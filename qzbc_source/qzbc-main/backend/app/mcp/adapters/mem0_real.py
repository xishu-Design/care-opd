from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import MemoryAccessLog
from app.mcp.contracts import ToolRequest


_CLIENT: Any | None = None
_CLIENT_LOCK = Lock()


def health(db: Session) -> dict:
    del db
    _memory_client()
    return {
        "backend": "mem0",
        "status": "ok",
        "storage": "local_qdrant",
        "collection": get_settings().mem0_collection_name,
    }


def execute(db: Session, request: ToolRequest) -> dict:
    owner_role = str(request.arguments.get("owner_role", request.context.role))
    owner_user_id = int(request.arguments.get("owner_user_id", request.context.user_id))
    if owner_role != request.context.role or owner_user_id != request.context.user_id:
        _log_access(db, request, owner_user_id, owner_role, False, "MEMORY_SCOPE_VIOLATION")
        return {"error_code": "MEMORY_SCOPE_VIOLATION", "message": "禁止跨角色读取记忆"}

    memory = _memory_client()
    user_scope = _user_scope(owner_role, owner_user_id)
    if request.tool_name in {"write_role_memory", "update_student_weakness"}:
        content = request.arguments.get("content") or _content_from_request(request)
        text = _memory_text(content)
        result = memory.add(
            text,
            user_id=user_scope,
            metadata={
                "owner_role": owner_role,
                "owner_user_id": owner_user_id,
                "memory_type": request.arguments.get("memory_type", request.tool_name),
                "tool_name": request.tool_name,
            },
            infer=False,
        )
        _log_access(db, request, owner_user_id, owner_role, True, "write")
        return {
            "memory_id": _first_memory_id(result),
            "memory_used": [_result_memory(result, owner_role, owner_user_id, content)],
            "__metadata__": _metadata(),
        }

    query = str(request.arguments.get("query") or _default_query(request)).strip()
    settings = get_settings()
    results = memory.search(query, top_k=settings.mem0_search_top_k, filters={"user_id": user_scope})
    rows = _relevant_search_rows(
        _result_rows(results),
        min_score=settings.mem0_min_relevance_score,
        limit=settings.mem0_search_top_k,
    )
    memory_used = [_default_memory(request)]
    memory_used.extend(_search_payload(row, owner_role, owner_user_id) for row in rows)
    _log_access(db, request, owner_user_id, owner_role, True, "read")
    return {"memory_used": memory_used, "__metadata__": _metadata()}


def write_chat_memory(*, role: str, user_id: int, question: str, answer: str, trace_id: str | None) -> dict:
    """Legacy compatibility only.

    Raw question/answer turns are authoritative in SQL and are deliberately not
    promoted into Mem0 anymore. Keeping this function as a no-op avoids accidental
    reintroduction by older call sites.
    """
    del role, user_id, question, answer, trace_id
    return {"stored": False, "reason": "raw_chat_is_short_term_archive"}


def sync_long_term_memory(*, role: str, user_id: int, memory_id: int, content_json: dict) -> dict:
    settings = get_settings()
    if settings.memory_backend != "mem0" or not settings.mem0_ingest_chat_history:
        return {"stored": False, "reason": "mem0_not_enabled"}
    memory = _memory_client()
    scope = _user_scope(role, user_id)
    rows = _result_rows(memory.get_all(filters={"user_id": scope}, top_k=500))
    deleted = 0
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if str(metadata.get("sql_memory_id") or "") != str(memory_id) or not row.get("id"):
            continue
        memory.delete(str(row["id"]))
        deleted += 1
    if content_json.get("status", "active") != "active":
        return {"stored": False, "deleted": deleted, "reason": "memory_inactive"}
    text = str(content_json.get("content") or "").strip()
    if not text:
        return {"stored": False, "deleted": deleted, "reason": "empty_memory"}
    result = memory.add(
        text,
        user_id=scope,
        metadata={
            "owner_role": role,
            "owner_user_id": user_id,
            "memory_type": content_json.get("category") or "long_term_memory",
            "memory_key": content_json.get("memory_key"),
            "sql_memory_id": memory_id,
            "status": "active",
            "source": "memory_consolidation",
        },
        infer=False,
    )
    return {"stored": True, "deleted": deleted, "memory_id": _first_memory_id(result)}


def delete_chat_memory(*, role: str, user_id: int, trace_id: str) -> dict:
    settings = get_settings()
    if settings.memory_backend != "mem0" or not settings.mem0_ingest_chat_history:
        return {"deleted": 0, "reason": "mem0_not_enabled"}
    memory = _memory_client()
    rows = _result_rows(memory.get_all(filters={"user_id": _user_scope(role, user_id)}, top_k=500))
    deleted = 0
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if str(metadata.get("trace_id") or "") != trace_id or not row.get("id"):
            continue
        memory.delete(str(row["id"]))
        deleted += 1
    return {"deleted": deleted}


def _memory_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        settings = get_settings()
        mem0_dir = _resolve_project_path(settings.mem0_dir)
        qdrant_path = _resolve_project_path(settings.mem0_qdrant_path)
        history_path = _resolve_project_path(settings.mem0_history_db_path)
        for path in (mem0_dir, qdrant_path, history_path.parent):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MEM0_DIR", str(mem0_dir))
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        # Dense Ollama embeddings are the configured retrieval path. Optional
        # spaCy/BM25 extras are intentionally not required for this deployment.
        logging.getLogger("mem0.utils.spacy_models").setLevel(logging.ERROR)
        logging.getLogger("mem0.vector_stores.qdrant").setLevel(logging.ERROR)
        from mem0 import Memory

        base_url = settings.ollama_base_url.rstrip("/")
        _CLIENT = Memory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": settings.mem0_collection_name,
                        "path": str(qdrant_path),
                        "embedding_model_dims": settings.mem0_embedding_dims,
                        "on_disk": True,
                    },
                },
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": settings.ollama_model,
                        "ollama_base_url": base_url,
                        "temperature": settings.llm_temperature,
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": settings.ollama_embedding_model,
                        "ollama_base_url": base_url,
                        "embedding_dims": settings.mem0_embedding_dims,
                    },
                },
                "history_db_path": str(history_path),
            }
        )
    return _CLIENT


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[4] / path


def _user_scope(role: str, user_id: int) -> str:
    return f"{role}:{user_id}"


def _content_from_request(request: ToolRequest) -> dict:
    if request.tool_name == "update_student_weakness":
        return {"weakness": request.arguments.get("weakness", "用法用量缺失识别")}
    return {"value": "角色长期记忆"}


def _memory_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _default_query(request: ToolRequest) -> str:
    if request.context.role == "doctor":
        return "医生的回答格式偏好和最近专业问答"
    if request.context.role == "patient":
        return "患者的科普解释偏好和最近问题"
    return "学生的学习进度、薄弱点和最近问题"


def _default_memory(request: ToolRequest) -> dict:
    if request.context.role == "doctor":
        content = {"format_preference": "risk_cards_first", "tone": "concise_doctor_internal"}
    elif request.context.role == "patient":
        content = {"explanation_level": "plain_language", "avoid_terms": ["复杂证候推理"]}
    else:
        content = {"progress": "phase_3_cases", "weakness": ["用法用量缺失识别"]}
    return {
        "source": "mem0_default",
        "owner_user_id": request.context.user_id,
        "owner_role": request.context.role,
        "content": content,
    }


def _result_rows(result: Any) -> list[dict]:
    if isinstance(result, dict):
        rows = result.get("results", [])
        return rows if isinstance(rows, list) else []
    return result if isinstance(result, list) else []


def _first_memory_id(result: Any) -> str | None:
    rows = _result_rows(result)
    if not rows and isinstance(result, dict):
        rows = result.get("results", [])
    if rows and isinstance(rows[0], dict):
        return str(rows[0].get("id") or "") or None
    return None


def _result_memory(result: Any, role: str, user_id: int, content: Any) -> dict:
    return {
        "source": "mem0",
        "memory_id": _first_memory_id(result),
        "owner_user_id": user_id,
        "owner_role": role,
        "content": content,
    }


def _search_payload(row: dict, role: str, user_id: int) -> dict:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "source": "mem0",
        "memory_id": row.get("id"),
        "owner_user_id": user_id,
        "owner_role": role,
        "memory_type": metadata.get("memory_type"),
        "content": row.get("memory") or row.get("text") or "",
        "score": row.get("score"),
        "metadata": metadata,
    }


def _relevant_search_rows(rows: list[dict], *, min_score: float, limit: int) -> list[dict]:
    selected: list[dict] = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("memory_type") == "ai_chat_turn" or metadata.get("status") == "deleted":
            continue
        score = row.get("score")
        if not isinstance(score, (int, float)) or float(score) < min_score:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _log_access(db: Session, request: ToolRequest, owner_user_id: int, owner_role: str, allowed: bool, reason: str) -> None:
    db.add(
        MemoryAccessLog(
            requester_user_id=request.context.user_id,
            requester_role=request.context.role,
            owner_user_id=owner_user_id,
            owner_role=owner_role,
            tool_name=request.tool_name,
            allowed=allowed,
            reason=reason,
        )
    )
    db.flush()


def _metadata() -> dict:
    return {
        "backend": "mem0",
        "degraded": False,
        "fallback_used": False,
        "storage": "local_qdrant",
    }
