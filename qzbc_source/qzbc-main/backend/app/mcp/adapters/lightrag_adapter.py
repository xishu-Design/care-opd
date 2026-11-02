from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import LightRAGRetrievalLog
from app.lightrag_integration.client import LightRAGClient
from app.lightrag_integration.health import lightrag_health
from app.lightrag_integration.schemas import LightRAGQueryResult
from app.knowledge.qa_knowledge_index import search_qa_knowledge
from app.mcp.adapters import local_knowledge
from app.mcp.contracts import ToolRequest


ENTRY_TYPE_BY_TOOL = {
    "search_herb": "herb",
    "search_formula": "formula",
    "search_contraindication": "contraindication",
    "search_diet_knowledge": "diet_precaution",
    "public_tcm_education": "public_education",
}


def health(db: Session) -> dict:
    del db
    return lightrag_health()


def execute(db: Session, request: ToolRequest) -> dict:
    query = request.arguments.get("query") or request.arguments.get("message") or ""
    entry_type = request.arguments.get("entry_type") or ENTRY_TYPE_BY_TOOL.get(request.tool_name)
    top_k = int(request.arguments.get("top_k") or 5)
    mode = str(request.arguments.get("mode") or "hybrid")
    response = LightRAGClient().query(query, mode=mode, top_k=top_k)
    if not response.success:
        return _fallback(db, request, response.error or "lightrag_query_failed")

    evidence, filtered_count, missing_source_count = _evidence_from_results(response.results, request, entry_type, mode)
    qa_evidence = search_qa_knowledge(query=query, role=request.context.role, limit=min(3, top_k)) if entry_type is None else []
    evidence = _merge_evidence(qa_evidence, evidence, limit=top_k)
    if missing_source_count and not evidence:
        return _fallback(db, request, "lightrag_result_missing_source_id")

    metadata = {
        "backend": "lightrag",
        "knowledge_backend": "lightrag",
        "lightrag_mode": response.metadata.get("lightrag_mode") or get_settings().lightrag_mode,
        "retrieval_mode": f"{mode}+qa_knowledge" if qa_evidence else mode,
        "degraded": filtered_count > 0 and not evidence,
        "degraded_reason": "visibility_filtered_all_results" if filtered_count > 0 and not evidence else None,
        "fallback_used": False,
        "filtered_count": filtered_count,
        "missing_source_count": missing_source_count,
        "qa_retrieval_count": len(qa_evidence),
        "latency_ms": response.metadata.get("latency_ms", 0),
    }
    db.add(
        LightRAGRetrievalLog(
            session_id=request.context.session_id,
            query=query,
            role=request.context.role,
            task_type=request.context.task_type,
            evidence_json=evidence,
        )
    )
    db.flush()
    if request.tool_name == "public_tcm_education":
        return {"answer": _public_answer(evidence), "evidence": evidence, "__metadata__": metadata}
    return {"evidence": evidence, "__metadata__": metadata}


def _merge_evidence(primary: list[dict[str, Any]], secondary: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    result = []
    seen: set[tuple[str, str]] = set()
    for item in [*primary, *secondary]:
        key = (str(item.get("source_id") or ""), str(item.get("entry_id") or item.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _fallback(db: Session, request: ToolRequest, reason: str) -> dict:
    settings = get_settings()
    if not settings.lightrag_allow_fallback:
        return {
            "error_code": "LIGHTRAG_UNAVAILABLE",
            "message": reason,
            "__metadata__": {
                "backend": "lightrag",
                "knowledge_backend": "lightrag",
                "degraded": True,
                "degraded_reason": reason,
                "fallback_used": False,
            },
        }
    data = local_knowledge.execute(db, request)
    metadata = data.pop("__metadata__", {}) if isinstance(data, dict) else {}
    data["__metadata__"] = {
        **metadata,
        "backend": metadata.get("backend", "local_knowledge"),
        "configured_backend": "lightrag",
        "knowledge_backend": "local_knowledge",
        "degraded": True,
        "degraded_reason": reason,
        "fallback_used": True,
        "fallback_backend": "local_knowledge",
        "fallback_reason": reason,
    }
    return data


def _evidence_from_results(results: list[LightRAGQueryResult], request: ToolRequest, entry_type: str | None, retrieval_mode: str) -> tuple[list[dict[str, Any]], int, int]:
    evidence = []
    filtered_count = 0
    missing_source_count = 0
    for result in results:
        if entry_type and result.entry_type and result.entry_type != entry_type:
            filtered_count += 1
            continue
        if not result.source_id:
            missing_source_count += 1
            continue
        if not _visible_to_role(result.metadata, request.context.role):
            filtered_count += 1
            continue
        evidence.append(_to_evidence(result, retrieval_mode))
    return evidence, filtered_count, missing_source_count


def _to_evidence(result: LightRAGQueryResult, retrieval_mode: str) -> dict[str, Any]:
    metadata = dict(result.metadata)
    metadata.setdefault("source_id", result.source_id)
    metadata.setdefault("entry_id", result.entry_id)
    metadata.setdefault("entry_type", result.entry_type)
    metadata.setdefault("retrieved_by", "lightrag")
    metadata.setdefault("retrieval_mode", retrieval_mode)
    metadata.setdefault("graph_entities", result.graph_entities)
    metadata.setdefault("graph_relations", result.graph_relations)
    metadata.setdefault("graph_path", result.graph_path)
    return {
        "source_id": result.source_id,
        "entry_id": result.entry_id,
        "entry_type": result.entry_type,
        "title": result.title,
        "snippet": result.content[:240],
        "retrieved_by": "lightrag",
        "retrieval_mode": retrieval_mode,
        "graph_entities": result.graph_entities,
        "graph_relations": result.graph_relations,
        "graph_path": result.graph_path,
        "confidence": round(max(result.score, 0.01), 2),
        "source_records": metadata.get("source_records", []),
        "original_source": metadata.get("source_title", ""),
        "metadata": metadata,
    }


def _visible_to_role(metadata: dict[str, Any], role: str) -> bool:
    visibility = metadata.get("visibility", [])
    values = visibility if isinstance(visibility, list) else [visibility] if visibility else []
    allowed = {str(value) for value in values if value}
    if not allowed:
        return True
    if role == "student":
        return bool({"student", "student_learning"} & allowed)
    if role == "patient":
        return "patient" in allowed or "patient_safe" in allowed
    return role in allowed


def _public_answer(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "我目前只能提供一般性科普，具体用药请以医生说明为准。"
    snippet = evidence[0].get("snippet") or "中医药科普可以帮助理解常见概念。"
    return f"{snippet} 以上仅为一般科普，不能替代医生诊疗意见。"
