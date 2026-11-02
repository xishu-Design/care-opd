from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import MemoryAccessLog, RoleMemory
from app.mcp.contracts import ToolRequest
from app.services.memory_consolidation_service import canonicalize_memory_content


def health(db: Session) -> dict:
    del db
    return {"backend": "persistent_memory", "status": "ok"}


def execute(db: Session, request: ToolRequest) -> dict:
    owner_role = request.arguments.get("owner_role", request.context.role)
    owner_user_id = int(request.arguments.get("owner_user_id", request.context.user_id))
    if owner_role != request.context.role or owner_user_id != request.context.user_id:
        _log_access(db, request, owner_user_id, owner_role, False, "MEMORY_SCOPE_VIOLATION")
        return {"error_code": "MEMORY_SCOPE_VIOLATION", "message": "cross-role memory access is not allowed"}

    if request.tool_name in {"write_role_memory", "update_student_weakness"}:
        memory = RoleMemory(
            owner_user_id=owner_user_id,
            owner_role=owner_role,
            memory_type=request.arguments.get("memory_type", request.tool_name),
            content_json=request.arguments.get("content") or _content_from_request(request),
            source="persistent_memory",
        )
        db.add(memory)
        db.flush()
        _log_access(db, request, owner_user_id, owner_role, True, "write")
        return {"memory_id": memory.id, "memory_used": [_memory_payload(memory)], "__metadata__": _metadata(False)}

    rows = db.scalars(
        select(RoleMemory)
        .where(RoleMemory.owner_user_id == owner_user_id, RoleMemory.owner_role == owner_role)
        .order_by(RoleMemory.updated_at.desc(), RoleMemory.id.desc())
        .limit(100)
    ).all()
    rows = [
        row
        for row in rows
        if row.memory_type != "ai_chat_turn"
        and (row.content_json or {}).get("status", "active") == "active"
    ][:20]
    memory_used = [_default_memory(request)]
    memory_used.extend(_memory_payload(row) for row in rows)
    _log_access(db, request, owner_user_id, owner_role, True, "read")
    return {"memory_used": memory_used, "__metadata__": _metadata(False)}


def _content_from_request(request: ToolRequest) -> dict:
    if request.tool_name == "update_student_weakness":
        return {"weakness": request.arguments.get("weakness", "用法用量缺失识别")}
    return request.arguments.get("content") or {"value": "persistent role memory"}


def _default_memory(request: ToolRequest) -> dict:
    if request.context.role == "doctor":
        content = {"format_preference": "risk_cards_first", "tone": "concise_doctor_internal"}
    elif request.context.role == "patient":
        content = {"explanation_level": "plain_language", "avoid_terms": ["复杂证候推理"]}
    else:
        content = {"progress": "phase_3_cases", "weakness": ["用法用量缺失识别"]}
    return {
        "source": "persistent_memory_default",
        "owner_user_id": request.context.user_id,
        "owner_role": request.context.role,
        "content": content,
    }


def _memory_payload(memory: RoleMemory) -> dict:
    content = dict(memory.content_json or {})
    content["content"] = canonicalize_memory_content(
        role=memory.owner_role,
        category=str(content.get("category") or memory.memory_type),
        memory_key=str(content.get("memory_key") or ""),
        content=str(content.get("content") or ""),
    )
    return {
        "source": memory.source,
        "owner_user_id": memory.owner_user_id,
        "owner_role": memory.owner_role,
        "memory_type": memory.memory_type,
        "content": content,
    }


def _log_access(db: Session, request: ToolRequest, owner_user_id: int | None, owner_role: str | None, allowed: bool, reason: str) -> None:
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


def _metadata(degraded: bool) -> dict:
    return {
        "backend": "persistent_memory",
        "degraded": degraded,
        "fallback_used": False,
        "degraded_reason": "memory_adapter_unavailable" if degraded else None,
    }
