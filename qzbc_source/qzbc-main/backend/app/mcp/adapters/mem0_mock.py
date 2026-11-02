from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import MockMemory
from app.mcp.contracts import ToolRequest


def _default_memory(request: ToolRequest) -> dict:
    if request.context.role == "doctor":
        return {"format_preference": "risk_cards_first", "tone": "concise_doctor_internal"}
    if request.context.role == "patient":
        return {"explanation_level": "plain_language", "avoid_terms": ["复杂证候推理"]}
    return {"progress": "phase_1_cases", "weakness": ["用法用量缺失识别"]}


def _memory_payload(request: ToolRequest) -> list[dict]:
    return [
        {
            "source": "mock_mem0",
            "owner_user_id": request.context.user_id,
            "owner_role": request.context.role,
            "content": _default_memory(request),
        }
    ]


def execute(db: Session, request: ToolRequest) -> dict:
    args = request.arguments
    owner_role = args.get("owner_role", request.context.role)
    owner_user_id = int(args.get("owner_user_id", request.context.user_id))
    if owner_role != request.context.role or owner_user_id != request.context.user_id:
        return {"error_code": "MEMORY_SCOPE_VIOLATION", "message": "cross-role memory access is not allowed"}

    if request.tool_name in {"mock_write_role_memory", "mock_update_student_weakness", "write_role_memory", "update_student_weakness"}:
        content = args.get("content") or {"value": args.get("weakness", "mock_memory")}
        memory = MockMemory(
            owner_user_id=request.context.user_id,
            owner_role=request.context.role,
            memory_type=args.get("memory_type", request.tool_name),
            content_json=content,
        )
        db.add(memory)
        db.flush()
        return {"memory_id": memory.id, "memory_used": _memory_payload(request)}

    existing = db.scalars(
        select(MockMemory).where(
            MockMemory.owner_user_id == request.context.user_id,
            MockMemory.owner_role == request.context.role,
        )
    ).all()
    memory_used = _memory_payload(request)
    memory_used.extend(
        {
            "source": memory.source,
            "owner_user_id": memory.owner_user_id,
            "owner_role": memory.owner_role,
            "content": memory.content_json,
        }
        for memory in existing
    )
    return {"memory_used": memory_used}
