from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fastapi import HTTPException, status

from app.db.models import RoleAIChat, RoleMemory
from app.services.patient_education_history_service import _delete_external_ai_memory
from app.services.patient_education_history_service import display_answer
from app.services.memory_consolidation_service import MemoryConsolidationAgent, sync_external_long_term_memory


SUPPORTED_CHAT_ROLES = {"doctor", "student"}


def save_role_ai_chat(db: Session, *, user_id: int, role: str, question: str, result: dict) -> RoleAIChat:
    _validate_role(role)
    answer = display_answer(result)
    chat = RoleAIChat(
        user_id=user_id,
        role=role,
        question=question,
        answer=answer,
        trace_id=result.get("trace_id"),
        evidence_packet=result.get("evidence_packet") or {},
        generated_response=result.get("generated_response") or {},
        llm_result=result.get("llm_result") or {},
    )
    db.add(chat)
    db.flush()
    consolidation = MemoryConsolidationAgent(db).consolidate_chat(
        owner_user_id=user_id,
        owner_role=role,
        user_message=question,
        source_chat_id=chat.id,
        trace_id=result.get("trace_id"),
    )
    db.commit()
    db.refresh(chat)
    if consolidation.external_sync_required and consolidation.memory_id:
        sync_external_long_term_memory(role=role, user_id=user_id, memory_id=consolidation.memory_id)
    _persist_external_ai_memory(
        role=role,
        user_id=user_id,
        question=question,
        answer=answer,
        trace_id=result.get("trace_id"),
    )
    return chat


def list_role_ai_chats(db: Session, *, user_id: int, role: str, limit: int = 100) -> list[dict]:
    _validate_role(role)
    rows = db.scalars(
        select(RoleAIChat)
        .where(RoleAIChat.user_id == user_id, RoleAIChat.role == role)
        .order_by(RoleAIChat.created_at.desc(), RoleAIChat.id.desc())
        .limit(limit)
    ).all()
    return [_serialize(row) for row in reversed(rows)]


def delete_role_ai_chat(db: Session, *, user_id: int, role: str, chat_id: int) -> None:
    _validate_role(role)
    row = db.scalar(
        select(RoleAIChat).where(
            RoleAIChat.id == chat_id,
            RoleAIChat.user_id == user_id,
            RoleAIChat.role == role,
        )
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat record not found")
    trace_id = row.trace_id
    memories = db.scalars(
        select(RoleMemory).where(
            RoleMemory.owner_user_id == user_id,
            RoleMemory.owner_role == role,
            RoleMemory.source == "role_ai_chat",
        )
    ).all()
    for memory in memories:
        if trace_id and str((memory.content_json or {}).get("trace_id") or "") == trace_id:
            db.delete(memory)
    consolidated_ids = MemoryConsolidationAgent(db).detach_chat_source(
        owner_user_id=user_id,
        owner_role=role,
        source_chat_id=chat_id,
    )
    db.delete(row)
    db.commit()
    for memory_id in consolidated_ids:
        sync_external_long_term_memory(role=role, user_id=user_id, memory_id=memory_id)
    _delete_external_ai_memory(role=role, user_id=user_id, trace_id=trace_id)


def _serialize(row: RoleAIChat) -> dict:
    return {
        "id": row.id,
        "role": row.role,
        "question": row.question,
        "answer": row.answer,
        "trace_id": row.trace_id,
        "evidence_packet": row.evidence_packet,
        "generated_response": row.generated_response,
        "llm_result": row.llm_result,
        "created_at": row.created_at.isoformat(),
    }


def _validate_role(role: str) -> None:
    if role not in SUPPORTED_CHAT_ROLES:
        raise ValueError(f"unsupported AI chat role: {role}")


def _persist_external_ai_memory(*, role: str, user_id: int, question: str, answer: str, trace_id: str | None) -> None:
    # SQL remains authoritative. Optional local Graphiti/Mem0 enrichment must
    # never make a completed chat request fail.
    try:
        from app.mcp.adapters.mem0_real import write_chat_memory

        write_chat_memory(role=role, user_id=user_id, question=question, answer=answer, trace_id=trace_id)
    except Exception:
        pass
    try:
        from app.mcp.adapters.graphiti_real import submit_chat_episode

        submit_chat_episode(role=role, user_id=user_id, question=question, answer=answer, trace_id=trace_id)
    except Exception:
        pass
