from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.db.models import PatientEducationChat
from app.services.memory_consolidation_service import MemoryConsolidationAgent, sync_external_long_term_memory


def save_education_chat(db: Session, *, patient_id: int, question: str, result: dict) -> PatientEducationChat:
    chat = PatientEducationChat(
        patient_id=patient_id,
        question=question,
        answer=display_answer(result),
        allowed=bool(result.get("allowed", False)),
        trace_id=result.get("trace_id"),
        evidence_packet=result.get("evidence_packet") or {},
        generated_response=result.get("generated_response") or {},
        llm_result=result.get("llm_result") or {},
    )
    db.add(chat)
    db.flush()
    consolidation = MemoryConsolidationAgent(db).consolidate_chat(
        owner_user_id=patient_id,
        owner_role="patient",
        user_message=question,
        source_chat_id=chat.id,
        trace_id=result.get("trace_id"),
    )
    db.commit()
    db.refresh(chat)
    if consolidation.external_sync_required and consolidation.memory_id:
        sync_external_long_term_memory(role="patient", user_id=patient_id, memory_id=consolidation.memory_id)
    _persist_external_ai_memory(
        patient_id=patient_id,
        question=question,
        answer=chat.answer,
        trace_id=result.get("trace_id"),
    )
    return chat


def list_education_chats(db: Session, *, patient_id: int, limit: int = 100) -> list[dict]:
    rows = db.scalars(
        select(PatientEducationChat)
        .where(PatientEducationChat.patient_id == patient_id)
        .order_by(PatientEducationChat.created_at.desc(), PatientEducationChat.id.desc())
        .limit(limit)
    ).all()
    return [_serialize(row) for row in reversed(rows)]


def delete_education_chat(db: Session, *, patient_id: int, chat_id: int) -> None:
    row = db.scalar(
        select(PatientEducationChat).where(
            PatientEducationChat.id == chat_id,
            PatientEducationChat.patient_id == patient_id,
        )
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat record not found")
    trace_id = row.trace_id
    consolidated_ids = MemoryConsolidationAgent(db).detach_chat_source(
        owner_user_id=patient_id,
        owner_role="patient",
        source_chat_id=chat_id,
    )
    db.delete(row)
    db.commit()
    for memory_id in consolidated_ids:
        sync_external_long_term_memory(role="patient", user_id=patient_id, memory_id=memory_id)
    _delete_external_ai_memory(role="patient", user_id=patient_id, trace_id=trace_id)


def display_answer(result: dict) -> str:
    generated = result.get("generated_response") or {}
    sections = generated.get("sections") or []
    section_text = ["\n".join(part for part in (item.get("heading"), item.get("content")) if part) for item in sections if isinstance(item, dict)]
    return "\n\n".join(part for part in (generated.get("summary"), *section_text) if part) or result.get("safe_reply") or result.get("review_summary") or "已收到。"


def _serialize(row: PatientEducationChat) -> dict:
    return {
        "id": row.id,
        "question": row.question,
        "answer": row.answer,
        "allowed": row.allowed,
        "trace_id": row.trace_id,
        "evidence_packet": row.evidence_packet,
        "generated_response": row.generated_response,
        "llm_result": row.llm_result,
        "created_at": row.created_at.isoformat(),
    }


def _persist_external_ai_memory(*, patient_id: int, question: str, answer: str, trace_id: str | None) -> None:
    # The complete chat remains in SQL. Graphiti receives the temporal episode;
    # Mem0 receives only memories approved by MemoryConsolidationAgent.
    try:
        from app.mcp.adapters.graphiti_real import submit_chat_episode

        submit_chat_episode(role="patient", user_id=patient_id, question=question, answer=answer, trace_id=trace_id)
    except Exception:
        pass


def _delete_external_ai_memory(*, role: str, user_id: int, trace_id: str | None) -> None:
    if not trace_id:
        return
    try:
        from app.mcp.adapters.mem0_real import delete_chat_memory

        delete_chat_memory(role=role, user_id=user_id, trace_id=trace_id)
    except Exception:
        pass
    try:
        from app.mcp.adapters.graphiti_real import delete_chat_episode

        delete_chat_episode(role=role, user_id=user_id, trace_id=trace_id)
    except Exception:
        pass
