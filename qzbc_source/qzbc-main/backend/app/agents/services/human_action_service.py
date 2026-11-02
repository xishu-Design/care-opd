from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import AgentTrace, DoctorPatientBinding, PendingHumanAction, utcnow
from app.harness.trace import record_event


PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
CHECKPOINT_LOST = "checkpoint_lost"
ALREADY_RESUMED = "already_resumed"


def make_action_id() -> str:
    return f"human_{uuid4().hex}"


def get_pending_action_by_trace(db: Session, trace_id: str) -> PendingHumanAction | None:
    return db.scalar(select(PendingHumanAction).where(PendingHumanAction.trace_id == trace_id).order_by(PendingHumanAction.created_at.desc()))


def list_pending_actions_for_doctor(db: Session, doctor_id: int) -> list[PendingHumanAction]:
    return db.scalars(
        select(PendingHumanAction)
        .join(DoctorPatientBinding, DoctorPatientBinding.patient_id == PendingHumanAction.patient_id)
        .where(
            DoctorPatientBinding.doctor_id == doctor_id,
            DoctorPatientBinding.is_active.is_(True),
            PendingHumanAction.status == PENDING,
        )
        .order_by(PendingHumanAction.created_at.desc())
    ).all()


def create_or_get_pending_action(
    db: Session,
    *,
    trace_id: str,
    thread_id: str,
    graph_name: str,
    node_name: str,
    prescription_id: int,
    patient_id: int,
    doctor_id: int,
    evidence_packet_id: str,
    risk_level: str | None,
    review_summary: dict[str, Any],
    interrupt_payload: dict[str, Any],
) -> PendingHumanAction:
    existing = get_pending_action_by_trace(db, trace_id)
    if existing:
        return existing
    now = utcnow()
    action = PendingHumanAction(
        action_id=make_action_id(),
        trace_id=trace_id,
        thread_id=thread_id,
        graph_name=graph_name,
        node_name=node_name,
        prescription_id=prescription_id,
        patient_id=patient_id,
        doctor_id=doctor_id,
        evidence_packet_id=evidence_packet_id,
        status=PENDING,
        risk_level=risk_level,
        review_summary_json=review_summary,
        interrupt_payload_json=interrupt_payload,
        expires_at=now + timedelta(minutes=get_settings().human_confirm_timeout_minutes),
    )
    db.add(action)
    _update_trace_metadata(
        db,
        trace_id,
        {
            "thread_id": thread_id,
            "checkpoint_backend": get_settings().langgraph_checkpoint_backend,
            "interrupt_enabled": get_settings().langgraph_enable_interrupt,
            "pending_action_id": action.action_id,
            "interrupted": False,
            "resumed": False,
        },
    )
    db.flush()
    return action


def mark_interrupted(db: Session, action: PendingHumanAction) -> None:
    trace = db.scalar(select(AgentTrace).where(AgentTrace.trace_id == action.trace_id))
    if trace:
        trace.status = "interrupted"
    _update_trace_metadata(
        db,
        action.trace_id,
        {"pending_action_id": action.action_id, "interrupted": True, "resumed": False},
    )
    record_event(
        db,
        action.trace_id,
        node_name=action.node_name,
        event_type="interrupt",
        status="interrupted",
        output_summary={"pending_action_id": action.action_id, "prescription_id": action.prescription_id},
        evidence_packet_id=action.evidence_packet_id,
        metadata={
            "thread_id": action.thread_id,
            "pending_action_id": action.action_id,
            "prescription_id": action.prescription_id,
            "evidence_packet_id": action.evidence_packet_id,
        },
    )
    db.flush()


def mark_resumed(
    db: Session,
    action: PendingHumanAction,
    *,
    approved: bool,
    comment: str | None,
    doctor_id: int,
    prescription_status: str,
) -> None:
    action.status = APPROVED if approved else REJECTED
    action.resume_payload_json = {"approved": approved, "comment": comment or "", "doctor_id": doctor_id}
    action.resume_comment = comment
    action.resumed_at = utcnow()
    _update_trace_metadata(
        db,
        action.trace_id,
        {
            "pending_action_id": action.action_id,
            "interrupted": True,
            "resumed": True,
            "resume_action": action.status,
            "human_confirmed": approved,
            "prescription_status": prescription_status,
        },
    )
    record_event(
        db,
        action.trace_id,
        node_name=action.node_name,
        event_type="resume",
        status="resumed",
        output_summary={"approved": approved, "prescription_status": prescription_status},
        evidence_packet_id=action.evidence_packet_id,
        metadata={"approved": approved, "comment": comment or "", "prescription_status": prescription_status},
    )
    record_event(
        db,
        action.trace_id,
        node_name=action.node_name,
        event_type="human_approved" if approved else "human_rejected",
        status="success",
        output_summary={"pending_action_id": action.action_id, "prescription_status": prescription_status},
        evidence_packet_id=action.evidence_packet_id,
        metadata={"doctor_id": doctor_id, "comment": comment or ""},
    )
    db.flush()


def mark_checkpoint_lost(db: Session, action: PendingHumanAction) -> None:
    action.status = CHECKPOINT_LOST
    action.resumed_at = utcnow()
    _update_trace_metadata(db, action.trace_id, {"pending_action_id": action.action_id, "checkpoint_lost": True})
    record_event(
        db,
        action.trace_id,
        node_name=action.node_name,
        event_type="checkpoint_missing",
        status="failed",
        output_summary={"pending_action_id": action.action_id},
        evidence_packet_id=action.evidence_packet_id,
        error_code="CHECKPOINT_NOT_FOUND",
        metadata={"thread_id": action.thread_id},
    )
    db.flush()


def ensure_pending(action: PendingHumanAction) -> None:
    if action.status != PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "HUMAN_ACTION_ALREADY_RESUMED",
                "message": "该医生确认动作已处理，不能重复提交。",
                "details": {"status": action.status},
            },
        )


def serialize_pending_action(action: PendingHumanAction) -> dict:
    return {
        "action_id": action.action_id,
        "trace_id": action.trace_id,
        "thread_id": action.thread_id,
        "graph_name": action.graph_name,
        "node_name": action.node_name,
        "prescription_id": action.prescription_id,
        "patient_id": action.patient_id,
        "doctor_id": action.doctor_id,
        "evidence_packet_id": action.evidence_packet_id,
        "status": action.status,
        "risk_level": action.risk_level,
        "review_summary": action.review_summary_json,
        "interrupt_payload": action.interrupt_payload_json,
        "resume_payload": action.resume_payload_json,
        "resume_comment": action.resume_comment,
        "created_at": action.created_at.isoformat(),
        "resumed_at": action.resumed_at.isoformat() if action.resumed_at else None,
        "expires_at": action.expires_at.isoformat() if action.expires_at else None,
    }


def _update_trace_metadata(db: Session, trace_id: str, updates: dict[str, Any]) -> None:
    trace = db.scalar(select(AgentTrace).where(AgentTrace.trace_id == trace_id))
    if not trace:
        return
    trace.metadata_json = {**(trace.metadata_json or {}), **updates}
    db.flush()
