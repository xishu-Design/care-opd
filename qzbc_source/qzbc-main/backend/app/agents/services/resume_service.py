from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.agents.graphs.doctor_review_graph import DoctorReviewGraph
from app.agents.services.checkpoint_service import get_langgraph_checkpointer
from app.agents.services.human_action_service import (
    ensure_pending,
    get_pending_action_by_trace,
    list_pending_actions_for_doctor,
    mark_checkpoint_lost,
    serialize_pending_action,
)
from app.core.config import get_settings
from app.core.enums import PrescriptionStatus
from app.db.models import Prescription, User
from app.services.patient_service import ensure_patient_bound


def interrupt_resume_enabled() -> bool:
    return bool(get_settings().langgraph_enable_interrupt)


def resume_doctor_review(db: Session, *, trace_id: str, approved: bool, comment: str | None, doctor: User) -> dict:
    action = get_pending_action_by_trace(db, trace_id)
    if not action:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pending human action not found")

    _ensure_resume_permission(db, action.patient_id, doctor)
    ensure_pending(action)
    prescription = db.get(Prescription, action.prescription_id)
    if not prescription or prescription.status != PrescriptionStatus.REVIEWING.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "PRESCRIPTION_STATE_INVALID_FOR_RESUME",
                "message": "当前处方状态不允许执行该恢复动作。",
                "details": {"status": prescription.status if prescription else None},
            },
        )

    config = {"configurable": {"thread_id": action.thread_id}}
    if get_langgraph_checkpointer().get_tuple(config) is None:
        mark_checkpoint_lost(db, action)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "CHECKPOINT_NOT_FOUND",
                "message": "当前 graph checkpoint 已丢失，请重新发起审方。",
            },
        )

    result = DoctorReviewGraph(db).resume(
        thread_id=action.thread_id,
        approved=approved,
        comment=comment,
        doctor_id=doctor.id,
    )
    prescription = db.get(Prescription, action.prescription_id)
    action = get_pending_action_by_trace(db, trace_id)
    return {
        "trace_id": trace_id,
        "thread_id": action.thread_id,
        "prescription_id": action.prescription_id,
        "resume_action": action.status,
        "human_confirmed": action.status == "approved",
        "prescription_status": prescription.status if prescription else None,
        "graph_node_path": result.get("graph_node_path", []),
        "pending_action_status": action.status,
        "evidence_packet_id": action.evidence_packet_id,
        "result": result,
    }


def pending_actions_for_doctor(db: Session, doctor: User) -> list[dict]:
    return [serialize_pending_action(action) for action in list_pending_actions_for_doctor(db, doctor.id)]


def pending_action_detail(db: Session, trace_id: str, doctor: User) -> dict:
    action = get_pending_action_by_trace(db, trace_id)
    if not action:
        return {}
    _ensure_resume_permission(db, action.patient_id, doctor)
    return serialize_pending_action(action)


def _ensure_resume_permission(db: Session, patient_id: int, doctor: User) -> None:
    if doctor.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_code": "RESUME_PERMISSION_DENIED", "message": "当前用户无权恢复该审方流程。"},
        )
    try:
        ensure_patient_bound(db, doctor.id, patient_id)
    except HTTPException as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_code": "RESUME_PERMISSION_DENIED", "message": "当前用户无权恢复该审方流程。"},
        ) from exc
