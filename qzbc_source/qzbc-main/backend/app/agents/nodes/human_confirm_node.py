from sqlalchemy.orm import Session
from langgraph.types import interrupt

from app.agents.nodes.common import append_node, copy_state
from app.agents.services.human_action_service import ensure_pending, get_pending_action_by_trace, mark_resumed
from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.enums import PrescriptionStatus
from app.db.models import Prescription
from app.evidence.evidence_packet import update_evidence_packet_human_confirm
from app.services.patient_service import ensure_patient_bound
from app.services.prescription_service import PrescriptionStateService, update_prescription_review_human_confirm
from app.services.user_service import get_user


class HumanConfirmNode:
    node_name = "HumanConfirmNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        if state["role"] == "doctor" and state["task_type"] == "prescription_review":
            state["requires_human_confirm"] = True
            if get_settings().langgraph_enable_interrupt:
                resume_value = interrupt(state.get("input_payload", {}).get("interrupt_payload", self._interrupt_payload(state)))
                return self._resume(state, resume_value or {})
            state["human_confirmed"] = False
        return append_node(
            self.db,
            state,
            self.node_name,
            output_summary={
                "requires_human_confirm": state.get("requires_human_confirm", False),
                "human_confirmed": state.get("human_confirmed", False),
            },
        )

    def _resume(self, state: AgentState, resume_value: dict) -> AgentState:
        action = get_pending_action_by_trace(self.db, state["trace_id"])
        if not action:
            state["should_block"] = True
            state["errors"] = state.get("errors", []) + [{"code": "PENDING_HUMAN_ACTION_NOT_FOUND", "message": "pending human action not found"}]
            return append_node(self.db, state, self.node_name, status="blocked", error_code="PENDING_HUMAN_ACTION_NOT_FOUND")
        ensure_pending(action)

        approved = bool(resume_value.get("approved"))
        comment = resume_value.get("comment") or ""
        doctor_id = int(resume_value.get("doctor_id") or state["user_id"])
        doctor = get_user(self.db, doctor_id)
        prescription = self.db.get(Prescription, action.prescription_id)
        if not doctor or doctor.role != "doctor" or not prescription:
            state["should_block"] = True
            state["errors"] = state.get("errors", []) + [{"code": "RESUME_PERMISSION_DENIED", "message": "resume permission denied"}]
            return append_node(self.db, state, self.node_name, status="blocked", error_code="RESUME_PERMISSION_DENIED")
        ensure_patient_bound(self.db, doctor.id, action.patient_id)

        if approved:
            if prescription.status != PrescriptionStatus.REVIEWING.value:
                state["should_block"] = True
                state["errors"] = state.get("errors", []) + [
                    {
                        "code": "PRESCRIPTION_STATE_INVALID_FOR_RESUME",
                        "message": f"invalid prescription status: {prescription.status}",
                    }
                ]
                return append_node(
                    self.db,
                    state,
                    self.node_name,
                    status="blocked",
                    error_code="PRESCRIPTION_STATE_INVALID_FOR_RESUME",
                )
            PrescriptionStateService(self.db).transition(
                prescription,
                PrescriptionStatus.DOCTOR_CONFIRMED.value,
                doctor,
                "doctor_confirm_via_langgraph_resume",
            )
        elif prescription.status not in {PrescriptionStatus.REVIEWING.value, PrescriptionStatus.DOCTOR_CONFIRMED.value}:
            state["should_block"] = True
            state["errors"] = state.get("errors", []) + [
                {
                    "code": "PRESCRIPTION_STATE_INVALID_FOR_RESUME",
                    "message": f"invalid prescription status: {prescription.status}",
                }
            ]
            return append_node(
                self.db,
                state,
                self.node_name,
                status="blocked",
                error_code="PRESCRIPTION_STATE_INVALID_FOR_RESUME",
            )

        prescription_status = prescription.status
        state["human_confirmed"] = approved
        state["interrupted"] = True
        state["resume_action"] = "approved" if approved else "rejected"
        state["input_payload"]["pending_action_status"] = state["resume_action"]
        state["input_payload"]["resume_comment"] = comment
        state["input_payload"]["prescription_status"] = prescription_status
        human_confirm = {
            "required": True,
            "interrupted": True,
            "resumed": True,
            "approved": approved,
            "doctor_id": doctor.id,
            "comment": comment,
        }
        packet = dict(state.get("evidence_packet") or {})
        packet["human_confirm"] = {**(packet.get("human_confirm") or {}), **human_confirm}
        state["evidence_packet"] = packet
        update_evidence_packet_human_confirm(self.db, action.evidence_packet_id, human_confirm)
        update_prescription_review_human_confirm(self.db, action.prescription_id, action.evidence_packet_id, human_confirm)
        mark_resumed(
            self.db,
            action,
            approved=approved,
            comment=comment,
            doctor_id=doctor.id,
            prescription_status=prescription_status,
        )
        self.db.commit()
        return append_node(
            self.db,
            state,
            self.node_name,
            output_summary={
                "requires_human_confirm": True,
                "human_confirmed": approved,
                "resume_action": state["resume_action"],
                "prescription_status": prescription_status,
            },
        )

    def _interrupt_payload(self, state: AgentState) -> dict:
        packet = state.get("evidence_packet") or {}
        payload = state.get("input_payload", {})
        return {
            "action": "doctor_review_confirm",
            "trace_id": state.get("trace_id"),
            "thread_id": state.get("thread_id"),
            "prescription_id": state.get("prescription_id"),
            "patient_id": state.get("patient_id"),
            "evidence_packet_id": packet.get("packet_id"),
            "risk_level": payload.get("risk_level"),
            "review_summary": {"text": payload.get("review_summary", "")},
            "message": "请医生确认是否通过本次审方。批准后处方状态将变为 doctor_confirmed，但不会自动发布给患者。",
        }
