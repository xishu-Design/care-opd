from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, block_with_http_error, copy_state
from app.agents.state import AgentState
from app.core.enums import PrescriptionStatus
from app.harness.trace import start_trace
from app.services.learning_service import get_learning_case
from app.services.prescription_service import PrescriptionStateService, get_prescription_for_doctor
from app.services.user_service import get_user


class AuthNode:
    node_name = "AuthNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        trace = start_trace(
            self.db,
            user_id=state.get("user_id"),
            role=state["role"],
            task_type=state["task_type"],
            request_path=state["request_path"],
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
            metadata={
                "graph_name": state.get("graph_name"),
                "graph_version": state.get("graph_version"),
                "thread_id": state.get("thread_id"),
                "prescription_id": state.get("prescription_id"),
                "patient_id": state.get("patient_id"),
                "learning_case_id": state.get("learning_case_id"),
            },
        )
        state["trace_id"] = trace.trace_id
        try:
            user = get_user(self.db, state["user_id"])
            if not user or not user.is_active or user.role != state["role"]:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Role scope check failed")

            if state["role"] == "doctor" and state["task_type"] == "prescription_review":
                prescription = get_prescription_for_doctor(self.db, user.id, int(state["prescription_id"]))
                if prescription.status == PrescriptionStatus.DRAFT.value:
                    PrescriptionStateService(self.db).transition(
                        prescription,
                        PrescriptionStatus.REVIEWING.value,
                        user,
                        "submit_review",
                    )
                elif prescription.status != PrescriptionStatus.REVIEWING.value:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Only draft or reviewing prescriptions can be reviewed",
                    )
                state["patient_id"] = prescription.patient_id
                state["resource_scope"] = {"is_doctor_bound_to_patient": True}
                state["input_payload"].update(
                    {
                        "prescription_title": prescription.title,
                        "prescription_status": prescription.status,
                    }
                )

            elif state["role"] == "patient" and state["task_type"] == "patient_education":
                requested_patient_id = state.get("patient_id") or user.id
                if requested_patient_id != user.id:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Patient can only access own resources")
                state["patient_id"] = user.id
                state["resource_scope"] = {"is_patient_self": True}

            elif state["role"] == "student" and state["task_type"] == "student_training":
                learning_case = get_learning_case(self.db, int(state["learning_case_id"]))
                state["resource_scope"] = {"is_learning_case_deidentified": True}
                state["input_payload"].update(
                    {
                        "learning_case_title": learning_case.title,
                        "anonymized_case": learning_case.anonymized_case,
                        "standard_analysis": learning_case.standard_analysis,
                        "risk_points": learning_case.risk_points,
                        "evidence_chain": learning_case.evidence_chain,
                    }
                )

            elif state["role"] == "doctor" and state["task_type"] == "doctor_ai_chat":
                state["resource_scope"] = {}

            elif state["role"] == "student" and state["task_type"] == "student_ai_chat":
                state["resource_scope"] = {"is_learning_case_deidentified": True}

            else:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported agent role or task")

            state["auth_passed"] = True
            state["should_block"] = False
            state["should_continue"] = True
            return append_node(
                self.db,
                state,
                self.node_name,
                output_summary={
                    "auth_passed": True,
                    "role": state["role"],
                    "task_type": state["task_type"],
                    "resource_scope": state.get("resource_scope", {}),
                },
            )
        except HTTPException as exc:
            state = block_with_http_error(state, exc)
            return append_node(
                self.db,
                state,
                self.node_name,
                status="blocked",
                output_summary={"auth_passed": False, "detail": exc.detail},
                error_code=f"HTTP_{exc.status_code}",
            )
