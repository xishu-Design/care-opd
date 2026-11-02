from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, copy_state, evidence_packet_id
from app.agents.services.human_action_service import create_or_get_pending_action
from app.agents.state import AgentState
from app.core.config import get_settings
from app.evidence.evidence_packet import update_evidence_packet_human_confirm
from app.services.prescription_service import persist_prescription_review_result, update_prescription_review_human_confirm


class ReviewPersistNode:
    node_name = "ReviewPersistNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        payload = state.get("input_payload", {})
        packet = dict(state.get("evidence_packet") or {})
        human_confirm = {
            "required": True,
            "interrupted": False,
            "resumed": False,
            "approved": None,
            "doctor_id": None,
            "comment": "",
        }
        packet["human_confirm"] = human_confirm
        state["evidence_packet"] = packet
        update_evidence_packet_human_confirm(self.db, packet.get("packet_id"), human_confirm)

        persisted = persist_prescription_review_result(
            self.db,
            prescription_id=int(state["prescription_id"]),
            review_result={
                "risk_level": payload.get("risk_level", "red"),
                "review_summary": payload.get("review_summary", "结构化规则引擎未返回结果，必须由医生人工复核。"),
            },
            rule_findings=payload.get("rule_findings", []),
            evidence_packet=packet,
        )
        state["input_payload"]["persisted_review_result"] = persisted
        state["requires_human_confirm"] = True

        if get_settings().langgraph_enable_interrupt:
            interrupt_payload = {
                "action": "doctor_review_confirm",
                "trace_id": state.get("trace_id"),
                "thread_id": state.get("thread_id"),
                "prescription_id": state.get("prescription_id"),
                "patient_id": state.get("patient_id"),
                "evidence_packet_id": evidence_packet_id(state),
                "risk_level": persisted.get("risk_level"),
                "review_summary": {"text": persisted.get("review_summary", "")},
                "message": "请医生确认是否通过本次审方。批准后处方状态将变为 doctor_confirmed，但不会自动发布给患者。",
            }
            action = create_or_get_pending_action(
                self.db,
                trace_id=state["trace_id"],
                thread_id=state["thread_id"],
                graph_name=state["graph_name"],
                node_name="HumanConfirmNode",
                prescription_id=int(state["prescription_id"]),
                patient_id=int(state["patient_id"]),
                doctor_id=int(state["user_id"]),
                evidence_packet_id=evidence_packet_id(state) or "",
                risk_level=persisted.get("risk_level"),
                review_summary={"text": persisted.get("review_summary", "")},
                interrupt_payload=interrupt_payload,
            )
            action.interrupt_payload_json = {**interrupt_payload, "pending_action_id": action.action_id}
            state["input_payload"]["pending_action_id"] = action.action_id
            state["input_payload"]["interrupt_payload"] = action.interrupt_payload_json
            self.db.flush()

        update_prescription_review_human_confirm(self.db, int(state["prescription_id"]), packet.get("packet_id"), human_confirm)
        self.db.commit()
        return append_node(
            self.db,
            state,
            self.node_name,
            output_summary={
                "review_persisted": True,
                "evidence_packet_id": evidence_packet_id(state),
                "pending_action_id": state["input_payload"].get("pending_action_id"),
            },
        )
