from sqlalchemy.orm import Session

from app.agents.nodes.common import add_error, append_node, copy_state
from app.agents.state import AgentState
from app.services.safety_service import check_patient_education_request
from app.services.user_service import get_user


class SafetyPreNode:
    node_name = "SafetyPreNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        if state["role"] != "patient":
            state["safety_decision"] = {"allowed": True, "reason": "pre_safety_not_required", "safe_reply": ""}
            return append_node(
                self.db,
                state,
                self.node_name,
                output_summary={"allowed": True, "reason": "pre_safety_not_required"},
                safety_decision=state["safety_decision"],
            )

        user = get_user(self.db, state["user_id"])
        if not user:
            state["should_block"] = True
            state["should_continue"] = False
            state = add_error(state, "USER_NOT_FOUND", "patient user not found")
            return append_node(self.db, state, self.node_name, status="blocked", error_code="USER_NOT_FOUND")

        safety = check_patient_education_request(self.db, user, state.get("input_text", ""))
        state["safety_decision"] = safety
        if not safety["allowed"]:
            state["route"] = "patient_high_risk_block"
            state["should_block"] = True
            state["should_continue"] = False
            state["final_response"] = dict(safety)

        return append_node(
            self.db,
            state,
            self.node_name,
            status="success" if safety["allowed"] else "blocked",
            input_summary={"message": state.get("input_text", "")},
            output_summary={"allowed": safety["allowed"], "reason": safety["reason"]},
            safety_decision=safety,
        )


class SafetyPostNode:
    node_name = "SafetyPostNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        packet = state.get("evidence_packet") or {}
        generated_text = str(state.get("generated_response") or "") + " " + str(state.get("llm_result") or "")
        packet_text = str(packet) + " " + generated_text
        unsafe_reason = ""
        if state["role"] == "patient":
            forbidden = ("rule_id", "doctor_internal", "graph_reasoning_path", "内部推理链", "可以加量", "可以停药", "处方如下", "我判断你是")
            if any(token in packet_text for token in forbidden):
                unsafe_reason = "patient_output_contains_internal_info"
        elif state["role"] == "student":
            forbidden = ("手机号", "身份证", "真实姓名", "联系方式", "精确就诊时间", "真实患者 ID")
            if any(token in packet_text for token in forbidden):
                unsafe_reason = "student_output_contains_real_patient_info"
            if packet and packet.get("visibility") != "student_learning":
                unsafe_reason = "student_evidence_visibility_invalid"
        elif state["role"] == "doctor" and packet and packet.get("visibility") != "doctor_internal":
            unsafe_reason = "doctor_evidence_visibility_invalid"
        elif state["role"] == "doctor" and any(token in generated_text for token in ("无需医生确认", "可直接发布", "保证有效")):
            unsafe_reason = "doctor_llm_output_boundary_violation"

        if unsafe_reason:
            state["should_block"] = True
            state["should_continue"] = False
            state = add_error(state, "SAFETY_POST_BLOCKED", unsafe_reason)

        return append_node(
            self.db,
            state,
            self.node_name,
            status="blocked" if unsafe_reason else "success",
            output_summary={"allowed": not bool(unsafe_reason), "reason": unsafe_reason or "post_safety_passed"},
            safety_decision={"allowed": not bool(unsafe_reason), "reason": unsafe_reason or "post_safety_passed"},
            error_code="SAFETY_POST_BLOCKED" if unsafe_reason else None,
        )
