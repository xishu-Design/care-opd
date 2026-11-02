from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, copy_state, evidence_packet_id
from app.agents.state import AgentState


class ResponseNode:
    node_name = "ResponseNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        if state.get("final_response") is None:
            if state["role"] == "doctor":
                state["final_response"] = self._doctor_response(state)
            elif state["role"] == "patient":
                state["final_response"] = self._patient_response(state)
            elif state["role"] == "student":
                state["final_response"] = self._student_response(state)
            else:
                state["final_response"] = {"errors": state.get("errors", [])}
        else:
            state["final_response"] = dict(state["final_response"])

        state["final_response"].update(
            {
                "trace_id": state.get("trace_id"),
                "thread_id": state.get("thread_id"),
                "graph_node_path": state.get("graph_node_path", []) + [self.node_name],
                "evidence_packet_id": evidence_packet_id(state),
                "degraded": state.get("degraded", False),
                "fallback_used": state.get("fallback_used", False),
            }
        )
        if state.get("evidence_packet"):
            state["final_response"]["evidence_packet"] = state["evidence_packet"]
        if state.get("generated_response"):
            state["final_response"]["generated_response"] = state["generated_response"]
        if state.get("llm_result"):
            state["final_response"]["llm_result"] = state["llm_result"]
        if state.get("requires_human_confirm"):
            state["final_response"]["requires_human_confirm"] = True

        return append_node(
            self.db,
            state,
            self.node_name,
            status="blocked" if state.get("should_block") else "success",
            output_summary={
                "has_response": True,
                "evidence_packet_id": evidence_packet_id(state),
                "requires_human_confirm": state.get("requires_human_confirm", False),
            },
        )

    def _doctor_response(self, state: AgentState) -> dict:
        if state["task_type"] == "doctor_ai_chat":
            return self._role_chat_response(state)

        from app.services.prescription_service import persist_prescription_review_result

        payload = state.get("input_payload", {})
        if payload.get("persisted_review_result"):
            persisted = dict(payload["persisted_review_result"])
            persisted["requires_human_confirm"] = state.get("requires_human_confirm", False)
            persisted["human_confirmed"] = state.get("human_confirmed", False)
            persisted["interrupted"] = state.get("interrupted", False)
            persisted["resume_action"] = state.get("resume_action")
            persisted["generated_response"] = state.get("generated_response")
            persisted["llm_result"] = state.get("llm_result")
            return persisted

        persisted = persist_prescription_review_result(
            self.db,
            prescription_id=int(state["prescription_id"]),
            review_result={
                "risk_level": payload.get("risk_level", "red"),
                "review_summary": payload.get("review_summary", "结构化规则引擎未返回结果，必须由医生人工复核。"),
            },
            rule_findings=payload.get("rule_findings", []),
            evidence_packet=state.get("evidence_packet") or {},
        )
        persisted["requires_human_confirm"] = state.get("requires_human_confirm", False)
        persisted["generated_response"] = state.get("generated_response")
        persisted["llm_result"] = state.get("llm_result")
        return persisted

    def _patient_response(self, state: AgentState) -> dict:
        response = dict(state.get("safety_decision") or {"allowed": True, "reason": "public_tcm_education_allowed"})
        if state.get("final_response"):
            response.update(state["final_response"])
        response.setdefault("safe_reply", "这是一般中医药科普信息，不能替代医生诊疗意见。如涉及用药调整，请咨询你的医生。")
        if state.get("generated_response"):
            response["generated_response"] = state["generated_response"]
        if state.get("llm_result"):
            response["llm_result"] = state["llm_result"]
        return response

    def _student_response(self, state: AgentState) -> dict:
        if state["task_type"] == "student_ai_chat":
            return self._role_chat_response(state)

        payload = state.get("input_payload", {})
        return {
            "id": state.get("learning_case_id"),
            "title": payload.get("learning_case_title"),
            "anonymized_case": payload.get("anonymized_case"),
            "standard_analysis": payload.get("standard_analysis"),
            "risk_points": payload.get("risk_points", []),
            "evidence_chain": payload.get("evidence_chain", []),
            "knowledge_cards": payload.get("knowledge_cards", []),
            "quiz": payload.get("quiz", []),
            "generated_response": state.get("generated_response"),
            "llm_result": state.get("llm_result"),
        }

    def _role_chat_response(self, state: AgentState) -> dict:
        generated = state.get("generated_response") or {}
        llm_result = state.get("llm_result") or {}
        return {
            "answer": generated.get("summary") or llm_result.get("content") or "当前证据不足，暂时无法生成可靠回答。",
            "generated_response": generated,
            "llm_result": llm_result,
        }
