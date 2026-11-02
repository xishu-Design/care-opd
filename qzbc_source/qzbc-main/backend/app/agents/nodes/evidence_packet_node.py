import hashlib

from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, copy_state, packet_visibility_for_role
from app.agents.state import AgentState
from app.evidence.evidence_packet import build_evidence_packet_v2
from app.mcp.contracts import ToolResult


class EvidencePacketNode:
    node_name = "EvidencePacketNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        state = append_node(
            self.db,
            state,
            self.node_name,
            output_summary={"tool_result_count": len(state.get("tool_results", []))},
        )
        visibility = packet_visibility_for_role(state["role"])
        tool_results = [ToolResult.model_validate(result) for result in state.get("tool_results", [])]
        request_bound_evidence, request_bound_contract = self._request_bound_case(
            state
        )
        packet = build_evidence_packet_v2(
            self.db,
            role=state["role"],
            task_type=state["task_type"],
            visibility=visibility,
            case_summary=self._case_summary(state),
            tool_results=tool_results,
            created_by_user_id=state["user_id"],
            trace_id=state.get("trace_id"),
            safety_decision=self._safety_decision(state, visibility),
            graph_node_path=state.get("graph_node_path", []),
            conversation_context=state.get("conversation_context") or {},
            request_bound_evidence=request_bound_evidence,
            request_bound_contract=request_bound_contract,
        )
        state["evidence_packet"] = packet
        state["degraded"] = bool(state.get("degraded")) or packet["degraded"]
        state["fallback_used"] = bool(state.get("fallback_used")) or packet["backend_summary"].get("fallback_used", False)
        return state

    def _request_bound_case(
        self, state: AgentState
    ) -> tuple[list[dict], dict | None]:
        raw = (state.get("input_payload") or {}).get("case_reasoning_context")
        if raw is None:
            return [], None
        if state.get("role") != "doctor" or state.get("task_type") != "doctor_ai_chat":
            raise RuntimeError("case reasoning material is allowed only on doctor_ai_chat")
        if not isinstance(raw, dict):
            raise RuntimeError("case reasoning context must be an object")
        checks = {
            "protocol": "public_anonymized_tcm_case_reasoning_v1",
            "source_kind": "public_anonymized_teaching_case",
            "inference_mode": "uncertain_teaching_hypothesis",
            "reference_answer_included": False,
            "clinical_use_authorized": False,
        }
        mismatches = {
            key: {"expected": expected, "actual": raw.get(key)}
            for key, expected in checks.items()
            if raw.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"invalid case reasoning contract: {mismatches}")
        case_material = str(raw.get("case_material") or "")
        expected_sha256 = str(raw.get("case_material_sha256") or "")
        observed_sha256 = hashlib.sha256(case_material.encode("utf-8")).hexdigest()
        if not case_material or observed_sha256 != expected_sha256:
            raise RuntimeError("case reasoning material hash failed closed")
        if case_material not in str(state.get("input_text") or ""):
            raise RuntimeError("complete case reasoning material is not bound to input_text")

        source_id = f"CASE_INPUT_{expected_sha256[:16]}"
        contract = {
            **checks,
            "task_token": raw.get("task_token"),
            "case_material_sha256": expected_sha256,
            "case_source_id": source_id,
            "current_request_bound": True,
            "complete_material_bound": True,
            "model_reference_answer_access": False,
            "product_adapters_bypassed": False,
        }
        evidence = {
            "source_id": source_id,
            "kind": "request_bound_case_material",
            "content": case_material,
            "summary": case_material,
            "snippet": case_material,
            "retrieved_by": "request_evidence_binding",
            "visibility": ["doctor", "doctor_internal"],
            "metadata": {
                "visibility": ["doctor", "doctor_internal"],
                "retrieved_by": "request_evidence_binding",
                "claim_status": "user_supplied_case_fact",
                **contract,
            },
        }
        return [evidence], contract

    def _case_summary(self, state: AgentState) -> str:
        if state["role"] == "doctor":
            if state["task_type"] == "doctor_ai_chat":
                return f"医生专业问答：{state.get('input_text', '')[:200]}"
            return f"Prescription {state.get('prescription_id')} for patient {state.get('patient_id')}"
        if state["role"] == "patient":
            if state.get("should_block"):
                return "patient education request blocked by Safety Guard"
            return "patient public TCM education"
        if state["role"] == "student":
            if state["task_type"] == "student_ai_chat":
                return f"学生教学问答：{state.get('input_text', '')[:200]}"
            return f"student learning case {state.get('learning_case_id')}"
        return "agent graph execution"

    def _safety_decision(self, state: AgentState, visibility: str) -> dict:
        if state.get("safety_decision"):
            safety = state["safety_decision"]
            return {
                "allow_answer": bool(safety.get("allowed", True)),
                "need_human_confirm": visibility == "doctor_internal",
                "refusal_reason": "" if safety.get("allowed", True) else safety.get("reason", ""),
            }
        return {
            "allow_answer": not bool(state.get("should_block")),
            "need_human_confirm": visibility == "doctor_internal",
            "refusal_reason": "",
        }
