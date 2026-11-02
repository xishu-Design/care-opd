from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, copy_state, evidence_packet_id
from app.agents.state import AgentState
from app.db.models import EvidencePacketLog
from app.llm.service import LLMService


class LLMGenerateNode:
    node_name = "LLMGenerateNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        if state.get("should_block"):
            return append_node(
                self.db,
                state,
                self.node_name,
                status="blocked",
                output_summary={"skipped": True, "reason": "blocked_before_llm"},
            )
        packet = state.get("evidence_packet") or {}
        service = LLMService(self.db)
        context = {
            "trace_id": state.get("trace_id"),
            "user_id": state.get("user_id"),
            "input_text": state.get("input_text", ""),
            "constraints": {"role": state.get("role"), "visibility": packet.get("visibility")},
        }
        if state["role"] == "student" and state["task_type"] == "student_training":
            case = state.get("input_payload", {}).get("anonymized_case") or {}
            profile = case.get("patient_profile") if isinstance(case, dict) and isinstance(case.get("patient_profile"), dict) else {}
            prescription = case.get("prescription") if isinstance(case, dict) and isinstance(case.get("prescription"), dict) else {}
            items = prescription.get("items") if isinstance(prescription.get("items"), list) else []
            herbs = [str(item.get("herb_name") or item.get("name") or "").strip() for item in items if isinstance(item, dict)]
            symptoms = case.get("symptoms") if isinstance(case, dict) else []
            context["input_text"] = (
                f"脱敏教学案例：年龄={profile.get('age') or profile.get('age_range') or '未提供'}；"
                f"性别={profile.get('gender') or '未提供'}；"
                f"症状={ '、'.join(str(item) for item in symptoms) if isinstance(symptoms, list) else symptoms or '未提供'}；"
                f"参考方药={'、'.join(item for item in herbs if item) or '未提供'}。"
                "请只解释结构化参考方和教学边界，不得把药材写成对该患者的治疗结论。"
            )
            context["constraints"]["student_training_reference_only"] = True
        if state["task_type"] == "doctor_ai_chat":
            result = service.generate_doctor_ai_chat(packet, context)
        elif state["task_type"] == "student_ai_chat":
            result = service.generate_student_ai_chat(packet, context)
        elif state["role"] == "doctor":
            result = service.generate_doctor_review(packet, context)
        elif state["role"] == "student":
            result = service.generate_student_analysis(packet, context)
        else:
            result = service.generate_patient_education(packet, context)

        return self.apply_result(state, result)

    def apply_result(self, state: AgentState, result) -> AgentState:
        state = copy_state(state)
        result_dict = result.model_dump(mode="json")
        generated = result.structured_output
        state["llm_result"] = result_dict
        state["generated_response"] = generated
        state["degraded"] = bool(state.get("degraded")) or result.metadata.degraded
        state["fallback_used"] = bool(state.get("fallback_used")) or result.metadata.fallback_used
        self._merge_generated_fields(state, generated, result_dict)
        self._update_packet_llm_metadata(state, result_dict)
        self.db.commit()
        return append_node(
            self.db,
            state,
            self.node_name,
            output_summary={
                "provider": result.provider,
                "fallback_used": result.metadata.fallback_used,
                "degraded": result.metadata.degraded,
                "safety_flags": result.safety_flags,
                "evidence_packet_id": evidence_packet_id(state),
            },
            metadata={
                "provider": result.provider,
                "model": result.model,
                "fallback_used": result.metadata.fallback_used,
                "degraded": result.metadata.degraded,
                "safety_flags": result.safety_flags,
            },
        )

    def _merge_generated_fields(self, state: AgentState, generated: dict, result: dict) -> None:
        payload = state["input_payload"]
        payload["generated_response"] = generated
        payload["llm_result"] = result
        if state["task_type"] in {"doctor_ai_chat", "student_ai_chat"}:
            state["final_response"] = {
                "answer": generated.get("summary") or result.get("content"),
                "generated_response": generated,
                "llm_result": result,
            }
        elif state["role"] == "doctor":
            if generated.get("summary"):
                # Structured rules own the persisted risk summary. LLM output is
                # explanatory copy and must never overwrite the rule-engine result.
                payload["llm_review_summary"] = generated["summary"]
        elif state["role"] == "patient":
            response = dict(state.get("safety_decision") or {"allowed": True, "reason": "public_tcm_education_allowed"})
            response["safe_reply"] = (
                result.get("content")
                if result.get("provider") == "patient_safety_policy"
                else generated.get("summary") or result.get("content")
            )
            response["generated_response"] = generated
            response["llm_result"] = result
            state["final_response"] = response
        elif state["role"] == "student":
            # The stored standard analysis is the reviewed reference.  Model
            # copy is supplementary and must never overwrite it.
            if not payload.get("standard_analysis"):
                payload["standard_analysis"] = generated.get("summary") or result.get("content")
            payload["knowledge_cards"] = [
                {"title": section.get("heading"), "content": section.get("content"), "source_ids": section.get("source_ids", [])}
                for section in generated.get("sections", [])
            ]
            generated_quiz = [
                {"question": action, "answer": "请参考 Evidence Packet 和标准解析。"} for action in generated.get("next_actions", [])
            ]
            from app.services.learning_service import contextualize_student_quiz

            payload["quiz"] = contextualize_student_quiz(
                payload.get("anonymized_case") or {},
                generated_quiz or payload.get("quiz", []),
            )

    def _update_packet_llm_metadata(self, state: AgentState, result: dict) -> None:
        packet = dict(state.get("evidence_packet") or {})
        backend_summary = dict(packet.get("backend_summary") or {})
        backend_summary["llm"] = {
            "provider": result.get("provider"),
            "model": result.get("model"),
            "fallback_used": result.get("metadata", {}).get("fallback_used", False),
            "degraded": result.get("metadata", {}).get("degraded", False),
            "fallback_reason": result.get("metadata", {}).get("fallback_reason"),
            "parse_success": result.get("metadata", {}).get("parse_success", True),
            "schema_valid": result.get("metadata", {}).get("schema_valid", True),
            "parser_error": result.get("metadata", {}).get("parser_error"),
            "safety_flags": result.get("safety_flags", []),
        }
        packet["backend_summary"] = backend_summary
        packet["llm_result"] = result
        state["evidence_packet"] = packet
        packet_id = packet.get("packet_id")
        if packet_id:
            log = self.db.scalar(select(EvidencePacketLog).where(EvidencePacketLog.packet_id == packet_id))
            if log:
                log.backend_summary_json = backend_summary
                log.degraded = bool(log.degraded or result.get("metadata", {}).get("degraded", False))
                self.db.flush()
