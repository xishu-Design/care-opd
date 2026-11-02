from app.evaluation.runners.base import BaseEvaluationRunner
from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult
from app.evidence.evidence_packet import build_evidence_packet_v2
from app.mcp.contracts import ResourceScope, ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway


class EvidenceEvaluationRunner(BaseEvaluationRunner):
    def run(self, cases: list[EvaluationCase]) -> list[EvaluationCaseResult]:
        return [self.timed(case, self._case) for case in cases]

    def _case(self, case: EvaluationCase) -> EvaluationCaseResult:
        patient_tool = case.input["tool_name"] in {"public_tcm_education", "search_diet_knowledge"}
        actor = self.seed["patient"] if patient_tool else self.seed["doctor"]
        role = "patient" if patient_tool else "doctor"
        task_type = "patient_education" if patient_tool else "prescription_review"
        context = ToolContext(
            user_id=actor.id,
            role=role,
            task_type=task_type,
            patient_id=self.seed["patient"].id,
            resource_scope=ResourceScope(is_doctor_bound_to_patient=not patient_tool, is_patient_self=patient_tool),
        )
        tool = MCPGateway(self.db).execute(
            ToolRequest(tool_name=str(case.input["tool_name"]), arguments={"query": str(case.input["query"])}, context=context)
        )
        packet = build_evidence_packet_v2(
            self.db,
            role=role,
            task_type=task_type,
            visibility="patient_safe" if patient_tool else "doctor_internal",
            case_summary=f"Phase 12 {case.case_id}",
            tool_results=[] if "evidence_packet_required" in self.disabled else [tool],
            created_by_user_id=actor.id,
        )
        source_ids = sorted({str(item.get("source_id")) for item in packet["retrieved_evidence"] if item.get("source_id")})
        source_required = bool(case.expected.get("source_id_required"))
        passed = bool(packet.get("packet_id")) and (bool(source_ids) if source_required else "degraded" in packet)
        if "evidence_packet_required" in self.disabled:
            passed = False
        actual = {
            "packet_created": bool(packet.get("packet_id")),
            "source_ids": source_ids,
            "source_id_covered": bool(source_ids),
            "degraded": packet.get("degraded"),
            "degraded_visible": "degraded" in packet and "degraded_reasons" in packet,
            "fallback_used": tool.metadata.get("fallback_used", False),
            "backend": tool.metadata.get("backend"),
        }
        return self.result(case, passed, actual, evidence_packet_id=packet.get("packet_id"))
