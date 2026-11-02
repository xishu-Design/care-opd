import json

from app.evaluation.runners.base import BaseEvaluationRunner
from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult
from app.llm.output_parser import parse_generated_response_result
from app.llm.safety import apply_output_safety
from app.llm.schemas import LLMRequest, LLMResult


class LLMEvaluationRunner(BaseEvaluationRunner):
    def run(self, cases: list[EvaluationCase]) -> list[EvaluationCaseResult]:
        return [self.timed(case, self._case) for case in cases]

    def _case(self, case: EvaluationCase) -> EvaluationCaseResult:
        role = str(case.input["role"])
        task_type = "doctor_review_text" if role == "doctor" else "patient_education" if role == "patient" else "student_analysis"
        raw = json.dumps(
            {
                "title": "Phase 12",
                "summary": str(case.input["unsafe_content"]),
                "sections": [],
                "warnings": [],
                "next_actions": [],
                "disclaimer": "仅用于评测",
            },
            ensure_ascii=False,
        )
        parsed = parse_generated_response_result(raw)
        result = LLMResult(
            request_id=f"eval_{case.case_id}",
            provider="template",
            content=str(case.input["unsafe_content"]),
            structured_output=parsed.structured_output or {},
        )
        request = LLMRequest(
            role=role,
            task_type=task_type,
            input_text="phase12 output safety",
            evidence_packet={"packet_id": f"packet_{case.case_id}", "visibility": "evaluation", "retrieved_evidence": [{"source_id": "SRC_EVAL"}]},
        )
        guarded = result if "safety_guard" in self.disabled else apply_output_safety(request, result)
        safety_expected = case.metric_name not in {"structured_output_parse_success_rate", "schema_valid_rate"}
        forbidden_still_present = safety_expected and str(case.input["unsafe_content"]) in guarded.content and bool(guarded.safety_flags) is False
        safety_passed = bool(guarded.safety_flags) if safety_expected else True
        passed = parsed.success and parsed.schema_valid and safety_passed and not forbidden_still_present
        if "safety_guard" in self.disabled and safety_expected:
            passed = False
        return self.result(
            case,
            passed,
            {
                "parse_success": parsed.success,
                "schema_valid": parsed.schema_valid,
                "safety_passed": safety_passed,
                "safety_flags": guarded.safety_flags,
                "fallback_used": guarded.metadata.fallback_used,
                "output": guarded.content,
            },
        )
