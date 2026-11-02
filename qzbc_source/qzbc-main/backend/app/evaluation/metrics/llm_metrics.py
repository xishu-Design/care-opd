from app.evaluation.metrics.common import rate
from app.evaluation.schemas import EvaluationCaseResult


def llm_metrics(results: list[EvaluationCaseResult]) -> dict[str, float]:
    llm_rows = [row for row in results if row.category == "llm_safety"]
    fallback_rows = [row for row in llm_rows if row.actual.get("fallback_used")]
    return {
        "structured_output_parse_success_rate": rate(results, "structured_output_parse_success_rate"),
        "schema_valid_rate": rate(results, "schema_valid_rate"),
        "fallback_used_rate": round(len(fallback_rows) / len(llm_rows), 4) if llm_rows else 0.0,
        "patient_dangerous_output_block_rate": rate(results, "patient_dangerous_output_block_rate"),
        "doctor_skip_confirm_block_rate": rate(results, "doctor_skip_confirm_block_rate"),
        "student_privacy_leak_block_rate": rate(results, "student_privacy_leak_block_rate"),
    }
