from app.evaluation.metrics.common import rate
from app.evaluation.schemas import EvaluationCaseResult


def safety_metrics(results: list[EvaluationCaseResult]) -> dict[str, float]:
    return {
        "patient_high_risk_refusal_rate": rate(results, "patient_high_risk_refusal_rate"),
        "mcp_unauthorized_block_rate": rate(results, "mcp_unauthorized_block_rate"),
        "student_identity_leak_rate": round(1 - rate(results, "student_deid_pass_rate"), 4),
        "doctor_unbound_access_block_rate": rate(results, "doctor_unbound_access_block_rate"),
        "prescription_status_consistency_rate": rate(results, "prescription_status_consistency_rate"),
        "duplicate_resume_block_rate": rate(results, "duplicate_resume_block_rate"),
        "memory_isolation_rate": rate(results, "memory_isolation_rate"),
    }
