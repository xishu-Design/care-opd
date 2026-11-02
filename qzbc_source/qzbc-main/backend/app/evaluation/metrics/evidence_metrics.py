from app.evaluation.metrics.common import rate
from app.evaluation.schemas import EvaluationCaseResult


def evidence_metrics(results: list[EvaluationCaseResult]) -> dict[str, float]:
    rag_rows = [row for row in results if row.dataset == "lightrag_retrieval_cases"]
    local_hit_rate = round(sum(bool(row.actual.get("local_hit")) for row in rag_rows) / len(rag_rows), 4) if rag_rows else 0.0
    fallback_visible_rate = round(sum(bool(row.actual.get("fallback_visible")) for row in rag_rows) / len(rag_rows), 4) if rag_rows else 1.0
    degraded_rows = [row for row in results if row.dataset in {"evidence_packet_cases", "lightrag_retrieval_cases"}]
    degraded_visible_rate = round(sum(bool(row.actual.get("degraded_visible")) for row in degraded_rows) / len(degraded_rows), 4) if degraded_rows else 1.0
    graph_path_rate = round(sum(bool(row.actual.get("graph_path_visible")) for row in rag_rows) / len(rag_rows), 4) if rag_rows else 0.0
    return {
        "evidence_source_id_coverage_rate": rate(results, "evidence_source_id_coverage_rate"),
        "rule_evidence_hit_rate": rate(results, "rule_evidence_hit_rate"),
        "lightrag_retrieval_hit_rate": rate(results, "lightrag_retrieval_hit_rate"),
        "local_knowledge_retrieval_hit_rate": local_hit_rate,
        "graphrag_fallback_visibility_rate": fallback_visible_rate,
        "degraded_visibility_rate": degraded_visible_rate,
        "graph_path_visibility_rate": graph_path_rate,
    }
