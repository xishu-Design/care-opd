from app.evaluation.metrics.common import rate
from app.evaluation.schemas import EvaluationCaseResult


def agent_metrics(results: list[EvaluationCaseResult]) -> dict[str, float]:
    agent_rows = [row for row in results if row.dataset == "human_confirm_cases"]
    trace_rate = round(sum(bool(row.trace_id) for row in agent_rows) / len(agent_rows), 4) if agent_rows else 0.0
    node_rate = round(sum("HumanConfirmNode" in row.actual.get("nodes", []) for row in agent_rows) / len(agent_rows), 4) if agent_rows else 0.0
    mcp_rows = [row for row in results if row.dataset == "mcp_permission_cases"]
    mcp_consistency = round(sum(row.actual.get("logged") is True for row in mcp_rows) / len(mcp_rows), 4) if mcp_rows else 0.0
    return {
        "langgraph_node_completeness_rate": node_rate,
        "trace_generation_rate": trace_rate,
        "human_confirm_interrupt_rate": rate(results, "human_confirm_interrupt_rate"),
        "resume_success_rate": rate(results, "resume_success_rate"),
        "mcp_tool_node_consistency_rate": mcp_consistency,
    }
