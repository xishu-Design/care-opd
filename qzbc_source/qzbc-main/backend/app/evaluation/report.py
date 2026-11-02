from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EvaluationResult, EvaluationRun
from app.evaluation.schemas import EvaluationCaseResult


def build_evaluation_report(
    *,
    run_id: str,
    suite: str,
    results: list[EvaluationCaseResult],
    metrics: dict[str, Any],
    scores: dict[str, float],
    duration_ms: int,
) -> dict[str, Any]:
    total = len(results)
    passed = sum(row.passed for row in results)
    failed_cases = [row.model_dump(mode="json") for row in results if not row.passed]
    light = metrics["evidence"]["lightrag_retrieval_hit_rate"]
    local = metrics["evidence"]["local_knowledge_retrieval_hit_rate"]
    improvement = round(light - local, 4)
    summary = (
        f"GuardTCM Phase 12 共评测 {total} 个系统行为，关键安全指标可逐项追溯。"
        f" LightRAG 相比 LocalKnowledge 的固定语料命中率变化为 {improvement:+.1%}，"
        "消融报告用于说明各治理层被移除后的能力下降。"
    )
    report = {
        "run_id": run_id,
        "phase": "phase_12_system_evaluation",
        "suite": suite,
        "dataset_version": "phase12-v1",
        "status": "completed",
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "duration_ms": duration_ms,
        **scores,
        "metrics": metrics,
        "ablation_summary": {},
        "competition_summary": summary,
        "key_failures": failed_cases[:20],
        "ppt_ready_tables": _ppt_tables(metrics, scores, total, passed),
        "ppt_ready_charts": _ppt_charts(metrics, scores),
    }
    report["markdown_report"] = render_markdown(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    safety = report["metrics"]["safety"]
    evidence = report["metrics"]["evidence"]
    agent = report["metrics"]["agent"]
    improvement = evidence["lightrag_retrieval_hit_rate"] - evidence["local_knowledge_retrieval_hit_rate"]
    return (
        "# GuardTCM Evaluation Report\n\n"
        f"本次评测共运行 {report['total_cases']} 个测试用例，通过率 {report['pass_rate']:.1%}。\n\n"
        "## 核心结论\n\n"
        f"1. 患者端高风险请求拒答率达到 {safety['patient_high_risk_refusal_rate']:.1%}。\n"
        f"2. MCP 越权工具调用拦截率达到 {safety['mcp_unauthorized_block_rate']:.1%}。\n"
        f"3. Evidence Packet source_id 覆盖率达到 {evidence['evidence_source_id_coverage_rate']:.1%}。\n"
        f"4. HumanConfirm 中断成功率 {agent['human_confirm_interrupt_rate']:.1%}，恢复成功率 {agent['resume_success_rate']:.1%}。\n"
        f"5. 使用 LightRAG 后证据命中率相比 LocalKnowledge 变化 {improvement:+.1%}。\n"
        f"6. LLM 输出安全分达到 {report['llm_safety_score']:.1%}。\n\n"
        "## 比赛结论\n\n"
        f"{report['competition_summary']}\n"
    )


def serialize_evaluation_run(run: EvaluationRun, include_report: bool = True) -> dict[str, Any]:
    payload = {
        "run_id": run.run_id,
        "suite": run.suite,
        "dataset_version": run.dataset_version,
        "status": run.status,
        "total_cases": run.total_cases,
        "passed": run.passed,
        "failed": run.failed,
        "pass_rate": run.pass_rate,
        "overall_score": run.overall_score,
        "safety_score": run.safety_score,
        "evidence_score": run.evidence_score,
        "agent_trace_score": run.agent_trace_score,
        "llm_safety_score": run.llm_safety_score,
        "metrics": run.metrics_json,
        "duration_ms": run.duration_ms,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }
    if include_report:
        payload.update(run.report_json)
        payload["markdown_report"] = run.markdown_report
    return payload


def evaluation_results(db: Session, run: EvaluationRun) -> list[dict[str, Any]]:
    rows = db.scalars(select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id).order_by(EvaluationResult.id)).all()
    return [
        {
            "case_id": row.case_id,
            "dataset": row.dataset,
            "category": row.category,
            "metric_name": row.metric_name,
            "severity": row.severity,
            "passed": row.passed,
            "actual": row.actual_json,
            "expected": row.expected_json,
            "trace_id": row.trace_id,
            "evidence_packet_id": row.evidence_packet_id,
            "failure_reason": row.failure_reason,
            "duration_ms": row.duration_ms,
        }
        for row in rows
    ]


def _ppt_tables(metrics: dict[str, Any], scores: dict[str, float], total: int, passed: int) -> list[dict[str, Any]]:
    score_rows = [{"dimension": key, "score": value} for key, value in scores.items()]
    metric_rows = [{"group": group, "metric": key, "rate": value} for group, values in metrics.items() for key, value in values.items()]
    return [
        {"title": "GuardTCM Phase 12 总览", "columns": ["total_cases", "passed", "pass_rate"], "rows": [{"total_cases": total, "passed": passed, "pass_rate": round(passed / total, 4) if total else 0.0}]},
        {"title": "系统能力评分", "columns": ["dimension", "score"], "rows": score_rows},
        {"title": "核心指标明细", "columns": ["group", "metric", "rate"], "rows": metric_rows},
    ]


def _ppt_charts(metrics: dict[str, Any], scores: dict[str, float]) -> list[dict[str, Any]]:
    evidence = metrics["evidence"]
    return [
        {"title": "GuardTCM 核心能力评分", "type": "bar", "categories": list(scores), "series": [{"name": "score", "data": list(scores.values())}]},
        {"title": "LightRAG 与 LocalKnowledge 命中率", "type": "bar", "categories": ["LocalKnowledge", "LightRAG"], "series": [{"name": "hit_rate", "data": [evidence["local_knowledge_retrieval_hit_rate"], evidence["lightrag_retrieval_hit_rate"]]}]},
    ]
