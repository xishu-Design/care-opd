from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import EvaluationResult, EvaluationRun, utcnow
from app.evaluation.dataset_loader import load_all_datasets
from app.evaluation.metrics import agent_metrics, evidence_metrics, llm_metrics, safety_metrics
from app.evaluation.report import build_evaluation_report
from app.evaluation.runners.agent_trace_eval_runner import AgentTraceEvaluationRunner
from app.evaluation.runners.doctor_eval_runner import DoctorEvaluationRunner
from app.evaluation.runners.evidence_eval_runner import EvidenceEvaluationRunner
from app.evaluation.runners.lightrag_eval_runner import LightRAGEvaluationRunner
from app.evaluation.runners.llm_eval_runner import LLMEvaluationRunner
from app.evaluation.runners.mcp_eval_runner import MCPEvaluationRunner
from app.evaluation.runners.memory_eval_runner import MemoryEvaluationRunner
from app.evaluation.runners.safety_eval_runner import SafetyEvaluationRunner
from app.evaluation.schemas import EvaluationCaseResult
from app.harness.demo import ensure_phase4_demo_seed, reset_phase4_demo_data


class EvaluationRunner:
    def __init__(self, db: Session, *, suite: str = "full", reset_demo_data: bool = True, actor_user_id: int | None = None, disabled: set[str] | None = None):
        self.db = db
        self.suite = suite
        self.actor_user_id = actor_user_id
        self.disabled = disabled or set()
        if reset_demo_data:
            reset_phase4_demo_data(db)
        self.seed = ensure_phase4_demo_seed(db)

    def run(self) -> EvaluationRun:
        started = perf_counter()
        started_at = utcnow()
        run_id = f"eval_{uuid4().hex}"
        datasets = load_all_datasets(self.suite)
        common = {"db": self.db, "seed": self.seed, "disabled": self.disabled}
        safety = SafetyEvaluationRunner(**common)
        results: list[EvaluationCaseResult] = []
        results.extend(safety.run_patient_cases(datasets["patient_safety_cases.json"]))
        results.extend(DoctorEvaluationRunner(**common).run(datasets["doctor_review_cases.json"]))
        results.extend(safety.run_student_cases(datasets["student_deid_cases.json"]))
        results.extend(MCPEvaluationRunner(**common).run(datasets["mcp_permission_cases.json"]))
        results.extend(EvidenceEvaluationRunner(**common).run(datasets["evidence_packet_cases.json"]))
        results.extend(LightRAGEvaluationRunner(**common).run(datasets["lightrag_retrieval_cases.json"]))
        results.extend(MemoryEvaluationRunner(**common).run(datasets["mem0_isolation_cases.json"]))
        results.extend(AgentTraceEvaluationRunner(**common).run(datasets["human_confirm_cases.json"]))
        results.extend(LLMEvaluationRunner(**common).run(datasets["llm_safety_cases.json"]))
        metrics = self._metrics(results)
        scores = self._scores(metrics)
        duration_ms = int((perf_counter() - started) * 1000)
        report = build_evaluation_report(run_id=run_id, suite=self.suite, results=results, metrics=metrics, scores=scores, duration_ms=duration_ms)
        passed = sum(row.passed for row in results)
        run = EvaluationRun(
            run_id=run_id,
            actor_user_id=self.actor_user_id,
            suite=self.suite,
            dataset_version="phase12-v1",
            status="completed",
            total_cases=len(results),
            passed=passed,
            failed=len(results) - passed,
            pass_rate=passed / len(results) if results else 0.0,
            metrics_json=metrics,
            report_json={key: value for key, value in report.items() if key != "markdown_report"},
            markdown_report=report["markdown_report"],
            started_at=started_at,
            ended_at=utcnow(),
            duration_ms=duration_ms,
            **scores,
        )
        self.db.add(run)
        self.db.flush()
        for result in results:
            self.db.add(
                EvaluationResult(
                    evaluation_run_id=run.id,
                    case_id=result.case_id,
                    dataset=result.dataset,
                    category=result.category,
                    metric_name=result.metric_name,
                    severity=result.severity,
                    passed=result.passed,
                    actual_json=result.actual,
                    expected_json=result.expected,
                    trace_id=result.trace_id,
                    evidence_packet_id=result.evidence_packet_id,
                    failure_reason=result.failure_reason,
                    duration_ms=result.duration_ms,
                )
            )
        self.db.commit()
        self.db.refresh(run)
        return run

    @staticmethod
    def _metrics(results: list[EvaluationCaseResult]) -> dict:
        return {
            "safety": safety_metrics(results),
            "evidence": evidence_metrics(results),
            "agent": agent_metrics(results),
            "llm": llm_metrics(results),
        }

    @staticmethod
    def _scores(metrics: dict) -> dict[str, float]:
        safety_values = [value for key, value in metrics["safety"].items() if key != "student_identity_leak_rate"]
        safety_values.append(1 - metrics["safety"]["student_identity_leak_rate"])
        safety_score = round(sum(safety_values) / len(safety_values), 4)
        evidence_score = round(sum(metrics["evidence"].values()) / len(metrics["evidence"]), 4)
        agent_score = round(sum(metrics["agent"].values()) / len(metrics["agent"]), 4)
        llm_score = round(sum(value for key, value in metrics["llm"].items() if key != "fallback_used_rate") / (len(metrics["llm"]) - 1), 4)
        return {
            "overall_score": round(0.35 * safety_score + 0.25 * evidence_score + 0.2 * agent_score + 0.2 * llm_score, 4),
            "safety_score": safety_score,
            "evidence_score": evidence_score,
            "agent_trace_score": agent_score,
            "llm_safety_score": llm_score,
        }
