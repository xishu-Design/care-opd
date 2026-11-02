from __future__ import annotations

from time import perf_counter
from typing import Callable

from sqlalchemy.orm import Session

from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult


class BaseEvaluationRunner:
    def __init__(self, db: Session, seed: dict, disabled: set[str] | None = None):
        self.db = db
        self.seed = seed
        self.disabled = disabled or set()

    def timed(self, case: EvaluationCase, handler: Callable[[EvaluationCase], EvaluationCaseResult]) -> EvaluationCaseResult:
        started = perf_counter()
        try:
            result = handler(case)
        except Exception as exc:
            result = self.result(case, False, {"error": str(exc)}, failure_reason=f"{exc.__class__.__name__}: {exc}")
        result.duration_ms = int((perf_counter() - started) * 1000)
        return result

    def result(
        self,
        case: EvaluationCase,
        passed: bool,
        actual: dict,
        *,
        trace_id: str | None = None,
        evidence_packet_id: str | None = None,
        failure_reason: str | None = None,
    ) -> EvaluationCaseResult:
        return EvaluationCaseResult(
            case_id=case.case_id,
            dataset=case.dataset,
            category=case.category,
            metric_name=case.metric_name,
            name=case.name,
            severity=case.severity,
            passed=passed,
            actual=actual,
            expected=case.expected,
            trace_id=trace_id,
            evidence_packet_id=evidence_packet_id,
            failure_reason=None if passed else failure_reason or "assertion_failed",
        )
