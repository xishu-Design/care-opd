from __future__ import annotations

from time import perf_counter
from typing import Callable

from sqlalchemy.orm import Session

from app.db.models import HarnessResult, HarnessRun, utcnow
from app.demo.schemas import DemoResetRequest
from app.demo.service import DemoService
from app.harness.case_schema import PHASE11, RunnerResult
from app.harness.phase5_runner import _category_summary


class Phase11HarnessRunner:
    def __init__(self, db: Session, suite: str = "full", reset_demo_data: bool = True):
        self.db = db
        self.suite = suite
        self.reset_demo_data = reset_demo_data
        self.demo = DemoService(db)

    def run(self) -> HarnessRun:
        started = perf_counter()
        started_at = utcnow()
        results = [
            self._timed("P11_STATUS_001", "demo_status", "Demo status is readable after reset", "critical", self._status),
            self._timed("P11_RESET_001", "demo_reset", "Demo reset creates idempotent demo data", "critical", self._reset),
            self._timed("P11_SMOKE_001", "demo_smoke", "Demo smoke checks critical demo paths", "critical", self._smoke),
            self._timed("P11_ACCOUNTS_001", "demo_accounts", "Demo accounts and binding exist", "critical", self._accounts),
            self._timed("P11_READINESS_001", "demo_frontend_readiness", "Readiness payload exposes frontend console fields", "high", self._frontend_readiness),
        ]
        if self.suite == "quick":
            results = results[:3]
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        metrics = {
            "demo_readiness_rate": passed / total if total else 0.0,
            "critical_failed": sum(1 for result in results if not result.passed and result.severity == "critical"),
            "smoke_passed": next((result.passed for result in results if result.case_id == "P11_SMOKE_001"), False),
        }
        run = HarnessRun(
            phase=PHASE11,
            suite=self.suite,
            status="completed",
            total_cases=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total else 0.0,
            categories=_category_summary(results),
            metrics_json=metrics,
            markdown_summary=_markdown_summary(total, passed, total - passed, metrics),
            started_at=started_at,
            ended_at=utcnow(),
            duration_ms=int((perf_counter() - started) * 1000),
        )
        self.db.add(run)
        self.db.flush()
        for result in results:
            self.db.add(
                HarnessResult(
                    run_id=run.id,
                    case_id=result.case_id,
                    category=result.category,
                    name=result.details,
                    severity=result.severity,
                    passed=result.passed,
                    actual_json=result.actual,
                    expected_json=result.expected,
                    details={"message": result.details},
                    failure_reason=result.failure_reason,
                    duration_ms=result.duration_ms,
                )
            )
        self.db.commit()
        self.db.refresh(run)
        return run

    def _timed(self, case_id: str, category: str, name: str, severity: str, handler: Callable[[str, str, str, str], RunnerResult]) -> RunnerResult:
        started = perf_counter()
        try:
            result = handler(case_id, category, name, severity)
        except Exception as exc:
            result = self._result(case_id, category, name, severity, False, {"error": str(exc)}, str(exc))
        result.duration_ms = int((perf_counter() - started) * 1000)
        return result

    def _result(self, case_id: str, category: str, name: str, severity: str, passed: bool, actual: dict, failure_reason: str | None = None) -> RunnerResult:
        return RunnerResult(
            case_id=case_id,
            category=category,
            passed=passed,
            severity=severity,
            actual=actual,
            expected={"name": name},
            details=name,
            failure_reason=None if passed else failure_reason or "assertion_failed",
        )

    def _status(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        if self.reset_demo_data:
            self.demo.reset_demo_data(DemoResetRequest())
        status = self.demo.get_demo_status()
        passed = status["backend"]["database_ok"] is True and status["readiness"]["ready"] is True
        return self._result(case_id, category, name, severity, passed, status)

    def _reset(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        first = self.demo.reset_demo_data(DemoResetRequest())
        second = self.demo.reset_demo_data(DemoResetRequest())
        passed = first["status"] == "success" and second["status"] == "success" and first["created"] == second["created"]
        return self._result(case_id, category, name, severity, passed, {"first": first, "second": second})

    def _smoke(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self.demo.reset_demo_data(DemoResetRequest())
        smoke = self.demo.run_demo_smoke().model_dump(mode="json")
        return self._result(case_id, category, name, severity, bool(smoke["passed"]), smoke)

    def _accounts(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        status = self.demo.get_demo_status()
        passed = all(status["accounts"].values()) and status["bindings"]["doctor_patient_binding_exists"]
        return self._result(case_id, category, name, severity, passed, {"accounts": status["accounts"], "bindings": status["bindings"]})

    def _frontend_readiness(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        status = self.demo.get_demo_status()
        required = {"environment", "backend", "accounts", "bindings", "demo_data", "content", "llm", "lightrag", "harness", "readiness"}
        passed = required.issubset(status.keys()) and "warnings" in status["readiness"]
        return self._result(case_id, category, name, severity, passed, {"keys": sorted(status.keys()), "readiness": status["readiness"]})


def _markdown_summary(total: int, passed: int, failed: int, metrics: dict) -> str:
    return (
        "# Phase 11 Demo Readiness Harness\n\n"
        f"- Total cases: {total}\n"
        f"- Passed: {passed}\n"
        f"- Failed: {failed}\n"
        f"- Demo readiness rate: {metrics['demo_readiness_rate']:.0%}\n"
        f"- Critical failed: {metrics['critical_failed']}\n"
    )
