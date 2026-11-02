from __future__ import annotations

from time import perf_counter

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.services.checkpoint_service import get_langgraph_checkpointer
from app.agents.services.resume_service import resume_doctor_review
from app.core.enums import PrescriptionStatus, UserRole
from app.core.security import hash_password
from app.db.models import EvidencePacketLog, HarnessResult, HarnessRun, PendingHumanAction, Prescription, User, utcnow
from app.harness.case_schema import PHASE6, RunnerResult
from app.harness.phase5_runner import Phase5HarnessRunner, _category_summary, _tool_calls_for_trace
from app.harness.trace import get_trace_detail
from app.services.prescription_service import get_patient_visible_prescription, review_prescription
from app.core.config import get_settings


class Phase6HarnessRunner(Phase5HarnessRunner):
    def run(self) -> HarnessRun:
        started = perf_counter()
        started_at = utcnow()
        results = [
            self._timed("P6_INTERRUPT_001", "agent_interrupt_resume", "DoctorReviewGraph interrupts at HumanConfirmNode", "critical", self._interrupt_review),
            self._timed("P6_APPROVE_001", "agent_interrupt_resume", "Approved resume confirms but does not publish", "critical", self._approved_resume),
            self._timed("P6_REJECT_001", "agent_human_confirm", "Rejected resume keeps prescription reviewing", "critical", self._rejected_resume),
            self._timed("P6_PERMISSION_001", "agent_resume_permission", "Patient, student and unbound doctor cannot resume", "critical", self._resume_permission),
            self._timed("P6_IDEMPOTENCY_001", "agent_resume_idempotency", "Repeated resume is rejected", "critical", self._resume_idempotency),
            self._timed("P6_CHECKPOINT_001", "agent_checkpoint_resume", "Missing checkpoint returns explainable 409", "high", self._checkpoint_missing),
            self._timed("P6_TRACE_001", "agent_resume_trace", "Trace and Evidence Packet record human confirmation", "high", self._trace_evidence),
            self._timed("P6_FALLBACK_001", "agent_human_confirm", "Interrupt disabled keeps Phase 5 confirmation path", "high", self._fallback_mode),
        ]
        if self.suite == "quick":
            results = results[:4]
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        metrics = _phase6_metrics(results)
        run = HarnessRun(
            phase=PHASE6,
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
                    details={"message": result.details, "related_tool_calls": result.related_tool_calls},
                    trace_id=result.trace_id,
                    evidence_packet_id=result.related_evidence_packet_id,
                    failure_reason=result.failure_reason,
                    duration_ms=result.duration_ms,
                )
            )
        self.db.commit()
        self.db.refresh(run)
        return run

    def _interrupt_review(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        prescription = self._create_phase5_prescription()
        review = review_prescription(self.db, self.doctor, prescription.id)
        action = self._pending_action(review["trace_id"])
        try:
            get_patient_visible_prescription(self.db, self.patient.id, prescription.id)
            patient_visible = True
        except HTTPException as exc:
            patient_visible = exc.status_code != 404
        payload = action.interrupt_payload_json if action else {}
        passed = (
            review.get("interrupted") is True
            and review.get("requires_human_confirm") is True
            and bool(review.get("thread_id"))
            and action is not None
            and action.status == "pending"
            and payload.get("prescription_id") == prescription.id
            and bool(payload.get("evidence_packet_id"))
            and "risk_level" in payload
            and bool(payload.get("review_summary"))
            and patient_visible is False
            and self.db.get(Prescription, prescription.id).status == PrescriptionStatus.REVIEWING.value
        )
        actual = {
            "trace_id": review.get("trace_id"),
            "thread_id": review.get("thread_id"),
            "evidence_packet_id": review.get("evidence_packet_id"),
            "interrupted": review.get("interrupted"),
            "requires_human_confirm": review.get("requires_human_confirm"),
            "pending_action_status": action.status if action else None,
            "interrupt_payload": payload,
            "patient_visible_before_confirm": patient_visible,
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _approved_resume(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        prescription = self._create_phase5_prescription()
        review = review_prescription(self.db, self.doctor, prescription.id)
        trace_id = review["trace_id"]
        tool_calls_before = _tool_calls_for_trace(self.db, trace_id)
        packet_count_before = self._packet_count(trace_id)
        resumed = resume_doctor_review(self.db, trace_id=trace_id, approved=True, comment="同意本次审方结果", doctor=self.doctor)
        action = self._pending_action(trace_id)
        prescription = self.db.get(Prescription, prescription.id)
        try:
            get_patient_visible_prescription(self.db, self.patient.id, prescription.id)
            patient_visible_after_resume = True
        except HTTPException as exc:
            patient_visible_after_resume = exc.status_code != 404
        tool_calls_after = _tool_calls_for_trace(self.db, trace_id)
        packet_count_after = self._packet_count(trace_id)
        passed = (
            resumed.get("human_confirmed") is True
            and resumed.get("resume_action") == "approved"
            and prescription.status == PrescriptionStatus.DOCTOR_CONFIRMED.value
            and patient_visible_after_resume is False
            and action is not None
            and action.status == "approved"
            and tool_calls_after == tool_calls_before
            and packet_count_after == packet_count_before
        )
        actual = {
            "trace_id": trace_id,
            "evidence_packet_id": review.get("evidence_packet_id"),
            "resume_action": resumed.get("resume_action"),
            "human_confirmed": resumed.get("human_confirmed"),
            "prescription_status": prescription.status,
            "pending_action_status": action.status if action else None,
            "patient_visible_after_resume": patient_visible_after_resume,
            "tool_calls_before": tool_calls_before,
            "tool_calls_after": tool_calls_after,
            "evidence_packet_count_before": packet_count_before,
            "evidence_packet_count_after": packet_count_after,
            "tool_calls": tool_calls_after,
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _rejected_resume(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        prescription = self._create_phase5_prescription()
        review = review_prescription(self.db, self.doctor, prescription.id)
        resumed = resume_doctor_review(self.db, trace_id=review["trace_id"], approved=False, comment="需要人工调整后再确认", doctor=self.doctor)
        action = self._pending_action(review["trace_id"])
        prescription = self.db.get(Prescription, prescription.id)
        detail = get_trace_detail(self.db, review["trace_id"])
        event_types = {event["event_type"] for event in detail.get("events", [])}
        passed = (
            resumed.get("human_confirmed") is False
            and resumed.get("resume_action") == "rejected"
            and prescription.status == PrescriptionStatus.REVIEWING.value
            and action is not None
            and action.status == "rejected"
            and "human_rejected" in event_types
        )
        actual = {
            "trace_id": review["trace_id"],
            "evidence_packet_id": review.get("evidence_packet_id"),
            "resume_action": resumed.get("resume_action"),
            "human_confirmed": resumed.get("human_confirmed"),
            "prescription_status": prescription.status,
            "pending_action_status": action.status if action else None,
            "event_types": sorted(event_types),
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _resume_permission(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        prescription = self._create_phase5_prescription()
        review = review_prescription(self.db, self.doctor, prescription.id)
        denials = {
            "patient": self._resume_error(review["trace_id"], self.patient),
            "student": self._resume_error(review["trace_id"], self.student),
            "unbound_doctor": self._resume_error(review["trace_id"], self._unbound_doctor()),
        }
        action = self._pending_action(review["trace_id"])
        passed = all(item["status_code"] == 403 and item["error_code"] == "RESUME_PERMISSION_DENIED" for item in denials.values()) and action.status == "pending"
        actual = {
            "trace_id": review["trace_id"],
            "evidence_packet_id": review.get("evidence_packet_id"),
            "denials": denials,
            "pending_action_status": action.status if action else None,
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _resume_idempotency(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        prescription = self._create_phase5_prescription()
        review = review_prescription(self.db, self.doctor, prescription.id)
        first = resume_doctor_review(self.db, trace_id=review["trace_id"], approved=True, comment="同意", doctor=self.doctor)
        second = self._resume_error(review["trace_id"], self.doctor, approved=True)
        third = self._resume_error(review["trace_id"], self.doctor, approved=False)
        action = self._pending_action(review["trace_id"])
        passed = (
            first.get("resume_action") == "approved"
            and second["status_code"] == 409
            and second["error_code"] == "HUMAN_ACTION_ALREADY_RESUMED"
            and third["status_code"] == 409
            and third["error_code"] == "HUMAN_ACTION_ALREADY_RESUMED"
            and action.status == "approved"
        )
        actual = {
            "trace_id": review["trace_id"],
            "evidence_packet_id": review.get("evidence_packet_id"),
            "first_resume": first.get("resume_action"),
            "second_resume": second,
            "third_resume": third,
            "pending_action_status": action.status if action else None,
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _checkpoint_missing(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        prescription = self._create_phase5_prescription()
        review = review_prescription(self.db, self.doctor, prescription.id)
        get_langgraph_checkpointer.cache_clear()
        error = self._resume_error(review["trace_id"], self.doctor, approved=True)
        action = self._pending_action(review["trace_id"])
        detail = get_trace_detail(self.db, review["trace_id"])
        event_types = {event["event_type"] for event in detail.get("events", [])}
        passed = (
            error["status_code"] == 409
            and error["error_code"] == "CHECKPOINT_NOT_FOUND"
            and action is not None
            and action.status == "checkpoint_lost"
            and "checkpoint_missing" in event_types
        )
        actual = {
            "trace_id": review["trace_id"],
            "evidence_packet_id": review.get("evidence_packet_id"),
            "error": error,
            "pending_action_status": action.status if action else None,
            "event_types": sorted(event_types),
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _trace_evidence(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        prescription = self._create_phase5_prescription()
        review = review_prescription(self.db, self.doctor, prescription.id)
        resume_doctor_review(self.db, trace_id=review["trace_id"], approved=True, comment="同意", doctor=self.doctor)
        detail = get_trace_detail(self.db, review["trace_id"])
        event_types = {event["event_type"] for event in detail.get("events", [])}
        packets = detail.get("evidence_packets", [])
        human_confirm = packets[0].get("human_confirm", {}) if packets else {}
        pending_actions = detail.get("pending_human_actions", [])
        passed = (
            {"interrupt", "resume", "human_approved"}.issubset(event_types)
            and bool(packets)
            and packets[0]["packet_id"] == review.get("evidence_packet_id")
            and human_confirm.get("resumed") is True
            and human_confirm.get("approved") is True
            and bool(pending_actions)
            and pending_actions[0]["status"] == "approved"
        )
        actual = {
            "trace_id": review["trace_id"],
            "evidence_packet_id": review.get("evidence_packet_id"),
            "event_types": sorted(event_types),
            "human_confirm": human_confirm,
            "pending_human_actions": pending_actions,
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _fallback_mode(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        settings = get_settings()
        original = settings.langgraph_enable_interrupt
        settings.langgraph_enable_interrupt = False
        try:
            prescription = self._create_phase5_prescription()
            review = review_prescription(self.db, self.doctor, prescription.id)
            action = self._pending_action(review["trace_id"])
            prescription = self.db.get(Prescription, prescription.id)
        finally:
            settings.langgraph_enable_interrupt = original
        passed = (
            review.get("requires_human_confirm") is True
            and review.get("interrupted") is False
            and action is None
            and prescription.status == PrescriptionStatus.REVIEWING.value
        )
        actual = {
            "trace_id": review.get("trace_id"),
            "evidence_packet_id": review.get("evidence_packet_id"),
            "interrupted": review.get("interrupted"),
            "requires_human_confirm": review.get("requires_human_confirm"),
            "pending_action_status": action.status if action else None,
            "prescription_status": prescription.status,
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _pending_action(self, trace_id: str) -> PendingHumanAction | None:
        return self.db.scalar(select(PendingHumanAction).where(PendingHumanAction.trace_id == trace_id).order_by(PendingHumanAction.created_at.desc()))

    def _packet_count(self, trace_id: str) -> int:
        return self.db.scalar(select(func.count()).select_from(EvidencePacketLog).where(EvidencePacketLog.trace_id == trace_id)) or 0

    def _resume_error(self, trace_id: str, actor: User, approved: bool = True) -> dict:
        try:
            resume_doctor_review(self.db, trace_id=trace_id, approved=approved, comment="harness negative test", doctor=actor)
            return {"status_code": 200, "error_code": None}
        except HTTPException as exc:
            if isinstance(exc.detail, dict):
                return {"status_code": exc.status_code, "error_code": exc.detail.get("error_code"), "message": exc.detail.get("message")}
            return {"status_code": exc.status_code, "error_code": "http_error", "message": str(exc.detail)}

    def _unbound_doctor(self) -> User:
        user = self.db.scalar(select(User).where(User.username == "phase6_unbound_doctor"))
        if user:
            return user
        user = User(
            username="phase6_unbound_doctor",
            hashed_password=hash_password("password123"),
            role=UserRole.DOCTOR.value,
            display_name="Phase6 Unbound Doctor",
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user


def _rate(results: list[RunnerResult], category: str) -> float:
    rows = [result for result in results if result.category == category]
    if not rows:
        return 1.0
    return sum(1 for result in rows if result.passed) / len(rows)


def _phase6_metrics(results: list[RunnerResult]) -> dict:
    return {
        "agent_interrupt_resume_rate": _rate(results, "agent_interrupt_resume"),
        "agent_checkpoint_resume_rate": _rate(results, "agent_checkpoint_resume"),
        "agent_resume_permission_rate": _rate(results, "agent_resume_permission"),
        "agent_human_confirm_rate": _rate(results, "agent_human_confirm"),
        "agent_resume_idempotency_rate": _rate(results, "agent_resume_idempotency"),
        "agent_resume_trace_rate": _rate(results, "agent_resume_trace"),
        "demo_scenario_passed": all(result.passed for result in results),
        "critical_failed": sum(1 for result in results if not result.passed and result.severity == "critical"),
        "high_failed": sum(1 for result in results if not result.passed and result.severity == "high"),
    }


def _markdown_summary(total: int, passed: int, failed: int, metrics: dict) -> str:
    pass_rate = passed / total if total else 0.0
    return (
        "# GuardTCM Phase 6 Harness Report\n\n"
        f"本次 Harness 共运行 {total} 个 Human-in-the-loop 测试用例，通过 {passed} 个，失败 {failed} 个，通过率 {pass_rate:.0%}。\n\n"
        "核心链路：DoctorReviewGraph -> ReviewPersistNode -> HumanConfirmNode interrupt -> resume API -> PrescriptionStateService。\n\n"
        "关键结论：\n"
        f"- Interrupt/Resume：{metrics['agent_interrupt_resume_rate']:.0%}。\n"
        f"- 权限边界：{metrics['agent_resume_permission_rate']:.0%}。\n"
        f"- 幂等控制：{metrics['agent_resume_idempotency_rate']:.0%}。\n"
        f"- Checkpoint 恢复：{metrics['agent_checkpoint_resume_rate']:.0%}。\n"
        f"- Trace/Evidence：{metrics['agent_resume_trace_rate']:.0%}。\n"
    )
