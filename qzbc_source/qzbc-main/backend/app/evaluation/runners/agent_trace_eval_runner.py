from fastapi import HTTPException
from sqlalchemy import select

from app.agents.services.resume_service import resume_doctor_review
from app.db.models import AgentTrace, AgentTraceEvent, PendingHumanAction
from app.evaluation.runners.base import BaseEvaluationRunner
from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult
from app.schemas.prescription import PrescriptionCreateRequest, PrescriptionItemIn
from app.services.prescription_service import create_draft_prescription, review_prescription


class AgentTraceEvaluationRunner(BaseEvaluationRunner):
    def run(self, cases: list[EvaluationCase]) -> list[EvaluationCaseResult]:
        if not cases:
            return []
        if "human_confirm" in self.disabled:
            return [self.result(case, False, {"assertion": case.input.get("assertion"), "ablation": "human_confirm"}) for case in cases]
        facts = self._exercise_flow()
        return [self.timed(case, lambda item, facts=facts: self._case(item, facts)) for case in cases]

    def _exercise_flow(self) -> dict:
        prescription = create_draft_prescription(
            self.db,
            self.seed["doctor"],
            PrescriptionCreateRequest(
                patient_id=self.seed["patient"].id,
                title="Phase12 HumanConfirm Evaluation",
                usage_method="每日一剂，遵医嘱",
                doctor_notes="Phase 12 interrupt/resume",
                items=[PrescriptionItemIn(herb_name="黄芪", dose="10", unit="g")],
            ),
        )
        reviewed = review_prescription(self.db, self.seed["doctor"], prescription.id)
        trace_id = str(reviewed.get("trace_id"))
        pending = self.db.scalar(select(PendingHumanAction).where(PendingHumanAction.trace_id == trace_id))
        trace = self.db.scalar(select(AgentTrace).where(AgentTrace.trace_id == trace_id))
        events_before = self.db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == trace_id)).all()
        resumed = resume_doctor_review(self.db, trace_id=trace_id, approved=True, comment="Phase 12 approved", doctor=self.seed["doctor"])
        duplicate_blocked = False
        try:
            resume_doctor_review(self.db, trace_id=trace_id, approved=True, comment="duplicate", doctor=self.seed["doctor"])
        except HTTPException as exc:
            duplicate_blocked = exc.status_code == 409
        events_after = self.db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == trace_id)).all()
        nodes = [event.node_name for event in events_after]
        return {
            "trace_id": trace_id,
            "evidence_packet_id": reviewed.get("evidence_packet_id"),
            "interrupt_created": reviewed.get("interrupted") is True,
            "pending_action_persisted": pending is not None,
            "trace_created": trace is not None,
            "human_confirm_node_present": "HumanConfirmNode" in nodes,
            "resume_approved": resumed.get("resume_action") == "approved",
            "prescription_confirmed": resumed.get("human_confirmed") is True,
            "duplicate_resume_blocked": duplicate_blocked,
            "resume_trace_continued": len(events_after) > len(events_before),
            "evidence_packet_linked": bool(reviewed.get("evidence_packet_id")),
            "publish_requires_confirmation": reviewed.get("requires_human_confirm") is True,
            "nodes": nodes,
        }

    def _case(self, case: EvaluationCase, facts: dict) -> EvaluationCaseResult:
        assertion = str(case.input["assertion"])
        passed = facts.get(assertion) is True
        return self.result(case, passed, {"assertion": assertion, "value": facts.get(assertion), "nodes": facts["nodes"]}, trace_id=facts["trace_id"], evidence_packet_id=facts["evidence_packet_id"])
