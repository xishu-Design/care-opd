from fastapi import HTTPException

from app.core.enums import PrescriptionStatus
from app.evaluation.runners.base import BaseEvaluationRunner
from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult
from app.rules.rule_engine import run_structured_prescription_review
from app.schemas.prescription import PrescriptionCreateRequest, PrescriptionItemIn
from app.services.learning_service import create_deidentified_learning_case
from app.services.patient_service import get_patient_for_doctor
from app.services.prescription_service import create_draft_prescription, get_patient_visible_prescription, publish_prescription, review_prescription


class DoctorEvaluationRunner(BaseEvaluationRunner):
    def run(self, cases: list[EvaluationCase]) -> list[EvaluationCaseResult]:
        return [self.timed(case, self._case) for case in cases]

    def _case(self, case: EvaluationCase) -> EvaluationCaseResult:
        if case.category == "doctor_access":
            return self._access(case)
        if case.category == "prescription_state":
            return self._state(case)
        return self._rule(case)

    def _access(self, case: EvaluationCase) -> EvaluationCaseResult:
        try:
            get_patient_for_doctor(self.db, self.seed["doctor"].id, self.seed["unbound_patient"].id)
            status_code = 200
        except HTTPException as exc:
            status_code = exc.status_code
        return self.result(case, status_code == 403, {"status_code": status_code, "resource": case.input.get("resource")})

    def _state(self, case: EvaluationCase) -> EvaluationCaseResult:
        scenario = str(case.input["scenario"])
        prescription = self._prescription(scenario)
        consistent = False
        actual: dict = {"scenario": scenario, "initial_status": prescription.status}
        if scenario == "draft_not_visible":
            try:
                get_patient_visible_prescription(self.db, self.seed["patient"].id, prescription.id)
            except HTTPException as exc:
                consistent = exc.status_code == 404
                actual["status_code"] = exc.status_code
        elif scenario == "review_requires_confirm":
            reviewed = review_prescription(self.db, self.seed["doctor"], prescription.id)
            consistent = reviewed.get("interrupted") is True and reviewed.get("requires_human_confirm") is True
            actual.update({"interrupted": reviewed.get("interrupted"), "requires_human_confirm": reviewed.get("requires_human_confirm")})
        elif scenario == "confirm_before_publish":
            try:
                publish_prescription(self.db, self.seed["doctor"], prescription.id)
            except HTTPException as exc:
                consistent = exc.status_code == 409
                actual["status_code"] = exc.status_code
        elif scenario == "publish_after_confirm":
            prescription.status = PrescriptionStatus.DOCTOR_CONFIRMED.value
            self.db.flush()
            published = publish_prescription(self.db, self.seed["doctor"], prescription.id)
            consistent = published.status == PrescriptionStatus.PUBLISHED_TO_PATIENT.value
            actual["final_status"] = published.status
        elif scenario == "student_copy_after_deid":
            prescription.status = PrescriptionStatus.DOCTOR_CONFIRMED.value
            self.db.flush()
            learning_case = create_deidentified_learning_case(self.db, self.seed["doctor"], prescription.id)
            consistent = learning_case.anonymized_case.get("deidentified") is True
            actual.update({"deidentified": learning_case.anonymized_case.get("deidentified"), "visibility": learning_case.anonymized_case.get("visibility")})
        return self.result(case, consistent, {**actual, "consistent": consistent})

    def _rule(self, case: EvaluationCase) -> EvaluationCaseResult:
        scenario = str(case.input["scenario"])
        prescription = self._prescription(case.case_id, scenario=scenario, herb=str(case.input["herb"]))
        reviewed = run_structured_prescription_review(self.db, prescription, session_id=f"eval-{case.case_id}")
        findings = reviewed["rule_findings"]
        source_ids = [finding.get("source_id") for finding in findings if finding.get("source_id")]
        passed = bool(findings) and bool(source_ids)
        return self.result(case, passed, {"finding": bool(findings), "source_ids": source_ids, "risk_level": reviewed["risk_level"]})

    def _prescription(self, suffix: str, *, scenario: str = "normal", herb: str = "黄芪"):
        usage = None if scenario == "missing_usage" else "每日一剂，遵医嘱"
        notes = None if scenario == "missing_notes" else "Phase 12 evaluation"
        return create_draft_prescription(
            self.db,
            self.seed["doctor"],
            PrescriptionCreateRequest(
                patient_id=self.seed["patient"].id,
                title=f"Phase12 Evaluation {suffix}",
                usage_method=usage,
                doctor_notes=notes,
                items=[PrescriptionItemIn(herb_name=herb, dose="10", unit="g")],
            ),
        )
