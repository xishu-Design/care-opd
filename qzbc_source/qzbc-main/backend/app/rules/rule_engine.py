from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AllergyRecord, Prescription, RuleExecutionLog
from app.rules.rule_loader import get_rules
from app.rules.rule_models import RuleFinding, StructuredRule
from app.services.allergy_service import allergy_conflicts_for_herbs
from app.services.herb_incompatibility_service import find_herb_incompatibilities

RISK_ORDER = {"green": 0, "yellow": 1, "orange": 2, "red": 3, "high": 3}


def run_structured_prescription_review(db: Session, prescription: Prescription, session_id: str | None = None) -> dict:
    findings: list[RuleFinding] = []
    for rule in get_rules(db, include_disabled=False):
        finding = _evaluate_rule(db, prescription, rule)
        if finding:
            findings.append(finding)

    risk_level = _overall_risk(findings)
    review_summary = _summary_for(findings)
    finding_payloads = [finding.model_dump() for finding in findings]
    db.add(
        RuleExecutionLog(
            session_id=session_id,
            prescription_id=prescription.id,
            patient_id=prescription.patient_id,
            triggered_rules_json=finding_payloads,
            risk_level=risk_level,
        )
    )
    db.flush()
    return {
        "risk_level": risk_level,
        "rule_findings": finding_payloads,
        "review_summary": review_summary,
        "prescription_snapshot": {
            "prescription_id": prescription.id,
            "patient_id": prescription.patient_id,
            "title": prescription.title,
            "symptom_context": prescription.symptom_context,
            "patient_age_at_visit": prescription.patient_age_at_visit,
            "patient_sex_at_visit": prescription.patient_sex_at_visit,
            "usage_method": prescription.usage_method,
            "doctor_notes": prescription.doctor_notes,
            "items": [
                {"herb_name": item.herb_name, "dose": item.dose, "unit": item.unit, "notes": item.notes}
                for item in prescription.items
            ],
        },
        "patient_allergy_summary": _allergy_summary(db, prescription),
        "allergy_matched_herbs": _allergy_conflicts(db, prescription),
        "herb_incompatibility_matches": _herb_incompatibilities(prescription),
    }


def _evaluate_rule(db: Session, prescription: Prescription, rule: StructuredRule) -> RuleFinding | None:
    condition = rule.condition
    if condition.get("field_missing") == "usage" and not (prescription.usage_method or "").strip():
        finding = _finding(rule, "administration_method_missing")
        payload = finding.model_dump()
        complete_dose_count = sum(bool((item.dose or "").strip() and (item.unit or "").strip()) for item in prescription.items)
        if prescription.items and complete_dose_count == len(prescription.items):
            payload["message"] = "各味药的剂量和单位已经填写；当前缺少的是处方级用法说明（如煎服方式、每日频次、分服方法）。"
        elif complete_dose_count:
            payload["message"] = f"已有 {complete_dose_count} 味药填写剂量和单位；仍有药材剂量信息不完整，且处方级用法说明尚未填写。"
        else:
            payload["message"] = "当前尚未填写完整的单味药剂量信息和处方级用法说明。"
        return RuleFinding.model_validate(payload)
    if condition.get("field_missing") == "doctor_notes" and not (prescription.doctor_notes or "").strip():
        return _finding(rule, "doctor_notes_missing")
    if condition.get("prescription_herb_in_patient_allergies"):
        matched = _allergy_conflicts(db, prescription)
        if matched:
            finding = _finding(rule, "allergy_conflict")
            payload = finding.model_dump()
            payload["matched_herbs"] = matched
            payload["message"] = f"{rule.message} 匹配药材：{'、'.join(matched)}。"
            return RuleFinding.model_validate(payload)
    if condition.get("prescription_has_traditional_incompatibility"):
        matched_pairs = _herb_incompatibilities(prescription)
        if matched_pairs:
            finding = _finding(rule, "traditional_herb_incompatibility")
            payload = finding.model_dump()
            payload["matched_pairs"] = matched_pairs
            pair_text = "；".join(
                f"{item['herb_a']}—{item['herb_b']}（{item['category']}）" for item in matched_pairs
            )
            payload["message"] = f"{rule.message} 匹配药对：{pair_text}。"
            return RuleFinding.model_validate(payload)
    return None


def _finding(rule: StructuredRule, trigger: str) -> RuleFinding:
    return RuleFinding(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type,
        rule_category=rule.rule_category,
        risk_level=rule.risk_level,
        trigger=trigger,
        message=rule.message,
        evidence_query=rule.evidence_query,
        source_id=rule.source_id,
        source_record_id=rule.source_record_id,
        review_status=rule.review_status,
        requires_doctor_confirmation=rule.requires_doctor_confirmation,
    )


def _allergy_conflicts(db: Session, prescription: Prescription) -> list[str]:
    return allergy_conflicts_for_herbs(
        _allergy_summary(db, prescription),
        (item.herb_name for item in prescription.items),
    )


def _allergy_summary(db: Session, prescription: Prescription) -> str:
    allergy = db.scalar(select(AllergyRecord).where(AllergyRecord.patient_id == prescription.patient_id))
    return allergy.summary if allergy else ""


def _herb_incompatibilities(prescription: Prescription) -> list[dict[str, str]]:
    return find_herb_incompatibilities(item.herb_name for item in prescription.items)


def _overall_risk(findings: list[RuleFinding]) -> str:
    if not findings:
        return "green"
    return max((finding.risk_level for finding in findings), key=lambda risk: RISK_ORDER.get(risk, 0))


def _summary_for(findings: list[RuleFinding]) -> str:
    if not findings:
        return "结构化规则未发现阻断性风险，仍需医生最终确认。"
    names = "；".join(finding.message for finding in findings)
    return f"结构化规则触发 {len(findings)} 项风险：{names}"
