from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.content.services.content_stats_service import content_stats
from app.content.validators.safety_validator import validate_patient_safety
from app.db.models import KnowledgeEntry, LearningCase, Quiz, RuleDefinition


def content_quality_report(db: Session) -> dict:
    stats = content_stats(db)
    critical_issues: list[dict] = []
    high_issues: list[dict] = []
    warnings: list[dict] = []
    for entry in db.scalars(select(KnowledgeEntry)).all():
        if not entry.source_id:
            critical_issues.append({"code": "ENTRY_MISSING_SOURCE", "entry_id": entry.entry_id})
        metadata = entry.metadata_json if isinstance(entry.metadata_json, dict) else {}
        if entry.entry_type in {"herb", "formula"} and "patient" in metadata.get("visibility", []):
            high_issues.append({"code": "OPEN_DATA_PATIENT_VISIBLE", "entry_id": entry.entry_id})
        for issue in validate_patient_safety(metadata):
            high_issues.append({"code": issue["code"], "entry_id": entry.entry_id, "message": issue["message"]})
    for case in db.scalars(select(LearningCase)).all():
        payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
        if payload.get("deidentified") is not True:
            critical_issues.append({"code": "CASE_NOT_DEIDENTIFIED", "case_id": payload.get("case_id")})
        if payload.get("visibility") != "student_learning":
            high_issues.append({"code": "CASE_INVALID_VISIBILITY", "case_id": payload.get("case_id")})
    for quiz in db.scalars(select(Quiz)).all():
        if quiz.answer not in quiz.options_json:
            high_issues.append({"code": "QUIZ_ANSWER_NOT_IN_OPTIONS", "quiz_id": quiz.quiz_id})
        if not quiz.explanation:
            high_issues.append({"code": "QUIZ_MISSING_EXPLANATION", "quiz_id": quiz.quiz_id})
    for rule in db.scalars(select(RuleDefinition)).all():
        if not rule.source_id:
            critical_issues.append({"code": "RULE_MISSING_SOURCE", "rule_id": rule.rule_id})
        if not rule.evidence_query:
            high_issues.append({"code": "RULE_MISSING_EVIDENCE_QUERY", "rule_id": rule.rule_id})
        if rule.rule_category == "medical_review" and "patient" in rule.applies_to_roles_json:
            high_issues.append({"code": "PATIENT_ROLE_IN_MEDICAL_RULE", "rule_id": rule.rule_id})
    metrics = {
        "content_valid_rate": 1.0 if not critical_issues and not high_issues else 0.0,
        "content_source_coverage_rate": stats["coverage"]["source_coverage_rate"],
        "quiz_case_binding_rate": stats["coverage"]["quiz_case_binding_rate"],
        "rule_evidence_binding_rate": stats["coverage"]["rule_source_binding_rate"],
        "patient_safe_rate": _patient_safe_rate(db),
        "learning_case_deid_pass_rate": 1.0 if stats["learning_cases"] == 0 else _deid_rate(db),
        "deidentification_pass_rate": 1.0 if stats["learning_cases"] == 0 else _deid_rate(db),
        "rule_valid_rate": _rule_valid_rate(db),
        "quiz_valid_rate": _quiz_valid_rate(db),
        "critical_issue_count": len(critical_issues),
        "high_issue_count": len(high_issues),
    }
    return {
        "report_id": "content_quality_latest",
        "metrics": metrics,
        "critical_issues": critical_issues,
        "high_issues": high_issues,
        "warnings": warnings,
    }


def _deid_rate(db: Session) -> float:
    cases = db.scalars(select(LearningCase)).all()
    if not cases:
        return 1.0
    passed = sum(1 for case in cases if isinstance(case.anonymized_case, dict) and case.anonymized_case.get("deidentified") is True)
    return passed / len(cases)


def _patient_safe_rate(db: Session) -> float:
    patient_entries = []
    for entry in db.scalars(select(KnowledgeEntry)).all():
        metadata = entry.metadata_json if isinstance(entry.metadata_json, dict) else {}
        visibility = metadata.get("visibility", [])
        values = visibility if isinstance(visibility, list) else [visibility]
        if "patient" in values:
            patient_entries.append(metadata)
    if not patient_entries:
        return 1.0
    return sum(1 for item in patient_entries if not validate_patient_safety(item)) / len(patient_entries)


def _rule_valid_rate(db: Session) -> float:
    rules = db.scalars(select(RuleDefinition)).all()
    if not rules:
        return 1.0
    valid = sum(1 for rule in rules if rule.source_id and rule.evidence_query and "patient" not in rule.applies_to_roles_json)
    return valid / len(rules)


def _quiz_valid_rate(db: Session) -> float:
    quizzes = db.scalars(select(Quiz)).all()
    if not quizzes:
        return 1.0
    valid = sum(1 for quiz in quizzes if quiz.answer in quiz.options_json and bool(quiz.explanation) and bool(quiz.knowledge_points_json))
    return valid / len(quizzes)
