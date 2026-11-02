from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.content.validators.safety_validator import validate_patient_safety
from app.db.models import KnowledgeEntry, KnowledgeSource, LearningCase, Quiz, RuleDefinition


def content_stats(db: Session) -> dict:
    source_count = db.scalar(select(func.count(KnowledgeSource.id))) or 0
    entry_count = db.scalar(select(func.count(KnowledgeEntry.id))) or 0
    herb_count = db.scalar(select(func.count(KnowledgeEntry.id)).where(KnowledgeEntry.entry_type == "herb")) or 0
    formula_count = db.scalar(select(func.count(KnowledgeEntry.id)).where(KnowledgeEntry.entry_type == "formula")) or 0
    contraindication_count = db.scalar(select(func.count(KnowledgeEntry.id)).where(KnowledgeEntry.entry_type == "contraindication")) or 0
    diet_count = db.scalar(select(func.count(KnowledgeEntry.id)).where(KnowledgeEntry.entry_type == "diet_precaution")) or 0
    education_count = db.scalar(select(func.count(KnowledgeEntry.id)).where(KnowledgeEntry.entry_type == "public_education")) or 0
    rule_count = db.scalar(select(func.count(RuleDefinition.id))) or 0
    learning_case_count = db.scalar(select(func.count(LearningCase.id))) or 0
    quiz_count = db.scalar(select(func.count(Quiz.id))) or 0
    student_learning_cases = 0
    for case in db.scalars(select(LearningCase)).all():
        if isinstance(case.anonymized_case, dict) and case.anonymized_case.get("visibility") == "student_learning":
            student_learning_cases += 1
    return {
        "sources": source_count,
        "knowledge_entries": entry_count,
        "herbs": herb_count,
        "formulas": formula_count,
        "contraindications": contraindication_count,
        "diet_precautions": diet_count,
        "public_education": education_count,
        "rules": rule_count,
        "learning_cases": learning_case_count,
        "quizzes": quiz_count,
        "patient_safe_entries": _patient_safe_entries(db),
        "student_learning_cases": student_learning_cases,
        "coverage": {
            "source_coverage_rate": 1.0 if entry_count == 0 else _source_coverage(db),
            "quiz_case_binding_rate": 1.0 if quiz_count == 0 else _quiz_case_binding(db),
            "rule_source_binding_rate": 1.0 if rule_count == 0 else _rule_source_binding(db),
        },
    }


def _source_coverage(db: Session) -> float:
    entries = db.scalars(select(KnowledgeEntry)).all()
    if not entries:
        return 1.0
    with_source = sum(1 for entry in entries if entry.source_id)
    return with_source / len(entries)


def _quiz_case_binding(db: Session) -> float:
    quizzes = db.scalars(select(Quiz)).all()
    case_ids = {case.anonymized_case.get("case_id") for case in db.scalars(select(LearningCase)).all() if isinstance(case.anonymized_case, dict)}
    if not quizzes:
        return 1.0
    bound = sum(1 for quiz in quizzes if quiz.case_id in case_ids)
    return bound / len(quizzes)


def _rule_source_binding(db: Session) -> float:
    rules = db.scalars(select(RuleDefinition)).all()
    if not rules:
        return 1.0
    bound = sum(1 for rule in rules if rule.source_id and rule.evidence_query)
    return bound / len(rules)


def _patient_safe_entries(db: Session) -> int:
    count = 0
    for entry in db.scalars(select(KnowledgeEntry)).all():
        metadata = entry.metadata_json or {}
        visibility = metadata.get("visibility", [])
        values = visibility if isinstance(visibility, list) else [visibility]
        if "patient" in values and not validate_patient_safety(metadata):
            count += 1
    return count
