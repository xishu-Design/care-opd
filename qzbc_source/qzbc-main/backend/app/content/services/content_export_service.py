from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeEntry, KnowledgeSource, LearningCase, Quiz, RuleDefinition


class ContentExportService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def export_content(self) -> dict[str, Any]:
        return {
            "sources": [_source(row) for row in self.db.scalars(select(KnowledgeSource).order_by(KnowledgeSource.source_id)).all()],
            "knowledge_entries": [_entry(row) for row in self.db.scalars(select(KnowledgeEntry).order_by(KnowledgeEntry.entry_type, KnowledgeEntry.entry_id)).all()],
            "review_rules": [_rule(row) for row in self.db.scalars(select(RuleDefinition).order_by(RuleDefinition.rule_id)).all()],
            "learning_cases": [_case(row) for row in self.db.scalars(select(LearningCase).order_by(LearningCase.id)).all()],
            "quizzes": [_quiz(row) for row in self.db.scalars(select(Quiz).order_by(Quiz.quiz_id)).all()],
        }


def _source(row: KnowledgeSource) -> dict[str, Any]:
    return {
        "source_id": row.source_id,
        "source_title": row.source_title,
        "source_type": row.source_type,
        "source_version": row.source_version,
        "source_year": row.source_year,
        "evidence_level": row.evidence_level,
        "license_note": row.license_note,
    }


def _entry(row: KnowledgeEntry) -> dict[str, Any]:
    return {
        "entry_id": row.entry_id,
        "entry_type": row.entry_type,
        "title": row.title,
        "content": row.content,
        "keywords": row.keywords_json,
        "source_id": row.source_id,
        "source_title": row.source_title,
        "source_type": row.source_type,
        "metadata": row.metadata_json,
    }


def _rule(row: RuleDefinition) -> dict[str, Any]:
    return {
        "rule_id": row.rule_id,
        "rule_type": row.rule_type,
        "rule_category": row.rule_category,
        "name": row.name,
        "enabled": row.enabled,
        "risk_level": row.risk_level,
        "source_id": row.source_id,
        "source_record_id": row.source_record_id,
        "review_status": row.review_status,
        "applies_to_roles": row.applies_to_roles_json,
        "condition": row.condition_json,
        "message": row.message,
        "evidence_query": row.evidence_query,
        "requires_doctor_confirmation": row.requires_doctor_confirmation,
        "metadata": row.metadata_json,
    }


def _case(row: LearningCase) -> dict[str, Any]:
    return {
        "id": row.id,
        "title": row.title,
        "anonymized_case": row.anonymized_case,
        "standard_analysis": row.standard_analysis,
        "risk_points": row.risk_points,
        "evidence_chain": row.evidence_chain,
    }


def _quiz(row: Quiz) -> dict[str, Any]:
    return {
        "quiz_id": row.quiz_id,
        "case_id": row.case_id,
        "question_type": row.question_type,
        "question": row.question,
        "options": row.options_json,
        "answer": row.answer,
        "explanation": row.explanation,
        "knowledge_points": row.knowledge_points_json,
        "source_id": row.source_id,
        "metadata": row.metadata_json,
    }
