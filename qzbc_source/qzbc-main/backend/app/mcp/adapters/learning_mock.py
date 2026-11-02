from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import LearningCase, Quiz
from app.mcp.contracts import ToolRequest
from app.services.learning_service import contextualize_student_quiz, is_student_learning_case


SENSITIVE_CASE_KEYS = {
    "address",
    "display_name",
    "full_name",
    "id_card",
    "patient_id",
    "patient_name",
    "phone",
    "real_name",
}


def _case_payload(case: LearningCase | None) -> dict:
    if not case or not is_student_learning_case(case):
        return {}
    anonymized_case = _strip_sensitive_fields(case.anonymized_case)
    return {
        "id": case.id,
        "title": case.title,
        "anonymized_case": anonymized_case,
        "standard_analysis": case.standard_analysis,
        "risk_points": case.risk_points,
        "evidence_chain": case.evidence_chain,
        "source_id": anonymized_case.get("source_id", ""),
        "source_record_id": anonymized_case.get("source_record_id", ""),
        "deidentified": True,
        "visibility": "student_learning",
    }


def execute(db: Session, request: ToolRequest) -> dict:
    case_id = request.arguments.get("learning_case_id") or request.context.learning_case_id
    case = db.get(LearningCase, case_id) if case_id else None
    if request.tool_name == "list_learning_cases":
        cases = db.scalars(select(LearningCase).order_by(LearningCase.created_at.desc())).all()
        return {"learning_cases": [_case_payload(item) for item in cases if is_student_learning_case(item)]}
    if request.tool_name == "get_learning_case":
        if not is_student_learning_case(case):
            return _case_not_found()
        return {"learning_case": _case_payload(case)}
    if request.tool_name == "get_standard_analysis":
        if not is_student_learning_case(case):
            return _case_not_found()
        payload = _case_payload(case)
        return {
            "standard_analysis": case.standard_analysis,
            "source_id": payload.get("source_id", ""),
            "source_record_id": payload.get("source_record_id", ""),
            "evidence_chain": case.evidence_chain,
        }
    if request.tool_name == "generate_mock_knowledge_cards":
        if not is_student_learning_case(case):
            return _case_not_found()
        return {"knowledge_cards": _knowledge_cards(case)}
    if request.tool_name == "generate_mock_quiz":
        if not is_student_learning_case(case):
            return _case_not_found()
        return {"quiz": _quiz_payloads(db, case)}
    if request.tool_name == "submit_review_answer":
        if not is_student_learning_case(case):
            return _case_not_found()
        return {"submitted": True, "feedback": "已记录作答并建议复盘标准解析。"}
    return {"learning_case": _case_payload(case)}


def _case_not_found() -> dict:
    return {"error_code": "LEARNING_CASE_NOT_FOUND", "message": "Learning case is not available for student learning"}


def _strip_sensitive_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_sensitive_fields(item)
            for key, item in value.items()
            if str(key).lower() not in SENSITIVE_CASE_KEYS
        }
    if isinstance(value, list):
        return [_strip_sensitive_fields(item) for item in value]
    return value


def _knowledge_cards(case: LearningCase) -> list[dict[str, Any]]:
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    points = payload.get("knowledge_points") or ["审方训练"]
    source_id = payload.get("source_id", "")
    return [
        {
            "title": str(point),
            "content": "该知识点来自脱敏教学案例，仅用于训练复盘，不作为临床建议。",
            "source_id": source_id,
            "visibility": "student_learning",
        }
        for point in points
    ]


def _quiz_payloads(db: Session, case: LearningCase) -> list[dict[str, Any]]:
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    external_case_id = payload.get("case_id", "")
    quizzes = db.scalars(select(Quiz).where(Quiz.case_id == external_case_id).order_by(Quiz.quiz_id)).all()
    if not quizzes:
        return contextualize_student_quiz(payload, [{"question": "该案例的主要复盘点是什么？", "answer": "需要结合标准解析复盘。", "source_id": payload.get("source_id", "")}])
    items = [
        {
            "quiz_id": quiz.quiz_id,
            "question_type": quiz.question_type,
            "question": quiz.question,
            "options": quiz.options_json,
            "answer": quiz.answer,
            "explanation": quiz.explanation,
            "knowledge_points": quiz.knowledge_points_json,
            "source_id": quiz.source_id,
            "visibility": "student_learning",
        }
        for quiz in quizzes
    ]
    return contextualize_student_quiz(payload, items)
