from __future__ import annotations

from typing import Any


TRUE_FALSE_ANSWERS = {"true", "false", "正确", "错误", True, False}


def validate_quiz(quiz: dict[str, Any], source_ids: set[str], case_ids: set[str]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not quiz.get("quiz_id"):
        issues.append(_issue("MISSING_QUIZ_ID", "Quiz must include quiz_id."))
    case_id = quiz.get("case_id")
    if not case_id:
        issues.append(_issue("MISSING_QUIZ_CASE_ID", "Quiz must include case_id."))
    elif case_id not in case_ids:
        issues.append(_issue("QUIZ_CASE_NOT_FOUND", "Quiz case_id must exist in learning_cases.json."))
    if not quiz.get("question"):
        issues.append(_issue("MISSING_QUIZ_QUESTION", "Quiz question must be non-empty."))
    question_type = quiz.get("question_type", "single_choice")
    options = quiz.get("options", [])
    answer = quiz.get("answer")
    if question_type == "true_false":
        if answer not in TRUE_FALSE_ANSWERS:
            issues.append(_issue("QUIZ_TRUE_FALSE_ANSWER_INVALID", "True/false quiz answer is invalid."))
    else:
        if not isinstance(options, list) or len(options) < 2:
            issues.append(_issue("QUIZ_OPTIONS_INVALID", "Quiz must include at least two options."))
        elif answer not in options:
            issues.append(_issue("QUIZ_ANSWER_NOT_IN_OPTIONS", "Quiz answer must be present in options."))
    if not quiz.get("explanation"):
        issues.append(_issue("MISSING_QUIZ_EXPLANATION", "Quiz explanation must be non-empty."))
    knowledge_points = quiz.get("knowledge_points", [])
    if not isinstance(knowledge_points, list) or not knowledge_points:
        issues.append(_issue("MISSING_QUIZ_KNOWLEDGE_POINTS", "Quiz knowledge_points must be non-empty."))
    source_id = str(quiz.get("source_id", ""))
    if not source_id:
        issues.append(_issue("MISSING_SOURCE_ID", "Quiz must include source_id."))
    elif source_id not in source_ids:
        issues.append(_issue("SOURCE_ID_NOT_FOUND", f"source_id {source_id} not found in sources.json."))
    return issues


def quiz_valid_rate(quizzes: list[dict[str, Any]], source_ids: set[str], case_ids: set[str]) -> float:
    if not quizzes:
        return 1.0
    valid = sum(1 for quiz in quizzes if not validate_quiz(quiz, source_ids, case_ids))
    return valid / len(quizzes)


def _issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}
