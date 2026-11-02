from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.content.validators.case_validator import validate_learning_case
from app.content.validators.deid_validator import ID_CARD_PATTERN, PHONE_PATTERN
from app.content.validators.quiz_validator import validate_quiz
from app.content.validators.rule_validator import SYSTEM_POLICY_SOURCE_ID, validate_review_rule
from app.content.validators.safety_validator import validate_patient_safety
from app.data_pipeline.validators.source_registry_validator import DATA_PIPELINE_DIR


DEFAULT_CONTENT_DATA_DIR = DATA_PIPELINE_DIR.parents[0] / "content" / "data"
VALID_VISIBILITY = {"doctor", "patient", "student", "student_learning"}
PLACEHOLDER_TERMS = ("TODO", "todo", "xxx", "待补充")
KNOWLEDGE_FILES = ("herbs.json", "formulas.json", "contraindications.json", "diet_precautions.json", "public_education.json")
OPTIONAL_CONTENT_FILES = ("contraindications.json", "diet_precautions.json", "public_education.json", "review_rules.json")


@dataclass
class ContentValidationIssue:
    code: str
    message: str
    file_name: str
    item_id: str = ""
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "file_name": self.file_name,
            "item_id": self.item_id,
            "severity": self.severity,
        }


@dataclass
class ContentValidationReport:
    valid: bool
    errors: list[ContentValidationIssue] = field(default_factory=list)
    warnings: list[ContentValidationIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "summary": self.summary,
        }


class ContentValidator:
    def __init__(self, content_data_dir: Path | str = DEFAULT_CONTENT_DATA_DIR) -> None:
        self.content_data_dir = Path(content_data_dir)

    def validate(self) -> ContentValidationReport:
        errors: list[ContentValidationIssue] = []
        warnings: list[ContentValidationIssue] = []
        sources = _items(self.content_data_dir / "sources.json")
        source_ids = {source.get("source_id") for source in sources}
        herbs = _items(self.content_data_dir / "herbs.json")
        formulas = _items(self.content_data_dir / "formulas.json")
        contraindications = _items(self.content_data_dir / "contraindications.json")
        diet_precautions = _items(self.content_data_dir / "diet_precautions.json")
        public_education = _items(self.content_data_dir / "public_education.json")
        review_rules = _items(self.content_data_dir / "review_rules.json")
        learning_cases = _items(self.content_data_dir / "learning_cases.json")
        quizzes = _items(self.content_data_dir / "quizzes.json")

        errors.extend(self._validate_source_items(sources))
        for file_name, items in {
            "herbs.json": herbs,
            "formulas.json": formulas,
            "contraindications.json": contraindications,
            "diet_precautions.json": diet_precautions,
            "public_education.json": public_education,
            "review_rules.json": review_rules,
            "learning_cases.json": learning_cases,
            "quizzes.json": quizzes,
        }.items():
            errors.extend(self._validate_unique_ids(file_name, items))
            for item in items:
                item_id = _item_id(item)
                errors.extend(self._validate_sources(file_name, item_id, item, source_ids))
                errors.extend(self._validate_visibility(file_name, item_id, item))
                errors.extend(self._validate_safety(file_name, item_id, item))
                errors.extend(self._validate_required_content(file_name, item_id, item))
                warnings.extend(self._validate_placeholders(file_name, item_id, item))

        for rule in review_rules:
            rule_id = str(rule.get("rule_id", ""))
            errors.extend(_issue_from_dict(issue, "review_rules.json", rule_id) for issue in validate_review_rule(rule, {str(source_id) for source_id in source_ids if source_id}))

        rule_ids = {rule.get("rule_id") for rule in review_rules}
        for item in learning_cases:
            item_id = str(item.get("case_id", ""))
            errors.extend(
                _issue_from_dict(issue, "learning_cases.json", item_id)
                for issue in validate_learning_case(item, {str(source_id) for source_id in source_ids if source_id}, {str(rule_id) for rule_id in rule_ids if rule_id})
            )

        case_ids = {case.get("case_id") for case in learning_cases}
        for quiz in quizzes:
            quiz_id = str(quiz.get("quiz_id", ""))
            errors.extend(
                _issue_from_dict(issue, "quizzes.json", quiz_id)
                for issue in validate_quiz(quiz, {str(source_id) for source_id in source_ids if source_id}, {str(case_id) for case_id in case_ids if case_id})
            )

        summary = {
            "sources": len(sources),
            "herbs": len(herbs),
            "formulas": len(formulas),
            "contraindications": len(contraindications),
            "diet_precautions": len(diet_precautions),
            "public_education": len(public_education),
            "review_rules": len(review_rules),
            "learning_cases": len(learning_cases),
            "quizzes": len(quizzes),
        }
        return ContentValidationReport(valid=not errors, errors=errors, warnings=warnings, summary=summary)

    def _validate_source_items(self, sources: list[dict[str, Any]]) -> list[ContentValidationIssue]:
        issues: list[ContentValidationIssue] = []
        seen: set[str] = set()
        for source in sources:
            source_id = str(source.get("source_id", ""))
            if not source_id:
                issues.append(_issue("MISSING_SOURCE_ID", "Source must include source_id.", "sources.json", ""))
                continue
            if source_id in seen:
                issues.append(_issue("DUPLICATE_ID", "Duplicate source_id.", "sources.json", source_id))
            seen.add(source_id)
            if not source.get("source_title") and not source.get("name"):
                issues.append(_issue("MISSING_SOURCE_TITLE", "Source must include source_title or name.", "sources.json", source_id))
        return issues

    def _validate_unique_ids(self, file_name: str, items: list[dict[str, Any]]) -> list[ContentValidationIssue]:
        seen: set[str] = set()
        issues: list[ContentValidationIssue] = []
        for item in items:
            item_id = _item_id(item)
            if not item_id:
                issues.append(_issue("MISSING_ITEM_ID", "Content item must include an ID field.", file_name, ""))
                continue
            if item_id in seen:
                issues.append(_issue("DUPLICATE_ID", f"Duplicate content ID: {item_id}", file_name, item_id))
            seen.add(item_id)
        return issues

    def _validate_sources(self, file_name: str, item_id: str, item: dict[str, Any], source_ids: set[Any]) -> list[ContentValidationIssue]:
        issues: list[ContentValidationIssue] = []
        if "source_ids" in item:
            ids = item.get("source_ids", [])
            if not ids:
                issues.append(_issue("MISSING_SOURCE_IDS", "Item must include source_ids.", file_name, item_id))
            for source_id in ids:
                if source_id not in source_ids:
                    issues.append(_issue("SOURCE_ID_NOT_FOUND", f"source_id {source_id} not found in sources.json.", file_name, item_id))
        elif "source_id" in item:
            source_id = item.get("source_id")
            if not source_id:
                issues.append(_issue("MISSING_SOURCE_ID", "Item must include source_id.", file_name, item_id))
            elif source_id not in source_ids and source_id != SYSTEM_POLICY_SOURCE_ID:
                issues.append(_issue("SOURCE_ID_NOT_FOUND", f"source_id {source_id} not found in sources.json.", file_name, item_id))
        else:
            issues.append(_issue("MISSING_SOURCE", "Item must include source_id or source_ids.", file_name, item_id))
        return issues

    def _validate_visibility(self, file_name: str, item_id: str, item: dict[str, Any]) -> list[ContentValidationIssue]:
        visibility = item.get("visibility")
        if visibility is None:
            return []
        values = visibility if isinstance(visibility, list) else [visibility]
        return [_issue("INVALID_VISIBILITY", f"Invalid visibility: {value}", file_name, item_id) for value in values if value not in VALID_VISIBILITY]

    def _validate_safety(self, file_name: str, item_id: str, item: dict[str, Any]) -> list[ContentValidationIssue]:
        text = json.dumps(item, ensure_ascii=False)
        issues = [_issue_from_dict(issue, file_name, item_id) for issue in validate_patient_safety(item, file_name=file_name)]
        if PHONE_PATTERN.search(text):
            issues.append(_issue("PHONE_NUMBER_DETECTED", "Phone-like personal data found.", file_name, item_id, "critical"))
        if ID_CARD_PATTERN.search(text):
            issues.append(_issue("ID_CARD_DETECTED", "ID-card-like personal data found.", file_name, item_id, "critical"))
        return issues

    def _validate_required_content(self, file_name: str, item_id: str, item: dict[str, Any]) -> list[ContentValidationIssue]:
        issues: list[ContentValidationIssue] = []
        if file_name in KNOWLEDGE_FILES:
            if not _title_value(file_name, item):
                issues.append(_issue("MISSING_TITLE", "Knowledge item must include a non-empty title/name/topic/message.", file_name, item_id))
            keywords = item.get("keywords", [])
            if not isinstance(keywords, list) or not keywords:
                issues.append(_issue("MISSING_KEYWORDS", "Knowledge item must include non-empty keywords.", file_name, item_id))
        if file_name == "review_rules.json" and not item.get("name"):
            issues.append(_issue("MISSING_RULE_NAME", "Rule must include name.", file_name, item_id))
        return issues

    def _validate_placeholders(self, file_name: str, item_id: str, item: dict[str, Any]) -> list[ContentValidationIssue]:
        text = json.dumps(item, ensure_ascii=False)
        return [
            ContentValidationIssue("PLACEHOLDER_TEXT", f"Placeholder text found: {term}", file_name, item_id, "warning")
            for term in PLACEHOLDER_TERMS
            if term in text
        ]


def _items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain an items list")
    return items


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("entry_id") or item.get("rule_id") or item.get("case_id") or item.get("quiz_id") or item.get("source_id") or item.get("id") or "")


def _title_value(file_name: str, item: dict[str, Any]) -> str:
    if file_name in {"herbs.json", "formulas.json"}:
        return str(item.get("name", "")).strip()
    if file_name == "contraindications.json":
        return str(item.get("title", "")).strip()
    if file_name == "diet_precautions.json":
        return str(item.get("title") or item.get("related_entity") or item.get("message") or "").strip()
    if file_name == "public_education.json":
        return str(item.get("topic", "")).strip()
    return str(item.get("title") or item.get("name") or "").strip()


def _issue(code: str, message: str, file_name: str, item_id: str, severity: str = "error") -> ContentValidationIssue:
    return ContentValidationIssue(code=code, message=message, file_name=file_name, item_id=item_id, severity=severity)


def _issue_from_dict(issue: dict[str, str], file_name: str, item_id: str) -> ContentValidationIssue:
    return _issue(issue["code"], issue["message"], file_name, item_id, issue.get("severity", "error"))
