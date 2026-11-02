from __future__ import annotations

import json
import re
from typing import Any


PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
PRECISE_DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b")
SENSITIVE_FIELD_NAMES = {"patient_name", "real_name", "name", "phone", "mobile", "id_card", "identity_number", "visit_date", "address", "hospital_internal_id"}


def validate_deidentification(payload: dict[str, Any]) -> list[dict[str, str]]:
    text = json.dumps(payload, ensure_ascii=False)
    issues: list[dict[str, str]] = []
    if payload.get("deidentified") is not True:
        issues.append(_issue("LEARNING_CASE_NOT_DEIDENTIFIED", "Learning case must be deidentified."))
    for field_name in sorted(_sensitive_fields(payload)):
        issues.append(_issue("LEARNING_CASE_SENSITIVE_FIELD", f"Learning case contains sensitive field: {field_name}"))
    if PHONE_PATTERN.search(text):
        issues.append(_issue("PHONE_NUMBER_DETECTED", "Phone-like personal data found."))
    if ID_CARD_PATTERN.search(text):
        issues.append(_issue("ID_CARD_DETECTED", "ID-card-like personal data found."))
    if PRECISE_DATE_PATTERN.search(text):
        issues.append(_issue("PRECISE_DATE_DETECTED", "Precise date found in learning case."))
    return issues


def deidentification_pass_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return 1.0
    passed = sum(1 for item in items if not validate_deidentification(item))
    return passed / len(items)


def _sensitive_fields(value: Any, prefix: str = "") -> set[str]:
    fields: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in SENSITIVE_FIELD_NAMES:
                fields.add(path)
            fields.update(_sensitive_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.update(_sensitive_fields(child, f"{prefix}[{index}]"))
    return fields


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message, "severity": "critical"}
