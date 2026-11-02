from __future__ import annotations

import json
from typing import Any


PATIENT_UNSAFE_TERMS = ("自行开方", "直接服用", "保证疗效", "替代医生", "无需就医", "加量", "停药", "换药", "治疗性食疗")
PATIENT_SAFE_VISIBILITIES = {"patient"}
PATIENT_SAFE_FILES = {"diet_precautions.json", "public_education.json"}
IGNORED_SAFETY_KEYS = {"forbidden_claims"}


def validate_patient_safety(payload: dict[str, Any], *, file_name: str = "") -> list[dict[str, str]]:
    text = _visible_text(payload)
    issues = []
    for term in PATIENT_UNSAFE_TERMS:
        if term in text:
            issues.append(
                {
                    "code": "PATIENT_UNSAFE_TERM",
                    "message": f"Unsafe patient-facing term found: {term}",
                    "severity": "high",
                }
            )
    if file_name == "diet_precautions.json" and payload.get("forbidden_as_treatment") is not True:
        issues.append(
            {
                "code": "DIET_TREATMENT_BOUNDARY_MISSING",
                "message": "Diet precaution items must set forbidden_as_treatment=true.",
                "severity": "high",
            }
        )
    return issues


def is_patient_visible(payload: dict[str, Any], *, file_name: str = "") -> bool:
    visibility = payload.get("visibility")
    values = visibility if isinstance(visibility, list) else [visibility] if visibility else []
    return file_name in PATIENT_SAFE_FILES or bool(PATIENT_SAFE_VISIBILITIES & {str(value) for value in values})


def patient_safe_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return 1.0
    safe = sum(1 for item in items if not validate_patient_safety(item))
    return safe / len(items)


def _visible_text(payload: dict[str, Any]) -> str:
    return json.dumps(_without_keys(payload, IGNORED_SAFETY_KEYS), ensure_ascii=False)


def _without_keys(value: Any, ignored_keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {key: _without_keys(item, ignored_keys) for key, item in value.items() if key not in ignored_keys}
    if isinstance(value, list):
        return [_without_keys(item, ignored_keys) for item in value]
    return value
