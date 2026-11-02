from __future__ import annotations

from typing import Any

from app.content.validators.deid_validator import validate_deidentification


def validate_learning_case(case: dict[str, Any], source_ids: set[str], rule_ids: set[str]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not case.get("case_id"):
        issues.append(_issue("MISSING_CASE_ID", "Learning case must include case_id."))
    if case.get("visibility") != "student_learning":
        issues.append(_issue("INVALID_LEARNING_CASE_VISIBILITY", "Learning case visibility must be student_learning."))
    source_id = str(case.get("source_id", ""))
    if not source_id:
        issues.append(_issue("MISSING_SOURCE_ID", "Learning case must include source_id."))
    elif source_id not in source_ids:
        issues.append(_issue("SOURCE_ID_NOT_FOUND", f"source_id {source_id} not found in sources.json."))
    expected_risks = case.get("expected_risks", [])
    if not isinstance(expected_risks, list):
        issues.append(_issue("INVALID_EXPECTED_RISKS", "expected_risks must be a list."))
    else:
        for rule_id in expected_risks:
            if rule_id not in rule_ids:
                issues.append(_issue("EXPECTED_RISK_RULE_NOT_FOUND", f"expected_risk {rule_id} not found in review_rules.json."))
    issues.extend(validate_deidentification(case))
    return issues


def learning_case_valid_rate(cases: list[dict[str, Any]], source_ids: set[str], rule_ids: set[str]) -> float:
    if not cases:
        return 1.0
    valid = sum(1 for case in cases if not validate_learning_case(case, source_ids, rule_ids))
    return valid / len(cases)


def _issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}
