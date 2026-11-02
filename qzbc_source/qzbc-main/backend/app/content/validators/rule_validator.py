from __future__ import annotations

from typing import Any


VALID_RISK_LEVELS = {"green", "yellow", "orange", "red", "low", "medium", "high"}
VALID_RULE_ROLES = {"doctor", "student", "patient"}
HIGH_RISK_LEVELS = {"red", "high"}
SYSTEM_POLICY_SOURCE_ID = "SRC_GUARDTCM_SYSTEM_POLICY"


def validate_review_rule(rule: dict[str, Any], source_ids: set[str]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    rule_id = str(rule.get("rule_id", ""))
    if not rule_id:
        issues.append(_issue("MISSING_RULE_ID", "Rule must include rule_id."))
    if not rule.get("rule_type"):
        issues.append(_issue("MISSING_RULE_TYPE", "Rule must include rule_type."))
    risk_level = str(rule.get("risk_level", ""))
    if risk_level not in VALID_RISK_LEVELS:
        issues.append(_issue("INVALID_RULE_RISK_LEVEL", f"Invalid rule risk_level: {risk_level}"))
    if not isinstance(rule.get("condition"), dict) or not rule.get("condition"):
        issues.append(_issue("MISSING_RULE_CONDITION", "Rule condition must be a non-empty object."))
    if not rule.get("evidence_query"):
        issues.append(_issue("MISSING_RULE_EVIDENCE_QUERY", "Rule must include evidence_query."))
    roles = rule.get("applies_to_roles", [])
    if not isinstance(roles, list) or not roles:
        issues.append(_issue("INVALID_RULE_ROLES", "Rule applies_to_roles must be a non-empty list."))
    for role in roles if isinstance(roles, list) else []:
        if role not in VALID_RULE_ROLES:
            issues.append(_issue("INVALID_RULE_ROLE", f"Invalid rule role: {role}"))
    if "patient" in roles and rule.get("rule_category", "medical_review") == "medical_review":
        issues.append(_issue("PATIENT_ROLE_IN_MEDICAL_RULE", "Medical review rules cannot apply to patient role.", "high"))
    if risk_level in HIGH_RISK_LEVELS and rule.get("requires_doctor_confirmation") is not True:
        issues.append(_issue("HIGH_RISK_RULE_REQUIRES_DOCTOR_CONFIRMATION", "High/red risk rules must require doctor confirmation.", "high"))
    source_id = str(rule.get("source_id", ""))
    if not source_id:
        issues.append(_issue("MISSING_RULE_SOURCE_ID", "Rule must include source_id."))
    elif source_id not in source_ids and source_id != SYSTEM_POLICY_SOURCE_ID:
        issues.append(_issue("RULE_SOURCE_ID_NOT_FOUND", f"source_id {source_id} not found in sources.json."))
    return issues


def rule_valid_rate(rules: list[dict[str, Any]], source_ids: set[str]) -> float:
    if not rules:
        return 1.0
    valid = sum(1 for rule in rules if not validate_review_rule(rule, source_ids))
    return valid / len(rules)


def _issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}
