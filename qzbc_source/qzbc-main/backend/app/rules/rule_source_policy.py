from __future__ import annotations

from dataclasses import dataclass

from app.rules.rule_models import StructuredRule

SYSTEM_POLICY_SOURCE_ID = "SRC_GUARDTCM_SYSTEM_POLICY"
SYSTEM_SAFETY_CATEGORY = "system_safety"
MEDICAL_REVIEW_CATEGORY = "medical_review"
HIGH_RISK_LEVELS = {"red", "high"}
VALID_RULE_ROLES = {"doctor", "patient", "student"}


@dataclass(frozen=True)
class RuleSourcePolicyIssue:
    code: str
    message: str
    rule_id: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "rule_id": self.rule_id}


def normalize_rule_for_source_policy(rule: StructuredRule) -> tuple[StructuredRule, list[RuleSourcePolicyIssue]]:
    issues: list[RuleSourcePolicyIssue] = []
    roles = _normalized_roles(rule.applies_to_roles)
    update: dict = {"applies_to_roles": roles}

    if rule.rule_category == SYSTEM_SAFETY_CATEGORY:
        source_id = rule.source_id or SYSTEM_POLICY_SOURCE_ID
        update["source_id"] = source_id
        if source_id != SYSTEM_POLICY_SOURCE_ID:
            update["enabled"] = False
            update["review_status"] = "manual_review_required"
            issues.append(
                RuleSourcePolicyIssue(
                    "SYSTEM_RULE_INVALID_SOURCE",
                    "System safety rule must use SRC_GUARDTCM_SYSTEM_POLICY.",
                    rule.rule_id,
                )
            )
        elif rule.enabled and not rule.review_status:
            update["review_status"] = "source_bound"
        return rule.model_copy(update=update), issues

    if rule.rule_category == MEDICAL_REVIEW_CATEGORY:
        if "patient" in roles:
            roles = [role for role in roles if role != "patient"]
            update["applies_to_roles"] = roles
            issues.append(
                RuleSourcePolicyIssue(
                    "PATIENT_ROLE_REMOVED_FROM_MEDICAL_RULE",
                    "Medical review rules cannot apply to patient role.",
                    rule.rule_id,
                )
            )
        if not roles:
            update["enabled"] = False
            update["review_status"] = "manual_review_required"
            issues.append(
                RuleSourcePolicyIssue(
                    "MEDICAL_RULE_HAS_NO_ALLOWED_ROLE",
                    "Medical review rule has no allowed role after role filtering.",
                    rule.rule_id,
                )
            )
        if rule.enabled and not rule.source_id:
            update["enabled"] = False
            update["review_status"] = "manual_review_required"
            issues.append(
                RuleSourcePolicyIssue(
                    "MEDICAL_RULE_MISSING_SOURCE",
                    "Medical review rule must include source_id before activation.",
                    rule.rule_id,
                )
            )
        if rule.enabled and rule.risk_level in HIGH_RISK_LEVELS and not rule.source_id:
            update["enabled"] = False
            update["review_status"] = "manual_review_required"
            issues.append(
                RuleSourcePolicyIssue(
                    "HIGH_RISK_MEDICAL_RULE_MISSING_SOURCE",
                    "High/red medical review rule must include source_id before activation.",
                    rule.rule_id,
                )
            )
        return rule.model_copy(update=update), issues

    if rule.enabled:
        update["enabled"] = False
        update["review_status"] = "manual_review_required"
        issues.append(
            RuleSourcePolicyIssue(
                "UNKNOWN_RULE_CATEGORY",
                "Unknown rule category cannot be activated.",
                rule.rule_id,
            )
        )
    return rule.model_copy(update=update), issues


def is_rule_active(rule: StructuredRule, role: str = "doctor") -> bool:
    normalized, _issues = normalize_rule_for_source_policy(rule)
    if not normalized.enabled:
        return False
    if role not in normalized.applies_to_roles:
        return False
    if normalized.rule_category == SYSTEM_SAFETY_CATEGORY:
        return normalized.source_id == SYSTEM_POLICY_SOURCE_ID
    if normalized.rule_category == MEDICAL_REVIEW_CATEGORY:
        return bool(normalized.source_id) and "patient" not in normalized.applies_to_roles
    return False


def _normalized_roles(roles: list[str]) -> list[str]:
    normalized = []
    for role in roles or ["doctor"]:
        role = str(role)
        if role in VALID_RULE_ROLES and role not in normalized:
            normalized.append(role)
    return normalized or ["doctor"]
