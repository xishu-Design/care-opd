from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import RuleDefinition, RuleSet, utcnow
from app.rules.rule_models import StructuredRule
from app.rules.rule_source_policy import is_rule_active, normalize_rule_for_source_policy

DATA_DIR = Path(__file__).resolve().parent / "data"
RULE_FILE = DATA_DIR / "prescription_rules.json"


def load_rules(db: Session, rule_file: Path | str | None = None) -> dict[str, Any]:
    payload = _load_rules_file(Path(rule_file) if rule_file else RULE_FILE)
    rule_set_payload = payload["rule_set"]
    rule_set = db.scalar(select(RuleSet).where(RuleSet.name == rule_set_payload["name"]))
    if not rule_set:
        rule_set = RuleSet(name=rule_set_payload["name"])
    rule_set.version = rule_set_payload.get("version", "0.3.0")
    rule_set.source = rule_set_payload.get("source", "local_rules")
    rule_set.loaded_at = utcnow()
    db.add(rule_set)
    db.flush()

    count = 0
    enabled_count = 0
    disabled_by_source_policy = 0
    policy_issues: list[dict[str, str]] = []
    for raw_rule in payload["rules"]:
        requested_enabled = bool(raw_rule.get("enabled", True))
        rule, issues = normalize_rule_for_source_policy(StructuredRule.model_validate(raw_rule))
        policy_issues.extend(issue.to_dict() for issue in issues)
        if requested_enabled and not rule.enabled:
            disabled_by_source_policy += 1
        row = db.scalar(select(RuleDefinition).where(RuleDefinition.rule_id == rule.rule_id))
        if not row:
            row = RuleDefinition(rule_id=rule.rule_id, rule_set_id=rule_set.id)
        row.rule_set_id = rule_set.id
        row.rule_type = rule.rule_type
        row.rule_category = rule.rule_category
        row.name = rule.name
        row.enabled = rule.enabled
        row.risk_level = rule.risk_level
        row.source_id = rule.source_id
        row.source_record_id = rule.source_record_id
        row.review_status = rule.review_status
        row.applies_to_roles_json = rule.applies_to_roles
        row.condition_json = rule.condition
        row.message = rule.message
        row.evidence_query = rule.evidence_query
        row.metadata_json = rule.metadata
        row.requires_doctor_confirmation = rule.requires_doctor_confirmation
        row.updated_at = utcnow()
        db.add(row)
        count += 1
        if rule.enabled:
            enabled_count += 1
    db.flush()
    return {
        "status": "loaded",
        "rule_set": rule_set.name,
        "version": rule_set.version,
        "rules_count": count,
        "enabled_count": enabled_count,
        "disabled_by_source_policy": disabled_by_source_policy,
        "policy_issues": policy_issues,
    }


def ensure_rules_loaded(db: Session) -> None:
    payload = _load_rules_file(RULE_FILE)
    rule_set_payload = payload["rule_set"]
    rule_set = db.scalar(select(RuleSet).where(RuleSet.name == rule_set_payload["name"]))
    rules_count = db.scalar(select(func.count(RuleDefinition.id))) or 0
    if rules_count == 0 or rule_set is None or rule_set.version != rule_set_payload.get("version", "0.3.0"):
        load_rules(db)


def get_rules(db: Session, include_disabled: bool = True) -> list[StructuredRule]:
    ensure_rules_loaded(db)
    query = select(RuleDefinition)
    if not include_disabled:
        query = query.where(RuleDefinition.enabled.is_(True))
    rows = db.scalars(query.order_by(RuleDefinition.rule_id)).all()
    rules = [
        StructuredRule(
            rule_id=row.rule_id,
            rule_type=row.rule_type,
            rule_category=row.rule_category,
            name=row.name,
            enabled=row.enabled,
            risk_level=row.risk_level,
            source_id=row.source_id,
            source_record_id=row.source_record_id,
            review_status=row.review_status,
            applies_to_roles=row.applies_to_roles_json,
            condition=row.condition_json,
            message=row.message,
            evidence_query=row.evidence_query,
            metadata=row.metadata_json,
            requires_doctor_confirmation=row.requires_doctor_confirmation,
        )
        for row in rows
    ]
    if not include_disabled:
        return [rule for rule in rules if is_rule_active(rule, role="doctor")]
    return rules


def _load_rules_file(rule_file: Path) -> dict[str, Any]:
    with rule_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or "rule_set" not in payload or "rules" not in payload:
        raise ValueError("prescription_rules.json must contain rule_set and rules")
    return payload
