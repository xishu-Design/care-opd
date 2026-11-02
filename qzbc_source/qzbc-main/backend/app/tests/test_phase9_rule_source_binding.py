import json

from sqlalchemy import select

from app.db.models import RuleDefinition
from app.rules.rule_loader import get_rules, load_rules
from app.rules.rule_source_policy import SYSTEM_POLICY_SOURCE_ID, is_rule_active


def test_rule_source_binding_disables_no_source_medical_rules(db_session, tmp_path):
    rule_file = _rule_file(tmp_path)

    report = load_rules(db_session, rule_file)
    db_session.flush()

    assert report["status"] == "loaded"
    assert report["disabled_by_source_policy"] == 1
    assert {issue["code"] for issue in report["policy_issues"]} >= {
        "MEDICAL_RULE_MISSING_SOURCE",
        "HIGH_RISK_MEDICAL_RULE_MISSING_SOURCE",
        "PATIENT_ROLE_REMOVED_FROM_MEDICAL_RULE",
    }

    no_source = db_session.scalar(select(RuleDefinition).where(RuleDefinition.rule_id == "RULE_NO_SOURCE_RED"))
    assert no_source is not None
    assert no_source.enabled is False
    assert no_source.review_status == "manual_review_required"

    active_rule_ids = {rule.rule_id for rule in get_rules(db_session, include_disabled=False)}
    assert "RULE_NO_SOURCE_RED" not in active_rule_ids
    assert "RULE_SOURCE_BOUND_RED" in active_rule_ids


def test_system_safety_rules_can_be_source_bound_to_project_policy(db_session, tmp_path):
    load_rules(db_session, _rule_file(tmp_path))

    system_rule = next(rule for rule in get_rules(db_session) if rule.rule_id == "RULE_PATIENT_CANNOT_PRESCRIBE")

    assert system_rule.enabled is True
    assert system_rule.rule_category == "system_safety"
    assert system_rule.source_id == SYSTEM_POLICY_SOURCE_ID
    assert is_rule_active(system_rule, role="patient") is True


def test_patient_role_removed_from_medical_review_rules(db_session, tmp_path):
    load_rules(db_session, _rule_file(tmp_path))

    sourced_rule = next(rule for rule in get_rules(db_session) if rule.rule_id == "RULE_SOURCE_BOUND_RED")

    assert sourced_rule.enabled is True
    assert sourced_rule.rule_category == "medical_review"
    assert sourced_rule.source_id == "SRC_TEST_MEDICAL_SOURCE"
    assert "patient" not in sourced_rule.applies_to_roles
    assert is_rule_active(sourced_rule, role="doctor") is True
    assert is_rule_active(sourced_rule, role="patient") is False


def _rule_file(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "rule_set": {
                    "name": "phase9_rule_source_binding",
                    "version": "0.9.0-test",
                    "source": "test_fixture",
                },
                "rules": [
                    {
                        "rule_id": "RULE_PATIENT_CANNOT_PRESCRIBE",
                        "rule_type": "patient_guardrail",
                        "rule_category": "system_safety",
                        "name": "患者端不能开方",
                        "enabled": True,
                        "risk_level": "high",
                        "source_id": SYSTEM_POLICY_SOURCE_ID,
                        "source_record_id": "policy:patient_cannot_prescribe",
                        "review_status": "source_bound",
                        "applies_to_roles": ["patient"],
                        "condition": {"patient_request_contains": "开方"},
                        "message": "患者端不能请求系统开方。",
                        "evidence_query": "system policy patient cannot prescribe",
                        "requires_doctor_confirmation": False,
                    },
                    {
                        "rule_id": "RULE_NO_SOURCE_RED",
                        "rule_type": "medical_demo",
                        "rule_category": "medical_review",
                        "name": "无来源高风险医学规则",
                        "enabled": True,
                        "risk_level": "red",
                        "source_id": "",
                        "review_status": "source_bound",
                        "applies_to_roles": ["doctor"],
                        "condition": {"field_missing": "usage"},
                        "message": "无来源医学规则不应启用。",
                        "evidence_query": "no source medical rule",
                        "requires_doctor_confirmation": True,
                    },
                    {
                        "rule_id": "RULE_SOURCE_BOUND_RED",
                        "rule_type": "medical_demo",
                        "rule_category": "medical_review",
                        "name": "有来源高风险医学规则",
                        "enabled": True,
                        "risk_level": "red",
                        "source_id": "SRC_TEST_MEDICAL_SOURCE",
                        "source_record_id": "rule_source:1",
                        "review_status": "source_bound",
                        "applies_to_roles": ["doctor", "patient"],
                        "condition": {"field_missing": "usage"},
                        "message": "有来源医学规则可进入医生审方规则集。",
                        "evidence_query": "source bound medical rule",
                        "requires_doctor_confirmation": True,
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
