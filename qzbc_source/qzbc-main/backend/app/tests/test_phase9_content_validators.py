import json

from app.content.validators.content_validator import ContentValidator
from app.content.validators.rule_validator import validate_review_rule


def test_content_schema_files_are_json_objects():
    schema_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "content" / "schemas"
    schema_names = {
        "source.schema.json",
        "herb.schema.json",
        "formula.schema.json",
        "contraindication.schema.json",
        "diet_precaution.schema.json",
        "public_education.schema.json",
        "review_rule.schema.json",
        "learning_case.schema.json",
        "quiz.schema.json",
    }

    assert {path.name for path in schema_dir.glob("*.schema.json")} >= schema_names
    for schema_name in schema_names:
        payload = json.loads((schema_dir / schema_name).read_text(encoding="utf-8"))
        assert payload["type"] == "object"
        assert payload["required"]


def test_content_validator_covers_rules_cases_quizzes_and_patient_safety(tmp_path):
    content_dir = _content_dir(tmp_path)
    _write(
        content_dir / "review_rules.json",
        {
            "items": [
                {
                    "rule_id": "RULE_BAD",
                    "rule_type": "demo",
                    "name": "非法规则",
                    "enabled": True,
                    "risk_level": "red",
                    "applies_to_roles": ["doctor", "patient"],
                    "condition": {},
                    "message": "非法规则",
                    "evidence_query": "",
                    "requires_doctor_confirmation": False,
                    "source_id": "",
                }
            ]
        },
    )
    _write(
        content_dir / "learning_cases.json",
        {
            "items": [
                {
                    "case_id": "CASE_BAD",
                    "title": "未脱敏案例",
                    "visibility": "student_learning",
                    "deidentified": True,
                    "patient_name": "真实姓名",
                    "prescription": {"items": []},
                    "expected_risks": ["RULE_MISSING"],
                    "source_id": "SRC_CONTENT_TEST",
                }
            ]
        },
    )
    _write(
        content_dir / "quizzes.json",
        {
            "items": [
                {
                    "quiz_id": "QUIZ_BAD",
                    "case_id": "CASE_BAD",
                    "question_type": "single_choice",
                    "question": "问题",
                    "options": ["A", "B"],
                    "answer": "C",
                    "explanation": "",
                    "knowledge_points": [],
                    "source_id": "SRC_CONTENT_TEST",
                }
            ]
        },
    )
    _write(
        content_dir / "public_education.json",
        {
            "items": [
                {
                    "entry_id": "EDU_BAD",
                    "topic": "危险科普",
                    "safe_answer": "保证疗效",
                    "source_id": "SRC_CONTENT_TEST",
                    "keywords": ["科普"],
                    "visibility": ["patient"],
                }
            ]
        },
    )

    report = ContentValidator(content_dir).validate()

    codes = {issue.code for issue in report.errors}
    assert not report.valid
    assert {
        "MISSING_RULE_EVIDENCE_QUERY",
        "PATIENT_ROLE_IN_MEDICAL_RULE",
        "HIGH_RISK_RULE_REQUIRES_DOCTOR_CONFIRMATION",
        "EXPECTED_RISK_RULE_NOT_FOUND",
        "LEARNING_CASE_SENSITIVE_FIELD",
        "QUIZ_ANSWER_NOT_IN_OPTIONS",
        "MISSING_QUIZ_EXPLANATION",
        "PATIENT_UNSAFE_TERM",
    } <= codes


def test_rule_validator_accepts_system_policy_source_id():
    issues = validate_review_rule(
        {
            "rule_id": "RULE_SYSTEM",
            "rule_type": "system_guard",
            "rule_category": "system_safety",
            "name": "系统规则",
            "enabled": True,
            "risk_level": "high",
            "applies_to_roles": ["patient"],
            "condition": {"request_contains": "开方"},
            "message": "系统安全规则",
            "evidence_query": "system policy",
            "requires_doctor_confirmation": True,
            "source_id": "SRC_GUARDTCM_SYSTEM_POLICY",
        },
        set(),
    )

    assert not issues


def _content_dir(tmp_path):
    content_dir = tmp_path / "content" / "data"
    content_dir.mkdir(parents=True)
    _write(
        content_dir / "sources.json",
        {
            "items": [
                {
                    "source_id": "SRC_CONTENT_TEST",
                    "source_title": "Content Test",
                    "source_type": "test",
                    "source_version": "0.9.0",
                    "source_year": "2026",
                    "evidence_level": "test",
                    "license_note": "test only",
                }
            ]
        },
    )
    for file_name in (
        "herbs.json",
        "formulas.json",
        "contraindications.json",
        "diet_precautions.json",
        "public_education.json",
        "review_rules.json",
        "learning_cases.json",
        "quizzes.json",
    ):
        _write(content_dir / file_name, {"items": []})
    return content_dir


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
