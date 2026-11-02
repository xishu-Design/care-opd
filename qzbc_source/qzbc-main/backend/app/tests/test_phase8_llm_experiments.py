from app.core.config import get_settings
from app.llm.experiments.runner import LLMExperimentRunner, load_experiment_cases
from app.tests.conftest import auth_header, login, register


def test_experiment_cases_cover_four_task_types():
    cases = load_experiment_cases("full")

    assert len(cases) == 12
    assert {case["task_type"] for case in cases} == {
        "doctor_review_text",
        "patient_education",
        "student_analysis",
        "quiz_generation",
    }


def test_quick_experiment_runs_with_template_provider(db_session):
    report = LLMExperimentRunner(db_session).run(provider="template", suite="quick")

    assert report["provider"] == "template"
    assert report["total_cases"] == 4
    assert report["success"] == 4
    assert report["parse_success_rate"] == 1.0
    assert report["schema_valid_rate"] == 1.0
    assert report["safety_pass_rate"] == 1.0
    assert set(report["categories"]) == {"doctor_review_text", "patient_education", "student_analysis", "quiz_generation"}


def test_ollama_unavailable_experiment_records_fallback(db_session):
    settings = get_settings()
    original_base = settings.ollama_base_url
    original_timeout = settings.ollama_generate_timeout_seconds
    settings.ollama_base_url = "http://127.0.0.1:1"
    settings.ollama_generate_timeout_seconds = 1
    try:
        report = LLMExperimentRunner(db_session).run(provider="ollama", model="qwen2.5:7b", suite="quick")
    finally:
        settings.ollama_base_url = original_base
        settings.ollama_generate_timeout_seconds = original_timeout

    assert report["provider"] == "ollama"
    assert report["total_cases"] == 4
    assert report["fallback_count"] == 4
    assert report["schema_valid_rate"] == 1.0
    assert all(item["fallback_used"] for item in report["items"])


def test_experiment_api_is_doctor_only_and_returns_latest(client):
    register(client, "exp_doctor", "doctor")
    doctor_token = login(client, "exp_doctor", "doctor")
    register(client, "exp_patient", "patient")
    patient_token = login(client, "exp_patient", "patient")

    denied = client.get("/api/llm/experiments/latest", headers=auth_header(patient_token))
    assert denied.status_code == 403

    run = client.post(
        "/api/llm/experiments/run",
        headers=auth_header(doctor_token),
        json={"provider": "template", "suite": "quick"},
    )
    assert run.status_code == 200, run.text
    report = run.json()["data"]
    latest = client.get("/api/llm/experiments/latest", headers=auth_header(doctor_token))
    by_id = client.get(f"/api/llm/experiments/{report['experiment_id']}", headers=auth_header(doctor_token))

    assert latest.status_code == 200
    assert by_id.status_code == 200
    assert latest.json()["data"]["experiment_id"] == report["experiment_id"]
    assert by_id.json()["data"]["experiment_id"] == report["experiment_id"]
