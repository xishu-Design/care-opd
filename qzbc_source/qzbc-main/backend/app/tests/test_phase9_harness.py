from sqlalchemy import select

from app.db.models import HarnessResult
from app.harness.case_schema import PHASE9
from app.harness.phase9_runner import Phase9HarnessRunner
from app.tests.conftest import auth_header, login, register


def test_phase9_harness_runner_persists_open_data_pipeline_results(db_session):
    run = Phase9HarnessRunner(db_session, suite="full").run()

    assert run.phase == PHASE9
    assert run.passed == run.total_cases
    assert run.pass_rate == 1.0
    assert run.metrics_json["approved_source_count"] == 4
    assert run.metrics_json["processed_item_traceability_rate"] == 1.0
    assert run.metrics_json["content_source_coverage_rate"] == 1.0
    assert run.metrics_json["open_data_search_coverage"] == 1.0
    assert run.metrics_json["rule_source_binding_pass_rate"] == 1.0
    assert run.markdown_summary.startswith("# GuardTCM Phase 9 Harness Report")

    rows = db_session.scalars(select(HarnessResult).where(HarnessResult.run_id == run.id)).all()
    assert len(rows) == run.total_cases
    assert {row.category for row in rows} >= {
        "source_registry_validation",
        "license_validation",
        "data_processing",
        "content_build",
        "content_import",
        "mcp_search_open_data",
        "rule_source_binding",
        "patient_safety_filter",
        "learning_case_deidentification",
        "open_data_reproducibility",
    }


def test_phase9_harness_api_can_run_explicit_phase(client):
    register(client, "phase9_harness_doctor_api", "doctor")
    token = login(client, "phase9_harness_doctor_api", "doctor")

    response = client.post(
        "/api/harness/run",
        headers=auth_header(token),
        json={"phase": PHASE9, "suite": "quick", "reset_demo_data": True},
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["phase"] == PHASE9
    assert data["suite"] == "quick"
    assert data["passed"] == data["total_cases"]
    assert data["metrics"]["open_data_search_coverage"] == 1.0
    assert data["markdown_summary"].startswith("# GuardTCM Phase 9 Harness Report")
