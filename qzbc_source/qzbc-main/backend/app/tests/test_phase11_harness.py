from sqlalchemy import select

from app.db.models import HarnessResult
from app.harness.case_schema import PHASE11
from app.harness.phase11_runner import Phase11HarnessRunner
from app.tests.conftest import auth_header, login, register


def test_phase11_harness_runner_persists_demo_readiness_results(db_session):
    run = Phase11HarnessRunner(db_session, suite="full").run()

    assert run.phase == PHASE11
    assert run.passed == run.total_cases
    assert run.pass_rate == 1.0
    assert run.metrics_json["demo_readiness_rate"] == 1.0
    assert run.markdown_summary.startswith("# Phase 11 Demo Readiness Harness")

    rows = db_session.scalars(select(HarnessResult).where(HarnessResult.run_id == run.id)).all()
    assert {row.category for row in rows} >= {
        "demo_status",
        "demo_reset",
        "demo_smoke",
        "demo_accounts",
        "demo_frontend_readiness",
    }


def test_phase11_harness_api_can_run_explicit_phase(client):
    register(client, "phase11_harness_api_doctor", "doctor")
    token = login(client, "phase11_harness_api_doctor", "doctor")

    response = client.post(
        "/api/harness/run",
        headers=auth_header(token),
        json={"phase": PHASE11, "suite": "quick", "reset_demo_data": True},
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["phase"] == PHASE11
    assert data["suite"] == "quick"
    assert data["passed"] == data["total_cases"]
    assert data["metrics"]["demo_readiness_rate"] == 1.0
