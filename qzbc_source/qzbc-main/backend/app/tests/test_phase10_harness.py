from sqlalchemy import select

from app.db.models import HarnessResult
from app.harness.case_schema import PHASE10
from app.harness.phase10_runner import Phase10HarnessRunner
from app.tests.conftest import auth_header, login, register


def test_phase10_harness_runner_persists_lightrag_results(db_session):
    run = Phase10HarnessRunner(db_session, suite="full").run()

    assert run.phase == PHASE10
    assert run.passed == run.total_cases
    assert run.pass_rate == 1.0
    assert run.metrics_json["lightrag_doc_build_rate"] == 1.0
    assert run.metrics_json["lightrag_retrieval_rate"] == 1.0
    assert run.metrics_json["lightrag_fallback_rate"] == 1.0
    assert run.markdown_summary.startswith("# GuardTCM Phase 10 Harness Report")

    rows = db_session.scalars(select(HarnessResult).where(HarnessResult.run_id == run.id)).all()
    assert {row.category for row in rows} >= {
        "lightrag_doc_build",
        "lightrag_index",
        "lightrag_retrieval",
        "lightrag_visibility_filter",
        "lightrag_fallback",
        "evidence_packet_graph_retrieval",
        "mcp_lightrag_adapter",
        "doctor_lightrag_review",
        "patient_lightrag_safety",
        "student_lightrag_learning",
    }


def test_phase10_harness_api_can_run_explicit_phase(client):
    register(client, "phase10_harness_api_doctor", "doctor")
    token = login(client, "phase10_harness_api_doctor", "doctor")

    response = client.post(
        "/api/harness/run",
        headers=auth_header(token),
        json={"phase": PHASE10, "suite": "quick", "reset_demo_data": True},
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["phase"] == PHASE10
    assert data["suite"] == "quick"
    assert data["passed"] == data["total_cases"]
    assert data["metrics"]["lightrag_doc_build_rate"] == 1.0
