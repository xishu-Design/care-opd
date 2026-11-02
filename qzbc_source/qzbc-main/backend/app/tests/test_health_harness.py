from app.tests.conftest import auth_header, login, register


def test_health_and_version(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["phase"] == "phase_11_demo_readiness"

    version = client.get("/api/version")
    assert version.status_code == 200
    assert version.json()["data"]["version"] == "1.1.0"
    assert version.json()["data"]["phase"] == "phase_11_demo_readiness"


def test_harness_run_summary_endpoint(client):
    register(client, "doctor_harness", "doctor")
    token = login(client, "doctor_harness", "doctor")

    run = client.post("/api/harness/run", headers=auth_header(token))
    assert run.status_code == 200
    data = run.json()["data"]
    assert data["phase"] == "phase_11_demo_readiness"
    assert data["suite"] == "full"
    assert data["total_cases"] >= 5
    assert data["passed"] == data["total_cases"]
    assert data["pass_rate"] == 1.0
    assert "demo_status" in data["categories"]
    assert "demo_smoke" in data["categories"]
    assert data["metrics"]["demo_readiness_rate"] == 1.0
    assert data["markdown_summary"].startswith("# Phase 11 Demo Readiness Harness")

    report = client.get(f"/api/harness/runs/{data['run_id']}/report", headers=auth_header(token))
    assert report.status_code == 200
    report_data = report.json()["data"]
    assert report_data["phase"] == "phase_11_demo_readiness"
    assert report_data["competition_metrics"]["demo_readiness_rate"] == 1.0
    assert report_data["category_results"]["demo_status"]["passed"] == 1

    traces = client.get("/api/harness/traces", headers=auth_header(token))
    assert traces.status_code == 200
    assert "items" in traces.json()["data"]

    runs = client.get("/api/harness/runs", headers=auth_header(token))
    assert runs.status_code == 200
    assert runs.json()["data"][0]["run_id"] == data["run_id"]
