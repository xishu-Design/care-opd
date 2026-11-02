from app.harness.case_schema import PHASE12
from app.tests.conftest import auth_header, login, register


def test_evaluation_api_is_doctor_only_and_returns_report(client) -> None:
    register(client, "phase12_doctor", "doctor")
    register(client, "phase12_patient", "patient")
    doctor_token = login(client, "phase12_doctor", "doctor")
    patient_token = login(client, "phase12_patient", "patient")

    forbidden = client.post(
        "/api/evaluation/run",
        headers=auth_header(patient_token),
        json={"suite": "quick", "reset_demo_data": True},
    )
    assert forbidden.status_code == 403

    response = client.post(
        "/api/evaluation/run",
        headers=auth_header(doctor_token),
        json={"suite": "quick", "reset_demo_data": True},
    )
    assert response.status_code == 200, response.text
    run = response.json()["data"]
    assert run["total_cases"] == 27
    assert run["status"] == "completed"
    assert run["ppt_ready_tables"]
    assert run["metrics"]["safety"]["mcp_unauthorized_block_rate"] == 1.0

    list_response = client.get("/api/evaluation/runs", headers=auth_header(doctor_token))
    assert list_response.status_code == 200
    assert list_response.json()["data"][0]["run_id"] == run["run_id"]

    detail = client.get(f"/api/evaluation/runs/{run['run_id']}", headers=auth_header(doctor_token))
    assert detail.status_code == 200
    assert len(detail.json()["data"]["results"]) == 27

    report = client.get(f"/api/evaluation/runs/{run['run_id']}/report", headers=auth_header(doctor_token))
    assert report.status_code == 200
    assert report.json()["data"]["markdown_report"].startswith("# GuardTCM Evaluation Report")


def test_ablation_api_runs_and_latest_returns_same_report(client) -> None:
    register(client, "phase12_ablation_doctor", "doctor")
    token = login(client, "phase12_ablation_doctor", "doctor")

    response = client.post(
        "/api/evaluation/ablation/run",
        headers=auth_header(token),
        json={"suite": "quick"},
    )
    assert response.status_code == 200, response.text
    report = response.json()["data"]
    assert report["experiment_count"] == 8
    assert len(report["experiments"]) == 8

    latest = client.get("/api/evaluation/ablation/latest", headers=auth_header(token))
    assert latest.status_code == 200
    assert latest.json()["data"]["ablation_id"] == report["ablation_id"]


def test_dataset_manifest_api(client) -> None:
    register(client, "phase12_manifest_doctor", "doctor")
    token = login(client, "phase12_manifest_doctor", "doctor")
    response = client.get("/api/evaluation/datasets", headers=auth_header(token))
    assert response.status_code == 200
    assert response.json()["data"]["total_cases"] == 185


def test_phase12_is_available_from_harness_entrypoint(client) -> None:
    register(client, "phase12_harness_doctor", "doctor")
    token = login(client, "phase12_harness_doctor", "doctor")
    response = client.post(
        "/api/harness/run",
        headers=auth_header(token),
        json={"phase": PHASE12, "suite": "quick", "reset_demo_data": True},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["phase"] == PHASE12
    assert data["total_cases"] == 27
    assert data["ppt_ready_tables"]
