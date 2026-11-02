from app.tests.conftest import auth_header, login, register


def test_demo_smoke_runs_for_doctor_after_reset(client):
    register(client, "smoke_doctor", "doctor")
    token = login(client, "smoke_doctor", "doctor")
    reset = client.post("/api/demo/reset", headers=auth_header(token))
    assert reset.status_code == 200, reset.text

    response = client.post("/api/demo/smoke", headers=auth_header(token))
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["passed"] is True
    assert data["total_checks"] >= 12
    assert any(check["name"] == "patient_cannot_see_unpublished" and check["passed"] for check in data["checks"])
    assert any(check["name"] == "student_can_see_deidentified_case" and check["passed"] for check in data["checks"])


def test_demo_smoke_is_doctor_only(client):
    register(client, "smoke_patient", "patient")
    register(client, "smoke_student", "student")
    patient_token = login(client, "smoke_patient", "patient")
    student_token = login(client, "smoke_student", "student")

    assert client.post("/api/demo/smoke", headers=auth_header(patient_token)).status_code == 403
    assert client.post("/api/demo/smoke", headers=auth_header(student_token)).status_code == 403


def test_demo_smoke_fails_when_critical_demo_data_missing(client):
    register(client, "smoke_missing_doctor", "doctor")
    token = login(client, "smoke_missing_doctor", "doctor")

    response = client.post("/api/demo/smoke", headers=auth_header(token))
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["passed"] is False
    assert any(check["severity"] == "critical" and not check["passed"] for check in data["checks"])

