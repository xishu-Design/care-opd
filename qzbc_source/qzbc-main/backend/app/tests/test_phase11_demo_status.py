from app.tests.conftest import auth_header, login, register


def test_demo_status_is_doctor_only(client):
    register(client, "status_doctor", "doctor")
    register(client, "status_patient", "patient")
    register(client, "status_student", "student")
    doctor_token = login(client, "status_doctor", "doctor")
    patient_token = login(client, "status_patient", "patient")
    student_token = login(client, "status_student", "student")

    doctor_response = client.get("/api/demo/status", headers=auth_header(doctor_token))
    assert doctor_response.status_code == 200, doctor_response.text
    data = doctor_response.json()["data"]
    assert {"environment", "backend", "accounts", "readiness"}.issubset(data.keys())
    assert data["environment"]["phase"] == "phase_11_demo_readiness"

    assert client.get("/api/demo/status", headers=auth_header(patient_token)).status_code == 403
    assert client.get("/api/demo/status", headers=auth_header(student_token)).status_code == 403
    assert client.get("/api/demo/status").status_code == 401


def test_demo_status_reports_readiness_after_reset(client):
    register(client, "status_reset_doctor", "doctor")
    token = login(client, "status_reset_doctor", "doctor")

    reset = client.post("/api/demo/reset", headers=auth_header(token), json={"reset_scope": "demo_only", "reseed": True})
    assert reset.status_code == 200, reset.text

    response = client.get("/api/demo/status", headers=auth_header(token))
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["readiness"]["ready"] is True
    assert data["demo_reset"]["last_reset_id"] == reset.json()["data"]["reset_id"]
    assert data["accounts"] == {
        "doctor_demo_exists": True,
        "patient_demo_exists": True,
        "student_demo_exists": True,
    }
    assert data["bindings"]["doctor_patient_binding_exists"] is True
    assert "lightrag_disabled_using_local_knowledge" in data["readiness"]["warnings"]
