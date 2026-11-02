from app.tests.conftest import auth_header, doctor_with_patient, login, register


def test_login_requires_matching_entry_role(client):
    register(client, "doctor_role", "doctor")
    register(client, "patient_role", "patient")
    register(client, "student_role", "student")

    assert login(client, "doctor_role", "doctor")

    doctor_as_patient = client.post(
        "/api/auth/login",
        json={"username": "doctor_role", "password": "password123", "entry_role": "patient"},
    )
    assert doctor_as_patient.status_code == 403

    patient_as_doctor = client.post(
        "/api/auth/login",
        json={"username": "patient_role", "password": "password123", "entry_role": "doctor"},
    )
    assert patient_as_doctor.status_code == 403

    student_as_patient = client.post(
        "/api/auth/login",
        json={"username": "student_role", "password": "password123", "entry_role": "patient"},
    )
    assert student_as_patient.status_code == 403


def test_doctor_created_patient_must_change_temporary_password(client):
    _doctor_token, patient = doctor_with_patient(client)
    login_response = client.post(
        "/api/auth/login",
        json={"username": patient["username"], "password": "password123", "entry_role": "patient"},
    )
    assert login_response.status_code == 200
    login_data = login_response.json()["data"]
    assert login_data["user"]["must_change_password"] is True
    assert client.get(
        "/api/patient/profile",
        headers=auth_header(login_data["access_token"]),
    ).status_code == 403

    wrong = client.post(
        "/api/auth/change-password",
        headers=auth_header(login_data["access_token"]),
        json={"current_password": "not-the-password", "new_password": "new-password-123", "confirm_password": "new-password-123"},
    )
    assert wrong.status_code == 400

    changed = client.post(
        "/api/auth/change-password",
        headers=auth_header(login_data["access_token"]),
        json={"current_password": "password123", "new_password": "new-password-123", "confirm_password": "new-password-123"},
    )
    assert changed.status_code == 200
    assert changed.json()["data"]["must_change_password"] is False
    assert client.post(
        "/api/auth/login",
        json={"username": patient["username"], "password": "password123", "entry_role": "patient"},
    ).status_code == 401
    assert login(client, patient["username"], "patient", password="new-password-123")


def test_self_registered_account_does_not_require_forced_password_change(client):
    registered = register(client, "self_registered_patient", "patient")
    assert registered["must_change_password"] is False
    response = client.post(
        "/api/auth/login",
        json={"username": "self_registered_patient", "password": "password123", "entry_role": "patient"},
    )
    assert response.json()["data"]["user"]["must_change_password"] is False
