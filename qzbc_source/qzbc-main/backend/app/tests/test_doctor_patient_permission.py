from app.tests.conftest import auth_header, doctor_with_patient, login, register


def test_doctor_can_access_only_bound_patients(client):
    doctor_token, patient = doctor_with_patient(client)
    response = client.get(f"/api/doctor/patients/{patient['id']}", headers=auth_header(doctor_token))
    assert response.status_code == 200
    assert response.json()["data"]["username"] == "patient_a"

    register(client, "doctor_b", "doctor", "Doctor B")
    doctor_b_token = login(client, "doctor_b", "doctor")
    forbidden = client.get(f"/api/doctor/patients/{patient['id']}", headers=auth_header(doctor_b_token))
    assert forbidden.status_code == 403


def test_patient_self_profile_and_student_forbidden(client):
    doctor_token, patient = doctor_with_patient(client)
    del doctor_token
    patient_token = login(client, "patient_a", "patient")
    profile = client.get("/api/patient/profile", headers=auth_header(patient_token))
    assert profile.status_code == 200
    assert profile.json()["data"]["id"] == patient["id"]

    register(client, "student_a", "student", "Student A")
    student_token = login(client, "student_a", "student")
    student_profile_attempt = client.get("/api/patient/profile", headers=auth_header(student_token))
    assert student_profile_attempt.status_code == 403
