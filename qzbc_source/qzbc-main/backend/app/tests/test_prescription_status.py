from sqlalchemy import select

from app.db.models import PrescriptionStatusLog
from app.tests.conftest import auth_header, create_reviewed_prescription, doctor_with_patient, login


def test_prescription_state_flow_and_logs(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    patient_token = login(client, "patient_a", "patient")

    create_response = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "草稿处方",
            "items": [{"herb_name": "茯苓", "dose": "9", "unit": "g"}],
        },
    )
    assert create_response.status_code == 200
    prescription_id = create_response.json()["data"]["id"]
    assert create_response.json()["data"]["status"] == "draft"

    invisible = client.get("/api/patient/prescriptions", headers=auth_header(patient_token))
    assert invisible.status_code == 200
    assert invisible.json()["data"] == []

    invalid_publish = client.post(
        f"/api/doctor/prescriptions/{prescription_id}/publish",
        headers=auth_header(doctor_token),
    )
    assert invalid_publish.status_code == 409

    review = client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(doctor_token))
    assert review.status_code == 200
    assert review.json()["data"]["risk_cards"]

    confirm = client.post(f"/api/doctor/prescriptions/{prescription_id}/confirm", headers=auth_header(doctor_token))
    assert confirm.status_code == 200
    assert confirm.json()["data"]["status"] == "doctor_confirmed"

    still_invisible = client.get("/api/patient/prescriptions", headers=auth_header(patient_token))
    assert still_invisible.json()["data"] == []

    publish = client.post(f"/api/doctor/prescriptions/{prescription_id}/publish", headers=auth_header(doctor_token))
    assert publish.status_code == 200
    assert publish.json()["data"]["status"] == "published_to_patient"

    visible = client.get("/api/patient/prescriptions", headers=auth_header(patient_token))
    assert len(visible.json()["data"]) == 1

    logs = db_session.scalars(
        select(PrescriptionStatusLog).where(PrescriptionStatusLog.prescription_id == prescription_id)
    ).all()
    assert [log.to_status for log in logs] == ["draft", "reviewing", "doctor_confirmed", "published_to_patient"]


def test_reviewed_prescription_helper(client):
    doctor_token, patient = doctor_with_patient(client)
    prescription_id = create_reviewed_prescription(client, doctor_token, patient["id"])
    assert prescription_id > 0


def test_doctor_can_add_prescription_usage_without_changing_item_doses(client):
    doctor_token, patient = doctor_with_patient(client)
    created = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "补充处方用法测试",
            "items": [{"herb_name": "茯苓", "dose": "9", "unit": "g"}],
        },
    )
    prescription_id = created.json()["data"]["id"]
    updated = client.request(
        "PATCH",
        f"/api/doctor/prescriptions/{prescription_id}",
        headers=auth_header(doctor_token),
        json={"usage_method": "每日一剂，水煎取汁，早晚分服。"},
    )
    assert updated.status_code == 200
    data = updated.json()["data"]
    assert data["usage_method"] == "每日一剂，水煎取汁，早晚分服。"
    assert data["items"][0]["dose"] == "9"
    assert data["items"][0]["unit"] == "g"

    review = client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(doctor_token))
    assert review.status_code == 200
    assert all(card["rule_id"] != "RULE_MISSING_USAGE" for card in review.json()["data"]["risk_cards"])
