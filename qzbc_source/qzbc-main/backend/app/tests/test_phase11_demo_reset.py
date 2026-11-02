from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import DemoResetLog, DoctorPatientBinding, Prescription, User
from app.demo.service import DEMO_PUBLISHED_TITLE, DEMO_REVIEW_TITLE, DEMO_UNPUBLISHED_TITLE
from app.tests.conftest import auth_header, login, register


def test_demo_reset_creates_demo_data_and_is_idempotent(client, db_session):
    register(client, "reset_actor", "doctor")
    register(client, "real_user_kept", "patient")
    token = login(client, "reset_actor", "doctor")

    first = client.post("/api/demo/reset", headers=auth_header(token), json={"reset_scope": "demo_only", "reseed": True})
    second = client.post("/api/demo/reset", headers=auth_header(token), json={"reset_scope": "demo_only", "reseed": True})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["data"]["created"] == second.json()["data"]["created"]
    assert db_session.scalar(select(User).where(User.username == "real_user_kept")) is not None

    doctor = db_session.scalar(select(User).where(User.username == "doctor_demo"))
    patient = db_session.scalar(select(User).where(User.username == "patient_demo"))
    student = db_session.scalar(select(User).where(User.username == "student_demo"))
    assert doctor is not None
    assert patient is not None
    assert student is not None
    assert db_session.scalar(select(DoctorPatientBinding).where(DoctorPatientBinding.doctor_id == doctor.id, DoctorPatientBinding.patient_id == patient.id)) is not None

    titles = [DEMO_UNPUBLISHED_TITLE, DEMO_PUBLISHED_TITLE, DEMO_REVIEW_TITLE]
    for title in titles:
        rows = db_session.scalars(select(Prescription).where(Prescription.title == title)).all()
        assert len(rows) == 1

    logs = db_session.scalars(select(DemoResetLog).order_by(DemoResetLog.created_at)).all()
    assert len(logs) == 2
    assert logs[-1].reset_id == second.json()["data"]["reset_id"]
    assert logs[-1].actor_user_id == db_session.scalar(select(User).where(User.username == "reset_actor")).id
    assert logs[-1].created_summary_json["prescriptions"] == 3


def test_demo_reset_rejects_non_doctor_and_production(client):
    register(client, "reset_doctor_prod", "doctor")
    register(client, "reset_patient", "patient")
    register(client, "reset_student", "student")
    doctor_token = login(client, "reset_doctor_prod", "doctor")
    patient_token = login(client, "reset_patient", "patient")
    student_token = login(client, "reset_student", "student")

    assert client.post("/api/demo/reset", headers=auth_header(patient_token)).status_code == 403
    assert client.post("/api/demo/reset", headers=auth_header(student_token)).status_code == 403

    settings = get_settings()
    original = settings.app_env
    settings.app_env = "production"
    try:
        response = client.post("/api/demo/reset", headers=auth_header(doctor_token))
        assert response.status_code == 403
    finally:
        settings.app_env = original


def test_demo_submodule_entrypoints_importable():
    from app.demo.seed import reset_demo_data
    from app.demo.smoke import run_demo_smoke
    from app.demo.status import get_demo_status

    assert reset_demo_data
    assert run_demo_smoke
    assert get_demo_status
