from sqlalchemy import select

from app.db.models import MCPToolCall
from app.tests.conftest import auth_header, doctor_with_patient, login, register


def test_mcp_gateway_allowlist_and_denied_logs(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    del patient
    patient_token = login(client, "patient_a", "patient")
    register(client, "student_mcp", "student")
    student_token = login(client, "student_mcp", "student")

    doctor_call = client.post(
        "/api/mcp/mock-call",
        headers=auth_header(doctor_token),
        json={"tool_name": "run_prescription_review", "arguments": {"prescription_id": 1}},
    )
    assert doctor_call.status_code == 200
    assert doctor_call.json()["data"]["allowed"] is True

    patient_denied = client.post(
        "/api/mcp/mock-call",
        headers=auth_header(patient_token),
        json={"tool_name": "run_prescription_review", "arguments": {"prescription_id": 1}},
    )
    assert patient_denied.status_code == 200
    assert patient_denied.json()["data"]["allowed"] is False

    student_denied = client.post(
        "/api/mcp/mock-call",
        headers=auth_header(student_token),
        json={"tool_name": "get_patient_profile", "arguments": {"patient_id": 1}},
    )
    assert student_denied.status_code == 200
    assert student_denied.json()["data"]["allowed"] is False

    calls = db_session.scalars(select(MCPToolCall)).all()
    assert len(calls) == 3
    assert sum(1 for call in calls if not call.allowed) == 2
