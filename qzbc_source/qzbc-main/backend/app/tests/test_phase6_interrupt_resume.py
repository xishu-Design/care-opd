from sqlalchemy import select

from app.agents.services.checkpoint_service import get_langgraph_checkpointer
from app.core.config import get_settings
from app.db.models import AgentTraceEvent, EvidencePacketLog, MCPToolCall, PatientHealthRecord, MedicalHistory, PendingHumanAction, Prescription
from app.tests.conftest import auth_header, doctor_with_patient, login, register


def _create_prescription(client, doctor_token: str, patient_id: int) -> int:
    response = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient_id,
            "title": "Phase6 审方测试",
            "symptom_context": "夜间咳嗽、咽干",
            "usage_method": "每日一剂，水煎服，遵医嘱。",
            "doctor_notes": "Phase6 internal note",
            "items": [{"herb_name": "黄芪", "dose": "10", "unit": "g"}],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]["id"]


def _review(client, doctor_token: str, prescription_id: int) -> dict:
    response = client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(doctor_token))
    assert response.status_code == 200, response.text
    return response.json()["data"]


def _resume(client, token: str, trace_id: str, approved: bool, comment: str = "同意本次审方结果"):
    return client.post(
        f"/api/agents/doctor/review/{trace_id}/resume",
        headers=auth_header(token),
        json={"approved": approved, "comment": comment},
    )


def _pending_action(db_session, trace_id: str) -> PendingHumanAction | None:
    return db_session.scalar(select(PendingHumanAction).where(PendingHumanAction.trace_id == trace_id))


def _event_types(db_session, trace_id: str) -> set[str]:
    rows = db_session.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == trace_id)).all()
    return {row.event_type for row in rows}


def _tool_calls(db_session, trace_id: str) -> list[str]:
    rows = db_session.scalars(select(MCPToolCall).where(MCPToolCall.trace_id == trace_id).order_by(MCPToolCall.created_at)).all()
    return [row.tool_name for row in rows]


def _packet_count(db_session, trace_id: str) -> int:
    return len(db_session.scalars(select(EvidencePacketLog).where(EvidencePacketLog.trace_id == trace_id)).all())


def test_doctor_review_interrupts_and_creates_pending_action(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    patient_token = login(client, "patient_a", "patient")
    prescription_id = _create_prescription(client, doctor_token, patient["id"])

    review = _review(client, doctor_token, prescription_id)

    assert review["interrupted"] is True
    assert review["requires_human_confirm"] is True
    assert review["trace_id"]
    assert review["thread_id"]
    assert "HumanConfirmNode" in review["graph_node_path"]
    action = _pending_action(db_session, review["trace_id"])
    assert action is not None
    assert action.status == "pending"
    assert action.thread_id == review["thread_id"]
    assert action.interrupt_payload_json["prescription_id"] == prescription_id
    assert action.interrupt_payload_json["evidence_packet_id"] == review["evidence_packet_id"]
    assert "risk_level" in action.interrupt_payload_json
    assert action.interrupt_payload_json["review_summary"]
    assert db_session.get(Prescription, prescription_id).status == "reviewing"

    patient_view = client.get(f"/api/patient/prescriptions/{prescription_id}", headers=auth_header(patient_token))
    assert patient_view.status_code == 404

    trace = client.get(f"/api/harness/traces/{review['trace_id']}", headers=auth_header(doctor_token))
    assert trace.status_code == 200
    trace_data = trace.json()["data"]
    assert trace_data["pending_human_actions"][0]["status"] == "pending"
    assert trace_data["evidence_packets"][0]["human_confirm"]["required"] is True


def test_approved_resume_confirms_without_publish_or_duplicate_mcp(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    patient_token = login(client, "patient_a", "patient")
    prescription_id = _create_prescription(client, doctor_token, patient["id"])
    review = _review(client, doctor_token, prescription_id)
    tools_before = _tool_calls(db_session, review["trace_id"])
    packets_before = _packet_count(db_session, review["trace_id"])

    response = _resume(client, doctor_token, review["trace_id"], True)

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["human_confirmed"] is True
    assert data["resume_action"] == "approved"
    assert data["prescription_status"] == "doctor_confirmed"
    db_session.expire_all()
    assert db_session.get(Prescription, prescription_id).status == "doctor_confirmed"
    action = _pending_action(db_session, review["trace_id"])
    assert action.status == "approved"
    assert action.resume_payload_json["approved"] is True
    assert {"interrupt", "resume", "human_approved"}.issubset(_event_types(db_session, review["trace_id"]))
    packet = db_session.scalar(select(EvidencePacketLog).where(EvidencePacketLog.packet_id == review["evidence_packet_id"]))
    assert packet.trace_id == review["trace_id"]
    assert packet.human_confirm_json["resumed"] is True
    assert packet.human_confirm_json["approved"] is True
    assert _tool_calls(db_session, review["trace_id"]) == tools_before
    assert _packet_count(db_session, review["trace_id"]) == packets_before
    history = db_session.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient["id"]))
    health = db_session.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == patient["id"]))
    assert "夜间咳嗽、咽干" in history.summary
    assert "【诊疗记录1】" in history.summary
    assert "RX-" not in history.summary
    assert health.current_symptoms == "夜间咳嗽、咽干"

    patient_view = client.get(f"/api/patient/prescriptions/{prescription_id}", headers=auth_header(patient_token))
    assert patient_view.status_code == 404


def test_rejected_resume_keeps_prescription_reviewing(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    prescription_id = _create_prescription(client, doctor_token, patient["id"])
    review = _review(client, doctor_token, prescription_id)

    response = _resume(client, doctor_token, review["trace_id"], False, "需要修改处方后再确认")

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["human_confirmed"] is False
    assert data["resume_action"] == "rejected"
    assert data["prescription_status"] == "reviewing"
    db_session.expire_all()
    assert db_session.get(Prescription, prescription_id).status == "reviewing"
    assert _pending_action(db_session, review["trace_id"]).status == "rejected"
    assert "human_rejected" in _event_types(db_session, review["trace_id"])


def test_resume_permission_denied_for_patient_student_and_unbound_doctor(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    patient_token = login(client, "patient_a", "patient")
    register(client, "phase6_student", "student")
    student_token = login(client, "phase6_student", "student")
    register(client, "phase6_unbound_doctor", "doctor")
    unbound_doctor_token = login(client, "phase6_unbound_doctor", "doctor")
    prescription_id = _create_prescription(client, doctor_token, patient["id"])
    review = _review(client, doctor_token, prescription_id)

    for token in [patient_token, student_token, unbound_doctor_token]:
        response = _resume(client, token, review["trace_id"], True)
        assert response.status_code == 403
        assert response.json()["error_code"] == "RESUME_PERMISSION_DENIED"

    assert _pending_action(db_session, review["trace_id"]).status == "pending"


def test_resume_is_idempotent(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    prescription_id = _create_prescription(client, doctor_token, patient["id"])
    review = _review(client, doctor_token, prescription_id)

    first = _resume(client, doctor_token, review["trace_id"], True)
    second = _resume(client, doctor_token, review["trace_id"], True)
    third = _resume(client, doctor_token, review["trace_id"], False)

    assert first.status_code == 200, first.text
    assert second.status_code == 409
    assert second.json()["error_code"] == "HUMAN_ACTION_ALREADY_RESUMED"
    assert third.status_code == 409
    assert third.json()["error_code"] == "HUMAN_ACTION_ALREADY_RESUMED"
    assert _pending_action(db_session, review["trace_id"]).status == "approved"


def test_resume_checkpoint_missing_returns_409(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    prescription_id = _create_prescription(client, doctor_token, patient["id"])
    review = _review(client, doctor_token, prescription_id)
    get_langgraph_checkpointer.cache_clear()

    response = _resume(client, doctor_token, review["trace_id"], True)

    assert response.status_code == 409
    assert response.json()["error_code"] == "CHECKPOINT_NOT_FOUND"
    assert _pending_action(db_session, review["trace_id"]).status == "checkpoint_lost"
    assert "checkpoint_missing" in _event_types(db_session, review["trace_id"])


def test_interrupt_disabled_keeps_phase5_fallback_path(client, db_session):
    settings = get_settings()
    original = settings.langgraph_enable_interrupt
    settings.langgraph_enable_interrupt = False
    try:
        doctor_token, patient = doctor_with_patient(client)
        prescription_id = _create_prescription(client, doctor_token, patient["id"])
        review = _review(client, doctor_token, prescription_id)
    finally:
        settings.langgraph_enable_interrupt = original

    assert review["requires_human_confirm"] is True
    assert review["interrupted"] is False
    assert _pending_action(db_session, review["trace_id"]) is None
    assert db_session.get(Prescription, prescription_id).status == "reviewing"
