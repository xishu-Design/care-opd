from sqlalchemy import select

from app.db.models import EvidencePacketLog, GraphPath, LightRAGRetrievalLog, MCPToolCall, MockMemory
from app.memory.memory_router import MemoryRouteRequest, MemoryRouter
from app.tests.conftest import auth_header, create_reviewed_prescription, doctor_with_patient, login, register


def test_memory_router_role_task_plans():
    router = MemoryRouter()
    doctor_plan = router.plan(MemoryRouteRequest(role="doctor", task_type="prescription_review"))
    patient_plan = router.plan(MemoryRouteRequest(role="patient", task_type="patient_education"))
    student_plan = router.plan(MemoryRouteRequest(role="student", task_type="student_training"))

    assert [item.tool_name for item in doctor_plan.memory_plan] == [
        "search_evidence",
        "get_reasoning_path",
        "search_role_memory",
    ]
    assert "public_tcm_education" in [item.tool_name for item in patient_plan.memory_plan]
    assert "search_diet_knowledge" in [item.tool_name for item in patient_plan.memory_plan]
    assert "get_reasoning_path" in [item.tool_name for item in student_plan.memory_plan]


def test_doctor_review_returns_complete_doctor_internal_packet(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    create_response = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "第三阶段缺失用法测试",
            "doctor_notes": "内部备注",
            "items": [{"herb_name": "酸枣仁", "dose": "10", "unit": "g"}],
        },
    )
    assert create_response.status_code == 200, create_response.text
    prescription_id = create_response.json()["data"]["id"]
    review_response = client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(doctor_token))
    assert review_response.status_code == 200, review_response.text
    assert any(card["rule_id"] == "RULE_MISSING_USAGE" for card in review_response.json()["data"]["risk_cards"])
    usage_card = next(card for card in review_response.json()["data"]["risk_cards"] if card["rule_id"] == "RULE_MISSING_USAGE")
    assert "剂量和单位已经填写" in usage_card["message"]
    assert "处方级用法" in usage_card["message"]

    response = client.get(f"/api/doctor/prescriptions/{prescription_id}", headers=auth_header(doctor_token))
    assert response.status_code == 200

    packet = db_session.scalar(select(EvidencePacketLog).where(EvidencePacketLog.task_type == "prescription_review"))
    assert packet is not None
    assert packet.visibility == "doctor_internal"
    assert packet.retrieved_evidence_json
    assert packet.graph_reasoning_path_json
    assert packet.rule_findings_json
    assert packet.memory_used_json
    assert packet.tool_calls_json
    assert packet.backend_summary_json["knowledge_backend"] == "local_knowledge"
    assert packet.backend_summary_json["rule_engine_backend"] == "local_rules"
    assert packet.degraded is False

    tool_names = {call.tool_name for call in db_session.scalars(select(MCPToolCall)).all()}
    assert {"parse_prescription", "run_prescription_review", "search_evidence", "get_reasoning_path", "search_role_memory"} <= tool_names
    assert db_session.scalar(select(LightRAGRetrievalLog)) is not None
    assert db_session.scalar(select(GraphPath)) is not None


def test_patient_education_patient_safe_packet_and_no_review_tools(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    del doctor_token, patient
    patient_token = login(client, "patient_a", "patient")

    response = client.post(
        "/api/patient/education-chat",
        headers=auth_header(patient_token),
        json={"message": "酸枣仁是什么？"},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["allowed"] is True
    packet = data["evidence_packet"]
    assert packet["visibility"] == "patient_safe"
    assert packet["retrieved_evidence"]
    assert packet["graph_reasoning_path"] == []
    assert packet["rule_findings"] == []
    assert all(item.get("owner_role") == "patient" for item in packet["memory_used"])

    tool_names = [call.tool_name for call in db_session.scalars(select(MCPToolCall)).all()]
    assert "public_tcm_education" in tool_names
    assert "search_diet_knowledge" in tool_names
    assert "run_prescription_review" not in tool_names
    assert "run_mock_prescription_review" not in tool_names
    assert "create_prescription_draft" not in tool_names


def test_patient_blocked_request_does_not_trigger_tools(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    del doctor_token, patient
    patient_token = login(client, "patient_a", "patient")
    before = len(db_session.scalars(select(MCPToolCall)).all())
    blocked = client.post(
        "/api/patient/education-chat",
        headers=auth_header(patient_token),
        json={"message": "我最近失眠，帮我开个方。"},
    )
    assert blocked.status_code == 200
    assert blocked.json()["data"]["allowed"] is False
    after = len(db_session.scalars(select(MCPToolCall)).all())
    assert after == before


def test_student_standard_analysis_packet_is_deidentified(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    prescription_id = create_reviewed_prescription(client, doctor_token, patient["id"])
    client.post(f"/api/doctor/prescriptions/{prescription_id}/confirm", headers=auth_header(doctor_token))
    deid = client.post(f"/api/doctor/prescriptions/{prescription_id}/deidentify", headers=auth_header(doctor_token))
    case_id = deid.json()["data"]["id"]

    register(client, "student_phase2", "student")
    student_token = login(client, "student_phase2", "student")
    response = client.get(f"/api/student/learning-cases/{case_id}/standard-analysis", headers=auth_header(student_token))
    assert response.status_code == 200
    data = response.json()["data"]
    packet = data["evidence_packet"]
    assert packet["visibility"] == "student_learning"
    assert "Patient A" not in str(packet)
    assert packet["graph_reasoning_path"]
    assert packet["retrieved_evidence"]
    assert packet["memory_used"]
    assert data["knowledge_cards"]
    assert data["quiz"]
    assert db_session.scalar(select(MockMemory).where(MockMemory.owner_role == "student")) is None or "Patient A" not in str(db_session.scalar(select(MockMemory).where(MockMemory.owner_role == "student")).content_json)
