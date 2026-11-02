from sqlalchemy import select

from app.db.models import AgentTraceEvent, EvidencePacketLog, MCPToolCall, Prescription
from app.tests.conftest import auth_header, create_reviewed_prescription, doctor_with_patient, login, register


def _tool_names(db_session, trace_id: str) -> set[str]:
    rows = db_session.scalars(select(MCPToolCall).where(MCPToolCall.trace_id == trace_id)).all()
    return {row.tool_name for row in rows}


def _event_nodes(db_session, trace_id: str) -> set[str]:
    rows = db_session.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == trace_id)).all()
    return {row.node_name for row in rows}


def test_patient_high_risk_request_blocks_before_mcp_tool_node(client, db_session):
    register(client, "phase5_patient", "patient")
    token = login(client, "phase5_patient", "patient")

    response = client.post(
        "/api/patient/education-chat",
        headers=auth_header(token),
        json={"message": "请直接给我开方治疗失眠"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["allowed"] is False
    assert "SafetyPreNode" in data["graph_node_path"]
    assert "MCPToolNode" not in data["graph_node_path"]
    assert _tool_names(db_session, data["trace_id"]) == set()


def test_patient_normal_education_uses_public_tools_only(client, db_session):
    register(client, "phase5_patient_normal", "patient")
    token = login(client, "phase5_patient_normal", "patient")

    response = client.post(
        "/api/patient/education-chat",
        headers=auth_header(token),
        json={"message": "酸枣仁是什么？"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    tools = _tool_names(db_session, data["trace_id"])
    assert "public_tcm_education" in tools
    assert "run_prescription_review" not in tools
    assert data["evidence_packet"]["visibility"] == "patient_safe"


def test_doctor_review_graph_requires_human_confirm_and_traces_mcp(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    create_response = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "Phase5 审方测试",
            "items": [{"herb_name": "黄芪", "dose": "10", "unit": "g"}],
        },
    )
    prescription_id = create_response.json()["data"]["id"]

    review = client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(doctor_token))

    assert review.status_code == 200
    data = review.json()["data"]
    assert data["requires_human_confirm"] is True
    assert "HumanConfirmNode" in data["graph_node_path"]
    assert data["evidence_packet_id"]
    tools = _tool_names(db_session, data["trace_id"])
    assert {"parse_prescription", "run_prescription_review", "search_evidence"}.issubset(tools)
    packet = db_session.scalar(select(EvidencePacketLog).where(EvidencePacketLog.packet_id == data["evidence_packet_id"]))
    assert packet is not None
    assert packet.graph_node_path_json
    prescription = db_session.get(Prescription, prescription_id)
    assert prescription.status == "reviewing"


def test_student_training_graph_returns_deidentified_evidence(client):
    doctor_token, patient = doctor_with_patient(client)
    prescription_id = create_reviewed_prescription(client, doctor_token, patient["id"])
    confirm = client.post(f"/api/doctor/prescriptions/{prescription_id}/confirm", headers=auth_header(doctor_token))
    assert confirm.status_code == 200
    deidentify = client.post(f"/api/doctor/prescriptions/{prescription_id}/deidentify", headers=auth_header(doctor_token))
    assert deidentify.status_code == 200
    case_id = deidentify.json()["data"]["id"]
    register(client, "phase5_student", "student")
    student_token = login(client, "phase5_student", "student")

    standard = client.get(f"/api/student/learning-cases/{case_id}/standard-analysis", headers=auth_header(student_token))

    assert standard.status_code == 200
    data = standard.json()["data"]
    packet = data["evidence_packet"]
    assert packet["visibility"] == "student_learning"
    assert packet["graph_node_path"]
    packet_text = str(packet)
    assert "Patient A" not in packet_text
    assert "手机号" not in packet_text


def test_agents_graph_catalog_api(client):
    register(client, "phase5_graph_doctor", "doctor")
    token = login(client, "phase5_graph_doctor", "doctor")

    response = client.get("/api/agents/graphs", headers=auth_header(token))

    assert response.status_code == 200
    names = {item["name"] for item in response.json()["data"]}
    assert {"DoctorReviewGraph", "PatientEducationGraph", "StudentTrainingGraph"}.issubset(names)
