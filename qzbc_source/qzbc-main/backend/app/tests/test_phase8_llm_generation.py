from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import AgentTraceEvent, LLMGenerationLog
from app.llm.schemas import LLMRequest
from app.llm.service import LLMService
from app.tests.conftest import auth_header, doctor_with_patient, login, register


def _evidence_packet(visibility: str = "patient_safe") -> dict:
    return {
        "packet_id": "test_phase8_packet",
        "visibility": visibility,
        "case_summary": "phase8 test evidence",
        "retrieved_evidence": [{"source_id": "SRC_TEST", "summary": "测试证据。"}],
        "rule_findings": [{"risk_level": "yellow", "message": "需要复核。"}],
        "graph_reasoning_path": [],
        "tool_calls": [],
    }


def test_template_provider_generates_three_role_outputs(db_session):
    service = LLMService(db_session)
    doctor = service.generate(
        LLMRequest(role="doctor", task_type="doctor_review_text", evidence_packet=_evidence_packet("doctor_internal"))
    )
    patient = service.generate(
        LLMRequest(role="patient", task_type="patient_education", evidence_packet=_evidence_packet("patient_safe"))
    )
    student = service.generate(
        LLMRequest(role="student", task_type="student_analysis", evidence_packet=_evidence_packet("student_learning"))
    )

    assert doctor.provider == "template"
    assert "医生" in doctor.structured_output["title"]
    assert patient.structured_output["disclaimer"]
    assert "教学训练" in student.structured_output["disclaimer"]
    assert doctor.grounding.source_ids == ["SRC_TEST"]
    assert db_session.scalar(select(LLMGenerationLog)) is not None


def test_ollama_failure_falls_back_to_template(db_session):
    settings = get_settings()
    original_provider = settings.llm_provider
    original_base = settings.ollama_base_url
    settings.llm_provider = "ollama"
    settings.ollama_base_url = "http://127.0.0.1:1"
    try:
        result = LLMService(db_session).generate(
            LLMRequest(role="patient", task_type="patient_education", evidence_packet=_evidence_packet())
        )
    finally:
        settings.llm_provider = original_provider
        settings.ollama_base_url = original_base

    assert result.provider == "template"
    assert result.metadata.fallback_used is True
    assert result.metadata.degraded is True


def test_llm_health_and_test_generate_api(client):
    register(client, "phase8_doctor", "doctor")
    token = login(client, "phase8_doctor", "doctor")

    health = client.get("/api/llm/health")
    assert health.status_code == 200
    assert health.json()["data"]["llm_enabled"] is True

    providers = client.get("/api/llm/providers", headers=auth_header(token))
    assert providers.status_code == 200
    assert {item["name"] for item in providers.json()["data"]["items"]} >= {"template", "ollama", "openai_compatible"}

    generated = client.post(
        "/api/llm/test-generate",
        headers=auth_header(token),
        json={"role": "patient", "task_type": "patient_education", "message": "酸枣仁是什么？"},
    )
    assert generated.status_code == 200
    assert generated.json()["data"]["provider"] == "template"


def test_patient_high_risk_request_does_not_enter_llm_node(client, db_session):
    register(client, "phase8_patient", "patient")
    token = login(client, "phase8_patient", "patient")

    response = client.post(
        "/api/patient/education-chat",
        headers=auth_header(token),
        json={"message": "我能不能把医生开的药加量？"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["allowed"] is False
    events = db_session.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == data["trace_id"])).all()
    assert "LLMGenerateNode" not in {event.node_name for event in events}
    assert db_session.scalar(select(LLMGenerationLog).where(LLMGenerationLog.trace_id == data["trace_id"])) is None


def test_doctor_review_graph_records_llm_generation(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    create_response = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={"patient_id": patient["id"], "title": "Phase8 审方", "items": [{"herb_name": "黄芪", "dose": "10", "unit": "g"}]},
    )
    prescription_id = create_response.json()["data"]["id"]

    review = client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(doctor_token))

    assert review.status_code == 200
    data = review.json()["data"]
    assert data["llm_result"]["provider"] == "template"
    assert "LLMGenerateNode" in data["graph_node_path"]
    llm_log = db_session.scalar(select(LLMGenerationLog).where(LLMGenerationLog.trace_id == data["trace_id"]))
    assert llm_log is not None
    events = db_session.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == data["trace_id"])).all()
    assert "llm_generation" in {event.event_type for event in events}
