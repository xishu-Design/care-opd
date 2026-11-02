from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import EvidencePacketLog, MCPToolCall, MemoryAccessLog
from app.tests.conftest import auth_header, login, register


def test_knowledge_reindex_and_search_api(client):
    register(client, "doctor_phase3_knowledge", "doctor")
    token = login(client, "doctor_phase3_knowledge", "doctor")

    reindex = client.post("/api/knowledge/reindex", headers=auth_header(token))
    assert reindex.status_code == 200
    assert reindex.json()["data"]["entries_count"] >= 5

    herb = client.get("/api/knowledge/search?query=黄芪", headers=auth_header(token))
    assert herb.status_code == 200
    evidence = herb.json()["data"]["evidence"]
    assert evidence
    assert evidence[0]["source_id"] == "SRC_MOCK_TCM_001"
    assert evidence[0]["entry_id"] == "HERB_001"

    missing = client.get("/api/knowledge/search?query=不存在的草药证据", headers=auth_header(token))
    assert missing.status_code == 200
    assert missing.json()["data"]["evidence"] == []
    assert missing.json()["data"]["degraded"] is True


def test_rules_reload_and_doctor_review_use_structured_rules(client, db_session):
    register(client, "doctor_phase3_rules", "doctor")
    token = login(client, "doctor_phase3_rules", "doctor")
    reload_rules = client.post("/api/rules/reload", headers=auth_header(token))
    assert reload_rules.status_code == 200

    patient = client.post(
        "/api/doctor/patients",
        headers=auth_header(token),
        json={
            "username": "patient_phase3_rules",
            "password": "password123",
            "display_name": "Phase3 Patient",
            "age": 40,
            "sex": "female",
            "allergy_history": "黄芪",
            "medical_history": "无特殊",
            "body_state": "体质偏虚",
            "current_symptoms": "乏力",
        },
    )
    assert patient.status_code == 200, patient.text
    patient_id = patient.json()["data"]["id"]
    prescription = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(token),
        json={
            "patient_id": patient_id,
            "title": "第三阶段过敏与缺失用法测试",
            "items": [{"herb_name": "黄芪", "dose": "10", "unit": "g"}],
        },
    )
    assert prescription.status_code == 200, prescription.text
    prescription_id = prescription.json()["data"]["id"]

    review = client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(token))
    assert review.status_code == 200, review.text
    data = review.json()["data"]
    rule_ids = {finding["rule_id"] for finding in data["rule_findings"]}
    assert {"RULE_MISSING_USAGE", "RULE_ALLERGY_CONFLICT"} <= rule_ids
    assert "RULE_DISABLED_DEMO" not in rule_ids
    assert data["risk_level"] == "red"
    allergy_finding = next(item for item in data["rule_findings"] if item["rule_id"] == "RULE_ALLERGY_CONFLICT")
    assert allergy_finding["matched_herbs"] == ["黄芪"]
    assert data["review_summary"].startswith("结构化规则触发")
    assert data["evidence_packet"]["structured_context"]["prescription"]["items"] == [
        {"herb_name": "黄芪", "dose": "10", "unit": "g", "notes": None}
    ]
    assert data["evidence_packet"]["structured_context"]["allergy_matched_herbs"] == ["黄芪"]
    assert "黄芪 10 g" in str(data["generated_response"])
    assert data["evidence_packet"]["backend_summary"]["rule_engine_backend"] == "local_rules"
    assert any(item["source_id"] == "SRC_MOCK_TCM_001" for item in data["evidence_packet"]["retrieved_evidence"])

    packet = db_session.scalar(select(EvidencePacketLog).where(EvidencePacketLog.task_type == "prescription_review"))
    assert packet.backend_summary_json["knowledge_backend"] == "local_knowledge"
    assert packet.backend_summary_json["fallback_used"] is False


def test_persistent_memory_isolated_and_cross_role_access_logged(client, db_session):
    doctor = register(client, "doctor_phase3_memory", "doctor")
    doctor_token = login(client, "doctor_phase3_memory", "doctor")
    register(client, "patient_phase3_memory", "patient")
    patient_token = login(client, "patient_phase3_memory", "patient")

    write = client.post(
        "/api/mcp/mock-call",
        headers=auth_header(doctor_token),
        json={
            "tool_name": "write_role_memory",
            "arguments": {"memory_type": "format_preference", "content": {"format": "risk_cards_first"}},
            "context": {"task_type": "prescription_review"},
        },
    )
    assert write.status_code == 200
    assert write.json()["data"]["success"] is True

    read = client.post(
        "/api/mcp/mock-call",
        headers=auth_header(doctor_token),
        json={
            "tool_name": "search_role_memory",
            "arguments": {"owner_role": "doctor", "owner_user_id": doctor["id"]},
            "context": {"task_type": "prescription_review"},
        },
    )
    assert read.status_code == 200
    assert "risk_cards_first" in str(read.json()["data"]["data"]["memory_used"])

    blocked = client.post(
        "/api/mcp/mock-call",
        headers=auth_header(patient_token),
        json={
            "tool_name": "search_role_memory",
            "arguments": {"owner_role": "doctor", "owner_user_id": doctor["id"]},
            "context": {"task_type": "patient_education"},
        },
    )
    assert blocked.status_code == 200
    assert blocked.json()["data"]["allowed"] is False
    assert blocked.json()["data"]["error_code"] == "MEMORY_SCOPE_VIOLATION"
    assert db_session.scalar(select(MemoryAccessLog).where(MemoryAccessLog.allowed.is_(False))) is not None


def test_adapter_fallback_metadata_is_explicit(client, db_session):
    settings = get_settings()
    original_backend = settings.knowledge_backend
    settings.knowledge_backend = "lightrag"
    try:
        register(client, "doctor_phase3_fallback", "doctor")
        token = login(client, "doctor_phase3_fallback", "doctor")
        response = client.post(
            "/api/mcp/mock-call",
            headers=auth_header(token),
            json={
                "tool_name": "search_evidence",
                "arguments": {"query": "黄芪"},
                "context": {"task_type": "prescription_review"},
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["metadata"]["fallback_used"] is True
        assert data["metadata"]["backend"] == "local_knowledge"
        assert data["metadata"]["configured_backend"] == "lightrag"
        call = db_session.scalar(select(MCPToolCall).where(MCPToolCall.tool_name == "search_evidence"))
        assert call.fallback_used is True
        assert call.fallback_reason.startswith("lightrag_health_")
    finally:
        settings.knowledge_backend = original_backend
