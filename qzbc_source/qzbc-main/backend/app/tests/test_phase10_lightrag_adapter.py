from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import EvidencePacketLog, MCPToolCall, User
from app.evidence.evidence_packet import build_evidence_packet_v2
from app.lightrag_integration.document_builder import build_lightrag_docs
from app.lightrag_integration.client import track_state
from app.lightrag_integration.index_builder import build_lightrag_index
from app.mcp.contracts import ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway
from app.tests.test_phase9_content_import import _content_fixture


_ORIGINAL_SETTINGS = None


def test_mcp_uses_lightrag_adapter_when_configured(db_session, tmp_path):
    _prepare_lightrag_settings(tmp_path)
    try:
        _build_docs_and_index(tmp_path)
        doctor = _user(db_session, "phase10_doctor", "doctor")

        result = _call(db_session, doctor, "search_herb", {"query": "黄芪"}, "prescription_review")

        assert result.success is True
        assert result.metadata["backend"] == "lightrag"
        assert result.evidence
        evidence = result.evidence[0]
        assert evidence["retrieved_by"] == "lightrag"
        assert evidence["retrieval_mode"] == "hybrid"
        assert evidence["source_id"] == "SRC_TCM_DATA_FOR_ML"
        assert evidence["entry_id"]
        assert evidence["graph_entities"]
    finally:
        _restore_settings()


def test_lightrag_adapter_filters_patient_doctor_only_evidence(db_session, tmp_path):
    _prepare_lightrag_settings(tmp_path)
    try:
        _build_docs_and_index(tmp_path)
        patient = _user(db_session, "phase10_patient", "patient")

        result = _call(db_session, patient, "search_herb", {"query": "黄芪"}, "patient_education")

        assert result.success is True
        assert result.metadata["backend"] == "lightrag"
        assert result.metadata["degraded"] is True
        assert result.metadata["degraded_reason"] == "visibility_filtered_all_results"
        assert result.evidence == []
        call = db_session.scalar(select(MCPToolCall).where(MCPToolCall.tool_name == "search_herb"))
        assert call.backend == "lightrag"
    finally:
        _restore_settings()


def test_lightrag_unavailable_falls_back_to_local_knowledge(db_session):
    settings = get_settings()
    original = _snapshot_settings()
    settings.knowledge_backend = "lightrag"
    settings.lightrag_enabled = False
    settings.lightrag_allow_fallback = True
    try:
        doctor = _user(db_session, "phase10_fallback_doctor", "doctor")

        result = _call(db_session, doctor, "search_herb", {"query": "黄芪"}, "prescription_review")

        assert result.success is True
        assert result.metadata["backend"] == "local_knowledge"
        assert result.metadata["configured_backend"] == "lightrag"
        assert result.metadata["fallback_used"] is True
        assert result.evidence
    finally:
        _apply_settings(original)


def test_evidence_packet_records_lightrag_graph_metadata(db_session, tmp_path):
    _prepare_lightrag_settings(tmp_path)
    try:
        _build_docs_and_index(tmp_path)
        doctor = _user(db_session, "phase10_packet_doctor", "doctor")
        result = _call(db_session, doctor, "search_herb", {"query": "黄芪"}, "prescription_review")

        packet = build_evidence_packet_v2(
            db_session,
            role="doctor",
            task_type="prescription_review",
            visibility="doctor_internal",
            case_summary="phase10 lightrag evidence",
            tool_results=[result],
            created_by_user_id=doctor.id,
        )

        assert packet["retrieved_evidence"][0]["retrieved_by"] == "lightrag"
        assert packet["retrieved_evidence"][0]["graph_entities"]
        assert packet["backend_summary"]["knowledge_backend"] == "lightrag"
        assert packet["backend_summary"]["lightrag_mode"] == "embedded"
        assert packet["backend_summary"]["graph_retrieval_count"] == 1
        row = db_session.scalar(select(EvidencePacketLog).where(EvidencePacketLog.packet_id == packet["packet_id"]))
        assert row.backend_summary_json["knowledge_backend"] == "lightrag"
    finally:
        _restore_settings()


def _prepare_lightrag_settings(tmp_path):
    global _ORIGINAL_SETTINGS
    settings = get_settings()
    _ORIGINAL_SETTINGS = _snapshot_settings()
    settings.knowledge_backend = "lightrag"
    settings.lightrag_enabled = True
    settings.lightrag_mode = "embedded"
    settings.lightrag_input_dir = str(tmp_path / "input_docs")
    settings.lightrag_working_dir = str(tmp_path / "working_dir")
    settings.lightrag_allow_fallback = True


def _restore_settings():
    global _ORIGINAL_SETTINGS
    if _ORIGINAL_SETTINGS:
        _apply_settings(_ORIGINAL_SETTINGS)
        _ORIGINAL_SETTINGS = None


def _snapshot_settings():
    settings = get_settings()
    return {
        "knowledge_backend": settings.knowledge_backend,
        "lightrag_enabled": settings.lightrag_enabled,
        "lightrag_mode": settings.lightrag_mode,
        "lightrag_input_dir": settings.lightrag_input_dir,
        "lightrag_working_dir": settings.lightrag_working_dir,
        "lightrag_allow_fallback": settings.lightrag_allow_fallback,
    }


def _apply_settings(values):
    settings = get_settings()
    for key, value in values.items():
        setattr(settings, key, value)


def _build_docs_and_index(tmp_path):
    content_dir = _content_fixture(tmp_path)
    docs_path = tmp_path / "input_docs" / "guardtcm_lightrag_docs.jsonl"
    build_lightrag_docs(content_data_dir=content_dir, output_path=docs_path, report_path=tmp_path / "reports" / "doc_report.json")
    build_lightrag_index(docs_path=docs_path, working_dir=tmp_path / "working_dir", report_path=tmp_path / "reports" / "index_report.json", rebuild=True)


def _user(db_session, username: str, role: str) -> User:
    user = User(username=username, hashed_password="test", role=role, display_name=username)
    db_session.add(user)
    db_session.flush()
    return user


def _call(db_session, user: User, tool_name: str, arguments: dict, task_type: str):
    context = ToolContext(user_id=user.id, role=user.role, task_type=task_type)
    return MCPGateway(db_session).execute(ToolRequest(tool_name=tool_name, arguments=arguments, context=context))


def test_track_state_supports_lightrag_154_document_payload():
    assert track_state({"documents": [{"status": "processed"}, {"status": "processed"}]}) == "processed"
    assert track_state({"status_summary": {"DocStatus.PROCESSED": 2}}) == "processed"
    assert track_state({"documents": [{"status": "processing"}]}) == "processing"
    assert track_state({"documents": [{"status": "failed"}]}) == "failed"
