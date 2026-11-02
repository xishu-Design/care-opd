from sqlalchemy import select

from app.content.services.content_import_service import ContentImportService
from app.db.models import LearningCase, User
from app.evidence.evidence_packet import build_evidence_packet_v2
from app.mcp.contracts import ResourceScope, ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway
from app.tests.test_phase9_content_import import _content_fixture


def test_search_herb_returns_imported_open_data_source_metadata(db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()
    doctor = _user(db_session, "phase9_mcp_doctor", "doctor")

    result = _call_tool(db_session, doctor, "search_herb", {"query": "黄芪"}, "prescription_review")

    assert result.success is True
    assert result.evidence
    evidence = result.evidence[0]
    assert evidence["title"] == "黄芪"
    assert evidence["source_ids"] == ["SRC_TCM_DATA_FOR_ML"]
    assert evidence["source_records"][0]["source_record_id"] == "row_1"
    assert evidence["original_source"] == "TCMDataForML"
    assert evidence["metadata"]["license"] == "MIT"
    assert evidence["metadata"]["commit_hash"] == "abc123"
    assert evidence["metadata"]["processing_script"] == "extract_tcmdataforml.py"
    assert evidence["metadata"]["functions"] == []
    assert evidence["metadata"]["cautions"] == []
    assert evidence["metadata"]["dose_range"] is None


def test_search_formula_returns_candidate_components_without_clinical_recommendation(db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()
    doctor = _user(db_session, "phase9_formula_doctor", "doctor")

    result = _call_tool(db_session, doctor, "search_formula", {"query": "黄芪"}, "prescription_review")

    assert result.success is True
    assert result.evidence
    evidence = result.evidence[0]
    assert evidence["not_clinical_recommendation"] is True
    assert evidence["components"] == [{"herb_name": "黄芪", "normalized_name": "黄芪", "dosage": None, "unit": None}]
    assert evidence["metadata"]["source_ids"] == ["SRC_TCM_DATA_FOR_ML"]
    assert evidence["metadata"]["source_records"][0]["source_file"] == "TCM.csv"


def test_learning_case_tools_return_only_deidentified_student_cases(db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()
    student = _user(db_session, "phase9_mcp_student", "student")
    case = db_session.scalar(select(LearningCase))
    unsafe_case = LearningCase(
        source_prescription_id=None,
        title="未脱敏案例",
        anonymized_case={"deidentified": False, "visibility": "student_learning", "patient_name": "Real Patient"},
        standard_analysis="unsafe",
        risk_points=[],
        evidence_chain=[],
    )
    db_session.add(unsafe_case)
    db_session.flush()

    detail = _call_tool(db_session, student, "get_learning_case", {"learning_case_id": case.id}, "student_training")
    listing = _call_tool(db_session, student, "list_learning_cases", {}, "student_training")
    quiz = _call_tool(db_session, student, "generate_mock_quiz", {"learning_case_id": case.id}, "student_training")
    unsafe_detail = _call_tool(db_session, student, "get_learning_case", {"learning_case_id": unsafe_case.id}, "student_training")

    assert detail.success is True
    learning_case = detail.data["learning_case"]
    assert learning_case["deidentified"] is True
    assert learning_case["visibility"] == "student_learning"
    assert learning_case["source_id"] == "SRC_TCM_DATA_FOR_ML"
    assert learning_case["anonymized_case"]["deidentified"] is True
    assert "patient_name" not in str(learning_case)
    assert "phone" not in str(learning_case).lower()
    assert [item["id"] for item in listing.data["learning_cases"]] == [case.id]
    assert quiz.data["quiz"][0]["quiz_id"] == "QUIZ_SRC_TCM_DATA_FOR_ML_000001"
    assert unsafe_detail.allowed is False
    assert unsafe_detail.error_code == "LEARNING_CASE_NOT_FOUND"


def test_patient_cannot_retrieve_doctor_student_open_data(db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()
    patient = _user(db_session, "phase9_mcp_patient", "patient")

    result = _call_tool(db_session, patient, "search_herb", {"query": "黄芪"}, "patient_education")

    assert result.allowed is True
    assert result.success is True
    assert result.evidence == []
    assert result.metadata["degraded"] is True
    assert result.metadata["degraded_reason"] == "no_evidence_found"


def test_evidence_packet_keeps_original_source_from_open_data_search(db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()
    doctor = _user(db_session, "phase9_packet_doctor", "doctor")

    result = _call_tool(db_session, doctor, "search_evidence", {"query": "黄芪"}, "prescription_review")
    packet = build_evidence_packet_v2(
        db_session,
        role="doctor",
        task_type="prescription_review",
        visibility="doctor_internal",
        case_summary="phase9 open data evidence",
        tool_results=[result],
        created_by_user_id=doctor.id,
    )

    assert result.success is True
    assert packet["retrieved_evidence"]
    assert packet["retrieved_evidence"][0]["original_source"] == "TCMDataForML"
    assert packet["retrieved_evidence"][0]["metadata"]["original_source"] == "TCMDataForML"


def _user(db_session, username: str, role: str) -> User:
    user = User(username=username, hashed_password="test", role=role, display_name=username)
    db_session.add(user)
    db_session.flush()
    return user


def _call_tool(db_session, user: User, tool_name: str, arguments: dict, task_type: str):
    learning_case_id = arguments.get("learning_case_id")
    context = ToolContext(
        user_id=user.id,
        role=user.role,
        task_type=task_type,
        learning_case_id=learning_case_id,
        resource_scope=ResourceScope(is_learning_case_deidentified=True) if learning_case_id else ResourceScope(),
    )
    return MCPGateway(db_session).execute(ToolRequest(tool_name=tool_name, arguments=arguments, context=context))
