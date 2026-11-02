from app.content.services.content_import_service import ContentImportService
from app.tests.conftest import auth_header, login, register
from app.tests.test_phase9_content_import import _content_fixture


def test_phase9_source_registry_api_returns_traceable_source_status(client, db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()
    db_session.commit()
    register(client, "phase9_source_doctor", "doctor")
    token = login(client, "phase9_source_doctor", "doctor")

    response = client.get("/api/harness/sources", headers=auth_header(token))

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["summary"]["approved_count"] == 4
    source = next(item for item in data["sources"] if item["source_id"] == "SRC_TCM_DATA_FOR_ML")
    assert source["approval_status"] == "approved"
    assert source["license"] == "MIT"
    assert source["can_process"] is True
    assert source["imported_count"] >= 4


def test_phase9_data_pipeline_api_returns_reports_and_counts(client):
    register(client, "phase9_pipeline_doctor", "doctor")
    token = login(client, "phase9_pipeline_doctor", "doctor")

    response = client.get("/api/harness/data-pipeline", headers=auth_header(token))

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert "download_report" in data["reports"]
    assert "license_report" in data["reports"]
    assert "processing_report" in data["reports"]
    assert "quality_report" in data["reports"]
    assert "processed_item_count" in data["summary"]
    assert "content_item_count" in data["summary"]
    assert "content_source_coverage_rate" in data["content_quality"]["metrics"]
