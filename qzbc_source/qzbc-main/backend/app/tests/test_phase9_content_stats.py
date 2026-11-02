from app.content.services.content_import_service import ContentImportService
from app.content.services.content_stats_service import content_stats
from app.tests.conftest import auth_header, login, register
from app.tests.test_phase9_content_import import _content_fixture


def test_content_stats_returns_imported_counts(db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()

    stats = content_stats(db_session)

    assert stats["sources"] == 1
    assert stats["knowledge_entries"] == 2
    assert stats["herbs"] == 1
    assert stats["formulas"] == 1
    assert stats["learning_cases"] == 1
    assert stats["quizzes"] == 1
    assert stats["student_learning_cases"] == 1
    assert stats["coverage"]["source_coverage_rate"] == 1.0
    assert stats["coverage"]["quiz_case_binding_rate"] == 1.0


def test_content_api_validate_stats_and_quality_report(client):
    register(client, "phase9_content_doctor", "doctor")
    token = login(client, "phase9_content_doctor", "doctor")

    validate = client.post("/api/content/validate", headers=auth_header(token))
    stats_response = client.get("/api/content/stats", headers=auth_header(token))
    quality = client.get("/api/content/quality-report", headers=auth_header(token))

    assert validate.status_code == 200
    assert "valid" in validate.json()["data"]
    assert stats_response.status_code == 200
    assert "knowledge_entries" in stats_response.json()["data"]
    assert quality.status_code == 200
    assert "metrics" in quality.json()["data"]
