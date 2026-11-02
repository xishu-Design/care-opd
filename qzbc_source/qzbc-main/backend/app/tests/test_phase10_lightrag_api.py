from app.core.config import get_settings
from app.tests.conftest import auth_header, login, register


def test_lightrag_health_and_query_api_return_degraded_when_disabled(client):
    register(client, "phase10_api_doctor", "doctor")
    token = login(client, "phase10_api_doctor", "doctor")
    settings = get_settings()
    original = _snapshot_settings()
    settings.lightrag_enabled = False
    try:
        health = client.get("/api/lightrag/health", headers=auth_header(token))
        query = client.get("/api/lightrag/query?query=黄芪", headers=auth_header(token))

        assert health.status_code == 200
        assert health.json()["data"]["status"] == "unavailable"
        assert query.status_code == 200
        assert query.json()["data"]["success"] is False
        assert query.json()["data"]["metadata"]["degraded"] is True
    finally:
        _apply_settings(original)


def test_lightrag_build_docs_and_reindex_api(client, tmp_path):
    register(client, "phase10_api_build_doctor", "doctor")
    token = login(client, "phase10_api_build_doctor", "doctor")
    settings = get_settings()
    original = _snapshot_settings()
    settings.lightrag_input_dir = str(tmp_path / "input_docs")
    settings.lightrag_working_dir = str(tmp_path / "working_dir")
    settings.lightrag_enabled = True
    settings.lightrag_mode = "embedded"
    try:
        build = client.post("/api/lightrag/build-docs", headers=auth_header(token))
        reindex = client.post("/api/lightrag/reindex", headers=auth_header(token))
        report = client.get("/api/lightrag/report", headers=auth_header(token))

        assert build.status_code == 200, build.text
        assert "docs_written" in build.json()["data"]
        assert reindex.status_code == 200, reindex.text
        assert reindex.json()["data"]["status"] in {"success", "degraded"}
        assert report.status_code == 200, report.text
        assert "index_report" in report.json()["data"]
    finally:
        _apply_settings(original)


def _snapshot_settings():
    settings = get_settings()
    return {
        "lightrag_enabled": settings.lightrag_enabled,
        "lightrag_mode": settings.lightrag_mode,
        "lightrag_input_dir": settings.lightrag_input_dir,
        "lightrag_working_dir": settings.lightrag_working_dir,
    }


def _apply_settings(values):
    settings = get_settings()
    for key, value in values.items():
        setattr(settings, key, value)
