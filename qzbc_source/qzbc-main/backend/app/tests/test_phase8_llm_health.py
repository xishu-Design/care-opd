from app.core.config import get_settings
from app.llm.service import LLMService


def test_template_health_returns_enhanced_fields(db_session):
    settings = get_settings()
    original_provider = settings.llm_provider
    settings.llm_provider = "template"
    try:
        health = LLMService(db_session).health()
    finally:
        settings.llm_provider = original_provider

    assert health["llm_enabled"] is True
    assert health["configured_provider"] == "template"
    assert health["active_provider"] == "template"
    assert health["provider_status"] == "ok"
    assert health["configured"] is True
    assert health["reachable"] is True
    assert health["fallback_available"] is True
    assert "latency_ms" in health
    assert "last_error" in health
    assert {item["name"] for item in health["providers"]} >= {"template", "ollama", "openai_compatible"}


def test_ollama_unavailable_health_does_not_raise(db_session):
    settings = get_settings()
    original_provider = settings.llm_provider
    original_base = settings.ollama_base_url
    original_timeout = settings.ollama_health_timeout_seconds
    settings.llm_provider = "ollama"
    settings.ollama_base_url = "http://127.0.0.1:1"
    settings.ollama_health_timeout_seconds = 1
    try:
        health = LLMService(db_session).health()
    finally:
        settings.llm_provider = original_provider
        settings.ollama_base_url = original_base
        settings.ollama_health_timeout_seconds = original_timeout

    assert health["configured_provider"] == "ollama"
    assert health["active_provider"] == "template"
    assert health["provider_status"] == "fallback"
    assert health["configured"] is True
    assert health["reachable"] is False
    assert health["fallback_available"] is True
    assert health["last_error"]


def test_llm_health_api_shape_remains_demo_compatible(client):
    response = client.get("/api/llm/health")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["llm_enabled"] is True
    assert "provider" in data
    assert "provider_status" in data
    assert "configured" in data
    assert "reachable" in data
    assert "fallback_available" in data
