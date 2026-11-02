import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.base import Base
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.schemas import LLMRequest
from app.llm.service import LLMService


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _packet() -> dict:
    return {
        "packet_id": "ollama_test_packet",
        "visibility": "patient_safe",
        "retrieved_evidence": [{"source_id": "SRC_OLLAMA", "summary": "测试证据。"}],
        "rule_findings": [],
        "graph_reasoning_path": [],
        "tool_calls": [],
    }


def _request() -> LLMRequest:
    return LLMRequest(role="patient", task_type="patient_education", input_text="解释酸枣仁", evidence_packet=_packet())


def test_ollama_health_reads_tags(monkeypatch):
    settings = get_settings()
    original_base = settings.ollama_base_url
    original_model = settings.ollama_model
    settings.ollama_base_url = "http://ollama.test"
    settings.ollama_model = "qwen2.5:7b"

    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse({"models": [{"name": "qwen2.5:7b"}]})

    monkeypatch.setattr("app.llm.providers.ollama_provider.urllib.request.urlopen", fake_urlopen)
    try:
        health = OllamaProvider().health()
    finally:
        settings.ollama_base_url = original_base
        settings.ollama_model = original_model

    assert health.configured is True
    assert health.reachable is True
    assert health.status == "ok"
    assert health.model == "qwen2.5:7b"


def test_ollama_provider_parses_valid_json(monkeypatch):
    settings = get_settings()
    original_base = settings.ollama_base_url
    original_model = settings.ollama_model
    original_temperature = settings.llm_temperature
    original_top_p = settings.llm_top_p
    original_seed = settings.llm_seed
    settings.ollama_base_url = "http://ollama.test"
    settings.ollama_model = "qwen2.5:7b"
    settings.llm_temperature = 0.0
    settings.llm_top_p = 0.9
    settings.llm_seed = 42
    captured = {}
    content = {
        "title": "中医药科普说明",
        "summary": "基于证据解释。",
        "sections": [{"heading": "说明", "content": "只做科普。", "source_ids": ["SRC_OLLAMA"]}],
        "warnings": [],
        "next_actions": [],
        "disclaimer": "不能替代医生诊疗。",
    }

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured.update(json.loads(req.data.decode("utf-8")))
        return _FakeResponse({"message": {"content": json.dumps(content, ensure_ascii=False)}})

    monkeypatch.setattr("app.llm.providers.ollama_provider.urllib.request.urlopen", fake_urlopen)
    try:
        result = OllamaProvider().generate(_request())
    finally:
        settings.ollama_base_url = original_base
        settings.ollama_model = original_model
        settings.llm_temperature = original_temperature
        settings.llm_top_p = original_top_p
        settings.llm_seed = original_seed

    assert result.provider == "ollama"
    assert result.structured_output["title"] == "中医药科普说明"
    assert result.metadata.parse_success is True
    assert result.metadata.schema_valid is True
    assert result.grounding.source_ids == ["SRC_OLLAMA"]
    assert captured["options"]["temperature"] == 0.0
    assert captured["options"]["top_p"] == 0.9
    assert captured["options"]["seed"] == 42


def test_ollama_parser_failure_falls_back_to_template(monkeypatch, db_session):
    settings = get_settings()
    original_provider = settings.llm_provider
    original_base = settings.ollama_base_url
    original_model = settings.ollama_model
    settings.llm_provider = "ollama"
    settings.ollama_base_url = "http://ollama.test"
    settings.ollama_model = "qwen2.5:7b"

    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse({"message": {"content": "非 JSON 输出"}})

    monkeypatch.setattr("app.llm.providers.ollama_provider.urllib.request.urlopen", fake_urlopen)
    try:
        result = LLMService(db_session).generate(_request())
    finally:
        settings.llm_provider = original_provider
        settings.ollama_base_url = original_base
        settings.ollama_model = original_model

    assert result.provider == "template"
    assert result.metadata.fallback_used is True
    assert result.metadata.degraded is True
    assert "parser_error" in result.metadata.fallback_reason


def test_parser_failure_without_fallback_has_truthful_metadata(monkeypatch):
    settings = get_settings()
    original = {
        "llm_provider": settings.llm_provider,
        "llm_allow_fallback": settings.llm_allow_fallback,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
    }
    settings.llm_provider = "ollama"
    settings.llm_allow_fallback = False
    settings.ollama_base_url = "http://ollama.test"
    settings.ollama_model = "qwen2.5:7b"

    def fake_urlopen(req, timeout):  # noqa: ARG001
        return _FakeResponse({"message": {"content": "非 JSON 输出"}})

    monkeypatch.setattr("app.llm.providers.ollama_provider.urllib.request.urlopen", fake_urlopen)
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db_session:
            result = LLMService(db_session).generate(_request())
    finally:
        engine.dispose()
        for key, value in original.items():
            setattr(settings, key, value)

    assert result.success is False
    assert result.provider == "ollama"
    assert result.model == "qwen2.5:7b"
    assert result.metadata.parse_success is False
    assert result.metadata.schema_valid is False
    assert result.metadata.parser_error == "invalid_json"
    assert result.metadata.provider_unavailable is False


def test_provider_failure_without_fallback_does_not_claim_parse_success(monkeypatch):
    settings = get_settings()
    original = {
        "llm_provider": settings.llm_provider,
        "llm_allow_fallback": settings.llm_allow_fallback,
        "ollama_model": settings.ollama_model,
    }
    settings.llm_provider = "ollama"
    settings.llm_allow_fallback = False
    settings.ollama_model = "qwen2.5:7b"

    def fail_before_payload(self, request):  # noqa: ARG001
        raise RuntimeError("generation_failed")

    monkeypatch.setattr(OllamaProvider, "generate", fail_before_payload)
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db_session:
            result = LLMService(db_session).generate(_request())
    finally:
        engine.dispose()
        for key, value in original.items():
            setattr(settings, key, value)

    assert result.success is False
    assert result.error_code == "LLM_PROVIDER_FAILED"
    assert result.metadata.provider_unavailable is True
    assert result.metadata.parse_success is False
    assert result.metadata.schema_valid is False
    assert result.metadata.parser_error is None
