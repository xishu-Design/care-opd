from __future__ import annotations

import json
import urllib.error
import urllib.request
from time import perf_counter

from app.core.config import get_settings
from app.llm.fallback import evidence_count, extract_source_ids
from app.llm.output_parser import parse_generated_response_result
from app.llm.providers.base import BaseLLMProvider, ProviderGenerationError, ProviderUnavailable
from app.llm.providers.ollama_provider import (
    _chat_messages,
    _normalize_case_reasoning_output,
)
from app.llm.schemas import LLMGrounding, LLMMetadata, LLMRequest, LLMResult, ProviderHealth


_SAFE_REMOTE_ERROR_CODES = {
    "context_length_exceeded",
    "generation_failed",
    "invalid_json",
    "invalid_max_tokens",
    "invalid_message",
    "invalid_request_size",
    "invalid_sampling_parameters",
    "invalid_seed",
    "max_tokens_out_of_range",
    "messages_must_be_a_nonempty_list",
    "only_temperature_zero_is_supported",
    "only_top_p_one_is_supported",
    "request_must_be_an_object",
    "streaming_not_supported",
    "unknown_model",
}


def _safe_http_error(exc: urllib.error.HTTPError) -> str:
    """Return an allowlisted diagnostic without retaining response content."""

    safe_code = "unspecified"
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        error = payload.get("error") if isinstance(payload, dict) else None
        candidate = error.get("message") if isinstance(error, dict) else None
        if candidate in _SAFE_REMOTE_ERROR_CODES:
            safe_code = str(candidate)
    except Exception:
        pass
    return f"http_{int(exc.code)}:{safe_code}"


class OpenAICompatibleProvider(BaseLLMProvider):
    name = "openai_compatible"

    def health(self) -> ProviderHealth:
        settings = get_settings()
        if not settings.openai_compatible_base_url or not settings.openai_compatible_model:
            return ProviderHealth(
                name=self.name,
                configured=False,
                reachable=False,
                status="misconfigured",
                model=settings.openai_compatible_model or None,
                last_error="OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL are required",
            )
        started = perf_counter()
        headers = {"Content-Type": "application/json"}
        if settings.openai_compatible_api_key:
            headers["Authorization"] = f"Bearer {settings.openai_compatible_api_key}"
        try:
            req = urllib.request.Request(settings.openai_compatible_base_url.rstrip("/") + "/models", headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=settings.ollama_health_timeout_seconds) as response:
                response.read()
        except Exception as exc:
            return ProviderHealth(
                name=self.name,
                configured=True,
                reachable=False,
                status="unavailable",
                model=settings.openai_compatible_model,
                last_error=str(exc) or exc.__class__.__name__,
            )
        return ProviderHealth(
            name=self.name,
            configured=True,
            reachable=True,
            status="ok",
            latency_ms=int((perf_counter() - started) * 1000),
            model=settings.openai_compatible_model,
        )

    def generate(self, request: LLMRequest) -> LLMResult:
        settings = get_settings()
        if not settings.openai_compatible_base_url or not settings.openai_compatible_model:
            raise ProviderUnavailable("OpenAI-compatible provider is not configured")
        started = perf_counter()
        payload = {
            "model": settings.openai_compatible_model,
            "messages": _chat_messages(request, streaming=False),
            "temperature": settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "seed": settings.llm_seed,
            "max_tokens": settings.llm_max_output_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if settings.openai_compatible_api_key:
            headers["Authorization"] = f"Bearer {settings.openai_compatible_api_key}"
        try:
            req = urllib.request.Request(
                settings.openai_compatible_base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=settings.llm_timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderGenerationError(_safe_http_error(exc)) from exc
        except Exception as exc:
            raise ProviderGenerationError(str(exc) or exc.__class__.__name__) from exc
        choice = (raw.get("choices") or [{}])[0]
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason == "length":
            raise ProviderGenerationError("output_truncated:finish_reason_length")
        content = choice.get("message", {}).get("content", "")
        if not content:
            raise ProviderGenerationError("OpenAI-compatible provider returned empty content")
        parsed = parse_generated_response_result(content)
        if not parsed.success or parsed.structured_output is None:
            raise ProviderGenerationError(f"parser_error:{parsed.parse_error or 'invalid_output'}")
        structured_output = _normalize_case_reasoning_output(
            request, parsed.structured_output
        )
        source_ids = extract_source_ids(request.evidence_packet)
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return LLMResult(
            request_id=request.request_id,
            success=True,
            provider=self.name,
            model=settings.openai_compatible_model,
            content=content,
            structured_output=structured_output,
            grounding=LLMGrounding(
                used_evidence_count=evidence_count(request.evidence_packet),
                missing_evidence=evidence_count(request.evidence_packet) == 0 or not source_ids,
                source_ids=source_ids,
            ),
            metadata=LLMMetadata(
                latency_ms=int((perf_counter() - started) * 1000),
                parse_success=parsed.success,
                schema_valid=parsed.schema_valid,
                parser_error=parsed.parse_error,
                token_usage={
                    "input_tokens": int(usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                },
            ),
        )
