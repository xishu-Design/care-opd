from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import LLMGenerationLog
from app.harness.trace import record_event
from app.llm.providers.base import BaseLLMProvider
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from app.llm.providers.template_provider import TemplateProvider
from app.llm.safety import apply_output_safety
from app.llm.doctor_response_policy import doctor_direct_response
from app.llm.patient_response_policy import patient_direct_response, response_display_text
from app.llm.schemas import GeneratedResponse, LLMGrounding, LLMMetadata, LLMRequest, LLMResult, ProviderHealth
from app.llm.fallback import evidence_count, extract_source_ids


class LLMService:
    def __init__(self, db: Session):
        self.db = db

    def generate_doctor_review(self, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMResult:
        request = self._request("doctor", "doctor_review_text", evidence_packet, context)
        return self.generate(request)

    def generate_patient_education(self, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMResult:
        request = self._request("patient", "patient_education", evidence_packet, context)
        return self.generate(request)

    def generate_patient_diet_guidance(self, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMResult:
        """Explain a doctor's confirmed diet note without inventing interactions."""
        request = self._request("patient", "patient_diet_guidance", evidence_packet, context)
        return self.generate(request)

    def generate_student_analysis(self, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMResult:
        request = self._request("student", "student_analysis", evidence_packet, context)
        return self.generate(request)

    def generate_student_quiz_explanation(self, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMResult:
        """Explain a source question after its deterministic answer has been shown.

        The answer key remains the authority; Ollama is used only to improve the
        teaching explanation and may safely fall back to the local template.
        """
        request = self._request("student", "student_quiz_explanation", evidence_packet, context)
        return self.generate(request)

    def generate_doctor_ai_chat(self, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMResult:
        request = self._request("doctor", "doctor_ai_chat", evidence_packet, context)
        return self.generate(request)

    def generate_student_ai_chat(self, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMResult:
        request = self._request("student", "student_ai_chat", evidence_packet, context)
        return self.generate(request)

    def generate_quiz(self, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMResult:
        request = self._request("student", "quiz_generation", evidence_packet, context)
        return self.generate(request)

    def build_chat_request(self, *, role: str, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMRequest:
        task_type = {
            "doctor": "doctor_ai_chat",
            "student": "student_ai_chat",
            "patient": "patient_education",
        }[role]
        return self._request(role, task_type, evidence_packet, context)

    def finalize_streamed(
        self,
        request: LLMRequest,
        *,
        content: str,
        provider: str,
        model: str,
        latency_ms: int,
        token_usage: dict[str, int] | None = None,
        interrupted: bool = False,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> LLMResult:
        if request.role == "doctor" and request.task_type == "doctor_ai_chat":
            policy = doctor_direct_response(request.input_text, request.evidence_packet)
            if policy is not None:
                response, reason = policy
                result = self._business_context_policy_result(
                    request,
                    response,
                    reason,
                    provider="doctor_business_context_policy",
                    model="doctor-context-v1",
                    latency_ms=latency_ms,
                )
                if interrupted:
                    result.metadata.degraded = True
                    result.safety_flags.append("generation_interrupted")
                return self._record(request, result)
        if request.role == "patient" and provider == "patient_safety_policy":
            policy = patient_direct_response(request.input_text, request.evidence_packet)
            if policy is not None:
                response, reason = policy
                result = self._patient_policy_result(request, response, reason, latency_ms=latency_ms)
                if interrupted:
                    result.metadata.degraded = True
                    result.safety_flags.append("generation_interrupted")
                return self._record(request, result)
        source_ids = extract_source_ids(request.evidence_packet)
        generated = GeneratedResponse(
            title={"doctor": "医生专业辅助回答", "student": "中医药教学回答", "patient": "中医药科普"}[request.role],
            summary=content.strip() or "回答生成已停止。",
            warnings=["回答由用户停止生成，内容可能不完整。"] if interrupted else [],
            disclaimer={
                "doctor": "仅供医生专业决策辅助，最终结论与处方仍需医生人工确认。",
                "student": "仅用于中医药教学，不作为临床诊疗依据。",
                "patient": "仅供一般中医药科普，不能替代医生诊疗或个体化用药指导。",
            }[request.role],
        ).model_dump(mode="json")
        result = LLMResult(
            request_id=request.request_id,
            success=True,
            provider=provider,
            model=model,
            content=content.strip() or "回答生成已停止。",
            structured_output=generated,
            grounding=LLMGrounding(
                used_evidence_count=evidence_count(request.evidence_packet),
                missing_evidence=evidence_count(request.evidence_packet) == 0 or not source_ids,
                source_ids=source_ids,
            ),
            metadata=LLMMetadata(
                latency_ms=latency_ms,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                degraded=interrupted or fallback_used,
                token_usage=token_usage or {"input_tokens": 0, "output_tokens": 0},
            ),
            safety_flags=["generation_interrupted"] if interrupted else [],
        )
        result = apply_output_safety(request, result)
        return self._record(request, result)

    def generate(self, request: LLMRequest) -> LLMResult:
        settings = get_settings()
        if request.role == "doctor" and request.task_type == "doctor_ai_chat":
            policy = doctor_direct_response(request.input_text, request.evidence_packet)
            if policy is not None:
                response, reason = policy
                return self._record(
                    request,
                    self._business_context_policy_result(
                        request,
                        response,
                        reason,
                        provider="doctor_business_context_policy",
                        model="doctor-context-v1",
                    ),
                )
        if request.role == "patient" and request.task_type != "patient_diet_guidance":
            policy = patient_direct_response(request.input_text, request.evidence_packet)
            if policy is not None:
                response, reason = policy
                return self._record(request, self._patient_policy_result(request, response, reason))
        if not request.evidence_packet:
            result = TemplateProvider().generate(request)
            self._attach_request_metadata(request, result)
            result.metadata.degraded = True
            result.error_code = "EVIDENCE_PACKET_REQUIRED"
            result.safety_flags.append("missing_evidence_packet")
            return self._record(request, result)

        providers = self._provider_chain()
        primary_error: str | None = None
        for index, provider in enumerate(providers):
            try:
                result = provider.generate(request)
                self._attach_request_metadata(request, result)
                result = apply_output_safety(request, result)
                if index > 0 or (provider.name == "template" and settings.llm_provider != "template"):
                    result.metadata.fallback_used = True
                    result.metadata.degraded = True
                    result.metadata.fallback_reason = primary_error or f"{settings.llm_provider}_not_used"
                    result.metadata.provider_unavailable = primary_error is not None
                return self._record(request, result)
            except Exception as exc:
                provider_error = str(exc) or exc.__class__.__name__
                primary_error = f"{provider.name}:{provider_error}"
                if not settings.llm_allow_fallback:
                    result = LLMResult(
                        request_id=request.request_id,
                        success=False,
                        provider=provider.name,
                        model=_model_for_settings(),
                        error_code="LLM_PROVIDER_FAILED",
                        content="LLM provider failed",
                    )
                    self._attach_request_metadata(request, result)
                    result.metadata.degraded = True
                    result.metadata.fallback_reason = primary_error
                    parser_error = _parser_error_marker(provider_error)
                    if parser_error is not None:
                        result.metadata.parse_success = False
                        result.metadata.schema_valid = False
                        result.metadata.parser_error = parser_error
                        result.metadata.provider_unavailable = False
                    else:
                        result.metadata.provider_unavailable = True
                        # No model payload reached the parser.  The boolean
                        # fields cannot express "not applicable", so false is
                        # the only truthful value for this failed generation.
                        result.metadata.parse_success = False
                        result.metadata.schema_valid = False
                    result.safety_flags.append("provider_failed")
                    return self._record(request, result)
                continue

        result = TemplateProvider().generate(request)
        self._attach_request_metadata(request, result)
        result.metadata.fallback_used = True
        result.metadata.degraded = True
        result.metadata.fallback_reason = primary_error
        result.metadata.provider_unavailable = primary_error is not None
        result.error_code = "LLM_PROVIDER_FALLBACK"
        if primary_error:
            result.safety_flags.append("provider_failed")
        return self._record(request, result)

    def _patient_policy_result(
        self,
        request: LLMRequest,
        response: dict[str, Any],
        reason: str,
        *,
        latency_ms: int = 0,
    ) -> LLMResult:
        source_ids = extract_source_ids(request.evidence_packet)
        return LLMResult(
            request_id=request.request_id,
            success=True,
            provider="patient_safety_policy",
            model="patient-safety-v1",
            content=response_display_text(response),
            structured_output=response,
            grounding=LLMGrounding(
                used_evidence_count=evidence_count(request.evidence_packet),
                missing_evidence=evidence_count(request.evidence_packet) == 0 or not source_ids,
                source_ids=source_ids,
            ),
            metadata=LLMMetadata(latency_ms=latency_ms),
            safety_flags=[reason],
        )

    def _business_context_policy_result(
        self,
        request: LLMRequest,
        response: dict[str, Any],
        reason: str,
        *,
        provider: str,
        model: str,
        latency_ms: int = 0,
    ) -> LLMResult:
        source_ids = extract_source_ids(request.evidence_packet)
        return LLMResult(
            request_id=request.request_id,
            success=True,
            provider=provider,
            model=model,
            content=response_display_text(response),
            structured_output=response,
            grounding=LLMGrounding(
                used_evidence_count=evidence_count(request.evidence_packet),
                missing_evidence=False,
                source_ids=source_ids,
            ),
            metadata=LLMMetadata(latency_ms=latency_ms),
            safety_flags=[reason],
        )

    def providers(self) -> list[dict[str, Any]]:
        return self.health()["providers"]

    def health(self) -> dict[str, Any]:
        settings = get_settings()
        provider_health = _provider_healths()
        health_by_name = {item.name: item for item in provider_health}
        configured_provider = settings.llm_provider
        configured_health = health_by_name.get(
            configured_provider,
            ProviderHealth(
                name=configured_provider,
                configured=False,
                reachable=False,
                status="misconfigured",
                last_error=f"unknown_provider:{configured_provider}",
            ),
        )
        fallback_health = health_by_name["template"]

        if not settings.llm_enabled:
            provider_status = "disabled"
            active_provider = "template"
            degraded = False
        elif configured_health.status == "ok":
            provider_status = "ok"
            active_provider = configured_provider
            degraded = False
        elif settings.llm_allow_fallback and fallback_health.reachable:
            provider_status = "fallback"
            active_provider = "template"
            degraded = True
        else:
            provider_status = configured_health.status
            active_provider = configured_provider
            degraded = configured_health.status != "ok"

        active_health = health_by_name.get(active_provider, configured_health)
        return {
            "llm_enabled": settings.llm_enabled,
            "provider": configured_provider,
            "active_provider": active_provider,
            "configured_provider": configured_provider,
            "provider_status": provider_status,
            "configured": configured_health.configured,
            "reachable": configured_health.reachable,
            "degraded": degraded,
            "fallback_available": settings.llm_allow_fallback and fallback_health.reachable,
            "fallback_provider": "template",
            "model": active_health.model,
            "latency_ms": configured_health.latency_ms,
            "last_error": configured_health.last_error,
            "allow_fallback": settings.llm_allow_fallback,
            "generation_parameters": {
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "seed": settings.llm_seed,
                "max_output_tokens": settings.llm_max_output_tokens,
                "max_input_tokens": settings.llm_max_input_tokens,
                "num_ctx": settings.ollama_num_ctx if configured_provider == "ollama" else None,
            },
            "providers": [item.model_dump(mode="json") for item in provider_health],
        }

    def _request(self, role: str, task_type: str, evidence_packet: dict[str, Any], context: dict[str, Any]) -> LLMRequest:
        return LLMRequest(
            role=role,
            task_type=task_type,
            input_text=context.get("input_text", ""),
            evidence_packet=evidence_packet,
            constraints=context.get("constraints", {}),
            trace_id=context.get("trace_id"),
            user_id=context.get("user_id"),
        )

    def _provider_chain(self) -> list[BaseLLMProvider]:
        settings = get_settings()
        if not settings.llm_enabled:
            return [TemplateProvider()]
        providers: list[BaseLLMProvider] = []
        if settings.llm_provider == "ollama":
            providers.append(OllamaProvider())
        elif settings.llm_provider == "openai_compatible":
            providers.append(OpenAICompatibleProvider())
        providers.append(TemplateProvider())
        return providers

    def _record(self, request: LLMRequest, result: LLMResult) -> LLMResult:
        self.db.add(
            LLMGenerationLog(
                request_id=result.request_id,
                trace_id=request.trace_id,
                user_id=request.user_id,
                role=request.role,
                task_type=request.task_type,
                provider=result.provider,
                model=result.model or "",
                success=result.success,
                fallback_used=result.metadata.fallback_used,
                degraded=result.metadata.degraded,
                input_summary_json={
                    "task_type": request.task_type,
                    "visibility": request.evidence_packet.get("visibility"),
                    "packet_id": request.evidence_packet.get("packet_id"),
                },
                output_summary_json={
                    "content": result.content[:240],
                    "structured_keys": sorted(result.structured_output.keys()),
                    "fallback_reason": result.metadata.fallback_reason,
                    "parse_success": result.metadata.parse_success,
                    "schema_valid": result.metadata.schema_valid,
                    "parser_error": result.metadata.parser_error,
                },
                source_ids_json=result.grounding.source_ids,
                safety_flags_json=result.safety_flags,
                latency_ms=result.metadata.latency_ms,
                token_usage_json=result.metadata.token_usage,
                error_code=result.error_code,
            )
        )
        record_event(
            self.db,
            request.trace_id,
            node_name="LLMGenerateNode",
            event_type="llm_generation",
            status="success" if result.success else "failed",
            output_summary={
                "provider": result.provider,
                "model": result.model,
                "fallback_used": result.metadata.fallback_used,
                "degraded": result.metadata.degraded,
                "fallback_reason": result.metadata.fallback_reason,
                "parse_success": result.metadata.parse_success,
                "schema_valid": result.metadata.schema_valid,
                "safety_flags": result.safety_flags,
            },
            evidence_packet_id=request.evidence_packet.get("packet_id"),
            metadata={
                "provider": result.provider,
                "model": result.model,
                "fallback_used": result.metadata.fallback_used,
                "degraded": result.metadata.degraded,
                "fallback_reason": result.metadata.fallback_reason,
                "parse_success": result.metadata.parse_success,
                "schema_valid": result.metadata.schema_valid,
                "parser_error": result.metadata.parser_error,
                "source_ids": result.grounding.source_ids,
                "safety_flags": result.safety_flags,
            },
        )
        self.db.flush()
        return result

    def _attach_request_metadata(self, request: LLMRequest, result: LLMResult) -> None:
        experiment_id = request.constraints.get("experiment_id") if isinstance(request.constraints, dict) else None
        if isinstance(experiment_id, str) and experiment_id:
            result.metadata.experiment_id = experiment_id


def _model_for_settings() -> str:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return settings.ollama_model
    if settings.llm_provider == "openai_compatible":
        return settings.openai_compatible_model
    return "guardtcm-template-v1"


def _parser_error_marker(provider_error: str) -> str | None:
    """Keep a content-free parser failure class in failed LLM metadata."""

    prefix = "parser_error:"
    if not provider_error.startswith(prefix):
        return None
    detail = provider_error[len(prefix) :]
    marker = detail.split(":", 1)[0].strip()
    return marker or "invalid_output"


def _provider_healths() -> list[ProviderHealth]:
    providers = [TemplateProvider(), OllamaProvider(), OpenAICompatibleProvider()]
    healths: list[ProviderHealth] = []
    for provider in providers:
        try:
            healths.append(provider.health())
        except Exception as exc:
            healths.append(
                ProviderHealth(
                    name=provider.name,
                    configured=False,
                    reachable=False,
                    status="unavailable",
                    last_error=str(exc) or exc.__class__.__name__,
                )
            )
    return healths
