from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


RoleName = Literal["doctor", "patient", "student"]
LLMTaskType = Literal[
    "doctor_review_text",
    "doctor_ai_chat",
    "patient_education",
    "patient_diet_guidance",
    "student_analysis",
    "student_quiz_explanation",
    "student_ai_chat",
    "quiz_generation",
]
ProviderStatus = Literal["ok", "unavailable", "degraded", "fallback", "disabled", "misconfigured"]


class GeneratedSection(BaseModel):
    heading: str = ""
    content: str = ""
    source_ids: list[str] = Field(default_factory=list)


class GeneratedResponse(BaseModel):
    title: str = ""
    summary: str = ""
    sections: list[GeneratedSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    disclaimer: str = ""


class LLMRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"llm_{uuid4().hex}")
    role: RoleName
    task_type: LLMTaskType
    input_text: str = ""
    evidence_packet: dict[str, Any]
    constraints: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    user_id: int | None = None


class LLMGrounding(BaseModel):
    used_evidence_count: int = 0
    missing_evidence: bool = False
    source_ids: list[str] = Field(default_factory=list)


class LLMMetadata(BaseModel):
    latency_ms: int = 0
    fallback_used: bool = False
    fallback_reason: str | None = None
    degraded: bool = False
    provider_unavailable: bool = False
    parse_success: bool = True
    schema_valid: bool = True
    parser_error: str | None = None
    experiment_id: str | None = None
    token_usage: dict[str, int] = Field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


class LLMResult(BaseModel):
    request_id: str
    success: bool = True
    provider: str = "template"
    model: str = ""
    content: str = ""
    structured_output: dict[str, Any] = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    grounding: LLMGrounding = Field(default_factory=LLMGrounding)
    metadata: LLMMetadata = Field(default_factory=LLMMetadata)
    error_code: str | None = None


class ProviderHealth(BaseModel):
    name: str
    configured: bool = False
    reachable: bool = False
    status: ProviderStatus = "unavailable"
    latency_ms: int | None = None
    model: str | None = None
    last_error: str | None = None


class ParserResult(BaseModel):
    success: bool
    structured_output: dict[str, Any] | None = None
    raw_text: str = ""
    parse_error: str | None = None
    schema_valid: bool = False


def generated_to_dict(response: GeneratedResponse) -> dict[str, Any]:
    return response.model_dump(mode="json")
