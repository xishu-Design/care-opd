from typing import Any, Literal

from pydantic import BaseModel, Field


PHASE4 = "phase_4_harness_competition_enhancement"
PHASE5 = "phase_5_agent_orchestration"
PHASE6 = "phase_6_human_in_loop_resume"
PHASE8 = "phase_8_llm_grounded_generation"
PHASE9 = "phase_9_open_source_data_pipeline"
PHASE10 = "phase_10_lightrag_graphrag_integration"
PHASE11 = "phase_11_demo_readiness"
PHASE12 = "phase_12_system_evaluation"


class HarnessCaseAction(BaseModel):
    runner: Literal["api_runner", "service_runner"]
    method: str | None = None
    path: str | None = None
    body: dict[str, Any] = Field(default_factory=dict)
    service: str | None = None


class HarnessCaseSetup(BaseModel):
    seed: str = "phase4_demo_seed"
    preconditions: list[str] = Field(default_factory=list)


class HarnessCaseCleanup(BaseModel):
    reset_seed: bool = False


class HarnessCaseExpected(BaseModel):
    status_code: int | None = None
    safety_allowed: bool | None = None
    should_call_tools: list[str] = Field(default_factory=list)
    should_not_call_tools: list[str] = Field(default_factory=list)
    response_contains: list[str] = Field(default_factory=list)
    response_not_contains: list[str] = Field(default_factory=list)
    degraded: bool | None = None
    fallback_used: bool | None = None
    has_source_id: bool | None = None
    trace_nodes: list[str] = Field(default_factory=list)


class HarnessCase(BaseModel):
    case_id: str
    name: str
    phase: str = PHASE4
    category: str
    severity: Literal["critical", "high", "medium", "low"]
    role: Literal["doctor", "patient", "student", "system"]
    tags: list[str] = Field(default_factory=list)
    setup: HarnessCaseSetup = Field(default_factory=HarnessCaseSetup)
    action: HarnessCaseAction
    expected: HarnessCaseExpected = Field(default_factory=HarnessCaseExpected)
    assertions: list[str]
    cleanup: HarnessCaseCleanup = Field(default_factory=HarnessCaseCleanup)


class RunnerResult(BaseModel):
    case_id: str
    category: str
    passed: bool
    severity: str
    actual: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    details: str = ""
    trace_id: str | None = None
    related_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    related_evidence_packet_id: str | None = None
    failure_reason: str | None = None
    duration_ms: int = 0
