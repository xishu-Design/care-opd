from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


EvaluationSuite = Literal["quick", "full"]


class EvaluationCase(BaseModel):
    case_id: str
    dataset: str
    category: str
    metric_name: str
    name: str
    severity: str = "high"
    input: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class EvaluationCaseResult(BaseModel):
    case_id: str
    dataset: str
    category: str
    metric_name: str
    name: str
    severity: str
    passed: bool
    actual: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    evidence_packet_id: str | None = None
    failure_reason: str | None = None
    duration_ms: int = 0


class EvaluationRunRequest(BaseModel):
    suite: EvaluationSuite = "full"
    reset_demo_data: bool = True


class AblationRunRequest(BaseModel):
    suite: EvaluationSuite = "quick"
    baseline_run_id: str | None = None


class EvaluationContext(BaseModel):
    suite: EvaluationSuite = "full"
    disabled: set[str] = Field(default_factory=set)
    experiment_name: str = "baseline"
