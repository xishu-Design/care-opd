from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PHASE11 = "phase_11_demo_readiness"
VERSION11 = "1.1.0"

DEMO_DOCTOR_USERNAME = "doctor_demo"
DEMO_PATIENT_USERNAME = "patient_demo"
DEMO_STUDENT_USERNAME = "student_demo"
DEMO_PASSWORD = "password123"
DEMO_TAG = "phase11_default_demo"


class DemoResetRequest(BaseModel):
    reset_scope: Literal["demo_only"] = "demo_only"
    reseed: bool = True


class DemoCheck(BaseModel):
    name: str
    passed: bool
    severity: Literal["critical", "high", "medium", "low"] = "critical"
    details: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DemoSmokeResult(BaseModel):
    smoke_id: str
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    checks: list[DemoCheck]
    warnings: list[str] = Field(default_factory=list)

