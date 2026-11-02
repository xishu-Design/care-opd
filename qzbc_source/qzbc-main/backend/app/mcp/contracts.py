from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def new_request_id() -> str:
    return str(uuid4())


class ResourceScope(BaseModel):
    is_doctor_bound_to_patient: bool | None = None
    is_patient_self: bool | None = None
    is_learning_case_deidentified: bool | None = None


class ToolContext(BaseModel):
    user_id: int
    role: str
    session_id: str = Field(default_factory=new_request_id)
    trace_id: str | None = None
    task_type: str = "generic"
    patient_id: int | None = None
    prescription_id: int | None = None
    learning_case_id: int | None = None
    resource_scope: ResourceScope = Field(default_factory=ResourceScope)


class ToolRequest(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    context: ToolContext


class ToolResult(BaseModel):
    request_id: str
    tool_name: str
    allowed: bool
    success: bool
    data: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    risk_level: str = "low"
    requires_confirmation: bool = False
    error_code: str | None = None
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def result(self) -> dict[str, Any]:
        return self.data or {}


class ToolSpec(BaseModel):
    name: str
    description: str
    server: str
    allowed_roles: list[str]
    allowed_task_types: list[str]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "low"
    requires_confirmation: bool = False
    audit_required: bool = True
    timeout_ms: int = 3000
    deprecated: bool = False
    remove_after_phase: int | None = None


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
