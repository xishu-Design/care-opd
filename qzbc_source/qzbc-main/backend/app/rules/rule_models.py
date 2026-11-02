from typing import Any

from pydantic import BaseModel, Field


class StructuredRule(BaseModel):
    rule_id: str
    rule_type: str
    rule_category: str = "medical_review"
    name: str
    enabled: bool = True
    risk_level: str
    source_id: str | None = None
    source_record_id: str = ""
    review_status: str = "manual_review_required"
    applies_to_roles: list[str] = Field(default_factory=lambda: ["doctor"])
    condition: dict[str, Any]
    message: str
    evidence_query: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    requires_doctor_confirmation: bool = True


class RuleFinding(BaseModel):
    rule_id: str
    rule_type: str
    rule_category: str = "medical_review"
    risk_level: str
    trigger: str
    message: str
    evidence_query: str
    source_id: str | None = None
    source_record_id: str = ""
    review_status: str = ""
    requires_doctor_confirmation: bool = True
    matched_herbs: list[str] = Field(default_factory=list)
    matched_pairs: list[dict[str, str]] = Field(default_factory=list)
