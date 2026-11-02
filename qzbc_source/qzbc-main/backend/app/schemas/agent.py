from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentGraphInfo(BaseModel):
    name: str
    role: str
    task_type: str
    version: str
    nodes: list[str]


class DoctorReviewAgentRequest(BaseModel):
    prescription_id: int


class PatientEducationAgentRequest(BaseModel):
    message: str


class PrivateEvalEvidenceControlRequest(BaseModel):
    """Opaque control plane for the loopback-only PRIVATE evaluation backend."""

    protocol: Literal["three_role_paired_fact_v1"]
    condition: Literal["clean", "evidence_unavailable_counterfactual"]
    task_token: str = Field(pattern=r"^T-[A-Za-z0-9_-]{8,80}$")
    case_material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_material: str = Field(min_length=1, max_length=2000)


class RoleAIChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    private_eval_evidence_control: PrivateEvalEvidenceControlRequest | None = None


class DoctorCaseReasoningRequest(BaseModel):
    """Audited input contract for an anonymized, non-clinical teaching case.

    The case material is first-party request evidence.  It is deliberately
    separate from retrieved knowledge and may never contain a reference answer.
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=5000)
    protocol: Literal["public_anonymized_tcm_case_reasoning_v1"]
    task_token: str = Field(pattern=r"^T-[A-Za-z0-9_-]{8,80}$")
    source_kind: Literal["public_anonymized_teaching_case"]
    case_material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_material: str = Field(min_length=1, max_length=3000)
    inference_mode: Literal["uncertain_teaching_hypothesis"]
    reference_answer_included: Literal[False] = False
    clinical_use_authorized: Literal[False] = False


class StudentTrainingAgentRequest(BaseModel):
    learning_case_id: int


class DoctorReviewResumeRequest(BaseModel):
    approved: bool
    comment: str | None = None


class PendingHumanActionResponse(BaseModel):
    action_id: str
    trace_id: str
    thread_id: str
    graph_name: str
    node_name: str
    prescription_id: int
    patient_id: int
    doctor_id: int
    evidence_packet_id: str
    status: str
    risk_level: str | None = None
    review_summary: dict
    interrupt_payload: dict
    resume_payload: dict
    resume_comment: str | None = None
