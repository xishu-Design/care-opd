from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy import JSON as SAJSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import PrescriptionStatus
from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    doctor_profile: Mapped["DoctorProfile | None"] = relationship(back_populates="user", cascade="all, delete-orphan")
    patient_profile: Mapped["PatientProfile | None"] = relationship(back_populates="user", cascade="all, delete-orphan")
    student_profile: Mapped["StudentProfile | None"] = relationship(back_populates="user", cascade="all, delete-orphan")


class DoctorProfile(Base):
    __tablename__ = "doctor_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    institution: Mapped[str | None] = mapped_column(String(160))
    department: Mapped[str | None] = mapped_column(String(120))
    title: Mapped[str | None] = mapped_column(String(120))
    doctor_code: Mapped[str | None] = mapped_column(String(80))
    specialty: Mapped[str | None] = mapped_column(String(240))

    user: Mapped[User] = relationship(back_populates="doctor_profile")


class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    age: Mapped[int | None] = mapped_column(Integer)
    sex: Mapped[str | None] = mapped_column(String(32))
    height_cm: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    body_state: Mapped[str | None] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="patient_profile")


class StudentProfile(Base):
    __tablename__ = "student_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    school: Mapped[str | None] = mapped_column(String(160))
    major: Mapped[str | None] = mapped_column(String(120))
    grade: Mapped[str | None] = mapped_column(String(80))
    learning_direction: Mapped[str | None] = mapped_column(String(160))

    user: Mapped[User] = relationship(back_populates="student_profile")


class DoctorPatientBinding(Base):
    __tablename__ = "doctor_patient_bindings"
    __table_args__ = (UniqueConstraint("doctor_id", "patient_id", name="uq_doctor_patient"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class PatientHealthRecord(Base):
    __tablename__ = "patient_health_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    current_symptoms: Mapped[str | None] = mapped_column(Text)
    body_state: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MedicalHistory(Base):
    __tablename__ = "medical_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AllergyRecord(Base):
    __tablename__ = "allergy_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Prescription(Base):
    __tablename__ = "prescriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(160), default="辅助拟方草稿", nullable=False)
    symptom_context: Mapped[str | None] = mapped_column(Text)
    patient_age_at_visit: Mapped[int | None] = mapped_column(Integer)
    patient_sex_at_visit: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(64), default=PrescriptionStatus.DRAFT.value, index=True, nullable=False)
    usage_method: Mapped[str | None] = mapped_column(Text)
    doctor_notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    items: Mapped[list["PrescriptionItem"]] = relationship(back_populates="prescription", cascade="all, delete-orphan")


class PrescriptionItem(Base):
    __tablename__ = "prescription_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prescription_id: Mapped[int] = mapped_column(ForeignKey("prescriptions.id"), index=True, nullable=False)
    herb_name: Mapped[str] = mapped_column(String(120), nullable=False)
    dose: Mapped[str | None] = mapped_column(String(80))
    unit: Mapped[str | None] = mapped_column(String(40))
    notes: Mapped[str | None] = mapped_column(Text)

    prescription: Mapped[Prescription] = relationship(back_populates="items")


class PrescriptionStatusLog(Base):
    __tablename__ = "prescription_status_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prescription_id: Mapped[int] = mapped_column(ForeignKey("prescriptions.id"), index=True, nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(64))
    to_status: Mapped[str] = mapped_column(String(64), nullable=False)
    changed_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    changed_by_role: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class PrescriptionReview(Base):
    __tablename__ = "prescription_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prescription_id: Mapped[int] = mapped_column(ForeignKey("prescriptions.id"), index=True, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(40), nullable=False)
    review_summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_packet: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RiskCard(Base):
    __tablename__ = "risk_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("prescription_reviews.id"), index=True, nullable=False)
    risk_type: Mapped[str] = mapped_column(String(80), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(40), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)


class PatientInstruction(Base):
    __tablename__ = "patient_instructions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prescription_id: Mapped[int] = mapped_column(ForeignKey("prescriptions.id"), unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    safety_statement: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class PatientEducationChat(Base):
    __tablename__ = "patient_education_chats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(80), index=True)
    evidence_packet: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    generated_response: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    llm_result: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RoleAIChat(Base):
    __tablename__ = "role_ai_chats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(80), index=True)
    evidence_packet: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    generated_response: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    llm_result: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True, nullable=False)


class DietPrecaution(Base):
    __tablename__ = "diet_precautions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prescription_id: Mapped[int] = mapped_column(ForeignKey("prescriptions.id"), unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class LearningCase(Base):
    __tablename__ = "learning_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_prescription_id: Mapped[int | None] = mapped_column(ForeignKey("prescriptions.id"), index=True)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    anonymized_case: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    standard_analysis: Mapped[str] = mapped_column(Text, nullable=False)
    risk_points: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    evidence_chain: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Quiz(Base):
    __tablename__ = "quizzes"
    __table_args__ = (UniqueConstraint("quiz_id", name="uq_quiz_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    quiz_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    case_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    question_type: Mapped[str] = mapped_column(String(80), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    options_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    knowledge_points_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    source_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class DeidentificationLog(Base):
    __tablename__ = "deidentification_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prescription_id: Mapped[int] = mapped_column(ForeignKey("prescriptions.id"), nullable=False)
    learning_case_id: Mapped[int] = mapped_column(ForeignKey("learning_cases.id"), nullable=False)
    removed_fields: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    checker_status: Mapped[str] = mapped_column(String(80), default="auto_checked", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class StudentAnswer(Base):
    __tablename__ = "student_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    learning_case_id: Mapped[int] = mapped_column(ForeignKey("learning_cases.id"), index=True, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    has_risk: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    risk_level: Mapped[str | None] = mapped_column(String(40))
    feedback: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MCPToolCall(Base):
    __tablename__ = "mcp_tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(80), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(80), index=True)
    tool_name: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    risk_level: Mapped[str | None] = mapped_column(String(40))
    requires_confirmation: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reason: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    arguments_summary: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    result_summary: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(120))
    blocked_by: Mapped[str | None] = mapped_column(String(120))
    backend: Mapped[str | None] = mapped_column(String(80))
    degraded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fallback_reason: Mapped[str | None] = mapped_column(String(160))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SafetyLog(Base):
    __tablename__ = "safety_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str] = mapped_column(String(160), nullable=False)
    safe_reply: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class HarnessRun(Base):
    __tablename__ = "harness_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    phase: Mapped[str] = mapped_column(String(80), default="phase_1_backend_foundation", nullable=False)
    suite: Mapped[str] = mapped_column(String(40), default="full", nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="completed", nullable=False)
    total_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    passed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    categories: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    markdown_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class HarnessResult(Base):
    __tablename__ = "harness_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("harness_runs.id"), index=True, nullable=False)
    case_id: Mapped[str | None] = mapped_column(String(120), index=True)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    severity: Mapped[str | None] = mapped_column(String(40))
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    actual_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    expected_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(80), index=True)
    evidence_packet_id: Mapped[str | None] = mapped_column(String(80), index=True)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class EvidencePacketLog(Base):
    __tablename__ = "evidence_packets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    packet_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(80), index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    task_type: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    visibility: Mapped[str] = mapped_column(String(80), nullable=False)
    case_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    retrieved_evidence_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    graph_reasoning_path_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    graph_node_path_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    rule_findings_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    memory_used_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    tool_calls_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    backend_summary_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    degraded_reasons_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    safety_decision_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    human_confirm_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class PendingHumanAction(Base):
    __tablename__ = "pending_human_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    trace_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(160), index=True, nullable=False)
    graph_name: Mapped[str] = mapped_column(String(120), nullable=False)
    node_name: Mapped[str] = mapped_column(String(120), nullable=False)
    prescription_id: Mapped[int] = mapped_column(ForeignKey("prescriptions.id"), index=True, nullable=False)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    evidence_packet_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="pending", index=True, nullable=False)
    risk_level: Mapped[str | None] = mapped_column(String(40))
    review_summary_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    interrupt_payload_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    resume_payload_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    resume_comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    resumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LLMGenerationLog(Base):
    __tablename__ = "llm_generation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(80), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    task_type: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    model: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    input_summary_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    output_summary_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    source_ids_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    safety_flags_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    token_usage_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class LLMExperimentRun(Base):
    __tablename__ = "llm_experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    provider: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    model: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    suite: Mapped[str] = mapped_column(String(40), default="quick", nullable=False)
    total_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    fallback_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parse_success_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    schema_valid_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    safety_pass_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    suite: Mapped[str] = mapped_column(String(40), default="full", nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(40), default="phase12-v1", nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="completed", index=True, nullable=False)
    total_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    passed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    safety_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    evidence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    agent_trace_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    llm_safety_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    markdown_report: Mapped[str] = mapped_column(Text, default="", nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"), index=True, nullable=False)
    case_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    dataset: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    category: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String(40), default="high", nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    actual_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    expected_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(80), index=True)
    evidence_packet_id: Mapped[str | None] = mapped_column(String(80), index=True)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AblationRun(Base):
    __tablename__ = "ablation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ablation_id: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(40), default="completed", index=True, nullable=False)
    baseline_run_id: Mapped[str | None] = mapped_column(String(120), index=True)
    experiment_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MockMemory(Base):
    __tablename__ = "mock_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    owner_role: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(80), nullable=False)
    content_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    source: Mapped[str] = mapped_column(String(80), default="mock_mem0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MockGraphPath(Base):
    __tablename__ = "mock_graph_paths"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    prescription_id: Mapped[int | None] = mapped_column(ForeignKey("prescriptions.id"), index=True)
    learning_case_id: Mapped[int | None] = mapped_column(ForeignKey("learning_cases.id"), index=True)
    path_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    visibility: Mapped[str] = mapped_column(String(80), default="doctor_internal", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class LightRAGRetrievalLog(Base):
    __tablename__ = "lightrag_retrieval_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    query: Mapped[str] = mapped_column(Text, default="", nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    task_type: Mapped[str] = mapped_column(String(80), nullable=False)
    evidence_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class KnowledgeSnapshot(Base):
    __tablename__ = "knowledge_snapshots"
    __table_args__ = (UniqueConstraint("snapshot_id", name="uq_knowledge_snapshot_id"), UniqueConstraint("version", name="uq_knowledge_snapshot_version"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    version: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(40), index=True, default="STAGING", nullable=False)
    source_manifest_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    record_counts_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    lightrag_workspace: Mapped[str] = mapped_column(Text, default="", nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(120), default="bge-m3", nullable=False)
    llm_model: Mapped[str] = mapped_column(String(120), default="qwen2.5:7b", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str] = mapped_column(Text, default="", nullable=False)


class LightRAGIndexJob(Base):
    __tablename__ = "lightrag_index_jobs"
    __table_args__ = (UniqueConstraint("run_id", name="uq_lightrag_index_job_run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(40), index=True, default="queued", nullable=False)
    stage: Mapped[str] = mapped_column(String(80), default="queued", nullable=False)
    track_ids_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    total_documents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_documents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_documents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)


class LightRAGDocManifest(Base):
    __tablename__ = "lightrag_doc_manifest"
    __table_args__ = (UniqueConstraint("snapshot_id", "doc_id", name="uq_lightrag_doc_manifest_snapshot_doc"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(160), index=True, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, default="", nullable=False)
    entry_ids_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    source_ids_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    visibility_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    license_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"
    __table_args__ = (UniqueConstraint("source_id", name="uq_knowledge_source_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    source_title: Mapped[str] = mapped_column(String(240), nullable=False)
    source_type: Mapped[str] = mapped_column(String(80), nullable=False)
    source_version: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    source_year: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    evidence_level: Mapped[str] = mapped_column(String(80), default="demo", nullable=False)
    license_note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"
    __table_args__ = (UniqueConstraint("source_id", "entry_id", name="uq_knowledge_source_entry"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entry_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    entry_type: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    keywords_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    source_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    source_title: Mapped[str] = mapped_column(String(240), default="", nullable=False)
    source_type: Mapped[str] = mapped_column(String(80), default="local_sample", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class KnowledgeClaim(Base):
    __tablename__ = "knowledge_claims"
    __table_args__ = (UniqueConstraint("claim_id", name="uq_knowledge_claim_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[str] = mapped_column(String(160), index=True, nullable=False)
    subject_label: Mapped[str] = mapped_column(String(240), index=True, nullable=False)
    subject_type: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    predicate: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    object_label: Mapped[str] = mapped_column(String(240), index=True, nullable=False)
    source_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(160), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    source_excerpt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    license_status: Mapped[str] = mapped_column(String(40), default="verified", nullable=False)
    distribution_scope: Mapped[str] = mapped_column(String(40), default="internal_only", nullable=False)
    content_risk_level: Mapped[str] = mapped_column(String(40), default="non_clinical", nullable=False)
    review_status: Mapped[str] = mapped_column(String(40), default="imported", nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(80), default="rule", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class KnowledgeIngestionRun(Base):
    __tablename__ = "knowledge_ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(80), default="running", nullable=False)
    sources_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    entries_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ContentImportRun(Base):
    __tablename__ = "content_import_runs"
    __table_args__ = (UniqueConstraint("run_id", name="uq_content_import_run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(80), default="running", nullable=False)
    total_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DemoResetLog(Base):
    __tablename__ = "demo_reset_logs"
    __table_args__ = (UniqueConstraint("reset_id", name="uq_demo_reset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reset_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    reset_scope: Mapped[str] = mapped_column(String(80), default="demo_only", nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="success", nullable=False)
    created_summary_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    demo_ids_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RuleSet(Base):
    __tablename__ = "rule_sets"
    __table_args__ = (UniqueConstraint("name", name="uq_rule_set_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(80), default="0.3.0", nullable=False)
    source: Mapped[str] = mapped_column(String(120), default="local_rules", nullable=False)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RuleDefinition(Base):
    __tablename__ = "rule_definitions"
    __table_args__ = (UniqueConstraint("rule_id", name="uq_rule_definition_rule_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_set_id: Mapped[int | None] = mapped_column(ForeignKey("rule_sets.id"), index=True)
    rule_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    rule_type: Mapped[str] = mapped_column(String(80), nullable=False)
    rule_category: Mapped[str] = mapped_column(String(80), default="medical_review", nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(40), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(120), index=True)
    source_record_id: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    review_status: Mapped[str] = mapped_column(String(80), default="manual_review_required", nullable=False)
    applies_to_roles_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    condition_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    evidence_query: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    requires_doctor_confirmation: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RuleExecutionLog(Base):
    __tablename__ = "rule_execution_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    prescription_id: Mapped[int | None] = mapped_column(ForeignKey("prescriptions.id"), index=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    triggered_rules_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(40), default="green", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class GraphContextNode(Base):
    __tablename__ = "graph_context_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    node_type: Mapped[str] = mapped_column(String(80), nullable=False)
    label: Mapped[str] = mapped_column(String(240), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    visibility: Mapped[str] = mapped_column(String(80), default="doctor_internal", nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class GraphContextEdge(Base):
    __tablename__ = "graph_context_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    from_node_id: Mapped[int | None] = mapped_column(ForeignKey("graph_context_nodes.id"), index=True)
    to_node_id: Mapped[int | None] = mapped_column(ForeignKey("graph_context_nodes.id"), index=True)
    relation: Mapped[str] = mapped_column(String(120), nullable=False)
    visibility: Mapped[str] = mapped_column(String(80), default="doctor_internal", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class GraphPath(Base):
    __tablename__ = "graph_paths"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    prescription_id: Mapped[int | None] = mapped_column(ForeignKey("prescriptions.id"), index=True)
    learning_case_id: Mapped[int | None] = mapped_column(ForeignKey("learning_cases.id"), index=True)
    visibility: Mapped[str] = mapped_column(String(80), default="doctor_internal", nullable=False)
    path_json: Mapped[list[Any]] = mapped_column(SAJSON, default=list, nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RoleMemory(Base):
    __tablename__ = "role_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    owner_role: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(80), nullable=False)
    content_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    source: Mapped[str] = mapped_column(String(80), default="persistent_memory", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MemoryAccessLog(Base):
    __tablename__ = "memory_access_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    requester_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    requester_role: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    owner_role: Mapped[str | None] = mapped_column(String(32))
    tool_name: Mapped[str] = mapped_column(String(120), nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AdapterHealthCheck(Base):
    __tablename__ = "adapter_health_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    adapter_name: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    backend: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(80), nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    task_type: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    request_path: Mapped[str] = mapped_column(String(240), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="running", nullable=False)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)


class AgentTraceEvent(Base):
    __tablename__ = "agent_trace_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    span_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    parent_span_id: Mapped[str | None] = mapped_column(String(80), index=True)
    node_name: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="success", nullable=False)
    input_summary: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    output_summary: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(120), index=True)
    evidence_packet_id: Mapped[str | None] = mapped_column(String(80), index=True)
    safety_decision: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(120))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
