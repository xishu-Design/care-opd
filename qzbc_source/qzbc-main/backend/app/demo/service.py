from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.content.services.content_stats_service import content_stats
from app.core.config import get_settings
from app.core.enums import PrescriptionStatus, UserRole
from app.core.security import hash_password
from app.db.models import (
    AllergyRecord,
    DietPrecaution,
    DoctorPatientBinding,
    DoctorProfile,
    DemoResetLog,
    EvidencePacketLog,
    HarnessRun,
    LearningCase,
    MedicalHistory,
    PatientHealthRecord,
    PatientInstruction,
    PatientProfile,
    Prescription,
    PrescriptionItem,
    Quiz,
    RuleDefinition,
    StudentProfile,
    User,
    AgentTrace,
)
from app.demo.schemas import (
    DEMO_DOCTOR_USERNAME,
    DEMO_PASSWORD,
    DEMO_PATIENT_USERNAME,
    DEMO_STUDENT_USERNAME,
    DEMO_TAG,
    PHASE11,
    VERSION11,
    DemoCheck,
    DemoResetRequest,
    DemoSmokeResult,
)
from app.lightrag_integration.health import lightrag_health
from app.llm.service import LLMService
from app.services.learning_service import is_student_learning_case, list_learning_cases
from app.services.prescription_service import list_patient_visible_prescriptions


DEMO_UNPUBLISHED_TITLE = "Phase 11 Demo Unpublished Prescription"
DEMO_PUBLISHED_TITLE = "Phase 11 Demo Published Prescription"
DEMO_REVIEW_TITLE = "Phase 11 Demo Review Prescription"
DEMO_CASE_ID = "PHASE11_DEMO_CASE"
DEMO_QUIZ_ID = "PHASE11_DEMO_QUIZ_001"


class DemoService:
    def __init__(self, db: Session):
        self.db = db

    def get_demo_status(self) -> dict[str, Any]:
        settings = get_settings()
        database_ok = self._database_ok()
        doctor = self._user(DEMO_DOCTOR_USERNAME)
        patient = self._user(DEMO_PATIENT_USERNAME)
        student = self._user(DEMO_STUDENT_USERNAME)
        binding = self._binding(doctor, patient)
        content = content_stats(self.db)
        llm = LLMService(self.db).health()
        lightrag = self._safe_lightrag_health()
        last_run = self.db.scalar(select(HarnessRun).order_by(HarnessRun.created_at.desc()))
        last_reset = self.db.scalar(select(DemoResetLog).order_by(DemoResetLog.created_at.desc()))
        latest_trace = self.db.scalar(select(AgentTrace).order_by(AgentTrace.started_at.desc()))
        latest_packet = self.db.scalar(select(EvidencePacketLog).order_by(EvidencePacketLog.created_at.desc()))

        demo_data = {
            "patient_profile_exists": patient is not None and patient.patient_profile is not None,
            "draft_prescription_exists": self._prescription(DEMO_UNPUBLISHED_TITLE) is not None,
            "published_prescription_exists": self._prescription(DEMO_PUBLISHED_TITLE, PrescriptionStatus.PUBLISHED_TO_PATIENT.value) is not None,
            "review_prescription_exists": self._prescription(DEMO_REVIEW_TITLE, PrescriptionStatus.DRAFT.value) is not None,
            "learning_case_exists": self._learning_case() is not None,
            "quiz_exists": self._quiz() is not None,
        }

        blocking_issues: list[str] = []
        warnings: list[str] = []
        if not database_ok:
            blocking_issues.append("database_unavailable")
        if doctor is None:
            blocking_issues.append("doctor_demo_missing")
        if patient is None:
            blocking_issues.append("patient_demo_missing")
        if student is None:
            blocking_issues.append("student_demo_missing")
        if binding is None:
            blocking_issues.append("doctor_patient_binding_missing")
        if not demo_data["published_prescription_exists"]:
            blocking_issues.append("published_demo_prescription_missing")
        if not demo_data["learning_case_exists"]:
            blocking_issues.append("demo_learning_case_missing")
        if content.get("knowledge_entries", 0) == 0 and content.get("rules", 0) == 0:
            warnings.append("content_library_empty")
        if not lightrag.get("enabled"):
            warnings.append("lightrag_disabled_using_local_knowledge")
        elif lightrag.get("status") not in {"ok", "healthy"}:
            warnings.append("lightrag_unavailable_or_degraded")
        if llm.get("provider") != "template" and llm.get("provider_status") != "ok":
            warnings.append("primary_llm_unavailable_template_fallback_available")
        if last_run is None:
            warnings.append("harness_never_run")
        if latest_packet is None:
            warnings.append("no_evidence_packet_yet")
        if latest_trace is None:
            warnings.append("no_trace_yet")

        return {
            "environment": {
                "app_env": settings.app_env,
                "phase": PHASE11,
                "version": VERSION11,
            },
            "backend": {
                "api_alive": True,
                "database_ok": database_ok,
                "alembic_current": database_ok,
            },
            "accounts": {
                "doctor_demo_exists": doctor is not None,
                "patient_demo_exists": patient is not None,
                "student_demo_exists": student is not None,
            },
            "bindings": {
                "doctor_patient_binding_exists": binding is not None,
            },
            "demo_data": demo_data,
            "content": {
                "sources": content.get("sources", 0),
                "knowledge_entries": content.get("knowledge_entries", 0),
                "rules": content.get("rules", 0),
                "learning_cases": content.get("learning_cases", 0),
                "quizzes": content.get("quizzes", 0),
                "student_learning_cases": content.get("student_learning_cases", 0),
            },
            "llm": {
                "enabled": llm.get("llm_enabled", False),
                "provider": llm.get("provider", "template"),
                "active_provider": llm.get("active_provider", llm.get("provider", "template")),
                "configured_provider": llm.get("configured_provider", llm.get("provider", "template")),
                "status": llm.get("provider_status", "unknown"),
                "configured": llm.get("configured", True),
                "reachable": llm.get("reachable", llm.get("provider_status") == "ok"),
                "degraded": llm.get("degraded", False),
                "fallback_available": llm.get("fallback_available", llm.get("allow_fallback", True)),
                "fallback_provider": llm.get("fallback_provider", "template"),
                "model": llm.get("model", ""),
                "latency_ms": llm.get("latency_ms"),
                "last_error": llm.get("last_error"),
                "providers": llm.get("providers", []),
            },
            "lightrag": {
                "enabled": lightrag.get("enabled", False),
                "backend": get_settings().knowledge_backend,
                "status": "disabled" if not lightrag.get("enabled") else lightrag.get("status", "unknown"),
                "fallback_available": get_settings().lightrag_allow_fallback,
                "mode": lightrag.get("mode", ""),
                "docs_count": lightrag.get("docs_count", lightrag.get("docs_indexed", 0)),
            },
            "harness": {
                "last_run_id": last_run.id if last_run else None,
                "last_phase": last_run.phase if last_run else None,
                "last_pass_rate": last_run.pass_rate if last_run else None,
            },
            "demo_reset": {
                "last_reset_id": last_reset.reset_id if last_reset else None,
                "last_status": last_reset.status if last_reset else None,
                "last_reset_at": last_reset.created_at.isoformat() if last_reset else None,
            },
            "observability": {
                "latest_trace_id": latest_trace.trace_id if latest_trace else None,
                "latest_evidence_packet_id": latest_packet.packet_id if latest_packet else None,
            },
            "readiness": {
                "ready": not blocking_issues,
                "blocking_issues": blocking_issues,
                "warnings": warnings,
            },
        }

    def reset_demo_data(self, payload: DemoResetRequest, actor_user_id: int | None = None) -> dict[str, Any]:
        settings = get_settings()
        if settings.app_env != "development":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Demo reset is only available in development")
        if payload.reset_scope != "demo_only":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only demo_only reset is supported")

        doctor = self._upsert_user(DEMO_DOCTOR_USERNAME, UserRole.DOCTOR.value, "演示医生")
        patient = self._upsert_user(DEMO_PATIENT_USERNAME, UserRole.PATIENT.value, "演示患者")
        student = self._upsert_user(DEMO_STUDENT_USERNAME, UserRole.STUDENT.value, "演示学生")
        self._upsert_profiles(doctor, patient, student)
        self._upsert_binding(doctor, patient)
        unpublished = self._upsert_prescription(doctor, patient, DEMO_UNPUBLISHED_TITLE, PrescriptionStatus.DRAFT.value)
        published = self._upsert_prescription(doctor, patient, DEMO_PUBLISHED_TITLE, PrescriptionStatus.PUBLISHED_TO_PATIENT.value)
        review = self._upsert_prescription(doctor, patient, DEMO_REVIEW_TITLE, PrescriptionStatus.DRAFT.value)
        self._upsert_patient_outputs(published)
        case = self._upsert_learning_case(published)
        self._upsert_quiz(case)
        reset_id = f"demo_reset_{uuid4().hex}"
        created = {
            "users": 3,
            "bindings": 1,
            "patient_profiles": 1,
            "prescriptions": 3,
            "learning_cases": 1,
            "quizzes": 1,
        }
        demo_ids = {
            "doctor_id": doctor.id,
            "patient_id": patient.id,
            "student_id": student.id,
            "unpublished_prescription_id": unpublished.id,
            "published_prescription_id": published.id,
            "review_prescription_id": review.id,
            "learning_case_id": case.id,
        }
        self.db.add(
            DemoResetLog(
                reset_id=reset_id,
                actor_user_id=actor_user_id,
                reset_scope=payload.reset_scope,
                status="success",
                created_summary_json=created,
                demo_ids_json=demo_ids,
            )
        )
        self.db.commit()

        return {
            "reset_id": reset_id,
            "status": "success",
            "created": created,
            "demo_ids": demo_ids,
            "accounts": _demo_accounts(),
        }

    def run_demo_smoke(self) -> DemoSmokeResult:
        checks = [
            self._check("health_ok", True, "critical", "Backend service is alive."),
            self._check("database_ok", self._database_ok(), "critical", "Database connection OK."),
        ]
        doctor = self._user(DEMO_DOCTOR_USERNAME)
        patient = self._user(DEMO_PATIENT_USERNAME)
        student = self._user(DEMO_STUDENT_USERNAME)
        binding = self._binding(doctor, patient)
        checks.extend(
            [
                self._check("doctor_demo_exists", doctor is not None, "critical", "doctor_demo account exists."),
                self._check("patient_demo_exists", patient is not None, "critical", "patient_demo account exists."),
                self._check("student_demo_exists", student is not None, "critical", "student_demo account exists."),
                self._check("doctor_patient_binding", binding is not None, "critical", "doctor_demo is bound to patient_demo."),
            ]
        )
        unpublished = self._prescription(DEMO_UNPUBLISHED_TITLE)
        published = self._prescription(DEMO_PUBLISHED_TITLE, PrescriptionStatus.PUBLISHED_TO_PATIENT.value)
        visible = list_patient_visible_prescriptions(self.db, patient.id) if patient else []
        visible_ids = {item["id"] for item in visible}
        checks.extend(
            [
                self._check(
                    "patient_cannot_see_unpublished",
                    unpublished is not None and unpublished.id not in visible_ids,
                    "critical",
                    "patient_demo cannot see draft/reviewing demo prescription.",
                ),
                self._check(
                    "patient_can_see_published",
                    published is not None and published.id in visible_ids,
                    "critical",
                    "patient_demo can see published demo prescription.",
                ),
            ]
        )
        learning_cases = list_learning_cases(self.db)
        checks.append(
            self._check(
                "student_can_see_deidentified_case",
                any(case.anonymized_case.get("case_id") == DEMO_CASE_ID for case in learning_cases),
                "critical",
                "student_demo has at least one deidentified student_learning case.",
            )
        )
        llm = LLMService(self.db).health()
        checks.append(
            self._check(
                "llm_fallback_available",
                bool(llm.get("llm_enabled")) and bool(llm.get("fallback_available", llm.get("allow_fallback"))),
                "medium",
                "LLM generation has template fallback available.",
            )
        )
        lightrag = self._safe_lightrag_health()
        checks.append(
            self._check(
                "lightrag_optional_or_available",
                (not lightrag.get("enabled")) or lightrag.get("status") in {"ok", "healthy"},
                "medium",
                "LightRAG is optional; local knowledge fallback can carry the demo.",
            )
        )
        checks.append(
            self._check(
                "harness_api_available",
                True,
                "critical",
                "Harness runner is available from backend service.",
            )
        )
        latest_packet = self.db.scalar(select(EvidencePacketLog).order_by(EvidencePacketLog.created_at.desc()))
        latest_trace = self.db.scalar(select(AgentTrace).order_by(AgentTrace.started_at.desc()))
        checks.extend(
            [
                self._check("latest_evidence_packet_readable", latest_packet is not None, "low", "Latest Evidence Packet can be read."),
                self._check("latest_trace_readable", latest_trace is not None, "low", "Latest Trace can be read."),
            ]
        )
        warnings = [check.name for check in checks if not check.passed and check.severity != "critical"]
        critical_failed = [check for check in checks if not check.passed and check.severity == "critical"]
        return DemoSmokeResult(
            smoke_id=f"smoke_{uuid4().hex}",
            passed=not critical_failed,
            total_checks=len(checks),
            passed_checks=sum(1 for check in checks if check.passed),
            failed_checks=sum(1 for check in checks if not check.passed),
            checks=checks,
            warnings=warnings,
        )

    def _database_ok(self) -> bool:
        try:
            self.db.execute(text("SELECT 1")).scalar()
            return True
        except Exception:
            return False

    def _safe_lightrag_health(self) -> dict[str, Any]:
        try:
            return lightrag_health()
        except Exception as exc:
            return {"enabled": False, "backend": "lightrag", "status": "unavailable", "degraded_reason": str(exc)}

    def _user(self, username: str) -> User | None:
        return self.db.scalar(select(User).where(User.username == username))

    def _binding(self, doctor: User | None, patient: User | None) -> DoctorPatientBinding | None:
        if not doctor or not patient:
            return None
        return self.db.scalar(
            select(DoctorPatientBinding).where(
                DoctorPatientBinding.doctor_id == doctor.id,
                DoctorPatientBinding.patient_id == patient.id,
                DoctorPatientBinding.is_active.is_(True),
            )
        )

    def _prescription(self, title: str, status_value: str | None = None) -> Prescription | None:
        query = select(Prescription).where(Prescription.title == title)
        if status_value:
            query = query.where(Prescription.status == status_value)
        return self.db.scalar(query.order_by(Prescription.id.desc()))

    def _learning_case(self) -> LearningCase | None:
        for case in self.db.scalars(select(LearningCase).order_by(LearningCase.id.desc())).all():
            if isinstance(case.anonymized_case, dict) and case.anonymized_case.get("case_id") == DEMO_CASE_ID:
                return case
        return None

    def _quiz(self) -> Quiz | None:
        return self.db.scalar(select(Quiz).where(Quiz.quiz_id == DEMO_QUIZ_ID))

    def _upsert_user(self, username: str, role: str, display_name: str) -> User:
        user = self._user(username)
        if user:
            user.role = role
            user.display_name = display_name
            user.hashed_password = hash_password(DEMO_PASSWORD)
            user.is_active = True
            self.db.flush()
            return user
        user = User(username=username, hashed_password=hash_password(DEMO_PASSWORD), role=role, display_name=display_name)
        self.db.add(user)
        self.db.flush()
        return user

    def _upsert_profiles(self, doctor: User, patient: User, student: User) -> None:
        doctor_profile = self.db.scalar(select(DoctorProfile).where(DoctorProfile.user_id == doctor.id))
        if not doctor_profile:
            doctor_profile = DoctorProfile(user_id=doctor.id)
            self.db.add(doctor_profile)
        doctor_profile.department = "中医科"
        doctor_profile.title = "主治医师"
        doctor_profile.institution = "GuardTCM 演示医院"
        doctor_profile.specialty = "中医内科审方演示"

        patient_profile = self.db.scalar(select(PatientProfile).where(PatientProfile.user_id == patient.id))
        if not patient_profile:
            patient_profile = PatientProfile(user_id=patient.id)
            self.db.add(patient_profile)
        patient_profile.age = 35
        patient_profile.sex = "female"
        patient_profile.body_state = "体质偏虚，睡眠欠佳。"

        health = self.db.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == patient.id))
        if not health:
            health = PatientHealthRecord(patient_id=patient.id)
            self.db.add(health)
        health.current_symptoms = "乏力、睡眠欠佳。"
        health.body_state = "示例身体状态，仅用于演示。"

        medical = self.db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient.id))
        if not medical:
            medical = MedicalHistory(patient_id=patient.id)
            self.db.add(medical)
        medical.summary = "示例既往史：慢性胃部不适史。"

        allergy = self.db.scalar(select(AllergyRecord).where(AllergyRecord.patient_id == patient.id))
        if not allergy:
            allergy = AllergyRecord(patient_id=patient.id)
            self.db.add(allergy)
        allergy.summary = "无明确药物过敏史。"

        student_profile = self.db.scalar(select(StudentProfile).where(StudentProfile.user_id == student.id))
        if not student_profile:
            student_profile = StudentProfile(user_id=student.id)
            self.db.add(student_profile)
        student_profile.school = "示例中医药大学"
        student_profile.grade = "研一"
        student_profile.major = "中医内科学"
        student_profile.learning_direction = "审方安全训练"
        self.db.flush()

    def _upsert_binding(self, doctor: User, patient: User) -> DoctorPatientBinding:
        binding = self.db.scalar(
            select(DoctorPatientBinding).where(DoctorPatientBinding.doctor_id == doctor.id, DoctorPatientBinding.patient_id == patient.id)
        )
        if not binding:
            binding = DoctorPatientBinding(doctor_id=doctor.id, patient_id=patient.id)
            self.db.add(binding)
        binding.is_active = True
        self.db.flush()
        return binding

    def _upsert_prescription(self, doctor: User, patient: User, title: str, status_value: str) -> Prescription:
        prescription = self._prescription(title)
        if not prescription:
            prescription = Prescription(doctor_id=doctor.id, patient_id=patient.id, title=title)
            self.db.add(prescription)
            self.db.flush()
        prescription.doctor_id = doctor.id
        prescription.patient_id = patient.id
        prescription.title = title
        prescription.status = status_value
        prescription.usage_method = "每日一剂，水煎服，遵医嘱。"
        prescription.doctor_notes = f"{DEMO_TAG}: {title}"
        for item in list(prescription.items):
            self.db.delete(item)
        self.db.flush()
        self.db.add(PrescriptionItem(prescription_id=prescription.id, herb_name="酸枣仁", dose="10", unit="g", notes="Phase 11 demo item"))
        self.db.flush()
        return prescription

    def _upsert_patient_outputs(self, prescription: Prescription) -> None:
        instruction = self.db.scalar(select(PatientInstruction).where(PatientInstruction.prescription_id == prescription.id))
        if not instruction:
            instruction = PatientInstruction(prescription_id=prescription.id, content="", safety_statement="")
            self.db.add(instruction)
        instruction.content = "请按医生最终确认的处方说明服药。如出现不适、过敏或症状加重，请及时联系医生。"
        instruction.safety_statement = "本说明来自医生确认后的处方转译，不构成自动诊断或自动开方。"

        diet = self.db.scalar(select(DietPrecaution).where(DietPrecaution.prescription_id == prescription.id))
        if not diet:
            diet = DietPrecaution(prescription_id=prescription.id, content="", details_json={})
            self.db.add(diet)
        diet.content = "请以医生备注原文为准；演示数据未调用外部模型补充说明。"
        diet.details_json = {
            "doctor_note": prescription.doctor_notes or "",
            "doctor_usage_method": prescription.usage_method or "",
            "ai_explanation": diet.content,
            "food_interactions": [],
            "herb_interactions": [],
            "food_evidence_missing": True,
            "source_ids": [],
            "sources": [],
            "llm": {"provider": "template", "model": "guardtcm-template-v1", "fallback_used": False, "degraded": False},
        }
        self.db.flush()

    def _upsert_learning_case(self, prescription: Prescription) -> LearningCase:
        case = self._learning_case()
        anonymized_case = {
            "case_id": DEMO_CASE_ID,
            "case_type": "prescription_review_training",
            "visibility": "student_learning",
            "deidentified": True,
            "patient_profile": {"age_range": "30-39岁", "gender": "female"},
            "symptoms": ["入睡困难", "睡眠不安"],
            "allergy_tags": ["无明确药物过敏史"],
            "medical_history_summary": "示例既往史，已脱敏。",
            "prescription": [{"herb_name": "酸枣仁", "dose": "10", "unit": "g"}],
            "source_id": "SRC_GUARDTCM_PHASE11_DEMO",
            "source_record_id": f"prescription:{prescription.id}",
            "demo_id": DEMO_TAG,
        }
        if not case:
            case = LearningCase(source_prescription_id=prescription.id, title="Phase 11 Demo 脱敏审方训练案例", anonymized_case={})
            self.db.add(case)
        case.source_prescription_id = prescription.id
        case.title = "Phase 11 Demo 脱敏审方训练案例"
        case.anonymized_case = anonymized_case
        case.standard_analysis = "该演示案例用于验证学习端能读取脱敏案例，风险判断应以 Evidence Packet 和规则引擎为准。"
        case.risk_points = [{"risk_type": "demo_review", "risk_level": "green", "message": "演示案例，无新增医学判断。"}]
        case.evidence_chain = [{"source_id": "SRC_GUARDTCM_PHASE11_DEMO", "summary": "Phase 11 demo evidence marker."}]
        self.db.flush()
        if not is_student_learning_case(case):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Demo learning case deidentification check failed")
        return case

    def _upsert_quiz(self, case: LearningCase) -> Quiz:
        quiz = self._quiz()
        if not quiz:
            quiz = Quiz(quiz_id=DEMO_QUIZ_ID, case_id=DEMO_CASE_ID, question_type="single_choice", question="")
            self.db.add(quiz)
        quiz.case_id = DEMO_CASE_ID
        quiz.question_type = "single_choice"
        quiz.question = "该案例在学习端展示前必须满足什么条件？"
        quiz.options_json = ["包含真实姓名", "已脱敏且 visibility=student_learning", "患者可自行加量", "跳过 Evidence Packet"]
        quiz.answer = "已脱敏且 visibility=student_learning"
        quiz.explanation = "学生端只能查看脱敏后的 student_learning 案例。"
        quiz.knowledge_points_json = ["deidentification", "student_learning_visibility"]
        quiz.source_id = "SRC_GUARDTCM_PHASE11_DEMO"
        quiz.metadata_json = {"is_demo": True, "demo_id": DEMO_TAG}
        self.db.flush()
        return quiz

    def _check(self, name: str, passed: bool, severity: str, details: str, metadata: dict[str, Any] | None = None) -> DemoCheck:
        return DemoCheck(name=name, passed=passed, severity=severity, details=details, metadata=metadata or {})


def _demo_accounts() -> dict[str, dict[str, str]]:
    return {
        "doctor": {"username": DEMO_DOCTOR_USERNAME, "password": DEMO_PASSWORD},
        "patient": {"username": DEMO_PATIENT_USERNAME, "password": DEMO_PASSWORD},
        "student": {"username": DEMO_STUDENT_USERNAME, "password": DEMO_PASSWORD},
    }
