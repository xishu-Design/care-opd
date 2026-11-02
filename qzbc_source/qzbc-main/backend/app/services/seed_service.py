from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import UserRole
from app.db.models import DoctorPatientBinding, LearningCase, PatientProfile, Prescription, User
from app.schemas.auth import RegisterRequest
from app.schemas.patient import PatientCreateRequest
from app.schemas.prescription import PrescriptionCreateRequest, PrescriptionItemIn
from app.services.auth_service import register_user
from app.services.learning_service import create_deidentified_learning_case
from app.services.patient_service import create_or_bind_patient
from app.services.prescription_service import confirm_prescription, create_draft_prescription, publish_prescription, review_prescription


def seed_demo_data(db: Session) -> dict:
    doctor = db.scalar(select(User).where(User.username == "doctor_demo"))
    if not doctor:
        doctor = register_user(
            db,
            RegisterRequest(username="doctor_demo", password="password123", role=UserRole.DOCTOR, display_name="Demo Doctor"),
        )
    patient = db.scalar(select(User).where(User.username == "patient_demo"))
    if not patient:
        create_or_bind_patient(
            db,
            doctor,
            PatientCreateRequest(
                username="patient_demo",
                password="password123",
                display_name="Demo Patient",
                age=42,
                sex="female",
                allergy_history="花粉",
                medical_history="慢性胃炎史",
                current_symptoms="失眠、乏力",
                body_state="体质偏虚",
            ),
        )
        patient = db.scalar(select(User).where(User.username == "patient_demo"))
    elif not db.scalar(select(DoctorPatientBinding).where(DoctorPatientBinding.doctor_id == doctor.id, DoctorPatientBinding.patient_id == patient.id)):
        db.add(DoctorPatientBinding(doctor_id=doctor.id, patient_id=patient.id))
        db.commit()

    student = db.scalar(select(User).where(User.username == "student_demo"))
    if not student:
        student = register_user(
            db,
            RegisterRequest(username="student_demo", password="password123", role=UserRole.STUDENT, display_name="Demo Student"),
        )

    if patient and not db.scalar(select(Prescription).where(Prescription.title == "Demo published prescription")):
        draft = create_draft_prescription(
            db,
            doctor,
            PrescriptionCreateRequest(
                patient_id=patient.id,
                title="Demo published prescription",
                symptom_context="失眠、乏力",
                usage_method="每日一剂，水煎服，遵医嘱。",
                doctor_notes="用于演示的医生确认处方。",
                items=[PrescriptionItemIn(herb_name="酸枣仁", dose="10", unit="g")],
            ),
        )
        review_prescription(db, doctor, draft.id)
        confirm_prescription(db, doctor, draft.id)
        publish_prescription(db, doctor, draft.id)
        create_deidentified_learning_case(db, doctor, draft.id)

    if patient:
        _refresh_demo_learning_case_context(db)

    return {
        "doctor": "doctor_demo / password123",
        "patient": "patient_demo / password123",
        "student": "student_demo / password123",
        "learning_cases": db.scalar(select(LearningCase).where(LearningCase.source_prescription_id.is_not(None))) is not None,
    }


def _refresh_demo_learning_case_context(db: Session) -> None:
    """Upgrade old demo snapshots into safe symptom-to-composition teaching cases."""
    for case in db.scalars(select(LearningCase).where(LearningCase.source_prescription_id.is_not(None))).all():
        payload = dict(case.anonymized_case or {})
        if payload.get("deidentified") is not True or payload.get("visibility") != "student_learning":
            continue
        payload["patient_profile"] = payload.get("patient_profile") or {
            "age_range": payload.pop("age_range", "40-49岁"),
            "gender": payload.pop("sex", "female"),
        }
        case_profile = payload["patient_profile"] if isinstance(payload.get("patient_profile"), dict) else {}
        source_prescription = db.get(Prescription, case.source_prescription_id) if case.source_prescription_id else None
        patient_profile = (
            db.scalar(select(PatientProfile).where(PatientProfile.user_id == source_prescription.patient_id))
            if source_prescription
            else None
        )
        if case_profile.get("age") is None and patient_profile and patient_profile.age is not None:
            case_profile["age"] = patient_profile.age
        payload["patient_profile"] = case_profile
        if not payload.get("symptoms"):
            payload["symptoms"] = ["失眠", "乏力"]
        prescription = payload.get("prescription")
        if isinstance(prescription, list):
            payload["prescription"] = {"items": prescription}
        payload.setdefault("knowledge_points", ["症状与处方组成对照", "脱敏教学边界"])
        payload.setdefault("case_note", "演示用脱敏教学案例，不构成临床处方建议。")
        case.anonymized_case = payload
    db.commit()
