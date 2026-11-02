from __future__ import annotations

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.core.enums import UserRole
from app.core.security import hash_password
from app.db.models import (
    AllergyRecord,
    DeidentificationLog,
    DietPrecaution,
    DoctorPatientBinding,
    LearningCase,
    MedicalHistory,
    PatientHealthRecord,
    PatientInstruction,
    PatientProfile,
    PendingHumanAction,
    Prescription,
    PrescriptionItem,
    PrescriptionReview,
    PrescriptionStatusLog,
    RuleExecutionLog,
    RiskCard,
    StudentAnswer,
    User,
    GraphPath,
    MockGraphPath,
)
from app.knowledge.ingestion import ingest_local_knowledge
from app.rules.rule_loader import load_rules


def ensure_phase4_demo_seed(db: Session) -> dict[str, User]:
    doctor = _ensure_user(db, "phase4_demo_doctor", UserRole.DOCTOR.value, "Phase4 Demo Doctor")
    patient = _ensure_user(db, "phase4_demo_patient", UserRole.PATIENT.value, "Phase4 Patient")
    student = _ensure_user(db, "phase4_demo_student", UserRole.STUDENT.value, "Phase4 Demo Student")
    unbound = _ensure_user(db, "phase4_unbound_patient", UserRole.PATIENT.value, "Phase4 Unbound Patient")

    _ensure_patient_profile(db, patient, allergy_history="黄芪")
    _ensure_patient_profile(db, unbound, allergy_history="花粉")
    if not db.scalar(select(DoctorPatientBinding).where(DoctorPatientBinding.doctor_id == doctor.id, DoctorPatientBinding.patient_id == patient.id)):
        db.add(DoctorPatientBinding(doctor_id=doctor.id, patient_id=patient.id))

    ingest_local_knowledge(db)
    load_rules(db)
    db.commit()
    return {"doctor": doctor, "patient": patient, "student": student, "unbound_patient": unbound}


def reset_phase4_demo_data(db: Session) -> None:
    prescriptions = db.scalars(select(Prescription).where(Prescription.title == "Phase4 Demo 风险拟方")).all()
    prescription_ids = [row.id for row in prescriptions]
    if not prescription_ids:
        return
    learning_case_ids = [
        row.id
        for row in db.scalars(select(LearningCase).where(LearningCase.source_prescription_id.in_(prescription_ids))).all()
    ]
    review_ids = [
        row.id
        for row in db.scalars(select(PrescriptionReview).where(PrescriptionReview.prescription_id.in_(prescription_ids))).all()
    ]
    if learning_case_ids:
        db.execute(update(GraphPath).where(GraphPath.learning_case_id.in_(learning_case_ids)).values(learning_case_id=None))
        db.execute(update(MockGraphPath).where(MockGraphPath.learning_case_id.in_(learning_case_ids)).values(learning_case_id=None))
        db.execute(delete(StudentAnswer).where(StudentAnswer.learning_case_id.in_(learning_case_ids)))
        db.execute(delete(DeidentificationLog).where(DeidentificationLog.learning_case_id.in_(learning_case_ids)))
        db.execute(delete(LearningCase).where(LearningCase.id.in_(learning_case_ids)))
    if review_ids:
        db.execute(delete(RiskCard).where(RiskCard.review_id.in_(review_ids)))
    db.execute(delete(PendingHumanAction).where(PendingHumanAction.prescription_id.in_(prescription_ids)))
    db.execute(delete(PrescriptionReview).where(PrescriptionReview.prescription_id.in_(prescription_ids)))
    db.execute(delete(PatientInstruction).where(PatientInstruction.prescription_id.in_(prescription_ids)))
    db.execute(delete(DietPrecaution).where(DietPrecaution.prescription_id.in_(prescription_ids)))
    db.execute(delete(PrescriptionStatusLog).where(PrescriptionStatusLog.prescription_id.in_(prescription_ids)))
    db.execute(update(RuleExecutionLog).where(RuleExecutionLog.prescription_id.in_(prescription_ids)).values(prescription_id=None))
    db.execute(update(GraphPath).where(GraphPath.prescription_id.in_(prescription_ids)).values(prescription_id=None))
    db.execute(update(MockGraphPath).where(MockGraphPath.prescription_id.in_(prescription_ids)).values(prescription_id=None))
    db.execute(delete(PrescriptionItem).where(PrescriptionItem.prescription_id.in_(prescription_ids)))
    db.execute(delete(Prescription).where(Prescription.id.in_(prescription_ids)))
    db.commit()


def _ensure_user(db: Session, username: str, role: str, display_name: str) -> User:
    user = db.scalar(select(User).where(User.username == username))
    if user:
        return user
    user = User(
        username=username,
        hashed_password=hash_password("password123"),
        role=role,
        display_name=display_name,
    )
    db.add(user)
    db.flush()
    return user


def _ensure_patient_profile(db: Session, patient: User, allergy_history: str) -> None:
    if not db.scalar(select(PatientProfile).where(PatientProfile.user_id == patient.id)):
        db.add(PatientProfile(user_id=patient.id, age=39, sex="female", body_state="体质偏虚"))
    health = db.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == patient.id))
    if not health:
        db.add(PatientHealthRecord(patient_id=patient.id, current_symptoms="乏力、失眠", body_state="体质偏虚"))
    medical = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient.id))
    if not medical:
        db.add(MedicalHistory(patient_id=patient.id, summary="慢性胃炎史"))
    allergy = db.scalar(select(AllergyRecord).where(AllergyRecord.patient_id == patient.id))
    if allergy:
        allergy.summary = allergy_history
    else:
        db.add(AllergyRecord(patient_id=patient.id, summary=allergy_history))
    db.flush()
