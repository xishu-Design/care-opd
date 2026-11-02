from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import UserRole
from app.core.security import hash_password
from app.db.models import (
    AllergyRecord,
    DoctorPatientBinding,
    MedicalHistory,
    PatientHealthRecord,
    PatientProfile,
    User,
)
from app.schemas.patient import PatientCreateRequest, PatientProfileResponse, PatientUpdateRequest
from app.services.medical_history_service import normalize_medical_history


def ensure_patient_bound(db: Session, doctor_id: int, patient_id: int) -> None:
    binding = db.scalar(
        select(DoctorPatientBinding).where(
            DoctorPatientBinding.doctor_id == doctor_id,
            DoctorPatientBinding.patient_id == patient_id,
            DoctorPatientBinding.is_active.is_(True),
        )
    )
    if not binding:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor is not bound to this patient")


def _patient_to_response(db: Session, patient: User) -> PatientProfileResponse:
    profile = db.scalar(select(PatientProfile).where(PatientProfile.user_id == patient.id))
    health = db.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == patient.id))
    medical = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient.id))
    allergy = db.scalar(select(AllergyRecord).where(AllergyRecord.patient_id == patient.id))
    return PatientProfileResponse(
        id=patient.id,
        username=patient.username,
        display_name=patient.display_name,
        age=profile.age if profile else None,
        sex=profile.sex if profile else None,
        height_cm=profile.height_cm if profile else None,
        weight_kg=profile.weight_kg if profile else None,
        body_state=(health.body_state if health else None) or (profile.body_state if profile else "") or "",
        current_symptoms=(health.current_symptoms if health else "") or "",
        medical_history=normalize_medical_history(medical.summary if medical else ""),
        allergy_history=allergy.summary if allergy else "",
    )


def create_or_bind_patient(db: Session, doctor: User, payload: PatientCreateRequest) -> PatientProfileResponse:
    patient = db.scalar(select(User).where(User.username == payload.username))
    if patient and patient.role != UserRole.PATIENT.value:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Existing username is not a patient account")

    if not patient:
        if not payload.password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="新患者尚未注册，请设置一次性临时密码；患者首次登录后必须自行修改",
            )
        patient = User(
            username=payload.username,
            hashed_password=hash_password(payload.password),
            role=UserRole.PATIENT.value,
            display_name=payload.display_name,
            must_change_password=True,
        )
        db.add(patient)
        db.flush()
        db.add(
            PatientProfile(
                user_id=patient.id,
                age=payload.age,
                sex=payload.sex,
                height_cm=payload.height_cm,
                weight_kg=payload.weight_kg,
                body_state=payload.body_state,
            )
        )
        db.add(PatientHealthRecord(patient_id=patient.id, current_symptoms=payload.current_symptoms, body_state=payload.body_state))
        db.add(MedicalHistory(patient_id=patient.id, summary=payload.medical_history))
        db.add(AllergyRecord(patient_id=patient.id, summary=payload.allergy_history))
    else:
        # A patient may have registered before the doctor binds the account. In
        # that flow the clinical profile must still receive the values entered
        # by the doctor instead of silently keeping an empty profile.
        profile = db.scalar(select(PatientProfile).where(PatientProfile.user_id == patient.id))
        if not profile:
            profile = PatientProfile(user_id=patient.id)
            db.add(profile)
        if payload.age is not None:
            profile.age = payload.age
        if payload.sex:
            profile.sex = payload.sex
        if payload.height_cm is not None:
            profile.height_cm = payload.height_cm
        if payload.weight_kg is not None:
            profile.weight_kg = payload.weight_kg
        if payload.body_state:
            profile.body_state = payload.body_state

        health = db.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == patient.id))
        if not health:
            health = PatientHealthRecord(patient_id=patient.id)
            db.add(health)
        if payload.current_symptoms:
            health.current_symptoms = payload.current_symptoms
        if payload.body_state:
            health.body_state = payload.body_state

        medical = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient.id))
        if not medical:
            medical = MedicalHistory(patient_id=patient.id, summary="")
            db.add(medical)
        if payload.medical_history:
            medical.summary = payload.medical_history

        allergy = db.scalar(select(AllergyRecord).where(AllergyRecord.patient_id == patient.id))
        if not allergy:
            allergy = AllergyRecord(patient_id=patient.id, summary="")
            db.add(allergy)
        if payload.allergy_history:
            allergy.summary = payload.allergy_history

    existing = db.scalar(
        select(DoctorPatientBinding).where(
            DoctorPatientBinding.doctor_id == doctor.id,
            DoctorPatientBinding.patient_id == patient.id,
        )
    )
    if existing and existing.is_active:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Patient is already bound to this doctor")
    if existing:
        existing.is_active = True
    else:
        db.add(DoctorPatientBinding(doctor_id=doctor.id, patient_id=patient.id))

    db.commit()
    return _patient_to_response(db, patient)


def list_bound_patients(db: Session, doctor_id: int) -> list[PatientProfileResponse]:
    rows = db.scalars(
        select(User)
        .join(DoctorPatientBinding, DoctorPatientBinding.patient_id == User.id)
        .where(DoctorPatientBinding.doctor_id == doctor_id, DoctorPatientBinding.is_active.is_(True))
    ).all()
    return [_patient_to_response(db, patient) for patient in rows]


def get_patient_for_doctor(db: Session, doctor_id: int, patient_id: int) -> PatientProfileResponse:
    ensure_patient_bound(db, doctor_id, patient_id)
    patient = db.get(User, patient_id)
    if not patient or patient.role != UserRole.PATIENT.value:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return _patient_to_response(db, patient)


def update_patient_for_doctor(
    db: Session,
    doctor_id: int,
    patient_id: int,
    payload: PatientUpdateRequest,
) -> PatientProfileResponse:
    ensure_patient_bound(db, doctor_id, patient_id)
    patient = db.get(User, patient_id)
    if not patient or patient.role != UserRole.PATIENT.value:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    if "display_name" in payload.model_fields_set and payload.display_name is not None:
        patient.display_name = payload.display_name.strip() or patient.display_name

    profile = db.scalar(select(PatientProfile).where(PatientProfile.user_id == patient_id))
    if not profile:
        profile = PatientProfile(user_id=patient_id)
        db.add(profile)
    for field in ("age", "sex", "height_cm", "weight_kg"):
        if field in payload.model_fields_set:
            setattr(profile, field, getattr(payload, field))
    if "body_state" in payload.model_fields_set:
        profile.body_state = payload.body_state or ""

    health = db.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == patient_id))
    if not health:
        health = PatientHealthRecord(patient_id=patient_id)
        db.add(health)
    if "body_state" in payload.model_fields_set:
        health.body_state = payload.body_state or ""
    if "current_symptoms" in payload.model_fields_set:
        health.current_symptoms = payload.current_symptoms or ""

    medical = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient_id))
    if not medical:
        medical = MedicalHistory(patient_id=patient_id, summary="")
        db.add(medical)
    if "medical_history" in payload.model_fields_set:
        medical.summary = payload.medical_history or ""

    allergy = db.scalar(select(AllergyRecord).where(AllergyRecord.patient_id == patient_id))
    if not allergy:
        allergy = AllergyRecord(patient_id=patient_id, summary="")
        db.add(allergy)
    if "allergy_history" in payload.model_fields_set:
        allergy.summary = payload.allergy_history or ""

    db.commit()
    return _patient_to_response(db, patient)


def get_own_patient_profile(db: Session, patient_id: int) -> PatientProfileResponse:
    patient = db.get(User, patient_id)
    if not patient or patient.role != UserRole.PATIENT.value:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return _patient_to_response(db, patient)
