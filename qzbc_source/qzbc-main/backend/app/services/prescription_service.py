from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import PrescriptionStatus
from app.db.models import (
    DietPrecaution,
    MedicalHistory,
    PatientHealthRecord,
    PatientInstruction,
    PatientProfile,
    Prescription,
    PrescriptionItem,
    PrescriptionReview,
    PrescriptionStatusLog,
    RiskCard,
    User,
)
from app.schemas.prescription import PrescriptionCreateRequest, PrescriptionUpdateRequest
from app.services.medical_history_service import append_confirmed_encounter
from app.services.patient_service import ensure_patient_bound
from app.services.patient_guidance_service import patient_guidance_payload, upsert_patient_guidance


ALLOWED_TRANSITIONS = {
    PrescriptionStatus.DRAFT.value: {PrescriptionStatus.REVIEWING.value},
    PrescriptionStatus.REVIEWING.value: {PrescriptionStatus.DOCTOR_CONFIRMED.value},
    PrescriptionStatus.DOCTOR_CONFIRMED.value: {
        PrescriptionStatus.PUBLISHED_TO_PATIENT.value,
        PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value,
    },
    PrescriptionStatus.PUBLISHED_TO_PATIENT.value: {PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value},
}


def serialize_prescription(prescription: Prescription) -> dict:
    return {
        "id": prescription.id,
        "patient_id": prescription.patient_id,
        "doctor_id": prescription.doctor_id,
        "title": prescription.title,
        "symptom_context": prescription.symptom_context or "",
        "patient_age_at_visit": prescription.patient_age_at_visit,
        "patient_sex_at_visit": prescription.patient_sex_at_visit,
        "status": prescription.status,
        "usage_method": prescription.usage_method,
        "doctor_notes": prescription.doctor_notes,
        "items": [
            {
                "id": item.id,
                "herb_name": item.herb_name,
                "dose": item.dose,
                "unit": item.unit,
                "notes": item.notes,
            }
            for item in prescription.items
        ],
    }


class PrescriptionStateService:
    def __init__(self, db: Session):
        self.db = db

    def transition(self, prescription: Prescription, to_status: str, actor: User, reason: str) -> Prescription:
        allowed = ALLOWED_TRANSITIONS.get(prescription.status, set())
        if to_status not in allowed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Invalid prescription status transition: {prescription.status} -> {to_status}",
            )
        from_status = prescription.status
        prescription.status = to_status
        self.db.add(
            PrescriptionStatusLog(
                prescription_id=prescription.id,
                from_status=from_status,
                to_status=to_status,
                changed_by_user_id=actor.id,
                changed_by_role=actor.role,
                reason=reason,
            )
        )
        self.db.flush()
        if to_status == PrescriptionStatus.DOCTOR_CONFIRMED.value:
            _record_confirmed_encounter(self.db, prescription)
        return prescription


def get_prescription_for_doctor(db: Session, doctor_id: int, prescription_id: int) -> Prescription:
    prescription = db.get(Prescription, prescription_id)
    if not prescription:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prescription not found")
    ensure_patient_bound(db, doctor_id, prescription.patient_id)
    if prescription.doctor_id != doctor_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor does not own this prescription")
    return prescription


def create_draft_prescription(db: Session, doctor: User, payload: PrescriptionCreateRequest) -> Prescription:
    ensure_patient_bound(db, doctor.id, payload.patient_id)
    profile = db.scalar(select(PatientProfile).where(PatientProfile.user_id == payload.patient_id))
    prescription = Prescription(
        doctor_id=doctor.id,
        patient_id=payload.patient_id,
        title=payload.title,
        symptom_context=payload.symptom_context.strip() or None,
        patient_age_at_visit=profile.age if profile else None,
        patient_sex_at_visit=profile.sex if profile else None,
        usage_method=payload.usage_method,
        doctor_notes=payload.doctor_notes,
    )
    db.add(prescription)
    db.flush()
    for item in payload.items:
        db.add(PrescriptionItem(prescription_id=prescription.id, **item.model_dump()))
    db.add(
        PrescriptionStatusLog(
            prescription_id=prescription.id,
            from_status=None,
            to_status=PrescriptionStatus.DRAFT.value,
            changed_by_user_id=doctor.id,
            changed_by_role=doctor.role,
            reason="create_draft",
        )
    )
    db.commit()
    db.refresh(prescription)
    return prescription


def update_draft_prescription(
    db: Session,
    doctor: User,
    prescription_id: int,
    payload: PrescriptionUpdateRequest,
) -> Prescription:
    prescription = get_prescription_for_doctor(db, doctor.id, prescription_id)
    if prescription.status not in {PrescriptionStatus.DRAFT.value, PrescriptionStatus.REVIEWING.value}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only draft or reviewing prescriptions can be edited",
        )
    if "symptom_context" in payload.model_fields_set:
        prescription.symptom_context = (payload.symptom_context or "").strip() or None
    if "usage_method" in payload.model_fields_set:
        prescription.usage_method = (payload.usage_method or "").strip() or None
    if "doctor_notes" in payload.model_fields_set:
        prescription.doctor_notes = (payload.doctor_notes or "").strip() or None
    db.add(prescription)
    db.commit()
    db.refresh(prescription)
    return prescription


def review_prescription(db: Session, doctor: User, prescription_id: int) -> dict:
    from app.agents.graphs.doctor_review_graph import DoctorReviewGraph

    return DoctorReviewGraph(db).invoke(doctor, prescription_id)


def persist_prescription_review_result(
    db: Session,
    *,
    prescription_id: int,
    review_result: dict,
    rule_findings: list[dict],
    evidence_packet: dict,
) -> dict:
    prescription = db.get(Prescription, prescription_id)
    if not prescription:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prescription not found")
    existing = _find_existing_review_for_packet(db, prescription_id, evidence_packet)
    if existing:
        return _serialize_review_result(existing, evidence_packet)

    review = PrescriptionReview(
        prescription_id=prescription.id,
        risk_level=review_result.get("risk_level", "red"),
        review_summary=review_result.get("review_summary", "结构化规则引擎未返回结果，必须由医生人工复核。"),
        evidence_packet=evidence_packet,
    )
    db.add(review)
    db.flush()
    risk_cards = []
    for card in rule_findings:
        risk_card = {
            "risk_type": card.get("risk_type") or card.get("rule_type") or card.get("rule_id", "structured_rule"),
            "risk_level": card.get("risk_level", "green"),
            "message": card.get("message", ""),
            "evidence": card.get("evidence") or card.get("evidence_query", ""),
        }
        risk = RiskCard(review_id=review.id, **risk_card)
        db.add(risk)
        risk_cards.append({**risk_card, **card})

    db.commit()
    return {
        "prescription_id": prescription.id,
        "risk_level": review.risk_level,
        "review_summary": review.review_summary,
        "risk_cards": risk_cards,
        "rule_findings": rule_findings,
        "evidence_packet": evidence_packet,
    }


def update_prescription_review_human_confirm(db: Session, prescription_id: int, packet_id: str | None, human_confirm: dict) -> None:
    if not packet_id:
        return
    reviews = db.scalars(
        select(PrescriptionReview)
        .where(PrescriptionReview.prescription_id == prescription_id)
        .order_by(PrescriptionReview.created_at.desc())
    ).all()
    for review in reviews:
        packet = dict(review.evidence_packet or {})
        if packet.get("packet_id") == packet_id:
            packet["human_confirm"] = {**(packet.get("human_confirm") or {}), **human_confirm}
            review.evidence_packet = packet
            db.flush()
            return


def _find_existing_review_for_packet(db: Session, prescription_id: int, evidence_packet: dict) -> PrescriptionReview | None:
    packet_id = evidence_packet.get("packet_id")
    trace_id = evidence_packet.get("trace_id")
    if not packet_id and not trace_id:
        return None
    reviews = db.scalars(
        select(PrescriptionReview)
        .where(PrescriptionReview.prescription_id == prescription_id)
        .order_by(PrescriptionReview.created_at.desc())
    ).all()
    for review in reviews:
        packet = review.evidence_packet or {}
        if packet_id and packet.get("packet_id") == packet_id:
            return review
        if trace_id and packet.get("trace_id") == trace_id:
            return review
    return None


def _serialize_review_result(review: PrescriptionReview, evidence_packet: dict) -> dict:
    return {
        "prescription_id": review.prescription_id,
        "risk_level": review.risk_level,
        "review_summary": review.review_summary,
        "risk_cards": [],
        "rule_findings": evidence_packet.get("rule_findings", []),
        "evidence_packet": evidence_packet,
    }


def _planned_tool_arguments(tool_name: str, prescription: Prescription, doctor: User) -> dict:
    if tool_name == "get_reasoning_path":
        return {
            "prescription_id": prescription.id,
            "patient_id": prescription.patient_id,
            "visibility": "doctor_internal",
        }
    if tool_name == "search_role_memory":
        return {
            "owner_role": doctor.role,
            "owner_user_id": doctor.id,
            "query": "doctor prescription review preference",
        }
    return {"prescription_id": prescription.id}


def confirm_prescription(db: Session, doctor: User, prescription_id: int) -> Prescription:
    prescription = get_prescription_for_doctor(db, doctor.id, prescription_id)
    PrescriptionStateService(db).transition(
        prescription,
        PrescriptionStatus.DOCTOR_CONFIRMED.value,
        doctor,
        "doctor_confirm",
    )
    db.commit()
    db.refresh(prescription)
    return prescription


def _record_confirmed_encounter(db: Session, prescription: Prescription) -> None:
    """Persist the doctor-confirmed symptom snapshot exactly once."""
    symptoms = (prescription.symptom_context or "").strip()
    if not symptoms:
        return

    medical = db.scalar(
        select(MedicalHistory)
        .where(MedicalHistory.patient_id == prescription.patient_id)
        .order_by(MedicalHistory.created_at.asc())
    )
    if not medical:
        medical = MedicalHistory(patient_id=prescription.patient_id, summary="")
        db.add(medical)
    visit_date = datetime.now(timezone.utc).date().isoformat()
    medical.summary = append_confirmed_encounter(
        medical.summary,
        visit_date=visit_date,
        prescription_title=prescription.title,
        symptoms=symptoms,
    )

    health = db.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == prescription.patient_id))
    if not health:
        health = PatientHealthRecord(patient_id=prescription.patient_id)
        db.add(health)
    health.current_symptoms = symptoms
    health.updated_at = datetime.now(timezone.utc)


def publish_prescription(db: Session, doctor: User, prescription_id: int) -> Prescription:
    prescription = get_prescription_for_doctor(db, doctor.id, prescription_id)
    PrescriptionStateService(db).transition(
        prescription,
        PrescriptionStatus.PUBLISHED_TO_PATIENT.value,
        doctor,
        "publish_to_patient",
    )
    # Generate from the final confirmed snapshot. Creating this during machine
    # review caused later doctor edits to be hidden behind stale boilerplate.
    upsert_patient_guidance(db, prescription)
    db.commit()
    db.refresh(prescription)
    return prescription


def list_patient_visible_prescriptions(db: Session, patient_id: int) -> list[dict]:
    prescriptions = db.scalars(
        select(Prescription).where(
            Prescription.patient_id == patient_id,
            Prescription.status.in_(
                [
                    PrescriptionStatus.PUBLISHED_TO_PATIENT.value,
                    PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value,
                ]
            ),
        )
    ).all()
    return [serialize_patient_prescription(db, prescription) for prescription in prescriptions]


def get_patient_visible_prescription(db: Session, patient_id: int, prescription_id: int) -> dict:
    prescription = db.get(Prescription, prescription_id)
    if (
        not prescription
        or prescription.patient_id != patient_id
        or prescription.status
        not in [PrescriptionStatus.PUBLISHED_TO_PATIENT.value, PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value]
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No published prescription found")
    return serialize_patient_prescription(db, prescription)


def serialize_patient_prescription(db: Session, prescription: Prescription) -> dict:
    instruction = db.scalar(select(PatientInstruction).where(PatientInstruction.prescription_id == prescription.id))
    diet = db.scalar(select(DietPrecaution).where(DietPrecaution.prescription_id == prescription.id))
    data = serialize_prescription(prescription)
    data.pop("doctor_notes", None)
    data["patient_instruction"] = {
        "content": instruction.content if instruction else "",
        "safety_statement": instruction.safety_statement if instruction else "",
        "doctor_note": (prescription.doctor_notes or "").strip(),
        "doctor_usage_method": (prescription.usage_method or "").strip(),
        "item_notes": [
            {"herb_name": item.herb_name, "note": item.notes.strip()}
            for item in prescription.items
            if item.notes and item.notes.strip()
        ],
    }
    data["diet_precaution"] = patient_guidance_payload(prescription, diet)
    return data
