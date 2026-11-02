from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AllergyRecord, MedicalHistory, PatientHealthRecord, PatientProfile, Prescription, User
from app.mcp.contracts import ToolRequest
from app.services.medical_history_service import normalize_medical_history


def execute(db: Session, request: ToolRequest) -> dict:
    if request.tool_name == "get_bound_patient_roster":
        from app.services.patient_service import list_bound_patients

        patients = list_bound_patients(db, request.context.user_id)
        roster = []
        for patient in patients[:50]:
            latest = db.scalar(
                select(Prescription)
                .where(Prescription.doctor_id == request.context.user_id, Prescription.patient_id == patient.id)
                .order_by(Prescription.updated_at.desc(), Prescription.id.desc())
                .limit(1)
            )
            roster.append(
                {
                    "patient_id": patient.id,
                    "username": patient.username,
                    "display_name": patient.display_name,
                    "age": patient.age,
                    "sex": patient.sex,
                    "current_symptoms": patient.current_symptoms,
                    "body_state": patient.body_state,
                    "medical_history": patient.medical_history,
                    "allergy_history": patient.allergy_history,
                    "latest_prescription": (
                        {
                            "prescription_id": latest.id,
                            "title": latest.title,
                            "status": latest.status,
                            "symptom_context": latest.symptom_context or "",
                            "usage_method": latest.usage_method or "",
                            "doctor_notes": latest.doctor_notes or "",
                            "items": [
                                {
                                    "herb_name": item.herb_name,
                                    "dose": item.dose,
                                    "unit": item.unit,
                                    "notes": item.notes or "",
                                }
                                for item in latest.items
                            ],
                            "updated_at": latest.updated_at.isoformat(),
                        }
                        if latest
                        else None
                    ),
                }
            )
        return {"doctor_patient_roster": roster, "patient_count": len(roster)}

    patient_id = request.arguments.get("patient_id") or request.context.patient_id or request.context.user_id
    patient = db.get(User, patient_id)
    profile = db.scalar(select(PatientProfile).where(PatientProfile.user_id == patient_id))
    medical = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient_id))
    allergy = db.scalar(select(AllergyRecord).where(AllergyRecord.patient_id == patient_id))
    health = db.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == patient_id))
    return {
        "patient_profile": {
            "id": patient_id,
            "display_name": patient.display_name if patient else None,
            "age": profile.age if profile else None,
            "sex": profile.sex if profile else None,
            "height_cm": profile.height_cm if profile else None,
            "weight_kg": profile.weight_kg if profile else None,
            "body_state": (health.body_state if health else None) or (profile.body_state if profile else None),
            "current_symptoms": health.current_symptoms if health else None,
        },
        "medical_history": normalize_medical_history(medical.summary if medical else ""),
        "allergy_history": allergy.summary if allergy else "",
    }
