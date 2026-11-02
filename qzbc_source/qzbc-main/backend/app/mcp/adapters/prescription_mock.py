from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import PrescriptionStatus
from app.db.models import DietPrecaution, PatientInstruction, Prescription, User
from app.mcp.contracts import ToolRequest
from app.services.prescription_service import serialize_patient_prescription


def _items(prescription: Prescription | None) -> list[dict]:
    if not prescription:
        return []
    return [
        {"herb_name": item.herb_name, "dose": item.dose, "unit": item.unit, "notes": item.notes}
        for item in prescription.items
    ]


def execute(db: Session, request: ToolRequest) -> dict:
    prescription_id = request.arguments.get("prescription_id") or request.context.prescription_id
    prescription = db.get(Prescription, prescription_id) if prescription_id else None
    if request.tool_name == "parse_prescription":
        return {"parsed_items": _items(prescription)}
    if request.tool_name == "generate_patient_instruction":
        return {
            "patient_instruction": "请按医生最终确认的处方说明服药，如出现不适请联系医生。",
            "safety_statement": "本说明不构成自动诊断或自动开方。",
        }
    if request.tool_name == "generate_diet_precaution":
        return {"diet_precaution": "服药期间保持清淡饮食，避免自行添加治疗性食疗方。"}
    if request.tool_name == "get_prescription_detail":
        return {"prescription_id": prescription_id, "items": _items(prescription), "status": prescription.status if prescription else None}
    if request.tool_name == "get_my_published_prescriptions":
        prescriptions = db.scalars(
            select(Prescription)
            .where(
                Prescription.patient_id == request.context.user_id,
                Prescription.status.in_(
                    [
                        PrescriptionStatus.PUBLISHED_TO_PATIENT.value,
                        PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value,
                    ]
                ),
            )
            .order_by(Prescription.updated_at.desc(), Prescription.id.desc())
            .limit(10)
        ).all()
        rows = []
        for index, item in enumerate(prescriptions):
            row = serialize_patient_prescription(db, item)
            doctor = db.get(User, item.doctor_id)
            row["doctor_display_name"] = doctor.display_name if doctor else ""
            row["is_current"] = index == 0
            row["published_at"] = item.updated_at.isoformat()
            rows.append(row)
        return {"published_prescriptions": rows, "prescription_count": len(rows)}
    if request.tool_name == "create_prescription_draft":
        return {"draft_preview": request.arguments}
    return {"tool_name": request.tool_name}
