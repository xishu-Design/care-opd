from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DoctorPatientBinding, LearningCase
from app.mcp.contracts import ToolRequest
from app.services.learning_service import is_student_learning_case


def execute(db: Session, request: ToolRequest) -> dict:
    if request.tool_name == "check_doctor_patient_binding":
        patient_id = request.arguments.get("patient_id") or request.context.patient_id
        binding = db.scalar(
            select(DoctorPatientBinding).where(
                DoctorPatientBinding.doctor_id == request.context.user_id,
                DoctorPatientBinding.patient_id == patient_id,
                DoctorPatientBinding.is_active.is_(True),
            )
        )
        return {"is_bound": bool(binding)}
    if request.tool_name == "check_patient_ownership":
        patient_id = request.arguments.get("patient_id") or request.context.patient_id or request.context.user_id
        return {"is_patient_self": int(patient_id) == request.context.user_id}
    if request.tool_name == "check_learning_case_access":
        case_id = request.arguments.get("learning_case_id") or request.context.learning_case_id
        return {"is_learning_case_deidentified": is_student_learning_case(db.get(LearningCase, case_id))}
    return {"role": request.context.role, "verified": True}
