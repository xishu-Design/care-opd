import hashlib
import json

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_doctor
from app.agents.graphs.role_ai_chat_graph import RoleAIChatGraph
from app.core.responses import success_response
from app.db.models import User
from app.evaluation.private_evidence_control import authorize_private_evidence_control
from app.evaluation.public_teaching_case_control import authorize_public_teaching_case
from app.schemas.agent import DoctorCaseReasoningRequest, RoleAIChatRequest
from app.schemas.memory import ExplicitMemoryRequest
from app.schemas.patient import PatientCreateRequest, PatientUpdateRequest
from app.schemas.prescription import PrescriptionCreateRequest, PrescriptionSuggestionRequest, PrescriptionUpdateRequest
from app.services.learning_service import create_deidentified_learning_case
from app.services.patient_service import create_or_bind_patient, get_patient_for_doctor, list_bound_patients, update_patient_for_doctor
from app.services.prescription_service import (
    confirm_prescription,
    create_draft_prescription,
    get_prescription_for_doctor,
    publish_prescription,
    review_prescription,
    serialize_prescription,
    update_draft_prescription,
)
from app.services.prescription_suggestion_service import suggest_prescription_candidates
from app.services.role_ai_chat_service import delete_role_ai_chat, list_role_ai_chats, save_role_ai_chat
from app.services.memory_consolidation_service import MemoryConsolidationAgent, sync_external_long_term_memory
from app.services.streaming_ai_chat_service import stream_ai_chat_events

router = APIRouter(prefix="/api/doctor", tags=["doctor"])


@router.get("/ai-memory")
async def ai_memory(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    return success_response(MemoryConsolidationAgent(db).list_active(owner_user_id=current_user.id, owner_role="doctor"))


@router.post("/ai-memory")
async def remember_ai_memory(payload: ExplicitMemoryRequest, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    result = MemoryConsolidationAgent(db).remember_explicit(owner_user_id=current_user.id, owner_role="doctor", content=payload.content)
    db.commit()
    if result.memory_id:
        sync_external_long_term_memory(role="doctor", user_id=current_user.id, memory_id=result.memory_id)
    return success_response({"decision": result.decision.model_dump(), "memory_id": result.memory_id}, "long-term memory saved")


@router.delete("/ai-memory/{memory_id}")
async def forget_ai_memory(memory_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    row = MemoryConsolidationAgent(db).forget(owner_user_id=current_user.id, owner_role="doctor", memory_id=memory_id)
    db.commit()
    sync_external_long_term_memory(role="doctor", user_id=current_user.id, memory_id=row.id)
    return success_response({"deleted": True, "memory_id": memory_id}, "long-term memory forgotten")


@router.get("/ai-chat/history")
async def ai_chat_history(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    return success_response(list_role_ai_chats(db, user_id=current_user.id, role="doctor"))


@router.delete("/ai-chat/{chat_id}")
async def delete_ai_chat(chat_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    delete_role_ai_chat(db, user_id=current_user.id, role="doctor", chat_id=chat_id)
    return success_response({"deleted": True, "chat_id": chat_id}, "chat deleted")


@router.post("/ai-chat")
async def ai_chat(
    payload: RoleAIChatRequest,
    request: Request,
    x_care_private_eval_key: str | None = Header(default=None, alias="X-Care-Private-Eval-Key"),
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    question = payload.message.strip()
    control = authorize_private_evidence_control(
        control=payload.private_eval_evidence_control,
        message=question,
        supplied_key=x_care_private_eval_key,
        client_host=request.client.host if request.client else None,
    )
    result = RoleAIChatGraph(db).invoke(
        current_user,
        question,
        private_eval_evidence_control=control,
    )
    chat = save_role_ai_chat(db, user_id=current_user.id, role="doctor", question=question, result=result)
    result["chat_id"] = chat.id
    result["created_at"] = chat.created_at.isoformat()
    return success_response(result, "doctor AI chat completed")


@router.post("/case-reasoning")
async def case_reasoning(
    payload: DoctorCaseReasoningRequest,
    request: Request,
    x_care_case_reasoning_key: str | None = Header(
        default=None, alias="X-Care-Case-Reasoning-Key"
    ),
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    """Run a doctor-internal teaching-case analysis with an auditable fact boundary."""

    question = payload.message.strip()
    case_material = payload.case_material.strip()
    if case_material != payload.case_material:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="case_material must be canonical text without surrounding whitespace",
        )
    observed_sha256 = hashlib.sha256(case_material.encode("utf-8")).hexdigest()
    if observed_sha256 != payload.case_material_sha256:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="case_material_sha256 mismatch",
        )
    if case_material not in question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="message is not bound to the complete case_material",
        )
    context = authorize_public_teaching_case(
        payload=payload,
        message=question,
        supplied_key=x_care_case_reasoning_key,
        client_host=request.client.host if request.client else None,
    )
    result = RoleAIChatGraph(db).invoke(
        current_user,
        question,
        case_reasoning_context=context,
    )
    chat = save_role_ai_chat(
        db,
        user_id=current_user.id,
        role="doctor",
        question=question,
        result=result,
    )
    result["chat_id"] = chat.id
    result["created_at"] = chat.created_at.isoformat()
    return success_response(result, "doctor teaching-case reasoning completed")


@router.post("/ai-chat/stream")
async def ai_chat_stream(
    payload: RoleAIChatRequest,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    if payload.private_eval_evidence_control is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="private evaluation evidence control requires the non-streaming endpoint",
        )
    return StreamingResponse(
        _stream_events(db, current_user, payload.message.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _stream_events(db: Session, user: User, message: str):
    async for event, data in stream_ai_chat_events(db, user, message):
        yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/patients")
async def list_patients(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    return success_response([patient.model_dump() for patient in list_bound_patients(db, current_user.id)])


@router.post("/patients")
async def create_patient(payload: PatientCreateRequest, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    patient = create_or_bind_patient(db, current_user, payload)
    return success_response(patient.model_dump(), "patient bound")


@router.get("/patients/{patient_id}")
async def get_patient(patient_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    return success_response(get_patient_for_doctor(db, current_user.id, patient_id).model_dump())


@router.patch("/patients/{patient_id}")
async def update_patient(
    patient_id: int,
    payload: PatientUpdateRequest,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    patient = update_patient_for_doctor(db, current_user.id, patient_id, payload)
    return success_response(patient.model_dump(), "patient updated")


@router.post("/prescriptions")
async def create_prescription(payload: PrescriptionCreateRequest, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    prescription = create_draft_prescription(db, current_user, payload)
    return success_response(serialize_prescription(prescription), "draft created")


@router.post("/prescription-suggestions")
def prescription_suggestions(payload: PrescriptionSuggestionRequest, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    return success_response(
        suggest_prescription_candidates(
            db,
            current_user,
            patient_id=payload.patient_id,
            symptoms=payload.symptoms,
            query=payload.query,
        )
    )


@router.get("/prescriptions/{prescription_id}")
async def get_prescription(prescription_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    prescription = get_prescription_for_doctor(db, current_user.id, prescription_id)
    return success_response(serialize_prescription(prescription))


@router.patch("/prescriptions/{prescription_id}")
async def update_prescription(
    prescription_id: int,
    payload: PrescriptionUpdateRequest,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    prescription = update_draft_prescription(db, current_user, prescription_id, payload)
    return success_response(serialize_prescription(prescription), "prescription updated; rerun review")


@router.post("/prescriptions/{prescription_id}/review")
async def review(prescription_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    return success_response(review_prescription(db, current_user, prescription_id), "review completed")


@router.post("/prescriptions/{prescription_id}/confirm")
async def confirm(prescription_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    prescription = confirm_prescription(db, current_user, prescription_id)
    return success_response(serialize_prescription(prescription), "doctor confirmed")


@router.post("/prescriptions/{prescription_id}/publish")
async def publish(prescription_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    prescription = publish_prescription(db, current_user, prescription_id)
    return success_response(serialize_prescription(prescription), "published to patient")


@router.post("/prescriptions/{prescription_id}/deidentify")
async def deidentify(prescription_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    case = create_deidentified_learning_case(db, current_user, prescription_id)
    return success_response(
        {
            "id": case.id,
            "title": case.title,
            "anonymized_case": case.anonymized_case,
            "standard_analysis": case.standard_analysis,
            "risk_points": case.risk_points,
            "evidence_chain": case.evidence_chain,
        },
        "learning case created",
    )
