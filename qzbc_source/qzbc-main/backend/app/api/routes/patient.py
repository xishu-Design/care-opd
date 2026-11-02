import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_patient
from app.agents.graphs.patient_education_graph import PatientEducationGraph
from app.core.responses import success_response
from app.db.models import User
from app.evaluation.private_evidence_control import authorize_private_evidence_control
from app.schemas.agent import PrivateEvalEvidenceControlRequest
from app.schemas.memory import ExplicitMemoryRequest
from app.services.memory_consolidation_service import MemoryConsolidationAgent, sync_external_long_term_memory
from app.services.patient_service import get_own_patient_profile
from app.services.patient_education_history_service import delete_education_chat, display_answer, list_education_chats, save_education_chat
from app.services.prescription_service import get_patient_visible_prescription, list_patient_visible_prescriptions
from app.services.streaming_ai_chat_service import stream_ai_chat_events

router = APIRouter(prefix="/api/patient", tags=["patient"])


class EducationChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    private_eval_evidence_control: PrivateEvalEvidenceControlRequest | None = None


@router.get("/ai-memory")
async def ai_memory(current_user: User = Depends(require_patient), db: Session = Depends(get_db)) -> dict:
    return success_response(MemoryConsolidationAgent(db).list_active(owner_user_id=current_user.id, owner_role="patient"))


@router.post("/ai-memory")
async def remember_ai_memory(payload: ExplicitMemoryRequest, current_user: User = Depends(require_patient), db: Session = Depends(get_db)) -> dict:
    result = MemoryConsolidationAgent(db).remember_explicit(owner_user_id=current_user.id, owner_role="patient", content=payload.content)
    db.commit()
    if result.memory_id:
        sync_external_long_term_memory(role="patient", user_id=current_user.id, memory_id=result.memory_id)
    return success_response({"decision": result.decision.model_dump(), "memory_id": result.memory_id}, "long-term memory saved")


@router.delete("/ai-memory/{memory_id}")
async def forget_ai_memory(memory_id: int, current_user: User = Depends(require_patient), db: Session = Depends(get_db)) -> dict:
    row = MemoryConsolidationAgent(db).forget(owner_user_id=current_user.id, owner_role="patient", memory_id=memory_id)
    db.commit()
    sync_external_long_term_memory(role="patient", user_id=current_user.id, memory_id=row.id)
    return success_response({"deleted": True, "memory_id": memory_id}, "long-term memory forgotten")


@router.get("/profile")
async def profile(current_user: User = Depends(require_patient), db: Session = Depends(get_db)) -> dict:
    return success_response(get_own_patient_profile(db, current_user.id).model_dump())


@router.get("/prescriptions")
async def prescriptions(current_user: User = Depends(require_patient), db: Session = Depends(get_db)) -> dict:
    return success_response(list_patient_visible_prescriptions(db, current_user.id))


@router.get("/prescriptions/{prescription_id}")
async def prescription_detail(prescription_id: int, current_user: User = Depends(require_patient), db: Session = Depends(get_db)) -> dict:
    return success_response(get_patient_visible_prescription(db, current_user.id, prescription_id))


@router.get("/education-chat/history")
async def education_chat_history(current_user: User = Depends(require_patient), db: Session = Depends(get_db)) -> dict:
    return success_response(list_education_chats(db, patient_id=current_user.id))


@router.delete("/education-chat/{chat_id}")
async def delete_education_chat_record(
    chat_id: int,
    current_user: User = Depends(require_patient),
    db: Session = Depends(get_db),
) -> dict:
    delete_education_chat(db, patient_id=current_user.id, chat_id=chat_id)
    return success_response({"deleted": True, "chat_id": chat_id}, "chat deleted")


@router.post("/education-chat")
async def education_chat(
    payload: EducationChatRequest,
    request: Request,
    x_care_private_eval_key: str | None = Header(default=None, alias="X-Care-Private-Eval-Key"),
    current_user: User = Depends(require_patient),
    db: Session = Depends(get_db),
) -> dict:
    control = authorize_private_evidence_control(
        control=payload.private_eval_evidence_control,
        message=payload.message,
        supplied_key=x_care_private_eval_key,
        client_host=request.client.host if request.client else None,
    )
    result = PatientEducationGraph(db).invoke(
        current_user,
        payload.message,
        private_eval_evidence_control=control,
    )
    save_education_chat(db, patient_id=current_user.id, question=payload.message, result=result)
    return success_response(result)


@router.post("/education-chat/stream")
async def education_chat_stream(
    payload: EducationChatRequest,
    current_user: User = Depends(require_patient),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Stream patient-safe education copy while preserving the existing graph audit trail."""
    if payload.private_eval_evidence_control is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="private evaluation evidence control requires the non-streaming endpoint",
        )
    return StreamingResponse(
        _education_stream(db, current_user.id, payload.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _education_stream(db: Session, user_id: int, message: str) -> AsyncIterator[str]:
    patient = db.get(User, user_id)
    if patient is None or patient.role != "patient" or not patient.is_active:
        yield _sse("error", {"message": "登录状态已失效，请重新登录。"})
        return
    async for event, data in stream_ai_chat_events(db, patient, message):
        yield _sse(event, data)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
