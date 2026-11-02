import json

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_student
from app.agents.graphs.role_ai_chat_graph import RoleAIChatGraph
from app.agents.graphs.student_training_graph import StudentTrainingGraph
from app.core.responses import success_response
from app.db.models import User
from app.evaluation.private_evidence_control import authorize_private_evidence_control
from app.mcp.contracts import ResourceScope, ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway
from app.schemas.agent import RoleAIChatRequest
from app.schemas.memory import ExplicitMemoryRequest
from app.schemas.learning import StudentAnswerRequest
from app.services.learning_service import (
    build_mtcmb_standard_analysis,
    build_syndrome_standard_analysis,
    generate_mtcmb_teaching_explanation,
    generate_student_submission_feedback,
    get_learning_case,
    get_student_learning_progress,
    is_mtcmb_knowledge_evaluation,
    is_syndrome_differentiation_training,
    list_learning_cases,
    submit_student_answer,
)
from app.services.role_ai_chat_service import delete_role_ai_chat, list_role_ai_chats, save_role_ai_chat
from app.services.memory_consolidation_service import MemoryConsolidationAgent, sync_external_long_term_memory
from app.services.streaming_ai_chat_service import stream_ai_chat_events

router = APIRouter(prefix="/api/student", tags=["student"])


@router.get("/ai-memory")
async def ai_memory(current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    return success_response(MemoryConsolidationAgent(db).list_active(owner_user_id=current_user.id, owner_role="student"))


@router.post("/ai-memory")
async def remember_ai_memory(payload: ExplicitMemoryRequest, current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    result = MemoryConsolidationAgent(db).remember_explicit(owner_user_id=current_user.id, owner_role="student", content=payload.content)
    db.commit()
    if result.memory_id:
        sync_external_long_term_memory(role="student", user_id=current_user.id, memory_id=result.memory_id)
    return success_response({"decision": result.decision.model_dump(), "memory_id": result.memory_id}, "long-term memory saved")


@router.delete("/ai-memory/{memory_id}")
async def forget_ai_memory(memory_id: int, current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    row = MemoryConsolidationAgent(db).forget(owner_user_id=current_user.id, owner_role="student", memory_id=memory_id)
    db.commit()
    sync_external_long_term_memory(role="student", user_id=current_user.id, memory_id=row.id)
    return success_response({"deleted": True, "memory_id": memory_id}, "long-term memory forgotten")


@router.get("/ai-chat/history")
async def ai_chat_history(current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    return success_response(list_role_ai_chats(db, user_id=current_user.id, role="student"))


@router.delete("/ai-chat/{chat_id}")
async def delete_ai_chat(chat_id: int, current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    delete_role_ai_chat(db, user_id=current_user.id, role="student", chat_id=chat_id)
    return success_response({"deleted": True, "chat_id": chat_id}, "chat deleted")


@router.post("/ai-chat")
async def ai_chat(
    payload: RoleAIChatRequest,
    request: Request,
    x_care_private_eval_key: str | None = Header(default=None, alias="X-Care-Private-Eval-Key"),
    current_user: User = Depends(require_student),
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
    chat = save_role_ai_chat(db, user_id=current_user.id, role="student", question=question, result=result)
    result["chat_id"] = chat.id
    result["created_at"] = chat.created_at.isoformat()
    return success_response(result, "student AI chat completed")


@router.post("/ai-chat/stream")
async def ai_chat_stream(
    payload: RoleAIChatRequest,
    current_user: User = Depends(require_student),
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


def serialize_case(case, include_standard: bool = False) -> dict:
    return {
        "id": case.id,
        "title": case.title,
        "anonymized_case": case.anonymized_case,
        "standard_analysis": case.standard_analysis if include_standard else None,
        "risk_points": case.risk_points,
        "evidence_chain": case.evidence_chain,
    }


@router.get("/learning-cases")
async def learning_cases(kind: str | None = None, current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response([serialize_case(case) for case in list_learning_cases(db, kind=kind)])


@router.get("/learning-cases/{case_id}")
async def learning_case(case_id: int, current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(serialize_case(get_learning_case(db, case_id)))


@router.post("/learning-cases/{case_id}/submit-answer")
async def submit_answer(
    case_id: int,
    payload: StudentAnswerRequest,
    current_user: User = Depends(require_student),
    db: Session = Depends(get_db),
) -> dict:
    answer, evaluation = submit_student_answer(
        db,
        current_user,
        case_id,
        payload.answer_text,
        payload.has_risk,
        payload.risk_level,
        payload.proposed_herbs,
    )
    evaluation = generate_student_submission_feedback(db, current_user, case_id, evaluation)
    answer.feedback = evaluation["feedback"]
    db.commit()
    context = ToolContext(
        user_id=current_user.id,
        role=current_user.role,
        task_type="student_training",
        learning_case_id=case_id,
        resource_scope=ResourceScope(is_learning_case_deidentified=True),
    )
    MCPGateway(db).execute(
        ToolRequest(
            tool_name="submit_review_answer",
            arguments={"learning_case_id": case_id, "answer_summary": payload.answer_text},
            context=context,
        )
    )
    return success_response(
        {
            "id": answer.id,
            "learning_case_id": answer.learning_case_id,
            "feedback": answer.feedback,
            "evaluation": evaluation,
        },
        "answer submitted",
    )


@router.get("/learning-cases/{case_id}/standard-analysis")
async def standard_analysis(case_id: int, current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    case = get_learning_case(db, case_id)
    if is_mtcmb_knowledge_evaluation(case.anonymized_case):
        return success_response(build_mtcmb_standard_analysis(case), "MTCMB 题库标准解析已生成")
    if is_syndrome_differentiation_training(case.anonymized_case):
        return success_response(build_syndrome_standard_analysis(case), "TCMEval-SDT 标准辨证解析已生成")
    return success_response(StudentTrainingGraph(db).invoke(current_user, case_id))


@router.post("/learning-cases/{case_id}/teaching-explanation")
async def mtcmb_teaching_explanation(case_id: int, current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    case = get_learning_case(db, case_id)
    return success_response(
        generate_mtcmb_teaching_explanation(db, case, user_id=current_user.id),
        "MTCMB AI 教学讲解已生成",
    )


@router.get("/learning-progress")
async def learning_progress(current_user: User = Depends(require_student), db: Session = Depends(get_db)) -> dict:
    return success_response(get_student_learning_progress(db, current_user))
