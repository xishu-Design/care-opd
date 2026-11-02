from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_doctor, require_patient, require_student
from app.agents.graphs.doctor_review_graph import DoctorReviewGraph
from app.agents.graphs.graph_factory import graph_detail, list_graphs
from app.agents.graphs.patient_education_graph import PatientEducationGraph
from app.agents.graphs.student_training_graph import StudentTrainingGraph
from app.agents.services.resume_service import pending_action_detail, pending_actions_for_doctor, resume_doctor_review
from app.core.responses import success_response
from app.db.models import User
from app.schemas.agent import DoctorReviewAgentRequest, DoctorReviewResumeRequest, PatientEducationAgentRequest, StudentTrainingAgentRequest

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/graphs")
async def graphs(current_user: User = Depends(require_doctor)) -> dict:
    del current_user
    return success_response(list_graphs())


@router.get("/graphs/{graph_name}")
async def graph(graph_name: str, current_user: User = Depends(require_doctor)) -> dict:
    del current_user
    return success_response(graph_detail(graph_name))


@router.post("/doctor/review")
async def doctor_review(
    payload: DoctorReviewAgentRequest,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    return success_response(DoctorReviewGraph(db).invoke(current_user, payload.prescription_id), "agent review completed")


@router.get("/doctor/review/pending")
async def doctor_review_pending(
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    return success_response({"items": pending_actions_for_doctor(db, current_user)})


@router.get("/doctor/review/pending/{trace_id}")
async def doctor_review_pending_detail(
    trace_id: str,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    return success_response(pending_action_detail(db, trace_id, current_user))


@router.post("/doctor/review/{trace_id}/resume")
async def doctor_review_resume(
    trace_id: str,
    payload: DoctorReviewResumeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return success_response(
        resume_doctor_review(db, trace_id=trace_id, approved=payload.approved, comment=payload.comment, doctor=current_user),
        "agent review resumed",
    )


@router.post("/patient/education")
async def patient_education(
    payload: PatientEducationAgentRequest,
    current_user: User = Depends(require_patient),
    db: Session = Depends(get_db),
) -> dict:
    return success_response(PatientEducationGraph(db).invoke(current_user, payload.message), "agent education completed")


@router.post("/student/training")
async def student_training(
    payload: StudentTrainingAgentRequest,
    current_user: User = Depends(require_student),
    db: Session = Depends(get_db),
) -> dict:
    return success_response(StudentTrainingGraph(db).invoke(current_user, payload.learning_case_id), "agent training completed")
