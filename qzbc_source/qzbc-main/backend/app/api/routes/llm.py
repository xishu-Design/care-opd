from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_doctor
from app.core.responses import success_response
from app.db.models import User
from app.llm.experiments.report import get_experiment_report, latest_experiment_report
from app.llm.experiments.runner import LLMExperimentRunner
from app.llm.schemas import LLMRequest
from app.llm.service import LLMService

router = APIRouter(prefix="/api/llm", tags=["llm"])


class TestGenerateRequest(BaseModel):
    role: str = "patient"
    task_type: str = "patient_education"
    message: str = ""
    evidence_packet: dict[str, Any] | None = None


class ExperimentRunRequest(BaseModel):
    provider: str = "ollama"
    model: str | None = None
    suite: str = "quick"


@router.get("/health")
async def llm_health(db: Session = Depends(get_db)) -> dict:
    return success_response(LLMService(db).health())


@router.get("/providers")
async def llm_providers(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response({"items": LLMService(db).providers()})


@router.post("/test-generate")
async def test_generate(payload: TestGenerateRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    evidence_packet = payload.evidence_packet or {
        "packet_id": "dev_test_packet",
        "visibility": "patient_safe" if payload.role == "patient" else "student_learning" if payload.role == "student" else "doctor_internal",
        "retrieved_evidence": [{"source_id": "dev_source", "summary": "用于开发测试的本地模板证据。"}],
        "rule_findings": [],
        "graph_reasoning_path": [],
        "tool_calls": [],
    }
    request = LLMRequest(
        role=payload.role,  # type: ignore[arg-type]
        task_type=payload.task_type,  # type: ignore[arg-type]
        input_text=payload.message,
        evidence_packet=evidence_packet,
        trace_id=None,
        user_id=current_user.id,
    )
    return success_response(LLMService(db).generate(request).model_dump(mode="json"))


@router.post("/experiments/run")
async def run_llm_experiment(payload: ExperimentRunRequest, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    try:
        report = LLMExperimentRunner(db).run(provider=payload.provider, model=payload.model, suite=payload.suite, actor_user_id=current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return success_response(report)


@router.get("/experiments/latest")
async def latest_llm_experiment(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    report = latest_experiment_report(db)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No LLM experiment report found")
    return success_response(report)


@router.get("/experiments/{experiment_id}")
async def get_llm_experiment(experiment_id: str, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    report = get_experiment_report(db, experiment_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM experiment report not found")
    return success_response(report)
