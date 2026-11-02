from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_doctor
from app.core.responses import success_response
from app.db.models import AblationRun, EvaluationRun, User
from app.evaluation.ablation import AblationRunner
from app.evaluation.dataset_loader import dataset_manifest
from app.evaluation.report import evaluation_results, serialize_evaluation_run
from app.evaluation.runner import EvaluationRunner
from app.evaluation.schemas import AblationRunRequest, EvaluationRunRequest


router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/datasets")
async def datasets(current_user: User = Depends(require_doctor)) -> dict:
    del current_user
    return success_response(dataset_manifest())


@router.post("/run")
async def run_evaluation(payload: EvaluationRunRequest, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    run = EvaluationRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data, actor_user_id=current_user.id).run()
    return success_response(serialize_evaluation_run(run), "evaluation run completed")


@router.get("/runs")
async def list_evaluation_runs(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    runs = db.scalars(select(EvaluationRun).order_by(EvaluationRun.created_at.desc())).all()
    return success_response([serialize_evaluation_run(run, include_report=False) for run in runs])


@router.get("/runs/{run_id}")
async def get_evaluation_run(run_id: str, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    run = db.scalar(select(EvaluationRun).where(EvaluationRun.run_id == run_id))
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found")
    payload = serialize_evaluation_run(run)
    payload["results"] = evaluation_results(db, run)
    return success_response(payload)


@router.get("/runs/{run_id}/report")
async def get_evaluation_report(run_id: str, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    run = db.scalar(select(EvaluationRun).where(EvaluationRun.run_id == run_id))
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found")
    payload = dict(run.report_json)
    payload["markdown_report"] = run.markdown_report
    return success_response(payload)


@router.post("/ablation/run")
async def run_ablation(payload: AblationRunRequest, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    try:
        report = AblationRunner(db).run(suite=payload.suite, baseline_run_id=payload.baseline_run_id, actor_user_id=current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return success_response(report, "ablation run completed")


@router.get("/ablation/latest")
async def latest_ablation(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    run = db.scalar(select(AblationRun).order_by(AblationRun.created_at.desc()))
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ablation report found")
    return success_response(run.report_json)
