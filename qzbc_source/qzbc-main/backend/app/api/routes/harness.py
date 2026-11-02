from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_doctor
from app.core.responses import success_response
from app.db.models import HarnessRun, User
from app.harness.case_schema import PHASE4, PHASE5, PHASE6, PHASE8, PHASE9, PHASE10, PHASE11, PHASE12
from app.harness.phase4_runner import Phase4HarnessRunner, build_harness_report
from app.harness.phase5_runner import Phase5HarnessRunner
from app.harness.phase6_runner import Phase6HarnessRunner
from app.harness.phase8_runner import Phase8HarnessRunner
from app.harness.phase9_runner import Phase9HarnessRunner
from app.harness.phase10_runner import Phase10HarnessRunner
from app.harness.phase11_runner import Phase11HarnessRunner
from app.harness.phase9_reports import phase9_data_pipeline_report, phase9_source_registry_report
from app.harness.trace import get_trace_detail, list_traces
from app.schemas.harness import HarnessRunRequest
from app.evaluation.report import serialize_evaluation_run
from app.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/api/harness", tags=["harness"])


def serialize_run(run: HarnessRun) -> dict:
    return {
        "run_id": run.id,
        "phase": run.phase,
        "suite": run.suite,
        "status": run.status,
        "total_cases": run.total_cases,
        "passed": run.passed,
        "failed": run.failed,
        "pass_rate": run.pass_rate,
        "categories": run.categories,
        "metrics": run.metrics_json,
        "markdown_summary": run.markdown_summary,
        "duration_ms": run.duration_ms,
    }


@router.post("/run")
async def run_harness(payload: HarnessRunRequest | None = None, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    payload = payload or HarnessRunRequest()
    if payload.phase == PHASE12:
        evaluation = EvaluationRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data, actor_user_id=current_user.id).run()
        return success_response(serialize_evaluation_run(evaluation), "phase 12 evaluation completed")
    if payload.phase == PHASE4:
        run = Phase4HarnessRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data).run()
    elif payload.phase == PHASE5:
        run = Phase5HarnessRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data).run()
    elif payload.phase == PHASE6:
        run = Phase6HarnessRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data).run()
    elif payload.phase == PHASE8:
        run = Phase8HarnessRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data).run()
    elif payload.phase == PHASE9:
        run = Phase9HarnessRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data).run()
    elif payload.phase == PHASE10:
        run = Phase10HarnessRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data).run()
    elif payload.phase == PHASE11:
        run = Phase11HarnessRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data).run()
    else:
        run = Phase11HarnessRunner(db, suite=payload.suite, reset_demo_data=payload.reset_demo_data).run()
    return success_response(serialize_run(run), "harness run completed")


@router.get("/runs")
async def list_runs(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    runs = db.scalars(select(HarnessRun).order_by(HarnessRun.created_at.desc())).all()
    return success_response([serialize_run(run) for run in runs])


@router.get("/runs/{run_id}")
async def get_run(run_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    run = db.get(HarnessRun, run_id)
    return success_response(serialize_run(run) if run else {})


@router.get("/runs/{run_id}/report")
async def get_run_report(run_id: int, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    run = db.get(HarnessRun, run_id)
    return success_response(build_harness_report(db, run) if run else {})


@router.get("/sources")
async def source_registry(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(phase9_source_registry_report(db))


@router.get("/data-pipeline")
async def data_pipeline(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(phase9_data_pipeline_report(db))


@router.get("/traces")
async def traces(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response({"items": list_traces(db)})


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(get_trace_detail(db, trace_id))
