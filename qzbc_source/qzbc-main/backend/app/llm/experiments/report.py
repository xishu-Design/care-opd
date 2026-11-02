from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import LLMExperimentRun


def latest_experiment_report(db: Session) -> dict[str, Any] | None:
    run = db.scalar(select(LLMExperimentRun).order_by(LLMExperimentRun.created_at.desc()))
    return run.report_json if run else None


def get_experiment_report(db: Session, experiment_id: str) -> dict[str, Any] | None:
    run = db.scalar(select(LLMExperimentRun).where(LLMExperimentRun.experiment_id == experiment_id))
    return run.report_json if run else None
