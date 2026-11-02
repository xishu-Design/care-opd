from __future__ import annotations

from sqlalchemy.orm import Session

from app.demo.schemas import DemoSmokeResult
from app.demo.service import DemoService


def run_demo_smoke(db: Session) -> DemoSmokeResult:
    return DemoService(db).run_demo_smoke()
