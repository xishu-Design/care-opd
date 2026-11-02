from sqlalchemy.orm import Session

from app.db.models import HarnessRun
from app.harness.phase8_runner import Phase8HarnessRunner


def run_harness_summary(db: Session) -> HarnessRun:
    return Phase8HarnessRunner(db, suite="full", reset_demo_data=True).run()
