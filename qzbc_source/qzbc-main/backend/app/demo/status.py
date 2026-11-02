from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.demo.service import DemoService


def get_demo_status(db: Session) -> dict[str, Any]:
    return DemoService(db).get_demo_status()

