from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.demo.schemas import DemoResetRequest
from app.demo.service import DemoService


def reset_demo_data(db: Session, payload: DemoResetRequest | None = None, actor_user_id: int | None = None) -> dict[str, Any]:
    return DemoService(db).reset_demo_data(payload or DemoResetRequest(), actor_user_id=actor_user_id)

