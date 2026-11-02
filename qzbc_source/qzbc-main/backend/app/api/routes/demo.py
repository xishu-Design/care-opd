from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_doctor
from app.core.responses import success_response
from app.db.models import User
from app.demo.schemas import DemoResetRequest
from app.demo.service import DemoService

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/status")
async def demo_status(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(DemoService(db).get_demo_status())


@router.post("/reset")
async def demo_reset(payload: DemoResetRequest | None = None, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    return success_response(DemoService(db).reset_demo_data(payload or DemoResetRequest(), actor_user_id=current_user.id), "demo data reset")


@router.post("/smoke")
async def demo_smoke(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(DemoService(db).run_demo_smoke().model_dump(mode="json"), "demo smoke completed")
