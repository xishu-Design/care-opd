from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_doctor
from app.core.responses import success_response
from app.db.models import User
from app.rules.rule_loader import get_rules, load_rules

router = APIRouter(prefix="/api/rules", tags=["rules"])


@router.get("")
async def rules(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response([rule.model_dump() for rule in get_rules(db)])


@router.post("/reload")
async def reload_rules(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    report = load_rules(db)
    db.commit()
    return success_response(report, "rules reloaded")
