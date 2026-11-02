from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_doctor
from app.content.services.content_import_service import ContentImportError, ContentImportService
from app.content.services.content_quality_service import content_quality_report
from app.content.services.content_stats_service import content_stats
from app.content.validators.content_validator import ContentValidator
from app.core.responses import success_response
from app.db.models import User

router = APIRouter(prefix="/api/content", tags=["content"])


@router.post("/validate")
async def validate_content(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user, db
    report = ContentValidator().validate().to_dict()
    return success_response(report)


@router.post("/import")
async def import_content(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    try:
        report = ContentImportService(db).import_content()
    except ContentImportError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return success_response(report, "content imported")


@router.get("/stats")
async def stats(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(content_stats(db))


@router.get("/quality-report")
async def quality_report(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(content_quality_report(db))
