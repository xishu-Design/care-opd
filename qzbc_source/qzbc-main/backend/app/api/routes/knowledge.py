from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_doctor
from app.core.responses import success_response
from app.db.models import User
from app.core.config import get_settings
from app.knowledge.internal_knowledge import import_shennong_jsonl
from app.knowledge.ingestion import get_knowledge_entry, ingest_local_knowledge, knowledge_health, list_knowledge_entries, list_sources, search_knowledge

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/health")
async def health(db: Session = Depends(get_db)) -> dict:
    return success_response(knowledge_health(db))


@router.get("/sources")
async def sources(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    del current_user
    return success_response(list_sources(db))


@router.get("/entries")
async def entries(
    entry_type: str | None = None,
    q: str | None = Query(default=None, max_length=120),
    source_id: str | None = Query(default=None, max_length=120),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _require_knowledge_viewer(current_user)
    return success_response(
        list_knowledge_entries(
            db,
            role=current_user.role,
            entry_type=entry_type,
            query=q,
            source_id=source_id,
            page=page,
            page_size=page_size,
        )
    )


@router.get("/entries/{source_id}/{entry_id}")
async def entry_detail(
    source_id: str,
    entry_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _require_knowledge_viewer(current_user)
    entry = get_knowledge_entry(db, role=current_user.role, source_id=source_id, entry_id=entry_id)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge entry not found")
    return success_response(entry)


@router.post("/reindex")
async def reindex(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    report = ingest_local_knowledge(db)
    db.commit()
    return success_response(report, "knowledge reindexed")


@router.post("/internal/shennong/import")
async def import_shennong(
    payload: dict,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    del current_user
    settings = get_settings()
    configured_root = Path(settings.internal_knowledge_dir).resolve()
    path = Path(str(payload.get("path", ""))).resolve()
    if configured_root not in path.parents or path.suffix not in {".json", ".jsonl"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Internal source path is invalid")
    revision = str(payload.get("revision", "")).strip()
    if not revision:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Immutable source revision is required")
    report = import_shennong_jsonl(db, path=path, revision=revision, allow_internal=settings.allow_unverified_internal_sources)
    db.commit()
    return success_response(report, "ShenNong internal knowledge imported")


@router.get("/search")
async def search(
    query: str = Query(..., min_length=1),
    entry_type: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    evidence = search_knowledge(
        db,
        query=query,
        entry_type=entry_type,
        retrieved_by="api_knowledge_search",
        log_request={"session_id": "api", "role": current_user.role, "task_type": "knowledge_search"},
    )
    db.commit()
    return success_response({"evidence": evidence, "degraded": not evidence, "degraded_reason": "no_evidence_found" if not evidence else None})


def _require_knowledge_viewer(user: User) -> None:
    if user.role not in {"doctor", "student"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor or student role required")
