from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import KnowledgeSnapshot


VALID_TRANSITIONS = {
    "STAGING": {"VERIFYING", "FAILED"},
    "VERIFYING": {"ACTIVE", "FAILED"},
    "ACTIVE": set(),
    "FAILED": set(),
}


def create_snapshot(
    db: Session,
    *,
    version: str,
    source_manifest: dict,
    record_counts: dict,
    workspace: str,
    content_dir: Path,
    embedding_model: str = "bge-m3",
    llm_model: str = "qwen2.5:7b",
) -> KnowledgeSnapshot:
    snapshot = KnowledgeSnapshot(
        snapshot_id=f"SNAP_{uuid4().hex[:20].upper()}",
        version=version,
        status="STAGING",
        source_manifest_json=source_manifest,
        content_hash=hash_directory(content_dir),
        record_counts_json=record_counts,
        lightrag_workspace=workspace,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot


def transition_snapshot(db: Session, snapshot_id: str, target: str, *, reason: str = "") -> KnowledgeSnapshot:
    snapshot = db.scalar(select(KnowledgeSnapshot).where(KnowledgeSnapshot.snapshot_id == snapshot_id))
    if snapshot is None:
        raise ValueError(f"snapshot not found: {snapshot_id}")
    if target not in VALID_TRANSITIONS.get(snapshot.status, set()):
        raise ValueError(f"invalid snapshot transition: {snapshot.status} -> {target}")
    if target == "ACTIVE":
        db.execute(update(KnowledgeSnapshot).where(KnowledgeSnapshot.status == "ACTIVE").values(status="SUPERSEDED"))
        snapshot.activated_at = datetime.now(UTC)
    snapshot.status = target
    snapshot.failure_reason = reason if target == "FAILED" else ""
    db.commit()
    db.refresh(snapshot)
    return snapshot


def active_snapshot(db: Session) -> KnowledgeSnapshot | None:
    return db.scalar(select(KnowledgeSnapshot).where(KnowledgeSnapshot.status == "ACTIVE").order_by(KnowledgeSnapshot.activated_at.desc()))


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def snapshot_payload(snapshot: KnowledgeSnapshot) -> dict:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "version": snapshot.version,
        "status": snapshot.status,
        "source_manifest": snapshot.source_manifest_json,
        "content_hash": snapshot.content_hash,
        "record_counts": snapshot.record_counts_json,
        "lightrag_workspace": snapshot.lightrag_workspace,
        "embedding_model": snapshot.embedding_model,
        "llm_model": snapshot.llm_model,
        "created_at": snapshot.created_at.isoformat(),
        "activated_at": snapshot.activated_at.isoformat() if snapshot.activated_at else None,
        "failure_reason": snapshot.failure_reason,
    }
