from __future__ import annotations

import time
import urllib.error
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import LightRAGIndexJob
from app.db.session import SessionLocal
from app.knowledge.snapshots import active_snapshot
from app.lightrag_integration.client import LightRAGClient, track_state
from app.lightrag_integration.document_builder import build_lightrag_docs, load_lightrag_docs


TERMINAL_TRACK_STATES = {"processed", "completed", "complete", "done", "success", "failed", "error", "cancelled", "canceled"}


def create_index_job(db: Session, *, snapshot_id: str | None = None) -> LightRAGIndexJob:
    snapshot = active_snapshot(db)
    resolved_snapshot = snapshot_id or (snapshot.snapshot_id if snapshot else "UNVERSIONED")
    job = LightRAGIndexJob(run_id=f"lightrag_{uuid4().hex[:20]}", snapshot_id=resolved_snapshot, status="queued", stage="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def run_index_job(run_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.scalar(select(LightRAGIndexJob).where(LightRAGIndexJob.run_id == run_id))
        if not job or job.status == "cancelled":
            return
        _update(db, job, status="preparing", stage="preparing", started_at=datetime.now(UTC), progress=0.05)
        client = LightRAGClient()
        compatibility = client.openapi_compatibility()
        if not compatibility.get("compatible"):
            raise RuntimeError(f"LightRAG OpenAPI incompatible: {compatibility}")
        build_lightrag_docs()
        docs = load_lightrag_docs()
        job.total_documents = len(docs)
        _update(db, job, status="uploading", stage="uploading", progress=0.1)
        track_ids = []
        for offset in range(0, len(docs), 20):
            current = _reload(db, run_id)
            if current.status == "cancelled":
                return
            batch = docs[offset : offset + 20]
            batch_track_ids, _ = _upload_batch_idempotently(client, batch)
            track_ids.extend(batch_track_ids)
            current.track_ids_json = track_ids
            current.status = "extracting"
            current.stage = "extracting"
            current.updated_at = datetime.now(UTC)
            db.commit()
            for track_id in batch_track_ids:
                _wait_track(client, track_id, db, run_id)
            current = _reload(db, run_id)
            if current.status == "cancelled":
                return
            current.status = "extracting"
            current.stage = "extracting"
            current.processed_documents = min(offset + len(batch), len(docs))
            current.progress = 0.1 + 0.75 * current.processed_documents / max(len(docs), 1)
            current.updated_at = datetime.now(UTC)
            db.commit()
        job = _reload(db, run_id)
        _update(db, job, status="verifying", stage="verifying", progress=0.9)
        for query in ("黄芪", "酸枣仁汤"):
            response = client.query_data(query, mode="mix", top_k=5)
            if not response:
                raise RuntimeError(f"fixed query returned no data: {query}")
        job = _reload(db, run_id)
        _update(db, job, status="ready", stage="ready", progress=1.0, completed_at=datetime.now(UTC))
    except Exception as exc:
        job = db.scalar(select(LightRAGIndexJob).where(LightRAGIndexJob.run_id == run_id))
        if job:
            _update(db, job, status="failed", stage="failed", error_message=str(exc), completed_at=datetime.now(UTC))
    finally:
        db.close()


def job_payload(job: LightRAGIndexJob) -> dict:
    return {
        "run_id": job.run_id,
        "snapshot_id": job.snapshot_id,
        "status": job.status,
        "stage": job.stage,
        "track_ids": job.track_ids_json,
        "total_documents": job.total_documents,
        "processed_documents": job.processed_documents,
        "failed_documents": job.failed_documents,
        "progress": job.progress,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "updated_at": job.updated_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message,
    }


def _wait_track(client: LightRAGClient, track_id: str, db: Session, run_id: str) -> None:
    for _ in range(720):
        job = _reload(db, run_id)
        if job.status == "cancelled":
            return
        payload = client.track_status(track_id)
        status = track_state(payload)
        if status in TERMINAL_TRACK_STATES:
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"LightRAG track failed: {track_id}: {payload}")
            return
        time.sleep(5)
    raise TimeoutError(f"LightRAG track timed out: {track_id}")


def _track_id(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    return str(payload.get("track_id") or payload.get("id") or data.get("track_id") or data.get("id") or "")


def _upload_batch_idempotently(client: LightRAGClient, docs: list) -> tuple[list[str], int]:
    try:
        response = client.upload_texts([doc.text for doc in docs], [doc.doc_id for doc in docs])
        track_id = _track_id(response)
        if not track_id:
            raise RuntimeError(f"LightRAG did not return track_id: {response}")
        return [track_id], 0
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise

    track_ids: list[str] = []
    reused = 0
    for doc in docs:
        try:
            response = client.upload_texts([doc.text], [doc.doc_id])
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                reused += 1
                continue
            raise
        track_id = _track_id(response)
        if not track_id:
            raise RuntimeError(f"LightRAG did not return track_id for {doc.doc_id}: {response}")
        track_ids.append(track_id)
    return track_ids, reused


def _reload(db: Session, run_id: str) -> LightRAGIndexJob:
    db.expire_all()
    job = db.scalar(select(LightRAGIndexJob).where(LightRAGIndexJob.run_id == run_id))
    if job is None:
        raise RuntimeError(f"LightRAG job missing: {run_id}")
    return job


def _update(db: Session, job: LightRAGIndexJob, **values: object) -> None:
    for key, value in values.items():
        setattr(job, key, value)
    job.updated_at = datetime.now(UTC)
    db.commit()
