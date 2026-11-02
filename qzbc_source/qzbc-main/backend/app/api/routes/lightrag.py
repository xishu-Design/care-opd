from __future__ import annotations

import logging
from threading import Thread

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_doctor
from app.core.responses import success_response
from app.db.models import LightRAGIndexJob, User
from app.knowledge.snapshots import active_snapshot
from app.lightrag_integration.client import LightRAGClient
from app.lightrag_integration.config import DEFAULT_REPORTS_DIR, get_lightrag_config
from app.lightrag_integration.document_builder import build_lightrag_docs
from app.lightrag_integration.health import lightrag_health
from app.lightrag_integration.jobs import create_index_job, job_payload, run_index_job
from app.lightrag_integration.index_builder import build_lightrag_index, read_index_report
from app.lightrag_integration.local_graph import local_graph, local_labels
from app.lightrag_integration.provenance import ProvenanceResolver


router = APIRouter(prefix="/api/lightrag", tags=["lightrag"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    result = lightrag_health()
    result["openapi"] = LightRAGClient().openapi_compatibility() if get_lightrag_config().enabled else {"compatible": False, "reason": "disabled"}
    snapshot = active_snapshot(db)
    result["snapshot_id"] = snapshot.snapshot_id if snapshot else ""
    return success_response(result)


@router.post("/build-docs")
async def build_docs(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user, db
    report_dir = _runtime_report_dir()
    return success_response(build_lightrag_docs(report_path=report_dir / "lightrag_doc_build_report.json"), "LightRAG 文档已生成")


@router.post("/reindex")
async def reindex(
    snapshot_id: str | None = None,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    del current_user
    if get_lightrag_config().mode != "api":
        report_dir = _runtime_report_dir()
        return success_response(build_lightrag_index(report_path=report_dir / "lightrag_index_report.json", rebuild=True), "本地兼容索引已生成")
    job = create_index_job(db, snapshot_id=snapshot_id)
    Thread(target=run_index_job, args=(job.run_id,), daemon=True, name=f"lightrag-{job.run_id}").start()
    return success_response(job_payload(job), "LightRAG 索引任务已创建")


@router.get("/jobs")
async def list_jobs(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    rows = db.scalars(select(LightRAGIndexJob).order_by(LightRAGIndexJob.updated_at.desc()).limit(50)).all()
    return success_response({"items": [job_payload(row) for row in rows]})


@router.get("/jobs/{run_id}")
async def get_job(run_id: str, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    row = db.scalar(select(LightRAGIndexJob).where(LightRAGIndexJob.run_id == run_id))
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="索引任务不存在")
    return success_response(job_payload(row))


@router.post("/jobs/{run_id}/cancel")
async def cancel_job(run_id: str, current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    row = db.scalar(select(LightRAGIndexJob).where(LightRAGIndexJob.run_id == run_id))
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="索引任务不存在")
    if row.status not in {"ready", "failed", "cancelled"}:
        row.status = "cancelled"
        row.stage = "cancelled"
        db.commit()
        db.refresh(row)
    return success_response(job_payload(row), "索引任务已取消")


@router.get("/graph/labels")
async def graph_labels(
    q: str = "",
    limit: int = Query(50, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    role = _graph_role(current_user)
    return success_response(
        {"items": local_labels(query=q, role=role, limit=limit, db=db), "backend": "local_knowledge", "degraded": True, "fallback_used": True}
    )


@router.get("/graph")
async def graph(
    label: str = "",
    max_depth: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(200, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    role = _graph_role(current_user)
    snapshot = active_snapshot(db)
    deterministic = local_graph(label=label, role=role, max_depth=max_depth, max_nodes=max_nodes, db=db)
    config = get_lightrag_config()
    if not config.enabled:
        deterministic.meta.update(
            {
                "lightrag_status": "disabled",
                "degraded_reason": "lightrag_disabled",
                "snapshot_id": snapshot.snapshot_id if snapshot else "",
            }
        )
        return success_response(deterministic.model_dump(mode="json"))
    if label.strip() and not deterministic.nodes:
        # The visual graph intentionally exposes only approved, role-visible
        # entities. Do not wait for an external graph request when the query is
        # outside that catalog; return a normal empty result instead of making
        # the browser hit its request timeout.
        deterministic.meta.update(
            {
                "lightrag_status": "skipped_no_local_anchor",
                "degraded_reason": "graph_label_not_found",
                "snapshot_id": snapshot.snapshot_id if snapshot else "",
            }
        )
        return success_response(deterministic.model_dump(mode="json"))
    try:
        raw = LightRAGClient().graph(label, max_depth, max_nodes)
        raw_node_count = _graph_item_count(raw, "nodes", "vertices", "entities")
        raw_edge_count = _graph_item_count(raw, "edges", "links", "relationships")
        result = ProvenanceResolver(
            db=db,
            snapshot_id=snapshot.snapshot_id if snapshot else "",
        ).resolve_graph(
            raw,
            role=role,
            max_depth=max_depth,
            max_nodes=max_nodes,
            snapshot_id=snapshot.snapshot_id if snapshot else "",
        )
        if result.nodes and deterministic.nodes:
            deterministic.meta.update(
                {
                    "backend": "lightrag",
                    "degraded": False,
                    "fallback_used": False,
                    "snapshot_id": snapshot.snapshot_id if snapshot else "",
                    "lightrag_anchor_nodes": len(result.nodes),
                    "lightrag_raw_nodes": raw_node_count,
                    "lightrag_raw_edges": raw_edge_count,
                    "lightrag_status": "available",
                    "relationship_backend": "local_knowledge",
                    "relationship_policy": "approved_deterministic_only",
                }
            )
            deterministic.meta.pop("degraded_reason", None)
            if raw_edge_count == 0:
                deterministic.meta.update(
                    {
                        "backend": "lightrag_hybrid",
                        "fallback_used": True,
                        "degraded_reason": "lightrag_no_relationships_for_query",
                    }
                )
            return success_response(deterministic.model_dump(mode="json"))
        deterministic.meta.update(
            {
                "lightrag_status": "available",
                "lightrag_raw_nodes": raw_node_count,
                "lightrag_raw_edges": raw_edge_count,
                "degraded_reason": "lightrag_results_filtered_by_provenance",
            }
        )
    except Exception as exc:
        logger.warning("LightRAG graph query failed for label=%r: %s", label, exc)
        deterministic.meta.update(
            {
                "lightrag_status": "unavailable",
                "degraded_reason": "lightrag_graph_query_failed",
            }
        )
    deterministic.meta["snapshot_id"] = snapshot.snapshot_id if snapshot else ""
    return success_response(deterministic.model_dump(mode="json"))


@router.get("/query")
async def query(
    query: str = Query(..., min_length=1),
    mode: str = "mix",
    top_k: int = 5,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
) -> dict:
    del current_user, db
    result = LightRAGClient().query(query, mode=mode, top_k=top_k)
    return success_response(result.model_dump(mode="json"))


@router.get("/report")
async def report(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user, db
    return success_response(
        {
            "doc_build_report": _read_report(DEFAULT_REPORTS_DIR / "lightrag_doc_build_report.json"),
            "index_report": read_index_report(),
            "query_report": _read_report(DEFAULT_REPORTS_DIR / "lightrag_query_report.json"),
        }
    )


def _graph_role(user: User) -> str:
    if user.role not in {"doctor", "student"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="患者无权访问知识图谱")
    return user.role


def _read_report(path):
    if not path.exists():
        return {"status": "not_run"}
    import json

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {"status": "invalid"}


def _runtime_report_dir():
    config = get_lightrag_config()
    return config.input_dir.parent / "reports"


def _graph_item_count(payload, *keys: str) -> int:
    if not isinstance(payload, dict):
        return 0
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            nested = _graph_item_count(value, *keys)
            if nested:
                return nested
    data = payload.get("data")
    return _graph_item_count(data, *keys) if data is not payload else 0
