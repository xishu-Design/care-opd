from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
if os.environ.get("DATABASE_URL", "") in {"", "sqlite:///./guardtcm_dev.db"}:
    os.environ["DATABASE_URL"] = f"sqlite:///{BACKEND / 'guardtcm_dev.db'}"

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.content.services.content_import_service import ContentImportService  # noqa: E402
from app.content.validators.content_validator import ContentValidator  # noqa: E402
from app.data_pipeline.scripts.build_guardtcm_kb import build_guardtcm_kb  # noqa: E402
from app.data_pipeline.scripts.download_sources import build_download_report  # noqa: E402
from app.data_pipeline.scripts.process_approved_sources import process_approved_sources  # noqa: E402
from app.data_pipeline.validators.license_validator import generate_license_report  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.init_db import init_db  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.knowledge.snapshots import create_snapshot, transition_snapshot  # noqa: E402
from app.lightrag_integration.client import LightRAGClient, track_state  # noqa: E402
from app.lightrag_integration.config import get_lightrag_config  # noqa: E402
from app.lightrag_integration.document_builder import (  # noqa: E402
    build_lightrag_docs,
    load_lightrag_docs,
    persist_lightrag_doc_manifest,
)


def run(args: argparse.Namespace) -> dict:
    run_id = args.resume_run_id or f"local_ai_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    run_dir = ROOT / "runtime" / "lightrag" / "staging" / run_id
    reports = run_dir / "reports"
    processed = run_dir / "processed"
    content = run_dir / "content"
    docs_path = run_dir / "input" / "guardtcm_lightrag_docs.jsonl"
    report = {"run_id": run_id, "snapshot_version": args.snapshot_version, "status": "running", "stages": [], "started_at": _now()}
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report.update(status="dry_run", planned_stages=_stages(), ended_at=_now())
        return _write_report(reports, report)
    try:
        if not args.skip_download:
            _stage(report, "download", build_download_report())
        download_report = BACKEND / "app/data_pipeline/reports/download_report.json"
        license_report = BACKEND / "app/data_pipeline/reports/license_report.json"
        _stage(report, "verify_license", generate_license_report(download_report_path=download_report, output_path=license_report))
        if content.exists():
            shutil.rmtree(content)
        shutil.copytree(BACKEND / "app/content/data", content)
        processing = process_approved_sources(
            license_report_path=license_report,
            download_report_path=download_report,
            processed_dir=processed,
            reports_dir=reports,
            source_id=args.source,
        )
        _stage(report, "extract_deidentify_normalize", processing)
        if args.pilot or args.max_records:
            _limit_processed(processed, args.max_records or 40)
        build = build_guardtcm_kb(processed_dir=processed, content_data_dir=content, reports_dir=reports)
        _stage(report, "merge_and_build", build)
        validation = ContentValidator(content).validate().to_dict()
        _stage(report, "validate_content", validation)
        if not validation["valid"]:
            raise RuntimeError("staging content validation failed")
        staging_db = run_dir / "staging.db"
        engine = create_engine(f"sqlite:///{staging_db}")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        try:
            import_result = ContentImportService(session, content).import_content()
            session.commit()
        finally:
            session.close()
        _stage(report, "import_staging", import_result)
        docs_report = build_lightrag_docs(content_data_dir=content, output_path=docs_path, report_path=reports / "lightrag_docs.json")
        _stage(report, "build_lightrag_docs", docs_report)
        init_db()
        live_db = SessionLocal()
        snapshot = create_snapshot(
            live_db,
            version=args.snapshot_version,
            source_manifest=json.loads((processed / "source_catalog.json").read_text(encoding="utf-8")),
            record_counts=build["output_counts"],
            workspace=str(run_dir / "lightrag_store"),
            content_dir=content,
        )
        report["snapshot_id"] = snapshot.snapshot_id
        manifest_count = persist_lightrag_doc_manifest(live_db, snapshot_id=snapshot.snapshot_id, docs_path=docs_path)
        _stage(report, "persist_document_manifest", {"documents": manifest_count, "snapshot_id": snapshot.snapshot_id})
        compatibility = LightRAGClient().openapi_compatibility()
        _stage(report, "lightrag_openapi", compatibility)
        if not compatibility.get("compatible"):
            transition_snapshot(live_db, snapshot.snapshot_id, "FAILED", reason="LightRAG OpenAPI incompatible")
            raise RuntimeError("LightRAG OpenAPI compatibility gate failed")
        transition_snapshot(live_db, snapshot.snapshot_id, "VERIFYING")
        index_result = _index_and_verify(
            LightRAGClient(),
            docs_path,
            document_limit=40 if args.pilot or args.max_records else None,
        )
        _stage(report, "index_and_verify", index_result)
        staged_graph = run_dir / "lightrag_store"
        _capture_lightrag_workspace(get_lightrag_config().working_dir, staged_graph)
        _stage(report, "capture_lightrag_workspace", {"path": str(staged_graph)})
        if args.activate_after_verify:
            _activate_artifacts(
                content,
                BACKEND / "app/content/data",
                staged_graph,
                ROOT / "runtime/lightrag/active/current",
                docs_path.parent,
                ROOT / "runtime/lightrag/active/input",
            )
            live_import = ContentImportService(live_db, BACKEND / "app/content/data").import_content()
            live_db.commit()
            _stage(report, "import_live_database", live_import)
            transition_snapshot(live_db, snapshot.snapshot_id, "ACTIVE")
            report["activated"] = True
            report["service_restart_required"] = True
        live_db.close()
        report.update(status="success", ended_at=_now())
    except Exception as exc:
        report.update(status="failed", error=str(exc), ended_at=_now())
        snapshot_id = report.get("snapshot_id")
        if snapshot_id:
            failure_db = SessionLocal()
            try:
                from sqlalchemy import select
                from app.db.models import KnowledgeSnapshot

                row = failure_db.scalar(select(KnowledgeSnapshot).where(KnowledgeSnapshot.snapshot_id == snapshot_id))
                if row is not None and row.status in {"STAGING", "VERIFYING"}:
                    transition_snapshot(failure_db, snapshot_id, "FAILED", reason=str(exc))
            finally:
                failure_db.close()
        failed_dir = ROOT / "runtime" / "lightrag" / "failed" / run_id
        if run_dir.exists() and run_dir != failed_dir:
            failed_dir.parent.mkdir(parents=True, exist_ok=True)
            if failed_dir.exists():
                shutil.rmtree(failed_dir)
            shutil.move(str(run_dir), str(failed_dir))
            reports = failed_dir / "reports"
    return _write_report(reports, report)


def _index_and_verify(client: LightRAGClient, docs_path: Path, *, document_limit: int | None = None) -> dict:
    docs = load_lightrag_docs(docs_path)
    if document_limit is not None:
        docs = docs[:document_limit]
    track_ids, reused = _upload_idempotently(client, docs)
    for track_id in track_ids:
        _wait_track(client, track_id)
    queries = {query: client.query_data(query, mode="mix", top_k=5) for query in ("黄芪", "酸枣仁汤")}
    graph = client.graph("黄芪", 2, 80)
    if not graph:
        raise RuntimeError("LightRAG graph gate returned empty data")
    return {"track_ids": track_ids, "documents": len(docs), "reused_documents": reused, "queries": queries, "graph_nonempty": True}


def _upload_idempotently(client: LightRAGClient, docs: list) -> tuple[list[str], int]:
    try:
        response = client.upload_texts([doc.text for doc in docs], [doc.doc_id for doc in docs])
        track_id = _track_id(response)
        if not track_id:
            raise RuntimeError(f"LightRAG upload missing track_id: {response}")
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
            raise RuntimeError(f"LightRAG upload missing track_id for {doc.doc_id}: {response}")
        track_ids.append(track_id)
        _wait_track(client, track_id)
    return track_ids, reused


def _wait_track(client: LightRAGClient, track_id: str) -> None:
    for _ in range(720):
        status_payload = client.track_status(track_id)
        state = track_state(status_payload)
        if state in {"processed", "completed", "complete", "done", "success"}:
            return
        if state in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"LightRAG index failed: {status_payload}")
        time.sleep(5)
    raise TimeoutError(f"LightRAG indexing timed out: {track_id}")


def _track_id(response: object) -> str:
    if not isinstance(response, dict):
        return ""
    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    return str(response.get("track_id") or response.get("id") or data.get("track_id") or data.get("id") or "")


def _limit_processed(processed: Path, limit: int) -> None:
    limits = {"herb_candidates.json": min(limit, 20), "formula_candidates.json": min(limit, 10), "learning_case_candidates.json": min(limit, 10), "approved_training_items.json": min(limit, 10)}
    for name, count in limits.items():
        path = processed / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items", [])
        selected = []
        for required in ("黄芪", "酸枣仁汤"):
            match = next((item for item in items if required in json.dumps(item, ensure_ascii=False)), None)
            if match is not None and match not in selected:
                selected.append(match)
        selected.extend(item for item in items if item not in selected)
        payload["items"] = selected[:count]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _capture_lightrag_workspace(source: Path, target: Path) -> None:
    if not source.exists() or not any(source.iterdir()):
        raise RuntimeError(f"LightRAG staging workspace is empty: {source}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _activate_artifacts(
    staged_content: Path,
    active_content: Path,
    staged_graph: Path,
    active_graph: Path,
    staged_input: Path | None = None,
    active_input: Path | None = None,
) -> None:
    pairs = [(staged_content, active_content), (staged_graph, active_graph)]
    if staged_input is not None and active_input is not None:
        pairs.append((staged_input, active_input))
    prepared: list[tuple[Path, Path, Path]] = []
    for staged, active in pairs:
        next_dir = active.with_name(f"{active.name}.next")
        backup = active.with_name(f"{active.name}.previous")
        active.parent.mkdir(parents=True, exist_ok=True)
        if next_dir.exists():
            shutil.rmtree(next_dir)
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(staged, next_dir)
        prepared.append((active, next_dir, backup))

    switched: list[tuple[Path, Path]] = []
    try:
        for active, next_dir, backup in prepared:
            if active.exists():
                active.rename(backup)
            next_dir.rename(active)
            switched.append((active, backup))
    except Exception:
        for active, backup in reversed(switched):
            if active.exists():
                shutil.rmtree(active)
            if backup.exists():
                backup.rename(active)
        raise
    for _, _, backup in prepared:
        if backup.exists():
            shutil.rmtree(backup)


def _stage(report: dict, name: str, result: object) -> None:
    report["stages"].append({"stage": name, "status": "completed", "result": result})


def _stages() -> list[str]:
    return ["preflight", "download", "verify_license", "extract", "deidentify", "normalize", "merge", "validate_content", "create_snapshot", "import_staging", "build_lightrag_docs", "persist_document_manifest", "upload_lightrag", "wait_index", "verify_queries", "verify_graph", "capture_lightrag_workspace", "import_live_database", "activate_snapshot"]


def _write_report(reports: Path, report: dict) -> dict:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "local_ai_stack_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 GuardTCM 本地 Ollama + LightRAG 数据栈。")
    parser.add_argument("--sources", default="TCMDataForML,MTCMB")
    parser.add_argument("--source")
    parser.add_argument("--snapshot-version", required=True)
    parser.add_argument("--activate-after-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume-run-id")
    result = run(parser.parse_args())
    return 0 if result["status"] in {"success", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
