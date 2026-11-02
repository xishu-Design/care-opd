#!/usr/bin/env python3
"""Build and index the frozen GuardTCM knowledge snapshot for the Agent trial.

This utility is deliberately independent of the 30 evaluation prompts.  It
uses only the QZBC content snapshot bundled before generation, writes a stable
JSONL document map, uploads it to the loopback LightRAG service in recoverable
batches, and records hashes/progress for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "guardtcm-lightrag-bootstrap-v1"
REQUIRED_OPENAPI_PATHS = {
    "/health",
    "/documents/texts",
    "/documents/track_status/{track_id}",
    "/query/data",
    "/graphs",
}


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class API:
    def __init__(self, base_url: str, api_key: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout: int | None = None) -> Any:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        payload = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=payload, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")


def _track_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    nested = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    return str(payload.get("track_id") or payload.get("id") or nested.get("track_id") or nested.get("id") or "")


def _track_state(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    direct = str(payload.get("status") or payload.get("track_status") or "").lower()
    if direct:
        return direct
    documents = payload.get("documents")
    if isinstance(documents, list) and documents:
        states = {str(row.get("status") or "").lower() for row in documents if isinstance(row, dict)}
        if states & {"failed", "error", "cancelled", "canceled"}:
            return "failed"
        if states and states <= {"processed", "completed", "complete", "done", "success"}:
            return "processed"
        return "processing"
    summary = payload.get("status_summary")
    if isinstance(summary, dict) and summary:
        states = {str(key).split(".")[-1].lower() for key, count in summary.items() if count}
        if states & {"failed", "error", "cancelled", "canceled"}:
            return "failed"
        if states and states <= {"processed", "completed", "complete", "done", "success"}:
            return "processed"
        return "processing"
    return ""


def _failed_track_summary(track_id: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        return f"LightRAG track failed: {track_id}"
    documents = payload.get("documents")
    errors: Counter[str] = Counter()
    if isinstance(documents, list):
        for row in documents:
            if isinstance(row, dict):
                message = str(row.get("error_msg") or "unspecified document failure").strip()
                if message:
                    errors[message] += 1
    compact_errors = [f"{count}x {message}" for message, count in errors.most_common(5)]
    return (
        f"LightRAG track failed: {track_id}; total={payload.get('total_count')}; "
        f"status_summary={payload.get('status_summary')}; errors={compact_errors}"
    )


def _ollama_preflight(base_url: str, timeout: int) -> dict[str, Any]:
    base = base_url.rstrip("/")
    with urllib.request.urlopen(base + "/api/tags", timeout=min(timeout, 30)) as response:
        payload = json.loads(response.read().decode("utf-8") or "{}")
    names = {str(row.get("name") or "") for row in payload.get("models", []) if isinstance(row, dict)}
    missing = [
        model
        for model in ("qwen2.5:7b", "bge-m3")
        if model not in names and model + ":latest" not in names
    ]
    if missing:
        raise RuntimeError(f"Ollama at {base} is missing required models: {missing}; visible={sorted(names)}")

    embed_request = urllib.request.Request(
        base + "/api/embed",
        data=json.dumps({"model": "bge-m3", "input": ["中医辨证论治"]}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(embed_request, timeout=min(timeout, 180)) as response:
        embed_payload = json.loads(response.read().decode("utf-8") or "{}")
    embeddings = embed_payload.get("embeddings") if isinstance(embed_payload, dict) else None
    dimensions = len(embeddings[0]) if isinstance(embeddings, list) and embeddings else 0
    if dimensions != 1024:
        raise RuntimeError(f"Ollama bge-m3 returned unexpected dimensions: {dimensions}")

    generate_request = urllib.request.Request(
        base + "/api/generate",
        data=json.dumps(
            {
                "model": "qwen2.5:7b",
                "prompt": "仅回答：就绪",
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0, "num_predict": 8},
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(generate_request, timeout=min(timeout, 300)) as response:
        generation = json.loads(response.read().decode("utf-8") or "{}")
    if not str(generation.get("response") or "").strip():
        raise RuntimeError(f"Ollama qwen2.5:7b generation probe failed: {generation}")
    return {"base_url": base, "visible_models": sorted(names), "embedding_dimensions": dimensions}


def _load_docs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or "")
            text = str(row.get("text") or "").strip()
            if not doc_id.startswith("DOC_") or not text:
                raise RuntimeError(f"invalid document at line {line_number}")
            if doc_id in seen:
                raise RuntimeError(f"duplicate doc_id: {doc_id}")
            seen.add(doc_id)
            rows.append(row)
    if not rows:
        raise RuntimeError("frozen LightRAG document map is empty")
    return rows


def _batch_id(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["doc_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["text"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _wait_track(api: API, track_id: str, *, poll_seconds: int, deadline_seconds: int) -> None:
    deadline = time.monotonic() + deadline_seconds
    last_state = ""
    while time.monotonic() < deadline:
        payload = api.request("GET", f"/documents/track_status/{urllib.parse.quote(track_id, safe='')}")
        state = _track_state(payload)
        if state != last_state:
            print(f"TRACK {track_id} state={state or 'unknown'}", flush=True)
            last_state = state
        if state in {"processed", "completed", "complete", "done", "success"}:
            return
        if state in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(_failed_track_summary(track_id, payload))
        time.sleep(poll_seconds)
    raise TimeoutError(f"LightRAG track timed out after {deadline_seconds}s: {track_id}")


def _submit(api: API, rows: list[dict[str, Any]]) -> str:
    response = api.request(
        "POST",
        "/documents/texts",
        {
            "texts": [str(row["text"]) for row in rows],
            "file_sources": [str(row["doc_id"]) for row in rows],
        },
    )
    track_id = _track_id(response)
    if not track_id:
        raise RuntimeError(f"LightRAG upload returned no track_id: {response}")
    return track_id


def _submit_resilient(api: API, rows: list[dict[str, Any]]) -> list[str]:
    try:
        return [_submit(api, rows)]
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise
    if len(rows) == 1:
        print(f"REUSE_EXISTING {rows[0]['doc_id']}", flush=True)
        return []
    midpoint = len(rows) // 2
    return _submit_resilient(api, rows[:midpoint]) + _submit_resilient(api, rows[midpoint:])


def _query_evidence(api: API, query: str, known_doc_ids: set[str]) -> dict[str, Any]:
    payload = api.request("POST", "/query/data", {"query": query, "mode": "mix", "top_k": 10}, timeout=300)
    data = payload.get("data") if isinstance(payload, dict) else None
    data = data if isinstance(data, dict) else payload
    chunks = data.get("chunks") if isinstance(data, dict) else None
    references = data.get("references") if isinstance(data, dict) else None
    if not isinstance(chunks, list) or not chunks or not isinstance(references, list):
        raise RuntimeError(f"LightRAG query returned no structured evidence for {query!r}")
    returned_doc_ids = {
        part.strip()
        for chunk in chunks
        if isinstance(chunk, dict)
        for part in str(chunk.get("file_path") or "").split("<SEP>")
        if part.strip().startswith("DOC_")
    }
    mapped = sorted(known_doc_ids & returned_doc_ids)
    if not mapped:
        raise RuntimeError(f"LightRAG query chunks do not map to frozen documents for {query!r}")
    return {
        "query": query,
        "chunk_count": len(chunks),
        "reference_count": len(references),
        "mapped_document_count": len(mapped),
        "mapped_document_ids": mapped[:10],
    }


def _graph_nonempty(payload: Any) -> bool:
    if isinstance(payload, list):
        return bool(payload)
    if not isinstance(payload, dict):
        return bool(payload)
    for key in ("nodes", "entities", "edges", "relations", "data"):
        value = payload.get(key)
        if isinstance(value, (list, dict)) and value:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and index the frozen GuardTCM LightRAG snapshot.")
    parser.add_argument("--qzbc-root", required=True)
    parser.add_argument("--content-data-dir")
    parser.add_argument("--docs-path", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:9621")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--api-key-env", default="LIGHTRAG_API_KEY")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--request-timeout-seconds", type=int, default=600)
    parser.add_argument("--track-timeout-seconds", type=int, default=21600)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--state-path")
    parser.add_argument("--manifest-path")
    args = parser.parse_args()

    qzbc_root = Path(args.qzbc_root).resolve()
    backend = qzbc_root / "backend"
    content = Path(args.content_data_dir).resolve() if args.content_data_dir else backend / "app/content/data"
    docs_path = Path(args.docs_path).resolve()
    working_dir = Path(args.working_dir).resolve()
    state_path = Path(args.state_path).resolve() if args.state_path else docs_path.with_suffix(".bootstrap_state.private.json")
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else docs_path.with_suffix(".bootstrap_manifest.private.json")
    if not (backend / "app/lightrag_integration/document_builder.py").is_file():
        raise SystemExit(f"missing frozen QZBC backend: {backend}")
    if not content.is_dir():
        raise SystemExit(f"missing frozen content directory: {content}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    sys.path.insert(0, str(backend))
    os.environ.setdefault("LIGHTRAG_ENABLED", "true")
    os.environ.setdefault("LIGHTRAG_MODE", "api")
    os.environ["LIGHTRAG_API_BASE_URL"] = args.api_base_url
    os.environ["LIGHTRAG_INPUT_DIR"] = str(docs_path.parent)
    os.environ["LIGHTRAG_WORKING_DIR"] = str(working_dir)
    os.environ["LIGHTRAG_TIMEOUT_SECONDS"] = str(args.request_timeout_seconds)
    api_key = os.environ.get(args.api_key_env, "").strip()
    os.environ["LIGHTRAG_API_KEY"] = api_key

    from app.lightrag_integration.document_builder import build_lightrag_docs  # noqa: PLC0415

    docs_path.parent.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    build_report_path = manifest_path.with_name("lightrag_document_build_report.private.json")
    build_report = build_lightrag_docs(
        content_data_dir=content,
        output_path=docs_path,
        report_path=build_report_path,
    )
    docs_sha256 = _sha256(docs_path)
    rows = _load_docs(docs_path)
    source_files = sorted(path for path in content.glob("*.json") if path.is_file())
    source_hashes = {path.name: _sha256(path) for path in source_files}
    counts = Counter(str(row.get("entry_type") or "unknown") for row in rows)

    api = API(args.api_base_url, api_key, args.request_timeout_seconds)
    ollama = _ollama_preflight(args.ollama_base_url, args.request_timeout_seconds)
    health = api.request("GET", "/health")
    status = str(health.get("status") or "ok").lower() if isinstance(health, dict) else ""
    if status not in {"ok", "healthy", "ready", "success"}:
        raise SystemExit(f"LightRAG is not healthy: {health}")
    openapi = api.request("GET", "/openapi.json")
    paths = set(openapi.get("paths", {})) if isinstance(openapi, dict) else set()
    missing_paths = sorted(REQUIRED_OPENAPI_PATHS - paths)
    if missing_paths:
        raise SystemExit(f"LightRAG OpenAPI missing paths: {missing_paths}")

    batches = [rows[offset : offset + args.batch_size] for offset in range(0, len(rows), args.batch_size)]
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("contract_version") != CONTRACT_VERSION or state.get("docs_sha256") != docs_sha256:
            raise SystemExit(f"existing bootstrap state does not match frozen docs: {state_path}")
    else:
        state = {
            "contract_version": CONTRACT_VERSION,
            "status": "INDEXING",
            "started_at": _now(),
            "docs_path": str(docs_path),
            "docs_sha256": docs_sha256,
            "document_count": len(rows),
            "batch_size": args.batch_size,
            "batches": {},
        }
        _atomic_json(state_path, state)

    for index, batch in enumerate(batches, start=1):
        batch_key = f"{index:05d}-{_batch_id(batch)[:16]}"
        record = state["batches"].get(batch_key, {})
        if record.get("status") in {"COMPLETE", "REUSED"}:
            print(f"SKIP_COMPLETE batch={index}/{len(batches)}", flush=True)
            continue
        track_ids = [str(value) for value in record.get("track_ids", []) if str(value)]
        if not track_ids:
            print(f"SUBMIT batch={index}/{len(batches)} docs={len(batch)}", flush=True)
            track_ids = _submit_resilient(api, batch)
            record = {
                "status": "PROCESSING" if track_ids else "REUSED",
                "doc_count": len(batch),
                "first_doc_id": str(batch[0]["doc_id"]),
                "last_doc_id": str(batch[-1]["doc_id"]),
                "track_ids": track_ids,
                "submitted_at": _now(),
            }
            state["batches"][batch_key] = record
            _atomic_json(state_path, state)
        for track_id in track_ids:
            _wait_track(
                api,
                track_id,
                poll_seconds=args.poll_seconds,
                deadline_seconds=args.track_timeout_seconds,
            )
        record["status"] = "COMPLETE" if track_ids else "REUSED"
        record["completed_at"] = _now()
        state["batches"][batch_key] = record
        _atomic_json(state_path, state)
        print(f"BATCH_COMPLETE batch={index}/{len(batches)}", flush=True)

    known_doc_ids = {str(row["doc_id"]) for row in rows}
    probes = [
        _query_evidence(api, "黄芪", known_doc_ids),
        _query_evidence(api, "酸枣仁汤", known_doc_ids),
    ]
    graph = api.request("GET", "/graphs?label=%E9%BB%84%E8%8A%AA&max_depth=2&max_nodes=80", timeout=120)
    if not _graph_nonempty(graph):
        raise RuntimeError("LightRAG graph verification returned empty data")

    state.update(status="COMPLETE", completed_at=_now())
    _atomic_json(state_path, state)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": "COMPLETE",
        "completed_at": _now(),
        "selection_dependency": "none; built before and independently of evaluation prompts",
        "qzbc_root": str(qzbc_root),
        "content_data_dir": str(content),
        "source_file_sha256": source_hashes,
        "docs_path": str(docs_path),
        "docs_sha256": docs_sha256,
        "document_count": len(rows),
        "entry_type_counts": dict(sorted(counts.items())),
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "build_report": build_report,
        "verification_queries": probes,
        "graph_nonempty": True,
        "api_base_url": args.api_base_url,
        "ollama_preflight": ollama,
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print("FULL_PRODUCT_LIGHTRAG_BOOTSTRAP=COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
