#!/usr/bin/env python3
"""Audit the frozen 30 cases against the exact active retrieval snapshots.

The task selection remains frozen.  This pre-inference qualification detects
whether any retrieval source enabled by the selected product profile contains
an exact/near duplicate of an evaluated case stem.  It records hashes and
similarity statistics only; no case or corpus text is copied to the report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_tcm_best4sdt_agent30 import (
    _best_overlap,
    _canonical_hash,
    _json_bytes,
    _load_local_corpus,
    _sha256_file,
    _sha256_text,
    _write_frozen,
)


SCHEMA_VERSION = "care_opd.competition.tcm_best4sdt_agent30.runtime_overlap.v2"

PROFILE_SCOPES = {
    "local_product": (
        "Frozen TCM-BEST4SDT 30-case stems versus the exact active local "
        "structured-knowledge JSON snapshots plus any optional QA SQLite "
        "snapshot enabled for this run"
    ),
    "live_product": (
        "Frozen TCM-BEST4SDT 30-case stems versus the exact active LightRAG "
        "document manifest and QA knowledge SQLite snapshot"
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    tasks_path = args.tasks.expanduser().resolve()
    sources = [path.expanduser().resolve() for path in args.overlap_source]
    output = args.output.expanduser().resolve()
    if not tasks_path.is_file():
        raise FileNotFoundError(tasks_path)
    if not sources:
        raise ValueError("runtime audit requires at least one active retrieval source")

    tasks = _read_jsonl(tasks_path)
    if len(tasks) != args.expected_tasks:
        raise ValueError(
            f"task count mismatch: expected={args.expected_tasks}, actual={len(tasks)}"
        )
    documents, source_meta = _load_local_corpus(sources)
    if not documents:
        raise ValueError("runtime retrieval snapshots contain no auditable text")
    inverted: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        for gram in document["grams"]:
            inverted[gram].append(index)

    rows = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        stem = str(task.get("source_text_private") or "")
        if not task_id or not stem:
            raise ValueError("frozen task is missing task_id/source_text_private")
        overlap = _best_overlap(stem, documents, inverted)
        rows.append(
            {
                "task_id": task_id,
                "case_material_sha256": _sha256_text(stem),
                **overlap,
            }
        )

    disqualified = [row for row in rows if row["near_duplicate"]]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL_EXACT_OR_NEAR_DUPLICATE" if disqualified else "QUALIFIED_BEFORE_INFERENCE",
        "retrieval_profile": args.retrieval_profile,
        "scope": PROFILE_SCOPES[args.retrieval_profile],
        "tasks_sha256": _sha256_file(tasks_path),
        "task_count": len(tasks),
        "runtime_sources": source_meta,
        "runtime_source_set_sha256": _canonical_hash(source_meta),
        "audited_local_string_count": len(documents),
        "exact_or_near_duplicate_count": len(disqualified),
        "rows": rows,
    }
    _write_frozen(output, _json_bytes(report))
    if disqualified:
        raise RuntimeError(
            "runtime retrieval leakage audit failed for task ids: "
            + ",".join(row["task_id"] for row in disqualified)
        )
    return {
        "status": report["status"],
        "retrieval_profile": args.retrieval_profile,
        "task_count": len(tasks),
        "runtime_source_count": len(source_meta),
        "runtime_source_set_sha256": report["runtime_source_set_sha256"],
        "output": str(output),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--overlap-source", type=Path, action="append", required=True)
    value.add_argument(
        "--retrieval-profile",
        choices=tuple(PROFILE_SCOPES),
        required=True,
    )
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--expected-tasks", type=int, default=30)
    return value


def main() -> None:
    print(json.dumps(audit(parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
