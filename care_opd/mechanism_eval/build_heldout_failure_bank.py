#!/usr/bin/env python3
"""Build a result-blind failure bank for parameter-preference diagnostics.

The input rollouts should come from the shared Clean-SFT checkpoint on the
held-out QZBC request file.  This script joins each failed output back to its
verified request/oracle and emits only data needed by ``recovery_margin.py``.
It does not train a model and does not execute the QZBC Agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from care_opd.qzbc_contract import canonical_response
from care_opd.qzbc_rollout import assess_response


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _index_unique(rows: Iterable[Mapping[str, Any]], source: Path) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id:
            raise ValueError(f"{source}: row without id")
        if row_id in indexed:
            raise ValueError(f"{source}: duplicate id {row_id}")
        indexed[row_id] = row
    return indexed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                count += 1
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return count


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build(requests_path: Path, rollouts_path: Path, output_path: Path) -> dict[str, Any]:
    requests = _index_unique(_read_jsonl(requests_path), requests_path)
    rollouts = _read_jsonl(rollouts_path)
    seen: set[str] = set()
    failures: list[dict[str, Any]] = []
    variants: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    inconsistent_saved_assessments = 0

    for rollout in rollouts:
        row_id = str(rollout.get("id") or "")
        if not row_id or row_id in seen:
            raise ValueError(f"{rollouts_path}: missing or duplicate id {row_id!r}")
        seen.add(row_id)
        request = requests.get(row_id)
        if request is None:
            raise ValueError(f"rollout id {row_id} is absent from {requests_path}")
        raw = str(rollout.get("rollout") or "")
        assessment = assess_response(request, raw)
        if bool(rollout.get("passed")) != assessment.passed:
            inconsistent_saved_assessments += 1
        if assessment.passed:
            continue

        oracle = canonical_response(request["oracle_response"])
        oracle_assessment = assess_response(request, oracle)
        if not oracle_assessment.passed:
            raise ValueError(f"request {row_id}: oracle does not pass the host verifier")
        variant = str(request.get("variant") or "clean")
        error_names = list(assessment.failure_types)
        variants[variant] += 1
        failure_types.update(error_names)
        failures.append(
            {
                "id": row_id,
                "seed_id": request.get("seed_id"),
                "variant": variant,
                "student_messages": request["student_messages"],
                "rollout": raw,
                "oracle_response": oracle,
                "failure_types": error_names,
                "selected_components": list(assessment.components),
                "risk_level": request.get("risk_level", "high"),
                "risk_weight": float(request.get("risk_weight", 1.0)),
                "source": "heldout_clean_sft_failure",
            }
        )

    if not failures:
        raise ValueError("no verifier-failed rollouts were found")
    count = _atomic_jsonl(output_path, failures)
    manifest = {
        "schema": "care-opd-heldout-failure-bank-v1",
        "requests": str(requests_path.resolve()),
        "requests_sha256": _sha256(requests_path),
        "rollouts": str(rollouts_path.resolve()),
        "rollouts_sha256": _sha256(rollouts_path),
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "rollouts_seen": len(rollouts),
        "failures_emitted": count,
        "variant_counts": dict(sorted(variants.items())),
        "failure_type_counts": dict(sorted(failure_types.items())),
        "saved_assessment_mismatches": inconsistent_saved_assessments,
        "interpretation": (
            "Failed Clean-SFT responses joined to verifier-valid held-out oracles; "
            "for parameter-preference diagnostics, not training or end-to-end claims."
        ),
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build(args.requests, args.rollouts, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
