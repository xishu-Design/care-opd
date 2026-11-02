#!/usr/bin/env python3
"""Audit failure-set dynamics across three CARE-OPD online rounds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "care-opd-round-dynamics-v1"
OUTPUT_NAMES = ("round_dynamics.json", "round_dynamics.csv", "round_dynamics.md")
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class InputError(ValueError):
    """Raised when a round archive violates the audit contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise InputError(f"{path}:{line_number}: expected a JSON object")
            value["__line__"] = line_number
            rows.append(value)
    if not rows:
        raise InputError(f"{path}: no JSONL records")
    return rows


def _parse_rounds(values: Sequence[str], expected_rounds: int) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" in value:
            label, raw_path = value.split("=", 1)
        else:
            raw_path = value
            label = Path(value).name
        label = label.strip()
        if not LABEL_RE.fullmatch(label):
            raise InputError(f"invalid round label {label!r}")
        if label in seen:
            raise InputError(f"duplicate round label {label!r}")
        seen.add(label)
        result.append((label, Path(raw_path).expanduser().resolve()))
    if len(result) != expected_rounds:
        raise InputError(
            f"expected exactly {expected_rounds} --round inputs, got {len(result)}"
        )
    return result


def _resolve_one(root: Path, relative_candidates: Sequence[str], name: str) -> Path:
    matches = [root / candidate for candidate in relative_candidates if (root / candidate).is_file()]
    if not matches:
        raise FileNotFoundError(
            f"{root}: could not find {name}; tried {list(relative_candidates)}"
        )
    if len(matches) > 1:
        raise InputError(f"{root}: ambiguous {name} inputs: {matches}")
    return matches[0]


def _round_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        "summary": _resolve_one(root, ("summary.json", "rollouts/summary.json"), "summary"),
        "rollouts": _resolve_one(
            root, ("qzbc_rollouts.jsonl", "rollouts/qzbc_rollouts.jsonl"), "rollouts"
        ),
        "opd": _resolve_one(root, ("qzbc_opd.jsonl", "rollouts/qzbc_opd.jsonl"), "OPD rows"),
        "metrics": _resolve_one(
            root,
            (
                "metrics.jsonl",
                "opd_train/metrics.jsonl",
                "../opd_train/metrics.jsonl",
            ),
            "training metrics",
        ),
    }


def _strict_int(summary: Mapping[str, Any], key: str, *, where: str) -> int:
    value = summary.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(f"{where}.{key}: expected a non-negative integer")
    return value


def _index_rollouts(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise InputError(f"{where}.id: expected a non-empty string")
        item_id = item_id.strip()
        if item_id in indexed:
            raise InputError(f"{where}: duplicate id {item_id!r}")
        passed = row.get("passed")
        if not isinstance(passed, bool):
            raise InputError(f"{where}.passed: expected a JSON boolean")
        failure_types = row.get("failure_types")
        if not isinstance(failure_types, list) or any(
            not isinstance(item, str) or not item for item in failure_types
        ):
            raise InputError(f"{where}.failure_types: expected a string list")
        if passed and failure_types:
            raise InputError(f"{where}: passed row must not contain failure_types")
        if not passed and not failure_types:
            raise InputError(f"{where}: failed row must contain failure_types")
        indexed[item_id] = {
            "passed": passed,
            "failure_types": tuple(dict.fromkeys(failure_types)),
        }
    return indexed


def _index_opd(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise InputError(f"{where}.id: expected a non-empty string")
        item_id = item_id.strip()
        if item_id in indexed:
            raise InputError(f"{where}: duplicate id {item_id!r}")
        clean_control = row.get("clean_control", False)
        if not isinstance(clean_control, bool):
            raise InputError(f"{where}.clean_control: expected a JSON boolean")
        indexed[item_id] = {"clean_control": clean_control}
    return indexed


def _metric_summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    event_counts: Counter[str] = Counter()
    numeric: dict[str, list[float]] = defaultdict(list)
    starts: list[dict[str, Any]] = []
    complete: dict[str, Any] | None = None
    for row in rows:
        line = row.pop("__line__")
        event = row.get("event")
        if not isinstance(event, str) or not event:
            raise InputError(f"{path}:{line}.event: expected a non-empty string")
        event_counts[event] += 1
        if event == "start":
            starts.append(dict(row))
        elif event == "complete":
            complete = dict(row)
        elif event == "train":
            for key, value in row.items():
                if key == "event" or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    numeric[key].append(float(value))
    if not starts:
        raise InputError(f"{path}: missing start event")
    if complete is None:
        raise InputError(f"{path}: missing complete event")
    methods = {str(row.get("method") or "") for row in starts}
    if "" in methods or len(methods) != 1:
        raise InputError(f"{path}: inconsistent or missing start-event method: {methods}")
    switch_fingerprints = {
        json.dumps(row.get("effective_switches"), sort_keys=True, separators=(",", ":"))
        for row in starts
        if row.get("effective_switches") is not None
    }
    if len(switch_fingerprints) > 1:
        raise InputError(f"{path}: effective switches changed across resumed starts")
    stats = {
        key: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }
        for key, values in sorted(numeric.items())
    }
    return {
        "rows": len(rows),
        "event_counts": dict(sorted(event_counts.items())),
        "start_events": starts,
        "latest_start": starts[-1],
        "complete": complete,
        "train_numeric": stats,
        "scope_note": (
            "metrics.jsonl contains main-process logged batches; *_local fields "
            "are diagnostics, not guaranteed global DDP aggregates"
        ),
    }


def _failure_sets(rollouts: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
    failed = {item_id for item_id, row in rollouts.items() if not row["passed"]}
    format_failed = {
        item_id
        for item_id in failed
        if "invalid_generated_response" in rollouts[item_id]["failure_types"]
    }
    semantic_failed = failed - format_failed
    return {"all": failed, "format": format_failed, "semantic": semantic_failed}


def _validate_summary(
    summary: Mapping[str, Any],
    rollouts: Mapping[str, Mapping[str, Any]],
    opd: Mapping[str, Mapping[str, Any]],
    *,
    where: str,
) -> None:
    passed = sum(bool(row["passed"]) for row in rollouts.values())
    failed = len(rollouts) - passed
    format_failed = sum(
        not row["passed"] and "invalid_generated_response" in row["failure_types"]
        for row in rollouts.values()
    )
    semantic_failed = failed - format_failed
    clean_controls = sum(bool(row["clean_control"]) for row in opd.values())
    opd_failures = len(opd) - clean_controls
    expected = {
        "rollouts": len(rollouts),
        "passed": passed,
        "opd_examples": len(opd),
        "opd_failures": opd_failures,
        "clean_controls": clean_controls,
        "format_failures_total": format_failed,
        "semantic_failures": semantic_failed,
    }
    for key, actual in expected.items():
        declared = _strict_int(summary, key, where=where)
        if declared != actual:
            raise InputError(
                f"{where}.{key}: declared {declared}, recomputed {actual}"
            )


def _round_record(label: str, root: Path) -> tuple[dict[str, Any], dict[str, set[str]], set[str]]:
    files = _round_files(root)
    summary = _load_json(files["summary"])
    rollouts = _index_rollouts(files["rollouts"])
    opd = _index_opd(files["opd"])
    _validate_summary(summary, rollouts, opd, where=str(files["summary"]))
    sets = _failure_sets(rollouts)
    record = {
        "label": label,
        "root": str(root),
        "files": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in files.items()
        },
        "summary": summary,
        "recomputed": {
            "rollouts": len(rollouts),
            "passed": sum(row["passed"] for row in rollouts.values()),
            "failed": len(sets["all"]),
            "format_failures": len(sets["format"]),
            "semantic_failures": len(sets["semantic"]),
            "opd_examples": len(opd),
            "opd_failures": sum(not row["clean_control"] for row in opd.values()),
            "clean_controls": sum(row["clean_control"] for row in opd.values()),
        },
        "training_metrics": _metric_summary(files["metrics"]),
    }
    return record, sets, set(rollouts)


def _transition(
    previous_label: str,
    current_label: str,
    failure_class: str,
    previous: set[str],
    current: set[str],
) -> dict[str, Any]:
    persistent = previous & current
    disappeared = previous - current
    new = current - previous
    union = previous | current
    return {
        "from": previous_label,
        "to": current_label,
        "failure_class": failure_class,
        "previous": len(previous),
        "current": len(current),
        "persistent": len(persistent),
        "disappeared": len(disappeared),
        "new": len(new),
        "jaccard": len(persistent) / len(union) if union else 1.0,
        "closure_rate": len(disappeared) / len(previous) if previous else None,
        "new_rate": len(new) / len(current) if current else None,
    }


def build_analysis(round_inputs: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    sets_by_label: dict[str, dict[str, set[str]]] = {}
    common_ids: set[str] | None = None
    for label, root in round_inputs:
        record, sets, ids = _round_record(label, root)
        if common_ids is None:
            common_ids = ids
        elif ids != common_ids:
            raise InputError(
                f"{label}: rollout ID set differs across rounds; "
                f"missing={sorted(common_ids - ids)[:10]}, "
                f"extra={sorted(ids - common_ids)[:10]}"
            )
        records.append(record)
        sets_by_label[label] = sets

    transitions: list[dict[str, Any]] = []
    for index in range(1, len(round_inputs)):
        previous_label = round_inputs[index - 1][0]
        current_label = round_inputs[index][0]
        for failure_class in ("all", "format", "semantic"):
            transitions.append(
                _transition(
                    previous_label,
                    current_label,
                    failure_class,
                    sets_by_label[previous_label][failure_class],
                    sets_by_label[current_label][failure_class],
                )
            )

    labels = [label for label, _ in round_inputs]
    across_all: dict[str, Any] = {}
    for failure_class in ("all", "format", "semantic"):
        class_sets = [sets_by_label[label][failure_class] for label in labels]
        persistent = set.intersection(*class_sets)
        ever = set.union(*class_sets)
        across_all[failure_class] = {
            "persistent_all_rounds": len(persistent),
            "ever_failed": len(ever),
            "persistent_fraction_of_ever": (
                len(persistent) / len(ever) if ever else None
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "definitions": {
            "persistent": "failed in both adjacent rounds",
            "disappeared": "failed in previous round and passed in current round",
            "new": "passed in previous round and failed in current round",
            "jaccard": "|failure intersection| / |failure union|",
            "format_failure": "failure_types contains invalid_generated_response",
            "semantic_failure": "failed and not a format_failure",
        },
        "rounds": records,
        "transitions": transitions,
        "across_all_rounds": across_all,
    }


def _atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in payload["rounds"]:
        values = record["recomputed"]
        rows.append({"row_type": "round", "round": record["label"], **values})
    for record in payload["transitions"]:
        rows.append({"row_type": "transition", "round": "", **record})
    return rows


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _pct(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CARE-OPD online-round failure dynamics",
        "",
        "## Per-round counts",
        "",
        "| Round | Rollouts | Passed | Failed | Format failures | Semantic failures | OPD failures | Clean controls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["rounds"]:
        row = record["recomputed"]
        lines.append(
            f"| {record['label']} | {row['rollouts']} | {row['passed']} | "
            f"{row['failed']} | {row['format_failures']} | "
            f"{row['semantic_failures']} | {row['opd_failures']} | "
            f"{row['clean_controls']} |"
        )
    lines.extend(
        [
            "",
            "## Adjacent-round set transitions",
            "",
            "| From → To | Class | Previous | Current | Persistent | Disappeared | New | Jaccard | Closure |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["transitions"]:
        lines.append(
            f"| {row['from']} → {row['to']} | {row['failure_class']} | "
            f"{row['previous']} | {row['current']} | {row['persistent']} | "
            f"{row['disappeared']} | {row['new']} | {row['jaccard']:.4f} | "
            f"{_pct(row['closure_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Counts are recomputed from rollout/OPD archives and checked against "
            "each round's summary before any report is written. Set turnover is "
            "descriptive and does not by itself establish a causal training effect.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    round_inputs = _parse_rounds(args.rounds, args.expected_rounds)
    payload = build_analysis(round_inputs)
    output_dir = args.output_dir.expanduser().resolve()
    targets = [output_dir / name for name in OUTPUT_NAMES]
    if not args.overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"outputs already exist: {existing}; pass --overwrite")
    rows = _csv_rows(payload)
    _atomic_text(
        targets[0],
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )
    _atomic_text(targets[1], _csv_text(rows), overwrite=args.overwrite)
    _atomic_text(targets[2], _markdown(payload), overwrite=args.overwrite)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round",
        dest="rounds",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="repeat in chronological order; DIR may be a round root or rollout root",
    )
    parser.add_argument("--expected-rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_rounds < 2:
        raise InputError("--expected-rounds must be at least 2")
    payload = run(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "rounds": len(payload["rounds"]),
                "transitions": len(payload["transitions"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
