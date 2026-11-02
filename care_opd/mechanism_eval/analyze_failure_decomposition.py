#!/usr/bin/env python3
"""Decompose paired QZBC outcomes into pass, rescue, and regression strata.

The script is deliberately result-agnostic: it consumes one frozen request
file and two or more already generated rollout archives.  Failure/component
strata are defined by the reference model (``clean_sft`` by default), avoiding
post-hoc regrouping by the candidate being evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "care-opd-failure-decomposition-v1"
DEFAULT_REQUIRED_LABELS = ("clean_sft", "gkd_full", "care_opd")
OUTPUT_NAMES = (
    "failure_decomposition.json",
    "failure_decomposition.csv",
    "failure_decomposition.md",
)
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class InputError(ValueError):
    """Raised when an input archive violates the paired-analysis contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                raise InputError(f"{path}:{line_number}: row must be a JSON object")
            value["__line__"] = line_number
            rows.append(value)
    if not rows:
        raise InputError(f"{path}: no JSONL records")
    return rows


def _nonempty_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}: expected a non-empty string")
    return value.strip()


def _string_list(value: Any, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise InputError(f"{where}: expected a list of non-empty strings")
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise InputError(f"{where}: duplicate values are not allowed")
    return normalized


def _task_type(row: Mapping[str, Any], *, where: str) -> str:
    request = row.get("request")
    nested = request.get("task_type") if isinstance(request, Mapping) else None
    top_level = row.get("task_type")
    if nested is not None and top_level is not None and nested != top_level:
        raise InputError(f"{where}: conflicting top-level and request.task_type")
    return _nonempty_string(nested if nested is not None else top_level, where=where)


def _index_requests(path: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = _nonempty_string(row.get("id"), where=f"{where}.id")
        if item_id in indexed:
            raise InputError(f"{where}: duplicate id {item_id!r}")
        indexed[item_id] = {
            "id": item_id,
            "variant": _nonempty_string(row.get("variant"), where=f"{where}.variant"),
            "task_type": _task_type(row, where=f"{where}.task_type"),
        }
    return indexed


def _index_rollouts(
    path: Path,
    requests: Mapping[str, Mapping[str, str]],
    *,
    require_all_requests: bool = False,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = _nonempty_string(row.get("id"), where=f"{where}.id")
        if item_id in indexed:
            raise InputError(f"{where}: duplicate id {item_id!r}")
        if item_id not in requests:
            raise InputError(f"{where}: id {item_id!r} is absent from requests")
        passed = row.get("passed")
        if not isinstance(passed, bool):
            raise InputError(f"{where}.passed: expected a JSON boolean")
        failure_types = _string_list(
            row.get("failure_types"), where=f"{where}.failure_types"
        )
        components = _string_list(
            row.get("selected_components"), where=f"{where}.selected_components"
        )
        if passed and failure_types:
            raise InputError(f"{where}: passed row must not contain failure_types")
        if not passed and not failure_types:
            raise InputError(f"{where}: failed row must contain at least one failure_type")
        variant = _nonempty_string(row.get("variant"), where=f"{where}.variant")
        task_type = _task_type(row, where=f"{where}.task_type")
        expected = requests[item_id]
        if variant != expected["variant"]:
            raise InputError(
                f"{where}: variant mismatch for {item_id!r}: "
                f"{variant!r} != {expected['variant']!r}"
            )
        if task_type != expected["task_type"]:
            raise InputError(
                f"{where}: task_type mismatch for {item_id!r}: "
                f"{task_type!r} != {expected['task_type']!r}"
            )
        indexed[item_id] = {
            "passed": passed,
            "failure_types": failure_types,
            "components": components,
        }

    if require_all_requests and set(indexed) != set(requests):
        missing = sorted(set(requests) - set(indexed))[:10]
        raise InputError(f"{path}: missing request IDs: {missing}")
    return indexed


def _parse_labeled_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise InputError(f"--rollouts expects LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not LABEL_RE.fullmatch(label):
            raise InputError(f"invalid rollout label {label!r}")
        if label in result:
            raise InputError(f"duplicate rollout label {label!r}")
        result[label] = Path(raw_path).expanduser().resolve()
    if len(result) < 2:
        raise InputError("at least two --rollouts LABEL=PATH inputs are required")
    return result


def _strata_for_id(
    item_id: str,
    requests: Mapping[str, Mapping[str, str]],
    reference: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    request = requests[item_id]
    ref = reference[item_id]
    failure_values = (
        ref["failure_types"] if not ref["passed"] else ("__reference_pass__",)
    )
    component_values = (
        ref["components"]
        if ref["components"]
        else (("__reference_pass__",) if ref["passed"] else ("__untagged__",))
    )
    values: list[tuple[str, str]] = [
        ("overall", "all"),
        (
            "track",
            "clean" if request["variant"] == "clean" else "perturbed",
        ),
        ("variant", request["variant"]),
        ("task_type", request["task_type"]),
    ]
    values.extend(("failure_type", value) for value in failure_values)
    values.extend(("component", value) for value in component_values)
    return tuple(values)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_analysis(
    requests: Mapping[str, Mapping[str, str]],
    archives: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    reference_label: str,
) -> list[dict[str, Any]]:
    reference = archives[reference_label]
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item_id in sorted(requests):
        for key in _strata_for_id(item_id, requests, reference):
            strata[key].append(item_id)

    rows: list[dict[str, Any]] = []
    for (dimension, stratum), ids in sorted(strata.items()):
        for label, candidate in archives.items():
            both_pass = rescued = regressed = both_fail = 0
            for item_id in ids:
                ref_passed = bool(reference[item_id]["passed"])
                candidate_passed = bool(candidate[item_id]["passed"])
                both_pass += int(ref_passed and candidate_passed)
                rescued += int((not ref_passed) and candidate_passed)
                regressed += int(ref_passed and (not candidate_passed))
                both_fail += int((not ref_passed) and (not candidate_passed))
            reference_passed = both_pass + regressed
            reference_failed = rescued + both_fail
            candidate_passed = both_pass + rescued
            rows.append(
                {
                    "dimension": dimension,
                    "stratum": stratum,
                    "model": label,
                    "reference": reference_label,
                    "n": len(ids),
                    "reference_passed": reference_passed,
                    "reference_failed": reference_failed,
                    "candidate_passed": candidate_passed,
                    "both_pass": both_pass,
                    "rescued": rescued,
                    "regressed": regressed,
                    "both_fail": both_fail,
                    "pass_rate": _rate(candidate_passed, len(ids)),
                    "rescue_rate": _rate(rescued, reference_failed),
                    "regression_rate": _rate(regressed, reference_passed),
                }
            )
    return rows


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


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _pct(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def _markdown(rows: Sequence[Mapping[str, Any]], reference_label: str) -> str:
    overall = [row for row in rows if row["dimension"] == "overall"]
    perturbed = [
        row
        for row in rows
        if row["dimension"] == "track" and row["stratum"] == "perturbed"
    ]
    lines = [
        "# CARE-OPD failure decomposition",
        "",
        f"Reference model: `{reference_label}`.",
        "",
        "Failure and component strata are frozen from the reference output. "
        "`rescue_rate=P(candidate passes | reference fails)` and "
        "`regression_rate=P(candidate fails | reference passes)`. `NA` means "
        "the corresponding denominator is zero.",
        "",
        "## Overall paired outcomes",
        "",
        "| Model | Passed | Pass rate | Rescued | Rescue rate | Regressed | Regression rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['model']} | {row['candidate_passed']}/{row['n']} | "
            f"{_pct(row['pass_rate'])} | {row['rescued']} | "
            f"{_pct(row['rescue_rate'])} | {row['regressed']} | "
            f"{_pct(row['regression_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Perturbed-only paired outcomes",
            "",
            "| Model | Passed | Pass rate | Rescued | Rescue rate | Regressed | Regression rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in perturbed:
        lines.append(
            f"| {row['model']} | {row['candidate_passed']}/{row['n']} | "
            f"{_pct(row['pass_rate'])} | {row['rescued']} | "
            f"{_pct(row['rescue_rate'])} | {row['regressed']} | "
            f"{_pct(row['regression_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Stratum table",
            "",
            "| Dimension | Stratum | Model | n | Pass | Rescue | Regression |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["dimension"] == "overall":
            continue
        lines.append(
            f"| {row['dimension']} | {row['stratum']} | {row['model']} | "
            f"{row['n']} | {_pct(row['pass_rate'])} | "
            f"{_pct(row['rescue_rate'])} | {_pct(row['regression_rate'])} |"
        )
    lines.extend(
        [
            "",
            "This report is descriptive. It does not isolate any single CARE-OPD "
            "component without a matched ablation checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rollout_paths = _parse_labeled_paths(args.rollouts)
    required = tuple(
        item.strip() for item in args.required_labels.split(",") if item.strip()
    )
    missing_labels = sorted(set(required) - set(rollout_paths))
    if missing_labels:
        raise InputError(f"missing required rollout labels: {missing_labels}")
    if args.reference_label not in rollout_paths:
        raise InputError(f"reference label {args.reference_label!r} was not supplied")

    request_path = args.requests.expanduser().resolve()
    requests = _index_requests(request_path)
    archives = {
        label: _index_rollouts(path, requests)
        for label, path in rollout_paths.items()
    }
    paired_ids: set[str] | None = None
    paired_label: str | None = None
    for label, archive in archives.items():
        ids = set(archive)
        if paired_ids is None:
            paired_ids = ids
            paired_label = label
        elif ids != paired_ids:
            raise InputError(
                f"rollout ID set mismatch: {label!r} versus {paired_label!r}; "
                f"missing={sorted(paired_ids - ids)[:10]}, "
                f"extra={sorted(ids - paired_ids)[:10]}"
            )
    assert paired_ids is not None
    paired_requests = {item_id: requests[item_id] for item_id in paired_ids}
    rows = build_analysis(
        paired_requests, archives, reference_label=args.reference_label
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "reference_label": args.reference_label,
        "definitions": {
            "pass_rate": "candidate_passed / n",
            "rescue_rate": "reference_failed_and_candidate_passed / reference_failed",
            "regression_rate": "reference_passed_and_candidate_failed / reference_passed",
            "track": "clean iff variant=clean; otherwise perturbed",
            "failure_strata": "failure_types emitted by the reference model",
            "component_strata": "selected_components emitted by the reference model",
        },
        "inputs": {
            "requests": {
                "path": str(request_path),
                "sha256": _sha256(request_path),
                "rows": len(requests),
                "paired_subset_rows": len(paired_requests),
                "rollouts_are_request_subset": True,
            },
            "rollouts": {
                label: {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "rows": len(archives[label]),
                }
                for label, path in rollout_paths.items()
            },
        },
        "rows": rows,
    }

    output_dir = args.output_dir.expanduser().resolve()
    targets = [output_dir / name for name in OUTPUT_NAMES]
    if not args.overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"outputs already exist: {existing}; pass --overwrite")
    _atomic_text(targets[0], _json_text(payload), overwrite=args.overwrite)
    _atomic_text(targets[1], _csv_text(rows), overwrite=args.overwrite)
    _atomic_text(
        targets[2], _markdown(rows, args.reference_label), overwrite=args.overwrite
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument(
        "--rollouts",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeat once per evaluated checkpoint",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-label", default="clean_sft")
    parser.add_argument(
        "--required-labels",
        default=",".join(DEFAULT_REQUIRED_LABELS),
        help="comma-separated labels that must be present",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run(args)
    overall = [row for row in payload["rows"] if row["dimension"] == "overall"]
    print(json.dumps({"status": "ok", "overall": overall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
