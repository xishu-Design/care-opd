#!/usr/bin/env python3
"""Build result-blind causal-mask and fresh/stale OPD data views.

The module intentionally does not import the training stack.  It validates the
published ``qzbc_opd.jsonl`` contract, transforms only ``repair_spans`` for the
localisation controls, and writes all artifacts atomically.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "care_opd.causal_ablation_views.v1"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_id(row: Mapping[str, Any], *, where: str) -> str:
    value = row.get("id")
    if value is None:
        value = row.get("sample_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: non-empty string id or sample_id is required")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rid = _row_id(row, where=f"{path}:{line_no}")
            if rid in seen:
                raise ValueError(f"{path}:{line_no}: duplicate row id {rid!r}")
            seen.add(rid)
            _validate_opd_row(row, where=f"{path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def _validate_opd_row(row: Mapping[str, Any], *, where: str) -> None:
    rollout = row.get("rollout")
    if not isinstance(rollout, str) or not rollout:
        raise ValueError(f"{where}: non-empty rollout is required")
    if row.get("teacher_messages") is None and row.get("teacher_prompt") is None:
        raise ValueError(f"{where}: teacher_messages or teacher_prompt is required")
    spans = row.get("repair_spans")
    if not isinstance(spans, list):
        raise ValueError(f"{where}: repair_spans must be a list")
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError(f"{where}: repair_spans[{index}] must be an object")
        start, end = span.get("start"), span.get("end")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{where}: repair_spans[{index}] bounds must be integers")
        if not (0 <= start < end <= len(rollout)):
            raise ValueError(
                f"{where}: repair_spans[{index}] out of bounds for rollout length {len(rollout)}"
            )
    if "clean_control" in row and not isinstance(row["clean_control"], bool):
        raise ValueError(f"{where}: clean_control must be boolean")
    if "passed" in row and not isinstance(row["passed"], bool):
        raise ValueError(f"{where}: passed must be boolean")


def _merged_spans(row: Mapping[str, Any]) -> list[tuple[int, int]]:
    intervals = sorted((int(x["start"]), int(x["end"])) for x in row["repair_spans"])
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _complement(length: int, intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    for start, end in intervals:
        if cursor < start:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length:
        result.append((cursor, length))
    return result


def _positions_to_spans(positions: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(positions))
    if not ordered:
        return []
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position != previous + 1:
            result.append((start, previous + 1))
            start = position
        previous = position
    result.append((start, previous + 1))
    return result


def _span_objects(
    intervals: Sequence[tuple[int, int]], *, component: str, risk_weight: float
) -> list[dict[str, Any]]:
    return [
        {"start": start, "end": end, "component": component, "risk_weight": risk_weight}
        for start, end in intervals
    ]


def _random_equal_length(row: dict[str, Any], *, seed: int) -> tuple[dict[str, Any], bool]:
    result = copy.deepcopy(row)
    if bool(row.get("clean_control")):
        return result, False
    selected = _merged_spans(row)
    target = sum(end - start for start, end in selected)
    outside = [position for start, end in _complement(len(row["rollout"]), selected) for position in range(start, end)]
    if target == 0:
        return result, True
    if len(outside) < target:
        # Exact equal length is mathematically impossible when the selected
        # region occupies more than half the response.  Preserve the target
        # and make the degeneracy auditable rather than silently dropping a row.
        result["ablation_control"] = {
            "kind": "random_equal_length",
            "degenerate": True,
            "reason": "insufficient_outside_characters",
            "target_characters": target,
            "available_outside_characters": len(outside),
        }
        return result, True
    rid = _row_id(row, where="row")
    local_seed = int.from_bytes(
        hashlib.sha256(f"{seed}\0{rid}\0{row['rollout']}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = random.Random(local_seed)
    chosen = rng.sample(outside, target)
    intervals = _positions_to_spans(chosen)
    risk_weight = float(row.get("risk_weight", 1.0))
    result["repair_spans"] = _span_objects(
        intervals, component="__random_equal_length__", risk_weight=risk_weight
    )
    result["ablation_control"] = {
        "kind": "random_equal_length",
        "degenerate": False,
        "seed": seed,
        "selected_characters": target,
    }
    return result, False


def _outside_only(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    result = copy.deepcopy(row)
    if bool(row.get("clean_control")):
        return result, False
    outside = _complement(len(row["rollout"]), _merged_spans(row))
    risk_weight = float(row.get("risk_weight", 1.0))
    result["repair_spans"] = _span_objects(
        outside, component="__outside_causal_span__", risk_weight=risk_weight
    )
    degenerate = not outside
    result["ablation_control"] = {
        "kind": "outside_only",
        "degenerate": degenerate,
        "selected_characters": sum(end - start for start, end in outside),
    }
    return result, degenerate


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _write_manifest(output_dir: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(output_dir / "manifest.json", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def build_causal_views(input_path: Path, output_dir: Path, *, seed: int) -> dict[str, Any]:
    rows = _read_jsonl(input_path)
    full = copy.deepcopy(rows)
    random_rows: list[dict[str, Any]] = []
    outside_rows: list[dict[str, Any]] = []
    random_degenerate = outside_degenerate = 0
    for row in rows:
        transformed, degenerate = _random_equal_length(row, seed=seed)
        random_rows.append(transformed)
        random_degenerate += int(degenerate)
        transformed, degenerate = _outside_only(row)
        outside_rows.append(transformed)
        outside_degenerate += int(degenerate)
    files = {
        "full_selected": output_dir / "full_selected.jsonl",
        "random_equal_length": output_dir / "random_equal_length.jsonl",
        "outside_only": output_dir / "outside_only.jsonl",
    }
    for name, data in (("full_selected", full), ("random_equal_length", random_rows), ("outside_only", outside_rows)):
        _atomic_text(files[name], _jsonl(data))
    ids = [_row_id(row, where="row") for row in rows]
    manifest = {
        "schema_version": SCHEMA,
        "mode": "causal",
        "result_blind": True,
        "source": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "seed": seed,
        "rows": len(rows),
        "clean_controls": sum(bool(row.get("clean_control")) for row in rows),
        "id_order_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "views": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path), "rows": len(rows)}
            for name, path in files.items()
        },
        "diagnostics": {
            "random_equal_length_degenerate_rows": random_degenerate,
            "outside_only_degenerate_rows": outside_degenerate,
            "length_unit": "unicode_codepoints",
            "token_length_caveat": "Exact equality is guaranteed in character offsets, not tokenizer tokens.",
        },
    }
    _write_manifest(output_dir, manifest)
    return manifest


def build_freshness_views(
    fresh_path: Path,
    stale_path: Path,
    output_dir: Path,
    *,
    seed: int,
    max_rows: int | None,
    min_rows: int,
) -> dict[str, Any]:
    fresh_rows = _read_jsonl(fresh_path)
    stale_rows = _read_jsonl(stale_path)
    if _sha256(fresh_path) == _sha256(stale_path):
        raise ValueError("fresh and stale inputs are byte-identical; freshness contrast is undefined")
    fresh = {_row_id(row, where="fresh row"): row for row in fresh_rows}
    stale = {_row_id(row, where="stale row"): row for row in stale_rows}
    common = sorted(
        rid
        for rid in set(fresh) & set(stale)
        if not bool(fresh[rid].get("clean_control"))
        and not bool(stale[rid].get("clean_control"))
    )
    rng = random.Random(seed)
    rng.shuffle(common)
    if max_rows is not None:
        common = common[:max_rows]
    if len(common) < min_rows:
        raise ValueError(f"only {len(common)} common IDs; require at least {min_rows}")
    # The same deterministic ID order is the sole matching rule. No outcome,
    # failure, risk, or teacher field participates in selection.
    fresh_matched = [fresh[rid] for rid in common]
    stale_matched = [stale[rid] for rid in common]
    files = {
        "fresh": output_dir / "fresh_matched.jsonl",
        "stale": output_dir / "stale_matched.jsonl",
    }
    _atomic_text(files["fresh"], _jsonl(fresh_matched))
    _atomic_text(files["stale"], _jsonl(stale_matched))
    manifest = {
        "schema_version": SCHEMA,
        "mode": "freshness",
        "result_blind": True,
        "matching_rule": "seeded shuffle of exact non-clean OPD row-ID intersection; no held-out evaluation result fields consulted",
        "cohort": "persistent_failure_only",
        "seed": seed,
        "max_rows": max_rows,
        "rows": len(common),
        "id_order_sha256": hashlib.sha256("\n".join(common).encode("utf-8")).hexdigest(),
        "inputs": {
            "fresh": {"path": str(fresh_path.resolve()), "sha256": _sha256(fresh_path), "rows": len(fresh_rows)},
            "stale": {"path": str(stale_path.resolve()), "sha256": _sha256(stale_path), "rows": len(stale_rows)},
        },
        "views": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path), "rows": len(common)}
            for name, path in files.items()
        },
        "chronology_caveat": "The script records but cannot infer collection chronology; the caller must supply genuinely current-policy fresh and prior-policy stale banks. Exact-ID matching conditions on failures present in both banks (the persistent-failure cohort), so it does not estimate all-request freshness effects.",
    }
    _write_manifest(output_dir, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    causal = sub.add_parser("causal", help="build selected/random/outside mask views")
    causal.add_argument("--input", type=Path, required=True)
    causal.add_argument("--output-dir", type=Path, required=True)
    causal.add_argument("--seed", type=int, default=42)
    freshness = sub.add_parser("freshness", help="build result-blind matched fresh/stale views")
    freshness.add_argument("--fresh", type=Path, required=True)
    freshness.add_argument("--stale", type=Path, required=True)
    freshness.add_argument("--output-dir", type=Path, required=True)
    freshness.add_argument("--seed", type=int, default=42)
    freshness.add_argument("--max-rows", type=int)
    freshness.add_argument("--min-rows", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "causal":
        manifest = build_causal_views(args.input, args.output_dir, seed=args.seed)
    else:
        if args.max_rows is not None and args.max_rows < 1:
            raise ValueError("--max-rows must be positive")
        if args.min_rows < 1:
            raise ValueError("--min-rows must be positive")
        manifest = build_freshness_views(
            args.fresh,
            args.stale,
            args.output_dir,
            seed=args.seed,
            max_rows=args.max_rows,
            min_rows=args.min_rows,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
