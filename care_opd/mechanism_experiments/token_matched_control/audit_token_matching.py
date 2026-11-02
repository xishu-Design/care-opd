#!/usr/bin/env python3
"""Audit exact per-example token matching for the random-mask control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from care_opd.collator import (
    OPDCollator,
    _encode_response_with_offsets,
    _random_token_matched_mask,
    _stable_row_identity,
    _token_span_mask,
)

SCHEMA = "care_opd.token_matched_mask_audit.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_PATHS = {
    "collator": PROJECT_ROOT / "src/care_opd/collator.py",
    "token_audit": Path(__file__).resolve(),
}


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
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    )
    try:
        with handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        raise


def _response(row: Mapping[str, Any]) -> str:
    for key in ("rollout", "student_response", "response"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError("row lacks rollout/student_response/response")


def _group_values(row: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    yield "overall", "all"
    for dimension in ("variant", "task_type", "risk_level"):
        value = row.get(dimension)
        if value is not None and str(value):
            yield dimension, str(value)
    failures = row.get("failure_types") or []
    if isinstance(failures, str):
        failures = [failures]
    for value in failures:
        yield "failure_type", str(value)


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = len(records)
    selected = sum(int(record["selected_tokens"]) for record in records)
    candidates = sum(int(record["candidate_tokens"]) for record in records)
    identical = sum(bool(record["identical_mask"]) for record in records)
    zero = sum(int(record["selected_tokens"]) == 0 for record in records)
    full = sum(
        int(record["selected_tokens"]) == int(record["candidate_tokens"])
        and int(record["candidate_tokens"]) > 0
        for record in records
    )
    jaccards = [float(record["jaccard"]) for record in records]
    return {
        "rows": rows,
        "selected_tokens": selected,
        "candidate_tokens": candidates,
        "selected_fraction_micro": selected / candidates if candidates else None,
        "exact_count_match_rate": sum(bool(record["count_match"]) for record in records)
        / rows,
        "exact_weight_multiset_match_rate": sum(
            bool(record["weight_multiset_match"]) for record in records
        )
        / rows,
        "identical_mask_rate": identical / rows,
        "zero_selected_rate": zero / rows,
        "full_selected_rate": full / rows,
        "response_truncation_rows": sum(
            bool(record["response_truncated"]) for record in records
        ),
        "zero_width_selected_tokens": sum(
            int(record["zero_width_selected_tokens"]) for record in records
        ),
        "semantic_weight_sum": sum(
            float(record["semantic_weight_sum"]) for record in records
        ),
        "random_weight_sum": sum(
            float(record["random_weight_sum"]) for record in records
        ),
        "active_semantic_tokens": sum(
            int(record["active_semantic_tokens"]) for record in records
        ),
        "active_random_tokens": sum(
            int(record["active_random_tokens"]) for record in records
        ),
        "mean_jaccard": statistics.fmean(jaccards),
        "median_jaccard": statistics.median(jaccards),
    }


def analyze(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    seed: int,
    max_length: int,
    max_response_tokens: int,
    add_eos: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collator = OPDCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        max_response_length=max_response_tokens,
        add_eos=add_eos,
        use_risk_weighting=True,
    )
    records: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    identities: set[str] = set()
    for row in rows:
        if row.get("id") is None or not str(row.get("id")):
            raise ValueError("formal token-matched audit requires a non-empty id")
        identity = _stable_row_identity(row)
        if identity in identities:
            raise ValueError(f"duplicate row id: {identity}")
        identities.add(identity)
        response = _response(row)
        response_ids, offsets = _encode_response_with_offsets(
            tokenizer, response, add_eos=add_eos
        )
        response_tokens_before_truncation = len(response_ids)
        cap = min(max_length - 1, max_response_tokens)
        response_ids, offsets = response_ids[:cap], offsets[:cap]
        if not response_ids:
            raise ValueError(f"{identity}: empty response after tokenization")
        risk_weight = collator._risk_weight(row)
        spans = row.get("repair_spans", row.get("component_spans", [])) or []
        selected, weights = _token_span_mask(
            offsets,
            spans,
            base_weight=risk_weight,
            use_explicit_risk_weight=True,
        )
        random_selected, random_weights = _random_token_matched_mask(
            offsets,
            selected,
            weights,
            row_identity=identity,
            seed=seed,
        )
        source = {index for index, value in enumerate(selected) if value}
        random = {index for index, value in enumerate(random_selected) if value}
        union = source | random
        candidate_tokens = sum(end > start for start, end in offsets)
        host_verified_value = row.get("host_repair_verified")
        if host_verified_value is None:
            host_verified_value = row.get("host_verified", False)
        host_verified = bool(host_verified_value)
        record = {
            "id": identity,
            "selected_tokens": len(source),
            "candidate_tokens": candidate_tokens,
            "count_match": len(source) == len(random),
            "weight_multiset_match": sorted(weights[index] for index in source)
            == sorted(random_weights[index] for index in random),
            "identical_mask": source == random,
            "intersection_tokens": len(source & random),
            "jaccard": len(source & random) / len(union) if union else 1.0,
            "response_tokens_before_truncation": response_tokens_before_truncation,
            "response_tokens_after_truncation": len(response_ids),
            "response_truncated": response_tokens_before_truncation > len(response_ids),
            "zero_width_selected_tokens": sum(
                bool(value) and offsets[index][1] <= offsets[index][0]
                for index, value in enumerate(random_selected)
            ),
            "semantic_weight_sum": sum(float(weights[index]) for index in source),
            "random_weight_sum": sum(
                float(random_weights[index]) for index in random
            ),
            "active_semantic_tokens": len(source) if host_verified else 0,
            "active_random_tokens": len(random) if host_verified else 0,
        }
        if not record["count_match"] or not record["weight_multiset_match"]:
            raise AssertionError(f"{identity}: token-matching contract violated")
        if record["zero_width_selected_tokens"]:
            raise AssertionError(f"{identity}: random mask selected zero-width token")
        if record["active_semantic_tokens"] != record["active_random_tokens"]:
            raise AssertionError(f"{identity}: effective active-token mismatch")
        records.append(record)
        for key in _group_values(row):
            grouped[key].append(record)

    strata = [
        {"dimension": dimension, "group": group, **_summarize(values)}
        for (dimension, group), values in sorted(grouped.items())
    ]
    return {
        "schema_version": SCHEMA,
        "status": "COMPLETE",
        "random_mask_seed": seed,
        "max_length": max_length,
        "max_response_tokens": max_response_tokens,
        "add_eos": add_eos,
        "overall": _summarize(records),
        "strata": strata,
        "claim_boundary": (
            "This proves exact tokenizer-level selected/effective token-count and "
            "weight matching for the trust-off localization control and reports "
            "mask overlap. It does not measure task performance or causal effect."
        ),
    }, records


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output_dir / "token_matching_audit.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    fieldnames = list(summary["strata"][0].keys())
    temp = output_dir / ".token_matching_audit.csv.tmp"
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["strata"])
    os.replace(temp, output_dir / "token_matching_audit.csv")
    record_fields = list(records[0].keys())
    temp = output_dir / ".token_matching_examples.csv.tmp"
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, output_dir / "token_matching_examples.csv")
    overall = summary["overall"]
    lines = [
        "# CARE-OPD exact token-matched mask audit",
        "",
        f"- Rows: `{overall['rows']}`",
        f"- Selected response tokens: `{overall['selected_tokens']}`",
        f"- Candidate response tokens: `{overall['candidate_tokens']}`",
        f"- Exact count match: `{overall['exact_count_match_rate']:.2%}`",
        "- Exact selected-weight multiset match: "
        f"`{overall['exact_weight_multiset_match_rate']:.2%}`",
        f"- Identical semantic/random mask: `{overall['identical_mask_rate']:.2%}`",
        f"- Mean mask Jaccard: `{overall['mean_jaccard']:.4f}`",
        f"- Response truncation rows: `{overall['response_truncation_rows']}`",
        f"- Zero-width selected tokens: `{overall['zero_width_selected_tokens']}`",
        f"- Effective selected tokens (semantic/random): "
        f"`{overall['active_semantic_tokens']}/{overall['active_random_tokens']}`",
        "",
        "The control is sampled after tokenization and preserves each row's exact "
        "selected-token count and loss-weight multiset. This is a mask audit, not a "
        "performance result.",
        "",
    ]
    _atomic_text(output_dir / "token_matching_audit.md", "\n".join(lines))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=1200)
    parser.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_length < 2 or args.max_response_tokens < 1:
        raise ValueError("length limits must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = _read_jsonl(args.data.resolve())
    summary, records = analyze(
        rows,
        tokenizer=tokenizer,
        seed=args.seed,
        max_length=args.max_length,
        max_response_tokens=args.max_response_tokens,
        add_eos=args.add_eos,
    )
    summary["source"] = {
        "path": str(args.data.resolve()),
        "sha256": _sha256(args.data.resolve()),
        "rows": len(rows),
    }
    summary["tokenizer"] = args.tokenizer
    from care_opd.models import tokenizer_fingerprint

    summary["tokenizer_sha256"] = tokenizer_fingerprint(tokenizer)
    summary["implementation"] = {
        label: {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for label, path in IMPLEMENTATION_PATHS.items()
    }
    _write_outputs(args.output_dir.resolve(), summary, records)
    print(f"TOKEN_MATCHING_AUDIT=COMPLETE rows={len(rows)}")
    print(args.output_dir.resolve() / "token_matching_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
