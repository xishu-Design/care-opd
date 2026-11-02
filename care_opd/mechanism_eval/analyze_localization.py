#!/usr/bin/env python3
"""Audit how verifier-selected CARE-OPD repairs localize onto response tokens.

This is a tokenizer-only diagnostic.  It reconstructs the same character spans
used by rollout collection and projects them with the same offset/mask helpers
used by :class:`care_opd.collator.OPDCollator`.  It never loads model weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from care_opd.collator import (
    _encode_response_with_offsets,
    _repair_components,
    _token_span_mask,
)
from care_opd.rollout import component_char_spans


SCHEMA = "care-opd-localization-analysis-v1"
DEFAULT_MAX_LENGTH = 4096
DEFAULT_MAX_RESPONSE_TOKENS = 1200


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
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
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"{path}:{line_number}: row lacks a non-empty id")
            if row_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {row_id!r}")
            seen.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _as_names(value: Any, *, default: str) -> list[str]:
    if value is None:
        return [default]
    if isinstance(value, str):
        return [value or default]
    if isinstance(value, Sequence):
        names = [str(item) for item in value if str(item)]
        return names or [default]
    return [str(value)]


def _union_length(spans: Sequence[Mapping[str, Any]], response_length: int) -> int:
    intervals: list[tuple[int, int]] = []
    for span in spans:
        start = int(span.get("start", span.get("char_start", 0)))
        end = int(span.get("end", span.get("char_end", 0)))
        start = min(max(start, 0), response_length)
        end = min(max(end, 0), response_length)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _finite_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _finite_median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def _project_row(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    max_response_tokens: int | None,
    add_eos: bool,
) -> dict[str, Any]:
    raw_response = row.get("rollout", row.get("student_response", row.get("response")))
    if raw_response is None:
        raise KeyError(f"row {row.get('id')!r} lacks rollout/student_response/response")
    raw_response = str(raw_response)
    risk_weight = float(row.get("risk_weight", 1.0))
    components = _repair_components(row)
    if not components:
        raise ValueError(f"row {row.get('id')!r} has no selected repair components")

    # Reconstruct exactly as online collection does.  Failure-bank rows contain
    # selected components but intentionally do not need persisted tokenizer spans.
    spans = component_char_spans(
        raw_response,
        components,
        risk_weight=risk_weight,
    )
    token_ids, offsets = _encode_response_with_offsets(
        tokenizer,
        raw_response,
        add_eos=add_eos,
    )
    response_cap = max_length - 1
    if max_response_tokens is not None:
        response_cap = min(response_cap, max_response_tokens)
    truncated = len(token_ids) > response_cap
    token_ids = token_ids[:response_cap]
    offsets = offsets[:response_cap]
    selected, weights = _token_span_mask(
        offsets,
        spans,
        base_weight=risk_weight,
        use_explicit_risk_weight=True,
    )

    full_tokens = len(token_ids)
    selected_tokens = sum(bool(value) for value in selected)
    outside_tokens = full_tokens - selected_tokens
    projected_components = sorted(
        {
            str(span.get("component"))
            for span in spans
            if span.get("component")
            and str(span.get("component")) != "__full_response_fallback__"
        }
    )
    full_fallback = any(
        str(span.get("component")) == "__full_response_fallback__" for span in spans
    )
    unprojected_components = sorted(
        component
        for component in set(components)
        if component not in projected_components and not component.startswith("__")
    )
    selected_chars = _union_length(spans, len(raw_response))

    return {
        "id": str(row.get("id")),
        "seed_id": row.get("seed_id"),
        "variant": str(row.get("variant") or "unknown"),
        "failure_types": _as_names(row.get("failure_types"), default="unknown"),
        "selected_components": components,
        "projected_components": projected_components,
        "unprojected_components": unprojected_components,
        "risk_level": str(row.get("risk_level") or "unknown"),
        "risk_weight": risk_weight,
        "response_chars": len(raw_response),
        "selected_chars": selected_chars,
        "char_coverage": _ratio(selected_chars, len(raw_response)),
        "repair_span_count": len(spans),
        "repair_spans": spans,
        "full_response_tokens": full_tokens,
        "selected_tokens": selected_tokens,
        "outside_selected_tokens": outside_tokens,
        "token_coverage": _ratio(selected_tokens, full_tokens),
        "supervision_pollution_multiplier": _ratio(full_tokens, selected_tokens),
        "outside_to_selected_ratio": _ratio(outside_tokens, selected_tokens),
        "risk_weighted_selected_tokens": sum(weights),
        "full_response_fallback": full_fallback,
        "partial_projection": bool(unprojected_components) and not full_fallback,
        "zero_selected_tokens": selected_tokens == 0,
        "response_truncated": truncated,
    }


def _summarize(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = len(examples)
    full_tokens = sum(int(item["full_response_tokens"]) for item in examples)
    selected_tokens = sum(int(item["selected_tokens"]) for item in examples)
    outside_tokens = full_tokens - selected_tokens
    response_chars = sum(int(item["response_chars"]) for item in examples)
    selected_chars = sum(int(item["selected_chars"]) for item in examples)
    fallback_rows = sum(bool(item["full_response_fallback"]) for item in examples)
    partial_rows = sum(bool(item["partial_projection"]) for item in examples)
    zero_rows = sum(bool(item["zero_selected_tokens"]) for item in examples)
    truncated_rows = sum(bool(item["response_truncated"]) for item in examples)
    row_multipliers = [item["supervision_pollution_multiplier"] for item in examples]
    return {
        "rows": rows,
        "repair_span_count": sum(int(item["repair_span_count"]) for item in examples),
        "response_chars": response_chars,
        "selected_chars": selected_chars,
        "char_coverage_micro": _ratio(selected_chars, response_chars),
        "char_coverage_macro": _finite_mean(item["char_coverage"] for item in examples),
        "full_response_tokens": full_tokens,
        "selected_tokens": selected_tokens,
        "outside_selected_tokens": outside_tokens,
        "token_coverage_micro": _ratio(selected_tokens, full_tokens),
        "token_coverage_macro": _finite_mean(item["token_coverage"] for item in examples),
        "supervision_pollution_multiplier": _ratio(full_tokens, selected_tokens),
        "outside_to_selected_ratio": _ratio(outside_tokens, selected_tokens),
        "mean_row_pollution_multiplier": _finite_mean(row_multipliers),
        "median_row_pollution_multiplier": _finite_median(row_multipliers),
        "risk_weighted_selected_tokens": sum(
            float(item["risk_weighted_selected_tokens"]) for item in examples
        ),
        "full_fallback_rows": fallback_rows,
        "full_fallback_rate": _ratio(fallback_rows, rows),
        "partial_projection_rows": partial_rows,
        "partial_projection_rate": _ratio(partial_rows, rows),
        "zero_selected_rows": zero_rows,
        "zero_selected_rate": _ratio(zero_rows, rows),
        "response_truncated_rows": truncated_rows,
        "response_truncated_rate": _ratio(truncated_rows, rows),
    }


def _stratify(
    examples: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        values = example[field]
        names = values if isinstance(values, list) else [values]
        for name in dict.fromkeys(str(value) for value in names):
            grouped[name].append(example)
    return {name: _summarize(grouped[name]) for name in sorted(grouped)}


def _distribution(
    examples: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    occurrence_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    for example in examples:
        values = example[field]
        names = [str(value) for value in values] if isinstance(values, list) else [str(values)]
        occurrence_counts.update(names)
        row_counts.update(set(names))
    return {
        name: {
            "rows": row_counts[name],
            "row_fraction": _ratio(row_counts[name], len(examples)),
            "occurrences": occurrence_counts[name],
        }
        for name in sorted(occurrence_counts)
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_response_tokens: int | None = DEFAULT_MAX_RESPONSE_TOKENS,
    add_eos: bool = True,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot analyze an empty failure bank")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if max_response_tokens is not None and max_response_tokens < 1:
        raise ValueError("max_response_tokens must be positive or None")
    examples = [
        _project_row(
            row,
            tokenizer,
            max_length=max_length,
            max_response_tokens=max_response_tokens,
            add_eos=add_eos,
        )
        for row in rows
    ]
    return {
        "overall": _summarize(examples),
        "field_distribution": _distribution(examples, "selected_components"),
        "failure_type_distribution": _distribution(examples, "failure_types"),
        "variant_distribution": _distribution(examples, "variant"),
        "strata": {
            "by_variant": _stratify(examples, "variant"),
            "by_failure_type": _stratify(examples, "failure_types"),
            "by_component": _stratify(examples, "selected_components"),
        },
        "examples": examples,
    }


def _tokenizer_metadata(tokenizer: Any, requested_path: str) -> dict[str, Any]:
    try:
        vocabulary_size = len(tokenizer)
    except (TypeError, NotImplementedError):
        vocabulary_size = getattr(tokenizer, "vocab_size", None)
    return {
        "requested_path": requested_path,
        "name_or_path": str(getattr(tokenizer, "name_or_path", requested_path)),
        "class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": vocabulary_size,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def _load_tokenizer(
    path: str,
    *,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    # Deliberately import AutoTokenizer only.  No AutoModel class is imported or
    # instantiated anywhere in this diagnostic.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=True,
    )


CSV_FIELDS = [
    "scope",
    "group",
    "rows",
    "repair_span_count",
    "full_response_tokens",
    "selected_tokens",
    "outside_selected_tokens",
    "token_coverage_micro",
    "token_coverage_macro",
    "supervision_pollution_multiplier",
    "outside_to_selected_ratio",
    "mean_row_pollution_multiplier",
    "median_row_pollution_multiplier",
    "full_fallback_rows",
    "full_fallback_rate",
    "partial_projection_rows",
    "partial_projection_rate",
    "zero_selected_rows",
    "zero_selected_rate",
    "response_truncated_rows",
    "response_truncated_rate",
    "response_chars",
    "selected_chars",
    "char_coverage_micro",
    "char_coverage_macro",
    "risk_weighted_selected_tokens",
]


def _csv_text(payload: Mapping[str, Any]) -> str:
    records: list[dict[str, Any]] = []
    records.append({"scope": "overall", "group": "all", **payload["overall"]})
    for scope, groups in payload["strata"].items():
        for group, summary in groups.items():
            records.append({"scope": scope, "group": group, **summary})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().replace("\r\n", "\n")


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _format_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(groups: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Group | Rows | Token coverage | Full/selected | Full fallback | Partial projection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, summary in groups.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(group),
                    str(summary["rows"]),
                    _format_percent(summary["token_coverage_micro"]),
                    _format_number(summary["supervision_pollution_multiplier"]),
                    _format_percent(summary["full_fallback_rate"]),
                    _format_percent(summary["partial_projection_rate"]),
                ]
            )
            + " |"
        )
    return lines


def _markdown_text(payload: Mapping[str, Any]) -> str:
    overall = payload["overall"]
    lines = [
        "# CARE-OPD localization audit",
        "",
        "This is a tokenizer-only mechanism diagnostic; it does not evaluate end-to-end task "
        "success and does not load model weights.",
        "",
        "## Overall",
        "",
        f"- Failure-bank rows: `{overall['rows']}`",
        f"- Repair spans: `{overall['repair_span_count']}`",
        f"- Repair character coverage (micro): `{_format_percent(overall['char_coverage_micro'])}`",
        f"- Repair token coverage (micro): `{_format_percent(overall['token_coverage_micro'])}`",
        "- Full-response/selected-token supervision multiplier: "
        f"`{_format_number(overall['supervision_pollution_multiplier'])}`",
        f"- Outside/selected-token ratio: `{_format_number(overall['outside_to_selected_ratio'])}`",
        f"- Full-response fallback rate: `{_format_percent(overall['full_fallback_rate'])}`",
        f"- Partial-projection rate: `{_format_percent(overall['partial_projection_rate'])}`",
        f"- Zero-selected-token rate: `{_format_percent(overall['zero_selected_rate'])}`",
        f"- Response truncation rate: `{_format_percent(overall['response_truncated_rate'])}`",
        "",
    ]
    headings = {
        "by_variant": "By variant",
        "by_failure_type": "By failure type",
        "by_component": "By selected component",
    }
    for key, heading in headings.items():
        lines.extend([f"## {heading}", ""])
        lines.extend(_markdown_table(payload["strata"][key]))
        lines.append("")
    lines.extend(
        [
            "## Metric definitions and claim boundary",
            "",
            "- Token masks are produced by CARE-OPD's existing response-offset encoder and "
            "span-mask projector after the configured response truncation.",
            "- `Full/selected` is the micro ratio of all response tokens that full-response "
            "training would supervise to verifier-selected repair tokens. EOS is included in "
            "the full-response count when configured, matching the collator's full mask.",
            "- Multi-label failures and components contribute once to every applicable stratum; "
            "stratum row counts therefore must not be summed.",
            "- Full fallback means no selected JSON field could be safely projected and the "
            "rollout helper selected the whole raw response.",
            "- These measurements establish localization sparsity and potential supervision "
            "contamination only; causal performance claims require a matched training ablation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(payload: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    paths = {
        "json": output_dir / "localization_analysis.json",
        "csv": output_dir / "localization_analysis.csv",
        "markdown": output_dir / "localization_analysis.md",
    }
    _atomic_json(paths["json"], payload)
    _atomic_text(paths["csv"], _csv_text(payload))
    _atomic_text(paths["markdown"], _markdown_text(payload))
    return {key: str(path.resolve()) for key, path in paths.items()}


def run(
    failure_bank: Path,
    tokenizer_path: str,
    output_dir: Path,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_response_tokens: int | None = DEFAULT_MAX_RESPONSE_TOKENS,
    add_eos: bool = True,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(failure_bank)
    if tokenizer is None:
        tokenizer = _load_tokenizer(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    analysis = analyze_rows(
        rows,
        tokenizer,
        max_length=max_length,
        max_response_tokens=max_response_tokens,
        add_eos=add_eos,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "failure_bank": str(failure_bank.resolve()),
            "failure_bank_sha256": _sha256(failure_bank),
            "rows": len(rows),
        },
        "tokenizer": _tokenizer_metadata(tokenizer, tokenizer_path),
        "settings": {
            "max_length": max_length,
            "max_response_tokens": max_response_tokens,
            "add_eos": add_eos,
            "mask_mode": "span",
            "span_builder": "care_opd.rollout.component_char_spans",
            "token_projector": "care_opd.collator._token_span_mask",
            "model_weights_loaded": False,
        },
        "definitions": {
            "token_coverage": "selected repair tokens / full response tokens",
            "supervision_pollution_multiplier": (
                "full response tokens / selected repair tokens (micro aggregate)"
            ),
            "outside_to_selected_ratio": (
                "tokens outside selected repair spans / selected repair tokens"
            ),
            "full_fallback": (
                "component_char_spans could not project any selected field and selected the "
                "whole response"
            ),
        },
        **analysis,
        "claim_boundary": (
            "Tokenizer-only offline mechanism diagnostic. It measures localization and the "
            "amount of counterfactual full-response supervision; it is not an end-to-end "
            "accuracy result or a causal OPD ablation."
        ),
    }
    outputs = write_reports(payload, output_dir)
    payload["outputs"] = outputs
    # Re-write JSON once so that the manifest records all three final paths.
    _atomic_json(Path(outputs["json"]), payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-bank", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        "--model",
        dest="tokenizer_path",
        required=True,
        help="Tokenizer directory or Hugging Face model id; model weights are never loaded.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=DEFAULT_MAX_RESPONSE_TOKENS,
        help="Response-axis cap used by the formal QZBC OPD collator (default: 1200).",
    )
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run(
        args.failure_bank,
        args.tokenizer_path,
        args.output_dir,
        max_length=args.max_length,
        max_response_tokens=args.max_response_tokens,
        add_eos=not args.no_eos,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"JSON: {payload['outputs']['json']}")
    print(f"CSV: {payload['outputs']['csv']}")
    print(f"Markdown: {payload['outputs']['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
