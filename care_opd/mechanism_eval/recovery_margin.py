#!/usr/bin/env python3
"""Teacher-forced recovery-preference diagnostic for frozen checkpoints.

For every row in a fixed held-out failure bank, the script scores the
verifier-valid oracle response and the original rejected response under the
same public QZBC chat prompt.  It performs forward passes only: there is no
generation, backward pass, optimizer, or parameter update.

The primary margin is::

    mean_logp(oracle response | public prompt)
      - mean_logp(rejected response | the same public prompt)

Selected-component margins are reported as localized only when every requested
top-level component can be projected on both responses.  Explicit full-response
fallbacks and partial projections are retained as separate diagnostics instead
of being mixed into the localized mean.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import re
import statistics
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from care_opd.collator import (
    _encode,
    _encode_response_with_offsets,
    _oracle_component_spans,
    _token_span_mask,
    render_prompt,
)
from care_opd.losses import selected_causal_log_probs, supports_selected_logits
from care_opd.models import (
    ModelArguments,
    load_causal_lm,
    load_tokenizer,
    tokenizer_fingerprint,
)
from care_opd.qzbc_contract import canonical_response
from care_opd.rollout import component_char_spans


SCHEMA = "care-opd-recovery-margin-v1"
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
OUTPUT_JSONL = "recovery_margin.jsonl"
SUMMARY_JSON = "recovery_margin_summary.json"
SUMMARY_CSV = "recovery_margin_summary.csv"
SUMMARY_MD = "recovery_margin_summary.md"
MANIFEST_JSON = "run_manifest.json"


class InputError(ValueError):
    """Raised when an input violates the fixed scoring contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _model_reference(value: str) -> str:
    """Canonicalize local paths while preserving Hugging Face identifiers."""

    path = Path(value)
    return str(path.resolve()) if path.exists() else value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
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


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        if allow_empty:
            return []
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
                raise InputError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows and not allow_empty:
        raise InputError(f"{path}: no records")
    return rows


def _load_failure_bank(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        where = f"{path}:{index}"
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise InputError(f"{where}: non-empty id required")
        if row_id in seen:
            raise InputError(f"{where}: duplicate id {row_id!r}")
        seen.add(row_id)
        if not isinstance(row.get("student_messages"), list):
            raise InputError(f"{where}: student_messages list required")
        if not isinstance(row.get("rollout"), str):
            raise InputError(f"{where}: raw rollout string required")
        if row.get("oracle_response") is None:
            raise InputError(f"{where}: oracle_response required")
        components = row.get("selected_components")
        if not isinstance(components, list) or not components or any(
            not isinstance(component, str) or not component for component in components
        ):
            raise InputError(f"{where}: non-empty selected_components string list required")
        copy = dict(row)
        copy["oracle_response"] = canonical_response(row["oracle_response"])
        validated.append(copy)
    return validated


def _select_rows(
    rows: Sequence[dict[str, Any]], *, max_examples: int, selection_seed: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{selection_seed}|{row['id']}".encode("utf-8")
            ).hexdigest(),
            row["id"],
        ),
    )
    return ranked if max_examples <= 0 else ranked[:max_examples]


def _parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(f"invalid model label {label!r}")
    if not path:
        raise argparse.ArgumentTypeError("model path must not be empty")
    return label, path


def _suite_binding(
    path: Path | None,
    models: Sequence[tuple[str, str]],
) -> dict[str, Any] | None:
    """Bind requested labels/paths to the frozen formal evaluation suite.

    This is intentionally a cheap contract check: the formal QZBC evaluator
    already produced the full checkpoint inventories.  We bind their recorded
    fingerprints here without re-hashing multi-GB weights.
    """

    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    entries = payload.get("models")
    suite_fingerprint = str(payload.get("suite_fingerprint") or "")
    if not isinstance(entries, list) or not re.fullmatch(
        r"[0-9a-f]{64}", suite_fingerprint
    ):
        raise InputError(f"invalid frozen model suite: {resolved}")
    bound: list[dict[str, str]] = []
    for label, model_path in models:
        matches = [entry for entry in entries if entry.get("label") == label]
        if len(matches) != 1:
            raise InputError(
                f"model suite must contain label {label!r} exactly once"
            )
        entry = matches[0]
        suite_path = _model_reference(str(entry.get("canonical_path") or ""))
        requested_path = _model_reference(model_path)
        fingerprint = str(entry.get("model_fingerprint") or "")
        if suite_path != requested_path:
            raise InputError(
                f"model path mismatch for {label}: {requested_path!r} != {suite_path!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise InputError(f"invalid model fingerprint for {label} in {resolved}")
        bound.append(
            {
                "label": label,
                "canonical_path": suite_path,
                "model_fingerprint": fingerprint,
            }
        )
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "suite_fingerprint": suite_fingerprint,
        "models": bound,
        "identity_scope": "recorded formal suite; weights not rehashed by this diagnostic",
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mask_from_spans(
    offsets: Sequence[tuple[int, int]], spans: Sequence[Mapping[str, Any]]
) -> list[bool]:
    selected, _ = _token_span_mask(
        offsets,
        spans,
        base_weight=1.0,
        use_explicit_risk_weight=False,
    )
    return [bool(value) for value in selected]


def _projection(
    response: str,
    offsets: Sequence[tuple[int, int]],
    components: Sequence[str],
    *,
    oracle: bool,
) -> dict[str, Any]:
    requested = list(dict.fromkeys(str(component) for component in components))
    synthetic = any(component.startswith("__") for component in requested)
    if oracle:
        if synthetic:
            spans = [
                {
                    "start": 0,
                    "end": max(len(response), 1),
                    "component": "__full_response_fallback__",
                }
            ]
        else:
            # The helper is fail-closed and returns a full span if any requested
            # field is absent.  For this diagnostic, distinguish a genuine
            # partial projection from that training fallback by projecting each
            # requested field independently.
            spans = []
            for component in requested:
                component_spans = _oracle_component_spans(response, [component])
                if component_spans and not any(
                    str(span.get("component")) == "__full_response_fallback__"
                    or (
                        int(span.get("start", 0)) == 0
                        and int(span.get("end", 0)) >= max(len(response), 1)
                    )
                    for span in component_spans
                ):
                    spans.extend(
                        {**span, "component": component} for span in component_spans
                    )
    else:
        spans = component_char_spans(response, requested, risk_weight=1.0)
    fallback = any(
        str(span.get("component")) == "__full_response_fallback__" for span in spans
    )
    projected = {
        str(span.get("component"))
        for span in spans
        if span.get("component")
        and str(span.get("component")) != "__full_response_fallback__"
    }
    real_requested = {component for component in requested if not component.startswith("__")}
    unprojected = sorted(real_requested - projected)
    coverage = (
        len(projected & real_requested) / len(real_requested)
        if real_requested
        else (0.0 if fallback else 1.0)
    )
    mask = _mask_from_spans(offsets, spans)
    per_component: dict[str, list[bool]] = {}
    for component in sorted(projected):
        component_spans = [
            span for span in spans if str(span.get("component")) == component
        ]
        per_component[component] = _mask_from_spans(offsets, component_spans)
    return {
        "mask": mask,
        "per_component_masks": per_component,
        "fallback": fallback,
        "coverage": coverage,
        "projected_components": sorted(projected),
        "unprojected_components": unprojected,
        "spans": [dict(span) for span in spans],
    }


def _prepare_pair(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    max_response_tokens: int | None,
    add_eos: bool,
) -> dict[str, Any]:
    prompt = render_prompt(
        tokenizer,
        row,
        text_field="student_prompt",
        messages_field="student_messages",
    )
    prompt_ids = _encode(tokenizer, prompt, add_special_tokens=False)
    if not prompt_ids:
        raise InputError(f"row {row.get('id')!r}: tokenizer produced empty prompt")
    oracle = canonical_response(row["oracle_response"])
    rejected = str(row["rollout"])
    oracle_ids, oracle_offsets = _encode_response_with_offsets(
        tokenizer, oracle, add_eos=add_eos
    )
    rejected_ids, rejected_offsets = _encode_response_with_offsets(
        tokenizer, rejected, add_eos=add_eos
    )
    cap = max_length - 1
    if max_response_tokens is not None:
        cap = min(cap, max_response_tokens)
    if cap < 1:
        raise InputError("response token cap must be positive")
    oracle_truncated = len(oracle_ids) > cap
    rejected_truncated = len(rejected_ids) > cap
    oracle_ids, oracle_offsets = oracle_ids[:cap], oracle_offsets[:cap]
    rejected_ids, rejected_offsets = rejected_ids[:cap], rejected_offsets[:cap]
    if not oracle_ids or not rejected_ids:
        raise InputError(
            f"row {row.get('id')!r}: oracle/rejected response has no scoreable token"
        )

    # Both alternatives receive exactly the same suffix of the public prompt.
    # Fitting against the longer alternative prevents response-dependent prompt
    # truncation from contaminating a preference margin.
    prompt_room = max_length - max(len(oracle_ids), len(rejected_ids))
    if prompt_room < 1:
        raise InputError(f"row {row.get('id')!r}: no room for a causal prompt token")
    fitted_prompt = prompt_ids[-prompt_room:]
    components = [str(component) for component in row["selected_components"]]
    oracle_projection = _projection(
        oracle, oracle_offsets, components, oracle=True
    )
    rejected_projection = _projection(
        rejected, rejected_offsets, components, oracle=False
    )
    return {
        "prompt": prompt,
        "prompt_ids": fitted_prompt,
        "oracle": {
            "text": oracle,
            "ids": oracle_ids,
            "offsets": oracle_offsets,
            "projection": oracle_projection,
            "truncated": oracle_truncated,
        },
        "rejected": {
            "text": rejected,
            "ids": rejected_ids,
            "offsets": rejected_offsets,
            "projection": rejected_projection,
            "truncated": rejected_truncated,
        },
    }


def _score_response_tokens(
    model: Any,
    *,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    device: torch.device,
    chunk_size: int,
) -> list[float]:
    """Return one causal log-probability per response token.

    Each forward selects only ``chunk_size`` response prediction positions via
    ``logits_to_keep``.  Qwen3.5 therefore never materializes a 4096 x vocab
    logit tensor for this diagnostic.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    sequence = list(prompt_ids) + list(response_ids)
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    prompt_length = len(prompt_ids)
    collected: list[float] = []
    for start in range(0, len(response_ids), chunk_size):
        stop = min(start + chunk_size, len(response_ids))
        label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        label_mask[:, prompt_length + start : prompt_length + stop] = True
        with torch.inference_mode():
            _, token_logps, selected_mask, efficient = selected_causal_log_probs(
                model, input_ids, attention, label_mask
            )
        if not efficient:
            raise RuntimeError(
                "checkpoint does not honor logits_to_keep; refusing to materialize "
                "full-sequence vocabulary logits"
            )
        values = token_logps[0][selected_mask[0]].detach().float().cpu().tolist()
        if len(values) != stop - start:
            raise RuntimeError(
                f"causal token alignment failed: expected {stop-start}, got {len(values)}"
            )
        collected.extend(float(value) for value in values)
    if len(collected) != len(response_ids):
        raise RuntimeError("response token scoring returned the wrong length")
    return collected


def _masked_stats(logps: Sequence[float], mask: Sequence[bool]) -> dict[str, Any]:
    if len(logps) != len(mask):
        raise ValueError("log-probability and mask lengths differ")
    selected = [float(value) for value, keep in zip(logps, mask) if keep]
    if not selected:
        return {"tokens": 0, "logp_sum": None, "mean_logp": None, "nll": None}
    total = float(sum(selected))
    mean_logp = total / len(selected)
    return {
        "tokens": len(selected),
        "logp_sum": total,
        "mean_logp": mean_logp,
        "nll": -mean_logp,
    }


def _margin(oracle: Mapping[str, Any], rejected: Mapping[str, Any]) -> float | None:
    left, right = oracle.get("mean_logp"), rejected.get("mean_logp")
    return float(left) - float(right) if left is not None and right is not None else None


def _score_example(
    row: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    *,
    model_label: str,
    model_path: str,
    device: torch.device,
    max_length: int,
    max_response_tokens: int | None,
    chunk_size: int,
    add_eos: bool,
) -> dict[str, Any]:
    pair = _prepare_pair(
        row,
        tokenizer,
        max_length=max_length,
        max_response_tokens=max_response_tokens,
        add_eos=add_eos,
    )
    oracle_logps = _score_response_tokens(
        model,
        prompt_ids=pair["prompt_ids"],
        response_ids=pair["oracle"]["ids"],
        device=device,
        chunk_size=chunk_size,
    )
    rejected_logps = _score_response_tokens(
        model,
        prompt_ids=pair["prompt_ids"],
        response_ids=pair["rejected"]["ids"],
        device=device,
        chunk_size=chunk_size,
    )
    oracle_full = _masked_stats(oracle_logps, [True] * len(oracle_logps))
    rejected_full = _masked_stats(rejected_logps, [True] * len(rejected_logps))
    op = pair["oracle"]["projection"]
    rp = pair["rejected"]["projection"]
    oracle_selected = _masked_stats(oracle_logps, op["mask"])
    rejected_selected = _masked_stats(rejected_logps, rp["mask"])
    oracle_outside = _masked_stats(oracle_logps, [not value for value in op["mask"]])
    rejected_outside = _masked_stats(
        rejected_logps, [not value for value in rp["mask"]]
    )
    localized = bool(
        not op["fallback"]
        and not rp["fallback"]
        and op["coverage"] == 1.0
        and rp["coverage"] == 1.0
        and oracle_selected["tokens"]
        and rejected_selected["tokens"]
    )
    fallback = bool(op["fallback"] or rp["fallback"])
    selected_margin = _margin(oracle_selected, rejected_selected)
    classification = (
        "localized_complete"
        if localized
        else "full_response_fallback"
        if fallback
        else "partial_or_missing_projection"
    )
    per_component: dict[str, Any] = {}
    shared_components = sorted(
        set(op["per_component_masks"]) & set(rp["per_component_masks"])
    )
    for component in shared_components:
        left = _masked_stats(oracle_logps, op["per_component_masks"][component])
        right = _masked_stats(rejected_logps, rp["per_component_masks"][component])
        per_component[component] = {
            "oracle": left,
            "rejected": right,
            "margin": _margin(left, right),
        }
    return {
        "schema": SCHEMA,
        "model_label": model_label,
        "model_path": _model_reference(model_path),
        "id": row["id"],
        "seed_id": row.get("seed_id"),
        "variant": row.get("variant"),
        "failure_types": list(row.get("failure_types") or []),
        "selected_components": list(row["selected_components"]),
        "prompt_sha256": _text_sha256(pair["prompt"]),
        "shared_prompt_tokens": len(pair["prompt_ids"]),
        "rejected_response": pair["rejected"]["text"],
        "rejected_response_sha256": _text_sha256(pair["rejected"]["text"]),
        "oracle_response": pair["oracle"]["text"],
        "oracle_response_sha256": _text_sha256(pair["oracle"]["text"]),
        "oracle_truncated": pair["oracle"]["truncated"],
        "rejected_truncated": pair["rejected"]["truncated"],
        "full": {
            "oracle": oracle_full,
            "rejected": rejected_full,
            "margin": _margin(oracle_full, rejected_full),
        },
        "selected": {
            "classification": classification,
            "localized_complete": localized,
            "full_response_fallback": fallback,
            "oracle_projection_coverage": op["coverage"],
            "rejected_projection_coverage": rp["coverage"],
            "oracle_projected_components": op["projected_components"],
            "rejected_projected_components": rp["projected_components"],
            "oracle_unprojected_components": op["unprojected_components"],
            "rejected_unprojected_components": rp["unprojected_components"],
            "oracle": oracle_selected,
            "rejected": rejected_selected,
            "margin": selected_margin,
            "localized_margin": selected_margin if localized else None,
            "fallback_margin": selected_margin if fallback else None,
        },
        "outside": {
            "oracle": oracle_outside,
            "rejected": rejected_outside,
            "margin": _margin(oracle_outside, rejected_outside),
        },
        "per_component": per_component,
    }


def _model_summary(label: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    localized = [row for row in rows if row["selected"]["localized_complete"]]
    fallback = [row for row in rows if row["selected"]["full_response_fallback"]]
    partial = [
        row
        for row in rows
        if row["selected"]["classification"] == "partial_or_missing_projection"
    ]
    full_margins = [row["full"]["margin"] for row in rows]
    local_margins = [row["selected"]["localized_margin"] for row in localized]
    fallback_margins = [row["selected"]["fallback_margin"] for row in fallback]
    outside_margins = [row["outside"]["margin"] for row in localized]
    return {
        "model_label": label,
        "examples": len(rows),
        "localized_examples": len(localized),
        "fallback_examples": len(fallback),
        "partial_or_missing_examples": len(partial),
        "full_margin_mean": _mean(full_margins),
        "full_margin_median": _median(full_margins),
        "full_positive_rate": _rate(
            sum(value is not None and value > 0 for value in full_margins),
            sum(value is not None for value in full_margins),
        ),
        "oracle_full_nll_mean": _mean(row["full"]["oracle"]["nll"] for row in rows),
        "rejected_full_nll_mean": _mean(
            row["full"]["rejected"]["nll"] for row in rows
        ),
        "localized_margin_mean": _mean(local_margins),
        "localized_margin_median": _median(local_margins),
        "localized_positive_rate": _rate(
            sum(value is not None and value > 0 for value in local_margins),
            sum(value is not None for value in local_margins),
        ),
        "fallback_margin_mean": _mean(fallback_margins),
        "outside_margin_mean": _mean(outside_margins),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_summaries(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    expected_per_model: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    by_label = {
        label: [row for row in records if row.get("model_label") == label]
        for label in labels
    }
    model_rows = [_model_summary(label, by_label[label]) for label in labels]
    complete = all(len(by_label[label]) == expected_per_model for label in labels)
    payload = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "expected_examples_per_model": expected_per_model,
        "records": len(records),
        "primary_margin_definition": (
            "length-normalized mean log P(oracle|same public prompt) minus "
            "length-normalized mean log P(rejected|same public prompt)"
        ),
        "localized_inclusion_rule": (
            "all requested top-level components project to non-empty token spans "
            "on both oracle and rejected responses; fallback/partial rows excluded"
        ),
        "manifest_sha256": _sha256(output_dir / MANIFEST_JSON),
        "models": model_rows,
    }
    _atomic_json(output_dir / SUMMARY_JSON, payload)
    columns = list(model_rows[0].keys()) if model_rows else ["model_label", "examples"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(model_rows)
    _atomic_text(output_dir / SUMMARY_CSV, buffer.getvalue())
    lines = [
        "# CARE-OPD recovery-margin diagnostic",
        "",
        (
            f"Status: **{'COMPLETE' if complete else 'PARTIAL'}**. "
            "Positive margins prefer the verifier-valid oracle."
        ),
        "",
        "| Model | n | Full margin | Localized n | Localized margin | "
        "Fallback n | Fallback margin | Outside margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            "| {model_label} | {examples} | {full} | {localized_examples} | "
            "{localized} | {fallback_examples} | {fallback} | {outside} |".format(
                model_label=row["model_label"],
                examples=row["examples"],
                full=_fmt(row["full_margin_mean"]),
                localized_examples=row["localized_examples"],
                localized=_fmt(row["localized_margin_mean"]),
                fallback_examples=row["fallback_examples"],
                fallback=_fmt(row["fallback_margin_mean"]),
                outside=_fmt(row["outside_margin_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "Margins are macro means of per-example length-normalized token log-probabilities.",
            "Fallback and partial projections are not mixed into the localized estimate.",
            "This is a frozen-parameter preference diagnostic, not generation or "
            "causal attribution.",
            "",
        ]
    )
    _atomic_text(output_dir / SUMMARY_MD, "\n".join(lines))
    return payload


@contextmanager
def _output_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".recovery_margin.lock"
    handle = lock_path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"output directory is locked: {output_dir}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"output directory is locked: {output_dir}") from exc
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _manifest_contract(
    args: argparse.Namespace,
    *,
    selected: Sequence[Mapping[str, Any]],
    models: Sequence[tuple[str, str]],
    suite_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected_ids = [str(row["id"]) for row in selected]
    return {
        "schema": SCHEMA,
        "failure_bank": str(args.failure_bank.resolve()),
        "failure_bank_sha256": _sha256(args.failure_bank),
        "selection_seed": args.selection_seed,
        "max_examples": args.max_examples,
        "selected_examples": len(selected),
        "selected_ids_sha256": _text_sha256("\n".join(selected_ids) + "\n"),
        "models": [
            {"label": label, "path": _model_reference(path)} for label, path in models
        ],
        "formal_model_suite": suite_binding,
        "device": str(args.device),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "max_response_tokens": args.max_response_tokens,
        "response_chunk_size": args.response_chunk_size,
        "add_eos": not args.no_eos,
        "no_generation": True,
        "no_training": True,
    }


def _ensure_manifest(path: Path, contract: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise InputError(
                f"resume contract differs from {path}; use a new output directory"
            )
    else:
        _atomic_json(path, contract)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-bank", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", type=_parse_model, required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-suite",
        type=Path,
        help="optional frozen formal model_suite.json used to bind labels and checkpoints",
    )
    parser.add_argument("--max-examples", type=int, default=200, help="0 means all")
    parser.add_argument("--selection-seed", type=int, default=20260813)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32", "auto"),
        default="bfloat16",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=0,
        help="optional response cap; 0 scores the complete response within max-length",
    )
    parser.add_argument("--response-chunk-size", type=int, default=768)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_examples < 0:
        raise InputError("--max-examples must be non-negative")
    if args.max_length < 2:
        raise InputError("--max-length must be at least 2")
    if args.max_response_tokens < 0:
        raise InputError("--max-response-tokens must be non-negative")
    if args.response_chunk_size < 1 or args.progress_every < 1:
        raise InputError("chunk size and progress interval must be positive")
    models = list(args.model)
    labels = [label for label, _ in models]
    if len(labels) != len(set(labels)):
        raise InputError("model labels must be unique")
    bank = _load_failure_bank(args.failure_bank)
    if args.max_response_tokens == 0:
        args.max_response_tokens = None
    selected = _select_rows(
        bank, max_examples=args.max_examples, selection_seed=args.selection_seed
    )
    if not selected:
        raise InputError("failure-bank selection is empty")
    args.device = _resolve_device(args.device)
    suite_binding = _suite_binding(args.model_suite, models)
    contract = _manifest_contract(
        args,
        selected=selected,
        models=models,
        suite_binding=suite_binding,
    )
    output_dir = args.output_dir
    records_path = output_dir / OUTPUT_JSONL

    with _output_lock(output_dir):
        _ensure_manifest(output_dir / MANIFEST_JSON, contract)
        records = _read_jsonl(records_path, allow_empty=True)
        completed: set[tuple[str, str]] = set()
        allowed_ids = {str(row["id"]) for row in selected}
        for record in records:
            key = (str(record.get("model_label")), str(record.get("id")))
            if key in completed:
                raise InputError(f"duplicate resumed result: {key}")
            if key[0] not in labels or key[1] not in allowed_ids:
                raise InputError(f"out-of-contract resumed result: {key}")
            completed.add(key)

        expected_tokenizer_hash: str | None = None
        device = args.device
        total_expected = len(selected) * len(models)
        print(
            f"[recovery-margin] selected={len(selected)} models={len(models)} "
            f"total={total_expected} resumed={len(completed)} device={device}",
            flush=True,
        )
        for model_index, (label, path) in enumerate(models, 1):
            pending = [row for row in selected if (label, str(row["id"])) not in completed]
            if not pending:
                print(f"[model {model_index}/{len(models)}] {label}: already complete", flush=True)
                continue
            print(
                f"[model {model_index}/{len(models)}] {label}: loading {path} "
                f"({len(pending)} pending)",
                flush=True,
            )
            model_args = ModelArguments(
                student_model_name_or_path=path,
                tokenizer_name_or_path=path,
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=args.dtype,
            )
            tokenizer = load_tokenizer(model_args, model_name_or_path=path)
            current_tokenizer_hash = tokenizer_fingerprint(tokenizer)
            if expected_tokenizer_hash is None:
                expected_tokenizer_hash = current_tokenizer_hash
            elif current_tokenizer_hash != expected_tokenizer_hash:
                raise InputError(
                    f"tokenizer mismatch for {label}: {current_tokenizer_hash} != "
                    f"{expected_tokenizer_hash}"
                )
            model = load_causal_lm(path, model_args, trainable=False)
            model.to(device)
            model.eval()
            if not supports_selected_logits(model):
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"{label} does not advertise logits_to_keep; full 4096 x vocab "
                    "logits are forbidden by this diagnostic"
                )

            started = time.monotonic()
            for local_index, row in enumerate(pending, 1):
                result = _score_example(
                    row,
                    tokenizer,
                    model,
                    model_label=label,
                    model_path=path,
                    device=device,
                    max_length=args.max_length,
                    max_response_tokens=args.max_response_tokens,
                    chunk_size=args.response_chunk_size,
                    add_eos=not args.no_eos,
                )
                records.append(result)
                completed.add((label, str(row["id"])))
                _atomic_jsonl(records_path, records)
                if local_index % args.progress_every == 0 or local_index == len(pending):
                    elapsed = time.monotonic() - started
                    rate = local_index / elapsed if elapsed > 0 else 0.0
                    remaining = (len(pending) - local_index) / rate if rate > 0 else 0.0
                    print(
                        f"[progress] model={label} {local_index}/{len(pending)} "
                        f"({100*local_index/len(pending):.1f}%) elapsed={elapsed:.1f}s "
                        f"eta={remaining:.1f}s",
                        flush=True,
                    )
            del model
            del tokenizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _write_summaries(
                output_dir,
                records,
                labels=labels,
                expected_per_model=len(selected),
                manifest=contract,
            )
            print(f"[model {model_index}/{len(models)}] {label}: complete", flush=True)

        summary = _write_summaries(
            output_dir,
            records,
            labels=labels,
            expected_per_model=len(selected),
            manifest=contract,
        )
        print(
            f"[recovery-margin] complete={summary['complete']} "
            f"records={summary['records']} output={output_dir.resolve()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
