#!/usr/bin/env python3
"""Aggregate the two private real-medical external-evaluation tracks safely.

The input JSONL files contain private model responses.  This program parses each
line, immediately projects it onto an explicit scalar allow-list, and never
emits case identifiers, report identifiers, source text, prompts, responses, or
per-case measurements.  Every output is aggregate-only and is checked by a
fail-closed privacy scan before the first byte is written.

The statistical unit for confidence intervals is the report family.  A sampled
family carries all of its selected variants together, preserving within-report
dependence.  The default ``legacy_frozen`` analysis exactly follows the public
V7 Python-RNG contract (seed 20260819) and reuses one family-draw matrix across
all scopes and metrics in a model comparison.  A different deterministic RNG
is available only through the explicitly supplemental ``independent_stable``
mode.  Intervals quantify evaluation-sample uncertainty for frozen checkpoints;
they are not cross-training-seed intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_STRUCTURED_RELATIVE = "eval_direct_gold_structured_six_models_v5/private"
DEFAULT_RAW_RELATIVE = "eval_direct_gold_raw_six_models_v5/private"
FROZEN_BOOTSTRAP_SEED = 20260819
DEFAULT_BOOTSTRAP_SEED = FROZEN_BOOTSTRAP_SEED
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
ALL_STATES = "all_states"

ANALYSIS_MODE_LEGACY_FROZEN = "legacy_frozen"
ANALYSIS_MODE_INDEPENDENT_STABLE = "independent_stable"
ANALYSIS_MODES = (
    ANALYSIS_MODE_LEGACY_FROZEN,
    ANALYSIS_MODE_INDEPENDENT_STABLE,
)
LEGACY_ANALYSIS_METHOD = "legacy_frozen_py_random_v1"
INDEPENDENT_ANALYSIS_METHOD = "independent_stable_numpy_v1"
AUTHORITATIVE_FROZEN = "authoritative_frozen_protocol"
SUPPLEMENTAL_REANALYSIS = "supplemental_reanalysis_not_frozen"

EXPECTED_TRACKS = {
    "structured_evidence_faithfulness",
    "raw_strict_extraction_challenge",
}
VARIANT_ORDER = (
    "clean",
    "bad_history",
    "source_outage",
    "evidence_conflict",
)

# These patterns are forbidden anywhere in serialized outputs.  Keep the
# public privacy-audit wording generic so the audit cannot trip over itself.
FORBIDDEN_OUTPUT_PATTERNS = (
    re.compile(r"raw_response_private", re.IGNORECASE),
    re.compile(r"input_text", re.IGNORECASE),
    re.compile(r"source_text_private", re.IGNORECASE),
    re.compile(r"clinical_gold[._]private", re.IGNORECASE),
    re.compile(r"cases[._]private", re.IGNORECASE),
    re.compile(r"source_record_private", re.IGNORECASE),
    re.compile(r"patient_name", re.IGNORECASE),
)


class StatsContractError(RuntimeError):
    """Raised when inputs or pairings violate the frozen analysis contract."""


class PrivacyError(RuntimeError):
    """Raised before writing when an output contains a forbidden field."""


@dataclass(frozen=True)
class SafeCase:
    """Scalar-only in-memory projection of one private JSONL row."""

    model: str
    track: str
    family_id: str
    variant: str
    report_exact: bool | None
    task_success: bool | None
    field_precision: float | None
    field_recall: float | None
    field_f1: float | None
    field_accuracy: float | None
    content_f1: float | None
    grounded_content_f1: float | None
    content_complete: bool | None
    grounded_content_complete: bool | None
    tp: int | None
    fp: int | None
    fn: int | None


# Each family is represented by a sum of these additive sufficient statistics.
STAT_NAMES = (
    "reports",
    "report_exact_sum",
    "report_exact_n",
    "task_success_sum",
    "task_success_n",
    "field_precision_sum",
    "field_precision_n",
    "field_recall_sum",
    "field_recall_n",
    "field_f1_sum",
    "field_f1_n",
    "field_accuracy_sum",
    "field_accuracy_n",
    "content_f1_sum",
    "content_f1_n",
    "grounded_content_f1_sum",
    "grounded_content_f1_n",
    "content_complete_sum",
    "content_complete_n",
    "grounded_content_complete_sum",
    "grounded_content_complete_n",
    "field_tp",
    "field_fp",
    "field_fn",
)
STAT_INDEX = {name: index for index, name in enumerate(STAT_NAMES)}

METRIC_NAMES = (
    "report_exact_rate",
    "task_success_rate",
    "macro_field_precision",
    "macro_field_recall",
    "macro_field_f1",
    "macro_field_accuracy",
    "micro_field_precision",
    "micro_field_recall",
    "micro_field_f1",
    "macro_content_f1",
    "macro_grounded_content_f1",
    "content_complete_rate",
    "grounded_content_complete_rate",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _as_nonnegative_int(value: Any) -> int | None:
    number = _as_finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _first_bool(*values: Any) -> bool | None:
    return next((value for value in values if isinstance(value, bool)), None)


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _as_finite_number(value)
        if number is not None:
            return number
    return None


def _first_count(*values: Any) -> int | None:
    for value in values:
        number = _as_nonnegative_int(value)
        if number is not None:
            return number
    return None


def _project_row(payload: Mapping[str, Any], *, source: Path, line_number: int) -> SafeCase:
    """Copy only approved scalar measurements from a private row."""

    assessment = payload.get("assessment")
    metrics = payload.get("metrics")
    if not isinstance(assessment, Mapping):
        raise StatsContractError(f"{source.name}:{line_number}: assessment must be an object")
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, Mapping):
        raise StatsContractError(f"{source.name}:{line_number}: metrics must be an object")

    model = str(payload.get("model_label") or "").strip()
    track = str(assessment.get("evaluation_track") or payload.get("evaluation_track") or "").strip()
    family_id = str(assessment.get("family_id") or payload.get("family_id") or "").strip()
    variant = str(assessment.get("variant") or payload.get("variant") or "").strip()
    if not model or not family_id:
        raise StatsContractError(f"{source.name}:{line_number}: model/family identity is missing")
    if track not in EXPECTED_TRACKS:
        raise StatsContractError(f"{source.name}:{line_number}: unsupported track {track!r}")
    if variant not in VARIANT_ORDER:
        raise StatsContractError(f"{source.name}:{line_number}: unsupported variant {variant!r}")

    # Assessment is authoritative; the compact metrics snapshot is fallback
    # only for old completed runs.
    return SafeCase(
        model=model,
        track=track,
        family_id=family_id,
        variant=variant,
        report_exact=_first_bool(
            assessment.get("report_exact_match"),
            metrics.get("report_exact"),
        ),
        task_success=_first_bool(
            assessment.get("state_task_success"),
            assessment.get("task_success"),
            assessment.get("passed"),
            metrics.get("task_success"),
        ),
        field_precision=_first_number(assessment.get("field_precision")),
        field_recall=_first_number(assessment.get("field_recall")),
        field_f1=_first_number(assessment.get("field_f1")),
        field_accuracy=_first_number(assessment.get("field_accuracy")),
        content_f1=_first_number(assessment.get("content_f1")),
        grounded_content_f1=_first_number(assessment.get("grounded_content_f1")),
        content_complete=_first_bool(assessment.get("content_complete")),
        grounded_content_complete=_first_bool(assessment.get("grounded_content_complete")),
        tp=_first_count(assessment.get("tp"), metrics.get("tp")),
        fp=_first_count(assessment.get("fp"), metrics.get("fp")),
        fn=_first_count(assessment.get("fn"), metrics.get("fn")),
    )


def load_cases(paths: Sequence[Path]) -> tuple[list[SafeCase], list[dict[str, Any]]]:
    cases: list[SafeCase] = []
    inventory: list[dict[str, Any]] = []
    for path in sorted(path.resolve() for path in paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        file_cases: list[SafeCase] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StatsContractError(f"{path.name}:{line_number}: invalid JSON") from exc
                if not isinstance(payload, Mapping):
                    raise StatsContractError(f"{path.name}:{line_number}: row must be an object")
                file_cases.append(_project_row(payload, source=path, line_number=line_number))
        if not file_cases:
            raise StatsContractError(f"{path}: no cases")
        identities = {(case.model, case.track) for case in file_cases}
        if len(identities) != 1:
            raise StatsContractError(f"{path.name}: one file must contain one model and one track")
        model, track = next(iter(identities))
        inventory.append(
            {
                "artifact_id": f"{track}::{model}",
                "sha256": _sha256_file(path),
                "rows": len(file_cases),
                "model": model,
                "track": track,
            }
        )
        cases.extend(file_cases)
    return cases, inventory


def discover_inputs(
    input_root: Path,
    *,
    structured_relative: str = DEFAULT_STRUCTURED_RELATIVE,
    raw_relative: str = DEFAULT_RAW_RELATIVE,
) -> list[Path]:
    root = input_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    directories = [root / structured_relative, root / raw_relative]
    paths: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        found = sorted(directory.glob("*.cases.private.jsonl"))
        if not found:
            raise StatsContractError(f"{directory}: no private case JSONL files")
        paths.extend(found)
    return paths


def _case_vector(case: SafeCase) -> np.ndarray:
    vector = np.zeros(len(STAT_NAMES), dtype=np.float64)
    vector[STAT_INDEX["reports"]] = 1.0

    def add_bool(prefix: str, value: bool | None) -> None:
        if value is not None:
            vector[STAT_INDEX[f"{prefix}_sum"]] = float(value)
            vector[STAT_INDEX[f"{prefix}_n"]] = 1.0

    def add_number(prefix: str, value: float | None) -> None:
        if value is not None:
            vector[STAT_INDEX[f"{prefix}_sum"]] = value
            vector[STAT_INDEX[f"{prefix}_n"]] = 1.0

    add_bool("report_exact", case.report_exact)
    add_bool("task_success", case.task_success)
    add_number("field_precision", case.field_precision)
    add_number("field_recall", case.field_recall)
    add_number("field_f1", case.field_f1)
    add_number("field_accuracy", case.field_accuracy)
    add_number("content_f1", case.content_f1)
    add_number("grounded_content_f1", case.grounded_content_f1)
    add_bool("content_complete", case.content_complete)
    add_bool("grounded_content_complete", case.grounded_content_complete)
    if case.tp is not None and case.fp is not None and case.fn is not None:
        vector[STAT_INDEX["field_tp"]] = case.tp
        vector[STAT_INDEX["field_fp"]] = case.fp
        vector[STAT_INDEX["field_fn"]] = case.fn
    return vector


def _safe_ratio(numerator: Any, denominator: Any) -> Any:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    shape = np.broadcast_shapes(numerator_array.shape, denominator_array.shape)
    result = np.full(shape, np.nan, dtype=np.float64)
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array != 0)
    return float(result) if result.ndim == 0 else result


def _metrics_from_total(total: np.ndarray) -> dict[str, Any]:
    def stat(name: str) -> Any:
        return total[..., STAT_INDEX[name]]

    tp, fp, fn = stat("field_tp"), stat("field_fp"), stat("field_fn")
    return {
        "report_exact_rate": _safe_ratio(stat("report_exact_sum"), stat("report_exact_n")),
        "task_success_rate": _safe_ratio(stat("task_success_sum"), stat("task_success_n")),
        "macro_field_precision": _safe_ratio(stat("field_precision_sum"), stat("field_precision_n")),
        "macro_field_recall": _safe_ratio(stat("field_recall_sum"), stat("field_recall_n")),
        "macro_field_f1": _safe_ratio(stat("field_f1_sum"), stat("field_f1_n")),
        "macro_field_accuracy": _safe_ratio(stat("field_accuracy_sum"), stat("field_accuracy_n")),
        "micro_field_precision": _safe_ratio(tp, tp + fp),
        "micro_field_recall": _safe_ratio(tp, tp + fn),
        "micro_field_f1": _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "macro_content_f1": _safe_ratio(stat("content_f1_sum"), stat("content_f1_n")),
        "macro_grounded_content_f1": _safe_ratio(
            stat("grounded_content_f1_sum"), stat("grounded_content_f1_n")
        ),
        "content_complete_rate": _safe_ratio(
            stat("content_complete_sum"), stat("content_complete_n")
        ),
        "grounded_content_complete_rate": _safe_ratio(
            stat("grounded_content_complete_sum"), stat("grounded_content_complete_n")
        ),
    }


def _optional_float(value: Any) -> float | None:
    if isinstance(value, np.ndarray):
        raise TypeError("expected scalar")
    number = float(value)
    return number if math.isfinite(number) else None


def aggregate_cases(cases: Sequence[SafeCase]) -> dict[str, Any]:
    if not cases:
        raise StatsContractError("cannot aggregate an empty case set")
    total = np.stack([_case_vector(case) for case in cases]).sum(axis=0)
    metrics = {name: _optional_float(value) for name, value in _metrics_from_total(total).items()}
    result: dict[str, Any] = {
        "families": len({case.family_id for case in cases}),
        "reports": int(total[STAT_INDEX["reports"]]),
        "report_exact_successes": int(total[STAT_INDEX["report_exact_sum"]]),
        "report_exact_evaluated": int(total[STAT_INDEX["report_exact_n"]]),
        "task_successes": int(total[STAT_INDEX["task_success_sum"]]),
        "task_success_evaluated": int(total[STAT_INDEX["task_success_n"]]),
        "field_tp": int(total[STAT_INDEX["field_tp"]]),
        "field_fp": int(total[STAT_INDEX["field_fp"]]),
        "field_fn": int(total[STAT_INDEX["field_fn"]]),
    }
    for prefix in (
        "field_precision",
        "field_recall",
        "field_f1",
        "field_accuracy",
        "content_f1",
        "grounded_content_f1",
        "content_complete",
        "grounded_content_complete",
    ):
        result[f"{prefix}_evaluated"] = int(total[STAT_INDEX[f"{prefix}_n"]])
    result.update(metrics)
    return result


def build_report_level_rows(cases: Sequence[SafeCase]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SafeCase]] = defaultdict(list)
    for case in cases:
        grouped[(case.track, case.model)].append(case)
    rows: list[dict[str, Any]] = []
    for (track, model), model_cases in sorted(grouped.items()):
        present = {case.variant for case in model_cases}
        for scope in (*VARIANT_ORDER, ALL_STATES):
            if scope != ALL_STATES and scope not in present:
                continue
            selected = model_cases if scope == ALL_STATES else [
                case for case in model_cases if case.variant == scope
            ]
            rows.append({"model": model, "track": track, "variant": scope, **aggregate_cases(selected)})
    return rows


def build_cohort_audit(
    cases: Sequence[SafeCase], *, provider_total: int | None = None
) -> dict[str, Any]:
    """Return overlap counts only; no report-family identifier leaves this function."""

    by_track_model: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for case in cases:
        by_track_model[case.track][case.model].add(case.family_id)
    canonical: dict[str, set[str]] = {}
    for track, models in by_track_model.items():
        sets = list(models.values())
        if not sets or any(values != sets[0] for values in sets[1:]):
            raise StatsContractError(f"models in {track!r} do not share one frozen cohort")
        canonical[track] = set(sets[0])
    structured = canonical.get("structured_evidence_faithfulness", set())
    raw = canonical.get("raw_strict_extraction_challenge", set())
    if not structured or not raw:
        raise StatsContractError("cohort audit requires both structured and raw-strict tracks")
    union = structured | raw
    if provider_total is not None:
        if provider_total < 1:
            raise StatsContractError("provider total must be positive")
        if provider_total < len(union):
            raise StatsContractError(
                f"provider total {provider_total} is smaller than evaluated union {len(union)}"
            )
    return {
        "provider_total": provider_total,
        "structured_eligible": len(structured),
        "raw_strict_eligible": len(raw),
        "track_intersection": len(structured & raw),
        "track_union": len(union),
        "structured_only": len(structured - raw),
        "raw_strict_only": len(raw - structured),
        "raw_strict_is_structured_subset": raw <= structured,
        "provider_not_in_structured": (
            provider_total - len(structured) if provider_total is not None else None
        ),
        "provider_not_in_raw_strict": (
            provider_total - len(raw) if provider_total is not None else None
        ),
        "provider_not_in_either_track": (
            provider_total - len(union) if provider_total is not None else None
        ),
        "counting_rule": "overlapping interface views; never add track counts",
        "identifier_columns_emitted": 0,
    }


def _index_model_cases(cases: Sequence[SafeCase]) -> dict[str, dict[str, dict[str, SafeCase]]]:
    index: dict[str, dict[str, dict[str, SafeCase]]] = defaultdict(lambda: defaultdict(dict))
    for case in cases:
        key = f"{case.family_id}\x1f{case.variant}"
        if key in index[case.track][case.model]:
            raise StatsContractError(
                f"duplicate family/variant for track={case.track!r}, model={case.model!r}"
            )
        index[case.track][case.model][key] = case
    return {track: {model: dict(values) for model, values in models.items()} for track, models in index.items()}


def _stable_seed(base_seed: int, *parts: str) -> int:
    token = "\x1f".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _family_vectors(
    model_index: Mapping[str, SafeCase], scope: str
) -> tuple[list[str], np.ndarray]:
    by_family: dict[str, list[SafeCase]] = defaultdict(list)
    for case in model_index.values():
        if scope == ALL_STATES or case.variant == scope:
            by_family[case.family_id].append(case)
    if not by_family:
        raise StatsContractError(f"scope {scope!r} contains no families")
    family_ids = sorted(by_family)
    vectors = np.stack(
        [sum((_case_vector(case) for case in by_family[family_id]), np.zeros(len(STAT_NAMES))) for family_id in family_ids]
    )
    return family_ids, vectors


def _comparison_inference(lower: float | None, upper: float | None) -> str:
    if lower is not None and lower > 0:
        return "candidate_higher"
    if upper is not None and upper < 0:
        return "candidate_lower"
    return "interval_includes_zero"


def _legacy_quantile(values: Sequence[float], probability: float) -> float:
    """Match the frozen evaluator's pure-Python linear quantile exactly."""

    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        raise StatsContractError("cannot take a quantile of an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_family_bootstrap_independent(
    reference_index: Mapping[str, SafeCase],
    candidate_index: Mapping[str, SafeCase],
    *,
    reference_model: str,
    candidate_model: str,
    track: str,
    scope: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    if replicates < 1:
        raise StatsContractError("bootstrap replicates must be positive")
    if set(reference_index) != set(candidate_index):
        missing_candidate = len(set(reference_index) - set(candidate_index))
        missing_reference = len(set(candidate_index) - set(reference_index))
        raise StatsContractError(
            f"unpaired models for {track}: missing candidate={missing_candidate}, "
            f"missing reference={missing_reference}"
        )
    reference_families, reference_vectors = _family_vectors(reference_index, scope)
    candidate_families, candidate_vectors = _family_vectors(candidate_index, scope)
    if reference_families != candidate_families:
        raise StatsContractError(f"family mismatch for track={track}, scope={scope}")

    point_reference = _metrics_from_total(reference_vectors.sum(axis=0))
    point_candidate = _metrics_from_total(candidate_vectors.sum(axis=0))
    paired_families = len(reference_families)
    combination_seed = _stable_seed(seed, track, scope, reference_model, candidate_model)
    rng = np.random.default_rng(combination_seed)

    distributions: dict[str, list[np.ndarray]] = {metric: [] for metric in METRIC_NAMES}
    # Chunking bounds peak memory while retaining vectorized ratio computation.
    remaining = replicates
    while remaining:
        chunk = min(remaining, 1_000)
        sampled = rng.integers(0, paired_families, size=(chunk, paired_families))
        reference_totals = reference_vectors[sampled].sum(axis=1)
        candidate_totals = candidate_vectors[sampled].sum(axis=1)
        reference_metrics = _metrics_from_total(reference_totals)
        candidate_metrics = _metrics_from_total(candidate_totals)
        for metric in METRIC_NAMES:
            distributions[metric].append(candidate_metrics[metric] - reference_metrics[metric])
        remaining -= chunk

    rows: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        reference_value = _optional_float(point_reference[metric])
        candidate_value = _optional_float(point_candidate[metric])
        values = np.concatenate(distributions[metric])
        values = values[np.isfinite(values)]
        delta = (
            candidate_value - reference_value
            if candidate_value is not None and reference_value is not None
            else None
        )
        lower = float(np.quantile(values, 0.025)) if values.size else None
        upper = float(np.quantile(values, 0.975)) if values.size else None
        rows.append(
            {
                "candidate_model": candidate_model,
                "reference_model": reference_model,
                "track": track,
                "variant": scope,
                "metric": metric,
                "direction": "higher_is_better",
                "paired_families": paired_families,
                "reference_value": reference_value,
                "candidate_value": candidate_value,
                "delta_candidate_minus_reference": delta,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_unit": "report_family",
                "bootstrap_replicates": replicates,
                "valid_bootstrap_replicates": int(values.size),
                "bootstrap_seed": combination_seed,
                "analysis_method": INDEPENDENT_ANALYSIS_METHOD,
                "authority": SUPPLEMENTAL_REANALYSIS,
                "inference": _comparison_inference(lower, upper),
            }
        )
    return rows


def paired_family_bootstrap_legacy_frozen(
    reference_index: Mapping[str, SafeCase],
    candidate_index: Mapping[str, SafeCase],
    *,
    reference_model: str,
    candidate_model: str,
    track: str,
    scopes: Sequence[str],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Reproduce the public frozen family bootstrap and extend its draw matrix.

    The original evaluator initializes ``random.Random(seed)`` once for one
    reference/candidate/track/stratum comparison and draws whole report
    families with replacement.  Every metric in that comparison consumes the
    same family draws.  This compatibility implementation also reuses those
    exact draws for the explicit variant scopes in the expanded matrix.  It is
    therefore possible to add fixed metrics without changing any already
    frozen overall/clean/source-outage confidence interval.
    """

    if replicates < 1:
        raise StatsContractError("bootstrap replicates must be positive")
    if seed != FROZEN_BOOTSTRAP_SEED:
        raise StatsContractError(
            "legacy_frozen analysis is bound to bootstrap seed "
            f"{FROZEN_BOOTSTRAP_SEED}; select independent_stable for a reanalysis"
        )
    if set(reference_index) != set(candidate_index):
        missing_candidate = len(set(reference_index) - set(candidate_index))
        missing_reference = len(set(candidate_index) - set(reference_index))
        raise StatsContractError(
            f"unpaired models for {track}: missing candidate={missing_candidate}, "
            f"missing reference={missing_reference}"
        )

    scope_vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    family_ids: list[str] | None = None
    for scope in scopes:
        reference_families, reference_vectors = _family_vectors(reference_index, scope)
        candidate_families, candidate_vectors = _family_vectors(candidate_index, scope)
        if reference_families != candidate_families:
            raise StatsContractError(f"family mismatch for track={track}, scope={scope}")
        if family_ids is None:
            family_ids = reference_families
        elif family_ids != reference_families:
            raise StatsContractError(
                f"legacy frozen scopes do not share one complete family set for {track}"
            )
        scope_vectors[scope] = (reference_vectors, candidate_vectors)
    if not family_ids:
        raise StatsContractError(f"no paired report families for {track}")

    paired_families = len(family_ids)
    rng = random.Random(seed)
    sampled = np.empty((replicates, paired_families), dtype=np.intp)
    for replicate in range(replicates):
        sampled[replicate] = [
            rng.randrange(paired_families) for _ in range(paired_families)
        ]

    rows: list[dict[str, Any]] = []
    for scope in scopes:
        reference_vectors, candidate_vectors = scope_vectors[scope]
        point_reference = _metrics_from_total(reference_vectors.sum(axis=0))
        point_candidate = _metrics_from_total(candidate_vectors.sum(axis=0))
        distributions: dict[str, list[np.ndarray]] = {
            metric: [] for metric in METRIC_NAMES
        }
        for start in range(0, replicates, 1_000):
            selected = sampled[start : start + 1_000]
            reference_totals = reference_vectors[selected].sum(axis=1)
            candidate_totals = candidate_vectors[selected].sum(axis=1)
            reference_metrics = _metrics_from_total(reference_totals)
            candidate_metrics = _metrics_from_total(candidate_totals)
            for metric in METRIC_NAMES:
                distributions[metric].append(
                    candidate_metrics[metric] - reference_metrics[metric]
                )

        for metric in METRIC_NAMES:
            reference_value = _optional_float(point_reference[metric])
            candidate_value = _optional_float(point_candidate[metric])
            values = np.concatenate(distributions[metric])
            values = values[np.isfinite(values)]
            delta = (
                candidate_value - reference_value
                if candidate_value is not None and reference_value is not None
                else None
            )
            lower = _legacy_quantile(values, 0.025) if values.size else None
            upper = _legacy_quantile(values, 0.975) if values.size else None
            rows.append(
                {
                    "candidate_model": candidate_model,
                    "reference_model": reference_model,
                    "track": track,
                    "variant": scope,
                    "metric": metric,
                    "direction": "higher_is_better",
                    "paired_families": paired_families,
                    "reference_value": reference_value,
                    "candidate_value": candidate_value,
                    "delta_candidate_minus_reference": delta,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_unit": "report_family",
                    "bootstrap_replicates": replicates,
                    "valid_bootstrap_replicates": int(values.size),
                    "bootstrap_seed": seed,
                    "analysis_method": LEGACY_ANALYSIS_METHOD,
                    "authority": AUTHORITATIVE_FROZEN,
                    "inference": _comparison_inference(lower, upper),
                }
            )
    return rows


def _normalize_references(values: Sequence[str] | None) -> list[str]:
    references: list[str] = []
    for value in values or ():
        for token in value.split(","):
            token = token.strip()
            if token and token not in references:
                references.append(token)
    return references


def build_fixed_matrix(
    cases: Sequence[SafeCase],
    *,
    candidate_model: str,
    reference_models: Sequence[str] | None,
    replicates: int,
    seed: int,
    analysis_mode: str = ANALYSIS_MODE_LEGACY_FROZEN,
) -> tuple[list[dict[str, Any]], list[str]]:
    if analysis_mode not in ANALYSIS_MODES:
        raise StatsContractError(f"unsupported analysis mode {analysis_mode!r}")
    if analysis_mode == ANALYSIS_MODE_LEGACY_FROZEN and seed != FROZEN_BOOTSTRAP_SEED:
        raise StatsContractError(
            "legacy_frozen analysis is bound to bootstrap seed "
            f"{FROZEN_BOOTSTRAP_SEED}; select independent_stable for a reanalysis"
        )
    index = _index_model_cases(cases)
    available = sorted({case.model for case in cases})
    references = _normalize_references(reference_models)
    if not references:
        references = [model for model in available if model != candidate_model]
    if candidate_model in references:
        raise StatsContractError("candidate model cannot also be a reference")
    if not references:
        raise StatsContractError("no reference models selected")

    rows: list[dict[str, Any]] = []
    for track, models in sorted(index.items()):
        if candidate_model not in models:
            raise StatsContractError(f"candidate {candidate_model!r} is absent from {track}")
        present_variants = {case.variant for case in models[candidate_model].values()}
        scopes = [variant for variant in VARIANT_ORDER if variant in present_variants]
        scopes.append(ALL_STATES)
        for reference in references:
            if reference not in models:
                raise StatsContractError(f"reference {reference!r} is absent from {track}")
            if analysis_mode == ANALYSIS_MODE_LEGACY_FROZEN:
                rows.extend(
                    paired_family_bootstrap_legacy_frozen(
                        models[reference],
                        models[candidate_model],
                        reference_model=reference,
                        candidate_model=candidate_model,
                        track=track,
                        scopes=scopes,
                        replicates=replicates,
                        seed=seed,
                    )
                )
            else:
                for scope in scopes:
                    rows.extend(
                        paired_family_bootstrap_independent(
                            models[reference],
                            models[candidate_model],
                            reference_model=reference,
                            candidate_model=candidate_model,
                            track=track,
                            scope=scope,
                            replicates=replicates,
                            seed=seed,
                        )
                    )
    return rows, references


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _render_markdown(
    report_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    cohort_audit: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
    analysis_method: str,
    authority: str,
) -> str:
    lines = [
        "# Real-medical external evaluation: aggregate statistics",
        "",
        "All rows below are aggregate measurements. Confidence intervals use a paired "
        "report-family percentile bootstrap, keeping all selected states of a sampled family "
        "together. They describe frozen-checkpoint evaluation-sample uncertainty, not training-seed variance.",
        "",
        f"Bootstrap configuration: {replicates:,} replicates; seed {seed}; "
        f"analysis method `{analysis_method}`; authority `{authority}`.",
        "",
        "## Cohort overlap audit",
        "",
        "| Provider total | Structured eligible | Raw-strict eligible | Intersection | Union | Structured only | Raw only | Raw subset of structured |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
        "| {provider} | {structured} | {raw} | {intersection} | {union} | {structured_only} | {raw_only} | {subset} |".format(
            provider=(
                cohort_audit["provider_total"]
                if cohort_audit["provider_total"] is not None
                else "not supplied"
            ),
            structured=cohort_audit["structured_eligible"],
            raw=cohort_audit["raw_strict_eligible"],
            intersection=cohort_audit["track_intersection"],
            union=cohort_audit["track_union"],
            structured_only=cohort_audit["structured_only"],
            raw_only=cohort_audit["raw_strict_only"],
            subset=str(cohort_audit["raw_strict_is_structured_subset"]).lower(),
        ),
        "",
        "The two tracks are overlapping interface views of one real-record cohort. Their eligible counts must not be added.",
        "",
        "## Model-by-track-by-variant summary",
        "",
        "| Track | Model | Variant | Families | Reports | Exact | Task success | Macro field F1 | Micro field F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            "| {track} | {model} | {variant} | {families} | {reports} | {exact} | "
            "{task} | {macro} | {micro} |".format(
                track=row["track"],
                model=row["model"],
                variant=row["variant"],
                families=row["families"],
                reports=row["reports"],
                exact=_format_percent(row["report_exact_rate"]),
                task=_format_percent(row["task_success_rate"]),
                macro=_format_percent(row["macro_field_f1"]),
                micro=_format_percent(row["micro_field_f1"]),
            )
        )
    lines.extend(
        [
            "",
            "## CARE-versus-every-reference fixed matrix",
            "",
            "The compact view below shows the two preselected headline metrics. The CSV and JSON "
            "contain every metric for every reference, track, and variant; no per-metric comparator selection is used.",
            "",
            "| Track | Variant | Reference | Metric | CARE | Reference | Delta | 95% CI |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        if row["metric"] not in {"task_success_rate", "micro_field_f1"}:
            continue
        delta = row["delta_candidate_minus_reference"]
        lower, upper = row["ci95_lower"], row["ci95_upper"]
        interval = "N/A" if lower is None or upper is None else f"[{100*lower:.2f}, {100*upper:.2f}] pp"
        lines.append(
            f"| {row['track']} | {row['variant']} | {row['reference_model']} | "
            f"{row['metric']} | {_format_percent(row['candidate_value'])} | "
            f"{_format_percent(row['reference_value'])} | "
            f"{'N/A' if delta is None else f'{100*delta:.2f} pp'} | {interval} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Macro values average report-level scores; micro values pool field counts.",
            "- Structured and raw-strict tracks are separate views and are never pooled.",
            "- Raw-strict is a subset of the Structured eligible cohort; 65 and 54 must not be added.",
            "- A confidence interval excluding zero describes a paired evaluation-sample difference only.",
            "- No case-level text, identifier, response, prompt, or per-case score is included.",
            "",
        ]
    )
    return "\n".join(lines)


def _assert_public_safe(value: Any, *, context: str = "output") -> None:
    """Fail closed if a serialized output could expose a forbidden field."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text.lower().endswith("_private"):
                raise PrivacyError(f"{context}: private-suffixed key is forbidden")
            _assert_public_safe(nested, context=context)
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_public_safe(nested, context=context)
        serialized = json.dumps(value, ensure_ascii=False)
    else:
        serialized = str(value)
    if any(pattern.search(serialized) for pattern in FORBIDDEN_OUTPUT_PATTERNS):
        raise PrivacyError(f"{context}: forbidden private-content marker detected")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run(
    *,
    input_root: Path,
    output_dir: Path,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    candidate_model: str = "care_opd",
    reference_models: Sequence[str] | None = None,
    provider_total: int | None = None,
    structured_relative: str = DEFAULT_STRUCTURED_RELATIVE,
    raw_relative: str = DEFAULT_RAW_RELATIVE,
    analysis_mode: str = ANALYSIS_MODE_LEGACY_FROZEN,
) -> dict[str, Any]:
    if bootstrap_replicates < 1:
        raise StatsContractError("bootstrap replicates must be positive")
    paths = discover_inputs(
        input_root,
        structured_relative=structured_relative,
        raw_relative=raw_relative,
    )
    cases, inventory = load_cases(paths)
    found_tracks = {case.track for case in cases}
    if found_tracks != EXPECTED_TRACKS:
        raise StatsContractError(
            f"expected both frozen tracks; found {sorted(found_tracks)}"
        )
    report_rows = build_report_level_rows(cases)
    cohort_audit = build_cohort_audit(cases, provider_total=provider_total)
    comparison_rows, references = build_fixed_matrix(
        cases,
        candidate_model=candidate_model,
        reference_models=reference_models,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        analysis_mode=analysis_mode,
    )

    if analysis_mode == ANALYSIS_MODE_LEGACY_FROZEN:
        analysis_method = LEGACY_ANALYSIS_METHOD
        authority = AUTHORITATIVE_FROZEN
        rng_contract = "python.random.Random; one draw matrix per track/reference/candidate"
        draw_reuse = "same report-family draws reused across all scopes and metrics"
    elif analysis_mode == ANALYSIS_MODE_INDEPENDENT_STABLE:
        analysis_method = INDEPENDENT_ANALYSIS_METHOD
        authority = SUPPLEMENTAL_REANALYSIS
        rng_contract = "numpy.random.Generator; deterministic per-scope derived seed"
        draw_reuse = "independent deterministic draw matrix per scope"
    else:  # build_fixed_matrix checks this too; keep payload construction fail closed.
        raise StatsContractError(f"unsupported analysis mode {analysis_mode!r}")

    payload = {
        "schema_version": "real_external_stats_v2",
        "privacy": {
            "status": "PASS",
            "release_granularity": "aggregate_only",
            "case_level_rows_emitted": 0,
        },
        "analysis": {
            "candidate_model": candidate_model,
            "reference_models": references,
            "tracks": sorted(found_tracks),
            "analysis_mode": analysis_mode,
            "analysis_method": analysis_method,
            "authority": authority,
            "bootstrap_unit": "report_family",
            "bootstrap_method": "paired percentile bootstrap",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_base_seed": bootstrap_seed,
            "rng_contract": rng_contract,
            "draw_reuse": draw_reuse,
            "interval_scope": "frozen-checkpoint evaluation-sample uncertainty",
        },
        "input_inventory": inventory,
        "cohort_audit": cohort_audit,
        "report_level_metrics": report_rows,
        "care_vs_references": comparison_rows,
    }
    report_fieldnames = list(report_rows[0])
    comparison_fieldnames = list(comparison_rows[0])
    cohort_rows = [cohort_audit]
    texts = {
        "real_external_report_metrics.csv": _csv_text(report_rows, report_fieldnames),
        "care_vs_all_references.csv": _csv_text(comparison_rows, comparison_fieldnames),
        "cohort_audit.csv": _csv_text(cohort_rows, list(cohort_audit)),
        "real_external_stats.json": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "SUMMARY.md": _render_markdown(
            report_rows,
            comparison_rows,
            cohort_audit,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            analysis_method=analysis_method,
            authority=authority,
        ),
    }
    _assert_public_safe(payload, context="JSON payload")
    for name, text in texts.items():
        _assert_public_safe(text, context=name)
    destination = output_dir.expanduser().resolve()
    for name, text in texts.items():
        _atomic_write(destination / name, text)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--analysis-mode",
        choices=ANALYSIS_MODES,
        default=ANALYSIS_MODE_LEGACY_FROZEN,
        help=(
            "legacy_frozen is the authoritative V7-compatible protocol and is bound "
            f"to seed {FROZEN_BOOTSTRAP_SEED}; independent_stable is an explicitly "
            "supplemental reanalysis"
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--candidate-model", default="care_opd")
    parser.add_argument(
        "--provider-total",
        type=int,
        default=None,
        help="optional independently supplied provider cohort total; never inferred",
    )
    parser.add_argument(
        "--reference-models",
        nargs="*",
        default=None,
        help="space- or comma-separated labels; default: every non-candidate model",
    )
    parser.add_argument("--structured-relative", default=DEFAULT_STRUCTURED_RELATIVE)
    parser.add_argument("--raw-relative", default=DEFAULT_RAW_RELATIVE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        input_root=args.input_root,
        output_dir=args.output_dir,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        candidate_model=args.candidate_model,
        reference_models=args.reference_models,
        provider_total=args.provider_total,
        structured_relative=args.structured_relative,
        raw_relative=args.raw_relative,
        analysis_mode=args.analysis_mode,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "models": 1 + len(payload["analysis"]["reference_models"]),
                "tracks": len(payload["analysis"]["tracks"]),
                "analysis_method": payload["analysis"]["analysis_method"],
                "authority": payload["analysis"]["authority"],
                "output_dir": str(args.output_dir.resolve()),
                "privacy": payload["privacy"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
