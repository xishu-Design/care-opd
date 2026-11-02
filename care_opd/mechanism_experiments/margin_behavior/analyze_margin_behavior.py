#!/usr/bin/env python3
"""Link frozen recovery margins to actual verifier-judged rescue outcomes.

The cohort must be the result-blind Clean-SFT failure bank already used by the
recovery-margin diagnostic.  This script performs no inference and loads no
model weights.  It joins records by request ID, verifies that every selected
request was in fact a Clean-SFT failure, and reports:

* positive-margin rates and medians for each model/response region;
* paired CARE-minus-GKD margin deltas with item-bootstrap confidence intervals;
* CARE rescue rates across deterministic within-score rank quartiles;
* AUROC and point-biserial association with actual CARE rescue; and
* paired localized-minus-outside association differences.

These are frozen-parameter/behavior associations, not training ablations or
causal estimates of a particular CARE-OPD loss term.
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
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA = "care-opd-margin-behavior-v1"
MODEL_LABELS = ("clean_sft", "gkd_full", "care_opd")
REGIONS = ("full", "localized", "outside")
OUTPUT_FILES = (
    "margin_behavior.json",
    "margin_behavior_summary.csv",
    "margin_behavior_examples.csv",
    "margin_behavior.md",
)


class InputError(ValueError):
    """Raised when paired input archives violate the analysis contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
                raise InputError(f"{path}:{line_number}: row must be an object")
            value["__line__"] = line_number
            rows.append(value)
    if not rows:
        raise InputError(f"{path}: no JSONL records")
    return rows


def _nonempty(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}: expected a non-empty string")
    return value.strip()


def _finite(value: Any, *, where: str, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{where}: expected a finite number")
    return result


def _nested(row: Mapping[str, Any], key: str, *, where: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise InputError(f"{where}.{key}: expected an object")
    return value


def _margin_index(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = _nonempty(row.get("id"), where=f"{where}.id")
        label = _nonempty(row.get("model_label"), where=f"{where}.model_label")
        if label not in MODEL_LABELS:
            continue
        by_label = indexed.setdefault(item_id, {})
        if label in by_label:
            raise InputError(f"{where}: duplicate ({item_id!r}, {label!r})")

        full = _nested(row, "full", where=where)
        selected = _nested(row, "selected", where=where)
        outside = _nested(row, "outside", where=where)
        localized_complete = selected.get("localized_complete")
        if not isinstance(localized_complete, bool):
            raise InputError(f"{where}.selected.localized_complete: expected boolean")
        full_margin = _finite(full.get("margin"), where=f"{where}.full.margin")
        localized_margin = _finite(
            selected.get("localized_margin"),
            where=f"{where}.selected.localized_margin",
            optional=True,
        )
        outside_margin = _finite(
            outside.get("margin"), where=f"{where}.outside.margin", optional=True
        )
        if localized_complete and (localized_margin is None or outside_margin is None):
            raise InputError(f"{where}: localized row is missing localized/outside margin")
        if not localized_complete and localized_margin is not None:
            raise InputError(f"{where}: non-localized row exposes localized_margin")
        by_label[label] = {
            "full": full_margin,
            "localized": localized_margin if localized_complete else None,
            "outside": outside_margin if localized_complete else None,
            "localized_complete": localized_complete,
        }

    if not indexed:
        raise InputError(f"{path}: no supported model labels {MODEL_LABELS}")
    for item_id, by_label in indexed.items():
        missing = sorted(set(MODEL_LABELS) - set(by_label))
        if missing:
            raise InputError(f"{path}: id {item_id!r} is missing margin labels {missing}")
    return indexed


def _rollout_index(path: Path) -> dict[str, bool]:
    indexed: dict[str, bool] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = _nonempty(row.get("id"), where=f"{where}.id")
        if item_id in indexed:
            raise InputError(f"{where}: duplicate id {item_id!r}")
        passed = row.get("passed")
        if not isinstance(passed, bool):
            raise InputError(f"{where}.passed: expected a JSON boolean")
        indexed[item_id] = passed
    return indexed


def _mean(values: Iterable[float]) -> float | None:
    sequence = list(values)
    return statistics.fmean(sequence) if sequence else None


def _median(values: Iterable[float]) -> float | None:
    sequence = list(values)
    return statistics.median(sequence) if sequence else None


def _positive_rate(values: Iterable[float]) -> float | None:
    sequence = list(values)
    return sum(value > 0 for value in sequence) / len(sequence) if sequence else None


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_ci(
    rows: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float | None],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"lower": None, "upper": None, "valid_replicates": 0}
    rng = random.Random(seed)
    estimates: list[float] = []
    size = len(rows)
    for _ in range(replicates):
        sample = [rows[rng.randrange(size)] for _ in range(size)]
        estimate = statistic(sample)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return {"lower": None, "upper": None, "valid_replicates": 0}
    return {
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
        "valid_replicates": len(estimates),
    }


def _auroc(scores: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    if len(scores) != len(outcomes):
        raise ValueError("score/outcome length mismatch")
    positive_count = sum(outcomes)
    negative_count = len(outcomes) - positive_count
    if not positive_count or not negative_count:
        return None
    # Mann-Whitney U with average ranks for ties: O(n log n), which keeps the
    # 10k paired bootstrap inexpensive even on a CPU-only login node.
    ordered = sorted(zip(scores, outcomes), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            bool(outcome) for _, outcome in ordered[index:end]
        )
        index = end
    u_statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return u_statistic / (positive_count * negative_count)


def _point_biserial(scores: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    if len(scores) != len(outcomes) or len(scores) < 2:
        return None
    numeric_outcomes = [1.0 if outcome else 0.0 for outcome in outcomes]
    mean_x = statistics.fmean(scores)
    mean_y = statistics.fmean(numeric_outcomes)
    centered_x = [value - mean_x for value in scores]
    centered_y = [value - mean_y for value in numeric_outcomes]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def _quartiles(rows: Sequence[Mapping[str, Any]], score_key: str) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get(score_key) is not None]
    eligible.sort(key=lambda row: (float(row[score_key]), str(row["id"])))
    groups: list[list[Mapping[str, Any]]] = [[] for _ in range(4)]
    for index, row in enumerate(eligible):
        group = min(3, index * 4 // len(eligible))
        groups[group].append(row)
    result: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        scores = [float(row[score_key]) for row in group]
        rescued = sum(bool(row["care_rescued"]) for row in group)
        result.append(
            {
                "quartile": index,
                "n": len(group),
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                "score_mean": _mean(scores),
                "care_rescued": rescued,
                "care_rescue_rate": rescued / len(group) if group else None,
            }
        )
    return result


def _prediction(rows: Sequence[Mapping[str, Any]], score_key: str) -> dict[str, Any]:
    eligible = [row for row in rows if row.get(score_key) is not None]
    scores = [float(row[score_key]) for row in eligible]
    outcomes = [bool(row["care_rescued"]) for row in eligible]
    return {
        "score": score_key,
        "n": len(eligible),
        "positive_outcomes": sum(outcomes),
        "negative_outcomes": len(outcomes) - sum(outcomes),
        "auroc": _auroc(scores, outcomes),
        "point_biserial": _point_biserial(scores, outcomes),
    }


def _paired_prediction_difference(
    rows: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    metric: str,
) -> float | None:
    eligible = [
        row for row in rows if row.get(left) is not None and row.get(right) is not None
    ]
    outcomes = [bool(row["care_rescued"]) for row in eligible]
    fn = _auroc if metric == "auroc" else _point_biserial
    left_value = fn([float(row[left]) for row in eligible], outcomes)
    right_value = fn([float(row[right]) for row in eligible], outcomes)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _stable_seed(base: int, label: str) -> int:
    payload = hashlib.sha256(f"{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(payload[:8], "big")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _ci_text(row: Mapping[str, Any], *, percent: bool = False) -> str:
    lower, upper = row.get("ci95_lower"), row.get("ci95_upper")
    if lower is None or upper is None:
        return "N/A"
    if percent:
        return f"[{100 * lower:+.2f}, {100 * upper:+.2f}] pp"
    return f"[{lower:+.4f}, {upper:+.4f}]"


def _summary_csv(payload: Mapping[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for row in payload["model_margin_summaries"]:
        rows.append({"section": "model_margin", **row})
    for row in payload["care_minus_gkd"]:
        rows.append({"section": "care_minus_gkd", **row})
    rows.append({"section": "behavior", **payload["behavior"]})
    for score, groups in payload["quartile_recovery"].items():
        for row in groups:
            rows.append({"section": "quartile", "score": score, **row})
    for row in payload["prediction"]:
        rows.append({"section": "prediction", **row})
    for row in payload["localized_vs_outside_prediction"]:
        rows.append({"section": "prediction_difference", **row})
    columns = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _examples_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "id",
        "clean_failed",
        "gkd_rescued",
        "care_rescued",
        "care_only_rescue",
        "gkd_only_rescue",
    ]
    for label in MODEL_LABELS:
        for region in REGIONS:
            columns.append(f"{label}_{region}")
    for region in REGIONS:
        columns.append(f"care_minus_gkd_{region}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in columns} for row in rows)
    return buffer.getvalue()


def _markdown(payload: Mapping[str, Any]) -> str:
    behavior = payload["behavior"]
    lines = [
        "# CARE-OPD margin-to-behavior closure",
        "",
        "Status: **COMPLETE**. The cohort is the frozen, result-blind Clean-SFT "
        "failure-bank sample used by the recovery-margin diagnostic.",
        "",
        "## Behavioral rescue on the aligned cohort",
        "",
        f"- Aligned examples: `{behavior['n']}`",
        f"- CARE rescue rate: `{_pct(behavior['care_rescue_rate'])}`",
        f"- GKD rescue rate: `{_pct(behavior['gkd_rescue_rate'])}`",
        (
            f"- Paired CARE−GKD rescue-rate difference: "
            f"`{100 * behavior['care_minus_gkd_rescue_rate']:+.2f} pp` "
            f"(`{_ci_text(behavior, percent=True)}`)"
        ),
        "",
        "## Frozen-parameter preference summaries",
        "",
        "| Model | Region | n | Mean margin | Median margin | Positive margin |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["model_margin_summaries"]:
        lines.append(
            "| {model} | {region} | {n} | {mean} | {median} | {positive} |".format(
                model=row["model"],
                region=row["region"],
                n=row["n"],
                mean=_fmt(row["mean_margin"]),
                median=_fmt(row["median_margin"]),
                positive=_pct(row["positive_margin_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired CARE−GKD margin changes",
            "",
            "| Region | n | Mean delta | Median delta | Positive delta | 95% paired bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["care_minus_gkd"]:
        lines.append(
            "| {region} | {n} | {mean} | {median} | {positive} | {ci} |".format(
                region=row["region"],
                n=row["n"],
                mean=_fmt(row["mean_delta"]),
                median=_fmt(row["median_delta"]),
                positive=_pct(row["positive_delta_rate"]),
                ci=_ci_text(row),
            )
        )
    lines.extend(
        [
            "",
            "## Actual CARE rescue by margin rank quartile",
            "",
            "Quartiles are deterministic equal-count rank groups within each eligible score; "
            "ties are ordered by request ID.",
            "",
            "| Score | Q1 | Q2 | Q3 | Q4 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for score, groups in payload["quartile_recovery"].items():
        values = [f"{_pct(row['care_rescue_rate'])} (n={row['n']})" for row in groups]
        lines.append(f"| {score} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Association with actual CARE rescue",
            "",
            "| Score | n | AUROC | Point-biserial r |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["prediction"]:
        lines.append(
            f"| {row['score']} | {row['n']} | {_fmt(row['auroc'])} | "
            f"{_fmt(row['point_biserial'])} |"
        )
    lines.extend(
        [
            "",
            "### Localized versus outside association",
            "",
            "| Metric | n | Localized−outside | 95% paired bootstrap CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["localized_vs_outside_prediction"]:
        lines.append(
            f"| {row['metric']} | {row['n']} | {_fmt(row['difference'])} | "
            f"{_ci_text(row)} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Margins are teacher-forced, length-normalized oracle-minus-rejected log-probabilities; they are not free-generation accuracy.",
            "- Rescue is the production verifier's paired outcome on the same fixed request ID.",
            "- AUROC and point-biserial values measure association inside a Clean-SFT-failure-conditioned cohort. They do not establish that a margin caused rescue.",
            "- CARE−GKD compares complete training packages. It does not isolate one loss coefficient or the span mask without a matched training ablation.",
            f"- Confidence intervals resample request IDs as paired units for `{payload['bootstrap']['replicates']}` fixed-seed bootstrap replicates.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    margin_path: Path,
    clean_rollouts_path: Path,
    gkd_rollouts_path: Path,
    care_rollouts_path: Path,
    output_dir: Path,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260813,
    overwrite: bool = False,
) -> dict[str, Any]:
    if bootstrap_replicates < 100:
        raise InputError("bootstrap_replicates must be at least 100")
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    margins = _margin_index(margin_path)
    clean = _rollout_index(clean_rollouts_path)
    gkd = _rollout_index(gkd_rollouts_path)
    care = _rollout_index(care_rollouts_path)
    margin_ids = set(margins)
    for label, archive in (("clean_sft", clean), ("gkd_full", gkd), ("care_opd", care)):
        missing = sorted(margin_ids - set(archive))
        if missing:
            raise InputError(f"{label} rollouts are missing selected IDs: {missing[:10]}")
    contaminated = sorted(item_id for item_id in margin_ids if clean[item_id])
    if contaminated:
        raise InputError(
            "margin cohort is not Clean-SFT-failure-conditioned; passed IDs include "
            f"{contaminated[:10]}"
        )

    examples: list[dict[str, Any]] = []
    for item_id in sorted(margin_ids):
        row: dict[str, Any] = {
            "id": item_id,
            "clean_failed": True,
            "gkd_rescued": bool(gkd[item_id]),
            "care_rescued": bool(care[item_id]),
            "care_only_rescue": bool(care[item_id] and not gkd[item_id]),
            "gkd_only_rescue": bool(gkd[item_id] and not care[item_id]),
        }
        for label in MODEL_LABELS:
            for region in REGIONS:
                row[f"{label}_{region}"] = margins[item_id][label][region]
        for region in REGIONS:
            left = row[f"care_opd_{region}"]
            right = row[f"gkd_full_{region}"]
            row[f"care_minus_gkd_{region}"] = (
                left - right if left is not None and right is not None else None
            )
        examples.append(row)

    model_summaries: list[dict[str, Any]] = []
    for label in MODEL_LABELS:
        for region in REGIONS:
            values = [
                float(row[f"{label}_{region}"])
                for row in examples
                if row[f"{label}_{region}"] is not None
            ]
            model_summaries.append(
                {
                    "model": label,
                    "region": region,
                    "n": len(values),
                    "mean_margin": _mean(values),
                    "median_margin": _median(values),
                    "positive_margin_rate": _positive_rate(values),
                }
            )

    deltas: list[dict[str, Any]] = []
    for region in REGIONS:
        key = f"care_minus_gkd_{region}"
        eligible = [row for row in examples if row[key] is not None]
        values = [float(row[key]) for row in eligible]
        ci = _bootstrap_ci(
            eligible,
            lambda sample, k=key: _mean(float(row[k]) for row in sample),
            replicates=bootstrap_replicates,
            seed=_stable_seed(bootstrap_seed, f"delta:{region}"),
        )
        deltas.append(
            {
                "region": region,
                "n": len(values),
                "mean_delta": _mean(values),
                "median_delta": _median(values),
                "positive_delta_rate": _positive_rate(values),
                "ci95_lower": ci["lower"],
                "ci95_upper": ci["upper"],
                "bootstrap_valid_replicates": ci["valid_replicates"],
            }
        )

    behavior_delta = _mean(
        float(row["care_rescued"]) - float(row["gkd_rescued"]) for row in examples
    )
    behavior_ci = _bootstrap_ci(
        examples,
        lambda sample: _mean(
            float(row["care_rescued"]) - float(row["gkd_rescued"])
            for row in sample
        ),
        replicates=bootstrap_replicates,
        seed=_stable_seed(bootstrap_seed, "behavior"),
    )
    behavior = {
        "n": len(examples),
        "care_rescued": sum(bool(row["care_rescued"]) for row in examples),
        "gkd_rescued": sum(bool(row["gkd_rescued"]) for row in examples),
        "care_only_rescued": sum(bool(row["care_only_rescue"]) for row in examples),
        "gkd_only_rescued": sum(bool(row["gkd_only_rescue"]) for row in examples),
        "care_rescue_rate": _mean(float(row["care_rescued"]) for row in examples),
        "gkd_rescue_rate": _mean(float(row["gkd_rescued"]) for row in examples),
        "care_minus_gkd_rescue_rate": behavior_delta,
        "ci95_lower": behavior_ci["lower"],
        "ci95_upper": behavior_ci["upper"],
        "bootstrap_valid_replicates": behavior_ci["valid_replicates"],
    }

    score_keys = {
        "care_full": "care_opd_full",
        "care_localized": "care_opd_localized",
        "care_outside": "care_opd_outside",
        "care_minus_gkd_full": "care_minus_gkd_full",
        "care_minus_gkd_localized": "care_minus_gkd_localized",
        "care_minus_gkd_outside": "care_minus_gkd_outside",
    }
    quartiles = {name: _quartiles(examples, key) for name, key in score_keys.items()}
    predictions = [_prediction(examples, key) for key in score_keys.values()]

    paired_rows = [
        row
        for row in examples
        if row["care_opd_localized"] is not None and row["care_opd_outside"] is not None
    ]
    prediction_differences: list[dict[str, Any]] = []
    for metric in ("auroc", "point_biserial"):
        difference = _paired_prediction_difference(
            paired_rows,
            left="care_opd_localized",
            right="care_opd_outside",
            metric=metric,
        )
        ci = _bootstrap_ci(
            paired_rows,
            lambda sample, m=metric: _paired_prediction_difference(
                sample,
                left="care_opd_localized",
                right="care_opd_outside",
                metric=m,
            ),
            replicates=bootstrap_replicates,
            seed=_stable_seed(bootstrap_seed, f"prediction:{metric}"),
        )
        prediction_differences.append(
            {
                "metric": metric,
                "n": len(paired_rows),
                "difference": difference,
                "ci95_lower": ci["lower"],
                "ci95_upper": ci["upper"],
                "bootstrap_valid_replicates": ci["valid_replicates"],
            }
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "inputs": {
            "margin_jsonl": str(margin_path.resolve()),
            "margin_sha256": _sha256(margin_path),
            "clean_rollouts": str(clean_rollouts_path.resolve()),
            "clean_rollouts_sha256": _sha256(clean_rollouts_path),
            "gkd_rollouts": str(gkd_rollouts_path.resolve()),
            "gkd_rollouts_sha256": _sha256(gkd_rollouts_path),
            "care_rollouts": str(care_rollouts_path.resolve()),
            "care_rollouts_sha256": _sha256(care_rollouts_path),
        },
        "cohort_contract": (
            "IDs fixed by the result-blind recovery-margin selection; every aligned "
            "ID is verified to fail under Clean-SFT before CARE/GKD outcomes are read"
        ),
        "behavior": behavior,
        "model_margin_summaries": model_summaries,
        "care_minus_gkd": deltas,
        "quartile_recovery": quartiles,
        "prediction": predictions,
        "localized_vs_outside_prediction": prediction_differences,
        "bootstrap": {
            "unit": "request_id",
            "paired": True,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
        },
        "claim_boundary": (
            "Associational closure between frozen teacher-forced preference and actual "
            "verifier rescue inside a Clean-SFT-failure-conditioned cohort; not causal "
            "attribution to a single CARE-OPD component"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "margin_behavior.json", payload)
    _atomic_text(output_dir / "margin_behavior_summary.csv", _summary_csv(payload))
    _atomic_text(output_dir / "margin_behavior_examples.csv", _examples_csv(examples))
    _atomic_text(output_dir / "margin_behavior.md", _markdown(payload))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin-jsonl", type=Path, required=True)
    parser.add_argument("--clean-rollouts", type=Path, required=True)
    parser.add_argument("--gkd-rollouts", type=Path, required=True)
    parser.add_argument("--care-rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = analyze(
        margin_path=args.margin_jsonl,
        clean_rollouts_path=args.clean_rollouts,
        gkd_rollouts_path=args.gkd_rollouts,
        care_rollouts_path=args.care_rollouts,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": payload["status"], **payload["behavior"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
