#!/usr/bin/env python3
"""Aggregate completed per-seed causal summaries across training runs."""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

try:
    from .common import (
        AGGREGATE_SCHEMA,
        EVAL_SUITE_SCHEMA,
        atomic_json,
        atomic_text,
        load_json_object,
        require_keys,
        sha256,
        verify_file_binding,
        verify_path_binding,
    )
except ImportError:
    from common import (
        AGGREGATE_SCHEMA,
        EVAL_SUITE_SCHEMA,
        atomic_json,
        atomic_text,
        load_json_object,
        require_keys,
        sha256,
        verify_file_binding,
        verify_path_binding,
    )


METRICS: dict[str, tuple[str, str]] = {
    "clean_tsr": ("higher", "unified_metrics.clean_task_success_rate"),
    "perturbed_tsr": ("higher", "unified_metrics.perturbed_task_success_rate"),
    "conditional_retention": ("higher", "unified_metrics.conditional_retention"),
    "internal_regression": ("lower", "unified_metrics.regression_rate"),
    "reference_rescue": ("higher", "reference_conditioned_request_metrics.rescue_rate"),
    "reference_regression": ("lower", "reference_conditioned_request_metrics.regression_rate"),
    "net_success_delta": ("higher", "reference_conditioned_request_metrics.net_success_delta"),
}


def _descriptive(values: Sequence[float], seeds: Sequence[int], direction: str) -> dict[str, Any]:
    if len(values) != len(seeds) or len(values) < 2:
        raise ValueError("descriptive cross-seed statistics require at least two paired values")
    worst_value = min(values) if direction == "higher" else max(values)
    return {
        "direction": direction,
        "n_training_seeds": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "worst_value": worst_value,
        "worst_training_seeds": [
            seed for seed, value in zip(seeds, values) if value == worst_value
        ],
    }


def _finite_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a numeric value, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where}: expected a finite value, got {value!r}")
    return result


def _extract(summary: dict[str, Any], *, path: Path) -> tuple[str, list[str], dict[str, dict[str, float]]]:
    if summary.get("schema_version") != "care_opd.causal_ablation_summary.v1":
        raise ValueError(f"{path}: unexpected causal summary schema")
    models = summary.get("models")
    relative = summary.get("reference_conditioned_request_metrics")
    reference = summary.get("reference")
    if not isinstance(models, dict) or not models or not isinstance(relative, dict):
        raise ValueError(f"{path}: models and reference-conditioned metrics are required")
    if not isinstance(reference, str) or reference not in models:
        raise ValueError(f"{path}: valid reference arm is required")
    arms = list(models)
    result: dict[str, dict[str, float]] = {}
    for arm in arms:
        model = models[arm]
        rel = relative.get(arm)
        if not isinstance(model, dict) or not isinstance(model.get("unified_metrics"), dict):
            raise ValueError(f"{path}: arm {arm!r} lacks unified_metrics")
        if not isinstance(rel, dict):
            raise ValueError(f"{path}: arm {arm!r} lacks reference-conditioned metrics")
        unified = model["unified_metrics"]
        result[arm] = {
            "clean_tsr": _finite_number(unified.get("clean_task_success_rate"), where=f"{path}:{arm}:clean_tsr"),
            "perturbed_tsr": _finite_number(unified.get("perturbed_task_success_rate"), where=f"{path}:{arm}:perturbed_tsr"),
            "conditional_retention": _finite_number(unified.get("conditional_retention"), where=f"{path}:{arm}:conditional_retention"),
            "internal_regression": _finite_number(unified.get("regression_rate"), where=f"{path}:{arm}:internal_regression"),
            "reference_rescue": _finite_number(rel.get("rescue_rate"), where=f"{path}:{arm}:reference_rescue"),
            "reference_regression": _finite_number(rel.get("regression_rate"), where=f"{path}:{arm}:reference_regression"),
            "net_success_delta": _finite_number(rel.get("net_success_delta"), where=f"{path}:{arm}:net_success_delta"),
        }
    return reference, arms, result


def aggregate(evaluation_plan: Path, output_dir: Path) -> dict[str, Any]:
    plan = load_json_object(evaluation_plan, schema=EVAL_SUITE_SCHEMA)
    verify_file_binding(plan.get("training_suite"), where="evaluation plan training_suite")
    verify_path_binding(plan.get("requests"), where="evaluation plan requests")
    verify_path_binding(plan.get("family_clean_ids"), where="evaluation plan family_clean_ids")
    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise ValueError("evaluation plan must contain at least two training seeds")

    source_records: list[dict[str, Any]] = []
    raw: dict[str, dict[str, dict[str, float]]] = {}
    canonical_arms: list[str] | None = None
    canonical_reference: str | None = None
    seen_seeds: set[int] = set()
    for index, entry in enumerate(seeds):
        if not isinstance(entry, dict):
            raise ValueError(f"seeds[{index}] must be an object")
        require_keys(
            entry,
            ("training_seed", "training_plan", "summary", "evaluation_plan", "arm_names"),
            where=f"seeds[{index}]",
        )
        seed = entry["training_seed"]
        if not isinstance(seed, int) or seed < 0 or seed in seen_seeds:
            raise ValueError(f"seeds[{index}]: training_seed must be a unique non-negative integer")
        seen_seeds.add(seed)
        _, nested_training = verify_file_binding(
            entry["training_plan"],
            where=f"seeds[{index}].training_plan",
            schema="care_opd.causal_ablation_plan.v1",
        )
        if nested_training.get("budget_contract", {}).get("seed") != seed:
            raise ValueError(f"seed {seed}: training plan seed mismatch")
        _, nested_evaluation = verify_file_binding(
            entry["evaluation_plan"],
            where=f"seeds[{index}].evaluation_plan",
            schema="care_opd.causal_ablation_eval_plan.v1",
        )
        if nested_evaluation.get("training_plan") != entry["training_plan"]:
            raise ValueError(f"seed {seed}: evaluation plan is not bound to the declared training plan")
        if nested_evaluation.get("requests") != plan.get("requests"):
            raise ValueError(f"seed {seed}: evaluation request binding differs from the suite")
        summary_binding = entry["summary"]
        if not isinstance(summary_binding, dict) or not isinstance(summary_binding.get("path"), str):
            raise ValueError(f"seeds[{index}].summary.path must be a string")
        summary_path = Path(summary_binding["path"])
        summary = load_json_object(summary_path, schema="care_opd.causal_ablation_summary.v1")
        if summary.get("requests") != plan.get("requests"):
            raise ValueError(f"seed {seed}: summary request binding differs from the suite")
        if summary.get("family_ids") != plan.get("family_clean_ids"):
            raise ValueError(f"seed {seed}: summary family-ID binding differs from the suite")
        reference, summary_arms, values = _extract(summary, path=summary_path)
        declared_arms = entry["arm_names"]
        evaluation_arms = nested_evaluation.get("arms")
        if (
            not isinstance(declared_arms, list)
            or not all(isinstance(arm, str) and arm for arm in declared_arms)
            or len(set(declared_arms)) != len(declared_arms)
            or set(summary_arms) != set(declared_arms)
        ):
            raise ValueError(f"seed {seed}: summary arms differ from the declared arm set")
        arms = list(declared_arms)
        if not isinstance(evaluation_arms, list) or [arm.get("name") for arm in evaluation_arms] != arms:
            raise ValueError(f"seed {seed}: evaluation arms differ from the summary arm order")
        for arm in evaluation_arms:
            expected_source = (Path(arm["rollout_output"]) / "qzbc_rollouts.jsonl").resolve()
            source = summary["models"][arm["name"]].get("source")
            source_path = verify_path_binding(
                source,
                where=f"seed {seed} arm {arm['name']} summary source",
            )
            if source_path.resolve() != expected_source:
                raise ValueError(
                    f"seed {seed} arm {arm['name']}: summary source is not the planned rollout"
                )
        if canonical_arms is None:
            canonical_arms, canonical_reference = arms, reference
        elif arms != canonical_arms or reference != canonical_reference:
            raise ValueError(f"seed {seed}: arm order or reference differs from the first seed")
        raw[str(seed)] = values
        source_records.append(
            {
                "training_seed": seed,
                "summary": {"path": str(summary_path.resolve()), "sha256": sha256(summary_path)},
                "within_seed_bootstrap_is_training_variance": False,
            }
        )

    assert canonical_arms is not None and canonical_reference is not None
    aggregates: dict[str, dict[str, Any]] = {}
    ordered_seeds = [entry["training_seed"] for entry in source_records]
    for arm in canonical_arms:
        aggregates[arm] = {}
        for metric, (direction, definition) in METRICS.items():
            values = [raw[str(seed)][arm][metric] for seed in ordered_seeds]
            aggregates[arm][metric] = {
                **_descriptive(values, ordered_seeds, direction),
                "source_definition": definition,
            }

    contrast_contract = plan.get("contrasts", {"source": None, "items": []})
    if not isinstance(contrast_contract, dict):
        raise ValueError("evaluation plan contrasts must be an object")
    contrast_source = contrast_contract.get("source")
    if contrast_source is not None:
        verify_path_binding(contrast_source, where="evaluation plan contrast source")
    contrast_specs = contrast_contract.get("items", [])
    if not isinstance(contrast_specs, list):
        raise ValueError("evaluation plan contrasts.items must be a list")
    contrast_results: dict[str, dict[str, Any]] = {}
    seen_contrasts: set[str] = set()
    for index, spec in enumerate(contrast_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"contrasts.items[{index}] must be an object")
        name, left, right = spec.get("name"), spec.get("left"), spec.get("right")
        if not isinstance(name, str) or not name or name in seen_contrasts:
            raise ValueError(f"contrasts.items[{index}] has an invalid/duplicate name")
        if left not in canonical_arms or right not in canonical_arms or left == right:
            raise ValueError(f"contrast {name!r} does not bind two distinct planned arms")
        seen_contrasts.add(name)
        per_seed_delta: dict[str, dict[str, float]] = {}
        metric_results: dict[str, Any] = {}
        for seed in ordered_seeds:
            per_seed_delta[str(seed)] = {
                metric: raw[str(seed)][left][metric] - raw[str(seed)][right][metric]
                for metric in METRICS
            }
        for metric, (direction, definition) in METRICS.items():
            values = [per_seed_delta[str(seed)][metric] for seed in ordered_seeds]
            metric_results[metric] = {
                **_descriptive(values, ordered_seeds, direction),
                "source_definition": definition,
                "delta_definition": f"{left} minus {right} within the same training seed",
            }
        contrast_results[name] = {
            "left_arm": left,
            "right_arm": right,
            "definition": spec.get("definition", ""),
            "per_seed_delta": per_seed_delta,
            "metrics": metric_results,
        }

    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "evaluation_plan": {"path": str(evaluation_plan.resolve()), "sha256": sha256(evaluation_plan)},
        "reference_arm": canonical_reference,
        "training_seeds": ordered_seeds,
        "n_training_seeds": len(ordered_seeds),
        "arms": canonical_arms,
        "metrics": {
            metric: {"direction": direction, "source_definition": definition}
            for metric, (direction, definition) in METRICS.items()
        },
        "per_seed": raw,
        "aggregate": aggregates,
        "contrast_contract": contrast_contract,
        "paired_contrasts": contrast_results,
        "sources": source_records,
        "statistical_contract": {
            "mean": "arithmetic mean over independently trained seed runs",
            "sample_std": "sample standard deviation over training seeds (ddof=1)",
            "worst_seed": "minimum for higher-is-better metrics; maximum for lower-is-better metrics",
            "confidence_interval": None,
            "paired_contrast": (
                "Every contrast is computed left-minus-right within each training seed before "
                "the mean and sample SD are calculated."
            ),
            "boundary": (
                "The paired request/seed-cluster bootstrap stored inside each source summary quantifies "
                "evaluation-sample uncertainty conditional on one trained checkpoint. It does not estimate "
                "between-training-seed variance and is not pooled here as if it did. With only three training "
                "seeds, mean, sample SD, and worst seed are descriptive rather than a stable population CI."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "multiseed_summary.json", result)
    per_seed_stream = io.StringIO(newline="")
    per_seed_writer = csv.DictWriter(
        per_seed_stream, fieldnames=["training_seed", "arm", "metric", "direction", "value"]
    )
    per_seed_writer.writeheader()
    aggregate_stream = io.StringIO(newline="")
    aggregate_writer = csv.DictWriter(
        aggregate_stream,
        fieldnames=[
            "arm", "metric", "direction", "n_training_seeds", "mean", "sample_std",
            "min", "max", "worst_value", "worst_training_seeds",
        ],
    )
    aggregate_writer.writeheader()
    contrast_per_seed_stream = io.StringIO(newline="")
    contrast_per_seed_writer = csv.DictWriter(
        contrast_per_seed_stream,
        fieldnames=[
            "training_seed", "contrast", "left_arm", "right_arm", "metric",
            "direction", "delta_left_minus_right",
        ],
    )
    contrast_per_seed_writer.writeheader()
    contrast_aggregate_stream = io.StringIO(newline="")
    contrast_aggregate_writer = csv.DictWriter(
        contrast_aggregate_stream,
        fieldnames=[
            "contrast", "left_arm", "right_arm", "metric", "direction",
            "n_training_seeds", "mean_delta", "sample_std_delta", "min_delta",
            "max_delta", "worst_delta", "worst_training_seeds",
        ],
    )
    contrast_aggregate_writer.writeheader()
    for arm in canonical_arms:
        for metric, (direction, _) in METRICS.items():
            for seed in ordered_seeds:
                per_seed_writer.writerow(
                    {
                        "training_seed": seed,
                        "arm": arm,
                        "metric": metric,
                        "direction": direction,
                        "value": raw[str(seed)][arm][metric],
                    }
                )
            item = aggregates[arm][metric]
            aggregate_writer.writerow(
                {
                    "arm": arm,
                    "metric": metric,
                    "direction": direction,
                    "n_training_seeds": item["n_training_seeds"],
                    "mean": item["mean"],
                    "sample_std": item["sample_std"],
                    "min": item["min"],
                    "max": item["max"],
                    "worst_value": item["worst_value"],
                    "worst_training_seeds": ";".join(str(seed) for seed in item["worst_training_seeds"]),
                }
            )
    atomic_text(output_dir / "per_seed.csv", per_seed_stream.getvalue())
    atomic_text(output_dir / "aggregate.csv", aggregate_stream.getvalue())
    for name, contrast in contrast_results.items():
        for metric, (direction, _) in METRICS.items():
            for seed in ordered_seeds:
                contrast_per_seed_writer.writerow(
                    {
                        "training_seed": seed,
                        "contrast": name,
                        "left_arm": contrast["left_arm"],
                        "right_arm": contrast["right_arm"],
                        "metric": metric,
                        "direction": direction,
                        "delta_left_minus_right": contrast["per_seed_delta"][str(seed)][metric],
                    }
                )
            item = contrast["metrics"][metric]
            contrast_aggregate_writer.writerow(
                {
                    "contrast": name,
                    "left_arm": contrast["left_arm"],
                    "right_arm": contrast["right_arm"],
                    "metric": metric,
                    "direction": direction,
                    "n_training_seeds": item["n_training_seeds"],
                    "mean_delta": item["mean"],
                    "sample_std_delta": item["sample_std"],
                    "min_delta": item["min"],
                    "max_delta": item["max"],
                    "worst_delta": item["worst_value"],
                    "worst_training_seeds": ";".join(
                        str(seed) for seed in item["worst_training_seeds"]
                    ),
                }
            )
    atomic_text(output_dir / "contrast_per_seed.csv", contrast_per_seed_stream.getvalue())
    atomic_text(output_dir / "contrast_aggregate.csv", contrast_aggregate_stream.getvalue())

    pct = lambda value: f"{100 * value:.2f}%"
    lines = [
        "# CARE-OPD multi-training-seed ablation summary",
        "",
        f"Training seeds: `{', '.join(str(seed) for seed in ordered_seeds)}`. Values below are mean ± sample SD; Worst is the explicitly direction-aware worst training seed.",
        "",
        "| Arm | Perturbed TSR ↑ | Conditional retention ↑ | Reference rescue ↑ | Reference regression ↓ | Net success Δ ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selected = (
        "perturbed_tsr", "conditional_retention", "reference_rescue",
        "reference_regression", "net_success_delta",
    )
    for arm in canonical_arms:
        cells = []
        for metric in selected:
            item = aggregates[arm][metric]
            worst = ",".join(str(seed) for seed in item["worst_training_seeds"])
            cells.append(f"{pct(item['mean'])} ± {pct(item['sample_std'])} (worst s={worst}: {pct(item['worst_value'])})")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    if contrast_results:
        lines += [
            "",
            "## Pre-specified paired arm contrasts",
            "",
            "Every delta is left minus right within the same training seed, then summarized across training seeds.",
            "",
            "| Contrast (left - right) | Perturbed TSR delta | Retention delta | Rescue delta | Regression delta | Net delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, contrast in contrast_results.items():
            cells = []
            for metric in selected:
                item = contrast["metrics"][metric]
                worst = ",".join(str(seed) for seed in item["worst_training_seeds"])
                cells.append(
                    f"{pct(item['mean'])} +/- {pct(item['sample_std'])} "
                    f"(worst s={worst}: {pct(item['worst_value'])})"
                )
            label = f"{name} ({contrast['left_arm']} - {contrast['right_arm']})"
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Statistical boundary",
        "",
        "The standard deviation above is computed across independently trained checkpoints (`ddof=1`). It is not reconstructed from a request-level or seed-ID-cluster bootstrap.",
        "",
        "The paired bootstrap intervals inside each per-seed `summary.json` remain conditional evaluation-sample uncertainty. They do **not** measure training variance and must not be substituted for repeated training seeds. With three training seeds, mean/SD/worst-seed are descriptive; no cross-seed confidence interval is reported.",
        "",
    ]
    atomic_text(output_dir / "multiseed_summary.md", "\n".join(lines))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = aggregate(args.evaluation_plan.resolve(), args.output_dir.resolve())
    print(f"training_seeds={','.join(str(seed) for seed in result['training_seeds'])}")
    print(f"summary={args.output_dir.resolve() / 'multiseed_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
