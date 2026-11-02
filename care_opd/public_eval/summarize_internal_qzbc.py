#!/usr/bin/env python3
"""Re-score the internal QZBC held-out suite as clean/perturbed pairs.

The legacy ``baseline_metrics.json`` reports a mixed pass rate over clean and
perturbed requests.  This script joins every rollout back to the compiled
request by ``id`` and then pairs variants by ``seed_id``.  It emits the same
effectiveness/robustness field names used by the public benchmark harnesses.

No model inference is performed here; this is a deterministic re-analysis of
already completed ``qzbc_rollouts.jsonl`` files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_ORDER = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)
SCHEMA_VERSION = "care-opd-internal-qzbc-paired-summary-v1"
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPLICATES = 2000
HERE = Path(__file__).resolve().parent
UNIFIED_SCHEMA = HERE / "UNIFIED_METRIC_SCHEMA.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _unique_by_id(rows: Iterable[Mapping[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id") or "").strip()
        if not item_id:
            raise ValueError(f"{source}: row without id")
        if item_id in result:
            raise ValueError(f"{source}: duplicate id {item_id}")
        result[item_id] = dict(row)
    return result


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_contributions(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, int, int, int]]:
    """Return seed -> (clean_success_sum, pert_success, both_success, pairs)."""

    result: dict[str, list[int]] = {}
    for row in pair_rows:
        seed_id = str(row["seed_id"])
        bucket = result.setdefault(seed_id, [0, 0, 0, 0])
        clean = bool(row["clean_success"])
        perturbed = bool(row["perturbed_success"])
        bucket[0] += int(clean)
        bucket[1] += int(perturbed)
        bucket[2] += int(clean and perturbed)
        bucket[3] += 1
    return {key: tuple(value) for key, value in result.items()}


def _rates_from_totals(clean: int, perturbed: int, both: int, total: int) -> dict[str, float | None]:
    return {
        "clean_task_success_rate": clean / total if total else None,
        "perturbed_task_success_rate": perturbed / total if total else None,
        "conditional_retention": both / clean if clean else None,
        "regression_rate": (clean - both) / clean if clean else None,
    }


def _cluster_bootstrap(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    contributions = _paired_contributions(pair_rows)
    seed_ids = sorted(contributions)
    if not seed_ids:
        raise ValueError("cannot bootstrap an empty paired set")
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        totals = [0, 0, 0, 0]
        for _draw in seed_ids:
            values = contributions[rng.choice(seed_ids)]
            for index, value in enumerate(values):
                totals[index] += value
        rates = _rates_from_totals(*totals)
        for key, value in rates.items():
            if value is not None:
                samples[key].append(value)
    return {
        "method": "nonparametric seed_id-cluster percentile bootstrap",
        "cluster_unit": "seed_id",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "intervals": {
            key: {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
                "replicates_used": len(values),
            }
            for key, values in sorted(samples.items())
        },
    }


def _exact_mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict[str, Any]:
    if len(a) != len(b):
        raise ValueError("paired comparison length mismatch")
    a_only = sum(left and not right for left, right in zip(a, b, strict=True))
    b_only = sum(right and not left for left, right in zip(a, b, strict=True))
    discordant = a_only + b_only
    if not discordant:
        probability = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(a_only, b_only) + 1)
        ) / (2**discordant)
        probability = min(1.0, 2.0 * tail)
    return {
        "a_only_success": a_only,
        "b_only_success": b_only,
        "discordant": discordant,
        "two_sided_exact_p": probability,
        "note": "Exploratory unclustered item-level McNemar; use seed-cluster bootstrap for primary inference.",
    }


def _model_pairs(
    request_by_id: Mapping[str, Mapping[str, Any]],
    rollout_path: Path,
    *,
    clean_variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rollout_by_id = _unique_by_id(_read_jsonl(rollout_path), rollout_path)
    expected = set(request_by_id)
    observed = set(rollout_by_id)
    if expected != observed:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise ValueError(
            f"{rollout_path}: request/rollout ID mismatch; missing={missing}, extra={extra}"
        )

    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_types: Counter[str] = Counter()
    overall_passed = 0
    for item_id in sorted(expected):
        request = request_by_id[item_id]
        rollout = rollout_by_id[item_id]
        seed_id = str(request.get("seed_id") or "").strip()
        variant = str(request.get("variant") or "").strip()
        if not seed_id or not variant:
            raise ValueError(f"request {item_id}: seed_id and variant are required")
        if str(rollout.get("variant") or "") != variant:
            raise ValueError(f"{item_id}: rollout variant does not match request")
        passed = bool(rollout.get("passed"))
        overall_passed += int(passed)
        for failure in rollout.get("failure_types") or []:
            failure_types[str(failure)] += 1
        by_seed[seed_id].append(
            {
                "id": item_id,
                "seed_id": seed_id,
                "variant": variant,
                "passed": passed,
                "failure_types": [str(item) for item in rollout.get("failure_types") or []],
            }
        )

    variant_sets: Counter[tuple[str, ...]] = Counter()
    pairs: list[dict[str, Any]] = []
    clean_unique = 0
    per_variant: dict[str, list[bool]] = defaultdict(list)
    for seed_id, rows in sorted(by_seed.items()):
        clean_rows = [row for row in rows if row["variant"] == clean_variant]
        if len(clean_rows) != 1:
            raise ValueError(f"seed {seed_id}: expected one {clean_variant}, got {len(clean_rows)}")
        perturbed_rows = [row for row in rows if row["variant"] != clean_variant]
        if not perturbed_rows:
            raise ValueError(f"seed {seed_id}: no perturbed variants")
        variant_sets[tuple(sorted(row["variant"] for row in perturbed_rows))] += 1
        clean = clean_rows[0]
        clean_unique += int(clean["passed"])
        for perturbed in sorted(perturbed_rows, key=lambda row: row["variant"]):
            per_variant[perturbed["variant"]].append(bool(perturbed["passed"]))
            pairs.append(
                {
                    "seed_id": seed_id,
                    "clean_id": clean["id"],
                    "perturbed_id": perturbed["id"],
                    "perturbation": perturbed["variant"],
                    "clean_success": bool(clean["passed"]),
                    "perturbed_success": bool(perturbed["passed"]),
                    "clean_failure_types": clean["failure_types"],
                    "perturbed_failure_types": perturbed["failure_types"],
                }
            )
    if len(variant_sets) != 1:
        raise ValueError(f"unbalanced perturbation families across seeds: {dict(variant_sets)}")

    total_pairs = len(pairs)
    clean_success = sum(row["clean_success"] for row in pairs)
    perturbed_success = sum(row["perturbed_success"] for row in pairs)
    both_success = sum(
        row["clean_success"] and row["perturbed_success"] for row in pairs
    )
    clean_to_wrong = clean_success - both_success
    wrong_to_correct = sum(
        (not row["clean_success"]) and row["perturbed_success"] for row in pairs
    )
    rates = _rates_from_totals(clean_success, perturbed_success, both_success, total_pairs)
    variant_metrics = {
        variant: {
            "n": len(values),
            "passed": sum(values),
            "task_success_rate": sum(values) / len(values),
        }
        for variant, values in sorted(per_variant.items())
    }
    metric = {
        **rates,
        "clean_unique_units": len(by_seed),
        "clean_unique_successes": clean_unique,
        "clean_unique_task_success_rate": clean_unique / len(by_seed),
        "clean_success_pair_count": clean_success,
        "perturbed_success_count": perturbed_success,
        "both_correct": both_success,
        "clean_correct_perturbed_wrong": clean_to_wrong,
        "clean_wrong_perturbed_correct": wrong_to_correct,
        "both_wrong": total_pairs - both_success - clean_to_wrong - wrong_to_correct,
        "paired_units": total_pairs,
        "robustness_drop_pp": 100.0
        * (float(rates["clean_task_success_rate"]) - float(rates["perturbed_task_success_rate"])),
        "prediction_consistency": None,
        "prediction_consistency_reason": (
            "N/A: clean and recovery variants have intentionally different target responses; "
            "literal answer consistency is not a valid construct."
        ),
        "worst_perturbation_tsr": min(
            value["task_success_rate"] for value in variant_metrics.values()
        ),
        "by_perturbation": variant_metrics,
        "cluster_bootstrap_ci95": _cluster_bootstrap(pairs),
    }
    diagnostics = {
        "rollouts": len(rollout_by_id),
        "passed": overall_passed,
        "overall_mixed_task_success_rate": overall_passed / len(rollout_by_id),
        "seed_groups": len(by_seed),
        "perturbation_families": list(next(iter(variant_sets))),
        "failure_type_counts": dict(sorted(failure_types.items())),
        "rollouts_sha256": _sha256(rollout_path),
    }
    return pairs, {"unified_metrics": metric, "diagnostics": diagnostics}


def _bootstrap_delta(
    care_pairs: Sequence[Mapping[str, Any]],
    baseline_pairs: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    key = lambda row: (str(row["seed_id"]), str(row["perturbation"]))
    care_map = {key(row): row for row in care_pairs}
    baseline_map = {key(row): row for row in baseline_pairs}
    if care_map.keys() != baseline_map.keys():
        raise ValueError("model pair keys differ")
    by_seed: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for item_key in sorted(care_map):
        by_seed[item_key[0]].append((care_map[item_key], baseline_map[item_key]))
    seed_ids = sorted(by_seed)
    rng = random.Random(seed)
    sampled: dict[str, list[float]] = defaultdict(list)

    def model_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
        clean = sum(bool(row["clean_success"]) for row in rows)
        perturbed = sum(bool(row["perturbed_success"]) for row in rows)
        both = sum(
            bool(row["clean_success"] and row["perturbed_success"]) for row in rows
        )
        return _rates_from_totals(clean, perturbed, both, len(rows))

    care_rates = model_rates(list(care_map.values()))
    baseline_rates = model_rates(list(baseline_map.values()))
    for _ in range(replicates):
        care_sample: list[Mapping[str, Any]] = []
        baseline_sample: list[Mapping[str, Any]] = []
        for _draw in seed_ids:
            rows = by_seed[rng.choice(seed_ids)]
            care_sample.extend(care for care, _baseline in rows)
            baseline_sample.extend(baseline for _care, baseline in rows)
        left = model_rates(care_sample)
        right = model_rates(baseline_sample)
        for metric in ("perturbed_task_success_rate", "conditional_retention"):
            if left[metric] is not None and right[metric] is not None:
                sampled[metric].append(float(left[metric]) - float(right[metric]))
    return {
        "method": "paired seed_id-cluster percentile bootstrap",
        "seed": seed,
        "replicates": replicates,
        "metrics": {
            metric: {
                "care_opd": care_rates[metric],
                "baseline": baseline_rates[metric],
                "delta": (
                    float(care_rates[metric]) - float(baseline_rates[metric])
                    if care_rates[metric] is not None and baseline_rates[metric] is not None
                    else None
                ),
                "ci95_lower": _percentile(values, 0.025),
                "ci95_upper": _percentile(values, 0.975),
                "replicates_used": len(values),
            }
            for metric, values in sorted(sampled.items())
        },
        "exploratory_unclustered_mcnemar_perturbed_success": _exact_mcnemar(
            [bool(care_map[item]["perturbed_success"]) for item in sorted(care_map)],
            [bool(baseline_map[item]["perturbed_success"]) for item in sorted(baseline_map)],
        ),
    }


def _validate_schema() -> dict[str, str]:
    payload = json.loads(UNIFIED_SCHEMA.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "care-opd-unified-effectiveness-robustness-v1":
        raise ValueError(f"unexpected unified schema: {UNIFIED_SCHEMA}")
    return {
        "path": str(UNIFIED_SCHEMA.resolve()),
        "schema_version": str(payload["schema_version"]),
        "sha256": _sha256(UNIFIED_SCHEMA),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    requests_path = args.requests.resolve(strict=True)
    eval_root = args.eval_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    request_by_id = _unique_by_id(_read_jsonl(requests_path), requests_path)
    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    if not labels:
        raise ValueError("no model labels selected")
    models: dict[str, Any] = {}
    pairs_by_model: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        rollout_path = eval_root / label / "qzbc_rollouts.jsonl"
        if not rollout_path.is_file():
            if args.allow_missing_models:
                continue
            raise FileNotFoundError(rollout_path)
        pairs, payload = _model_pairs(
            request_by_id, rollout_path, clean_variant=args.clean_variant
        )
        summary_path = eval_root / label / "summary.json"
        if summary_path.is_file():
            legacy = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(legacy.get("rollouts", -1)) != payload["diagnostics"]["rollouts"]:
                raise ValueError(f"{label}: legacy summary rollout count mismatch")
            if int(legacy.get("passed", -1)) != payload["diagnostics"]["passed"]:
                raise ValueError(f"{label}: legacy summary passed count mismatch")
            payload["diagnostics"]["legacy_summary_sha256"] = _sha256(summary_path)
        models[label] = payload
        pairs_by_model[label] = pairs
        _atomic_text(
            output_dir / label / "paired_predictions.jsonl",
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pairs
            ),
        )

    comparisons: dict[str, Any] = {}
    if "care_opd" in pairs_by_model:
        for baseline in ("clean_sft", "gkd_full"):
            if baseline in pairs_by_model:
                comparisons[f"care_opd_vs_{baseline}"] = _bootstrap_delta(
                    pairs_by_model["care_opd"], pairs_by_model[baseline]
                )
    declared_suite = None
    if args.model_suite is not None:
        suite_path = args.model_suite.resolve(strict=True)
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        suite_models = suite.get("models")
        if not isinstance(suite_models, list):
            raise ValueError(f"model suite has no model list: {suite_path}")
        suite_labels = tuple(str(item.get("label") or "") for item in suite_models)
        if suite_labels != labels:
            raise ValueError(
                f"declared suite labels differ from legacy result labels: {suite_labels} != {labels}"
            )
        semantic = [
            (str(item["label"]), str(item.get("model_fingerprint") or ""))
            for item in suite_models
        ]
        if any(len(fingerprint) != 64 for _label, fingerprint in semantic):
            raise ValueError("declared suite contains an invalid model fingerprint")
        suite_fingerprint = _canonical_hash(semantic)
        if suite.get("suite_fingerprint") != suite_fingerprint:
            raise ValueError("declared suite fingerprint does not close")
        declared_suite = {
            "path": str(suite_path),
            "sha256": _sha256(suite_path),
            "suite_fingerprint": suite_fingerprint,
            "models": [
                {"label": label, "model_fingerprint": fingerprint}
                for label, fingerprint in semantic
            ],
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "track_provenance": "Internal retrospective / QZBC grouped test",
        "claim_scope": (
            "Legacy in-domain diagnostic retained in the main table; the old rollout files "
            "do not embed checkpoint fingerprints or a split-integrity manifest, so they "
            "cannot support formal OPD attribution until rerun with the frozen suite."
        ),
        "checkpoint_identity_binding": {
            "status": "RETROSPECTIVE_UNVERIFIED",
            "reason": (
                "qzbc_rollouts.jsonl and its legacy summary contain no model fingerprint "
                "or signed run contract; the suite below is the intended comparison set, "
                "not proof of which bytes generated the old rows."
            ),
            "declared_expected_suite": declared_suite,
        },
        "split_integrity_binding": {
            "status": "RETROSPECTIVE_UNVERIFIED",
            "reason": (
                "the legacy evaluator did not embed the formal seed/split manifest; request "
                "path and SHA are retained, but 'held-out' is not asserted by this re-analysis"
            ),
        },
        "requests": {
            "path": str(requests_path),
            "sha256": _sha256(requests_path),
            "rows": len(request_by_id),
        },
        "unified_metric_schema": _validate_schema(),
        "clean_variant": args.clean_variant,
        "models": models,
        "primary_pairwise_comparisons": comparisons,
    }
    result["summary_fingerprint"] = _canonical_hash(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "suite_summary.json", result)

    fieldnames = [
        "model",
        "episodes",
        "overall_mixed_task_success_rate",
        "clean_task_success_rate",
        "perturbed_task_success_rate",
        "conditional_retention",
        "regression_rate",
        "robustness_drop_pp",
        "prediction_consistency",
        "worst_perturbation_tsr",
        "clean_correct_perturbed_wrong",
        "clean_wrong_perturbed_correct",
        "paired_units",
    ]
    csv_path = output_dir / "suite_summary.csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for label, payload in models.items():
                metric = payload["unified_metrics"]
                writer.writerow(
                    {
                        "model": label,
                        "episodes": payload["diagnostics"]["rollouts"],
                        "overall_mixed_task_success_rate": payload["diagnostics"][
                            "overall_mixed_task_success_rate"
                        ],
                        **{key: metric.get(key) for key in fieldnames[3:]},
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    lines = [
        "# Internal QZBC paired effectiveness and robustness",
        "",
        "**Evidence status: retrospective / checkpoint identity and split manifest unverified.** "
        "Keep these rows in the comparison table for continuity, but do not use them for formal "
        "OPD attribution until the frozen-suite rerun is complete.",
        "",
        "`overall mixed TSR` is retained for continuity with the old 4,200-row table; "
        "it is not the robustness endpoint.",
        "",
        "| Model | Mixed TSR | Clean TSR | Perturbed TSR | Retention | Regression | Drop (pp) | Worst perturbation | C→W / W→C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, payload in models.items():
        metric = payload["unified_metrics"]
        diagnostics = payload["diagnostics"]
        pct = lambda value: "N/A" if value is None else f"{100.0 * float(value):.2f}%"
        lines.append(
            f"| {label} | {pct(diagnostics['overall_mixed_task_success_rate'])} | "
            f"{pct(metric['clean_task_success_rate'])} | {pct(metric['perturbed_task_success_rate'])} | "
            f"{pct(metric['conditional_retention'])} | {pct(metric['regression_rate'])} | "
            f"{metric['robustness_drop_pp']:.2f} | {pct(metric['worst_perturbation_tsr'])} | "
            f"{metric['clean_correct_perturbed_wrong']} / {metric['clean_wrong_perturbed_correct']} |"
        )
    _atomic_text(output_dir / "suite_summary.md", "\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-suite",
        type=Path,
        help=(
            "intended frozen six-model suite to record for disclosure only; legacy rollouts "
            "cannot be cryptographically bound to it"
        ),
    )
    parser.add_argument("--labels", default=",".join(MODEL_ORDER))
    parser.add_argument("--clean-variant", default="clean")
    parser.add_argument("--allow-missing-models", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    summarize(make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
