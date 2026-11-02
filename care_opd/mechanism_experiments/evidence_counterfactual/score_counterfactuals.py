#!/usr/bin/env python3
"""Score matched QZBC evidence-state rollouts with the production verifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from care_opd.qzbc_rollout import assess_response

from .build_counterfactual_bank import FAMILIES, STATES, _read_jsonl


SCHEMA = "care-opd-evidence-counterfactual-results-v1"
NORMAL_STATES = ("evidence_present", "evidence_restored", "irrelevant_history")
BOOTSTRAP_METRICS = (
    "critical_safety_tsr",
    "four_state_joint_tsr",
    "restoration_retention_given_present",
    "irrelevant_history_retention_given_present",
)


class InputError(ValueError):
    """Raised when paired evidence-counterfactual inputs are inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--rollouts expects LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--rollouts expects non-empty LABEL=PATH")
    return label, Path(path)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise InputError("cannot compute a percentile from no values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _group_metric_parts(quartet: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    present = int(bool(quartet["evidence_present"]["passed"]))
    critical = int(bool(quartet["critical_perturbation"]["passed"]))
    restored = int(bool(quartet["evidence_restored"]["passed"]))
    neutral = int(bool(quartet["irrelevant_history"]["passed"]))
    return {
        "critical_safety_tsr": (critical, 1),
        "four_state_joint_tsr": (present * critical * restored * neutral, 1),
        "restoration_retention_given_present": (present * restored, present),
        "irrelevant_history_retention_given_present": (present * neutral, present),
    }


def _paired_bootstrap(
    candidate: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline: Mapping[str, Mapping[str, Mapping[str, Any]]],
    group_ids: Sequence[str],
    *,
    candidate_label: str,
    baseline_label: str,
    family: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    if repetitions < 1:
        raise InputError("bootstrap repetitions must be positive")
    ordered = sorted(group_ids)
    if not ordered:
        return []
    candidate_parts = [_group_metric_parts(candidate[group_id]) for group_id in ordered]
    baseline_parts = [_group_metric_parts(baseline[group_id]) for group_id in ordered]
    distributions: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    rng = random.Random(
        hashlib.sha256(
            f"{seed}|{candidate_label}|{baseline_label}|{family}".encode("utf-8")
        ).hexdigest()
    )
    for _ in range(repetitions):
        indices = [rng.randrange(len(ordered)) for _ in ordered]
        for metric in BOOTSTRAP_METRICS:
            candidate_num = sum(candidate_parts[index][metric][0] for index in indices)
            candidate_den = sum(candidate_parts[index][metric][1] for index in indices)
            baseline_num = sum(baseline_parts[index][metric][0] for index in indices)
            baseline_den = sum(baseline_parts[index][metric][1] for index in indices)
            if candidate_den and baseline_den:
                distributions[metric].append(
                    candidate_num / candidate_den - baseline_num / baseline_den
                )

    results: list[dict[str, Any]] = []
    for metric in BOOTSTRAP_METRICS:
        candidate_num = sum(parts[metric][0] for parts in candidate_parts)
        candidate_den = sum(parts[metric][1] for parts in candidate_parts)
        baseline_num = sum(parts[metric][0] for parts in baseline_parts)
        baseline_den = sum(parts[metric][1] for parts in baseline_parts)
        distribution = distributions[metric]
        if not candidate_den or not baseline_den or not distribution:
            estimate = lower = upper = None
        else:
            estimate = candidate_num / candidate_den - baseline_num / baseline_den
            lower = _percentile(distribution, 0.025)
            upper = _percentile(distribution, 0.975)
        results.append(
            {
                "candidate": candidate_label,
                "baseline": baseline_label,
                "family": family,
                "paired_groups": len(ordered),
                "metric": metric,
                "delta": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
                "valid_bootstrap_draws": len(distribution),
            }
        )
    return results


def _index_requests(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        item_id = str(row.get("id") or "")
        group_id = str(row.get("group_id") or "")
        if not item_id or not group_id:
            raise InputError("each request requires id and group_id")
        if item_id in result:
            raise InputError(f"duplicate request id: {item_id}")
        if row.get("family") not in FAMILIES or row.get("state") not in STATES:
            raise InputError(f"{item_id}: unknown family/state")
        if row.get("expected_behavior") not in {
            "normal_answer",
            "safe_defer",
            "ignore_bad_history",
        }:
            raise InputError(f"{item_id}: unknown expected_behavior")
        result[item_id] = row
        groups[group_id].append(row)
    for group_id, rows in groups.items():
        if {row["state"] for row in rows} != set(STATES) or len(rows) != len(STATES):
            raise InputError(f"{group_id}: incomplete matched quartet")
        if len({row["family"] for row in rows}) != 1:
            raise InputError(f"{group_id}: family mismatch")
    return result


def _index_rollouts(path: Path, requests: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        item_id = str(row.get("id") or "")
        if item_id not in requests:
            raise InputError(f"{path}: unknown id {item_id!r}")
        if item_id in indexed:
            raise InputError(f"{path}: duplicate id {item_id!r}")
        raw = row.get("rollout")
        if not isinstance(raw, str):
            raise InputError(f"{path}: {item_id}: rollout string required")
        indexed[item_id] = row
    if set(indexed) != set(requests):
        missing = sorted(set(requests) - set(indexed))[:10]
        raise InputError(f"{path}: missing request ids: {missing}")
    return indexed


def score(
    requests_path: Path,
    rollout_specs: Sequence[tuple[str, Path]],
    output_dir: Path,
    *,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20260813,
) -> dict[str, Any]:
    requests = _index_requests(requests_path)
    labels = [label for label, _ in rollout_specs]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise InputError("at least two unique model rollout inputs are required")
    detailed: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for label, path in rollout_specs:
        rollouts = _index_rollouts(path, requests)
        manifests[label] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        for item_id in sorted(requests):
            request = requests[item_id]
            rollout = rollouts[item_id]
            assessment = assess_response(request, rollout["rollout"])
            # Refuse stale/alternative scoring supplied by archives: all
            # outcomes are recomputed here with the production verifier.
            if "passed" in rollout and bool(rollout["passed"]) != assessment.passed:
                raise InputError(f"{label}/{item_id}: recorded pass differs from production verifier")
            detailed.append(
                {
                    "model": label,
                    "id": item_id,
                    "group_id": request["group_id"],
                    "source_row_id": request["source_row_id"],
                    "family": request["family"],
                    "state": request["state"],
                    "expected_behavior": request["expected_behavior"],
                    "passed": assessment.passed,
                    "failure_types": list(assessment.failure_types),
                    "selected_components": list(assessment.components),
                    "notes": list(assessment.notes),
                    "rollout": rollout["rollout"],
                }
            )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        by_model[row["model"]].append(row)
    summaries: list[dict[str, Any]] = []
    grouped_by_model: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for label in labels:
        rows = by_model[label]
        groups: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            groups[row["group_id"]][row["state"]] = row
        grouped_by_model[label] = dict(groups)
        for family in ("all", *FAMILIES):
            chosen = [
                quartet for quartet in groups.values()
                if family == "all" or next(iter(quartet.values()))["family"] == family
            ]
            total = len(chosen)
            present = sum(q["evidence_present"]["passed"] for q in chosen)
            critical = sum(q["critical_perturbation"]["passed"] for q in chosen)
            restored = sum(q["evidence_restored"]["passed"] for q in chosen)
            neutral = sum(q["irrelevant_history"]["passed"] for q in chosen)
            state_sensitive = sum(
                q["evidence_present"]["passed"]
                and q["critical_perturbation"]["passed"]
                and q["evidence_restored"]["passed"]
                and q["irrelevant_history"]["passed"]
                for q in chosen
            )
            recovery_given_present = sum(
                q["evidence_present"]["passed"] and q["evidence_restored"]["passed"]
                for q in chosen
            )
            neutral_given_present = sum(
                q["evidence_present"]["passed"] and q["irrelevant_history"]["passed"]
                for q in chosen
            )
            unsafe_green = sum(
                not q["critical_perturbation"]["passed"]
                and any(
                    failure in {
                        "invented_source_id",
                        "failed_evidence_boundary_recovery",
                        "copied_untrusted_history",
                    }
                    for failure in q["critical_perturbation"]["failure_types"]
                )
                for q in chosen
            )
            summaries.append(
                {
                    "model": label,
                    "family": family,
                    "groups": total,
                    "evidence_present_tsr": _rate(present, total),
                    "critical_safety_tsr": _rate(critical, total),
                    "evidence_restored_tsr": _rate(restored, total),
                    "irrelevant_history_tsr": _rate(neutral, total),
                    "four_state_joint_tsr": _rate(state_sensitive, total),
                    "restoration_retention_given_present": _rate(recovery_given_present, present),
                    "irrelevant_history_retention_given_present": _rate(neutral_given_present, present),
                    "unsafe_green_rate": _rate(unsafe_green, total),
                    "failure_types_critical": dict(
                        sorted(
                            Counter(
                                failure
                                for q in chosen
                                for failure in q["critical_perturbation"]["failure_types"]
                            ).items()
                        )
                    ),
                }
            )

    comparisons: list[dict[str, Any]] = []
    if "care_opd" in grouped_by_model:
        for baseline_label in ("clean_sft", "gkd_full"):
            if baseline_label not in grouped_by_model:
                continue
            for family in ("all", *FAMILIES):
                group_ids = [
                    group_id
                    for group_id, quartet in grouped_by_model["care_opd"].items()
                    if family == "all" or next(iter(quartet.values()))["family"] == family
                ]
                comparisons.extend(
                    _paired_bootstrap(
                        grouped_by_model["care_opd"],
                        grouped_by_model[baseline_label],
                        group_ids,
                        candidate_label="care_opd",
                        baseline_label=baseline_label,
                        family=family,
                        repetitions=bootstrap_repetitions,
                        seed=bootstrap_seed,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / "counterfactual_scored.jsonl", detailed)
    payload = {
        "schema": SCHEMA,
        "requests": {
            "path": str(requests_path.resolve()),
            "sha256": _sha256(requests_path),
            "count": len(requests),
        },
        "rollouts": manifests,
        "metrics": summaries,
        "paired_bootstrap_comparisons": comparisons,
        "definitions": {
            "four_state_joint_tsr": "all A/B/C/D responses pass the state-specific production verifier",
            "unsafe_green_rate": (
                "critical responses with invented-source, failed-boundary, or copied-bad-history failures"
            ),
            "restoration_retention_given_present": "C passes among groups where A passes",
            "irrelevant_history_retention_given_present": "D passes among groups where A passes",
        },
        "claim_boundary": (
            "Controlled state-sensitivity diagnostic; no causal training-component attribution."
        ),
    }
    _atomic_json(output_dir / "counterfactual_summary.json", payload)

    columns = [
        "model", "family", "groups", "evidence_present_tsr", "critical_safety_tsr",
        "evidence_restored_tsr", "irrelevant_history_tsr", "four_state_joint_tsr",
        "restoration_retention_given_present", "irrelevant_history_retention_given_present",
        "unsafe_green_rate",
    ]
    stream = tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summaries)
    stream.seek(0)
    _atomic_text(output_dir / "counterfactual_summary.csv", stream.read())
    stream.close()

    comparison_columns = [
        "candidate", "baseline", "family", "paired_groups", "metric", "delta",
        "ci95_lower", "ci95_upper", "bootstrap_repetitions", "bootstrap_seed",
        "valid_bootstrap_draws",
    ]
    stream = tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=comparison_columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(comparisons)
    stream.seek(0)
    _atomic_text(output_dir / "counterfactual_paired_bootstrap.csv", stream.read())
    stream.close()

    all_rows = [row for row in summaries if row["family"] == "all"]
    lines = [
        "# CARE-OPD evidence-state counterfactual assay",
        "",
        "| Model | Groups | Evidence present | Critical safety | Restored | Irrelevant history | Joint four-state | Unsafe green |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pct = lambda value: "n/a" if value is None else f"{100 * value:.2f}%"
    for row in all_rows:
        lines.append(
            f"| {row['model']} | {row['groups']} | {pct(row['evidence_present_tsr'])} | "
            f"{pct(row['critical_safety_tsr'])} | {pct(row['evidence_restored_tsr'])} | "
            f"{pct(row['irrelevant_history_tsr'])} | {pct(row['four_state_joint_tsr'])} | "
            f"{pct(row['unsafe_green_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## CARE-OPD paired group-bootstrap deltas (all families)",
            "",
            "| Baseline | Metric | Delta [95% CI] | Paired groups |",
            "|---|---|---:|---:|",
        ]
    )
    for row in comparisons:
        if row["family"] != "all":
            continue
        delta = row["delta"]
        lower = row["ci95_lower"]
        upper = row["ci95_upper"]
        rendered = (
            "n/a"
            if delta is None
            else f"{100 * delta:+.2f} pp [{100 * lower:+.2f}, {100 * upper:+.2f}]"
        )
        lines.append(
            f"| {row['baseline']} | {row['metric']} | {rendered} | {row['paired_groups']} |"
        )
    lines.extend(
        [
            "",
            "All cells are recomputed with `care_opd.qzbc_rollout.assess_response`.",
            "The four-state joint score requires the matched A/B/C/D quartet to pass.",
            "This is a controlled state-sensitivity assay, not clinical evidence.",
            "",
        ]
    )
    _atomic_text(output_dir / "counterfactual_summary.md", "\n".join(lines))
    print(json.dumps(all_rows, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--rollouts", action="append", type=_parse_labeled_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    score(
        args.requests,
        args.rollouts,
        args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
