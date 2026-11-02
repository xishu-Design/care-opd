#!/usr/bin/env python3
"""Summarize prespecified hard-oracle attribution contrasts from real rollouts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .contract import CONTRASTS, EXPECTED
except ImportError:
    from contract import CONTRASTS, EXPECTED

from mechanism_experiments.causal_ablation import summarize_arms


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def summarize(
    *,
    requests: Path,
    family_clean_ids: Path,
    rollouts: Sequence[tuple[str, Path]],
    output_dir: Path,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    labels = [label for label, _path in rollouts]
    if labels != list(EXPECTED):
        raise ValueError(
            "--rollout order/labels must match the frozen matrix exactly: "
            + ", ".join(EXPECTED)
        )
    if bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")

    # Reuse the existing production-verifier family-clean summarizer rather
    # than introducing a second metric implementation.
    base = summarize_arms.summarize(
        requests,
        family_clean_ids,
        rollouts,
        output_dir,
        reference="care_without_oracle_ce",
    )
    request_rows = summarize_arms._read_rows(requests)
    request_map = summarize_arms._index(request_rows, source=requests)
    ids = summarize_arms._family_ids(family_clean_ids)
    subset = {row_id: request_map[row_id] for row_id in ids}
    pairs: dict[str, list[dict[str, Any]]] = {}
    outcomes: dict[str, dict[str, bool]] = {}
    for label in labels:
        derived = output_dir / "family_clean_views" / f"{label}.jsonl"
        rows = summarize_arms._read_rows(derived)
        by_id = summarize_arms._index(rows, source=derived)
        pairs[label], _payload = summarize_arms.shared._model_pairs(
            subset, derived, clean_variant="clean"
        )
        outcomes[label] = {row_id: bool(by_id[row_id].get("passed")) for row_id in ids}

    contrast_results: dict[str, Any] = {}
    for contrast in CONTRASTS:
        name = str(contrast["name"])
        left = str(contrast["left"])
        right = str(contrast["right"])
        metric_bootstrap = summarize_arms.shared._bootstrap_delta(
            pairs[left],
            pairs[right],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        request_bootstrap = summarize_arms._reference_bootstrap(
            arm=outcomes[left],
            reference=outcomes[right],
            request_map=request_map,
            ids=ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        left_only = sum(outcomes[left][row_id] and not outcomes[right][row_id] for row_id in ids)
        right_only = sum(outcomes[right][row_id] and not outcomes[left][row_id] for row_id in ids)
        contrast_results[name] = {
            **contrast,
            "left_only_successes": left_only,
            "right_only_successes": right_only,
            "all_request_success_delta": (left_only - right_only) / len(ids),
            "family_clean_metric_bootstrap": metric_bootstrap,
            "all_request_cluster_bootstrap": request_bootstrap,
        }

    result = {
        "schema_version": "care_opd.hard_oracle_attribution_summary.v1",
        "status": "COMPLETE",
        "base_summary_schema": base["schema_version"],
        "arm_order": labels,
        "bootstrap": {
            "unit": "seed_id cluster",
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "confidence_level": 0.95,
        },
        "contrasts": contrast_results,
        "claim_boundary": (
            "Contrasts are paired within one on-policy bank and one training seed. "
            "They isolate configured one-round interventions but do not estimate "
            "training-seed variance or cross-domain effects."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summarize_arms._atomic_text(
        output_dir / "attribution_summary.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    columns = [
        "contrast",
        "left",
        "right",
        "left_perturbed_tsr",
        "right_perturbed_tsr",
        "perturbed_delta",
        "perturbed_ci95_lower",
        "perturbed_ci95_upper",
        "left_retention",
        "right_retention",
        "retention_delta",
        "retention_ci95_lower",
        "retention_ci95_upper",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    table_rows: list[dict[str, Any]] = []
    for name, contrast in contrast_results.items():
        metrics = contrast["family_clean_metric_bootstrap"]["metrics"]
        perturbed = metrics["perturbed_task_success_rate"]
        retention = metrics["conditional_retention"]
        row = {
            "contrast": name,
            "left": contrast["left"],
            "right": contrast["right"],
            "left_perturbed_tsr": perturbed["care_opd"],
            "right_perturbed_tsr": perturbed["baseline"],
            "perturbed_delta": perturbed["delta"],
            "perturbed_ci95_lower": perturbed["ci95_lower"],
            "perturbed_ci95_upper": perturbed["ci95_upper"],
            "left_retention": retention["care_opd"],
            "right_retention": retention["baseline"],
            "retention_delta": retention["delta"],
            "retention_ci95_lower": retention["ci95_lower"],
            "retention_ci95_upper": retention["ci95_upper"],
        }
        table_rows.append(row)
        writer.writerow(row)
    summarize_arms._atomic_text(output_dir / "attribution_summary.csv", stream.getvalue())

    md = [
        "# CARE-OPD hard-oracle attribution",
        "",
        "All deltas are left minus right on the same family-clean QZBC units. "
        "Intervals use the prespecified paired seed-ID cluster bootstrap.",
        "",
        "| Contrast | Perturbed TSR (L / R) | Delta [95% CI] | Retention (L / R) | Delta [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        md.append(
            f"| {row['contrast']} | {_pct(row['left_perturbed_tsr'])} / {_pct(row['right_perturbed_tsr'])} "
            f"| {_pct(row['perturbed_delta'])} [{_pct(row['perturbed_ci95_lower'])}, {_pct(row['perturbed_ci95_upper'])}] "
            f"| {_pct(row['left_retention'])} / {_pct(row['right_retention'])} "
            f"| {_pct(row['retention_delta'])} [{_pct(row['retention_ci95_lower'])}, {_pct(row['retention_ci95_upper'])}] |"
        )
    md += ["", result["claim_boundary"], ""]
    summarize_arms._atomic_text(output_dir / "attribution_summary.md", "\n".join(md))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--family-clean-ids", type=Path, required=True)
    parser.add_argument(
        "--rollout", action="append", type=summarize_arms._label_path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summarize(
        requests=args.requests,
        family_clean_ids=args.family_clean_ids,
        rollouts=args.rollout,
        output_dir=args.output_dir,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(f"summary={args.output_dir.resolve() / 'attribution_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

