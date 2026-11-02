"""Aggregate archived rollout logs into auditable formal evaluation reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import (
    aggregate,
    group_bootstrap,
    paired_permutation_test,
    paired_risk_adjusted_recovery,
)


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault("_source", str(path))
                row.setdefault("_line", line_number)
                if not row.get("method"):
                    raise ValueError(f"{path}:{line_number}: missing method")
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_baseline(
    by_method_split: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    requested: str | None,
) -> tuple[str | None, dict[str, Any], tuple[str, ...]]:
    """Resolve an explicit baseline or select one without test-set tuning."""

    if requested != "auto":
        available_splits = tuple(
            sorted({split for _, split in by_method_split})
        )
        return (
            requested,
            {
                "requested": requested,
                "method": requested,
                "selection_split": None,
                "criterion": "explicit" if requested is not None else "none",
                "candidate_prar": {},
                "comparison_split": "all",
                "exploratory": False,
            },
            available_splits,
        )

    methods = sorted(
        {
            method
            for method, _ in by_method_split
            if method != "care_opd"
        }
    )
    if not methods:
        raise ValueError(
            "baseline=auto requires at least one non-care_opd method"
        )
    dev_available = any(
        by_method_split.get((method, "dev")) for method in methods
    )
    selection_split = "dev" if dev_available else "test"
    exploratory = selection_split != "dev"
    scores: dict[str, float] = {}
    for method in methods:
        subset = by_method_split.get((method, selection_split), ())
        if not subset:
            continue
        score = paired_risk_adjusted_recovery(subset)
        if math.isfinite(score):
            scores[method] = score
    if not scores:
        raise ValueError(
            "baseline=auto found no finite non-care_opd PRAR on "
            f"{selection_split}"
        )
    # The secondary method key makes exact-score ties deterministic.
    selected = min(scores, key=lambda method: (-scores[method], method))
    if selection_split == "dev" and not by_method_split.get(
        (selected, "test")
    ):
        raise ValueError(
            f"auto-selected baseline {selected!r} has no test rows"
        )
    comparison_splits = ("test",)
    return (
        selected,
        {
            "requested": "auto",
            "method": selected,
            "selection_split": selection_split,
            "criterion": "maximum paired risk-adjusted recovery",
            "candidate_prar": scores,
            "comparison_split": "test",
            "exploratory": exploratory,
        },
        comparison_splits,
    )


def build_report(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    baseline: str | None,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    by_method_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method_split[(str(row["method"]), str(row.get("split", "all")))].append(row)

    table: list[dict[str, Any]] = []
    intervals: dict[str, Any] = {}
    for (method, split), subset in sorted(by_method_split.items()):
        summary = aggregate(subset)
        ci = group_bootstrap(
            subset,
            paired_risk_adjusted_recovery,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )
        record = {"method": method, "split": split, **summary}
        record.update(
            {
                "prar_ci_low": ci.low,
                "prar_ci_high": ci.high,
                "prar_n_groups": ci.n_groups,
            }
        )
        table.append(record)
        intervals[f"{method}/{split}"] = ci.as_dict()

    resolved_baseline, selection, comparison_splits = _resolve_baseline(
        by_method_split,
        baseline,
    )
    tests: dict[str, Any] = {}
    if resolved_baseline is not None:
        methods = sorted(
            {
                method
                for method, _ in by_method_split
                if method != resolved_baseline
            }
        )
        for split in comparison_splits:
            base_rows = by_method_split.get((resolved_baseline, split), [])
            for method in methods:
                candidate_rows = by_method_split.get((method, split), [])
                if base_rows and candidate_rows:
                    tests[
                        f"{method}_vs_{resolved_baseline}/{split}"
                    ] = paired_permutation_test(
                        candidate_rows,
                        base_rows,
                        samples=max(bootstrap_samples * 10, 10_000),
                        seed=seed,
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "main_table.csv", table)
    report = {
        "schema_version": "care-opd-eval-v1",
        "rows": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "baseline_selection": selection,
        "intervals": intervals,
        "paired_tests": tests,
    }
    (output_dir / "statistical_tests.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    report = build_report(
        rows,
        output_dir=args.output_dir,
        baseline=args.baseline,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
