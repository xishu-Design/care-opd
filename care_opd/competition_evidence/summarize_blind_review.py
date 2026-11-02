#!/usr/bin/env python3
"""Validate and summarize completed method-blind expert ratings.

The public outputs are aggregate-only.  The script never copies review-unit,
case, family, source, or response identifiers into a public artifact.  Binary
reviewer disagreements are written to a private adjudication queue and remain
unresolved until an adjudicator file is supplied; no score is fabricated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SCHEMA_VERSION = "care_opd.competition.blind_review_summary.v1"
FIELD_SPECS: dict[str, tuple[int, int, str]] = {
    "factual_fidelity_0_4": (0, 4, "ordinal_high"),
    "critical_omission_0_1": (0, 1, "binary_low"),
    "unsupported_major_claim_0_1": (0, 1, "binary_low"),
    "safe_correct_0_1": (0, 1, "binary_high"),
    "harm_severity_0_3": (0, 3, "ordinal_low"),
    "role_fit_0_4": (0, 4, "ordinal_high"),
    "evidence_handling_0_4": (0, 4, "ordinal_high"),
    "review_burden_0_4": (0, 4, "ordinal_low"),
    "overall_quality_1_5": (1, 5, "ordinal_high"),
}
BINARY_FIELDS = tuple(
    field for field, (_, _, kind) in FIELD_SPECS.items() if kind.startswith("binary")
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_score(raw: str, field: str) -> int:
    if raw is None or not str(raw).strip():
        raise ValueError(f"blank required score: {field}")
    text = str(raw).strip()
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"score {field} must be an integer, got {text!r}") from exc
    low, high, _ = FIELD_SPECS[field]
    if not low <= value <= high:
        raise ValueError(f"score {field}={value} outside [{low}, {high}]")
    return value


def _weighted_kappa(
    left: list[int], right: list[int], low: int, high: int, quadratic: bool
) -> float | None:
    if len(left) != len(right) or not left:
        return None
    categories = list(range(low, high + 1))
    size = len(categories)
    index = {value: idx for idx, value in enumerate(categories)}
    observed = [[0.0] * size for _ in range(size)]
    left_counts = [0.0] * size
    right_counts = [0.0] * size
    for a, b in zip(left, right):
        i, j = index[a], index[b]
        observed[i][j] += 1.0
        left_counts[i] += 1.0
        right_counts[j] += 1.0
    n = float(len(left))
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    denominator = max(1, size - 1)
    for i in range(size):
        for j in range(size):
            if quadratic:
                weight = ((i - j) / denominator) ** 2
            else:
                weight = 0.0 if i == j else 1.0
            observed_disagreement += weight * observed[i][j] / n
            expected_disagreement += weight * (left_counts[i] * right_counts[j]) / (n * n)
    if math.isclose(expected_disagreement, 0.0):
        return 1.0 if math.isclose(observed_disagreement, 0.0) else None
    return 1.0 - observed_disagreement / expected_disagreement


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap_difference(
    paired_rows: list[dict[str, Any]], replicates: int, seed: int
) -> tuple[float, float, float]:
    if not paired_rows:
        raise ValueError("no paired rows for bootstrap")
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in paired_rows:
        by_group[row["group_private"]].append(float(row["difference"]))
    groups = sorted(by_group)
    point = mean(row["difference"] for row in paired_rows)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        values = [value for group in selected for value in by_group[group]]
        samples.append(mean(values))
    return point, _percentile(samples, 0.025), _percentile(samples, 0.975)


def _mcnemar_exact(primary: list[int], comparator: list[int]) -> dict[str, Any]:
    if len(primary) != len(comparator):
        raise ValueError("McNemar inputs must be paired")
    primary_only = sum(a == 1 and b == 0 for a, b in zip(primary, comparator))
    comparator_only = sum(a == 0 and b == 1 for a, b in zip(primary, comparator))
    discordant = primary_only + comparator_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) * (0.5**discordant)
            for k in range(min(primary_only, comparator_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "primary_success_comparator_failure": primary_only,
        "primary_failure_comparator_success": comparator_only,
        "discordant_pair_count": discordant,
        "two_sided_exact_p": p_value,
    }


def _parse_adjudication(path: Path | None) -> dict[tuple[str, str, str], int]:
    if path is None:
        return {}
    result: dict[tuple[str, str, str], int] = {}
    for row in _read_csv(path):
        key = (row.get("review_unit_id", ""), row.get("candidate_code", ""), row.get("field", ""))
        if key[2] not in BINARY_FIELDS:
            raise ValueError(f"invalid adjudication field {key[2]!r}")
        if key in result:
            raise ValueError(f"duplicate adjudication row for {key}")
        result[key] = _parse_score(row.get("final_score", ""), key[2])
    return result


def summarize(args: argparse.Namespace) -> dict[str, Path]:
    manifest = _read_json(args.manifest.expanduser().resolve())
    mapping_rows = _read_jsonl(args.mapping.expanduser().resolve())
    score_rows = _read_csv(args.scores.expanduser().resolve())
    reviewers = list(manifest.get("reviewers_private", []))
    if len(reviewers) != 2:
        raise ValueError("private manifest must lock exactly two reviewers")

    map_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in mapping_rows:
        key = (str(row["review_unit_id"]), str(row["candidate_code"]))
        if key in map_index:
            raise ValueError(f"duplicate mapping key {key}")
        map_index[key] = row
    expected = {
        (reviewer, unit, candidate)
        for unit, candidate in map_index
        for reviewer in reviewers
    }
    observed: set[tuple[str, str, str]] = set()
    parsed: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(dict)
    for row_number, row in enumerate(score_rows, start=2):
        key = (
            row.get("reviewer_id", ""),
            row.get("review_unit_id", ""),
            row.get("candidate_code", ""),
        )
        if key in observed:
            raise ValueError(f"duplicate score row at CSV line {row_number}: {key}")
        if key not in expected:
            raise ValueError(f"unexpected score row at CSV line {row_number}: {key}")
        observed.add(key)
        parsed[(key[1], key[2])][key[0]] = {
            field: _parse_score(row.get(field, ""), field) for field in FIELD_SPECS
        }
    if observed != expected:
        raise ValueError(
            f"incomplete score table: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )

    agreement_rows: list[dict[str, Any]] = []
    for field, (low, high, kind) in FIELD_SPECS.items():
        left = [parsed[key][reviewers[0]][field] for key in sorted(parsed)]
        right = [parsed[key][reviewers[1]][field] for key in sorted(parsed)]
        exact = sum(a == b for a, b in zip(left, right)) / len(left)
        kappa = _weighted_kappa(
            left, right, low, high, quadratic=not kind.startswith("binary")
        )
        agreement_rows.append(
            {
                "field": field,
                "rating_pair_count": len(left),
                "exact_agreement": exact,
                "kappa_type": (
                    "quadratic_weighted_cohen" if not kind.startswith("binary") else "cohen"
                ),
                "kappa": kappa,
            }
        )

    adjudicated = _parse_adjudication(args.adjudication)
    disagreement_rows: list[dict[str, Any]] = []
    resolved_binary: dict[tuple[str, str, str], int | None] = {}
    for unit_candidate in sorted(parsed):
        for field in BINARY_FIELDS:
            values = [parsed[unit_candidate][reviewer][field] for reviewer in reviewers]
            key = (unit_candidate[0], unit_candidate[1], field)
            if values[0] == values[1]:
                resolved_binary[key] = values[0]
            elif key in adjudicated:
                resolved_binary[key] = adjudicated[key]
            else:
                resolved_binary[key] = None
                disagreement_rows.append(
                    {
                        "review_unit_id": unit_candidate[0],
                        "candidate_code": unit_candidate[1],
                        "field": field,
                        f"{reviewers[0]}_score": values[0],
                        f"{reviewers[1]}_score": values[1],
                        "final_score": "",
                        "adjudicator_id": "",
                        "adjudication_notes_private": "",
                    }
                )
    unused_adjudications = sorted(set(adjudicated) - set(resolved_binary))
    if unused_adjudications:
        raise ValueError(f"adjudication file contains unknown rows: {unused_adjudications[:3]}")

    # Candidate-level consensus: binary fields require agreement/adjudication;
    # ordinal fields use the prespecified mean of the two blinded reviewers.
    candidate_values: dict[tuple[str, str], dict[str, float | int | None]] = {}
    for key, by_reviewer in parsed.items():
        values: dict[str, float | int | None] = {}
        for field, (_, _, kind) in FIELD_SPECS.items():
            if kind.startswith("binary"):
                values[field] = resolved_binary[(key[0], key[1], field)]
            else:
                values[field] = mean(by_reviewer[reviewer][field] for reviewer in reviewers)
        candidate_values[key] = values

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_unit_model: dict[tuple[str, str], dict[str, Any]] = {}
    for key, values in candidate_values.items():
        mapping = map_index[key]
        model = str(mapping["model_label_private"])
        record = {
            **values,
            "unit_private": key[0],
            "group_private": str(mapping["report_group_hmac_private"]),
            "variant": str(mapping["variant"]),
        }
        by_model[model].append(record)
        by_unit_model[(key[0], model)] = record
    if args.primary_model not in by_model:
        raise ValueError(f"primary model {args.primary_model!r} absent from mapping")
    comparators = args.comparator or sorted(set(by_model) - {args.primary_model})
    absent = sorted(set(comparators) - set(by_model))
    if absent:
        raise ValueError(f"comparators absent from mapping: {absent}")

    aggregate_rows: list[dict[str, Any]] = []
    for model in sorted(by_model):
        rows = by_model[model]
        safe = [row["safe_correct_0_1"] for row in rows if row["safe_correct_0_1"] is not None]
        aggregate_rows.append(
            {
                "model": model,
                "review_unit_count": len(rows),
                "resolved_primary_endpoint_count": len(safe),
                "unresolved_primary_endpoint_count": len(rows) - len(safe),
                "safe_correct_rate": mean(safe) if safe else None,
                "mean_overall_quality": mean(row["overall_quality_1_5"] for row in rows),
                "mean_factual_fidelity": mean(row["factual_fidelity_0_4"] for row in rows),
                "mean_harm_severity": mean(row["harm_severity_0_3"] for row in rows),
                "mean_review_burden": mean(row["review_burden_0_4"] for row in rows),
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    for comparator in comparators:
        shared_units = sorted(
            {unit for unit, model in by_unit_model if model == args.primary_model}
            & {unit for unit, model in by_unit_model if model == comparator}
        )
        wins = ties = losses = 0
        paired_binary: list[dict[str, Any]] = []
        primary_binary: list[int] = []
        comparator_binary: list[int] = []
        for unit in shared_units:
            primary = by_unit_model[(unit, args.primary_model)]
            other = by_unit_model[(unit, comparator)]
            primary_quality = float(primary["overall_quality_1_5"])
            other_quality = float(other["overall_quality_1_5"])
            if primary_quality > other_quality:
                wins += 1
            elif primary_quality < other_quality:
                losses += 1
            else:
                ties += 1
            a = primary["safe_correct_0_1"]
            b = other["safe_correct_0_1"]
            if a is not None and b is not None:
                primary_binary.append(int(a))
                comparator_binary.append(int(b))
                paired_binary.append(
                    {
                        "group_private": primary["group_private"],
                        "difference": int(a) - int(b),
                    }
                )
        if paired_binary:
            diff, ci_low, ci_high = _cluster_bootstrap_difference(
                paired_binary, args.bootstrap_replicates, args.bootstrap_seed
            )
            mcnemar = _mcnemar_exact(primary_binary, comparator_binary)
        else:
            diff = ci_low = ci_high = None
            mcnemar = {
                "primary_success_comparator_failure": None,
                "primary_failure_comparator_success": None,
                "discordant_pair_count": None,
                "two_sided_exact_p": None,
            }
        unresolved_pairs = len(shared_units) - len(paired_binary)
        pairwise_rows.append(
            {
                "primary_model": args.primary_model,
                "comparator_model": comparator,
                "shared_review_unit_count": len(shared_units),
                "overall_quality_wins": wins,
                "overall_quality_ties": ties,
                "overall_quality_losses": losses,
                "resolved_binary_pair_count": len(paired_binary),
                "unresolved_binary_pair_count": unresolved_pairs,
                "safe_correct_rate_difference": diff,
                "cluster_bootstrap_ci95_low": ci_low,
                "cluster_bootstrap_ci95_high": ci_high,
                **mcnemar,
                "analysis_status": (
                    "final" if unresolved_pairs == 0 else "provisional_pending_adjudication"
                ),
            }
        )

    private_dir = args.output_private.expanduser().resolve()
    public_dir = args.output_public.expanduser().resolve()
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    if private_dir == public_dir:
        raise ValueError("private and public output directories must differ")
    adjudication_path = private_dir / "adjudication_queue.private.csv"
    adjudication_fields = [
        "review_unit_id",
        "candidate_code",
        "field",
        f"{reviewers[0]}_score",
        f"{reviewers[1]}_score",
        "final_score",
        "adjudicator_id",
        "adjudication_notes_private",
    ]
    _write_csv(adjudication_path, adjudication_fields, disagreement_rows)

    aggregate_path = public_dir / "blind_review_model_aggregate.csv"
    pairwise_path = public_dir / "blind_review_pairwise.csv"
    agreement_path = public_dir / "blind_review_agreement.csv"
    summary_path = public_dir / "blind_review_summary.json"
    _write_csv(aggregate_path, list(aggregate_rows[0]), aggregate_rows)
    _write_csv(pairwise_path, list(pairwise_rows[0]), pairwise_rows)
    _write_csv(agreement_path, list(agreement_rows[0]), agreement_rows)
    status = "FINAL" if not disagreement_rows else "ADJUDICATION_REQUIRED"
    public_summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "data_scope": {
            "origin": "real-world medical records provided by Benyuan Zhishu",
            "processing": "de-identified and structured",
            "evaluation_track": manifest.get("evaluation_track"),
            "variants": manifest.get("variants"),
            "report_group_count": manifest.get("sampling", {}).get("selected_family_count"),
            "review_unit_count": manifest.get("unit_count"),
            "method_count": len(by_model),
            "reviewer_count": len(reviewers),
        },
        "analysis": {
            "primary_binary_endpoint": "safe_correct_0_1",
            "win_tie_loss_endpoint": "mean reviewer overall_quality_1_5",
            "bootstrap": {
                "cluster": "report group",
                "replicates": args.bootstrap_replicates,
                "seed": args.bootstrap_seed,
            },
            "binary_test": "two-sided exact McNemar",
            "ordinal_agreement": "quadratic-weighted Cohen kappa",
            "unresolved_binary_disagreement_count": len(disagreement_rows),
            "warning": (
                None
                if status == "FINAL"
                else (
                    "Primary binary comparisons are provisional until private "
                    "adjudication is complete."
                )
            ),
        },
        "model_aggregates": aggregate_rows,
        "pairwise_comparisons": pairwise_rows,
        "inter_rater_agreement": agreement_rows,
        "privacy_statement": (
            "Aggregate output only: no blind-code mapping, row-level identifier, clinical text, "
            "model response, reviewer identity, or free-text note is included."
        ),
    }
    _write_json(summary_path, public_summary)
    return {
        "adjudication_queue_private": adjudication_path,
        "model_aggregate_public": aggregate_path,
        "pairwise_public": pairwise_path,
        "agreement_public": agreement_path,
        "summary_public": summary_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--primary-model", required=True)
    parser.add_argument("--comparator", action="append")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--output-private", required=True, type=Path)
    parser.add_argument("--output-public", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    outputs = summarize(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
