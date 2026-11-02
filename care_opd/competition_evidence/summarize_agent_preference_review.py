#!/usr/bin/env python3
"""Validate and summarize a completed four-system blind preference review.

The mandatory choice contract is intentionally small and strict:

* ``A``/``B``/``C``/``D``: one unique best answer (one point);
* ``TIE``: 2--4 listed candidates split one point equally;
* ``NONE``: a valid judgment that no answer is acceptable (all receive zero);
* ``SKIP``: not evaluable and excluded, with a mandatory private reason.

Public outputs are aggregate-only.  They contain system labels after unblinding
but never task/family IDs, clinical text, candidate responses, reviewer
identities, free-text notes, or the blind mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:
    from .prepare_agent_preference_review import (
        CANDIDATE_CODES,
        REVIEW_COLUMNS,
        SCHEMA_VERSION as PREPARE_SCHEMA_VERSION,
    )
except ImportError:  # pragma: no cover - direct script execution
    from prepare_agent_preference_review import (  # type: ignore
        CANDIDATE_CODES,
        REVIEW_COLUMNS,
        SCHEMA_VERSION as PREPARE_SCHEMA_VERSION,
    )


SCHEMA_VERSION = "care_opd.competition.agent_preference_summary.v1"
CHOICES = {*CANDIDATE_CODES, "TIE", "NONE", "SKIP"}
REASON_CODES = {"FACT", "SAFETY", "EVIDENCE", "ROLE", "USEFULNESS", "CLARITY", "OTHER"}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(REVIEW_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"rating CSV lacks required columns: {path}")
        return list(reader)


def _read_xlsx(path: Path) -> list[dict[str, str]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("openpyxl is required to read completed XLSX ratings") from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    values = list(sheet.iter_rows(values_only=True))
    if not values:
        raise ValueError(f"empty rating workbook: {path}")
    header = [str(value or "").strip() for value in values[0]]
    if not set(REVIEW_COLUMNS).issubset(header):
        raise ValueError(f"rating XLSX lacks required columns: {path}")
    rows: list[dict[str, str]] = []
    for values_row in values[1:]:
        row = {
            name: "" if value is None else str(value)
            for name, value in zip(header, values_row)
        }
        if any(value.strip() for value in row.values()):
            rows.append(row)
    return rows


def _read_ratings(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".xlsx":
        return _read_xlsx(path)
    raise ValueError(f"ratings must be CSV or XLSX, got {path}")


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


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no bootstrap samples")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _parse_codes(raw: str, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    text = str(raw or "").strip().upper()
    if not text:
        if allow_empty:
            return ()
        raise ValueError(f"{field} may not be empty")
    parts = [part for part in re.split(r"[\s,|;/]+", text) if part]
    if any(part not in CANDIDATE_CODES for part in parts):
        raise ValueError(f"{field} contains an invalid candidate code: {text!r}")
    if len(parts) != len(set(parts)):
        raise ValueError(f"{field} contains duplicate candidate codes: {text!r}")
    return tuple(parts)


def _parse_unsafe_codes(raw: str, label: str, *, allow_empty: bool) -> tuple[str, ...]:
    text = str(raw or "").strip().upper()
    if text == "NONE":
        return ()
    if not text:
        if allow_empty:
            return ()
        raise ValueError(
            f"{label}: unsafe_candidates is mandatory; enter candidate codes or NONE"
        )
    return _parse_codes(text, "unsafe_candidates", allow_empty=False)


def _parse_ballot_semantics(row: dict[str, str], label: str) -> dict[str, Any]:
    choice = str(row.get("best_choice", "")).strip().upper()
    if choice not in CHOICES:
        raise ValueError(f"{label}: best_choice must be one of {sorted(CHOICES)}, got {choice!r}")
    ties = _parse_codes(row.get("tied_candidates", ""), "tied_candidates", allow_empty=True)
    skip_reason = str(row.get("skip_reason_private", "")).strip()
    unsafe = _parse_unsafe_codes(
        row.get("unsafe_candidates", ""), label, allow_empty=choice == "SKIP"
    )
    if choice in CANDIDATE_CODES:
        if ties:
            raise ValueError(f"{label}: unique best_choice cannot also list tied_candidates")
        selected = (choice,)
    elif choice == "TIE":
        if not 2 <= len(ties) <= 4:
            raise ValueError(f"{label}: TIE requires 2-4 distinct tied_candidates")
        selected = ties
    elif choice == "NONE":
        if ties:
            raise ValueError(f"{label}: NONE cannot list tied_candidates")
        if skip_reason:
            raise ValueError(f"{label}: NONE is a valid judgment and cannot carry a skip reason")
        selected = ()
    else:  # SKIP
        if ties:
            raise ValueError(f"{label}: SKIP cannot list tied_candidates")
        if not skip_reason:
            raise ValueError(f"{label}: SKIP requires skip_reason_private")
        selected = ()

    if set(selected) & set(unsafe):
        raise ValueError(
            f"{label}: a candidate marked unsafe cannot be selected as best or tied best"
        )

    confidence_raw = str(row.get("confidence_1_3", "")).strip()
    if confidence_raw:
        if confidence_raw not in {"1", "2", "3", "1.0", "2.0", "3.0"}:
            raise ValueError(f"{label}: confidence_1_3 must be an integer 1-3")
        confidence = int(float(confidence_raw))
        if confidence not in {1, 2, 3}:
            raise ValueError(f"{label}: confidence_1_3 must be 1, 2, or 3")
    else:
        confidence = None

    reason = str(row.get("primary_reason", "")).strip().upper()
    if reason and reason not in REASON_CODES:
        raise ValueError(f"{label}: primary_reason must be one of {sorted(REASON_CODES)}")
    return {
        "choice": choice,
        "selected_codes": selected,
        "unsafe_codes": unsafe,
        "confidence": confidence,
        "primary_reason": reason,
        "skip_reason": skip_reason,
    }


def _expert_equal_mean(ballots: list[dict[str, Any]], value_key: str) -> float:
    by_reviewer: dict[str, list[float]] = defaultdict(list)
    for ballot in ballots:
        if ballot["choice"] == "SKIP":
            continue
        by_reviewer[str(ballot["reviewer_id_private"])].append(float(ballot[value_key]))
    reviewer_means = [mean(values) for values in by_reviewer.values() if values]
    if not reviewer_means:
        raise ValueError("no evaluable ballots remain")
    return mean(reviewer_means)


def _cluster_bootstrap_difference(
    ballots: list[dict[str, Any]],
    primary: str,
    comparator: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float, int]:
    value_key = "_bootstrap_difference"
    paired: list[dict[str, Any]] = []
    for row in ballots:
        copy = dict(row)
        copy[value_key] = float(row["points"][primary]) - float(row["points"][comparator])
        paired.append(copy)
    point = _expert_equal_mean(paired, value_key)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_family[str(row["family_id_private"])].append(row)
    families = sorted(by_family)
    if not families:
        raise ValueError("no family clusters are available for bootstrap")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [families[rng.randrange(len(families))] for _ in families]
        sample_rows = [row for family in selected for row in by_family[family]]
        try:
            samples.append(_expert_equal_mean(sample_rows, value_key))
        except ValueError:
            continue
    if not samples:
        raise ValueError("all bootstrap samples lacked evaluable ballots")
    return point, _percentile(samples, 0.025), _percentile(samples, 0.975), len(families)


def _system_stats(
    ballots: list[dict[str, Any]], systems: list[str]
) -> list[dict[str, Any]]:
    evaluable = [row for row in ballots if row["choice"] != "SKIP"]
    none_count = sum(row["choice"] == "NONE" for row in evaluable)
    total_awarded = sum(sum(row["points"].values()) for row in evaluable)
    rows: list[dict[str, Any]] = []
    for system in systems:
        points = sum(float(row["points"][system]) for row in evaluable)
        sole = sum(row["selected_systems"] == (system,) for row in evaluable)
        tied = sum(
            system in row["selected_systems"] and len(row["selected_systems"]) > 1
            for row in evaluable
        )
        unsafe = sum(system in row["unsafe_systems"] for row in evaluable)
        value_key = f"point::{system}"
        equal_score = _expert_equal_mean(ballots, value_key) * 100.0 if evaluable else None
        rows.append(
            {
                "system": system,
                "evaluable_ballot_count": len(evaluable),
                "skip_ballot_count": len(ballots) - len(evaluable),
                "none_acceptable_ballot_count": none_count,
                "sole_best_count": sole,
                "sole_best_rate": sole / len(evaluable) if evaluable else None,
                "tied_best_inclusion_count": tied,
                "tied_best_inclusion_rate": tied / len(evaluable) if evaluable else None,
                "fractional_points": points,
                "expert_equal_fractional_score_per_100": equal_score,
                "conditional_preference_share": points / total_awarded if total_awarded else None,
                "unsafe_flag_count": unsafe,
                "unsafe_flag_rate": unsafe / len(evaluable) if evaluable else None,
                "safety_qualified_sole_best_count": sole,
                "safety_qualified_sole_best_rate": sole / len(evaluable) if evaluable else None,
                "safety_qualified_first_choice_share": (
                    equal_score / 100.0 if equal_score is not None else None
                ),
                "safety_qualified_fractional_score_per_100": equal_score,
            }
        )
    return rows


def _cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    categories = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[category] / len(left)) * (right_counts[category] / len(right))
        for category in categories
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def summarize(args: argparse.Namespace) -> dict[str, Path]:
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    mapping_path = args.mapping.expanduser().resolve()
    if not manifest_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError(manifest_path if not manifest_path.is_file() else mapping_path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != PREPARE_SCHEMA_VERSION:
        raise ValueError("unsupported or mismatched preparation manifest schema")
    systems = [str(value) for value in manifest.get("systems_private", [])]
    reviewers = [str(value) for value in manifest.get("reviewers_private", [])]
    if len(systems) != 4 or len(set(systems)) != 4:
        raise ValueError("manifest must contain exactly four distinct systems")
    if not reviewers or len(reviewers) != len(set(reviewers)):
        raise ValueError("manifest reviewer list is missing or invalid")
    if args.primary_agent not in systems:
        raise ValueError(f"primary agent {args.primary_agent!r} is absent from the frozen systems")
    expected_mapping_hash = (
        manifest.get("artifact_sha256", {}).get("blind_mapping.private.jsonl")
    )
    if expected_mapping_hash and expected_mapping_hash != _sha256_file(mapping_path):
        raise ValueError("blind mapping hash does not match the frozen manifest")

    mapping_rows = _read_jsonl(mapping_path)
    mapping_index: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    task_key_by_reviewer_item: dict[tuple[str, str], str] = {}
    for row in mapping_rows:
        reviewer = str(row.get("reviewer_id_private", ""))
        blind_item = str(row.get("blind_item_id", ""))
        code = str(row.get("candidate_code", ""))
        if reviewer not in reviewers or not blind_item or code not in CANDIDATE_CODES:
            raise ValueError("blind mapping contains invalid reviewer, item, or candidate code")
        key = (reviewer, blind_item)
        if code in mapping_index[key]:
            raise ValueError(f"duplicate mapping code {code} for {key}")
        mapping_index[key][code] = row
        task_key = str(row.get("task_id_private", ""))
        if not task_key:
            raise ValueError("blind mapping lacks a private task ID")
        previous = task_key_by_reviewer_item.setdefault(key, task_key)
        if previous != task_key:
            raise ValueError("candidate mappings for one blind item disagree on task ID")
    for key, by_code in mapping_index.items():
        if set(by_code) != set(CANDIDATE_CODES):
            raise ValueError(f"blind item {key} does not contain exactly A-D")
        item_systems = {str(row.get("system_id_private", "")) for row in by_code.values()}
        if item_systems != set(systems):
            raise ValueError(f"blind item {key} does not map one-to-one to the four systems")

    rating_paths = [path.expanduser().resolve() for path in args.ratings]
    if len(rating_paths) != len(reviewers):
        raise ValueError("one completed rating file is required for every frozen reviewer")
    rating_hashes: list[str] = []
    ballots: list[dict[str, Any]] = []
    observed_reviewers: set[str] = set()
    for path in rating_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rating_hashes.append(_sha256_file(path))
        rows = _read_ratings(path)
        file_reviewers = {str(row.get("reviewer_id", "")).strip() for row in rows}
        if len(file_reviewers) != 1:
            raise ValueError(f"rating file must contain exactly one reviewer ID: {path}")
        reviewer = next(iter(file_reviewers))
        if reviewer not in reviewers or reviewer in observed_reviewers:
            raise ValueError(f"unknown or duplicate completed reviewer file: {reviewer!r}")
        observed_reviewers.add(reviewer)
        expected_items = {
            blind_item
            for frozen_reviewer, blind_item in mapping_index
            if frozen_reviewer == reviewer
        }
        observed_items: set[str] = set()
        for row_number, row in enumerate(rows, start=2):
            blind_item = str(row.get("blind_item_id", "")).strip()
            key = (reviewer, blind_item)
            label = f"{path.name}:{row_number}"
            if key not in mapping_index or blind_item in observed_items:
                raise ValueError(f"{label}: unexpected or duplicate blind_item_id")
            observed_items.add(blind_item)
            by_code = mapping_index[key]
            task_text = str(row.get("task_text_private", ""))
            expected_task_hashes = {str(value["task_text_sha256"]) for value in by_code.values()}
            if (
                len(expected_task_hashes) != 1
                or _sha256_text(task_text) not in expected_task_hashes
            ):
                raise ValueError(f"{label}: task text was changed after review preparation")
            for code in CANDIDATE_CODES:
                answer = str(row.get(f"answer_{code}", ""))
                if _sha256_text(answer) != str(by_code[code].get("response_sha256", "")):
                    raise ValueError(f"{label}: answer_{code} was changed after review preparation")
            role = str(row.get("target_role", "")).strip()
            variant = str(row.get("evaluation_condition", "")).strip()
            frozen_roles = {str(value["target_role"]) for value in by_code.values()}
            frozen_variants = {str(value["evaluation_condition"]) for value in by_code.values()}
            if frozen_roles != {role} or frozen_variants != {variant}:
                raise ValueError(f"{label}: role or evaluation condition was changed")
            parsed = _parse_ballot_semantics(row, label)
            code_to_system = {
                code: str(value["system_id_private"]) for code, value in by_code.items()
            }
            selected_systems = tuple(code_to_system[code] for code in parsed["selected_codes"])
            unsafe_systems = tuple(code_to_system[code] for code in parsed["unsafe_codes"])
            points = {system: 0.0 for system in systems}
            if parsed["choice"] != "SKIP" and selected_systems:
                credit = 1.0 / len(selected_systems)
                for system in selected_systems:
                    points[system] = credit
            mapping_sample = by_code[CANDIDATE_CODES[0]]
            ballot: dict[str, Any] = {
                "reviewer_id_private": reviewer,
                "blind_item_id_private": blind_item,
                "task_id_private": str(mapping_sample["task_id_private"]),
                "family_id_private": str(mapping_sample["family_id_private"]),
                "role": role,
                "variant": variant,
                "choice": parsed["choice"],
                "selected_systems": selected_systems,
                "unsafe_systems": unsafe_systems,
                "points": points,
                "primary_reason": parsed["primary_reason"],
                "confidence_1_3": parsed["confidence"],
                "skip_reason_private": parsed["skip_reason"],
                "review_notes_private": str(row.get("review_notes_private", "")),
            }
            for system in systems:
                ballot[f"point::{system}"] = points[system]
            ballots.append(ballot)
        if observed_items != expected_items:
            raise ValueError(
                f"incomplete rating file {path}: missing={len(expected_items - observed_items)}, "
                f"extra={len(observed_items - expected_items)}"
            )
    if observed_reviewers != set(reviewers):
        raise ValueError("completed rating files do not cover every frozen reviewer")

    public_reviewer = {reviewer: f"R{index}" for index, reviewer in enumerate(reviewers, start=1)}
    aggregate_rows = _system_stats(ballots, systems)
    ranking = sorted(
        aggregate_rows,
        key=lambda row: (
            -float(row["expert_equal_fractional_score_per_100"] or 0.0),
            -int(row["sole_best_count"]),
            str(row["system"]),
        ),
    )
    rank_by_system = {str(row["system"]): index for index, row in enumerate(ranking, start=1)}
    for row in aggregate_rows:
        row["numeric_rank"] = rank_by_system[str(row["system"])]

    comparators = [system for system in systems if system != args.primary_agent]
    pairwise_rows: list[dict[str, Any]] = []
    for comparator in comparators:
        point, low, high, family_count = _cluster_bootstrap_difference(
            ballots,
            args.primary_agent,
            comparator,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
        evaluable = [row for row in ballots if row["choice"] != "SKIP"]
        pairwise_rows.append(
            {
                "primary_system": args.primary_agent,
                "comparator_system": comparator,
                "evaluable_ballot_count": len(evaluable),
                "family_cluster_count": family_count,
                "primary_selected_comparator_not_count": sum(
                    args.primary_agent in row["selected_systems"]
                    and comparator not in row["selected_systems"]
                    for row in evaluable
                ),
                "comparator_selected_primary_not_count": sum(
                    comparator in row["selected_systems"]
                    and args.primary_agent not in row["selected_systems"]
                    for row in evaluable
                ),
                "both_selected_tie_count": sum(
                    args.primary_agent in row["selected_systems"]
                    and comparator in row["selected_systems"]
                    for row in evaluable
                ),
                "neither_selected_count": sum(
                    args.primary_agent not in row["selected_systems"]
                    and comparator not in row["selected_systems"]
                    for row in evaluable
                ),
                "expert_equal_fractional_score_difference": point,
                "difference_per_100": point * 100.0,
                "cluster_bootstrap_ci95_low_per_100": low * 100.0,
                "cluster_bootstrap_ci95_high_per_100": high * 100.0,
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )

    breakdown_rows: list[dict[str, Any]] = []
    dimensions: list[tuple[str, list[str], Any]] = [
        ("role", sorted({str(row["role"]) for row in ballots}), lambda row: str(row["role"])),
        (
            "variant",
            sorted({str(row["variant"]) for row in ballots}),
            lambda row: str(row["variant"]),
        ),
        (
            "reviewer",
            [public_reviewer[reviewer] for reviewer in reviewers],
            lambda row: public_reviewer[str(row["reviewer_id_private"])],
        ),
    ]
    for dimension, values, getter in dimensions:
        for value in values:
            subset = [row for row in ballots if getter(row) == value]
            for stats in _system_stats(subset, systems):
                breakdown_rows.append(
                    {
                        "dimension": dimension,
                        "dimension_value": value,
                        **stats,
                    }
                )

    by_reviewer_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in ballots:
        by_reviewer_task[str(row["reviewer_id_private"])][str(row["task_id_private"])] = row
    agreement_rows: list[dict[str, Any]] = []
    for left_reviewer, right_reviewer in combinations(reviewers, 2):
        shared = sorted(
            set(by_reviewer_task[left_reviewer]) & set(by_reviewer_task[right_reviewer])
        )
        both_evaluable: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for task_id in shared:
            left = by_reviewer_task[left_reviewer][task_id]
            right = by_reviewer_task[right_reviewer][task_id]
            if left["choice"] != "SKIP" and right["choice"] != "SKIP":
                both_evaluable.append((left, right))
        exact = sum(
            set(left["selected_systems"]) == set(right["selected_systems"])
            for left, right in both_evaluable
        )
        jaccards: list[float] = []
        categories_left: list[str] = []
        categories_right: list[str] = []
        for left, right in both_evaluable:
            left_set, right_set = set(left["selected_systems"]), set(right["selected_systems"])
            union = left_set | right_set
            jaccards.append(len(left_set & right_set) / len(union) if union else 1.0)
            if len(left_set) <= 1 and len(right_set) <= 1:
                categories_left.append(next(iter(left_set), "NONE"))
                categories_right.append(next(iter(right_set), "NONE"))
        agreement_rows.append(
            {
                "reviewer_left": public_reviewer[left_reviewer],
                "reviewer_right": public_reviewer[right_reviewer],
                "shared_item_count": len(shared),
                "both_evaluable_item_count": len(both_evaluable),
                "exact_selected_set_agreement": exact / len(both_evaluable)
                if both_evaluable
                else None,
                "mean_selected_set_jaccard": mean(jaccards) if jaccards else None,
                "unique_or_none_pair_count": len(categories_left),
                "cohen_kappa_unique_or_none": _cohen_kappa(categories_left, categories_right),
            }
        )

    private_dir = args.output_private.expanduser().resolve()
    public_dir = args.output_public.expanduser().resolve()
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    if private_dir == public_dir:
        raise ValueError("private and public output directories must differ")
    if any(private_dir.iterdir()) or any(public_dir.iterdir()):
        raise FileExistsError(
            "summary output directories must be empty; use new directories so frozen aggregate "
            "evidence is never overwritten after unblinding"
        )

    ballot_path = private_dir / "expert_preference_ballots.private.csv"
    private_fields = [
        "reviewer_id_private",
        "blind_item_id_private",
        "task_id_private",
        "family_id_private",
        "role",
        "variant",
        "choice",
        "selected_systems",
        "unsafe_systems",
        *[f"point::{system}" for system in systems],
        "primary_reason",
        "confidence_1_3",
        "skip_reason_private",
        "review_notes_private",
    ]
    private_rows: list[dict[str, Any]] = []
    for row in ballots:
        private_rows.append(
            {
                **{field: row.get(field, "") for field in private_fields},
                "selected_systems": "|".join(row["selected_systems"]),
                "unsafe_systems": "|".join(row["unsafe_systems"]),
            }
        )
    _write_csv(ballot_path, private_fields, private_rows)

    aggregate_path = public_dir / "expert_preference_system_aggregate.csv"
    pairwise_path = public_dir / "expert_preference_pairwise.csv"
    breakdown_path = public_dir / "expert_preference_breakdown.csv"
    agreement_path = public_dir / "expert_preference_agreement.csv"
    summary_path = public_dir / "expert_preference_summary.json"
    _write_csv(aggregate_path, list(aggregate_rows[0]), aggregate_rows)
    _write_csv(pairwise_path, list(pairwise_rows[0]), pairwise_rows)
    _write_csv(breakdown_path, list(breakdown_rows[0]), breakdown_rows)
    agreement_fields = (
        list(agreement_rows[0])
        if agreement_rows
        else [
            "reviewer_left",
            "reviewer_right",
            "shared_item_count",
            "both_evaluable_item_count",
            "exact_selected_set_agreement",
            "mean_selected_set_jaccard",
            "unique_or_none_pair_count",
            "cohen_kappa_unique_or_none",
        ]
    )
    _write_csv(agreement_path, agreement_fields, agreement_rows)

    top_score = float(ranking[0]["expert_equal_fractional_score_per_100"] or 0.0)
    second_score = float(ranking[1]["expert_equal_fractional_score_per_100"] or 0.0)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "FINAL",
        "data_scope": {
            "system_count": len(systems),
            "reviewer_count": len(reviewers),
            "task_count": manifest.get("task_count"),
            "ballot_count": len(ballots),
            "evaluable_ballot_count": sum(row["choice"] != "SKIP" for row in ballots),
            "skip_ballot_count": sum(row["choice"] == "SKIP" for row in ballots),
            "none_acceptable_ballot_count": sum(row["choice"] == "NONE" for row in ballots),
            "roles": manifest.get("roles", []),
            "conditions": manifest.get("variants", []),
        },
        "analysis": {
            "evaluation_unit": "end-to-end agent system",
            "primary_endpoint": (
                "safety-qualified expert first-choice share, with 1/k credit for k-way ties "
                "and equal weighting across experts"
            ),
            "tie_credit": "1/k for each of k tied-best systems",
            "none_credit": "zero for every system; ballot remains in denominator",
            "skip_handling": "excluded; private reason required",
            "safety_gate": (
                "unsafe_candidates was explicitly completed on every non-SKIP ballot; "
                "a safety-flagged candidate was prohibited from receiving best-answer credit"
            ),
            "bootstrap": {
                "cluster": "case family",
                "replicates": args.bootstrap_replicates,
                "seed": args.bootstrap_seed,
            },
            "numeric_leader": str(ranking[0]["system"]),
            "numeric_leader_score_per_100": top_score,
            "numeric_margin_over_second_per_100": top_score - second_score,
            "interpretation_guardrail": (
                "Numeric leadership is not a significance claim; inspect all prespecified pairwise "
                "family-cluster bootstrap intervals. This is an end-to-end project-level "
                "preference comparison and does not isolate the causal contribution of CARE-OPD."
            ),
        },
        "system_aggregates": aggregate_rows,
        "pairwise_comparisons": pairwise_rows,
        "agreement": agreement_rows,
        "integrity": {
            "preparation_manifest_sha256": _sha256_file(manifest_path),
            "blind_mapping_sha256": _sha256_file(mapping_path),
            "completed_rating_file_sha256": sorted(rating_hashes),
            "answer_and_task_text_hashes_verified": True,
            "strict_choice_contract_verified": True,
        },
        "privacy_statement": (
            "Aggregate-only output: no task or family identifier, clinical text, answer, blind "
            "code mapping, reviewer identity, skip reason, or free-text note is included."
        ),
    }
    _write_json(summary_path, summary)
    return {
        "ballots_private": ballot_path,
        "aggregate_public": aggregate_path,
        "pairwise_public": pairwise_path,
        "breakdown_public": breakdown_path,
        "agreement_public": agreement_path,
        "summary_public": summary_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument(
        "--ratings",
        action="append",
        required=True,
        type=Path,
        help="one completed CSV/XLSX per reviewer",
    )
    parser.add_argument("--primary-agent", required=True)
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
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
