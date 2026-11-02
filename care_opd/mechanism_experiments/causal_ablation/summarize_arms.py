#!/usr/bin/env python3
"""Summarize 3,900-row family-clean QZBC ablation rollouts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
PUBLIC_ROOT = HERE.parents[1] / "public_eval"
for root in (HERE.parents[1] / "src", PUBLIC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import summarize_internal_qzbc as shared  # type: ignore  # noqa: E402

try:
    from .build_views import _atomic_text, _sha256
except ImportError:
    from build_views import _atomic_text, _sha256


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def _index(rows: Sequence[Mapping[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(rows, 1):
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"{source}:{line_no}: non-empty id is required")
        if rid in result:
            raise ValueError(f"{source}:{line_no}: duplicate id {rid!r}")
        result[rid] = dict(row)
    return result


def _label_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--rollout must be LABEL=PATH")
    label, path = raw.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--rollout must be LABEL=PATH")
    return label, Path(path)


def _family_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value: Any = payload
    if isinstance(payload, dict):
        value = payload.get("family_clean_view", payload).get("request_ids")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("family-clean ID file must be a JSON list or DATA_AUDIT-style object")
    if len(value) != 3900 or len(set(value)) != 3900:
        raise ValueError(f"family-clean view must contain exactly 3900 unique IDs, got {len(value)}")
    return value


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _reference_bootstrap(
    *,
    arm: Mapping[str, bool],
    reference: Mapping[str, bool],
    request_map: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    seed: int = 20260813,
    replicates: int = 10_000,
) -> dict[str, Any]:
    by_seed: dict[str, list[str]] = {}
    for rid in ids:
        seed_id = str(request_map[rid].get("seed_id") or "")
        if not seed_id:
            raise ValueError(f"request {rid}: seed_id is required for cluster bootstrap")
        by_seed.setdefault(seed_id, []).append(rid)
    clusters = sorted(by_seed)
    rng = random.Random(seed)
    samples = {"rescue_rate": [], "regression_rate": [], "net_success_delta": []}
    for _ in range(replicates):
        rescue = regression = reference_fail = reference_pass = total = 0
        for _draw in clusters:
            for rid in by_seed[rng.choice(clusters)]:
                total += 1
                if reference[rid]:
                    reference_pass += 1
                    regression += int(not arm[rid])
                else:
                    reference_fail += 1
                    rescue += int(arm[rid])
        if reference_fail:
            samples["rescue_rate"].append(rescue / reference_fail)
        if reference_pass:
            samples["regression_rate"].append(regression / reference_pass)
        samples["net_success_delta"].append((rescue - regression) / total)
    return {
        "method": "paired seed_id-cluster percentile bootstrap",
        "seed": seed,
        "replicates": replicates,
        "intervals": {
            name: {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
                "replicates_used": len(values),
            }
            for name, values in samples.items()
            if values
        },
    }


def summarize(
    requests_path: Path,
    family_ids_path: Path,
    rollouts: Sequence[tuple[str, Path]],
    output_dir: Path,
    *,
    reference: str,
) -> dict[str, Any]:
    requests = _read_rows(requests_path)
    request_map = _index(requests, source=requests_path)
    family_ids = _family_ids(family_ids_path)
    missing = [rid for rid in family_ids if rid not in request_map]
    if missing:
        raise ValueError(f"family-clean IDs absent from requests: {missing[:5]}")
    subset = {rid: request_map[rid] for rid in family_ids}
    labels = [label for label, _ in rollouts]
    if len(labels) != len(set(labels)) or reference not in labels:
        raise ValueError("rollout labels must be unique and include --reference")
    models: dict[str, Any] = {}
    pairs: dict[str, list[dict[str, Any]]] = {}
    outcomes: dict[str, dict[str, bool]] = {}
    for label, path in rollouts:
        rows = _read_rows(path)
        by_id = _index(rows, source=path)
        if not set(family_ids).issubset(by_id):
            raise ValueError(f"{label}: rollout does not cover the 3900 family-clean IDs")
        derived = output_dir / "family_clean_views" / f"{label}.jsonl"
        _atomic_text(
            derived,
            "".join(json.dumps(by_id[rid], ensure_ascii=False, sort_keys=True) + "\n" for rid in family_ids),
        )
        model_pairs, payload = shared._model_pairs(subset, derived, clean_variant="clean")
        payload["source"] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        models[label] = payload
        pairs[label] = model_pairs
        outcomes[label] = {rid: bool(by_id[rid].get("passed")) for rid in family_ids}
    reference_outcomes = outcomes[reference]
    relative: dict[str, Any] = {}
    for label in labels:
        arm = outcomes[label]
        reference_fail = sum(not reference_outcomes[rid] for rid in family_ids)
        reference_pass = len(family_ids) - reference_fail
        rescue_count = sum(arm[rid] and not reference_outcomes[rid] for rid in family_ids)
        regression_count = sum((not arm[rid]) and reference_outcomes[rid] for rid in family_ids)
        relative[label] = {
            "scope": "all 3900 family-clean requests",
            "reference": reference,
            "rescue_definition": "P(arm pass | reference fail)",
            "regression_definition": "P(arm fail | reference pass)",
            "rescue_count": rescue_count,
            "reference_failure_count": reference_fail,
            "rescue_rate": rescue_count / reference_fail if reference_fail else None,
            "regression_count": regression_count,
            "reference_success_count": reference_pass,
            "regression_rate": regression_count / reference_pass if reference_pass else None,
            "net_success_delta": (rescue_count - regression_count) / len(family_ids),
            "net_success_delta_definition": "(rescued - regressed) / 3900; equals mixed-request TSR arm-minus-reference",
            "paired_ci95": _reference_bootstrap(
                arm=arm,
                reference=reference_outcomes,
                request_map=request_map,
                ids=family_ids,
            ),
        }
    def comparison_payload(label: str) -> dict[str, Any]:
        perturbations = sorted(
            {str(row["perturbation"]) for row in pairs[label]}
        )
        by_perturbation = {}
        for perturbation in perturbations:
            left_rows = [
                row
                for row in pairs[label]
                if str(row["perturbation"]) == perturbation
            ]
            reference_rows = [
                row
                for row in pairs[reference]
                if str(row["perturbation"]) == perturbation
            ]
            by_perturbation[perturbation] = shared._bootstrap_delta(
                left_rows, reference_rows
            )
        return {
            "left_arm": label,
            "reference_arm": reference,
            "shared_implementation_field_note": (
                "Within metrics, legacy keys care_opd/baseline mean "
                "left_arm/reference_arm."
            ),
            **shared._bootstrap_delta(pairs[label], pairs[reference]),
            "by_perturbation": by_perturbation,
        }

    comparisons = {
        f"{label}_vs_{reference}": comparison_payload(label)
        for label in labels
        if label != reference
    }
    result = {
        "schema_version": "care_opd.causal_ablation_summary.v1",
        "view": "QZBC family-clean 3900-request retrospective sensitivity view",
        "requests": {"path": str(requests_path.resolve()), "sha256": _sha256(requests_path)},
        "family_ids": {"path": str(family_ids_path.resolve()), "sha256": _sha256(family_ids_path)},
        "reference": reference,
        "models": models,
        "reference_conditioned_request_metrics": relative,
        "paired_cluster_bootstrap_comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / "summary.json", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    columns = ["model", "clean_tsr", "perturbed_tsr", "conditional_retention", "internal_regression", "reference_rescue", "reference_regression", "net_success_delta", "paired_units"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    table: list[dict[str, Any]] = []
    for label in labels:
        metric = models[label]["unified_metrics"]
        row = {
            "model": label,
            "clean_tsr": metric["clean_task_success_rate"],
            "perturbed_tsr": metric["perturbed_task_success_rate"],
            "conditional_retention": metric["conditional_retention"],
            "internal_regression": metric["regression_rate"],
            "reference_rescue": relative[label]["rescue_rate"],
            "reference_regression": relative[label]["regression_rate"],
            "net_success_delta": relative[label]["net_success_delta"],
            "paired_units": metric["paired_units"],
        }
        table.append(row)
        writer.writerow(row)
    _atomic_text(output_dir / "summary.csv", stream.getvalue())
    md = ["# Causal ablation results", "", f"Reference-conditioned metrics use `{reference}` on the same 3,900 request IDs: Rescue = P(arm pass | reference fail), Regression = P(arm fail | reference pass), Net = (rescued - regressed) / 3,900.", "", "| Arm | Clean TSR | Perturbed TSR | Conditional retention | Ref rescue | Ref regression | Net | Pairs |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    pct = lambda value: "N/A" if value is None else f"{100 * value:.2f}%"
    for row in table:
        md.append(f"| {row['model']} | {pct(row['clean_tsr'])} | {pct(row['perturbed_tsr'])} | {pct(row['conditional_retention'])} | {pct(row['reference_rescue'])} | {pct(row['reference_regression'])} | {pct(row['net_success_delta'])} | {row['paired_units']} |")
    md += ["", "Paired percentile CIs use the existing seed-ID cluster bootstrap implementation; inspect `summary.json`.", ""]
    _atomic_text(output_dir / "summary.md", "\n".join(md))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--family-clean-ids", type=Path, required=True)
    parser.add_argument("--rollout", action="append", type=_label_path, required=True)
    parser.add_argument("--reference", default="full_selected")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summarize(args.requests, args.family_clean_ids, args.rollout, args.output_dir, reference=args.reference)
    print(f"summary={args.output_dir.resolve() / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
