#!/usr/bin/env python3
"""Build the CARE-OPD main effectiveness/robustness tables from real results.

The canonical output is a long table with one method × benchmark × track per
row.  Two compact views expose task-success effectiveness and conditional
retention without manufacturing a cross-benchmark aggregate score.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from preregistration import MODEL_ORDER, sha256_file, validate_static_lock


PROJECT_ROOT = HERE.parent
STANDARD_METRICS = (
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
)
PRIMARY_REQUIRED_SOURCES = (
    "internal_qzbc",
    "tcmeval_official",
    "tcmeval_controlled",
    "tcmbench",
    "mtcmb",
)
PRIMARY_GATE_BASELINES = ("clean_sft", "gkd_full")
PRIMARY_GATE_ENDPOINTS = (
    "perturbed_task_success_rate",
    "conditional_retention",
)
PRIMARY_GATE_BENCHMARKS = (
    "TCMEval-PA v3",
    "TCMBench V2",
    "MTCMB-Clean",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
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
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _assert_order(labels: Iterable[str], source: str) -> None:
    observed = tuple(labels)
    if observed != MODEL_ORDER:
        raise ValueError(f"{source}: expected model order {MODEL_ORDER}, got {observed}")


def _blank_metrics() -> dict[str, Any]:
    return {key: None for key in STANDARD_METRICS}


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: expected an object")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location}: expected a finite number")
    return result


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location}: expected a boolean")
    return value


def _paired_units(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location}: expected a positive integer")
    return value


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _gate_row(
    *,
    benchmark: str,
    track: str,
    baseline: str,
    paired_units: Any,
    margin: Any,
    endpoints: Mapping[str, Mapping[str, Any]],
    joint_pass: Any,
) -> dict[str, Any]:
    if baseline not in PRIMARY_GATE_BASELINES:
        raise ValueError(f"{benchmark}: unexpected primary baseline {baseline!r}")
    margin_value = _number(margin, f"{benchmark}/{baseline}/margin")
    row: dict[str, Any] = {
        "benchmark": benchmark,
        "track": track,
        "baseline": baseline,
        "paired_units": _paired_units(
            paired_units, f"{benchmark}/{baseline}/paired_units"
        ),
        "margin_absolute": margin_value,
    }
    endpoint_passes = []
    if set(endpoints) != set(PRIMARY_GATE_ENDPOINTS):
        raise ValueError(
            f"{benchmark}/{baseline}: primary endpoint set is not closed: "
            f"{sorted(endpoints)}"
        )
    for endpoint in PRIMARY_GATE_ENDPOINTS:
        block = _mapping(endpoints[endpoint], f"{benchmark}/{baseline}/{endpoint}")
        delta = _number(block.get("delta"), f"{benchmark}/{baseline}/{endpoint}/delta")
        low = _number(block.get("ci95_low"), f"{benchmark}/{baseline}/{endpoint}/ci95_low")
        high = _number(
            block.get("ci95_high"), f"{benchmark}/{baseline}/{endpoint}/ci95_high"
        )
        endpoint_margin = _number(
            block.get("margin"), f"{benchmark}/{baseline}/{endpoint}/margin"
        )
        endpoint_pass = _boolean(
            block.get("pass"), f"{benchmark}/{baseline}/{endpoint}/pass"
        )
        # A percentile-bootstrap interval may exclude its plug-in estimate
        # when the bootstrap distribution is biased, so only interval order is
        # a structural invariant here.
        if low > high:
            raise ValueError(f"{benchmark}/{baseline}/{endpoint}: CI bounds reversed")
        if not _same_number(endpoint_margin, margin_value):
            raise ValueError(
                f"{benchmark}/{baseline}/{endpoint}: endpoint/top-level margins differ"
            )
        if endpoint_pass != bool(low > margin_value):
            raise ValueError(
                f"{benchmark}/{baseline}/{endpoint}: non-inferiority pass is inconsistent"
            )
        prefix = f"{endpoint}_care_minus_baseline"
        row[f"{prefix}_delta"] = delta
        row[f"{prefix}_ci95_low"] = low
        row[f"{prefix}_ci95_high"] = high
        row[f"{prefix}_noninferiority_pass"] = endpoint_pass
        endpoint_passes.append(endpoint_pass)
    observed_joint = _boolean(joint_pass, f"{benchmark}/{baseline}/joint_pass")
    if observed_joint != all(endpoint_passes):
        raise ValueError(f"{benchmark}/{baseline}: joint gate is inconsistent")
    row["joint_pass"] = observed_joint
    return row


def _closed_gate_rows(rows: Sequence[Mapping[str, Any]], source: str) -> None:
    baselines = [str(row.get("baseline")) for row in rows]
    if len(rows) != len(PRIMARY_GATE_BASELINES) or set(baselines) != set(
        PRIMARY_GATE_BASELINES
    ):
        raise ValueError(
            f"{source}: expected exactly {PRIMARY_GATE_BASELINES}, got {baselines}"
        )


def extract_tcmeval_gates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    block = _mapping(payload.get("controlled_derived_results"), "TCMEval controlled")
    comparisons = block.get("care_paired_comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("TCMEval controlled: care_paired_comparisons must be a list")
    rows = []
    for comparison_value in comparisons:
        comparison = _mapping(comparison_value, "TCMEval controlled comparison")
        baseline = str(comparison.get("baseline"))
        intervals = _mapping(
            comparison.get("delta_and_95_ci"), f"TCMEval/{baseline}/delta_and_95_ci"
        )
        gate = _mapping(
            comparison.get("noninferiority"), f"TCMEval/{baseline}/noninferiority"
        )
        gate_endpoints = _mapping(
            gate.get("endpoints"), f"TCMEval/{baseline}/noninferiority/endpoints"
        )
        endpoints = {}
        for endpoint in PRIMARY_GATE_ENDPOINTS:
            interval = _mapping(intervals.get(endpoint), f"TCMEval/{baseline}/{endpoint}")
            endpoint_gate = _mapping(
                gate_endpoints.get(endpoint), f"TCMEval/{baseline}/{endpoint}/gate"
            )
            low = _number(interval.get("low"), f"TCMEval/{baseline}/{endpoint}/low")
            gate_low = _number(
                endpoint_gate.get("ci95_low"),
                f"TCMEval/{baseline}/{endpoint}/gate/ci95_low",
            )
            if not _same_number(low, gate_low):
                raise ValueError(f"TCMEval/{baseline}/{endpoint}: gate CI differs")
            endpoints[endpoint] = {
                "delta": interval.get("delta"),
                "ci95_low": low,
                "ci95_high": interval.get("high"),
                "margin": endpoint_gate.get("margin"),
                "pass": endpoint_gate.get("pass"),
            }
        rows.append(
            _gate_row(
                benchmark="TCMEval-PA v3",
                track="controlled_robustness_v1 / normalized-stem cluster bootstrap",
                baseline=baseline,
                paired_units=comparison.get("paired_units"),
                margin=gate.get("margin_absolute"),
                endpoints=endpoints,
                joint_pass=gate.get("joint_pass"),
            )
        )
    _closed_gate_rows(rows, "TCMEval controlled")
    return rows


def extract_tcmbench_gates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    comparisons = _mapping(
        payload.get("main_pairwise_comparisons"), "TCMBench main_pairwise_comparisons"
    )
    expected_keys = {f"care_opd_vs_{baseline}" for baseline in PRIMARY_GATE_BASELINES}
    if set(comparisons) != expected_keys:
        raise ValueError(
            "TCMBench main_pairwise_comparisons is not a closed two-baseline block"
        )
    rows = []
    for baseline in PRIMARY_GATE_BASELINES:
        root = _mapping(comparisons[f"care_opd_vs_{baseline}"], f"TCMBench/{baseline}")
        overlap = _mapping(root.get("overlap_clean"), f"TCMBench/{baseline}/overlap_clean")
        bootstrap = _mapping(
            overlap.get("primary_case_cluster_bootstrap"),
            f"TCMBench/{baseline}/primary_case_cluster_bootstrap",
        )
        metrics = _mapping(bootstrap.get("metrics"), f"TCMBench/{baseline}/metrics")
        endpoints = {}
        for endpoint in PRIMARY_GATE_ENDPOINTS:
            metric = _mapping(
                metrics.get(f"{endpoint}_delta_care_minus_baseline"),
                f"TCMBench/{baseline}/{endpoint}",
            )
            endpoints[endpoint] = {
                "delta": metric.get("estimate"),
                "ci95_low": metric.get("ci95_lower"),
                "ci95_high": metric.get("ci95_upper"),
                "margin": metric.get("noninferiority_margin"),
                "pass": metric.get("noninferiority_pass"),
            }
        rows.append(
            _gate_row(
                benchmark="TCMBench V2",
                track="overlap_clean SDT→reverse/shuffle / case-cluster bootstrap",
                baseline=baseline,
                paired_units=bootstrap.get("paired_rows"),
                margin=endpoints[PRIMARY_GATE_ENDPOINTS[0]]["margin"],
                endpoints=endpoints,
                joint_pass=bootstrap.get("joint_primary_noninferiority_pass"),
            )
        )
    _closed_gate_rows(rows, "TCMBench")
    return rows


def extract_mtcmb_gates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    comparisons = payload.get("primary_paired_noninferiority")
    if not isinstance(comparisons, list):
        raise ValueError("MTCMB primary_paired_noninferiority must be a list")
    rows = []
    for comparison_value in comparisons:
        comparison = _mapping(comparison_value, "MTCMB paired comparison")
        baseline = str(comparison.get("baseline"))
        raw_endpoints = _mapping(
            comparison.get("endpoints"), f"MTCMB/{baseline}/endpoints"
        )
        endpoints = {}
        for endpoint in PRIMARY_GATE_ENDPOINTS:
            metric = _mapping(raw_endpoints.get(endpoint), f"MTCMB/{baseline}/{endpoint}")
            endpoints[endpoint] = {
                "delta": metric.get("delta"),
                "ci95_low": metric.get("ci95_low"),
                "ci95_high": metric.get("ci95_high"),
                "margin": metric.get("margin"),
                "pass": metric.get("noninferiority_pass"),
            }
        rows.append(
            _gate_row(
                benchmark="MTCMB-Clean",
                track="Clean9 SE-B option permutation / paired item bootstrap",
                baseline=baseline,
                paired_units=comparison.get("paired_units"),
                margin=comparison.get("margin_absolute"),
                endpoints=endpoints,
                joint_pass=comparison.get("joint_primary_noninferiority_pass"),
            )
        )
    _closed_gate_rows(rows, "MTCMB")
    return rows


def _row(
    *,
    model: str,
    benchmark: str,
    track: str,
    provenance: str,
    evidence_role: str,
    official_or_derived: str,
    clean_n: int | None,
    perturbed_n: int | None,
    metrics: Mapping[str, Any],
    effectiveness_display_key: str,
    robustness_display_key: str | None,
    full_view_n: int | None = None,
    full_view_tsr: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    values = _blank_metrics()
    for key in STANDARD_METRICS:
        if key in metrics:
            values[key] = metrics[key]
    return {
        "model": model,
        "benchmark": benchmark,
        "track": track,
        "track_provenance": provenance,
        "official_or_derived": official_or_derived,
        "evidence_role": evidence_role,
        "clean_n": clean_n,
        "perturbed_n": perturbed_n,
        **values,
        "full_view_n": full_view_n,
        "full_view_tsr": full_view_tsr,
        "effectiveness_display_key": effectiveness_display_key,
        "robustness_display_key": robustness_display_key,
        "notes": notes,
    }


def extract_internal(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") == "care-opd-qzbc-grouped-suite-v1":
        records = payload.get("main_table_records")
        if not isinstance(records, list):
            raise ValueError("formal internal QZBC summary has no main_table_records")
        checkpoint = _mapping(
            payload.get("checkpoint_identity_binding"),
            "formal internal QZBC checkpoint binding",
        )
        split = _mapping(
            payload.get("split_integrity_binding"),
            "formal internal QZBC split binding",
        )
        if checkpoint.get("status") != "PASS_CURRENT_CHECKPOINT_BYTES":
            raise ValueError("formal internal QZBC checkpoint binding did not pass")
        if split.get("status") != "PASS_CURRENT_SPLIT_AND_TEST_RECOMPILE":
            raise ValueError("formal internal QZBC split binding did not pass")
        training = _mapping(
            split.get("training_provenance"),
            "formal internal QZBC training provenance",
        )
        if training.get("status") != "CURRENT_DATA_AUDITED_TRAINING_BYTES_UNVERIFIED":
            raise ValueError("formal internal QZBC training-provenance disclosure changed")
        labels = [str(item.get("model")) for item in records if isinstance(item, Mapping)]
        if len(labels) != len(records):
            raise ValueError("formal internal QZBC contains an invalid record")
        _assert_order(labels, "formal internal QZBC")
        rows = []
        for item in records:
            metrics = _mapping(
                item.get("unified_metrics"),
                f"formal internal QZBC/{item.get('model')}/metrics",
            )
            rows.append(
                _row(
                    model=str(item["model"]),
                    benchmark=str(item.get("benchmark") or "QZBC grouped family-clean test"),
                    track=str(item.get("track") or "identity-bound paired variants"),
                    provenance=str(item.get("track_provenance") or payload.get("track_provenance")),
                    evidence_role=str(item.get("evidence_role") or "in-domain mechanism evidence"),
                    official_or_derived=str(item.get("official_or_derived") or "internal_formal_derived"),
                    clean_n=int(item["clean_n"]),
                    perturbed_n=int(item["perturbed_n"]),
                    metrics=metrics,
                    effectiveness_display_key=str(
                        item.get("effectiveness_display_key") or "QZBC perturbed TSR"
                    ),
                    robustness_display_key=item.get("robustness_display_key") or "QZBC retention",
                    full_view_n=(
                        int(item["full_view_n"])
                        if item.get("full_view_n") is not None
                        else None
                    ),
                    full_view_tsr=(
                        float(item["full_view_tsr"])
                        if item.get("full_view_tsr") is not None
                        else None
                    ),
                    notes=str(item.get("notes") or ""),
                )
            )
        return rows

    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("internal QZBC summary has no model object")
    _assert_order(models.keys(), "internal QZBC")
    rows = []
    for label, item in models.items():
        metrics = item["unified_metrics"]
        diagnostics = item["diagnostics"]
        rows.append(
            _row(
                model=str(label),
                benchmark="QZBC grouped test (legacy)",
                track="paired clean→recovery variants",
                provenance="Internal retrospective / checkpoint identity unverified",
                evidence_role="retrospective in-domain diagnostic",
                official_or_derived="internal_legacy",
                clean_n=int(metrics["clean_unique_units"]),
                perturbed_n=int(metrics["paired_units"]),
                metrics=metrics,
                effectiveness_display_key="QZBC perturbed TSR",
                robustness_display_key="QZBC retention",
                full_view_n=int(diagnostics["rollouts"]),
                full_view_tsr=float(diagnostics["overall_mixed_task_success_rate"]),
                notes=(
                    "The legacy mixed pass rate is shown only as full_view_tsr; paired perturbed TSR "
                    "and retention are re-analysis endpoints. Legacy rollout artifacts do not embed "
                    "checkpoint fingerprints or the split-integrity manifest, so this row is not "
                    "used for formal OPD attribution until an identity-bound rerun is complete."
                ),
            )
        )
    return rows


def extract_tcmeval_official(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("dataset_sha256") != (
        "905cb28f145ba50f3b1c912265ff6649e0bb38f56adf6cd588d77ca768db92c1"
    ):
        raise ValueError("TCMEval official dataset fingerprint changed")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("TCMEval official summary has no model list")
    _assert_order((str(item.get("model")) for item in models), "TCMEval official")
    return [
        _row(
            model=str(item["model"]),
            benchmark="TCMEval-PA v3",
            track="official clean",
            provenance="Public official",
            evidence_role="external TCM prescription-audit effectiveness",
            official_or_derived="official",
            clean_n=int(item["n"]),
            perturbed_n=None,
            metrics={"clean_task_success_rate": item["option_set_accuracy"]},
            effectiveness_display_key="TCMEval official TSR",
            robustness_display_key=None,
            notes="Official option-set exact; no controlled perturbation in this row.",
        )
        for item in models
    ]


def extract_tcmeval_controlled(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    block = payload.get("controlled_derived_results")
    if not isinstance(block, Mapping) or not isinstance(block.get("models"), list):
        raise ValueError("TCMEval controlled summary is malformed")
    models = block["models"]
    _assert_order((str(item.get("model")) for item in models), "TCMEval controlled")
    return [
        _row(
            model=str(item["model"]),
            benchmark="TCMEval-PA v3",
            track="controlled_robustness_v1",
            provenance="Public derived / label-blind option permutation",
            evidence_role="external decision robustness",
            official_or_derived="derived",
            clean_n=int(item["paired_units"]),
            perturbed_n=int(item["paired_units"]),
            metrics=item,
            effectiveness_display_key="TCMEval perturbed TSR",
            robustness_display_key="TCMEval retention",
            notes="Not an official leaderboard score; clean and permuted prompts are label-blind.",
        )
        for item in models
    ]


def extract_cmb(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("CMB summary has no model list")
    _assert_order((str(item.get("model")) for item in models), "CMB")
    rows = []
    for item in models:
        label = str(item["model"])
        rows.append(
            _row(
                model=label,
                benchmark="CMB-Exam",
                track="overlap-clean official",
                provenance="Public official / exact-overlap-excluded view",
                evidence_role="broad Chinese medical knowledge preservation",
                official_or_derived="official",
                clean_n=int(item["overlap_clean_n"]),
                perturbed_n=None,
                metrics={"clean_task_success_rate": item["overlap_clean_accuracy"]},
                effectiveness_display_key="CMB clean TSR",
                robustness_display_key=None,
                full_view_n=int(item["official_full_n"]),
                full_view_tsr=float(item["official_full_accuracy"]),
                notes="Exact-overlap exclusion is not claimed to remove semantic or pretraining contamination.",
            )
        )
        rows.append(
            _row(
                model=label,
                benchmark="CMB-Exam",
                track="controlled_robustness_v1",
                provenance="Public derived / category-stratified option permutation",
                evidence_role="broad medical decision robustness",
                official_or_derived="derived",
                clean_n=int(item["paired_units"]),
                perturbed_n=int(item["paired_units"]),
                metrics=item,
                effectiveness_display_key="CMB perturbed TSR",
                robustness_display_key="CMB retention",
                notes="Derived 2,800-pair track; not an official CMB score.",
            )
        )
    return rows


def extract_tcmbench(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("TCMBench summary has no model object")
    _assert_order(models.keys(), "TCMBench")
    rows = []
    for label, item in models.items():
        view = item.get("overlap_clean")
        if not isinstance(view, Mapping):
            raise ValueError(f"TCMBench overlap_clean missing for {label}")
        metrics = view.get("public_paired_robustness_track")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"TCMBench paired track missing for {label}")
        micro = view.get("micro") or {}
        rows.append(
            _row(
                model=str(label),
                benchmark="TCMBench V2",
                track="audited SDT→reverse/shuffle pair map",
                provenance="Public official tasks + project-derived semantic pair map",
                evidence_role="public TCM reasoning perturbation transfer",
                official_or_derived="derived",
                clean_n=int(metrics["paired_units"]),
                perturbed_n=int(metrics["paired_units"]),
                metrics=metrics,
                effectiveness_display_key="TCMBench perturbed TSR",
                robustness_display_key="TCMBench retention",
                full_view_n=int(micro.get("n") or 0),
                full_view_tsr=micro.get("accuracy"),
                notes=(
                    "Primary inference is case_index-clustered. DDST/SDDT defect slices are diagnostic only."
                ),
            )
        )
    return rows


def extract_mtcmb(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Accept the shared long-record contract emitted by mtcmb_clean."""

    records = payload.get("main_table_records")
    if records is None:
        return []
    if not isinstance(records, list):
        raise ValueError("MTCMB main_table_records must be a list")
    labels = []
    rows = []
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError("invalid MTCMB main table record")
        labels.append(str(item.get("model")))
        metrics = item.get("unified_metrics") or item
        rows.append(
            _row(
                model=str(item["model"]),
                benchmark="MTCMB-Clean",
                track=str(item.get("track") or "provenance-clean objective tasks"),
                provenance=str(item.get("track_provenance") or "Public filtered supplement"),
                evidence_role="Clean9 SE-B controlled robustness",
                official_or_derived=str(item.get("official_or_derived") or "filtered_official"),
                clean_n=int(item.get("clean_n") or item.get("n") or 0),
                perturbed_n=(
                    int(item["perturbed_n"]) if item.get("perturbed_n") is not None else None
                ),
                metrics=metrics,
                effectiveness_display_key=str(
                    item.get("effectiveness_display_key") or "MTCMB-Clean TSR"
                ),
                robustness_display_key=item.get("robustness_display_key"),
                full_view_n=item.get("full_view_n"),
                full_view_tsr=item.get("full_view_tsr"),
                notes=str(item.get("notes") or ""),
            )
        )
    # MTCMB may emit more than one track; validate each complete six-model block.
    if len(labels) % len(MODEL_ORDER) != 0:
        raise ValueError("MTCMB records do not form complete six-model blocks")
    for start in range(0, len(labels), len(MODEL_ORDER)):
        _assert_order(labels[start : start + len(MODEL_ORDER)], "MTCMB")
    return rows


def _format_rate(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _format_delta_interval(row: Mapping[str, Any], endpoint: str) -> str:
    prefix = f"{endpoint}_care_minus_baseline"
    return (
        f"{100.0 * float(row[f'{prefix}_delta']):+.2f} pp "
        f"[{100.0 * float(row[f'{prefix}_ci95_low']):+.2f}, "
        f"{100.0 * float(row[f'{prefix}_ci95_high']):+.2f}]"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def _compact_matrix(
    rows: Sequence[Mapping[str, Any]], key_field: str, value_field: str
) -> tuple[list[str], list[dict[str, Any]]]:
    columns = []
    by_model = {label: {"model": label} for label in MODEL_ORDER}
    for row in rows:
        column = row.get(key_field)
        if not column:
            continue
        column = str(column)
        if column not in columns:
            columns.append(column)
        if column in by_model[str(row["model"])]:
            raise ValueError(f"duplicate compact metric {column} for {row['model']}")
        by_model[str(row["model"])][column] = row.get(value_field)
    return columns, [by_model[label] for label in MODEL_ORDER]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    supplied = {
        "internal_qzbc": args.internal_qzbc,
        "tcmeval_official": args.tcmeval_official,
        "tcmeval_controlled": args.tcmeval_controlled,
        "cmb": args.cmb,
        "tcmbench": args.tcmbench,
        "mtcmb": args.mtcmb,
    }
    lock_identity = None
    if args.preregistration_lock:
        lock_identity = validate_static_lock(
            args.preregistration_lock,
            project_root=PROJECT_ROOT,
        )
    extractors = {
        "internal_qzbc": extract_internal,
        "tcmeval_official": extract_tcmeval_official,
        "tcmeval_controlled": extract_tcmeval_controlled,
        "cmb": extract_cmb,
        "tcmbench": extract_tcmbench,
        "mtcmb": extract_mtcmb,
    }
    gate_extractors = {
        "tcmeval_controlled": extract_tcmeval_gates,
        "tcmbench": extract_tcmbench_gates,
        "mtcmb": extract_mtcmb_gates,
    }
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    source_identities = {}
    missing = []
    for label, raw_path in supplied.items():
        if raw_path is None:
            missing.append(label)
            continue
        path = raw_path.resolve(strict=True)
        payload = _load(path)
        if lock_identity and label in {
            "internal_qzbc",
            "tcmeval_official",
            "tcmeval_controlled",
            "cmb",
            "tcmbench",
            "mtcmb",
        }:
            suite_fp = payload.get("suite_fingerprint")
            if not suite_fp:
                raise ValueError(f"{label}: model suite fingerprint is missing")
            if suite_fp != lock_identity["model_suite_fingerprint"]:
                raise ValueError(f"{label}: model suite differs from preregistration")
            # The official TCMEval run predates the new static lock and is
            # rebuilt from validated per-model artifacts by the master
            # launcher.  Every newly generated/derived source must additionally
            # carry the exact lock SHA.
            if label != "tcmeval_official":
                nested_lock = payload.get("preregistration_lock")
                lock_sha = payload.get("preregistration_lock_sha256")
                if lock_sha is None and isinstance(nested_lock, Mapping):
                    lock_sha = nested_lock.get("sha256")
                if lock_sha != lock_identity["sha256"]:
                    raise ValueError(f"{label}: preregistration lock mismatch")
        rows.extend(extractors[label](payload))
        if label in gate_extractors:
            gate_rows.extend(gate_extractors[label](payload))
        source_identities[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": payload.get("schema_version"),
        }
    missing_core = [
        name for name in PRIMARY_REQUIRED_SOURCES if supplied[name] is None
    ]
    if args.require_complete and missing_core:
        raise ValueError(f"required core summaries missing: {missing_core}")
    if args.require_complete:
        observed_gate_benchmarks = {
            str(row["benchmark"]) for row in gate_rows
        }
        if observed_gate_benchmarks != set(PRIMARY_GATE_BENCHMARKS):
            raise ValueError(
                "required public non-inferiority gate summaries missing or duplicated: "
                f"expected {PRIMARY_GATE_BENCHMARKS}, got {sorted(observed_gate_benchmarks)}"
            )
        if len(gate_rows) != len(PRIMARY_GATE_BENCHMARKS) * len(
            PRIMARY_GATE_BASELINES
        ):
            raise ValueError("public non-inferiority gate table is not structurally closed")
    if not rows:
        raise ValueError("no summaries supplied")

    effectiveness_columns, effectiveness = _compact_matrix(
        rows, "effectiveness_display_key", "perturbed_task_success_rate"
    )
    # Official/clean-only rows expose clean TSR as their effectiveness endpoint.
    for model_row in effectiveness:
        model = model_row["model"]
        for row in rows:
            if row["model"] != model or row["perturbed_task_success_rate"] is not None:
                continue
            model_row[row["effectiveness_display_key"]] = row["clean_task_success_rate"]
    robustness_columns, robustness = _compact_matrix(
        rows, "robustness_display_key", "conditional_retention"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "main_table_long.csv", rows)
    _write_csv(output / "main_table_effectiveness.csv", effectiveness)
    _write_csv(output / "main_table_robustness.csv", robustness)
    _write_csv(output / "noninferiority_gates.csv", gate_rows)

    clean_sft_gate_rows = [
        row for row in gate_rows if row["baseline"] == "clean_sft"
    ]
    clean_sft_gate_summary = {
        "baseline": "clean_sft",
        "benchmarks_evaluated": len(clean_sft_gate_rows),
        "benchmarks_joint_passed": sum(
            bool(row["joint_pass"]) for row in clean_sft_gate_rows
        ),
        "benchmark_passes": {
            str(row["benchmark"]): bool(row["joint_pass"])
            for row in clean_sft_gate_rows
        },
        "interpretation": (
            "Count of public benchmark-specific joint non-inferiority gates only; "
            "it is not a cross-benchmark average, ranking score, or total generalization conclusion."
        ),
    }
    gate_payload = {
        "schema_version": "care-opd-public-noninferiority-gates-v1",
        "effect_direction": "CARE-minus-baseline; positive favors CARE",
        "primary_endpoints": list(PRIMARY_GATE_ENDPOINTS),
        "required_baselines": list(PRIMARY_GATE_BASELINES),
        "no_cross_benchmark_average": True,
        "rows": gate_rows,
        "clean_sft_primary_baseline_gate_summary": clean_sft_gate_summary,
    }
    _atomic_json(output / "noninferiority_gates.json", gate_payload)

    markdown = [
        "# CARE-OPD main results table",
        "",
        "All numeric cells are Task Success Rates. No cross-benchmark average is used for ranking or claims.",
        "",
        "## Effectiveness",
        "",
        "| Model | " + " | ".join(effectiveness_columns) + " |",
        "|---|" + "|".join(["---:"] * len(effectiveness_columns)) + "|",
    ]
    for item in effectiveness:
        markdown.append(
            "| " + str(item["model"]) + " | "
            + " | ".join(_format_rate(item.get(column)) for column in effectiveness_columns)
            + " |"
        )
    markdown.extend(["", "## Robustness (conditional retention)", ""])
    if robustness_columns:
        markdown.extend(
            [
                "| Model | " + " | ".join(robustness_columns) + " |",
                "|---|" + "|".join(["---:"] * len(robustness_columns)) + "|",
            ]
        )
        for item in robustness:
            markdown.append(
                "| " + str(item["model"]) + " | "
                + " | ".join(_format_rate(item.get(column)) for column in robustness_columns)
                + " |"
            )
    else:
        markdown.append("No paired robustness summary was supplied.")
    markdown.extend(["", "## Pre-registered public non-inferiority gates", ""])
    if gate_rows:
        markdown.extend(
            [
                "CARE-minus-baseline deltas are percentage points. Each endpoint passes only when "
                "its paired-bootstrap 95% CI lower bound is strictly above the benchmark margin.",
                "",
                "| Benchmark / track | Baseline | Paired units | Margin | Perturbed TSR Δ [95% CI] | Pass | Retention Δ [95% CI] | Pass | Joint |",
                "|---|---|---:|---:|---:|:---:|---:|:---:|:---:|",
            ]
        )
        for item in gate_rows:
            markdown.append(
                "| "
                + f"{item['benchmark']} / {item['track']} | {item['baseline']} | "
                + f"{item['paired_units']} | {100.0 * float(item['margin_absolute']):+.2f} pp | "
                + _format_delta_interval(item, "perturbed_task_success_rate")
                + " | "
                + ("PASS" if item[
                    "perturbed_task_success_rate_care_minus_baseline_noninferiority_pass"
                ] else "FAIL")
                + " | "
                + _format_delta_interval(item, "conditional_retention")
                + " | "
                + ("PASS" if item[
                    "conditional_retention_care_minus_baseline_noninferiority_pass"
                ] else "FAIL")
                + " | "
                + ("PASS" if item["joint_pass"] else "FAIL")
                + " |"
            )
        markdown.extend(
            [
                "",
                "Clean-SFT primary-baseline public gate count: "
                f"**{clean_sft_gate_summary['benchmarks_joint_passed']} / "
                f"{clean_sft_gate_summary['benchmarks_evaluated']}** benchmarks passed. "
                "This is a benchmark-wise gate count, not a cross-benchmark average or a total "
                "generalization conclusion.",
            ]
        )
    else:
        markdown.append("No public non-inferiority gate summary was supplied.")
    markdown.extend(
        ["", (
            "The QZBC row comes from the identity-bound rerun. Its 3,900-request family-clean "
            "view is a retrospective, result-blind contamination-sensitivity analysis; the "
            "full 4,200-row entity holdout remains diagnostic, and historical training-input "
            "bytes were not contemporaneously hashed."
            if source_identities.get("internal_qzbc", {}).get("schema_version")
            == "care-opd-qzbc-grouped-suite-v1"
            else
            "The QZBC grouped-test rows are retrospective legacy results: the old rollout "
            "files do not embed checkpoint fingerprints or the split-integrity manifest. "
            "They remain visible for continuity but are not used for formal OPD attribution."
        ), "", (
            "Public official and Derived tracks are kept visibly separate; MTCMB-Full is never "
            "treated as independent generalization evidence."
            if source_identities.get("internal_qzbc", {}).get("schema_version")
            == "care-opd-qzbc-grouped-suite-v1"
            else
            "An identity-bound QZBC rerun is the intended primary mechanism test. Public "
            "official and Derived tracks are kept visibly separate; MTCMB-Full is never "
            "treated as independent generalization evidence."
        )]
    )
    _atomic_text(output / "main_table.md", "\n".join(markdown) + "\n")
    payload = {
        "schema_version": "care-opd-generalization-main-table-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_order": list(MODEL_ORDER),
        "required_primary_sources": list(PRIMARY_REQUIRED_SOURCES),
        "preregistration_lock": lock_identity,
        "sources": source_identities,
        "missing_optional_sources": missing,
        "no_cross_benchmark_primary_aggregate": True,
        "public_noninferiority_gates": gate_payload,
        "rows": rows,
        "effectiveness_columns": effectiveness_columns,
        "robustness_columns": robustness_columns,
    }
    _atomic_json(output / "main_table.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-qzbc", type=Path)
    parser.add_argument("--tcmeval-official", type=Path)
    parser.add_argument("--tcmeval-controlled", type=Path)
    parser.add_argument("--cmb", type=Path)
    parser.add_argument("--tcmbench", type=Path)
    parser.add_argument("--mtcmb", type=Path)
    parser.add_argument("--preregistration-lock", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "require the five primary sources (internal QZBC, TCMEval official and "
            "controlled, TCMBench, MTCMB); CMB remains an explicitly optional appendix"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    summarize(make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
