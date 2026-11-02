from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "competition_evidence" / "real_external_stats.py"
SPEC = importlib.util.spec_from_file_location("real_external_stats", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stats = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stats
SPEC.loader.exec_module(stats)


TRACKS = {
    "eval_direct_gold_structured_six_models_v5/private": "structured_evidence_faithfulness",
    "eval_direct_gold_raw_six_models_v5/private": "raw_strict_extraction_challenge",
}
VARIANTS = ("clean", "bad_history", "source_outage", "evidence_conflict")


def _assessment(*, strong: bool, family: str, variant: str, track: str) -> dict:
    success = strong or variant == "source_outage"
    tp = 2 if strong else 1
    fp = 0 if strong else 1
    fn = 0 if strong else 1
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator
    return {
        "family_id": family,
        "variant": variant,
        "evaluation_track": track,
        "report_exact_match": success,
        "state_task_success": success,
        "field_precision": tp / (tp + fp),
        "field_recall": tp / (tp + fn),
        "field_f1": f1,
        "field_accuracy": f1,
        "content_f1": f1,
        "grounded_content_f1": f1,
        "content_complete": success,
        "grounded_content_complete": success,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        # Nested case material must never be copied into an output.
        "field_results": [{"value": "DO_NOT_RELEASE"}],
    }


def _write_fixture(root: Path) -> None:
    for relative, track in TRACKS.items():
        directory = root / relative
        directory.mkdir(parents=True)
        for model, strong in (("care_opd", True), ("clean_sft", False)):
            path = directory / f"{model}.cases.private.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for family_index in range(2):
                    family = f"family-{family_index}"
                    for variant in VARIANTS:
                        assessment = _assessment(
                            strong=strong,
                            family=family,
                            variant=variant,
                            track=track,
                        )
                        row = {
                            "model_label": model,
                            "evaluation_track": track,
                            "family_id": family,
                            "variant": variant,
                            "assessment": assessment,
                            "metrics": {
                                "tp": assessment["tp"],
                                "fp": assessment["fp"],
                                "fn": assessment["fn"],
                                "report_exact": assessment["report_exact_match"],
                                "task_success": assessment["state_task_success"],
                            },
                            "raw_response_private": "secret-response",
                            "input_text": "secret-record",
                        }
                        stream.write(json.dumps(row) + "\n")


def test_aggregate_macro_micro_exact_and_task_success(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    paths = stats.discover_inputs(tmp_path)
    cases, _ = stats.load_cases(paths)
    rows = stats.build_report_level_rows(cases)
    clean = next(
        row
        for row in rows
        if row["model"] == "care_opd"
        and row["track"] == "structured_evidence_faithfulness"
        and row["variant"] == "clean"
    )
    assert clean["families"] == 2
    assert clean["reports"] == 2
    assert clean["report_exact_rate"] == pytest.approx(1.0)
    assert clean["task_success_rate"] == pytest.approx(1.0)
    assert clean["macro_field_f1"] == pytest.approx(1.0)
    assert clean["micro_field_f1"] == pytest.approx(1.0)
    assert clean["field_tp"] == 4


def test_cli_run_writes_deterministic_fixed_matrix_without_private_content(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    first = tmp_path / "out-one"
    second = tmp_path / "out-two"
    stats.run(
        input_root=tmp_path,
        output_dir=first,
        bootstrap_replicates=100,
        bootstrap_seed=17,
        reference_models=["clean_sft"],
        analysis_mode=stats.ANALYSIS_MODE_INDEPENDENT_STABLE,
    )
    stats.run(
        input_root=tmp_path,
        output_dir=second,
        bootstrap_replicates=100,
        bootstrap_seed=17,
        reference_models=["clean_sft"],
        analysis_mode=stats.ANALYSIS_MODE_INDEPENDENT_STABLE,
    )
    expected = {
        "real_external_report_metrics.csv",
        "care_vs_all_references.csv",
        "cohort_audit.csv",
        "real_external_stats.json",
        "SUMMARY.md",
    }
    assert {path.name for path in first.iterdir()} == expected
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        lowered = (first / name).read_text(encoding="utf-8").lower()
        assert "secret-response" not in lowered
        assert "secret-record" not in lowered
        assert "do_not_release" not in lowered

    with (first / "care_vs_all_references.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # 2 tracks x 5 scopes x 13 fixed metrics x 1 reference.
    assert len(rows) == 2 * 5 * len(stats.METRIC_NAMES)
    headline = next(
        row
        for row in rows
        if row["track"] == "structured_evidence_faithfulness"
        and row["variant"] == "clean"
        and row["metric"] == "micro_field_f1"
    )
    assert float(headline["delta_candidate_minus_reference"]) > 0
    assert int(headline["paired_families"]) == 2
    assert int(headline["valid_bootstrap_replicates"]) == 100

    payload = json.loads((first / "real_external_stats.json").read_text(encoding="utf-8"))
    audit = payload["cohort_audit"]
    assert audit["structured_eligible"] == 2
    assert audit["raw_strict_eligible"] == 2
    assert audit["track_intersection"] == 2
    assert audit["raw_strict_is_structured_subset"] is True
    assert payload["analysis"]["analysis_method"] == stats.INDEPENDENT_ANALYSIS_METHOD
    assert payload["analysis"]["authority"] == stats.SUPPLEMENTAL_REANALYSIS


def test_privacy_scan_is_fail_closed() -> None:
    forbidden_key = "raw" + "_response_private"
    with pytest.raises(stats.PrivacyError):
        stats._assert_public_safe({forbidden_key: "x"})
    with pytest.raises(stats.PrivacyError):
        stats._assert_public_safe({"safe": "prefix input" + "_text suffix"})


def test_pairing_mismatch_fails(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    paths = stats.discover_inputs(tmp_path)
    cases, _ = stats.load_cases(paths)
    trimmed = [
        case
        for case in cases
        if not (
            case.model == "clean_sft"
            and case.track == "structured_evidence_faithfulness"
            and case.family_id == "family-1"
            and case.variant == "clean"
        )
    ]
    with pytest.raises(stats.StatsContractError, match="unpaired models"):
        stats.build_fixed_matrix(
            trimmed,
            candidate_model="care_opd",
            reference_models=["clean_sft"],
            replicates=10,
            seed=stats.FROZEN_BOOTSTRAP_SEED,
            analysis_mode=stats.ANALYSIS_MODE_LEGACY_FROZEN,
        )


def test_legacy_frozen_is_seed_locked_and_records_authority(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    output = tmp_path / "legacy"
    payload = stats.run(
        input_root=tmp_path,
        output_dir=output,
        bootstrap_replicates=100,
        reference_models=["clean_sft"],
    )
    analysis = payload["analysis"]
    assert analysis["bootstrap_base_seed"] == stats.FROZEN_BOOTSTRAP_SEED
    assert analysis["analysis_method"] == stats.LEGACY_ANALYSIS_METHOD
    assert analysis["authority"] == stats.AUTHORITATIVE_FROZEN
    assert "same report-family draws" in analysis["draw_reuse"]
    rows = payload["care_vs_references"]
    assert rows
    assert {row["bootstrap_seed"] for row in rows} == {stats.FROZEN_BOOTSTRAP_SEED}
    assert {row["analysis_method"] for row in rows} == {stats.LEGACY_ANALYSIS_METHOD}
    assert {row["authority"] for row in rows} == {stats.AUTHORITATIVE_FROZEN}

    with pytest.raises(stats.StatsContractError, match="bound to bootstrap seed"):
        stats.run(
            input_root=tmp_path,
            output_dir=tmp_path / "invalid",
            bootstrap_replicates=10,
            bootstrap_seed=17,
            reference_models=["clean_sft"],
        )


def test_legacy_frozen_reproduces_public_headline_intervals() -> None:
    workspace = SCRIPT.parents[2]
    structured_private = workspace / "eval_direct_gold_structured_six_models_v5" / "private"
    raw_private = workspace / "eval_direct_gold_raw_six_models_v5" / "private"
    public_root = (
        workspace
        / "competition_project_document"
        / "formal"
        / "competition_submission_v7"
        / "supporting_public_v7"
        / "03_真实场景外测"
    )
    if not structured_private.is_dir() or not raw_private.is_dir() or not public_root.is_dir():
        pytest.skip("private frozen real-evaluation artifacts are not available")

    paths = stats.discover_inputs(workspace)
    cases, _ = stats.load_cases(paths)
    matrix, _ = stats.build_fixed_matrix(
        cases,
        candidate_model="care_opd",
        reference_models=["clean_sft", "gkd_full"],
        replicates=10_000,
        seed=stats.FROZEN_BOOTSTRAP_SEED,
        analysis_mode=stats.ANALYSIS_MODE_LEGACY_FROZEN,
    )

    def expanded(track: str, reference: str, variant: str, metric: str) -> dict:
        return next(
            row
            for row in matrix
            if row["track"] == track
            and row["reference_model"] == reference
            and row["variant"] == variant
            and row["metric"] == metric
        )

    metric_map = {
        "clean_task_success_rate": ("clean", "task_success_rate"),
        "bad_history_task_success_rate": ("bad_history", "task_success_rate"),
        "source_outage_task_success_rate": ("source_outage", "task_success_rate"),
        "evidence_conflict_task_success_rate": ("evidence_conflict", "task_success_rate"),
        "clean_field_micro_precision": ("clean", "micro_field_precision"),
        "clean_field_micro_recall": ("clean", "micro_field_recall"),
        "clean_field_micro_f1": ("clean", "micro_field_f1"),
        "bad_history_field_micro_precision": ("bad_history", "micro_field_precision"),
        "bad_history_field_micro_recall": ("bad_history", "micro_field_recall"),
        "bad_history_field_micro_f1": ("bad_history", "micro_field_f1"),
        "secondary_field_micro_precision": (stats.ALL_STATES, "micro_field_precision"),
        "secondary_field_micro_recall": (stats.ALL_STATES, "micro_field_recall"),
        "secondary_field_micro_f1": (stats.ALL_STATES, "micro_field_f1"),
    }
    checked = 0
    for public_track, track in (
        ("structured_public", "structured_evidence_faithfulness"),
        ("raw_strict_public", "raw_strict_extraction_challenge"),
    ):
        with (public_root / public_track / "paired_comparisons.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            frozen_rows = list(csv.DictReader(stream))
        for frozen in frozen_rows:
            if (
                frozen["candidate_model"] != "care_opd"
                or frozen["reference_model"] not in {"clean_sft", "gkd_full"}
                or frozen["stratum"] != "overall"
                or frozen["metric"] not in metric_map
            ):
                continue
            variant, metric = metric_map[frozen["metric"]]
            actual = expanded(track, frozen["reference_model"], variant, metric)
            for field in (
                "reference_value",
                "candidate_value",
                "delta_candidate_minus_reference",
                "ci95_lower",
                "ci95_upper",
            ):
                assert actual[field] == float(frozen[field])
            assert actual["valid_bootstrap_replicates"] == int(
                frozen["valid_bootstrap_replicates"]
            )
            checked += 1
    assert checked == 52
