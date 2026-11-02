from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from competition_evidence.prepare_blind_review import prepare
from competition_evidence.summarize_blind_review import summarize


TRACK = "structured_evidence_faithfulness"
VARIANTS = ("clean", "source_outage")
MODELS = ("care_opd", "clean_sft", "gkd_full")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fixture_inputs(root: Path) -> tuple[Path, dict[str, Path]]:
    locked_rows: list[dict] = []
    model_rows: dict[str, list[dict]] = {model: [] for model in MODELS}
    for family_index in range(2):
        family = f"family-hmac-{family_index}"
        for variant in VARIANTS:
            case_id = f"case-{family_index}-{variant}"
            locked_rows.append(
                {
                    "case_id": case_id,
                    "id": case_id,
                    "family_id": f"family-{family_index}",
                    "report_group_hmac": family,
                    "source_id": f"source-{family_index}",
                    "evaluation_track": TRACK,
                    "variant": variant,
                    "modality": "ultrasound",
                    "request": {
                        "input_text": f"synthetic private report {family_index}",
                        "evidence_packet": {"source": f"synthetic evidence {family_index}"},
                    },
                }
            )
            for model in MODELS:
                # The candidate text intentionally carries no method self-identification.
                # Different hashes preserve distinct model outputs without defeating blinding.
                response = f"synthetic candidate {MODELS.index(model)} for {case_id}"
                model_rows[model].append(
                    {
                        "case_id": case_id,
                        "family_id": f"family-{family_index}",
                        "report_group_hmac": family,
                        "evaluation_track": TRACK,
                        "variant": variant,
                        "model_label": model,
                        "raw_response_private": response,
                        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                        "generation_error": None,
                    }
                )
    locked = root / "locked.private.jsonl"
    _write_jsonl(locked, locked_rows)
    cases: dict[str, Path] = {}
    for model, rows in model_rows.items():
        path = root / f"{model}.cases.private.jsonl"
        _write_jsonl(path, rows)
        cases[model] = path
    return locked, cases


def _prepare_args(root: Path, locked: Path, cases: dict[str, Path]) -> argparse.Namespace:
    return argparse.Namespace(
        locked=locked,
        case=[f"{model}={path}" for model, path in cases.items()],
        track=TRACK,
        variant=list(VARIANTS),
        sample_mode="full",
        sample_size=None,
        sample_seed=17,
        blind_seed=29,
        reviewers=["clinician_a,clinician_b"],
        output_private=root / "private",
        output_public=root / "public",
    )


def _fill_scores(template: Path, output: Path, disagree: bool = False) -> None:
    with template.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        candidate_number = int(row["candidate_code"][1:])
        # Candidate codes are deliberately per-unit random and are not model IDs;
        # these deterministic synthetic scores only exercise the statistic code.
        row.update(
            {
                "factual_fidelity_0_4": "3",
                "critical_omission_0_1": "0",
                "unsupported_major_claim_0_1": "0",
                "safe_correct_0_1": "1",
                "harm_severity_0_3": "0",
                "role_fit_0_4": "3",
                "evidence_handling_0_4": "3",
                "review_burden_0_4": "1",
                "overall_quality_1_5": str(6 - candidate_number),
                "completed_at_utc": "2026-08-24T00:00:00Z",
            }
        )
    if disagree:
        target = next(row for row in rows if row["reviewer_id"] == "clinician_b")
        target["safe_correct_0_1"] = "0"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_is_blind_private_and_reproducible(tmp_path: Path) -> None:
    locked, cases = _fixture_inputs(tmp_path)
    first = prepare(_prepare_args(tmp_path / "run1", locked, cases))
    second = prepare(_prepare_args(tmp_path / "run2", locked, cases))

    assert first["queue"].read_bytes() == second["queue"].read_bytes()
    assert first["key"].read_bytes() == second["key"].read_bytes()
    queue_text = first["queue"].read_text(encoding="utf-8")
    assert "synthetic private report" in queue_text
    assert "model_label_private" not in queue_text
    assert "clean_sft" not in queue_text
    assert "gkd_full" not in queue_text

    mapping_text = first["key"].read_text(encoding="utf-8")
    assert all(model in mapping_text for model in MODELS)
    public_text = first["public_manifest"].read_text(encoding="utf-8")
    assert "synthetic private report" not in public_text
    assert "source-0" not in public_text
    assert "family-hmac" not in public_text
    assert "clean_sft" not in public_text
    assert "gkd_full" not in public_text
    assert '"models_private"' not in public_text
    manifest = json.loads(public_text)
    assert manifest["data_scope"]["selected_report_group_count"] == 2
    assert manifest["data_scope"]["review_unit_count"] == 4
    assert manifest["data_scope"]["candidate_method_count"] == 3


def test_preregistered_sample_selects_whole_report_groups(tmp_path: Path) -> None:
    locked, cases = _fixture_inputs(tmp_path)
    args = _prepare_args(tmp_path / "sampled", locked, cases)
    args.sample_mode = "preregistered"
    args.sample_size = 1
    outputs = prepare(args)
    private_manifest = json.loads(outputs["private_manifest"].read_text(encoding="utf-8"))
    assert private_manifest["sampling"]["selected_family_count"] == 1
    assert private_manifest["unit_count"] == len(VARIANTS)


def test_summary_computes_aggregate_pairwise_and_agreement(tmp_path: Path) -> None:
    locked, cases = _fixture_inputs(tmp_path)
    prepared = prepare(_prepare_args(tmp_path / "prepared", locked, cases))
    completed = tmp_path / "completed.private.csv"
    _fill_scores(prepared["scores"], completed)
    args = argparse.Namespace(
        manifest=prepared["private_manifest"],
        mapping=prepared["key"],
        scores=completed,
        adjudication=None,
        primary_model="care_opd",
        comparator=["clean_sft", "gkd_full"],
        bootstrap_replicates=250,
        bootstrap_seed=31,
        output_private=tmp_path / "summary_private",
        output_public=tmp_path / "summary_public",
    )
    outputs = summarize(args)
    summary = json.loads(outputs["summary_public"].read_text(encoding="utf-8"))
    assert summary["status"] == "FINAL"
    assert len(summary["model_aggregates"]) == 3
    assert len(summary["pairwise_comparisons"]) == 2
    assert all(row["resolved_binary_pair_count"] == 4 for row in summary["pairwise_comparisons"])
    assert all(row["two_sided_exact_p"] == 1.0 for row in summary["pairwise_comparisons"])
    assert all(row["exact_agreement"] == 1.0 for row in summary["inter_rater_agreement"])
    public_text = outputs["summary_public"].read_text(encoding="utf-8")
    assert "review_unit_id" not in public_text
    assert "family-hmac" not in public_text
    with outputs["adjudication_queue_private"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        assert list(csv.DictReader(handle)) == []


def test_binary_disagreement_is_not_fabricated_and_is_queued(tmp_path: Path) -> None:
    locked, cases = _fixture_inputs(tmp_path)
    prepared = prepare(_prepare_args(tmp_path / "prepared", locked, cases))
    completed = tmp_path / "completed.private.csv"
    _fill_scores(prepared["scores"], completed, disagree=True)
    args = argparse.Namespace(
        manifest=prepared["private_manifest"],
        mapping=prepared["key"],
        scores=completed,
        adjudication=None,
        primary_model="care_opd",
        comparator=["clean_sft"],
        bootstrap_replicates=100,
        bootstrap_seed=37,
        output_private=tmp_path / "summary_private",
        output_public=tmp_path / "summary_public",
    )
    outputs = summarize(args)
    summary = json.loads(outputs["summary_public"].read_text(encoding="utf-8"))
    assert summary["status"] == "ADJUDICATION_REQUIRED"
    assert summary["analysis"]["unresolved_binary_disagreement_count"] == 1
    with outputs["adjudication_queue_private"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["field"] == "safe_correct_0_1"
    assert rows[0]["final_score"] == ""


def test_blank_template_cannot_be_summarized(tmp_path: Path) -> None:
    locked, cases = _fixture_inputs(tmp_path)
    prepared = prepare(_prepare_args(tmp_path / "prepared", locked, cases))
    args = argparse.Namespace(
        manifest=prepared["private_manifest"],
        mapping=prepared["key"],
        scores=prepared["scores"],
        adjudication=None,
        primary_model="care_opd",
        comparator=["clean_sft"],
        bootstrap_replicates=10,
        bootstrap_seed=1,
        output_private=tmp_path / "summary_private",
        output_public=tmp_path / "summary_public",
    )
    with pytest.raises(ValueError, match="blank required score"):
        summarize(args)
