from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mechanism_experiments.real_thyroid_external import reaggregate_existing_eval as reaggregate
from mechanism_experiments.real_thyroid_external import thyroid_external_eval as evaluation


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path, *, stale_fingerprint: bool = False) -> Path:
    run_dir = root / "completed_eval"
    private_dir = run_dir / "private"
    public_dir = run_dir / "public"
    label = "care_opd"
    run_fingerprint = "a" * 64
    per_model_fingerprint = evaluation._sha256_bytes(
        evaluation._canonical_json({"run": run_fingerprint, "model": label})
    )
    if stale_fingerprint:
        per_model_fingerprint = "b" * 64
    variants = list(evaluation.FOUR_STATE_VARIANTS)
    rows = []
    for index, variant in enumerate(variants):
        is_boundary = variant in evaluation.BOUNDARY_VARIANTS
        raw_response = f"PRIVATE_RESPONSE_SENTINEL_{variant}"
        assessment = {
            "case_id": f"PRIVATE_CASE_SENTINEL::{variant}",
            "family_id": "PRIVATE_FAMILY_SENTINEL",
            "variant": variant,
            "passed": is_boundary,
            "task_success": is_boundary,
            "report_exact_match": is_boundary,
            "tp": 0,
            "fp": 0 if is_boundary else 1,
            "fn": 0 if is_boundary else 1,
            "expected_field_count": 0 if is_boundary else 1,
            "emitted_section_count": 0 if is_boundary else 1,
            "schema_valid": True,
            "source_valid": True,
            "failure_types": [] if is_boundary else ["unsupported_field", "missing_gold_field"],
            "unsupported_errors": [] if is_boundary else ["PRIVATE_UNSUPPORTED_SENTINEL"],
            "boundary_errors": [],
            "boundary_recovery_valid": True if is_boundary else None,
            "boundary_semantics_valid": True if is_boundary else None,
            "boundary_structure_valid": True if is_boundary else None,
            "controlled_sentinel_free": True,
            "unsafe_clinical_conclusion": False,
        }
        rows.append(
            {
                "schema_version": "care_opd.real_thyroid_external_eval.v2.private_case",
                "run_fingerprint": per_model_fingerprint,
                "model_label": label,
                "index": index,
                "case_id": f"PRIVATE_CASE_SENTINEL::{variant}",
                "family_id": "PRIVATE_FAMILY_SENTINEL",
                "variant": variant,
                "modality": "thyroid_ultrasound",
                "report_group_hmac": "PRIVATE_GROUP_SENTINEL",
                "seed_record_sha256": evaluation._sha256_bytes(variant.encode("utf-8")),
                "raw_response_private": raw_response,
                "response_sha256": evaluation._sha256_bytes(raw_response.encode("utf-8")),
                "generation_error": None,
                "latency_seconds": 0.1,
                "input_tokens": 10,
                "output_tokens": 5,
                "assessment": assessment,
                # Reproduce the v2 stale cache that the new aggregation must ignore.
                "metrics": {
                    **evaluation._case_metrics(assessment),
                    "unsupported": 0,
                },
            }
        )
    private_manifest = {
        "schema_version": "care_opd.real_thyroid_external_eval.v2.run_contract",
        "run_fingerprint": run_fingerprint,
        "input": {
            "case_count": 4,
            "family_count": 1,
            "family_mode": "four_state",
            "qzbc_contract_fingerprint": "qzbc-test",
            "variants": variants,
        },
        "models": [
            {
                "label": label,
                "fingerprint": "1" * 64,
                "file_count": 1,
                "bytes": 1,
                "canonical_path_private": "/private/model",
            }
        ],
        "generation": {},
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "care_opd_qzbc_contract_sha256": "4" * 64,
        "bootstrap": {"replicates": 5, "seed": 17},
        "reference_model": label,
        "reference_models": [label],
    }
    public_manifest = {
        "schema_version": "care_opd.real_thyroid_external_eval.v2.public_manifest",
        "qzbc_contract_fingerprint": "qzbc-test",
        "case_count": 4,
        "family_mode": "four_state",
        "family_count": 1,
        "variants": variants,
        "bootstrap_unit": "family_id",
        "models": [
            {
                "label": label,
                "fingerprint": "1" * 64,
                "file_count": 1,
                "bytes": 1,
            }
        ],
        "generation": {},
        "verifier_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "care_opd_qzbc_contract_sha256": "4" * 64,
        "bootstrap": {"replicates": 5, "seed": 17},
        "reference_model": label,
        "reference_models": [label],
        "private_binding_disclosure": "private bindings omitted",
    }
    public_manifest["public_protocol_fingerprint"] = evaluation._sha256_bytes(
        evaluation._canonical_json(public_manifest)
    )
    _write_json(private_dir / "run_manifest.private.json", private_manifest)
    cases_path = private_dir / f"{label}.cases.private.jsonl"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(public_dir / "run_manifest.json", public_manifest)
    return run_dir


class ReaggregateExistingEvaluationTest(unittest.TestCase):
    def test_reaggregates_without_generation_and_preserves_private_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _fixture(Path(temporary))
            private_hashes_before = {
                path.name: evaluation._sha256_file(path)
                for path in (run_dir / "private").iterdir()
                if path.is_file()
            }
            with mock.patch.object(
                evaluation,
                "_build_generation_runtime",
                side_effect=AssertionError("reaggregation must not load a model"),
            ):
                result = reaggregate.reaggregate_existing_run(
                    run_dir=run_dir,
                    acknowledge_private_clinical_data=True,
                )
            self.assertEqual(result["status"], "REAGGREGATED")
            output = run_dir / reaggregate.DEFAULT_PUBLIC_DIRNAME
            with (output / "secondary_field_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                overall = next(
                    row
                    for row in csv.DictReader(handle)
                    if row["model"] == "care_opd" and row["stratum"] == "overall"
                )
            self.assertEqual(float(overall["field_micro_f1"]), 0.0)
            self.assertEqual(int(overall["unsupported_fields"]), 2)
            self.assertEqual(float(overall["unsupported_field_rate"]), 1.0)
            with (output / "three_axis_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                axes = next(
                    row
                    for row in csv.DictReader(handle)
                    if row["model"] == "care_opd" and row["stratum"] == "overall"
                )
            self.assertEqual(axes["content_scope"], "clean+bad_history")
            self.assertEqual(int(axes["content_eligible_reports"]), 2)
            self.assertEqual(int(axes["content_evaluated_reports"]), 0)
            self.assertEqual(axes["content_micro_f1"], "")
            self.assertEqual(int(axes["slot_schema_valid_eligible_reports"]), 2)
            self.assertEqual(int(axes["slot_schema_valid_evaluated_reports"]), 0)
            self.assertEqual(axes["slot_schema_valid_rate"], "")
            combined_public = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.iterdir()
                if path.is_file()
            )
            for private_sentinel in (
                "PRIVATE_RESPONSE_SENTINEL",
                "PRIVATE_CASE_SENTINEL",
                "PRIVATE_FAMILY_SENTINEL",
                "PRIVATE_GROUP_SENTINEL",
                "PRIVATE_UNSUPPORTED_SENTINEL",
                "/private/model",
                "a" * 64,
            ):
                self.assertNotIn(private_sentinel, combined_public)
            private_hashes_after = {
                path.name: evaluation._sha256_file(path)
                for path in (run_dir / "private").iterdir()
                if path.is_file()
            }
            self.assertEqual(private_hashes_before, private_hashes_after)
            with self.assertRaises(FileExistsError):
                reaggregate.reaggregate_existing_run(
                    run_dir=run_dir,
                    acknowledge_private_clinical_data=True,
                )

    def test_rejects_stale_per_model_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _fixture(Path(temporary), stale_fingerprint=True)
            with self.assertRaisesRegex(
                evaluation.EvaluationContractError, "stale private row fingerprint"
            ):
                reaggregate.reaggregate_existing_run(
                    run_dir=run_dir,
                    acknowledge_private_clinical_data=True,
                )

    def test_requires_private_data_acknowledgement_before_reading(self) -> None:
        with self.assertRaisesRegex(
            evaluation.EvaluationContractError,
            "acknowledge-private-clinical-data",
        ):
            reaggregate.reaggregate_existing_run(run_dir=Path("missing"))


if __name__ == "__main__":
    unittest.main()
