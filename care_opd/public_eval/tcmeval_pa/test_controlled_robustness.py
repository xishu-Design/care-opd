#!/usr/bin/env python3
"""CPU-only tests for the paired TCMEval-PA controlled protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_controlled_robustness as controlled  # noqa: E402
import evaluate_tcmeval_pa as official  # noqa: E402


class MappingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = HERE / "data" / "TCMEval-PA.xlsx"
        cls.questions = official.load_questions(cls.data)

    def write_valid_lock(self, path: Path) -> None:
        project_root = controlled.HERE.parents[1]
        inventory = []
        for relative in controlled.TCMEVAL_CONTROLLED_PREREGISTERED_PATHS:
            target = project_root / relative
            inventory.append(
                {
                    "path": relative,
                    "bytes": target.stat().st_size,
                    "sha256": official.sha256_file(target),
                }
            )
        contract = {
            "schema_version": "care-opd-public-preregistration-lock-v1",
            "scope": "test",
            "timing_disclosure": {
                "tcmeval_pa_controlled_predictions_observed": False
            },
            "fixed_model_order": list(controlled.EXPECTED_LABELS),
            "model_suite": {
                "suite_fingerprint": official.canonical_hash(
                    [(label, "0" * 64) for label in controlled.EXPECTED_LABELS]
                ),
                "models": [
                    {"label": label, "model_fingerprint": "0" * 64}
                    for label in controlled.EXPECTED_LABELS
                ],
            },
            "locked_files": inventory,
        }
        path.write_text(
            json.dumps(
                {
                    **contract,
                    "contract_fingerprint": official.canonical_hash(contract),
                    "created_at_utc": "2026-08-08T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

    def test_every_mapping_is_deterministic_bijective_and_nonidentity(self) -> None:
        first = [controlled.mapping_item(question) for question in self.questions]
        second = [controlled.mapping_item(question) for question in self.questions]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 328)
        for question, item in zip(self.questions, first, strict=True):
            allowed = list(question.allowed_options)
            order = item["displayed_canonical_order"]
            self.assertCountEqual(order, allowed)
            self.assertNotEqual(order, allowed)
            display_to_canonical = item["display_to_canonical"]
            canonical_to_display = item["canonical_to_display"]
            self.assertEqual(set(display_to_canonical), set(allowed))
            self.assertEqual(set(display_to_canonical.values()), set(allowed))
            for display, canonical in display_to_canonical.items():
                self.assertEqual(canonical_to_display[canonical], display)

    def test_multiline_b1_remains_option_content_not_a_label(self) -> None:
        question = next(item for item in self.questions if item.id == "294")
        item = controlled.mapping_item(question)
        perturbed = controlled.candidate_answers_for_order(
            question, item["displayed_canonical_order"]
        )
        parsed = official._option_inventory(perturbed)
        self.assertEqual(parsed, ("A", "B", "C", "D", "E"))
        self.assertIn("B1", perturbed.replace("\n", ""))

    def test_display_prediction_maps_back_to_canonical(self) -> None:
        mapping = {"A": "D", "B": "A", "C": "E", "D": "B", "E": "C"}
        self.assertEqual(
            controlled.map_display_prediction_to_canonical(("A", "D"), mapping),
            ("B", "D"),
        )

    def test_frozen_manifest_records_lock_and_schema_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "PREREGISTRATION_LOCK.json"
            self.write_valid_lock(lock)
            output = root / "output"
            manifest = controlled.freeze_mapping_manifest(self.data, output, lock)
            self.assertEqual(manifest["paired_units"], 328)
            self.assertEqual(
                manifest["preregistration_lock_sha256"],
                hashlib.sha256(lock.read_bytes()).hexdigest(),
            )
            schema_snapshot = output / "UNIFIED_METRIC_SCHEMA.json"
            self.assertEqual(
                manifest["metric_schema_sha256"],
                hashlib.sha256(schema_snapshot.read_bytes()).hexdigest(),
            )
            analysis_snapshot = output / "ANALYSIS_SPEC.json"
            self.assertEqual(
                manifest["analysis_spec_sha256"],
                hashlib.sha256(analysis_snapshot.read_bytes()).hexdigest(),
            )
            first_bytes = (output / "mapping_manifest.json").read_bytes()
            reused = controlled.freeze_mapping_manifest(self.data, output, lock)
            self.assertEqual(manifest, reused)
            self.assertEqual(first_bytes, (output / "mapping_manifest.json").read_bytes())
            lock.write_text('{"status":"changed"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                controlled.freeze_mapping_manifest(self.data, output, lock)

    def test_missing_preregistration_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                controlled.freeze_mapping_manifest(
                    self.data, root / "output", root / "missing.json"
                )

    def test_normalized_stem_cluster_count_is_314(self) -> None:
        clusters = {
            controlled.normalized_stem_cluster(question.question)
            for question in self.questions
        }
        self.assertEqual(len(clusters), 314)

    def test_four_gpu_execution_topology_is_fingerprinted(self) -> None:
        args = argparse.Namespace(
            num_workers=controlled.DEFAULT_NUM_WORKERS,
            devices=",".join(controlled.DEFAULT_DEVICES),
        )
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
            controlled._normalize_execution_topology(args)
        contract = controlled._execution_contract(args)
        self.assertEqual(
            contract,
            {
                "num_workers": 4,
                "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                "visible_physical_devices": ["4", "5", "6", "7"],
            },
        )
        changed = dict(contract, visible_physical_devices=["0", "1", "2", "3"])
        self.assertNotEqual(
            official.canonical_hash(contract), official.canonical_hash(changed)
        )


def synthetic_row(
    index: int,
    *,
    clean: bool,
    perturbed: bool,
    consistent: bool,
    cluster: str | None = None,
) -> dict:
    if clean and perturbed:
        flip = "both_correct"
    elif clean:
        flip = "clean_correct_to_wrong"
    elif perturbed:
        flip = "clean_wrong_to_correct"
    else:
        flip = "both_wrong"
    return {
        "index": index,
        "id": str(index),
        "normalized_stem_cluster": cluster or f"cluster-{index}",
        "clean": {"task_success": clean},
        "perturbed": {"task_success": perturbed},
        "flip": flip,
        "canonical_prediction_consistent": consistent,
    }


class UnifiedMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            synthetic_row(1, clean=True, perturbed=True, consistent=True),
            synthetic_row(2, clean=True, perturbed=False, consistent=False),
            synthetic_row(3, clean=False, perturbed=True, consistent=False),
            synthetic_row(4, clean=False, perturbed=False, consistent=True),
        ]

    def test_exact_unified_metric_fields_and_values(self) -> None:
        metrics = controlled.unified_metrics(self.rows)
        self.assertEqual(tuple(metrics), controlled.UNIFIED_FIELDS)
        persisted = json.loads(json.dumps(metrics, sort_keys=True))
        self.assertEqual(set(persisted), set(controlled.UNIFIED_FIELDS))
        self.assertNotEqual(tuple(persisted), controlled.UNIFIED_FIELDS)
        self.assertEqual(metrics["clean_task_success_rate"], 0.5)
        self.assertEqual(metrics["perturbed_task_success_rate"], 0.5)
        self.assertEqual(metrics["conditional_retention"], 0.5)
        self.assertEqual(metrics["regression_rate"], 0.5)
        self.assertEqual(metrics["clean_correct_perturbed_wrong"], 1)
        self.assertEqual(metrics["clean_wrong_perturbed_correct"], 1)
        self.assertEqual(metrics["robustness_drop_pp"], 0.0)
        self.assertEqual(metrics["prediction_consistency"], 0.5)
        self.assertEqual(metrics["worst_perturbation_tsr"], 0.5)
        self.assertEqual(metrics["paired_units"], 4)

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        first = controlled.cluster_bootstrap_intervals(
            self.rows, replicates=200, seed=17
        )
        second = controlled.cluster_bootstrap_intervals(
            self.rows, replicates=200, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 4)
        self.assertEqual(
            set(first["intervals"]),
            {
                "clean_task_success_rate",
                "perturbed_task_success_rate",
                "conditional_retention",
                "regression_rate",
            },
        )

    def test_bootstrap_clusters_repeated_stems(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["normalized_stem_cluster"] = "repeated"
        rows[1]["normalized_stem_cluster"] = "repeated"
        result = controlled.cluster_bootstrap_intervals(rows, replicates=50, seed=1)
        self.assertEqual(result["cluster_count"], 3)

    def test_paired_comparison_has_required_deltas_and_flip_test(self) -> None:
        baseline = copy.deepcopy(self.rows)
        baseline[0] = synthetic_row(
            1, clean=True, perturbed=False, consistent=False
        )
        result = controlled._paired_comparison(
            self.rows, baseline, "clean_sft", replicates=200, seed=5
        )
        self.assertEqual(result["paired_units"], 4)
        self.assertEqual(
            set(result["delta_and_95_ci"]),
            {
                "perturbed_task_success_rate",
                "conditional_retention",
                "regression_rate",
            },
        )
        self.assertGreater(
            result["delta_and_95_ci"]["perturbed_task_success_rate"]["delta"], 0
        )
        self.assertEqual(result["noninferiority"]["margin_absolute"], -0.02)
        self.assertEqual(
            set(result["noninferiority"]["endpoints"]),
            {"perturbed_task_success_rate", "conditional_retention"},
        )
        self.assertIsInstance(result["noninferiority"]["joint_pass"], bool)
        self.assertIn(
            "two_sided_exact_p",
            result[
                "exploratory_exact_mcnemar_perturbed_task_success_rate"
            ],
        )

    def test_metric_schema_is_exact_and_hashable(self) -> None:
        schema = controlled._validate_metric_schema()
        self.assertEqual(
            tuple(schema["required_fields"]), controlled.UNIFIED_FIELDS
        )
        analysis = controlled._validate_analysis_spec()
        self.assertEqual(analysis["confidence_interval"]["replicates"], 10_000)
        self.assertEqual(analysis["confidence_interval"]["seed"], 20260808)
        self.assertEqual(
            analysis["comparison"]["noninferiority_margin_absolute"], -0.02
        )


class SuiteReuseTest(unittest.TestCase):
    def test_six_model_suite_is_validated_and_copied_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for label in controlled.EXPECTED_LABELS:
                model = (root / "models" / label).resolve()
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}\n", encoding="utf-8")
                (model / "tokenizer_config.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (model / "model.safetensors").write_bytes(b"test-weight")
                if label != "frozen":
                    (model / "export_manifest.json").write_text(
                        "{}\n", encoding="utf-8"
                    )
                with redirect_stdout(io.StringIO()):
                    entry = dict(
                        official.fingerprint_model(
                            model, require_export_manifest=label != "frozen"
                        )
                    )
                entry["label"] = label
                entries.append(entry)
            suite = {
                "schema_version": "tcmeval-pa-model-suite-v1",
                "models": entries,
                "suite_fingerprint": official.canonical_hash(
                    [
                        (entry["label"], entry["model_fingerprint"])
                        for entry in entries
                    ]
                ),
            }
            source = root / "source_suite.json"
            source.write_text(
                json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            source_before = source.read_bytes()
            destination = root / "controlled" / "model_suite.json"
            provenance = controlled.reuse_suite(source, destination)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(destination.read_bytes(), source_before)
            self.assertEqual(
                provenance["suite_fingerprint"], suite["suite_fingerprint"]
            )
            self.assertEqual(
                tuple(
                    item["label"]
                    for item in controlled.validate_suite(destination)["models"]
                ),
                controlled.EXPECTED_LABELS,
            )


if __name__ == "__main__":
    unittest.main()
