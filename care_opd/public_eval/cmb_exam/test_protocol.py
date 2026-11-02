#!/usr/bin/env python3
"""CPU-only regression checks for the CMB-Exam evaluation contract."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_cmb_exam as evaluation  # noqa: E402
import prepare_data  # noqa: E402


class DatasetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = HERE / "data"
        cls.questions = evaluation.load_questions(cls.data)
        cls.all_items = evaluation.load_evaluation_items(cls.data)

    def test_official_cardinality_and_hierarchy(self) -> None:
        self.assertEqual(len(self.questions), 11200)
        self.assertEqual(len({row.id for row in self.questions}), 11200)
        self.assertEqual(
            Counter(row.question_type for row in self.questions),
            Counter({"single": 10010, "multiple": 1190}),
        )
        self.assertEqual(len({row.class_key for row in self.questions}), 28)
        self.assertEqual(
            set(Counter(row.class_key for row in self.questions).values()), {400}
        )

    def test_official_type_is_not_gold_cardinality(self) -> None:
        self.assertEqual(
            Counter(row.gold_cardinality_type for row in self.questions),
            Counter({"single": 10035, "multiple": 1165}),
        )
        mismatched = [
            row
            for row in self.questions
            if row.question_type != row.gold_cardinality_type
        ]
        self.assertEqual(len(mismatched), 25)
        self.assertTrue(
            all(row.question_type == "multiple" for row in mismatched)
        )
        self.assertEqual(
            evaluation.prompt_spec()["question_type_source"],
            "public CMB answer-file metadata (never gold cardinality)",
        )

    def test_c_type_and_blank_option_source_quirks_are_explicit(self) -> None:
        self.assertEqual(
            sum(row.raw_question_type == "C型选择题" for row in self.questions), 11
        )
        # Upstream stores blank trailing option slots. They are not answer
        # choices and never contain a gold label; ID 13 is a representative.
        item = next(row for row in self.questions if row.id == "13")
        self.assertEqual(item.allowed_options, ("A", "B", "C", "D"))
        self.assertEqual(item.gold_options, ("A", "B", "C"))

    def test_overlap_manifest_was_rederived_from_report(self) -> None:
        manifest = json.loads(
            (self.data / prepare_data.EXCLUSION_FILENAME).read_text(encoding="utf-8")
        )
        prepare_data.verify_exclusion_manifest(
            manifest,
            self.data / prepare_data.QUESTION_FILENAME,
            self.data / prepare_data.OVERLAP_REPORT_FILENAME,
        )
        self.assertEqual(manifest["excluded_n"], 4)
        self.assertEqual(manifest["overlap_clean_n"], 11196)
        self.assertEqual(
            sum(not row.overlap_excluded for row in self.questions), 11196
        )

    def test_controlled_robustness_is_balanced_and_result_blind(self) -> None:
        manifest = json.loads(
            (self.data / prepare_data.ROBUSTNESS_MANIFEST_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        exclusion = json.loads(
            (self.data / prepare_data.EXCLUSION_FILENAME).read_text(encoding="utf-8")
        )
        prepare_data.verify_robustness_manifest(
            manifest,
            self.data / prepare_data.QUESTION_FILENAME,
            exclusion,
        )
        self.assertFalse(manifest["official_track"])
        self.assertEqual(manifest["n"], 2800)
        self.assertEqual(set(manifest["class_counts"].values()), {100})
        self.assertEqual(len(manifest["class_counts"]), 28)
        self.assertFalse(
            set(manifest["canonical_ids"]) & set(exclusion["excluded_ids"])
        )
        self.assertTrue(
            all(
                any(old != new for old, new in row["canonical_to_perturbed"].items())
                for row in manifest["entries"]
            )
        )
        self.assertEqual(len(self.all_items), 14000)
        self.assertEqual(len({row.id for row in self.all_items}), 14000)

    def test_option_permutation_preserves_text_and_remaps_gold(self) -> None:
        clean = {row.id: row for row in self.questions}
        perturbed = [
            row for row in self.all_items if row.perturbation == "option_permutation_v1"
        ]
        self.assertEqual(len(perturbed), 2800)
        for row in perturbed[:100]:
            source = clean[row.canonical_id]
            self.assertEqual(row.question, source.question)
            self.assertEqual(set(row.option_texts), set(source.option_texts))
            inverse = {new: old for old, new in row.canonical_to_displayed}
            self.assertEqual(
                tuple(sorted(inverse[label] for label in row.gold_options)),
                source.gold_options,
            )


class AnswerParserTest(unittest.TestCase):
    def test_option_set_normalization(self) -> None:
        parsed = evaluation.parse_prediction("C,A", ("A", "B", "C", "D", "E"))
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["parsed_options"], ["A", "C"])
        self.assertFalse(parsed["strict_format"])

    def test_fullwidth_letters(self) -> None:
        parsed = evaluation.parse_prediction("ＡＣ", ("A", "B", "C", "D", "E"))
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["parsed_options"], ["A", "C"])
        self.assertTrue(parsed["strict_format"])

    def test_english_prefix_fails_closed(self) -> None:
        parsed = evaluation.parse_prediction("Answer: A", ("A", "B", "C", "D"))
        self.assertFalse(parsed["parse_ok"])

    def test_out_of_inventory_letter_fails(self) -> None:
        parsed = evaluation.parse_prediction("E", ("A", "B", "C", "D"))
        self.assertFalse(parsed["parse_ok"])
        self.assertEqual(parsed["invalid_letters"], ["E"])


class MetricContractTest(unittest.TestCase):
    def _row(self, *, exact: bool, prediction: list[str], kind: str) -> dict:
        return {
            "option_set_exact": exact,
            "strict_exact": exact,
            "strict_format": True,
            "parse_ok": True,
            "canonical_parsed_options": prediction,
            "perturbation": kind,
            "exam_type": "医师考试",
            "class_key": "医师考试/规培结业",
            "canonical_id": "1",
        }

    def test_unified_robustness_formulas(self) -> None:
        pairs = [
            (
                self._row(exact=True, prediction=["A"], kind="clean"),
                self._row(
                    exact=False, prediction=["B"], kind="option_permutation_v1"
                ),
            ),
            (
                self._row(exact=False, prediction=["B"], kind="clean"),
                self._row(
                    exact=True, prediction=["A"], kind="option_permutation_v1"
                ),
            ),
            (
                self._row(exact=True, prediction=["C"], kind="clean"),
                self._row(
                    exact=True, prediction=["C"], kind="option_permutation_v1"
                ),
            ),
        ]
        metric = evaluation._robustness_metric(pairs)
        self.assertEqual(metric["paired_units"], 3)
        self.assertAlmostEqual(metric["clean_task_success_rate"], 2 / 3)
        self.assertAlmostEqual(metric["perturbed_task_success_rate"], 2 / 3)
        self.assertAlmostEqual(metric["conditional_retention"], 1 / 2)
        self.assertAlmostEqual(metric["regression_rate"], 1 / 2)
        self.assertAlmostEqual(metric["robustness_drop_pp"], 0.0)
        self.assertEqual(metric["clean_correct_perturbed_wrong"], 1)
        self.assertEqual(metric["clean_wrong_perturbed_correct"], 1)
        self.assertAlmostEqual(metric["prediction_consistency"], 1 / 3)

    def test_unified_schema_is_pinned(self) -> None:
        spec = evaluation.analysis_spec()
        self.assertEqual(
            spec["unified_metric_schema"]["sha256"],
            evaluation.EXPECTED_UNIFIED_METRIC_SCHEMA_SHA256,
        )

    def test_paired_bootstrap_is_deterministic(self) -> None:
        ids = [str(index) for index in range(10)]
        care = {item: index % 2 == 0 for index, item in enumerate(ids)}
        base = {item: index % 3 == 0 for index, item in enumerate(ids)}
        stems = {item: item for item in ids}
        first = evaluation._paired_bootstrap(
            care, base, stems, ids, comparison_label="cpu-regression"
        )
        second = evaluation._paired_bootstrap(
            care, base, stems, ids, comparison_label="cpu-regression"
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["care_minus_baseline"], 0.1)

    def test_preregistration_lock_fails_closed_and_is_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                evaluation.load_preregistration_lock(missing)
            lock_path = Path(temporary) / "lock.json"
            project_root = evaluation.HERE.parents[1]
            inventory = []
            for relative in evaluation.CMB_PREREGISTERED_PATHS:
                target = project_root / relative
                inventory.append(
                    {
                        "path": relative,
                        "bytes": target.stat().st_size,
                        "sha256": evaluation.sha256_file(target),
                    }
                )
            contract = {
                "schema_version": "care-opd-public-preregistration-lock-v1",
                "scope": "test",
                "timing_disclosure": {"cmb_exam_predictions_observed": False},
                "fixed_model_order": list(evaluation.MODEL_ORDER),
                "model_suite": {
                    "suite_fingerprint": evaluation.canonical_hash(
                        [(label, "0" * 64) for label in evaluation.MODEL_ORDER]
                    ),
                    "models": [
                        {"label": label, "model_fingerprint": "0" * 64}
                        for label in evaluation.MODEL_ORDER
                    ],
                },
                "locked_files": inventory,
            }
            lock_path.write_text(
                json.dumps(
                    {
                        **contract,
                        "contract_fingerprint": evaluation.canonical_hash(contract),
                        "created_at_utc": "2026-08-08T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            lock = evaluation.load_preregistration_lock(lock_path)
            self.assertEqual(
                lock["preregistration_lock_sha256"],
                evaluation.sha256_file(lock_path),
            )
            locked = evaluation.locked_analysis_spec(lock)
            self.assertEqual(
                locked["preregistration_lock"]["preregistration_lock_sha256"],
                lock["preregistration_lock_sha256"],
            )


class ResumeContractTest(unittest.TestCase):
    def test_only_corrupt_terminal_line_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker.jsonl"
            row = {"index": 0, "id": "1", "run_fingerprint": "run"}
            path.write_bytes((json.dumps(row) + "\n{broken").encode("utf-8"))
            recovered = evaluation._recover_shard(path, {0: "1"}, "run")
            self.assertEqual(recovered, [row])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
