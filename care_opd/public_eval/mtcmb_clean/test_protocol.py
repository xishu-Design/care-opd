#!/usr/bin/env python3
"""CPU-only integrity tests for the frozen MTCMB protocol."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import evaluate_mtcmb as evaluation  # noqa: E402
import protocol  # noqa: E402
import summarize_generalization as root_summary  # noqa: E402
from preregistration import validate_static_lock  # noqa: E402


class MTCMBProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        default_repo = WORKSPACE / ".tmp_benchmark_overlap_public" / "MTCMB"
        cls.repo = Path(os.environ.get("MTCMB_REPO", default_repo)).resolve()
        if not cls.repo.is_dir():
            raise unittest.SkipTest(
                "set MTCMB_REPO to the pinned official checkout for full CPU tests"
            )
        cls.exclusions = protocol.load_exclusions(HERE / "overlap_exclusions.json")
        cls.train_seed = (
            WORKSPACE
            / ".tmp_benchmark_overlap_public"
            / "qzbc_formal"
            / "qzbc_native_seeds.train.jsonl"
        )
        cls.records = protocol.load_all_records(cls.repo)
        cls.items = protocol.build_inference_items(
            cls.repo, cls.records, cls.exclusions
        )

    def test_official_files_counts_hashes_and_prompts(self) -> None:
        report = protocol.verify_upstream_repo(self.repo, verify_git=False)
        self.assertEqual(report["records"], 7100)
        self.assertEqual(report["litdata_annotation_calls"], 306)
        self.assertEqual(report["task_counts"], dict(sorted(protocol.TASK_COUNTS.items())))
        self.assertEqual(set(report["prompt_sources"]), set(protocol.PROMPT_SOURCE_SHA256))

    def test_exclusion_manifest_and_views(self) -> None:
        ids = self.exclusions["source_record_ids"]
        self.assertEqual(len(ids), 1062)
        self.assertEqual(
            Counter(protocol.parse_source_record_id(value)[0] for value in ids),
            Counter({"TCM-ED-A": 231, "TCM-ED-B": 831}),
        )
        additional = self.exclusions["overlap_additional_record_ids"]
        self.assertEqual(len(additional), 124)
        self.assertFalse(set(ids) & set(additional))
        counts = Counter()
        excluded = protocol.overlap_excluded_ids(self.exclusions)
        self.assertEqual(len(excluded), 1186)
        self.assertEqual(
            Counter(protocol.parse_source_record_id(value)[0] for value in excluded),
            Counter({"TCM-ED-A": 249, "TCM-ED-B": 937}),
        )
        for task, rows in self.records.items():
            for row in rows:
                for view in protocol.record_views(task, row["id"], excluded):
                    counts[view] += 1
        self.assertEqual(counts["mtcmb_full"], 7100)
        self.assertEqual(counts["overlap_clean"], 5914)
        self.assertEqual(counts["provenance_clean9"], 1050)

    def test_inference_cardinality_and_unique_ids(self) -> None:
        counts = Counter(item.condition for item in self.items)
        self.assertEqual(counts, Counter({"clean": 7256, "option_permutation": 4862}))
        self.assertEqual(len(self.items), 12118)
        self.assertEqual(len({item.item_id for item in self.items}), 12118)
        self.assertEqual(
            sum(
                item.condition == "option_permutation" and item.task == "TCM-SE-B"
                for item in self.items
            ),
            50,
        )

    def test_every_controlled_permutation_is_nonidentity_and_gold_preserving(self) -> None:
        excluded = protocol.overlap_excluded_ids(self.exclusions)
        checked = 0
        for task in sorted(protocol.MCQ_TASKS):
            for row in self.records[task]:
                key = protocol.record_key(task, row["id"])
                if key in excluded or key in protocol.CONTROLLED_PERTURBATION_EXCLUDED_IDS:
                    continue
                labels, original = protocol.parse_mcq_options(task, row)
                perturbed, mapping = protocol.perturb_mcq_row(task, row)
                new_labels, displayed = protocol.parse_mcq_options(task, perturbed)
                self.assertEqual(labels, new_labels)
                self.assertEqual(set(mapping), set(labels))
                self.assertEqual(set(mapping.values()), set(labels))
                self.assertTrue(any(mapping[label] != label for label in labels))
                for canonical, shown in mapping.items():
                    self.assertEqual(displayed[shown], original[canonical])
                self.assertEqual(
                    perturbed["answer"], mapping[str(row["answer"]).strip().upper()]
                )
                checked += 1
        self.assertEqual(checked, 4862)

    def test_result_blind_overlap_audit_recomputes_frozen_union(self) -> None:
        audit = protocol.compute_overlap_audit(self.records, self.train_seed)
        self.assertEqual(audit["counts"]["union_direct_exact_near"], 1186)
        self.assertEqual(len(audit["additional_exact_or_near_record_ids"]), 124)
        self.assertEqual(
            audit["overlap_excluded_by_task"],
            {"TCM-ED-A": 249, "TCM-ED-B": 937},
        )

    def test_ed_b_known_upstream_prefix_defects_are_positional(self) -> None:
        rows = {int(row["id"]): row for row in self.records["TCM-ED-B"]}
        for record_id in (108, 109):
            self.assertTrue(str(rows[record_id]["options"][3]).startswith("C."))
            labels, texts = protocol.parse_mcq_options("TCM-ED-B", rows[record_id])
            self.assertEqual(labels, tuple("ABCDE"))
            self.assertTrue(texts["D"])

    def test_prompt_extraction_is_dependency_free_and_deterministic(self) -> None:
        row = self.records["TCM-ED-A"][0]
        first = protocol._capture_official_prompt(self.repo, "TCM-ED-A", row)
        second = protocol._capture_official_prompt(self.repo, "TCM-ED-A", row)
        self.assertEqual(first, second)
        self.assertIn(str(row["question"]), first)
        self.assertIn(str(row["options"]), first)
        self.assertIn("禁止返回其他内容", first)

    def test_official_mcq_parser_is_reproduced_separately_from_format_diagnostic(self) -> None:
        raw = "先排除A，最终答案是B"
        self.assertEqual(protocol.parse_official_choice(raw, tuple("ABCDE")), "A")
        parsed_diagnostic, strict = protocol.parse_choice(raw, tuple("ABCDE"))
        self.assertEqual(parsed_diagnostic, "B")
        self.assertFalse(strict)

    def test_msdd_upstream_aliases_and_exact_endpoint(self) -> None:
        vocab = protocol.msdd_vocab(self.records["TCM-MSDD"])
        row = self.records["TCM-MSDD"][0]
        short = []
        for key in ("证型", "疾病"):
            gold = str(row["answer"][key]).split("|")
            reverse = {value: alias for alias, value in protocol.MSDD_ALIASES[key].items()}
            short.extend(reverse[value] for value in gold)
        raw = "、".join(short)
        self.assertTrue(protocol.exact_task_success("TCM-MSDD", row, raw, vocab=vocab))
        partial, exact, _ = evaluation._msdd_scores(row["answer"], raw, vocab)
        self.assertEqual(partial, 1.0)
        self.assertTrue(exact)

    def test_pr_official_and_derived_exact_endpoints_are_separate(self) -> None:
        row = self.records["TCM-PR"][0]
        gold = row["answer"]
        raw = str(gold)
        official = evaluation._pr_scores(gold, raw)
        self.assertEqual(official["Task2_Score"], 1.0)
        vocab = protocol.msdd_vocab(self.records["TCM-MSDD"])
        self.assertTrue(protocol.exact_task_success("TCM-PR", row, raw, vocab=vocab))

    def test_unified_metric_field_names_are_exact(self) -> None:
        schema = json.loads((HERE.parent / "UNIFIED_METRIC_SCHEMA.json").read_text(encoding="utf-8"))
        self.assertEqual(set(protocol.unified_na()), set(schema["required_fields"]))
        pairs = [
            {
                "clean_task_success": True,
                "perturbed_task_success": False,
                "clean_canonical_prediction": "A",
                "perturbed_canonical_prediction": "B",
                "clean_parse_ok": True,
                "perturbed_parse_ok": True,
            },
            {
                "clean_task_success": False,
                "perturbed_task_success": True,
                "clean_canonical_prediction": "B",
                "perturbed_canonical_prediction": "A",
                "clean_parse_ok": True,
                "perturbed_parse_ok": True,
            },
        ]
        result = protocol.controlled_metrics(pairs)
        self.assertEqual(set(result), set(schema["required_fields"]))
        self.assertEqual(result["paired_units"], 2)
        self.assertEqual(result["conditional_retention"], 0.0)
        self.assertEqual(result["regression_rate"], 1.0)
        self.assertEqual(result["clean_correct_perturbed_wrong"], 1)
        self.assertEqual(result["clean_wrong_perturbed_correct"], 1)
        both_unparsed = protocol.controlled_metrics(
            [
                {
                    "clean_task_success": False,
                    "perturbed_task_success": False,
                    "clean_canonical_prediction": "",
                    "perturbed_canonical_prediction": "",
                    "clean_parse_ok": False,
                    "perturbed_parse_ok": False,
                }
            ]
        )
        self.assertEqual(both_unparsed["prediction_consistency"], 0.0)
        intervals = protocol.controlled_rate_intervals(pairs)
        self.assertEqual(intervals["clean_task_success_rate"]["numerator"], 1)
        self.assertEqual(intervals["clean_task_success_rate"]["denominator"], 2)
        self.assertEqual(intervals["regression_rate"]["estimate"], 1.0)
        self.assertLess(
            intervals["regression_rate"]["ci95_low"],
            intervals["regression_rate"]["ci95_high"],
        )

    def test_full_offline_scoring_views_and_main_endpoints(self) -> None:
        lookup = {
            (task, int(row["id"])): row
            for task, rows in self.records.items()
            for row in rows
        }
        predictions = []
        for index, item in enumerate(self.items):
            source = lookup[(item.task, item.record_id)]
            mapping = dict(item.canonical_to_displayed)
            if item.annotation_index is not None:
                raw = str(source["annotations"][item.annotation_index]["A"])
            elif item.task in protocol.MCQ_TASKS:
                gold = str(source["answer"]).strip().upper()
                raw = mapping.get(gold, gold)
            elif item.task == "TCM-FT":
                raw = str(source.get("points") or source.get("answer") or "")
            elif item.task == "TCM-PR":
                raw = str(source["answer"])
            else:
                raw = json.dumps(source["answer"], ensure_ascii=False)
            predictions.append(
                {
                    "index": index,
                    "item_id": item.item_id,
                    "task": item.task,
                    "record_id": item.record_id,
                    "annotation_index": item.annotation_index,
                    "condition": item.condition,
                    "canonical_to_displayed": mapping,
                    "raw_output": raw,
                }
            )

        def perfect_bert(pairs, *_args, **_kwargs):
            return {identifier: 1.0 for identifier, _, _ in pairs}

        perfect_lexical = {"rouge_1": 1.0, "rouge_l": 1.0, "bleu": 100.0}
        with mock.patch.object(evaluation, "_batch_bert_scores", side_effect=perfect_bert), mock.patch.object(
            evaluation, "_lexical_scores", return_value=perfect_lexical
        ):
            scores, pairs, summary = evaluation.score_predictions(
                predictions,
                self.records,
                self.exclusions,
                {"canonical_path": "unused"},
                bert_device="cpu",
                bert_batch_size=4,
            )
        self.assertEqual(len(scores), 7050)
        self.assertEqual(len(pairs), 4862)
        self.assertEqual(summary["views"]["overlap_clean"]["offline_scored_records"], 5864)
        self.assertEqual(summary["views"]["provenance_clean9"]["offline_scored_records"], 1050)
        self.assertEqual(
            summary["views"]["provenance_clean9"][
                "natural_exact_endpoints_by_task"
            ]["eligible_records_descriptive_only"],
            250,
        )
        strict = summary["controlled_robustness"]["provenance_clean9_mcq"]
        self.assertEqual(strict["paired_units"], 50)
        self.assertEqual(strict["perturbed_task_success_rate"], 1.0)
        self.assertEqual(strict["conditional_retention"], 1.0)
        self.assertEqual(
            strict["rate_estimates_95ci"]["clean_task_success_rate"]["numerator"],
            50,
        )

    def test_task_manifest_declares_official_and_derived_separation(self) -> None:
        manifest = json.loads((HERE / "task_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["reporting"]["official_and_derived_separate"])
        self.assertTrue(manifest["reporting"]["sample_micro_average_across_tasks_forbidden"])
        self.assertEqual(
            [task for task, value in manifest["tasks"].items() if value["clean9"]],
            list(protocol.CLEAN9_TASKS),
        )
        self.assertFalse(manifest["tasks"]["TCM-SE-A"]["offline"])

    def test_main_table_schema_is_root_aggregator_compatible(self) -> None:
        metrics = protocol.controlled_metrics(
            [
                {
                    "clean_task_success": True,
                    "perturbed_task_success": True,
                    "clean_canonical_prediction": "A",
                    "perturbed_canonical_prediction": "A",
                    "clean_parse_ok": True,
                    "perturbed_parse_ok": True,
                }
            ]
        )
        rows = evaluation._main_table_records(
            [
                {
                    "model": "care_opd",
                    **metrics,
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["track"], "provenance_clean9_option_permutation")
        self.assertEqual(rows[0]["official_or_derived"], "derived")
        self.assertEqual(rows[0]["robustness_display_key"], "MTCMB Clean9 SE-B retention")
        self.assertEqual(rows[0]["clean_n"], 1)
        self.assertEqual(set(rows[0]["unified_metrics"]), set(protocol.unified_na()))

    def test_root_extractor_accepts_complete_six_model_block(self) -> None:
        records = []
        for model in protocol.MODEL_ORDER:
            metrics = protocol.unified_na(paired_units=50)
            metrics.update(
                {
                    "clean_task_success_rate": 0.8,
                    "perturbed_task_success_rate": 0.7,
                    "conditional_retention": 0.75,
                    "regression_rate": 0.25,
                    "robustness_drop_pp": 10.0,
                    "prediction_consistency": 0.7,
                    "worst_perturbation_tsr": 0.7,
                    "clean_correct_perturbed_wrong": 10,
                    "clean_wrong_perturbed_correct": 5,
                }
            )
            records.extend(evaluation._main_table_records([{"model": model, **metrics}]))
        extracted = root_summary.extract_mtcmb({"main_table_records": records})
        self.assertEqual(len(extracted), 6)
        self.assertTrue(all(row["official_or_derived"] == "derived" for row in extracted))
        self.assertTrue(
            all(
                row["robustness_display_key"] == "MTCMB Clean9 SE-B retention"
                for row in extracted
            )
        )

    def test_bert_suite_must_match_root_auxiliary_lock(self) -> None:
        models = [
            {"label": label, "model_fingerprint": f"{index + 1:064x}"}
            for index, label in enumerate(protocol.MODEL_ORDER)
        ]
        suite = {
            "models": models,
            "bert_scorer": {
                "canonical_path": "/frozen/bert-base-chinese",
                "model_fingerprint": "b" * 64,
            },
        }
        suite["suite_fingerprint"] = evaluation._suite_model_fingerprint(suite)
        suite["evaluation_assets_fingerprint"] = evaluation.evaluation_assets_fingerprint(
            suite
        )
        lock = {
            "model_suite_fingerprint": suite["suite_fingerprint"],
            "auxiliary_models": {
                "mtcmb_bert_scorer": {
                    "status": "LOCKED",
                    "canonical_path": "/frozen/bert-base-chinese",
                    "artifact_fingerprint": "b" * 64,
                }
            },
        }
        evaluation._assert_suite_matches_lock(suite, lock)
        swapped_path = {**suite, "bert_scorer": {**suite["bert_scorer"]}}
        swapped_path["bert_scorer"]["canonical_path"] = "/swapped/bert"
        swapped_path["evaluation_assets_fingerprint"] = (
            evaluation.evaluation_assets_fingerprint(swapped_path)
        )
        with self.assertRaises(ValueError):
            evaluation._assert_suite_matches_lock(swapped_path, lock)
        swapped_weights = {**suite, "bert_scorer": {**suite["bert_scorer"]}}
        swapped_weights["bert_scorer"]["model_fingerprint"] = "c" * 64
        swapped_weights["evaluation_assets_fingerprint"] = (
            evaluation.evaluation_assets_fingerprint(swapped_weights)
        )
        with self.assertRaises(ValueError):
            evaluation._assert_suite_matches_lock(swapped_weights, lock)

    def test_complete_validator_rejects_same_count_wrong_or_duplicate_keys(self) -> None:
        score_keys = evaluation.expected_score_record_keys(self.records)
        pair_keys = evaluation.expected_controlled_pair_keys(
            self.records, self.exclusions
        )
        excluded = protocol.overlap_excluded_ids(self.exclusions)
        scores = []
        for task in sorted(protocol.OFFLINE11_TASKS):
            for row in self.records[task]:
                eligible = task in protocol.EXACT_TASKS
                scores.append(
                    {
                        "task": task,
                        "record_id": int(row["id"]),
                        "record_key": protocol.record_key(task, row["id"]),
                        "views": list(protocol.record_views(task, row["id"], excluded)),
                        "official_score_0_100": 0.0,
                        "binary_task_success_eligible": eligible,
                        "binary_task_success": False if eligible else None,
                        "strict_format": False if task in protocol.MCQ_TASKS else None,
                        "parse_failure": True if task in protocol.MCQ_TASKS else None,
                    }
                )
        pairs = []
        for key in pair_keys:
            task, record_id = protocol.parse_source_record_id(key)
            pairs.append(
                {
                    "task": task,
                    "record_id": record_id,
                    "record_key": key,
                    "clean_task_success": False,
                    "perturbed_task_success": False,
                    "clean_parse_ok": False,
                    "perturbed_parse_ok": False,
                    "clean_canonical_prediction": "",
                    "perturbed_canonical_prediction": "",
                    "views": ["overlap_clean_mcq"]
                    + (["provenance_clean9_mcq"] if task == "TCM-SE-B" else []),
                }
            )
        aggregate = evaluation.aggregate_scored_rows(scores, pairs)
        fingerprints = {"contract": "locked"}
        run_fingerprint = "run-fingerprint"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            evaluation.atomic_jsonl(
                output / "predictions.jsonl",
                [{"item_id": "prediction-1", **fingerprints}],
            )
            evaluation.atomic_jsonl(output / "record_scores.jsonl", scores)
            evaluation.atomic_jsonl(
                output / "controlled_robustness_pairs.jsonl", pairs
            )
            evaluation.atomic_json(
                output / "summary.json",
                {
                    "run_fingerprint": run_fingerprint,
                    "fingerprints": fingerprints,
                    "official_results": aggregate["views"],
                    "derived_controlled_robustness": aggregate[
                        "controlled_robustness"
                    ],
                },
            )
            evaluation.atomic_json(
                output / "run_manifest.json",
                {"run_fingerprint": run_fingerprint},
            )

            def reclose() -> None:
                artifacts = {
                    name: output / name
                    for name in (
                        "predictions.jsonl",
                        "record_scores.jsonl",
                        "controlled_robustness_pairs.jsonl",
                        "summary.json",
                        "run_manifest.json",
                    )
                }
                evaluation.atomic_json(
                    output / "COMPLETE.json",
                    {
                        "status": "complete",
                        "run_fingerprint": run_fingerprint,
                        "artifact_sha256": {
                            name: protocol.sha256_file(path)
                            for name, path in artifacts.items()
                        },
                    },
                )

            def validates() -> bool:
                return evaluation._completed_valid(
                    output,
                    run_fingerprint=run_fingerprint,
                    expected_item_ids={"prediction-1"},
                    expected_score_keys=score_keys,
                    expected_pair_keys=pair_keys,
                    fingerprints=fingerprints,
                )

            reclose()
            self.assertTrue(validates())

            original_score_key = scores[0]["record_key"]
            scores[0]["record_key"] = "TCM-FT:999999"
            evaluation.atomic_jsonl(output / "record_scores.jsonl", scores)
            reclose()
            self.assertFalse(validates())
            scores[0]["record_key"] = original_score_key
            evaluation.atomic_jsonl(output / "record_scores.jsonl", scores)

            pairs[0]["record_key"] = pairs[1]["record_key"]
            evaluation.atomic_jsonl(
                output / "controlled_robustness_pairs.jsonl", pairs
            )
            reclose()
            self.assertFalse(validates())

    def test_clean9_paired_noninferiority_contract(self) -> None:
        care = []
        baseline = []
        for index in range(protocol.EXPECTED_CLEAN9_MCQ_PAIRS):
            common = {"record_key": f"TCM-SE-B:{index + 1}"}
            care.append(
                {
                    **common,
                    "clean_task_success": index < 45,
                    "perturbed_task_success": index < 43,
                }
            )
            baseline.append(
                {
                    **common,
                    "clean_task_success": index < 44,
                    "perturbed_task_success": index < 41,
                }
            )
        report = evaluation._paired_robustness_noninferiority(
            care, baseline, "clean_sft", replicates=200, seed=17
        )
        self.assertEqual(report["paired_units"], 50)
        self.assertEqual(report["margin_absolute"], -0.05)
        self.assertEqual(
            set(report["endpoints"]),
            {"perturbed_task_success_rate", "conditional_retention"},
        )
        self.assertIsInstance(report["joint_primary_noninferiority_pass"], bool)
        self.assertEqual(report["replicates"], 200)
        self.assertEqual(report["seed"], 17)

    def test_arbitrary_json_is_not_a_preregistration_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bogus = root / "lock.json"
            bogus.write_text('{"status":"ok"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_static_lock(bogus, project_root=root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
