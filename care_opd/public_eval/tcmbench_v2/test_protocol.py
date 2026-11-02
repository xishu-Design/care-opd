#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import evaluate_tcmbench_v2 as evaluation
from evaluate_tcmbench_v2 import parse_letter, parse_numeric, summarize_model
from protocol import (
    ROBUSTNESS_TASKS,
    build_sdt_pair_manifest,
    inspect_known_data_issues,
    load_episodes,
    load_overlap_exclusions,
    semantic_id,
    verify_snapshot,
)


HERE = Path(__file__).resolve().parent


class ParserTests(unittest.TestCase):
    def test_answer_only_letter_contract(self) -> None:
        self.assertEqual(parse_letter("<eoa>C</eoa>", "answer_only")["prediction"], "C")
        self.assertTrue(parse_letter("<eoa>C</eoa>", "answer_only")["strict_format"])
        self.assertFalse(parse_letter("答案是 C", "answer_only")["parse_ok"])

    def test_reconstructed_official_contract(self) -> None:
        parsed = parse_letter("【解析】辨析。<eoe>\n【答案】: D <eoa>", "reconstructed_official_cot")
        self.assertEqual(parsed["prediction"], "D")
        self.assertTrue(parsed["strict_format"])

    def test_numeric_contract_and_range(self) -> None:
        parsed = parse_numeric("<eoa>[4]</eoa>", "answer_only", 4)
        self.assertEqual(parsed["prediction"], "4")
        self.assertTrue(parsed["strict_format"])
        self.assertFalse(parse_numeric("<eoa>[5]</eoa>", "answer_only", 4)["parse_ok"])

    def test_semantic_id_ignores_presentation_number(self) -> None:
        first = semantic_id("同一病例", "1)．其治法是（ ）。\nA．甲\nB．乙")
        second = semantic_id("同一病例", "3）其治法是（ ）。\nA．甲\nB．乙")
        self.assertEqual(first, second)

    def test_overlap_manifest_has_identifier_only_payload(self) -> None:
        loaded = load_overlap_exclusions(HERE / "overlap_exclusions.json")
        self.assertEqual(len(loaded["item_ids"]), 39)
        payload = loaded["manifest"]
        self.assertNotIn("questions", payload)

    def test_summary_uses_preregistered_robustness_fields(self) -> None:
        def item(item_id: str, task: str, prediction: str, gold: str) -> dict:
            return {
                "item_id": item_id, "episode_id": item_id, "semantic_id": "same",
                "task": task, "level": "level", "case_index": 0, "kind": "sequential",
                "presentation_index": 0, "gold": gold, "prediction": prediction,
                "raw_output": prediction, "parse_ok": True, "strict_format": True,
                "parse_route": "strict", "correct": prediction == gold,
                "latency_seconds": 0.0, "generated_tokens": 1, "context_messages": 2,
            }
        rows = [
            item("c", "SDT", "A", "A"),
            item("r", "SDT_reverse", "B", "A"),
            item("s", "SDT_shuffle", "A", "A"),
        ]
        pairs = {
            "available": True,
            "pairs": [
                {"perturbation": "SDT_reverse", "case_index": 0, "semantic_id": "same",
                 "gold": "A", "clean_item_id": "c", "perturbed_item_id": "r"},
                {"perturbation": "SDT_shuffle", "case_index": 0, "semantic_id": "same",
                 "gold": "A", "clean_item_id": "c", "perturbed_item_id": "s"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "predictions.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = summarize_model(path, pairs, set())["official_full"]
        main = summary["public_paired_robustness_track"]
        self.assertEqual(main["perturbed_task_success_rate"], 0.5)
        self.assertEqual(main["conditional_retention"], 0.5)
        self.assertEqual(main["regression_rate"], 0.5)
        self.assertEqual(main["clean_correct_perturbed_wrong"], 1)
        self.assertEqual(main["clean_wrong_perturbed_correct"], 0)
        self.assertEqual(main["paired_units"], 2)
        self.assertEqual(main["robustness_drop_pp"], 50.0)

    def test_shared_preregistration_lock_is_accepted_and_closed(self) -> None:
        project_root = evaluation.HERE.parents[1]
        required = (
            "public_eval/UNIFIED_METRIC_SCHEMA.json",
            "public_eval/PUBLIC_GENERALIZATION_PREREGISTRATION.md",
            "public_eval/tcmbench_v2/protocol.py",
            "public_eval/tcmbench_v2/evaluate_tcmbench_v2.py",
            "public_eval/tcmbench_v2/run_existing_gpu4_5.sh",
            "public_eval/tcmbench_v2/overlap_exclusions.json",
        )
        inventory = []
        for relative in required:
            target = project_root / relative
            inventory.append(
                {
                    "path": relative,
                    "bytes": target.stat().st_size,
                    "sha256": evaluation.sha256_file(target),
                }
            )
        contract = {
            "schema_version": evaluation.ROOT_PREREGISTRATION_SCHEMA,
            "scope": "test",
            "timing_disclosure": {"tcmbench_v2_predictions_observed": False},
            "fixed_model_order": list(evaluation.EXPECTED_MODEL_ORDER),
            "model_suite": {
                "suite_fingerprint": evaluation.canonical_hash(
                    [(label, "0" * 64) for label in evaluation.EXPECTED_MODEL_ORDER]
                ),
                "models": [
                    {"label": label, "model_fingerprint": "0" * 64}
                    for label in evaluation.EXPECTED_MODEL_ORDER
                ],
            },
            "locked_files": inventory,
        }
        payload = {
            **contract,
            "contract_fingerprint": evaluation.canonical_hash(contract),
            "created_at_utc": "2026-08-08T00:00:00+00:00",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            identity = evaluation.validate_preregistration_lock(
                path,
                model_suite_fingerprint=contract["model_suite"]["suite_fingerprint"],
                benchmark_contract={"protocol": "fixed"},
            )
        self.assertEqual(identity["schema_version"], evaluation.ROOT_PREREGISTRATION_SCHEMA)
        self.assertEqual(identity["lock_kind"], "shared_static_protocol_lock")

    def test_completion_validator_rejects_self_consistent_short_result(self) -> None:
        expected_episode = {
            "episode_id": "ep-1",
            "items": [{"item_id": "item-1"}],
        }
        task_manifest = {"manifest_fingerprint": "task-fp"}
        pair_manifest = {"pair_manifest_fingerprint": "pair-fp"}
        episode_row = {
            "episode_id": "ep-1",
            "task": "SDT",
            "level": "L1",
            "case_index": 0,
            "kind": "single",
            "predictions": [{"item_id": "item-1", "prediction": "A"}],
        }
        run_fp = "run-fp"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {
                "episodes.jsonl": root / "episodes.jsonl",
                "predictions.jsonl": root / "predictions.jsonl",
                "run_manifest.json": root / "run_manifest.json",
                "task_manifest.json": root / "task_manifest.json",
                "pair_manifest.json": root / "pair_manifest.json",
            }
            flat = evaluation._flatten([episode_row])
            artifacts["episodes.jsonl"].write_text(
                json.dumps(episode_row) + "\n", encoding="utf-8"
            )
            artifacts["predictions.jsonl"].write_text(
                json.dumps(flat[0]) + "\n", encoding="utf-8"
            )
            artifacts["run_manifest.json"].write_text(
                json.dumps(
                    {
                        "run_fingerprint": run_fp,
                        "episodes": 1,
                        "items": 1,
                        "task_manifest_fingerprint": "task-fp",
                        "pair_manifest_fingerprint": "pair-fp",
                    }
                ),
                encoding="utf-8",
            )
            artifacts["task_manifest.json"].write_text(
                json.dumps(task_manifest), encoding="utf-8"
            )
            artifacts["pair_manifest.json"].write_text(
                json.dumps(pair_manifest), encoding="utf-8"
            )
            marker = {
                "status": "complete",
                "run_fingerprint": run_fp,
                "episodes": 1,
                "items": 1,
                "artifact_sha256": {
                    name: evaluation.sha256_file(path)
                    for name, path in artifacts.items()
                },
            }
            (root / "COMPLETE.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            self.assertTrue(
                evaluation._completed_valid(
                    root,
                    run_fp,
                    episodes=[expected_episode],
                    task_manifest_expected=task_manifest,
                    pair_manifest_expected=pair_manifest,
                )
            )
            longer_expected = [
                expected_episode,
                {"episode_id": "ep-2", "items": [{"item_id": "item-2"}]},
            ]
            self.assertFalse(
                evaluation._completed_valid(
                    root,
                    run_fp,
                    episodes=longer_expected,
                    task_manifest_expected=task_manifest,
                    pair_manifest_expected=pair_manifest,
                )
            )


class OfficialSnapshotIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw = os.environ.get("TCMBENCH_DATA_ROOT")
        cls.data_root = Path(raw).resolve() if raw else None

    def test_official_snapshot_pair_and_defect_audit(self) -> None:
        if self.data_root is None:
            self.skipTest("set TCMBENCH_DATA_ROOT for official-data integration test")
        snapshot = verify_snapshot(self.data_root)
        self.assertEqual(len(snapshot["files"]), 14)
        episodes, _ = load_episodes(self.data_root, ROBUSTNESS_TASKS)
        self.assertEqual(len(episodes), 774)
        self.assertEqual(sum(len(row["items"]) for row in episodes), 1335)
        pairs = build_sdt_pair_manifest(episodes)
        self.assertEqual(pairs["case_pairs"], 154)
        self.assertEqual(pairs["question_pairs"], 528)
        self.assertEqual(pairs["pairs_by_perturbation"]["SDT_reverse"], 264)
        self.assertEqual(pairs["pairs_by_perturbation"]["SDT_shuffle"], 264)
        issues = inspect_known_data_issues(self.data_root)
        self.assertEqual(issues["DDST_hard"]["position_shortcut_rate"], 1.0)
        self.assertEqual(
            issues["herb_predict"]["rows_whose_embedded_question_requests_syndrome_and_therapy"],
            239,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
