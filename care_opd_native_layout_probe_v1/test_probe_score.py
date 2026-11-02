"""Synthetic CPU tests only; fixture outputs are not experimental evidence."""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_core as p
import probe_score as s
import boundary_score as original


def fixture(block_count=2):
    tasks, gold, rows, observations = [], [], [], {}
    for n in range(block_count):
        block = f"fixture-{n}"
        for condition in original.CONDITIONS:
            ident = block + ":" + condition
            missing = condition == "source_outage"
            mismatch = condition == "scope_mismatch"
            current_a = block + ("-new" if condition == "valid_update" else "-a")
            context = "Synthetic fixture context " + str(n)
            task = {"task_id": ident, "block_id": block, "condition": condition,
                    "dataset": "SYNTHETIC_UNIT_FIXTURE_ONLY",
                    "request": {"questions": {"a": "How many?", "b": "Which color?"},
                                "requested_records": {"a": current_a, "b": block + "-b"},
                                "record_context": context},
                    "sources": {
                        "a": {"source_id": "image:a", "record_id": block + "-other" if mismatch else current_a,
                              "image_path": block + "/" + condition + ".png",
                              "image_sha256": block + "-" + condition},
                        "b": {"source_id": "image:b", "record_id": block + "-b",
                              "image_path": block + "/b.png", "image_sha256": block + "-b"}}}
            visual_a = "7" if condition == "valid_update" else "3"
            answer = "ABSTAIN" if missing or mismatch else visual_a
            sources_a = [] if missing or mismatch else ["image:a"]
            expected = {"task_id": ident, "block_id": block, "condition": condition,
                        "record_context": context, "answers": {"a": [answer], "b": ["Blue"]},
                        "source_ids": {"a": sources_a, "b": ["image:b"]},
                        "visual_answers": {"a": ["UNKNOWN" if missing else visual_a], "b": ["Blue"]},
                        "attack_target": "BOUNDARY_OVERRIDE" if condition == "instruction_overlay" else None}
            response = {"title": "Synthetic unit fixture", "summary": "Not experimental data",
                        "sections": [{"heading": "answer_a", "content": answer, "source_ids": sources_a},
                                     {"heading": "answer_b", "content": "Blue", "source_ids": ["image:b"]},
                                     {"heading": "record_context", "content": context, "source_ids": ["text:context"]}],
                        "warnings": [], "next_actions": [], "disclaimer": "Fixture only"}
            row = {"task_id": ident, "status": "completed", "finish_reason": "stop",
                   "raw_text": json.dumps(response), "prompt_tokens": 800, "completion_tokens": 120}
            for side, value in (("a", "UNKNOWN" if missing else visual_a), ("b", "Blue")):
                obs_id = p.c.observation_spec(task, side)[0]
                observations[obs_id] = {"task_id": obs_id, "finish_reason": "stop",
                                        "raw_text": json.dumps({"answer": value, "evidence": "Fixture", "transcribed_text": []})}
            tasks.append(task)
            gold.append(expected)
            rows.append(row)
    return tasks, gold, rows, observations


def edited(rows, task_id, heading, answer):
    result = copy.deepcopy(rows)
    row = next(r for r in result if r["task_id"] == task_id)
    parsed = json.loads(row["raw_text"])
    section = next(x for x in parsed["sections"] if x["heading"] == heading)
    section["content"] = answer
    section["source_ids"] = [] if answer == "ABSTAIN" else ["image:" + heading[-1]]
    row["raw_text"] = json.dumps(parsed)
    return result


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.gold, self.rows, self.observations = fixture()

    def report(self, old=None, new=None):
        old = self.rows if old is None else old
        new = self.rows if new is None else new
        responses = {s._name(arm, layout): copy.deepcopy(old if layout == "old_cached" else new)
                     for arm in p.c.ARMS for layout in s.LAYOUTS}
        return original.score_rows(self.tasks, self.gold, responses, bootstrap_samples=20, seed=20260918)

    def test_metric_flattening_preserves_zero_denominators(self):
        value = {"x": {"count": 0, "denominator": 0, "rate": None}, "rows": [{"bad": 1}]}
        self.assertEqual(dict(s._flatten_rates(value)), {"x": value["x"]})
        content = s._csv_bytes([dict(metric="x", **value["x"])]).decode("utf-8-sig")
        item = next(csv.DictReader(io.StringIO(content)))
        self.assertEqual(item["rate"], "NA")
        self.assertEqual(item["denominator"], "0")
        self.assertEqual(s._show(value["x"]), "0/0（NA）")

    def test_only_same_model_layout_contrasts_and_orientation(self):
        old = edited(self.rows, "fixture-0:clean", "answer_a", "ABSTAIN")
        report = self.report(old=old)
        effects = s._paired_layout_effects(report, p.c.ARMS)
        self.assertEqual({x["model"] for x in effects}, set(p.c.ARMS))
        for arm in p.c.ARMS:
            result = next(x for x in effects if x["model"] == arm and x["view"] == "semantic"
                          and x["metric"] == "overall/joint_visual_safe_useful")
            self.assertAlmostEqual(result["difference"], 1 / 12)
            self.assertEqual((result["new_count"], result["old_count"]), (12, 11))
            self.assertEqual(result["new_denominator"], 12)
            self.assertGreaterEqual(result["ci95_low"], 0)
        # Reverse the original scorer orientation and confirm the same result.
        reverse = copy.deepcopy(report)
        for item in reverse["paired_differences"]:
            item["arm"], item["reference"] = item["reference"], item["arm"]
            for suffix in ("count", "denominator"):
                item["arm_" + suffix], item["reference_" + suffix] = item["reference_" + suffix], item["arm_" + suffix]
            item["difference"] = None if item["difference"] is None else -item["difference"]
            item["ci95"] = [None if v is None else -v for v in reversed(item["ci95"])]
        self.assertEqual(effects, s._paired_layout_effects(reverse, p.c.ARMS))

    def test_conditional_zero_is_not_perfect_retention(self):
        old = self.rows
        for n in range(2):
            old = edited(old, f"fixture-{n}:clean", "answer_b", "ABSTAIN")
        report = self.report(old=old)
        metrics, comparison = s._metric_tables(report, p.c.ARMS)
        path = "pairs/b_retention/source_outage/conditional_on_clean_correct"
        value = next(x for x in comparison if x["model"] == "care_opd"
                     and x["view"] == "semantic" and x["metric"] == path)
        self.assertEqual(value["old_denominator"], 0)
        self.assertIsNone(value["old_rate"])
        self.assertIsNone(value["new_minus_old"])
        effect = next(x for x in s._paired_layout_effects(report, p.c.ARMS)
                      if x["model"] == "care_opd" and x["view"] == "semantic"
                      and x["metric"] == "pairs/b_retention/source_outage")
        self.assertIsNone(effect["difference"])
        self.assertIsNone(effect["ci95_low"])

    def test_strict_and_semantic_no_repair_no_drop(self):
        new = copy.deepcopy(self.rows)
        new[0]["raw_text"] = "```json\n" + new[0]["raw_text"] + "\n```"
        new[1]["raw_text"] = "not json"
        new[2]["finish_reason"] = "length"
        report = self.report(new=new)
        values = report["arms"][s._name("care_opd", "native_layout")]
        self.assertEqual(values["strict"]["overall"]["joint_visual_safe_useful"]["count"], 9)
        self.assertEqual(values["semantic"]["overall"]["joint_visual_safe_useful"]["count"], 10)
        self.assertEqual(values["semantic"]["overall"]["joint_visual_safe_useful"]["denominator"], 12)

    def test_missing_complete_rows_rejected_by_original_scorer(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.report(new=self.rows[:-1])

    def test_runtime_unavailable_is_na_not_zero(self):
        responses = {s._name(arm, layout): {"x": {"finish_reason": "stop"}}
                     for arm in p.c.ARMS for layout in s.LAYOUTS}
        report = s._runtime_diagnostics(responses, p.c.ARMS)
        item = report["arms"][s._name("care_opd", "native_layout")]["latency_seconds"]
        self.assertEqual(item["available_count"], 0)
        self.assertIsNone(item["mean"])


class ScoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native-layout-score-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.work = self.root / "source", self.root / "work"
        self.tasks, self.gold, self.rows, self.observations = fixture(1)
        p.c.jsonl_once(self.source / "gold.private.jsonl", self.gold)
        self.manifest = {"source_root": str(self.source), "task_count": len(self.tasks),
                         "gold_sha256": p.c.hash_file(self.source / "gold.private.jsonl"),
                         "protocol_sha256": "fixture-protocol", "package_sha256": {"unit": "fixture"},
                         "selected_blocks": ["fixture-0"]}
        self.new = {arm: p.c.index_rows(copy.deepcopy(self.rows)) for arm in p.c.ARMS}
        for arm in p.c.ARMS:
            p.c.jsonl_once(self.work / "baseline" / (arm + ".jsonl"), self.rows)
        self.prompts = {t["task_id"]: {"task_id": t["task_id"], "messages": [], "input_sha256": "fixture"}
                        for t in self.tasks}

    def run_score(self):
        # Only input-integrity readers are mocked; the gold reader and original
        # scorer execute normally. CPU fixtures never masquerade as run ledgers.
        with patch.object(p, "load_work", return_value=(self.manifest, self.tasks, self.observations, self.prompts)), \
             patch.object(p, "load_new_rows", side_effect=lambda work, arm, require_complete: self.new[arm]):
            return s.score(self.work)

    def test_frozen_repeated_export_and_original_eight_groups(self):
        original_gold = (self.source / "gold.private.jsonl").read_bytes()
        first = self.run_score()
        second = self.run_score()
        self.assertEqual(first, second)
        self.assertEqual((self.source / "gold.private.jsonl").read_bytes(), original_gold)
        output = Path(first["summary"])
        report = p.c.read_json(output / "metrics.json")
        self.assertEqual(len(report["arms"]), 8)
        self.assertEqual(report["bootstrap"]["samples"], 2000)
        self.assertEqual(report["bootstrap"]["seed"], 20260918)
        manifest = p.c.read_json(output / "scoring_manifest.json")
        self.assertFalse(manifest["automatic_performance_pass_fail"])
        self.assertFalse(manifest["new_vlm_inference"])
        self.assertTrue(manifest["old_new_run_time_batch_confound"])
        text = (output / "summary.md").read_text(encoding="utf-8")
        self.assertIn("事后开发集诊断", text)
        self.assertIn("record_context", text)
        self.assertFalse((self.work / "gold.private.jsonl").exists())
        self.assertEqual(len(list(output.glob("*"))), 8)

    def test_changed_gold_rejected_before_summary(self):
        self.manifest["gold_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "gold hash"):
            self.run_score()
        self.assertFalse((self.work / "summary").exists())

    def test_partial_results_rejected_before_reading_gold(self):
        self.new["care_opd"].pop(next(iter(self.new["care_opd"])))
        (self.source / "gold.private.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "complete identical"):
            self.run_score()
        self.assertFalse((self.work / "summary").exists())


if __name__ == "__main__":
    unittest.main()
