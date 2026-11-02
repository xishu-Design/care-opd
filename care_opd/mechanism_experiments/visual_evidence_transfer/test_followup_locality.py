"""CPU-only tests using synthetic records, never source evaluation results."""
from __future__ import annotations

import copy
import json
import unittest

import followup_locality as locality


def native(sections):
    return {"title": "研究", "summary": "所给证据", "sections": sections,
            "warnings": [], "next_actions": [], "disclaimer": "仅供研究"}


def source_fixture(count=6):
    tasks, gold, observations, responses, attributes = [], {}, {}, {}, {}
    for index in range(count):
        family = f"family-{index}"
        attributes[family] = [
            {"field": "modality", "question": "该独立记录的成像类型？", "value": "CT",
             "source_qid": f"attr-{index}-1", "source_json_sha256": "f" * 64},
            {"field": "location", "question": "该独立记录的标注部位？", "value": "上腹部",
             "source_qid": f"attr-{index}-2", "source_json_sha256": "f" * 64},
        ]
        for condition, answer in (("clean", "YES"), ("no_image", "ABSTAIN"), ("swapped_image", "NO")):
            ident = family + ":" + condition
            task = {"task_id": ident, "family_id": family, "cluster_id": f"cluster-{index}",
                    "condition": condition, "question": f"目标 {index} 是否存在？", "context_value": f"CTX{index:05d}",
                    "image_path": None if condition == "no_image" else f"/fake/{ident}.png"}
            tasks.append(task)
            gold[ident] = {"answer": answer, "context_value": task["context_value"]}
            observations[ident] = {"task_id": ident, "status": "completed", "raw_text": "OBS:" + ident}
            output = native([
                {"heading": "visual_answer", "content": answer,
                 "source_ids": [] if answer == "ABSTAIN" else ["image:current"]},
                {"heading": "context_value", "content": task["context_value"], "source_ids": ["text:context"]}])
            responses[ident] = {"task_id": ident, "status": "completed", "raw_text": json.dumps(output),
                                "finish_reason": "stop"}
    return {"tasks": tasks, "task_by_id": {t["task_id"]: t for t in tasks}, "gold_by_id": gold,
            "observations": observations, "responses": {"care_opd": responses},
            "text_attributes_by_family": attributes, "binding": {"source": "synthetic"},
            "selection": {"dataset": {"source_split": "validation", "source_json_sha256": "f" * 64}},
            "manifest": {}, "config": {}}


def perfect_rows(bundle):
    rows = []
    for task in bundle["tasks"]:
        gold = bundle["gold_by_id"][task["task_id"]]
        sections = []
        for heading in locality.HEADINGS:
            value = gold[heading]
            if heading in locality.VISUAL_FIELDS:
                sources = [] if value == "ABSTAIN" else ["image:" + heading[-1]]
            elif heading in locality.TEXT_FIELDS:
                sources = ["text:attribute:" + heading[-1]]
            else:
                sources = ["text:context"]
            sections.append({"heading": heading, "content": value, "source_ids": sources})
        rows.append({"task_id": task["task_id"], "status": "completed", "finish_reason": "stop",
                     "raw_text": json.dumps(native(sections), ensure_ascii=False)})
    return rows


def replace_field(row, heading, **changes):
    output = json.loads(row["raw_text"])
    next(section for section in output["sections"] if section["heading"] == heading).update(changes)
    row["raw_text"] = json.dumps(output, ensure_ascii=False)


class LocalityPreparationTest(unittest.TestCase):
    def test_deterministic_disjoint_triples_and_frozen_first_observations(self):
        source = source_fixture()
        bundle = locality.prepare_locality(source)
        reversed_source = copy.deepcopy(source)
        reversed_source["tasks"].reverse()
        self.assertEqual(bundle, locality.prepare_locality(reversed_source))
        self.assertEqual(bundle["selection"]["block_count"], 2)
        seen = set()
        for block in bundle["selection"]["blocks"]:
            self.assertFalse(seen & set(block["source_clusters"]))
            seen.update(block["source_clusters"])
            tasks = [task for task in bundle["tasks"] if task["block_id"] == block["block_id"]]
            self.assertEqual({task["condition"] for task in tasks}, set(locality.CONDITIONS))
            self.assertEqual(len({task["question_a"] for task in tasks}), 1)
            self.assertEqual(len({task["question_b"] for task in tasks}), 1)
            self.assertEqual(len({task["context_value"] for task in tasks}), 1)
            self.assertEqual(len(tasks[0]["context_value"]), 8)
            for task in tasks:
                for slot in ("a", "b"):
                    self.assertEqual(task["evidence_" + slot]["raw_text"],
                                     source["observations"][task["source_task_ids"][slot]]["raw_text"])
                for slot in ("a", "b", "text"):
                    original = source["task_by_id"][task["source_task_ids"][slot]]
                    self.assertIn(original["cluster_id"], task["source_clusters"])
            self.assertEqual(tasks[0]["evidence_b"], tasks[1]["evidence_b"])
            self.assertEqual(tasks[0]["text_attributes"], tasks[2]["text_attributes"])

    def test_selection_does_not_read_responses_or_labels(self):
        source = source_fixture()
        expected = locality.prepare_locality(source)
        source["responses"] = {"care_opd": {"anything": {"raw_text": "corrupt"}}}
        for gold in source["gold_by_id"].values():
            if gold["answer"] != "ABSTAIN":
                gold["answer"] = "NO" if gold["answer"] == "YES" else "YES"
        actual = locality.prepare_locality(source)
        self.assertEqual(expected["tasks"], actual["tasks"])
        self.assertEqual(expected["selection"], actual["selection"])

    def test_projection_omits_private_and_diagnostic_metadata(self):
        task = locality.prepare_locality(source_fixture())["tasks"][0]
        expected = locality.locality_messages(task)
        changed = copy.deepcopy(task)
        for key in ("condition", "task_id", "block_id", "gold", "answer", "source_clusters", "source_task_ids"):
            changed[key] = "PRIVATE_CANARY"
        changed["text_attributes"][0]["source_qid"] = "PRIVATE_CANARY"
        changed["text_attributes"][0]["source_json_sha256"] = "PRIVATE_CANARY"
        changed["evidence_a"]["gold"] = "PRIVATE_CANARY"
        self.assertEqual(expected, locality.locality_messages(changed))
        self.assertNotIn("PRIVATE_CANARY", json.dumps(expected))
        packet = json.loads(expected[1]["content"])["evidence_packet"]
        self.assertEqual(packet["text:attribute:1"]["record_scope"], "independent_record_c_not_visual_a_or_b")

    def test_wrong_split_fails_closed(self):
        source = source_fixture()
        source["selection"]["dataset"]["source_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            locality.prepare_locality(source)

    def test_replacement_must_keep_opposite_gold_and_different_image(self):
        source = source_fixture()
        for task in source["tasks"]:
            if task["condition"] == "swapped_image":
                source["gold_by_id"][task["task_id"]]["answer"] = "YES"
        with self.assertRaisesRegex(ValueError, "opposite source"):
            locality.prepare_locality(source)
        source = source_fixture()
        for task in source["tasks"]:
            if task["condition"] != "no_image":
                task["image_sha256"] = "same-image"
        with self.assertRaisesRegex(ValueError, "different source image"):
            locality.prepare_locality(source)

    def test_insufficient_independent_attributes_fails_closed(self):
        source = source_fixture()
        for family in source["text_attributes_by_family"]:
            source["text_attributes_by_family"][family] = source["text_attributes_by_family"][family][:1]
        with self.assertRaisesRegex(ValueError, "two distinct"):
            locality.prepare_locality(source)

    def test_no_repair_of_missing_or_misbound_frozen_observation(self):
        source = source_fixture()
        source["observations"] = {}
        with self.assertRaisesRegex(ValueError, "completed frozen observation"):
            locality.prepare_locality(source)
        source = source_fixture()
        for row in source["observations"].values():
            row["task_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            locality.prepare_locality(source)


class LocalityScoringTest(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture()
        self.bundle = locality.prepare_locality(self.source)
        self.rows = perfect_rows(self.bundle)

    def score(self, rows):
        return locality.score_locality(self.bundle["tasks"], self.bundle["gold_by_id"], {"model": rows})["arms"]["model"]

    def test_perfect_joint_and_conditional_retention(self):
        scores = self.score(self.rows)
        self.assertEqual(scores["overall"]["strict_success"]["rate"], 1.0)
        self.assertEqual(scores["overall"]["text_attribute_field_exact"]["denominator"], 12)
        self.assertEqual(scores["clean_replaced_both_a_correct"]["count"], 2)
        for condition in locality.CONDITIONS[1:]:
            self.assertEqual(scores["protected_b"][condition]["protected_b_damaged"], 0)
            self.assertEqual(scores["protected_text_attributes"][condition]["all_attribute_fields"]["rate"], 1)
        self.assertEqual(scores["overall"]["visual_a"]["selective_accuracy_available"]["denominator"], 4)
        self.assertEqual(scores["overall"]["visual_a"]["missing_abstention_correct"]["count"], 2)
        self.assertEqual(scores["overall"]["visual_b"]["selective_accuracy_available"]["denominator"], 6)

    def test_b_damage_and_text_damage_are_separate(self):
        row = next(row for row in self.rows if row["task_id"].endswith(":missing_a"))
        replace_field(row, "visual_b", content="NO")
        replace_field(row, "text_attribute_1", content="MRI")
        scores = self.score(self.rows)
        protected = scores["protected_b"]["missing_a"]
        self.assertEqual(protected["eligible_clean_b_correct"], 2)
        self.assertEqual(protected["protected_b_damaged"], 1)
        self.assertEqual(protected["damage_transition_counts"], {"YES->NO": 1})
        text = scores["protected_text_attributes"]["missing_a"]
        self.assertEqual(text["text_attribute_1"]["damaged"], 1)
        self.assertEqual(text["text_attribute_2"]["damaged"], 0)

    def test_missing_malformed_and_truncated_keep_denominators(self):
        rows = copy.deepcopy(self.rows)
        rows.pop()
        rows[0]["raw_text"] = "{broken"
        rows[1]["finish_reason"] = "length"
        scores = self.score(rows)
        self.assertEqual(scores["overall"]["n"], 6)
        self.assertEqual(scores["overall"]["visual_b"]["unknown"]["count"], 2)
        self.assertEqual(scores["overall"]["strict_success"]["count"], 3)
        self.assertEqual(scores["overall"]["truncated"]["count"], 1)

    def test_sources_schema_and_context_are_strict_not_answer_accuracy(self):
        rows = copy.deepcopy(self.rows)
        replace_field(rows[0], "visual_b", source_ids=["image:a"])
        replace_field(rows[1], "context_value", content="incorrect")
        value = json.loads(rows[2]["raw_text"])
        value["unexpected"] = "extra"
        rows[2]["raw_text"] = json.dumps(value)
        scores = self.score(rows)
        self.assertEqual(scores["overall"]["visual_joint_correct"]["count"], 6)
        self.assertEqual(scores["overall"]["strict_success"]["count"], 3)

    def test_absent_clean_success_gives_none_not_zero_conditional_rate(self):
        rows = [row for row in self.rows if not row["task_id"].endswith(":clean")]
        scores = self.score(rows)
        self.assertIsNone(scores["protected_b"]["missing_a"]["conditional_protected_b_correct_rate"])
        self.assertEqual(scores["protected_b"]["missing_a"]["eligible_clean_b_correct"], 0)

    def test_duplicate_responses_and_reused_original_clusters_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate response"):
            self.score(self.rows + self.rows[:1])
        tasks = copy.deepcopy(self.bundle["tasks"])
        tasks[-1]["source_clusters"] = tasks[0]["source_clusters"]
        with self.assertRaisesRegex(ValueError, "cluster reused"):
            locality.score_locality(tasks, self.bundle["gold_by_id"], {})


class IndependentBaselineTest(unittest.TestCase):
    def test_visual_only_no_gold_and_no_granted_text_success(self):
        source = source_fixture()
        bundle = locality.prepare_locality(source)
        source.pop("gold_by_id")
        rows = locality.independent_baseline(source, bundle["tasks"])
        metrics = locality.score_locality(bundle["tasks"], bundle["gold_by_id"], {"reuse": rows})["arms"]["reuse"]
        self.assertEqual(metrics["overall"]["visual_joint_correct"]["rate"], 1)
        self.assertIsNone(metrics["overall"]["text_attribute_field_exact"]["count"])
        self.assertEqual(metrics["overall"]["text_attribute_field_exact"]["denominator"], 0)
        self.assertEqual(metrics["overall"]["strict_success"]["status"], "NOT_EVALUATED")
        self.assertIsNone(metrics["overall"]["strict_success"]["rate"])
        self.assertIn("NOT_EVALUATED", metrics["comparison_scope"])
        self.assertTrue(all(row["new_model_inference_calls"] == 0 for row in rows))

    def test_parse_failure_and_illegal_source_not_repaired(self):
        source = source_fixture()
        bundle = locality.prepare_locality(source)
        first = bundle["tasks"][0]
        source["responses"]["care_opd"][first["source_task_ids"]["a"]]["raw_text"] = "bad JSON"
        donor = source["responses"]["care_opd"][first["source_task_ids"]["b"]]
        replace_field(donor, "visual_answer", source_ids=["invented:source"])
        rows = locality.independent_baseline(source, bundle["tasks"])
        output = json.loads(rows[0]["raw_text"])
        self.assertEqual(output["sections"][0]["content"], "UNKNOWN")
        self.assertEqual(output["sections"][1]["source_ids"], ["invented:source"])
        self.assertEqual(output["sections"][-1]["content"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
