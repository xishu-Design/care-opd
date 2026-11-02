"""Synthetic CPU tests; fixtures are not experimental measurements."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
import followup_attributes as attributes
import followup_run as runner


def annotation_source(root):
    path = root / "validation.json"
    rows = [
        {"qid": 1, "img_name": "xmlab1/source.jpg", "question": "目标是否存在？", "answer": "是", "base_type": "vqa", "q_lang": "zh", "content_type": "Presence"},
        {"qid": 2, "img_name": "xmlab1/source.jpg", "question": "是什么成像类型？", "answer": "CT", "base_type": "vqa", "q_lang": "zh", "content_type": "Modality"},
        {"qid": 3, "img_name": "xmlab1/source.jpg", "question": "主要部位是什么？", "answer": "胸部", "base_type": "vqa", "q_lang": "zh", "content_type": "Organ"},
        {"qid": 4, "img_name": "xmlab1/source.jpg", "question": "这是正常的吗？", "answer": "否", "base_type": "vqa", "q_lang": "zh", "content_type": "Normal"},
        {"qid": 5, "img_name": "xmlab2/source.jpg", "question": "部位？", "answer": "腹部", "base_type": "vqa", "q_lang": "zh", "content_type": "Organ"},
    ]
    runner.json_once(path, rows)
    return {"selection": {"dataset": {"source_split": "validation", "source_json": str(path),
                                      "source_json_sha256": common.file_sha256(path)}},
            "tasks": [{"task_id": "t", "dataset_qid": "1", "question": "目标是否存在？", "condition": "clean", "family_id": "f", "cluster_id": "c"}]}


class AttributeTests(unittest.TestCase):
    def test_multiple_source_attributes_exclude_binary_and_other_image(self):
        with tempfile.TemporaryDirectory() as directory:
            source = attributes.attach_attributes(annotation_source(Path(directory)))
            fields = source["text_attributes_by_family"]["f"]
            self.assertEqual({x["value"] for x in fields}, {"CT", "胸部"})
            self.assertEqual({x["source_qid"] for x in fields}, {"2", "3"})
            self.assertTrue(all(x["source_json_sha256"] for x in fields))

    def test_reject_annotation_drift_and_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            source = annotation_source(Path(directory))
            altered = copy.deepcopy(source)
            altered["selection"]["dataset"]["source_json_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash"):
                attributes.attach_attributes(altered)
            source["selection"]["dataset"]["source_split"] = "test"
            with self.assertRaisesRegex(ValueError, "validation"):
                attributes.attach_attributes(source)

    def test_deterministic_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = annotation_source(Path(directory))
            self.assertEqual(attributes.attach_attributes(copy.deepcopy(source)),
                             attributes.attach_attributes(copy.deepcopy(source)))


class RunnerTests(unittest.TestCase):
    def test_prepare_stub_inference_score_integration(self):
        from test_followup_locality import source_fixture, perfect_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_fixture()
            source["config"] = {"tasks_dir": str(root / "old/tasks"), "run_root": str(root / "old/run"),
                                "models": {arm: "fake-model-" + arm for arm in common.TEXT_ARMS},
                                "generation": dict(runner.evaluate.DEFAULT_GENERATION)}
            source["manifest"] = {"model_fingerprints": {arm: "synthetic-model" for arm in common.TEXT_ARMS}}
            source["text_attribute_provenance"] = {"synthetic_test_only": True}
            source["responses"] = {arm: copy.deepcopy(source["responses"]["care_opd"]) for arm in common.TEXT_ARMS}
            sentinel = root / "old/run/sentinel"
            runner.write_once(sentinel, b"old outputs must not change\n")
            before = (sentinel.read_bytes(), sentinel.stat().st_mtime_ns)
            work = root / "new"
            with patch.object(runner, "export_audit", return_value={}), \
                 patch.object(runner.followup_attributes, "attach_attributes", side_effect=lambda x: x):
                runner.prepare(root / "old/config.json", work, source=source)
                runner.prepare(root / "old/config.json", work, source=source)
            config, tasks = runner.load_work(work)
            with patch.object(runner.evaluate, "check_stack"), \
                 patch.object(runner.evaluate, "environment", return_value={"synthetic_test_only": True}), \
                 patch.object(runner.evaluate, "fingerprint_model", return_value={"sha256": "synthetic-model", "files": {}, "stat": {}}):
                manifest = runner.bind_runtime(work, config)
                self.assertEqual(manifest, runner.bind_runtime(work, config))
            gold = common.unique_rows(work / "gold.private.jsonl")
            output_by_id = {r["task_id"]: r for r in perfect_rows({"tasks": tasks, "gold_by_id": gold})}
            outputs_by_message = {common.canonical(runner.followup_locality.locality_messages(t)): output_by_id[t["task_id"]]
                                  for t in tasks}
            class Fake:
                def __init__(self, *args, **kwargs):
                    pass
                def generate(self, messages, image, budget):
                    return outputs_by_message[common.canonical(messages)]
                def close(self):
                    pass
            for arm in common.TEXT_ARMS:
                runner.generate_arm(work, config, tasks, manifest, arm, Fake)
            report = runner.score(work)
            self.assertEqual(report["arms"]["care_opd"]["overall"]["strict_success"]["rate"], 1)
            summary = next((work / "summary").iterdir())
            self.assertTrue((summary / "locality_metrics.csv").is_file())
            self.assertTrue((summary / "care_opd_protection.csv").is_file())
            bytes_before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in summary.iterdir()}
            runner.score(work)
            self.assertEqual(bytes_before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in summary.iterdir()})
            self.assertEqual(before, (sentinel.read_bytes(), sentinel.stat().st_mtime_ns))
            for arm in common.TEXT_ARMS:
                def never_load(*args, **kwargs):
                    self.fail("completed rows must not load a model again")
                runner.generate_arm(work, config, tasks, manifest, arm, never_load)

    def test_write_once_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.json"
            runner.json_once(path, {"a": 1})
            runner.json_once(path, {"a": 1})
            with self.assertRaisesRegex(ValueError, "frozen"):
                runner.json_once(path, {"a": 2})

    def test_output_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {"config": {"tasks_dir": str(root / "old/tasks"), "run_root": str(root / "old/run")}}
            runner.isolated_output(root / "new", source)
            for target in (root / "old", root / "old/run/sub", root / "old/tasks"):
                with self.assertRaises(ValueError):
                    runner.isolated_output(target, source)

    @patch.object(runner.followup_locality, "locality_messages", lambda task: [{"role": "user", "content": task["task_id"]}])
    def test_resume_freezes_invalid_first_output_and_skips_it(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            tasks = [{"task_id": "a", "block_id": "x", "condition": "clean"},
                     {"task_id": "b", "block_id": "x", "condition": "missing_a"}]
            config = {"models": {"care_opd": "fake"}, "generation": {"max_new_tokens": 1024}}
            manifest = {"protocol_sha256": "protocol", "model_fingerprints": {"care_opd": "checkpoint"}}
            calls = []
            class Fake:
                def __init__(self, *args, **kwargs):
                    pass
                def generate(self, messages, image, budget):
                    ident = messages[0]["content"]
                    calls.append(ident)
                    if ident == "b" and calls.count("b") == 1:
                        raise RuntimeError("simulated infrastructure error")
                    return {"raw_text": "malformed", "finish_reason": "length"}
                def close(self):
                    pass
            with self.assertRaisesRegex(RuntimeError, "technical failure"):
                runner.generate_arm(work, config, tasks, manifest, "care_opd", Fake)
            saved = common.read_jsonl(work / "arms/care_opd/responses.jsonl")
            self.assertEqual(saved[0]["raw_text"], "malformed")
            runner.generate_arm(work, config, tasks, manifest, "care_opd", Fake)
            self.assertEqual(calls, ["a", "b", "b"])
            self.assertEqual(len(runner.checked_rows(work, tasks, "care_opd", manifest)), 2)
            changed = copy.deepcopy(tasks)
            changed[0]["condition"] = "replaced_a"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                runner.checked_rows(work, changed, "care_opd", manifest)

    def test_load_work_hashes_gold_without_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            runner.write_once(work / "gold.private.jsonl", b"intentionally not JSON\n")
            runner.jsonl_once(work / "tasks.jsonl", [{"task_id": "a"}])
            config = {"schema_version": runner.SCHEMA, "formal_claim_allowed": False,
                      "task_count": 1, "code_sha256": runner.code_hashes(),
                      "files": {name: common.file_sha256(work / name) for name in ("tasks.jsonl", "gold.private.jsonl")}}
            runner.json_once(work / "experiment.json", config)
            with patch.object(runner.followup_locality, "locality_messages", return_value=[]):
                _, tasks = runner.load_work(work)
            self.assertEqual(len(tasks), 1)

    def test_unterminated_ledger_rejected_before_append(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            path = work / "arms/care_opd/responses.jsonl"
            runner.write_once(path, b'{"task_id":"a"}')
            with self.assertRaisesRegex(ValueError, "unterminated"):
                runner.checked_rows(work, [], "care_opd", {})
            self.assertEqual(path.read_bytes(), b'{"task_id":"a"}')


if __name__ == "__main__":
    unittest.main()
