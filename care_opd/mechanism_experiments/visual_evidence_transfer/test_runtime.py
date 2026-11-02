import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
import evaluate


def task(condition="clean"):
    return {"task_id": "example:" + condition, "family_id": "example", "cluster_id": "image-component",
            "condition": condition, "question": "图像中是否存在某器官？", "context_value": "A7C02B91",
            "image_path": None if condition == "no_image" else "/unused/image.jpg",
            "image_sha256": None if condition == "no_image" else "image-hash"}


class FakeGenerator:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, messages, image_path, max_tokens):
        type(self).calls += 1
        return {"raw_text": "", "finish_reason": "length", "latency_seconds": 0.01,
                "prompt_tokens": 9, "completion_tokens": max_tokens, "image_seen": image_path is not None}

    def close(self):
        pass


class RuntimeTest(unittest.TestCase):
    def test_prompt_whitelist_and_no_swap_label(self):
        item = task("swapped_image")
        item.update(answer="GOLD_SENTINEL", counterpart_qid="PRIVATE_QUESTION", source_image_id="LABELLED_IMAGE")
        obs = {"status": "completed", "raw_text": "观察内容"}
        for messages in (common.observation_messages(item), common.answer_messages(item, obs),
                         common.answer_messages(item, None, direct=True)):
            text = common.canonical(messages)
            for excluded in ("GOLD_SENTINEL", "PRIVATE_QUESTION", "LABELLED_IMAGE", "swapped_image", "/unused/"):
                self.assertNotIn(excluded, text)

    def test_contract_and_question_differ_from_observation(self):
        item = task()
        obs = {"status": "completed", "raw_text": "观察内容"}
        payload = json.loads(common.answer_messages(item, obs)[1]["content"])
        self.assertEqual(payload["evidence_packet"]["text:context"]["context_value"], item["context_value"])
        self.assertEqual(payload["evidence_packet"]["image:current"]["observation"], "观察内容")
        self.assertNotIn(item["context_value"], common.canonical(common.observation_messages(item)))

    def test_unknown_missing_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            common.append_row(path, {"task_id": "a"})
            common.append_row(path, {"task_id": "a"})
            with self.assertRaises(ValueError):
                common.unique_rows(path)

    def test_immutable_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            common.write_json_once(path, {"a": 1})
            common.write_json_once(path, {"a": 1})
            with self.assertRaises(ValueError):
                common.write_json_once(path, {"a": 2})

    def test_lock_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with common.run_lock(root):
                with self.assertRaises(RuntimeError):
                    with common.run_lock(root):
                        pass
            self.assertFalse((root / ".execution.lock").exists())

    def test_freeze_bad_first_output_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"run_root": folder, "models": {"vlm": "fake"}, "generation": evaluate.DEFAULT_GENERATION}
            manifest = {"protocol_sha256": "protocol", "model_fingerprints": {"vlm": "fingerprint"}}
            tasks = [task()]
            FakeGenerator.calls = 0
            with patch.object(evaluate, "HFGenerator", FakeGenerator):
                evaluate.run_phase(config, manifest, tasks, "direct")
                evaluate.run_phase(config, manifest, tasks, "direct")
                self.assertEqual(FakeGenerator.calls, 1)
                path = Path(folder) / "arms/vlm_direct/responses.jsonl"
                row = common.read_jsonl(path)[0]
                self.assertEqual(row["status"], "completed")
                self.assertEqual(row["raw_text"], "")
                self.assertEqual(row["finish_reason"], "length")
                tasks[0]["question"] += "改变"
                with self.assertRaises(ValueError):
                    evaluate.run_phase(config, manifest, tasks, "direct")

    def test_no_image_observation_has_no_model_call(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"run_root": folder, "models": {"vlm": "fake"}, "generation": evaluate.DEFAULT_GENERATION}
            manifest = {"protocol_sha256": "p", "model_fingerprints": {"vlm": "v"}}
            with patch.object(evaluate, "HFGenerator") as generator:
                evaluate.run_phase(config, manifest, [task("no_image")], "observe")
                generator.assert_not_called()
            row = common.read_jsonl(Path(folder) / "observations/observations.jsonl")[0]
            self.assertFalse(row["image_seen"])

    def test_infrastructure_failure_preserved_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"run_root": folder, "models": {"vlm": "fake"}, "generation": evaluate.DEFAULT_GENERATION}
            manifest = {"protocol_sha256": "p", "model_fingerprints": {"vlm": "v"}}
            with patch.object(evaluate, "HFGenerator", side_effect=RuntimeError("OOM")):
                with self.assertRaises(RuntimeError):
                    evaluate.run_phase(config, manifest, [task()], "direct")
            root = Path(folder) / "arms/vlm_direct"
            self.assertFalse((root / "responses.jsonl").exists())
            self.assertEqual(common.read_jsonl(root / "infrastructure_errors.jsonl")[0]["status"], "infra_error")

    def test_weight_hash_detects_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text('{}', encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"a" * 2048)
            old = evaluate.fingerprint_model(root)
            (root / "model.safetensors").write_bytes(b"b" * 2048)
            new = evaluate.fingerprint_model(root, previous=old, full=True)
            self.assertNotEqual(old["sha256"], new["sha256"])


if __name__ == "__main__":
    unittest.main()
