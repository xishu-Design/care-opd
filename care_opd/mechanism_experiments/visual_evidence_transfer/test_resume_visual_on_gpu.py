"""Synthetic CPU-only tests; never scientific experiment results."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import common
import evaluate
import resume_visual_on_gpu as resume
import test_gpu45 as fixtures

EmptyGenerator = fixtures.EmptyGenerator
device = fixtures.device


class MigrationTest(unittest.TestCase):
    setUp = fixtures.GPU45Test.setUp
    bind = fixtures.GPU45Test.bind

    def fixture(self):
        first = self.tasks[0]
        first.update(condition="no_image", image_path=None, image_sha256=None)
        second = {**first, "task_id": "second:no_image", "family_id": "second", "cluster_id": "second"}
        self.tasks = [first, second]
        self.manifest = self.bind()
        with patch.object(evaluate, "HFGenerator", EmptyGenerator):
            evaluate.run_phase(self.config, self.manifest, self.tasks, "observe")
            evaluate.run_phase(self.config, self.manifest, self.tasks, "direct")
            for arm in common.TEXT_ARMS:
                evaluate.run_phase(self.config, self.manifest, self.tasks, "generate", arm)
        self.run = Path(self.config["run_root"])
        self.care = self.run / "arms" / "care_opd" / "responses.jsonl"
        self.care.write_bytes(self.care.read_bytes().splitlines(keepends=True)[0])
        self.original = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        EmptyGenerator.calls = 0

    def execute(self, phase="resume", actual=None):
        with patch.dict(os.environ), \
                patch.object(common, "load_tasks", return_value=(self.tasks, self.selection)), \
                patch.object(evaluate, "verify_images"), \
                patch.object(evaluate, "check_stack"), \
                patch.object(evaluate, "environment", return_value=actual or device("0")), \
                patch.object(evaluate, "HFGenerator", EmptyGenerator), \
                patch("score_results.score_results", return_value={"status": "COMPLETE"}):
            resume.execute(self.config, "0", phase)

    def test_resume_only_missing_preserves_invalid_first_output_and_manifest(self):
        self.fixture()
        self.execute()
        self.assertEqual(EmptyGenerator.calls, 1)
        for name, payload in self.original.items():
            if Path(name) == self.care:
                self.assertTrue(self.care.read_bytes().startswith(payload))
            else:
                self.assertEqual(Path(name).read_bytes(), payload)
        rows = common.read_jsonl(self.care)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["raw_text"], "")
        audit = common.read_jsonl(self.run / "runtime" / "device_migrations.jsonl")
        self.assertEqual([a["event"] for a in audit], ["started", "completed"])
        self.assertEqual(audit[0]["actual_environment"]["cuda_visible_devices"], "0")
        self.assertEqual(audit[0]["original_environment"]["cuda_visible_devices"], "5")
        self.assertEqual(audit[0]["missing_task_ids"], ["second:no_image"])
        self.assertFalse((self.run / ".execution.lock").exists())
        self.execute()
        self.assertEqual(EmptyGenerator.calls, 1)

    def test_preflight_writes_no_audit_or_outputs(self):
        self.fixture()
        self.execute("preflight")
        self.assertEqual(EmptyGenerator.calls, 0)
        after = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(after, self.original)

    def test_changed_generation_is_rejected(self):
        self.fixture()
        self.config["generation"]["max_new_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "config differs"):
            self.execute()
        self.assertEqual(EmptyGenerator.calls, 0)

    def test_environment_drift_is_rejected_except_identity(self):
        for key, value in (("torch", "other"), ("gpu_name", "other"),
                           ("gpu_total_memory_bytes", 1), ("cuda_visible_devices", "1")):
            actual = device("0")
            actual[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume.validate_device(device("5"), actual, "0")
        resume.validate_device(device("5"), device("0"), "0")

    def test_partial_other_arm_is_rejected(self):
        self.fixture()
        path = self.run / "arms" / "gkd_full" / "responses.jsonl"
        path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "gkd_full incomplete"):
            self.execute()
        self.assertEqual(EmptyGenerator.calls, 0)

    def test_saved_row_tampering_is_rejected(self):
        self.fixture()
        row = common.read_jsonl(self.care)[0]
        row["input_sha256"] = "changed"
        self.care.write_text(common.canonical(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            self.execute()
        self.assertEqual(EmptyGenerator.calls, 0)

    def test_protocol_hash_tampering_is_rejected(self):
        self.fixture()
        path = self.run / "run_manifest.json"
        doc = common.read_json(path)
        doc["protocol_sha256"] = "0" * 64
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash closure"):
            self.execute()

    def test_no_terminal_newline_is_rejected_before_append(self):
        self.fixture()
        payload = self.care.read_bytes().rstrip(b"\n")
        self.care.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "terminal newline"):
            self.execute()
        self.assertEqual(self.care.read_bytes(), payload)
        self.assertEqual(EmptyGenerator.calls, 0)

    def test_code_change_is_rejected(self):
        self.fixture()
        with patch.object(evaluate, "source_fingerprints", return_value={}):
            with self.assertRaisesRegex(ValueError, "code_sha256 differs"):
                self.execute()
        self.assertEqual(EmptyGenerator.calls, 0)

    def test_lock_never_removed_for_other_owner(self):
        self.fixture()
        lock = self.run / ".execution.lock"
        lock.write_text("other-owner\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "run lock exists"):
            self.execute()
        self.assertEqual(lock.read_text(encoding="utf-8"), "other-owner\n")


if __name__ == "__main__":
    unittest.main()
