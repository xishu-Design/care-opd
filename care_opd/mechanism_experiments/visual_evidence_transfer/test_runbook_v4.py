"""CPU-only V4 initialization smoke tests; no actual models or GPU work."""
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import common
import evaluate


SOURCE = Path(__file__).resolve().parent


def python_blocks(path):
    return re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", path.read_text(encoding="utf-8"), re.S)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class RunbookV4Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old = self.root / "v3"
        self.work = self.root / "v4"
        tasks = self.old / "tasks_validation10"
        tasks.mkdir(parents=True)
        common.append_row(tasks / "tasks.jsonl", {
            "task_id": "synthetic:no_image", "family_id": "synthetic", "cluster_id": "synthetic",
            "question": "合成测试问题", "context_value": "SYNTH001", "condition": "no_image",
            "image_path": None, "image_sha256": None,
        })
        common.append_row(tasks / "gold.private.jsonl", {"synthetic_test_only": True})
        self.selection = {
            "dataset": {"split": "validation", "source_split": "validation"},
            "tasks_sha256": common.file_sha256(tasks / "tasks.jsonl"),
            "gold_sha256": common.file_sha256(tasks / "gold.private.jsonl"),
        }
        common.write_json_once(tasks / "selection_manifest.json", self.selection)
        models = {}
        for index, name in enumerate(("vlm", *common.TEXT_ARMS)):
            path = self.root / "models" / name
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(bytes([index + 1]) * 2048)
            models[name] = str(path.resolve())
        self.config = {
            "schema_version": "care_opd.visual_transfer.config.v1",
            "tasks_dir": str(tasks.resolve()), "run_root": str((self.old / "runs_test").resolve()),
            "models": models, "generation": dict(evaluate.DEFAULT_GENERATION),
        }
        self.manifest_path = self.old / "runs_test" / "run_manifest.json"
        common.write_json_once(self.manifest_path, {"config": self.config})

    def initialize(self, *, work=None):
        blocks = python_blocks(SOURCE / "run_validation_v4.sh")
        self.assertEqual(len(blocks), 1)
        return subprocess.run([sys.executable, "-B", "-c", blocks[0], str(SOURCE),
                               str(self.old), str(work or self.work)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_init_inherits_assets_and_budget_then_resumes_without_changing_old(self):
        before = snapshot(self.old)
        result = self.initialize()
        self.assertEqual(result.returncode, 0, result.stderr)
        config_path = self.work / "config_validation10.json"
        config = evaluate.configuration(config_path)
        self.assertEqual(config["models"], self.config["models"])
        self.assertEqual(config["generation"], self.config["generation"])
        self.assertEqual(config["protocol_version"], common.PROTOCOL_VERSION)
        self.assertEqual(config["evaluation_use"], "validation")
        self.assertEqual(config["tasks_dir"], self.config["tasks_dir"])
        self.assertEqual(Path(config["run_root"]), self.work / "runs_validation10")
        self.assertEqual(snapshot(self.old), before)
        frozen = snapshot(self.work)
        repeated = self.initialize()
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(snapshot(self.work), frozen)
        self.assertFalse((self.work / "runs_validation10" / "observations").exists())

    def test_fallback_only_when_prior_manifest_is_absent(self):
        self.manifest_path.unlink()  # Only this test's synthetic temporary fixture.
        common.write_json_once(self.old / "config_validation10.json", self.config)
        result = self.initialize()
        self.assertEqual(result.returncode, 0, result.stderr)
        provenance = common.read_json(self.work / "config_validation10.provenance.json")
        self.assertEqual(provenance["source_kind"], "prior_validation_config_fallback_manifest_absent")

    def test_invalid_manifest_does_not_silently_fall_back(self):
        self.manifest_path.write_text("{broken", encoding="utf-8")
        common.write_json_once(self.old / "config_validation10.json", self.config)
        result = self.initialize()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.work.exists())

    def test_overlap_old_outputs_and_changed_budget_fail_closed(self):
        before = snapshot(self.old)
        result = self.initialize(work=self.old / "nested_v4")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot(self.old), before)
        run_root = self.work / "runs_validation10"
        run_root.mkdir(parents=True)
        marker = run_root / "old_result.json"
        marker.write_text("{\"keep\":true}", encoding="utf-8")
        frozen = snapshot(self.work)
        result = self.initialize()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot(self.work), frozen)

    def test_existing_config_cannot_be_overwritten(self):
        result = self.initialize()
        self.assertEqual(result.returncode, 0, result.stderr)
        frozen = snapshot(self.work)
        changed = copy.deepcopy(self.config)
        changed["generation"]["max_new_tokens"] += 1
        self.manifest_path.write_text(json.dumps({"config": changed}), encoding="utf-8")
        result = self.initialize()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot(self.work), frozen)

    def test_all_documented_python_blocks_compile(self):
        for path in (SOURCE / "run_validation_v4.sh", SOURCE / "SERVER_COMMANDS_GPU45_V4.md"):
            blocks = python_blocks(path)
            self.assertTrue(blocks)
            for index, block in enumerate(blocks):
                compile(block, f"{path.name}:python:{index}", "exec")


if __name__ == "__main__":
    unittest.main()
