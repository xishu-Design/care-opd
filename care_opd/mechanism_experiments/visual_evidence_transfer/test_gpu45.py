"""CPU-only coordinator contract tests, never model evaluation results."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import common
import evaluate
import gpu45


def device(gpu):
    return {
        "python": "3.10.19", "torch": "2.7.1+cu118", "transformers": "5.14.1",
        "Pillow": "11.0.0", "accelerate": "1.10.0", "cuda_runtime": "11.8",
        "cuda_visible_devices": gpu, "gpu_name": "synthetic A800",
        "gpu_total_memory_bytes": 80 * 1024 ** 3, "gpu_uuid": "synthetic-" + gpu,
    }


class EmptyGenerator:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, messages, image_path, max_tokens):
        type(self).calls += 1
        return {"raw_text": "", "finish_reason": "length", "latency_seconds": 0.01,
                "prompt_tokens": 4, "completion_tokens": max_tokens,
                "image_seen": image_path is not None, "peak_gpu_memory_bytes": 0}

    def close(self):
        pass


class GPU45Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tasks_root = self.root / "tasks"
        tasks_root.mkdir()
        self.selection = {"tasks_sha256": "synthetic-tasks", "gold_sha256": "synthetic-gold",
                          "dataset": {"source_split": "validation"}}
        common.write_json_once(tasks_root / "selection_manifest.json", self.selection)
        self.tasks = [{"task_id": "example:clean", "family_id": "example", "cluster_id": "example-image",
                       "condition": "clean", "question": "图中是否有器官？", "context_value": "A7B9",
                       "image_path": "/synthetic/image.jpg", "image_sha256": "synthetic-image"}]
        models = {}
        for index, arm in enumerate(("vlm", *common.TEXT_ARMS)):
            path = self.root / "models" / arm
            path.mkdir(parents=True)
            (path / "config.json").write_text('{}', encoding="utf-8")
            (path / "model.safetensors").write_bytes(bytes([index + 1]) * 2048)
            models[arm] = str(path)
        self.config = {
            "schema_version": "care_opd.visual_transfer.config.v1", "models": models,
            "protocol_version": common.PROTOCOL_VERSION, "evaluation_use": "validation",
            "tasks_dir": str(tasks_root), "run_root": str(self.root / "run"),
            "generation": dict(evaluate.DEFAULT_GENERATION),
        }
        self.environments = {gpu: device(gpu) for gpu in ("4", "5")}
        self.fleet = gpu45.fleet_environment(self.environments)

    def bind(self, fleet=None):
        return evaluate.bind_run(self.config, self.tasks, self.selection,
                                 execution_environment=self.fleet if fleet is None else fleet,
                                 extra_sources=gpu45.EXTRA_SOURCES)

    def test_schedule_covers_each_arm_exactly_once(self):
        self.assertEqual(set(gpu45.GPU_PLAN), {"4", "5"})
        self.assertEqual(tuple(gpu45.GPU_PLAN["4"]), ("direct", "gkd_full"))
        self.assertEqual(tuple(gpu45.GPU_PLAN["5"]), ("clean_sft", "tropd_full", "care_opd"))
        phases = [phase for phases in gpu45.GPU_PLAN.values() for phase in phases]
        arms = ["vlm_direct" if phase == "direct" else phase for phase in phases]
        self.assertCountEqual(arms, common.ARMS)
        self.assertEqual(len(arms), len(set(arms)))
        self.assertEqual(self.fleet["devices"], self.environments)
        self.assertIn("schedule", self.fleet)

    def test_fleet_rejects_missing_wrong_or_mismatched_device(self):
        for key, value in (("cuda_visible_devices", "4"), ("torch", "different"),
                           ("gpu_name", "different"), ("gpu_total_memory_bytes", 1)):
            environments = copy.deepcopy(self.environments)
            environments["5"][key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                gpu45.fleet_environment(environments)
        with self.assertRaises((ValueError, RuntimeError)):
            gpu45.fleet_environment({"4": self.environments["4"]})

    def test_injected_fleet_binds_without_importing_gpu_stack(self):
        with patch.object(evaluate, "environment", side_effect=AssertionError("must not probe parent CUDA")):
            manifest = self.bind()
            self.assertEqual(self.bind(), manifest)
        self.assertEqual(manifest["environment"], self.fleet)
        self.assertEqual(manifest["code_sha256"]["gpu45.py"],
                         common.file_sha256(Path(gpu45.__file__)))
        changed = copy.deepcopy(self.fleet)
        changed["devices"]["5"]["gpu_uuid"] = "replacement-device"
        with self.assertRaises(ValueError):
            self.bind(changed)

    def test_original_single_gpu_binding_default_is_unchanged(self):
        with patch.object(evaluate, "environment", return_value=device("4")) as probe:
            manifest = evaluate.bind_run(self.config, self.tasks, self.selection)
        probe.assert_called_once_with()
        self.assertEqual(manifest["environment"], device("4"))
        self.assertNotIn("gpu45.py", manifest["code_sha256"])

    def test_worker_environment_drift_is_rejected(self):
        manifest = self.bind()
        with patch.object(evaluate, "environment", return_value=device("5")):
            gpu45.verify_worker(self.config, manifest, "5")
        changed = device("5")
        changed["gpu_uuid"] = "replacement-device"
        with patch.object(evaluate, "environment", return_value=changed):
            with self.assertRaises((ValueError, RuntimeError)):
                gpu45.verify_worker(self.config, manifest, "5")

    def test_worker_checkpoint_content_drift_is_rejected(self):
        manifest = self.bind()
        weights = Path(self.config["models"]["care_opd"]) / "model.safetensors"
        weights.write_bytes(b"changed-checkpoint" * 256)
        with patch.object(evaluate, "environment", return_value=device("5")):
            with self.assertRaises((ValueError, RuntimeError)):
                gpu45.verify_worker(self.config, manifest, "5")

    def test_worker_code_drift_is_rejected(self):
        manifest = self.bind()
        changed = copy.deepcopy(manifest)
        changed["code_sha256"]["gpu45.py"] = "different-source"
        with patch.object(evaluate, "environment", return_value=device("4")):
            with self.assertRaises((ValueError, RuntimeError)):
                gpu45.verify_worker(self.config, changed, "4")

    def test_parent_lock_and_first_outputs_survive_both_lanes(self):
        root = Path(self.config["run_root"])
        EmptyGenerator.calls = 0
        with common.run_lock(root):
            manifest = self.bind()
            with self.assertRaises(RuntimeError):
                with common.run_lock(root):
                    self.fail("a second coordinator must not acquire the parent lock")
            with patch.object(evaluate, "HFGenerator", EmptyGenerator):
                for _ in range(2):
                    # Workers reuse run_phase under the parent's ownership, not its CLI lock.
                    evaluate.run_phase(self.config, manifest, self.tasks, "observe")
                    for gpu, phases in gpu45.GPU_PLAN.items():
                        with patch.object(evaluate, "environment", return_value=device(gpu)):
                            gpu45.verify_worker(self.config, manifest, gpu)
                        for phase in phases:
                            evaluate.run_phase(self.config, manifest, self.tasks,
                                               "direct" if phase == "direct" else "generate",
                                               None if phase == "direct" else phase)
                self.assertEqual(EmptyGenerator.calls, 1 + len(common.ARMS))
            self.assertTrue((root / ".execution.lock").is_file())
        self.assertFalse((root / ".execution.lock").exists())
        for arm in common.ARMS:
            rows = common.read_jsonl(root / "arms" / arm / "responses.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["raw_text"], "")
            self.assertEqual(rows[0]["finish_reason"], "length")
            self.assertEqual(rows[0]["status"], "completed")

    def test_execute_holds_parent_lock_through_probe_workers_and_score(self):
        lock = Path(self.config["run_root"]) / ".execution.lock"
        events = []

        def probe(config, gpu):
            self.assertTrue(lock.is_file())
            events.append("probe" + gpu)
            return device(gpu)

        def workers(config, manifest, tasks, plan, *, stage_label):
            self.assertTrue(lock.is_file())
            events.append(stage_label)
            self.assertEqual(plan, {"4": ("observe",)} if stage_label == "observe" else gpu45.GPU_PLAN)

        def score(**kwargs):
            self.assertTrue(lock.is_file())
            events.append("score")
            return {"status": "COMPLETE"}

        with patch.object(gpu45, "load_tasks", return_value=(self.tasks, self.selection)), \
                patch.object(evaluate, "verify_images"), \
                patch.object(gpu45, "inspect_environment", side_effect=probe), \
                patch.object(gpu45, "run_workers", side_effect=workers), \
                patch("score_results.score_results", side_effect=score):
            gpu45.execute(self.config, "all")
        self.assertEqual(events, ["probe4", "probe5", "observe", "answers", "score"])
        self.assertFalse(lock.exists())

    def test_observation_failure_prevents_answer_workers_and_score(self):
        with patch.object(gpu45, "load_tasks", return_value=(self.tasks, self.selection)), \
                patch.object(evaluate, "verify_images"), \
                patch.object(gpu45, "inspect_environment", side_effect=lambda config, gpu: device(gpu)), \
                patch.object(gpu45, "run_workers", side_effect=RuntimeError("synthetic observation failure")) as workers, \
                patch("score_results.score_results") as score:
            with self.assertRaisesRegex(RuntimeError, "observation failure"):
                gpu45.execute(self.config, "all")
            workers.assert_called_once()
            score.assert_not_called()
        self.assertFalse((Path(self.config["run_root"]) / ".execution.lock").exists())

    def test_worker_reuses_parent_lock_without_reentering_it(self):
        root = Path(self.config["run_root"])
        with common.run_lock(root):
            manifest = self.bind()
            with patch.dict(os.environ), \
                    patch.object(evaluate, "environment", return_value=device("4")), \
                    patch.object(evaluate, "verify_images"), \
                    patch.object(evaluate, "run_phase") as run_phase:
                gpu45._worker(self.config, manifest, self.tasks, "4", ("direct",), str(root / "lane.log"))
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "4")
                run_phase.assert_called_once_with(self.config, manifest, self.tasks, "direct")
            self.assertTrue((root / ".execution.lock").is_file())

    def test_spawn_workers_all_join_even_if_one_lane_fails(self):
        root = Path(self.config["run_root"])
        children = [Mock(pid=104, exitcode=1), Mock(pid=105, exitcode=0)]
        context = Mock()
        context.Process.side_effect = children
        with common.run_lock(root):
            manifest = self.bind()
            for child in children:
                child.join.side_effect = lambda: self.assertTrue((root / ".execution.lock").is_file())
            with patch.object(gpu45.mp, "get_context", return_value=context) as get_context:
                with self.assertRaisesRegex(RuntimeError, "stages failed"):
                    gpu45.run_workers(self.config, manifest, self.tasks, gpu45.GPU_PLAN, stage_label="answers")
                get_context.assert_called_once_with("spawn")
            for child in children:
                child.start.assert_called_once_with()
                child.join.assert_called_once_with()
                child.terminate.assert_not_called()

    def test_exit_audit_failure_occurs_only_after_all_workers_joined(self):
        root = Path(self.config["run_root"])
        children = [Mock(pid=104, exitcode=0), Mock(pid=105, exitcode=0)]
        context = Mock()
        context.Process.side_effect = children
        joined = []
        for index, child in enumerate(children):
            child.join.side_effect = lambda index=index: joined.append(index)

        def append_event(path, row):
            if row["event"] == "exited":
                self.assertEqual(joined, [0, 1], "audit failure must never skip a live worker")
                self.assertTrue((root / ".execution.lock").is_file())
                raise OSError("synthetic audit disk failure")

        with common.run_lock(root):
            manifest = self.bind()
            with patch.object(gpu45.mp, "get_context", return_value=context), \
                    patch.object(gpu45, "append_row", side_effect=append_event):
                with self.assertRaisesRegex(OSError, "audit disk failure"):
                    gpu45.run_workers(self.config, manifest, self.tasks, gpu45.GPU_PLAN, stage_label="answers")
            for child in children:
                child.join.assert_called_once_with()
            self.assertTrue((root / ".execution.lock").is_file())
        self.assertFalse((root / ".execution.lock").exists())


if __name__ == "__main__":
    unittest.main()
