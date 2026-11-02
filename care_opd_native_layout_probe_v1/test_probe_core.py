"""CPU-only regression tests for the frozen evidence-layout development probe.

Uses copies of the existing small cached ledgers, never models or image files.
All generated strings below are artificial unit-test fixtures, not experiments.
Run: python -B -m unittest discover -s . -p test_probe_core.py -v
"""
from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import probe_core as p
import probe_run as runner


SOURCE = p.ROOT.parent / "care_opd_multimedia_boundary_gpu1_v1_diskfix"
CACHE_FILES = (
    "config.json", "selection_manifest.json", "run_manifest.json", "tasks.jsonl",
    "observations.jsonl", *[f"responses/{arm}/responses.jsonl" for arm in p.ARMS],
)


def fake_runtime(*, fail_at=None):
    """Return a stateful non-ML runtime producing deliberately invalid outputs."""
    state = types.SimpleNamespace(calls=0, instances=0, closes=0, seen_messages=[])

    class Generator:
        def __init__(self, model, generation, *, visual):
            if visual:
                raise AssertionError("visual inference must never be requested")
            state.instances += 1

        def generate(self, messages, image, max_new_tokens):
            if image is not None:
                raise AssertionError("text probe must not pass images")
            if max_new_tokens != 1024:
                raise AssertionError("original generation budget changed")
            state.calls += 1
            state.seen_messages.append(copy.deepcopy(messages))
            if state.calls == fail_at:
                raise RuntimeError("unit-test simulated infrastructure interruption")
            # Bad JSON, empty output and length termination remain first outputs.
            return {"raw_text": "" if state.calls % 2 else "UNIT_TEST_INVALID_JSON",
                    "finish_reason": "length" if state.calls % 3 == 0 else "stop",
                    "image_seen": False, "prompt_tokens": 1, "completion_tokens": 1}

        def close(self):
            state.closes += 1

    return types.SimpleNamespace(HFGenerator=Generator), state


@unittest.skipUnless(SOURCE.is_dir(), "original local cached pilot fixture unavailable")
class ProbeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = p.c.read_jsonl(SOURCE / "tasks.jsonl")
        cls.observations = p.c.index_rows(p.c.read_jsonl(SOURCE / "observations.jsonl"))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="native_layout_probe_test_")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source = root / "source"
        self.work = root / "work"
        for name in CACHE_FILES:
            destination = self.source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / name, destination)

    def prepare(self, blocks=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return p.prepare(self.source, self.work, blocks)

    def run_fake(self, work=None, *, fail_at=None):
        runtime, state = fake_runtime(fail_at=fail_at)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(work or self.work, runtime=runtime, skip_preflight=True)
        return state

    def test_all_systems_byte_identical_and_observations_roundtrip(self):
        for task in self.tasks:
            obs = p.c.task_observations(task, self.observations)
            original = p.c.answer_messages(task, obs)
            adapted = p.adapter_messages(task, obs)
            self.assertEqual(original[0]["content"].encode(), adapted[0]["content"].encode())
            self.assertEqual(p.c.strict_json(original[1]["content"]), p.roundtrip_payload(adapted))
            payload = p.c.strict_json(adapted[1]["content"])
            self.assertEqual(set(payload), {"questions", "requested_records", "record_context", "evidence_packet"})
            entries = payload["evidence_packet"]["retrieved_evidence"]
            self.assertEqual(len(entries), 2)
            for side, entry in zip(("a", "b"), entries):
                self.assertEqual(entry["summary"].encode(), obs[side]["raw_text"].encode())
                self.assertEqual(set(entry), {"source_id", "record_id", "summary", "origin"})

    def test_direct_adapter_whitelist_ignores_labels_conditions_and_outputs(self):
        task = self.tasks[0]
        obs = p.c.task_observations(task, self.observations)
        expected = p.adapter_messages(task, obs)
        changed = copy.deepcopy(task)
        changed.update(gold="GOLD_SENTINEL_NOT_IN_PROMPT", condition="CHANGED_CONDITION",
                       block_id="CHANGED_BLOCK", task_id="CHANGED_TASK", output="OUTPUT_SENTINEL")
        changed["request"]["reference_answer"] = "REQUEST_GOLD_SENTINEL"
        changed["sources"]["a"]["expected_answer"] = "SOURCE_GOLD_SENTINEL"
        obs_changed = copy.deepcopy(obs)
        obs_changed["a"]["gold"] = "OBS_GOLD_SENTINEL"
        obs_changed["a"]["condition"] = "OBS_CONDITION_SENTINEL"
        self.assertEqual(expected, p.adapter_messages(changed, obs_changed))

    def test_selection_is_order_and_output_independent(self):
        expected = p.selected_block_ids(self.tasks, 3)
        mutated = copy.deepcopy(list(reversed(self.tasks)))
        for i, task in enumerate(mutated):
            task["output"] = f"ARTIFICIAL_OUTPUT_{i}"
            task["gold"] = "ARTIFICIAL_LABEL"
            task["condition"] = "UNUSED_BY_SELECTION"
        self.assertEqual(expected, p.selected_block_ids(mutated, 3))
        self.assertEqual(expected, p.selected_block_ids(mutated + mutated[:5], 3))
        self.assertEqual(3, len(set(expected)))

    def test_selection_rejects_invalid_counts(self):
        for count in (0, -1, 11, True, False):
            with self.subTest(count=count), self.assertRaises(ValueError):
                p.selected_block_ids(self.tasks, count)

    def test_prepare_does_not_open_gold_and_does_not_load_ml(self):
        self.assertFalse((self.source / "gold.private.jsonl").exists())
        originally_loaded = set(sys.modules)
        original_open = Path.open

        def guard(path, *args, **kwargs):
            if path.name == "gold.private.jsonl":
                raise AssertionError("gold file access during prepare")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guard):
            manifest = self.prepare(blocks=3)
        self.assertEqual(manifest["task_count"], 18)
        self.assertEqual(manifest["blocks"], 3)
        self.assertFalse(manifest["gold_opened_during_prepare_or_inference"])
        self.assertFalse(manifest["new_visual_inference"])
        newly_loaded = set(sys.modules) - originally_loaded
        self.assertFalse(any(name.split(".")[0] in {"torch", "transformers", "vllm"} for name in newly_loaded))

    def test_source_checksum_rejects_task_byte_change(self):
        path = self.source / "tasks.jsonl"
        with path.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "source tasks changed"):
            p.inspect_source(self.source)

    def test_original_row_seal_rejects_modified_output(self):
        path = self.source / f"responses/{p.ARMS[0]}/responses.jsonl"
        rows = p.c.read_jsonl(path)
        rows[0]["raw_text"] = "UNIT_TEST_TAMPERED"
        path.write_text("".join(p.c.canonical(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "row seal mismatch"):
            p.inspect_source(self.source)

    def test_label_contaminated_task_rejected_by_source_validator(self):
        selection = p.c.read_json(self.source / "selection_manifest.json")
        tasks = copy.deepcopy(self.tasks)
        tasks[0]["gold"] = "UNIT_TEST_LABEL"
        with self.assertRaisesRegex(ValueError, "possible label contamination"):
            p.validate_tasks(tasks, selection)

    def test_disjoint_output_guard(self):
        for work in (self.source, self.source / "new", self.source.parent):
            with self.subTest(work=work), self.assertRaisesRegex(ValueError, "disjoint"):
                p.prepare(self.source, work)

    def test_prepare_idempotent_but_selection_change_rejected(self):
        original = self.prepare()
        frozen = (self.work / "experiment.json").read_bytes()
        again = self.prepare()
        self.assertEqual(original, again)
        self.assertEqual(frozen, (self.work / "experiment.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "frozen selection differs"):
            p.prepare(self.source, self.work, 2)

    def test_bad_input_and_source_changes_rejected_after_prepare(self):
        self.prepare()
        path = self.source / "observations.jsonl"
        with path.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "frozen file changed"):
            p.load_work(self.work)

    def test_empty_invalid_and_length_first_outputs_are_not_retried(self):
        self.prepare()
        state = self.run_fake()
        self.assertEqual(state.calls, 24)
        before = {arm: (self.work / f"responses/{arm}.jsonl").read_bytes() for arm in p.ARMS}
        resumed = self.run_fake()
        self.assertEqual(resumed.calls, 0)
        self.assertEqual(resumed.instances, 0)
        for arm in p.ARMS:
            self.assertEqual(before[arm], (self.work / f"responses/{arm}.jsonl").read_bytes())
            rows = list(p.load_new_rows(self.work, arm).values())
            self.assertTrue(any(not row["raw_text"] for row in rows))
            self.assertTrue(any(row["finish_reason"] == "length" for row in rows))
            old = p.c.index_rows(p.c.read_jsonl(self.work / f"baseline/{arm}.jsonl"))
            for row in rows:
                baseline = old[row["task_id"]]
                for suffix in ("row_sha256", "input_sha256", "protocol_sha256"):
                    self.assertEqual(row["paired_baseline_" + suffix], baseline[suffix])

    def test_resume_after_infrastructure_failure_preserves_saved_first_rows(self):
        self.prepare()
        runtime, failed = fake_runtime(fail_at=3)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "simulated infrastructure"):
            runner.run(self.work, runtime=runtime, skip_preflight=True)
        self.assertEqual(failed.calls, 3)
        self.assertEqual(failed.closes, 1)
        first_path = self.work / f"responses/{p.ARMS[0]}.jsonl"
        prefix = first_path.read_bytes()
        self.assertEqual(len(p.c.read_jsonl(first_path)), 2)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            p.load_new_rows(self.work, p.ARMS[0])
        resumed = self.run_fake()
        self.assertEqual(resumed.calls, 22)
        self.assertTrue(first_path.read_bytes().startswith(prefix))
        self.assertEqual(len(p.c.read_jsonl(self.work / "infrastructure_errors.jsonl")), 1)

    def test_resealed_wrong_pairing_is_rejected(self):
        self.prepare()
        self.run_fake()
        path = self.work / f"responses/{p.ARMS[0]}.jsonl"
        original = path.read_text(encoding="utf-8")
        for suffix in ("row_sha256", "input_sha256", "protocol_sha256"):
            with self.subTest(field=suffix):
                rows = [p.c.strict_json(line) for line in original.splitlines()]
                rows[0]["paired_baseline_" + suffix] = "0" * 64
                rows[0]["row_sha256"] = p.c.digest({k: v for k, v in rows[0].items() if k != "row_sha256"})
                path.write_text("".join(p.c.canonical(row) + "\n" for row in rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "row binding mismatch"):
                    p.load_new_rows(self.work, p.ARMS[0])


if __name__ == "__main__":
    unittest.main()
