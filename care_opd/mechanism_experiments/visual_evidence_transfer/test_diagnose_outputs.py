"""Synthetic CPU-only tests of aggregate, gold-free runtime diagnostics."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import diagnose_outputs as diagnostic


def native(answer="YES", context="CTX00001", visual_sources=None, context_sources=None):
    return json.dumps({"title": "private title", "summary": "private summary",
                       "sections": [
                           {"heading": "visual_answer", "content": answer,
                            "source_ids": (["image:current"] if answer in ("YES", "NO") else [])
                            if visual_sources is None else visual_sources},
                           {"heading": "context_value", "content": context,
                            "source_ids": ["text:context"] if context_sources is None else context_sources}],
                       "warnings": [], "next_actions": [], "disclaimer": "synthetic only"})


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class Fixture:
    def __init__(self, root):
        self.root = root
        self.tasks_dir = root / "tasks"
        self.run_root = root / "run"
        self.tasks = [{"task_id": condition, "family_id": "family",
                       "condition": condition, "context_value": "CTX00001",
                       "question": "private question"} for condition in diagnostic.CONDITIONS]
        self.task_path = self.tasks_dir / "tasks.jsonl"
        write_rows(self.task_path, self.tasks)
        self.observations = [{**task, "status": "completed", "raw_text": "无法辨认 PRIVATE_OBSERVATION",
                              "finish_reason": "length" if task["condition"] == "clean" else "stop",
                              "completion_tokens": 8, "prompt_tokens": 20} for task in self.tasks]
        write_rows(self.observation_path, self.observations)
        self.responses = [{**task, "status": "completed", "raw_text": native(answer),
                           "input_sha256": "a" * 64} for task, answer in zip(self.tasks, ("YES", "ABSTAIN", "NO"))]
        for arm in diagnostic.ARMS:
            write_rows(self.arm_path(arm), self.responses)
        self.config_path = root / "config.json"
        self.config_path.write_text(json.dumps({"tasks_dir": str(self.tasks_dir), "run_root": str(self.run_root)}), encoding="utf-8")
        # Deliberately unreadable as JSON. No manifest, image or checkpoint exists.
        (self.tasks_dir / "gold.private.jsonl").write_bytes(b"PRIVATE_GOLD_DO_NOT_READ")

    @property
    def observation_path(self):
        return self.run_root / "observations" / "observations.jsonl"

    def arm_path(self, arm="clean_sft"):
        return self.run_root / "arms" / arm / "responses.jsonl"

    def run(self):
        return diagnostic.diagnose(self.config_path)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def test_all_arms_conditions_counts_without_gold_or_private_output(self):
        original_open = Path.open

        def guard(path, *args, **kwargs):
            self.assertNotIn("gold", path.name)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guard):
            report = self.fixture.run()
        self.assertTrue(report["snapshot_complete"])
        self.assertFalse(report["gold_used"])
        self.assertFalse(report["model_selection_gate"])
        for arm in diagnostic.ARMS:
            for condition, answer in zip(diagnostic.CONDITIONS, ("YES", "ABSTAIN", "NO")):
                item = report["arms"][arm]["by_condition"][condition]
                self.assertEqual(item["first_completions"], 1)
                self.assertEqual(item["final_answer_distribution"][answer], 1)
                self.assertEqual(item["source_ids_error_count"], 0)
                self.assertEqual(item["context_field"]["accuracy_among_first_completions"], 1)
        serialized = json.dumps(report)
        for private in ("PRIVATE_GOLD", "PRIVATE_OBSERVATION", "private title", "private question", "CTX00001", "a" * 64):
            self.assertNotIn(private, serialized)

    def test_fences_sources_context_and_invalid_answers_are_separate(self):
        responses = [dict(row) for row in self.fixture.responses]
        responses[0]["raw_text"] = "```json\n" + native("YES", "WRONG", visual_sources=[]) + "\n```"
        responses[1]["raw_text"] = native("ABSTAIN", visual_sources=["image:current"], context_sources=[])
        responses[2]["raw_text"] = "not JSON PRIVATE_PARSE_INPUT"
        write_rows(self.fixture.arm_path(), responses)
        report = self.fixture.run()
        stats = report["arms"]["clean_sft"]["by_condition"]
        self.assertEqual(stats["clean"]["outer_markdown_fence_count"], 1)
        self.assertEqual(stats["clean"]["final_answer_distribution"]["YES"], 1)
        self.assertEqual(stats["clean"]["visual_source_ids_error_count"], 1)
        self.assertEqual(stats["clean"]["context_field"]["exact_count"], 0)
        self.assertEqual(stats["no_image"]["context_source_ids_error_count"], 1)
        self.assertEqual(stats["swapped_image"]["final_answer_distribution"]["invalid"], 1)
        self.assertEqual(stats["swapped_image"]["parse_error_count"], 1)
        self.assertNotIn("PRIVATE_PARSE_INPUT", json.dumps(report))
        # Output validity is not completion coverage or scientific success.
        self.assertTrue(report["snapshot_complete"])

    def test_partial_tail_is_not_counted_even_when_json_is_syntactically_valid(self):
        path = self.fixture.arm_path()
        for tail in ('{"task_id":', json.dumps(self.fixture.responses[-1])):
            with self.subTest(tail=tail[:12]):
                write_rows(path, self.fixture.responses[:-1])
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(tail)
                report = self.fixture.run()
                ledger = report["arms"]["clean_sft"]["ledger"]
                self.assertEqual(ledger["first_completions"], 2)
                self.assertTrue(ledger["partial_trailing_line"])
                self.assertFalse(report["snapshot_complete"])

    def test_fenced_direct_empty_sources_still_count_answers(self):
        responses = [dict(row) for row in self.fixture.responses]
        for row, answer in zip(responses, ("YES", "ABSTAIN", "NO")):
            row["raw_text"] = "```json\n" + native(answer, visual_sources=[], context_sources=[]) + "\n```"
        write_rows(self.fixture.arm_path("vlm_direct"), responses)
        stats = self.fixture.run()["arms"]["vlm_direct"]["by_condition"]
        for condition, answer in zip(diagnostic.CONDITIONS, ("YES", "ABSTAIN", "NO")):
            self.assertEqual(stats[condition]["final_answer_distribution"][answer], 1)
            self.assertEqual(stats[condition]["outer_markdown_fence_count"], 1)
            self.assertEqual(stats[condition]["context_source_ids_error_count"], 1)
            self.assertEqual(stats[condition]["visual_source_ids_error_count"], int(answer != "ABSTAIN"))

    def test_unable_prefix_counts_are_literal_not_semantic(self):
        self.fixture.observations[0]["raw_text"] = "  无法确定 PRIVATE"
        self.fixture.observations[1]["raw_text"] = "图像概况：无法确定 PRIVATE"
        self.fixture.observations[2]["raw_text"] = "unable PRIVATE"
        write_rows(self.fixture.observation_path, self.fixture.observations)
        stats = self.fixture.run()["observations"]["by_condition"]
        self.assertEqual(stats["clean"]["literal_unable_prefix_counts"]["无法确定"], 1)
        self.assertEqual(stats["clean"]["literal_unable_prefix_count"], 1)
        self.assertEqual(stats["no_image"]["literal_unable_prefix_count"], 0)
        self.assertEqual(stats["swapped_image"]["literal_unable_prefix_count"], 1)

    def test_duplicate_unknown_and_noncompleted_never_choose_a_better_retry(self):
        first = {**self.fixture.responses[0], "status": "infra_error"}
        duplicate = {**self.fixture.responses[1], "raw_text": native("YES")}
        unknown = {**self.fixture.responses[0], "task_id": "UNKNOWN_PRIVATE_ID"}
        write_rows(self.fixture.arm_path(), [first, *self.fixture.responses, duplicate, unknown])
        report = self.fixture.run()
        ledger = report["arms"]["clean_sft"]["ledger"]
        self.assertEqual(ledger["first_completions"], 2)
        self.assertEqual(ledger["row_warnings"]["duplicate_task_id"], 2)
        self.assertEqual(ledger["row_warnings"]["unknown_task_id"], 1)
        self.assertFalse(report["snapshot_complete"])
        no_image = report["arms"]["clean_sft"]["by_condition"]["no_image"]
        self.assertEqual(no_image["final_answer_distribution"]["ABSTAIN"], 1)
        self.assertNotIn("UNKNOWN_PRIVATE_ID", json.dumps(report))

    def test_observation_finish_lengths_tokens_prefix_and_pair_hashes(self):
        report = self.fixture.run()
        obs = report["observations"]["by_condition"]["clean"]
        self.assertEqual(obs["length_finish_count"], 1)
        self.assertEqual(obs["completion_tokens"]["mean"], 8)
        self.assertEqual(obs["text_length_characters"]["mean"], len(self.fixture.observations[0]["raw_text"]))
        self.assertEqual(obs["literal_unable_prefix_count"], 1)
        pairs = report["clean_swapped_identity"]
        self.assertEqual(pairs["observation_text"]["identical_pairs"], 1)
        for arm in diagnostic.ARMS[1:]:
            self.assertEqual(pairs["per_text_arm_input_hash"][arm]["identical_pairs"], 1)
        self.assertEqual(pairs["cross_text_arm_same_task_input_hash"]["all_four_identical_tasks"], 3)
        self.fixture.responses[-1]["input_sha256"] = "b" * 64
        write_rows(self.fixture.arm_path(), self.fixture.responses)
        pairs = self.fixture.run()["clean_swapped_identity"]
        self.assertEqual(pairs["per_text_arm_input_hash"]["clean_sft"]["different_pairs"], 1)
        self.assertEqual(pairs["cross_text_arm_same_task_input_hash"]["mismatched_tasks"], 1)

    def test_missing_files_and_malformed_interior_lines_never_complete(self):
        self.fixture.arm_path("care_opd").unlink()
        path = self.fixture.observation_path
        content = path.read_bytes()
        path.write_bytes(b'{"raw_text":"PRIVATE_BAD_LINE", broken}\n' + content)
        report = self.fixture.run()
        self.assertFalse(report["snapshot_complete"])
        self.assertEqual(report["observations"]["ledger"]["invalid_lines"], 1)
        self.assertFalse(report["arms"]["care_opd"]["ledger"]["exists"])
        self.assertNotIn("PRIVATE_BAD_LINE", json.dumps(report))

    def test_duplicate_tasks_and_ambiguous_families_warn(self):
        extra = {**self.fixture.tasks[0], "task_id": "second-clean"}
        write_rows(self.fixture.task_path, [*self.fixture.tasks, self.fixture.tasks[0], extra])
        report = self.fixture.run()
        self.assertFalse(report["snapshot_complete"])
        self.assertEqual(report["tasks"]["rejected_rows"], 1)
        self.assertEqual(report["clean_swapped_identity"]["ambiguous_family_count"], 1)

    def test_reader_limits_and_duplicate_json_keys(self):
        warnings = diagnostic.Warnings()
        path = self.fixture.root / "limit.jsonl"
        path.write_bytes(b'{"task_id":"a","task_id":"b"}\n')
        rows, state = diagnostic.read_snapshot(path, "test", warnings)
        self.assertFalse(rows)
        self.assertEqual(state["invalid_lines"], 1)
        with patch.object(diagnostic, "MAX_LINE_BYTES", 5):
            rows, state = diagnostic.read_snapshot(path, "test", warnings)
        self.assertTrue(state["read_limit_reached"])
        self.assertFalse(state["readable_snapshot"])

    def test_cli_json_only_no_third_party_packages_or_file_changes(self):
        before = {str(path): path.read_bytes() for path in self.fixture.root.rglob("*") if path.is_file()}
        proc = subprocess.run([sys.executable, "-S", str(Path(diagnostic.__file__).resolve()),
                               "--config", str(self.fixture.config_path)], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertTrue(json.loads(proc.stdout)["snapshot_complete"])
        after = {str(path): path.read_bytes() for path in self.fixture.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertNotIn("torch", sys.modules)

    def test_invalid_config_error_is_safe_json(self):
        self.fixture.config_path.write_text('{"PRIVATE_CONFIG_DATA": invalid}', encoding="utf-8")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = diagnostic.main(["--config", str(self.fixture.config_path)])
        self.assertEqual(code, 2)
        report = json.loads(stdout.getvalue())
        self.assertFalse(report["snapshot_complete"])
        self.assertNotIn("PRIVATE_CONFIG_DATA", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
