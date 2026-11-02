"""Synthetic CPU checks only; these fixtures are not empirical results."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import common
import evaluate
import followup_audit as audit
import score_results
from test_score_results import FrozenFixture, _native, _write_json, _write_jsonl


class FollowupFixture(FrozenFixture):
    """Upgrade the original scorer fixture to a fully bound V4 source."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.config = {
            "schema_version": "care_opd.visual_transfer.config.v1",
            "protocol_version": common.PROTOCOL_VERSION,
            "evaluation_use": "validation",
            "tasks_dir": str(self.tasks_dir.resolve()),
            "run_root": str(self.run_root.resolve()),
            "models": {name: str((root / "synthetic-models" / name).resolve())
                       for name in ("vlm", *common.TEXT_ARMS)},
        }
        _write_json(self.config_path, self.config)
        self.config = evaluate.configuration(self.config_path)
        selection_path = self.tasks_dir / "selection_manifest.json"
        selection = common.read_json(selection_path)
        selection["dataset"] = {"source_split": "validation"}
        _write_json(selection_path, selection)
        self.manifest.update({
            "protocol_version": common.PROTOCOL_VERSION,
            "config": self.config,
            "task_count": len(self.tasks),
            "selection_manifest_sha256": common.file_sha256(selection_path),
            "code_sha256": evaluate.source_fingerprints(),
            "environment": {"synthetic_original_gpu": "5"},
        })
        self.rebind()

    def rebind(self):
        self.manifest.pop("protocol_sha256", None)
        self.manifest["protocol_sha256"] = common.digest(self.manifest)
        _write_json(self.run_root / "run_manifest.json", self.manifest)
        for observation in self.observations.values():
            observation["protocol_sha256"] = self.manifest["protocol_sha256"]
        _write_jsonl(self.run_root / "observations" / "observations.jsonl", list(self.observations.values()))
        for arm in common.ARMS:
            for row in self.response_rows[arm]:
                row["protocol_sha256"] = self.manifest["protocol_sha256"]
                if arm != "vlm_direct":
                    row["observation_sha256"] = common.digest(self.observations[row["task_id"]])
            self.write_arm(arm)


def tree_hashes(root: Path):
    return {str(path.relative_to(root)): common.file_sha256(path)
            for path in root.rglob("*") if path.is_file()}


class FollowupAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = FollowupFixture(Path(self.temp.name))

    def load(self):
        return audit.load_source(self.fixture.config_path)

    def summary(self, report, arm="care_opd", condition="clean"):
        return next(row for row in report["per_arm_condition"]
                    if row["arm"] == arm and row["condition"] == condition)

    def test_complete_source_is_read_only_and_no_runtime_calls(self):
        before = tree_hashes(self.fixture.root)
        with mock.patch.object(evaluate, "HFGenerator", side_effect=AssertionError("model loading forbidden")), \
             mock.patch.object(evaluate, "environment", side_effect=AssertionError("GPU inspection forbidden")), \
             mock.patch.object(evaluate, "fingerprint_model", side_effect=AssertionError("model hashing forbidden")):
            source = self.load()
            report = audit.build_audit(source)
        self.assertEqual(tree_hashes(self.fixture.root), before)
        self.assertEqual(set(source["responses"]), set(common.ARMS))
        self.assertEqual(len(source["observations"]), 5)
        self.assertEqual(len(report["row_records"]), 25)
        self.assertEqual(source["selection"]["dataset"]["source_split"], "validation")
        self.assertEqual(source["binding"]["source_manifest_sha256"], common.file_sha256(self.fixture.run_root / "run_manifest.json"))
        self.assertTrue(report["diagnostic_only"])
        self.assertEqual({row["pixel_grounding"] for row in report["row_records"]}, {"unknown_not_reviewed"})

    def test_binding_repeats_identically(self):
        first, second = self.load(), self.load()
        self.assertEqual(first["binding"], second["binding"])
        self.assertEqual(audit.build_audit(first), audit.build_audit(second))

    def test_old_config_protocol_refused(self):
        config = common.read_json(self.fixture.config_path)
        config["protocol_version"] = "care_opd.visual_transfer.v1"
        _write_json(self.fixture.config_path, config)
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.load()

    def test_old_manifest_protocol_refused(self):
        self.fixture.manifest["protocol_version"] = "care_opd.visual_transfer.v1"
        self.fixture.rebind()
        with self.assertRaisesRegex(ValueError, "V4"):
            self.load()

    def test_post_test_protocol_revision_refused(self):
        config = common.read_json(self.fixture.config_path)
        config["evaluation_use"] = "post_test_protocol_revision"
        _write_json(self.fixture.config_path, config)
        with self.assertRaisesRegex(ValueError, "validation"):
            self.load()

    def test_source_split_not_relabeled_validation(self):
        path = self.fixture.tasks_dir / "selection_manifest.json"
        selection = common.read_json(path)
        selection["dataset"]["source_split"] = "test"
        _write_json(path, selection)
        self.fixture.manifest["selection_manifest_sha256"] = common.file_sha256(path)
        self.fixture.rebind()
        with self.assertRaisesRegex(ValueError, "source_split"):
            self.load()

    def test_config_cannot_diverge_from_manifest(self):
        config = common.read_json(self.fixture.config_path)
        config["generation"] = {"max_new_tokens": 513}
        _write_json(self.fixture.config_path, config)
        with self.assertRaisesRegex(ValueError, "config differs"):
            self.load()

    def test_changed_frozen_code_identity_refused(self):
        self.fixture.manifest["code_sha256"]["common.py"] = "0" * 64
        self.fixture.rebind()
        with self.assertRaisesRegex(ValueError, "adapter differs"):
            self.load()

    def test_incomplete_arm_refused(self):
        self.fixture.response_rows["care_opd"].pop()
        self.fixture.write_arm("care_opd")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.load()

    def test_missing_observation_refused(self):
        rows = list(self.fixture.observations.values())[:-1]
        _write_jsonl(self.fixture.run_root / "observations" / "observations.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "observation"):
            self.load()

    def test_observation_content_must_match_bound_input(self):
        rows = copy.deepcopy(list(self.fixture.observations.values()))
        rows[0]["raw_text"] = "changed observation"
        _write_jsonl(self.fixture.run_root / "observations" / "observations.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "observation SHA"):
            self.load()

    def test_response_input_must_match_frozen_prompt(self):
        self.fixture.response_rows["care_opd"][0]["input_sha256"] = "0" * 64
        self.fixture.write_arm("care_opd")
        with self.assertRaisesRegex(ValueError, "input SHA"):
            self.load()

    def test_changes_during_validation_refused(self):
        original = score_results._check_observation_identity
        def mutate_after_validation(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.fixture.config_path
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return result
        with mock.patch.object(score_results, "_check_observation_identity", side_effect=mutate_after_validation):
            with self.assertRaisesRegex(ValueError, "changed during audit"):
                self.load()

    def test_unknown_label_is_not_abstention_or_schema_failure(self):
        rows = self.fixture.response_rows["care_opd"]
        rows[0]["raw_text"] = _native("MAYBE", self.fixture.tasks[0]["context_value"])
        rows[3]["raw_text"] = _native("ABSTAIN", self.fixture.tasks[3]["context_value"])
        self.fixture.write_arm("care_opd")
        report = audit.build_audit(self.load())
        row = self.summary(report)
        self.assertEqual(row["total"], 2)
        self.assertEqual(row["answer_unrecognized"], 1)
        self.assertEqual(row["answer_abstain"], 1)
        self.assertEqual(row["schema_invalid_count"], 0)
        self.assertEqual(row["content_incorrect_count"], 2)

    def test_content_correct_but_truncated_is_separate(self):
        self.fixture.response_rows["care_opd"][0]["finish_reason"] = "length"
        self.fixture.write_arm("care_opd")
        row = self.summary(audit.build_audit(self.load()))
        self.assertEqual(row["answer_correct_count"], 2)
        self.assertEqual(row["strict_contract_success_count"], 1)
        self.assertEqual(row["truncated_count"], 1)
        self.assertEqual(row["content_correct_contract_failure_count"], 1)

    def test_goldblind_rules_and_condition_denominators(self):
        report = audit.build_audit(self.load())
        rows = {(row["rule"], row["condition"]): row for row in report["rule_baselines"]}
        self.assertEqual(rows[("always_abstain", "all")]["answer_correct_count"], 2)
        self.assertEqual(rows[("always_abstain", "all")]["total"], 5)
        self.assertEqual(rows[("missing_abstain_else_yes", "all")]["answer_correct_count"], 4)
        self.assertEqual(rows[("missing_abstain_else_no", "swapped_image")]["answer_correct_count"], 1)
        for rule in audit.RULES:
            self.assertEqual(audit._rule_prediction(rule, {"image_path": None, "answer": "YES"}), "ABSTAIN")
        task = {"image_path": "/synthetic.png", "answer": "NO"}
        self.assertEqual(audit._rule_prediction("missing_abstain_else_yes", task), "YES")
        self.assertFalse(any(row["prediction_uses_gold"] for row in report["rule_baselines"]))

    def test_runtime_migration_ledger_bound_without_gpu_revalidation(self):
        path = self.fixture.run_root / "runtime" / "device_migrations.jsonl"
        _write_jsonl(path, [{"event": "completed", "physical_gpu": "0", "synthetic_test_only": True}])
        source = self.load()
        self.assertEqual(source["manifest"]["environment"]["synthetic_original_gpu"], "5")
        self.assertEqual(source["binding"]["files"]["optional/runtime/device_migrations.jsonl"]["sha256"], common.file_sha256(path))

    def test_visible_coverage_and_missing_guesses_not_mixed(self):
        rows = self.fixture.response_rows["care_opd"]
        rows[1]["raw_text"] = _native("YES", self.fixture.tasks[1]["context_value"])
        self.fixture.write_arm("care_opd")
        report = audit.build_audit(self.load())
        missing = self.summary(report, condition="no_image")
        self.assertEqual(missing["answered_count"], 1)
        self.assertEqual(missing["answer_coverage"], 0.5)
        self.assertEqual(missing["selective_accuracy_denominator"], 0)
        self.assertIsNone(missing["selective_accuracy"])
        self.assertEqual(missing["original_answer_accuracy_among_guesses"], 1.0)
        all_rows = self.summary(report, condition="all")
        self.assertEqual(all_rows["answered_count"], 4)
        self.assertEqual(all_rows["selective_accuracy_denominator"], 3)
        self.assertEqual(all_rows["answered_correct_count"], 3)
        self.assertEqual(all_rows["selective_accuracy"], 1.0)

    def test_zero_visible_answers_have_null_selective_accuracy(self):
        rows = self.fixture.response_rows["care_opd"]
        for index in (0, 3):
            rows[index]["raw_text"] = _native("ABSTAIN", self.fixture.tasks[index]["context_value"])
        self.fixture.write_arm("care_opd")
        row = self.summary(audit.build_audit(self.load()))
        self.assertEqual(row["answer_coverage"], 0)
        self.assertEqual(row["abstain_rate"], 1)
        self.assertEqual(row["unknown_rate"], 0)
        self.assertIsNone(row["selective_accuracy"])

    def test_clean_accuracy_coverage_and_selective_have_explicit_denominators(self):
        rows = self.fixture.response_rows["care_opd"]
        rows[3]["raw_text"] = _native("ABSTAIN", self.fixture.tasks[3]["context_value"])
        self.fixture.write_arm("care_opd")
        row = self.summary(audit.build_audit(self.load()))
        self.assertEqual(row["total"], 2)
        self.assertEqual(row["answer_correct_count"], 1)
        self.assertEqual(row["answer_accuracy"], 0.5)
        self.assertEqual(row["answered_count"], 1)
        self.assertEqual(row["answer_coverage"], 0.5)
        self.assertEqual(row["answered_correct_count"], 1)
        self.assertEqual(row["selective_accuracy_denominator"], 1)
        self.assertEqual(row["selective_accuracy"], 1.0)

    def test_correctly_relabelled_pair_uses_both_correct_not_flip_only(self):
        source = self.load()
        report = audit.build_audit(source)
        row = next(row for row in report["valid_image_update"] if row["arm"] == "care_opd")
        self.assertEqual(row["paired_families"], 1)
        self.assertEqual(row["both_correct_count"], 1)
        self.assertEqual(row["correct_update_given_old_correct"], 1)
        self.assertEqual(row["wrong_flip_count"], 0)
        rows = self.fixture.response_rows["care_opd"]
        rows[0]["raw_text"] = _native("NO", self.fixture.tasks[0]["context_value"])
        rows[2]["raw_text"] = _native("YES", self.fixture.tasks[2]["context_value"])
        self.fixture.write_arm("care_opd")
        report = audit.build_audit(self.load())
        row = next(row for row in report["valid_image_update"] if row["arm"] == "care_opd")
        self.assertEqual(row["binary_flip_count"], 1)
        self.assertEqual(row["wrong_flip_count"], 1)
        self.assertEqual(row["both_correct_count"], 0)
        self.assertEqual(row["old_correct_count"], 0)
        self.assertIsNone(row["correct_update_given_old_correct"])

    def test_image_update_unknown_is_not_abstention(self):
        rows = self.fixture.response_rows["care_opd"]
        rows[2]["raw_text"] = _native("MAYBE", self.fixture.tasks[2]["context_value"])
        self.fixture.write_arm("care_opd")
        report = audit.build_audit(self.load())
        row = next(row for row in report["valid_image_update"] if row["arm"] == "care_opd")
        self.assertEqual(row["old_correct_count"], 1)
        self.assertEqual(row["new_unknown_count"], 1)
        self.assertEqual(row["new_abstain_count"], 0)
        self.assertEqual(row["correct_update_given_old_correct"], 0)

    def test_swapped_abstention_is_not_a_correct_label_update(self):
        rows = self.fixture.response_rows["care_opd"]
        rows[2]["raw_text"] = _native("ABSTAIN", self.fixture.tasks[2]["context_value"])
        self.fixture.write_arm("care_opd")
        report = audit.build_audit(self.load())
        row = next(row for row in report["valid_image_update"] if row["arm"] == "care_opd")
        self.assertEqual(row["old_correct_count"], 1)
        self.assertEqual(row["new_abstain_count"], 1)
        self.assertEqual(row["new_unknown_count"], 0)
        self.assertEqual(row["both_correct_count"], 0)
        self.assertEqual(row["binary_flip_count"], 0)
        self.assertEqual(row["correct_update_given_old_correct"], 0)

    def test_build_does_not_mutate_loaded_source(self):
        source = self.load()
        before = copy.deepcopy(source)
        audit.build_audit(source)
        self.assertEqual(source, before)

    def test_changed_image_bytes_refused(self):
        path = Path(self.fixture.tasks[0]["image_path"])
        path.write_bytes(b"different pixels")
        with self.assertRaisesRegex(ValueError, "image SHA"):
            self.load()


if __name__ == "__main__":
    unittest.main()
