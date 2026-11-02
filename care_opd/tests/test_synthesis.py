import json
from pathlib import Path
import shutil
import tempfile
import unittest

from care_opd.schemas import Action, Split, canonical_json
from care_opd.synthesis import (
    POLICY_SYSTEM_PROMPT,
    PRIVILEGED_TEACHER_START,
    SynthesisConfig,
    build_dataset,
    main,
    make_case,
    serialize_policy_prompt,
)
from care_opd.verifier import validate_dataset


class SynthesisTest(unittest.TestCase):
    def test_case_generation_is_deterministic(self) -> None:
        left = make_case(Split.TEST, 17, 20260728)
        right = make_case(Split.TEST, 17, 20260728)
        self.assertEqual(canonical_json(left), canonical_json(right))

    def test_canonical_json_orders_sets_and_rejects_nan(self) -> None:
        self.assertEqual(
            canonical_json({"values": {"b", "a"}}),
            '{"values":["a","b"]}',
        )
        with self.assertRaises(ValueError):
            canonical_json({"invalid": float("nan")})

    def test_prompt_serializer_rejects_hidden_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "privileged field"):
            serialize_policy_prompt(
                {
                    "case_id": "synthetic",
                    "hidden": {"true_decision": "block"},
                }
            )

    def test_composition_splits_are_disjoint(self) -> None:
        compositions = {}
        for split in Split:
            compositions[split] = {
                make_case(split, index, 42).composition_key
                for index in range(48)
            }
        self.assertFalse(compositions[Split.TRAIN] & compositions[Split.DEV])
        self.assertFalse(compositions[Split.TRAIN] & compositions[Split.TEST])
        self.assertFalse(compositions[Split.DEV] & compositions[Split.TEST])

    def test_dataset_has_all_formal_views_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated"
            config = SynthesisConfig(
                seed=42,
                train=28,
                dev=20,
                test=24,
                recoveries_per_case=2,
            )
            manifest = build_dataset(output, config)
            for split in Split:
                for view in (
                    "clean_oracle",
                    "full_recovery",
                    "first_failure_revision",
                    "transition_splice",
                    "verified_pairs",
                    "supervised_kd",
                    "paired_eval",
                    "paired_prefix_plan",
                ):
                    self.assertTrue(
                        (output / f"{view}_{split.value}.jsonl").exists()
                    )
            report = validate_dataset(output, replay_sample=18)
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertEqual(
                manifest["splits"]["train"]["oracle_resolution_rate"], 1.0
            )

            clean = json.loads(
                (output / "clean_oracle_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(clean["target"], canonical_json(clean["action"]))
            self.assertEqual(
                [message["role"] for message in clean["messages"]],
                ["system", "user", "assistant"],
            )
            self.assertEqual(
                clean["messages"][0]["content"],
                POLICY_SYSTEM_PROMPT,
            )
            self.assertNotIn(
                POLICY_SYSTEM_PROMPT,
                clean["messages"][1]["content"],
            )
            self.assertEqual(
                clean["messages"][2]["content"],
                clean["target"],
            )
            supervised_kd = json.loads(
                (output / "supervised_kd_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                supervised_kd["student_prompt"],
                supervised_kd["teacher_prompt"],
            )
            self.assertEqual(supervised_kd["rollout"], clean["target"])
            self.assertEqual(
                supervised_kd["oracle_action"],
                clean["action"],
            )
            self.assertEqual(
                supervised_kd["repair_spans"],
                [
                    {
                        "start": 0,
                        "end": len(supervised_kd["rollout"]),
                        "component": "full_response",
                        "risk_weight": 1.0,
                    }
                ],
            )
            self.assertTrue(supervised_kd["host_verified"])
            self.assertTrue(supervised_kd["teacher_verified"])
            self.assertFalse(supervised_kd["teacher_output_verified"])
            self.assertIn(
                "not_generated_output",
                supervised_kd["teacher_verified_semantics"],
            )

            recovery = json.loads(
                (output / "recovery_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            context = recovery["privileged_teacher_context"]
            self.assertEqual(context["scope"], "teacher_training_only")
            self.assertEqual(
                context["verified_oracle_repair"],
                recovery["repair_certificate"]["intervened_action"],
            )
            self.assertTrue(
                context["verification"]["verified_minimal"]
            )
            self.assertTrue(recovery["host_repair_verified"])
            self.assertFalse(recovery["teacher_output_verified"])
            self.assertFalse(recovery["teacher_verified"])
            self.assertEqual(recovery["teacher_confidence"], 0.0)
            self.assertIn(
                PRIVILEGED_TEACHER_START,
                recovery["teacher_prompt"],
            )
            self.assertIn(
                canonical_json(context),
                recovery["teacher_prompt"],
            )
            for forbidden in (
                PRIVILEGED_TEACHER_START,
                '"verified_oracle_repair":',
                '"certificate_hash":',
                '"required_fact_keys":',
                '"true_decision":',
                '"world_id":',
            ):
                self.assertNotIn(forbidden, recovery["student_prompt"])

            preference = json.loads(
                (output / "verified_pairs_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                preference["chosen"],
                canonical_json(preference["preferred"]),
            )
            self.assertEqual(
                preference["rejected"],
                canonical_json(preference["rejected_action"]),
            )

            full_rows = [
                json.loads(line)
                for line in (
                    output / "full_recovery_train.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreater(len(full_rows), len(
                (output / "recovery_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ))
            for full_row in full_rows:
                parsed = Action.from_json(full_row["target"])
                self.assertEqual(
                    canonical_json(parsed),
                    full_row["target"],
                )
            first_full = next(
                row
                for row in full_rows
                if row["recovery_step_index"] == 0
            )
            self.assertTrue(
                first_full["prompt_includes_failure_feedback"]
            )
            self.assertIn('"rejected_action":', first_full["prompt"])
            later_full = next(
                row
                for row in full_rows
                if row["recovery_step_index"] > 0
            )
            self.assertFalse(
                later_full["prompt_includes_failure_feedback"]
            )
            self.assertNotIn('"rejected_action":', later_full["prompt"])

            prefix = json.loads(
                (output / "paired_prefix_plan_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertTrue(prefix["action_cost"]["equal"])
            self.assertEqual(
                prefix["critical"]["remaining_budget"],
                prefix["nuisance"]["remaining_budget"],
            )
            self.assertEqual(
                prefix["clean"]["remaining_budget"] - 1,
                prefix["critical"]["remaining_budget"],
            )
            self.assertEqual(
                prefix["critical"]["failure_type"],
                "premature_decision",
            )
            self.assertEqual(prefix["nuisance"]["failure_type"], "none")

    def test_config_cli_resolves_project_relative_paths(self) -> None:
        project_source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            config_dir = project / "configs"
            knowledge_dir = project / "data" / "knowledge"
            config_dir.mkdir(parents=True)
            knowledge_dir.mkdir(parents=True)
            shutil.copyfile(
                project_source / "data" / "knowledge" / "tcm_workflow_seed.json",
                knowledge_dir / "seed.json",
            )
            config = config_dir / "smoke.yaml"
            config.write_text(
                "\n".join(
                    (
                        "project:",
                        "  seed: 17",
                        "data:",
                        "  knowledge: data/knowledge/seed.json",
                        "  output_dir: data/generated_from_config",
                        "  train_episodes: 4",
                        "  dev_episodes: 4",
                        "  test_episodes: 4",
                        "  max_steps: 5",
                        "  counterfactuals_per_failure: 1",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(main(["--config", str(config)]), 0)
            output = project / "data" / "generated_from_config"
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["config"]["seed"], 17)
            self.assertEqual(
                manifest["knowledge_binding"]["status"],
                "BOUND_AND_SCHEMA_CHECKED",
            )
            self.assertTrue((output / "paired_prefix_plan_test.jsonl").exists())

    def test_seed_is_explicitly_non_clinical(self) -> None:
        seed_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "knowledge"
            / "tcm_workflow_seed.json"
        )
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        self.assertTrue(seed["research_only"])
        self.assertFalse(seed["clinical_use_permitted"])
        self.assertEqual(
            seed["review_status"]["status"],
            "NOT_CLINICALLY_REVIEWED",
        )
        for rule in seed["rules"]:
            self.assertTrue(rule["rule_id"].startswith("SIM-"))
            self.assertIn("本仿真环境", rule["simulator_assertion"])


if __name__ == "__main__":
    unittest.main()
