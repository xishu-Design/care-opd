from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for item in (HERE, PROJECT_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import audit_completed_run
import contract
import plan_experiment


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _view_rows():
    shared = {
        "rollout": "{}",
        "teacher_messages": [{"role": "user", "content": "x"}],
        "host_repair_verified": True,
        "host_verification_confidence": 1.0,
    }
    return [
        {
            **shared,
            "id": "clean",
            "clean_control": True,
            "variant": "clean",
            "repair_spans": [],
        },
        {
            **shared,
            "id": "failure",
            "clean_control": False,
            "variant": "source_outage",
            "repair_spans": [{"start": 0, "end": 2, "component": "summary"}],
            "component_mask": ["summary"],
            "oracle_response": {"summary": "repair"},
        },
    ]


class ContractTests(unittest.TestCase):
    def test_frozen_matrix_passes_and_has_two_exact_pairs(self):
        report = contract.audit_matrix(HERE / "oracle_attribution_matrix.json")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["arm_order"]), 5)
        self.assertEqual(len(report["matched_pairs"]), 2)

    def test_pair_mutation_fails_closed(self):
        payload = json.loads(
            (HERE / "oracle_attribution_matrix.json").read_text(encoding="utf-8")
        )
        payload = deepcopy(payload)
        payload["variants"][3]["overrides"]["use_teacher_trust"] = True
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "use_teacher_trust"):
                contract.audit_matrix(path)

    def test_multiseed_contrasts_match_frozen_attribution_contract(self):
        path = (
            PROJECT_ROOT
            / "mechanism_experiments"
            / "multiseed_ablation"
            / "hard_oracle_contrasts.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = lambda item: (item["name"], item["left"], item["right"])
        self.assertEqual(
            [fields(item) for item in payload["contrasts"]],
            [fields(item) for item in contract.CONTRASTS],
        )

    def test_view_requires_oracle_and_host_certificate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "full_selected.jsonl"
            rows = _view_rows()
            _write_jsonl(path, rows)
            self.assertEqual(contract.audit_training_view(path)["status"], "PASS")
            rows[1].pop("oracle_response")
            _write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "lack a verified oracle"):
                contract.audit_training_view(path)


class PlanTests(unittest.TestCase):
    def test_plan_is_never_executed_and_embeds_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            views = root / "views"
            views.mkdir()
            _write_jsonl(views / "full_selected.jsonl", _view_rows())
            config = root / "config.yaml"
            config.write_text("{}\n", encoding="utf-8")
            args = argparse.Namespace(
                matrix=HERE / "oracle_attribution_matrix.json",
                views_dir=views,
                train_config=config,
                student_model="student",
                teacher_model="teacher",
                reference_model="reference",
                output_root=root / "out",
                max_steps=3,
                save_steps=2,
                seed=42,
                nproc_per_node=1,
                python="python",
                resume=False,
            )
            plan, commands = plan_experiment.build_plan(args)
            self.assertFalse(plan["execution_requested"])
            self.assertEqual(plan["scientific_contract"]["status"], "PASS")
            self.assertEqual(
                set(plan["training_code"]),
                set(audit_completed_run.REQUIRED_TRAINING_CODE),
            )
            self.assertTrue(
                all(len(item["sha256"]) == 64 for item in plan["training_code"].values())
            )
            self.assertEqual(plan["oracle_training_view_audit"]["on_policy_failure_rows"], 1)
            self.assertEqual(len(commands), 5)
            gkd_plus = commands[3]
            self.assertEqual(
                gkd_plus[gkd_plus.index("--oracle_risk_weighting_mode") + 1],
                "enabled",
            )
            self.assertEqual(
                gkd_plus[gkd_plus.index("--use_risk_weighting") + 1], "false"
            )
            self.assertEqual(
                gkd_plus[gkd_plus.index("--hard_oracle_ce_weight") + 1], "0.25"
            )


class CompletionAuditTests(unittest.TestCase):
    def test_completed_run_checks_effective_switches_and_oracle_activity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arms = []
            for name, expected in contract.EXPECTED.items():
                arm_root = root / name
                final = arm_root / "final"
                final.mkdir(parents=True)
                (final / "adapter_model.safetensors").write_bytes(b"weight")
                (final / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "trainer": "opd",
                            "global_step": 3,
                            "opd": expected,
                        }
                    ),
                    encoding="utf-8",
                )
                active = float(expected["hard_oracle_ce_weight"]) > 0
                _write_jsonl(
                    arm_root / "metrics.jsonl",
                    [
                        {
                            "event": "start",
                            "method": "care_opd",
                            "effective_switches": {
                                "mask_mode": expected["mask_mode"],
                                "random_mask_seed": expected["random_mask_seed"],
                                "risk_weighting": expected["use_risk_weighting"],
                                "oracle_risk_weighting_mode": expected[
                                    "oracle_risk_weighting_mode"
                                ],
                                "teacher_trust": expected["use_teacher_trust"],
                                "retain_outside_repair": expected[
                                    "retain_outside_repair"
                                ],
                                "distillation_coefficient": expected[
                                    "distillation_coefficient"
                                ],
                                "retention_coefficient": expected[
                                    "retention_coefficient"
                                ],
                                "hard_oracle_ce_weight": expected[
                                    "hard_oracle_ce_weight"
                                ],
                            },
                        },
                        {
                            "event": "train",
                            "step": 1,
                            "hard_oracle_tokens_local": 2 if active else 0,
                            "hard_oracle_ce": 1.0 if active else 0.0,
                        },
                        {"event": "complete", "global_step": 3},
                    ],
                )
                arms.append({"name": name, "output_dir": str(arm_root)})
            (root / "experiment_plan.json").write_text(
                json.dumps(
                    {
                        "scientific_contract": {
                            "schema_version": contract.CONTRACT_SCHEMA,
                            "status": "PASS",
                        },
                        "training_code": {
                            relative: {
                                "sha256": audit_completed_run._sha256(
                                    audit_completed_run.PROJECT_ROOT / relative
                                ),
                                "bytes": (
                                    audit_completed_run.PROJECT_ROOT / relative
                                ).stat().st_size,
                            }
                            for relative in audit_completed_run.REQUIRED_TRAINING_CODE
                        },
                        "budget_contract": {"max_steps_per_arm": 3},
                        "arms": arms,
                    }
                ),
                encoding="utf-8",
            )
            result = audit_completed_run.audit(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(set(result["arms"]), set(contract.EXPECTED))


if __name__ == "__main__":
    unittest.main()
