from __future__ import annotations

import json
import statistics
import tempfile
import unittest
from pathlib import Path

try:
    from .verify_results import (
        TOKEN_AUDIT_SCHEMA,
        VerificationError,
        _sha256,
        _verify_token_training_bindings,
        verify,
    )
except ImportError:
    from verify_results import (
        TOKEN_AUDIT_SCHEMA,
        VerificationError,
        _sha256,
        _verify_token_training_bindings,
        verify,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


class VerifyResultsTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        seeds = [42, 43]
        arms = ["semantic", "random"]
        matrix_path = root / "matrix.json"
        variants = [
            {
                "name": name,
                "data_view": "view.jsonl",
                "overrides": {
                    "baseline_mode": "care_opd",
                    "mask_mode": "span" if name == "semantic" else "random_token_matched",
                    "hard_oracle_ce_weight": 0.0,
                },
            }
            for name in arms
        ]
        _write_json(
            matrix_path,
            {"schema_version": "care_opd.causal_ablation_matrix.v1", "variants": variants},
        )
        view = root / "view.jsonl"
        view.write_text('{"id":"train-1"}\n', encoding="utf-8")

        request_path = root / "requests.jsonl"
        request_rows = [{"id": f"r{index}"} for index in range(3)]
        request_path.write_text(
            "".join(json.dumps(row) + "\n" for row in request_rows), encoding="utf-8"
        )
        family_path = root / "family.json"
        _write_json(family_path, [row["id"] for row in request_rows])
        request_binding = _binding(request_path)
        family_binding = _binding(family_path)

        training_entries = []
        training_plans: dict[int, Path] = {}
        for seed in seeds:
            nested_path = root / f"seed_{seed}" / "train" / "experiment_plan.json"
            nested_arms = []
            for arm, spec in zip(arms, variants, strict=True):
                output = root / f"seed_{seed}" / "train" / arm
                final = output / "final"
                final.mkdir(parents=True)
                _write_json(final / "adapter_config.json", {"ok": True})
                (final / "adapter_model.safetensors").write_bytes(b"weights")
                _write_json(
                    final / "run_metadata.json",
                    {
                        "global_step": 2,
                        "training": {"seed": seed, "max_steps": 2},
                        "opd": spec["overrides"],
                    },
                )
                (output / "metrics.jsonl").write_text(
                    json.dumps(
                        {
                            "event": "train",
                            "step": 1,
                            "hard_oracle_tokens_local": 0,
                        }
                    )
                    + "\n"
                    + json.dumps({"event": "complete", "global_step": 2})
                    + "\n",
                    encoding="utf-8",
                )
                nested_arms.append(
                    {
                        "name": arm,
                        "data_path": str(view),
                        "data_sha256": _sha256(view),
                        "rows": 1,
                        "id_order_sha256": "same-order",
                        "output_dir": str(output),
                        "overrides": spec["overrides"],
                        "command": ["python", "train"],
                    }
                )
            _write_json(
                nested_path,
                {
                    "schema_version": "care_opd.causal_ablation_plan.v1",
                    "matrix": _binding(matrix_path),
                    "budget_contract": {
                        "seed": seed,
                        "max_steps_per_arm": 2,
                    },
                    "arms": nested_arms,
                },
            )
            training_plans[seed] = nested_path
            training_entries.append(
                {
                    "training_seed": seed,
                    "plan": _binding(nested_path),
                    "arms": [],
                }
            )
        training_suite_path = root / "training" / "multiseed_plan.json"
        _write_json(
            training_suite_path,
            {
                "schema_version": "care_opd.multiseed_ablation_plan.v1",
                "design": {"training_seeds": seeds, "arm_names": arms},
                "common_inputs": {"matrix": _binding(matrix_path), "max_steps": 2},
                "seeds": training_entries,
            },
        )

        evaluation_entries = []
        per_seed: dict[str, dict[str, dict[str, float]]] = {}
        for seed in seeds:
            eval_plan_path = root / "eval" / f"seed_{seed}" / "evaluation_plan.json"
            eval_arms = []
            models = {}
            relative = {}
            per_seed[str(seed)] = {}
            for arm_index, arm in enumerate(arms):
                training_output = root / f"seed_{seed}" / "train" / arm
                final = training_output / "final"
                rollout_dir = root / "eval" / f"seed_{seed}" / "rollouts" / arm
                rollout_dir.mkdir(parents=True)
                rollout = rollout_dir / "qzbc_rollouts.jsonl"
                rollout.write_text(
                    "".join(
                        json.dumps({**row, "passed": (index + arm_index + seed) % 2 == 0})
                        + "\n"
                        for index, row in enumerate(request_rows)
                    ),
                    encoding="utf-8",
                )
                _write_json(rollout_dir / "summary.json", {"rollouts": 3, "passed": 2})
                eval_arms.append(
                    {
                        "name": arm,
                        "training_output": str(training_output),
                        "adapter": str(final),
                        "adapter_manifest": {
                            "metadata_sha256": {
                                "adapter_config.json": _sha256(final / "adapter_config.json"),
                                "run_metadata.json": _sha256(final / "run_metadata.json"),
                            },
                            "weight_files": [
                                {
                                    "name": "adapter_model.safetensors",
                                    "bytes": (final / "adapter_model.safetensors").stat().st_size,
                                }
                            ],
                        },
                        "rollout_output": str(rollout_dir),
                    }
                )
                metric = 0.5 + 0.01 * (seed - 42) + 0.1 * arm_index
                models[arm] = {
                    "source": _binding(rollout),
                    "unified_metrics": {
                        "clean_task_success_rate": metric,
                        "perturbed_task_success_rate": metric,
                        "conditional_retention": metric,
                        "regression_rate": 1.0 - metric,
                    },
                }
                relative[arm] = {
                    "rescue_rate": metric,
                    "regression_rate": 1.0 - metric,
                    "net_success_delta": metric - 0.5,
                }
                per_seed[str(seed)][arm] = {"perturbed_tsr": metric}
            _write_json(
                eval_plan_path,
                {
                    "schema_version": "care_opd.causal_ablation_eval_plan.v1",
                    "training_plan": _binding(training_plans[seed]),
                    "requests": request_binding,
                    "arms": eval_arms,
                },
            )
            summary_path = root / "eval" / f"seed_{seed}" / "summary" / "summary.json"
            _write_json(
                summary_path,
                {
                    "schema_version": "care_opd.causal_ablation_summary.v1",
                    "requests": request_binding,
                    "family_ids": family_binding,
                    "reference": "semantic",
                    "models": models,
                    "reference_conditioned_request_metrics": relative,
                },
            )
            evaluation_entries.append(
                {
                    "training_seed": seed,
                    "training_plan": _binding(training_plans[seed]),
                    "evaluation_plan": _binding(eval_plan_path),
                    "summary": {"path": str(summary_path)},
                    "arm_names": arms,
                }
            )
        evaluation_suite_path = root / "eval" / "multiseed_evaluation_plan.json"
        _write_json(
            evaluation_suite_path,
            {
                "schema_version": "care_opd.multiseed_ablation_eval_plan.v1",
                "training_suite": _binding(training_suite_path),
                "requests": request_binding,
                "family_clean_ids": family_binding,
                "seeds": evaluation_entries,
            },
        )

        aggregated = {}
        for arm in arms:
            values = [per_seed[str(seed)][arm]["perturbed_tsr"] for seed in seeds]
            worst = min(values)
            aggregated[arm] = {
                "perturbed_tsr": {
                    "n_training_seeds": 2,
                    "mean": statistics.fmean(values),
                    "sample_std": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                    "worst_value": worst,
                    "worst_training_seeds": [
                        seed for seed, value in zip(seeds, values, strict=True) if value == worst
                    ],
                }
            }
        aggregate_path = root / "eval" / "aggregate" / "multiseed_summary.json"
        _write_json(
            aggregate_path,
            {
                "schema_version": "care_opd.multiseed_ablation_aggregate.v1",
                "evaluation_plan": _binding(evaluation_suite_path),
                "training_seeds": seeds,
                "n_training_seeds": 2,
                "arms": arms,
                "metrics": {"perturbed_tsr": {"direction": "higher"}},
                "per_seed": per_seed,
                "aggregate": aggregated,
            },
        )
        token_audit = root / "token_matching_audit.json"
        _write_json(
            token_audit,
            {
                "schema_version": "care_opd.token_matched_mask_audit.v1",
                "status": "COMPLETE",
                "tokenizer_sha256": "fixture-tokenizer",
                "overall": {
                    "rows": 1,
                    "exact_count_match_rate": 1.0,
                    "exact_weight_multiset_match_rate": 1.0,
                },
            },
        )
        return {
            "matrix": matrix_path,
            "training": training_suite_path,
            "evaluation": evaluation_suite_path,
            "aggregate": aggregate_path,
            "token_audit": token_audit,
        }

    def test_complete_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            result = verify(
                matrix_path=paths["matrix"],
                training_plan_path=paths["training"],
                evaluation_plan_path=paths["evaluation"],
                aggregate_path=paths["aggregate"],
                expected_seeds=[42, 43],
                expected_requests=3,
                token_audit_path=paths["token_audit"],
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["training"]["arms"], 4)
            self.assertEqual(result["evaluation"]["rollout_rows"], 12)

    def test_missing_completion_event_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            metric = Path(directory) / "seed_42" / "train" / "semantic" / "metrics.jsonl"
            metric.write_text('{"event":"train","step":1}\n', encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "complete event"):
                verify(
                    matrix_path=paths["matrix"],
                    training_plan_path=paths["training"],
                    evaluation_plan_path=paths["evaluation"],
                    aggregate_path=paths["aggregate"],
                    expected_seeds=[42, 43],
                    expected_requests=3,
                )

    def test_reordered_rollout_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            rollout = Path(directory) / "eval" / "seed_42" / "rollouts" / "semantic" / "qzbc_rollouts.jsonl"
            rows = rollout.read_text(encoding="utf-8").splitlines()
            rollout.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
            # Source hash in the family summary now also becomes stale. Either
            # condition is correctly fail-closed; ID-order is checked first.
            with self.assertRaisesRegex(VerificationError, "IDs/order"):
                verify(
                    matrix_path=paths["matrix"],
                    training_plan_path=paths["training"],
                    evaluation_plan_path=paths["evaluation"],
                    aggregate_path=paths["aggregate"],
                    expected_seeds=[42, 43],
                    expected_requests=3,
                )

    def test_token_control_rejects_completed_tokenizer_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "token_matching_audit.json"
            audit.write_text("{}\n", encoding="utf-8")
            output = root / "localized"
            _write_json(
                output / "final" / "run_metadata.json",
                {"tokenizer_sha256": "different-completed-tokenizer"},
            )
            nested_path = root / "experiment_plan.json"
            _write_json(
                nested_path,
                {
                    "schema_version": "care_opd.causal_ablation_plan.v1",
                    "token_matching_audit": {
                        "status": "PASS",
                        "path": str(audit.resolve()),
                        "sha256": _sha256(audit),
                        "rows": 1,
                        "tokenizer_sha256": "audited-tokenizer",
                    },
                    "arms": [
                        {
                            "name": "localized_kd_only",
                            "output_dir": str(output),
                        }
                    ],
                },
            )
            suite = {
                "design": {"contract_mode": "token_matched"},
                "prerequisite_audits": [
                    {
                        "path": str(audit.resolve()),
                        "sha256": _sha256(audit),
                        "schema_version": TOKEN_AUDIT_SCHEMA,
                        "status": "COMPLETE",
                    }
                ],
                "seeds": [
                    {
                        "training_seed": 42,
                        "plan": _binding(nested_path),
                    }
                ],
            }
            with self.assertRaisesRegex(VerificationError, "completed tokenizer_sha256"):
                _verify_token_training_bindings(
                    suite,
                    token_audit_path=audit,
                    audited_rows=1,
                    audited_tokenizer_sha256="audited-tokenizer",
                )


if __name__ == "__main__":
    unittest.main()
