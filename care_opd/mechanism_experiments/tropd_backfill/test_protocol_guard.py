from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tropd_protocol_guard", HERE / "protocol_guard.py")
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class ProtocolGuardTest(unittest.TestCase):
    def test_canonical_hash_is_order_stable(self) -> None:
        self.assertEqual(
            guard.canonical_hash({"b": 2, "a": 1}),
            guard.canonical_hash({"a": 1, "b": 2}),
        )

    def test_immutable_lock_rejects_changed_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "lock.json"
            guard.immutable_json(target, {"fingerprint": "a"}, fingerprint_key="fingerprint")
            guard.immutable_json(target, {"fingerprint": "a"}, fingerprint_key="fingerprint")
            with self.assertRaisesRegex(ValueError, "choose a new BACKFILL_ROOT"):
                guard.immutable_json(
                    target, {"fingerprint": "b"}, fingerprint_key="fingerprint"
                )

    def test_real_manifest_comparison_fails_on_generation_change(self) -> None:
        stable = {
            "schema_version": "care_opd.real_thyroid_external_eval.v4.run_contract",
            "input": {
                "seed_sha256": "1" * 64,
                "field_gold_sha256": "2" * 64,
                "input_fingerprint": "3" * 64,
            },
            "evaluation_track": "structured_evidence_faithfulness",
            "challenge_mode": False,
            "generation": {"seed": 42},
            "verifier_sha256": "4" * 64,
            "evaluator_sha256": "5" * 64,
            "care_opd_qzbc_contract_sha256": "6" * 64,
            "bootstrap": {"replicates": 10000, "seed": 20260819},
            "models": [{"label": "old"}],
        }
        changed = json.loads(json.dumps(stable))
        changed["models"] = [{"label": "tropd_full"}]
        self.assertEqual(
            guard._real_stable_contract(stable), guard._real_stable_contract(changed)
        )
        changed["generation"]["seed"] = 7
        self.assertNotEqual(
            guard._real_stable_contract(stable), guard._real_stable_contract(changed)
        )

    def test_empty_real_manifests_cannot_compare_as_pass(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks required stable fields"):
            guard._real_stable_contract({})

    def test_model_lock_rejects_lineage_vs_thyroid_schema_mix(self) -> None:
        real_inventory = [
            {"relative_path": "config.json", "bytes": 10, "sha256": "a" * 64}
        ]
        lineage_inventory = [
            {"name": "config.json", "bytes": 10, "sha256": "a" * 64}
        ]
        self.assertNotEqual(
            guard.canonical_hash(real_inventory), guard.canonical_hash(lineage_inventory)
        )
        model = {
            "canonical_path": "/model",
            "model_fingerprint": guard.canonical_hash(real_inventory),
            "inventory": real_inventory,
        }
        contract = {
            "schema_version": guard.SCHEMA + ".model_contract",
            "fingerprint_scheme": "thyroid_external_eval.fingerprint_local_model",
            "models": {"tropd_full": model, "care_opd": dict(model)},
        }
        payload = {**contract, "contract_fingerprint": guard.canonical_hash(contract)}
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model_lock.json"
            target.write_text(json.dumps(payload), encoding="utf-8")
            guard._validate_model_lock(target)
            payload["models"]["tropd_full"]["model_fingerprint"] = guard.canonical_hash(
                lineage_inventory
            )
            payload["contract_fingerprint"] = guard.canonical_hash(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "contract_fingerprint"
                }
            )
            target.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory fingerprint"):
                guard._validate_model_lock(target)

    def test_parser_has_only_documented_commands(self) -> None:
        parser = guard.make_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(
            set(subparsers.choices),
            {
                "prepare-qzbc",
                "prepare-model-lock",
                "finalize-qzbc-model",
                "verify-qzbc-model",
                "summarize-qzbc",
                "compare-real-manifests",
                "inspect-real-inputs",
                "verify-real",
            },
        )

    def test_shell_has_inference_only_stages_and_no_destructive_shortcuts(self) -> None:
        script = (HERE / "run_tropd_backfill.sh").read_text(encoding="utf-8")
        for stage in ("preflight", "qzbc", "structured", "raw", "all"):
            self.assertIn(stage, script)
        for forbidden in ("rm -rf", "pkill", "kill -9", "huggingface-cli download", "pip install"):
            self.assertNotIn(forbidden, script)
        self.assertIn("care_opd.qzbc_rollout", script)
        self.assertIn("thyroid_external_eval.py", script)
        self.assertIn('--model "tropd_full=$TROPD_MODEL"', script)
        self.assertIn('--model "care_opd=$CARE_MODEL"', script)
        self.assertIn('adapter_config.json', script)
        self.assertIn('adapter-only/PEFT checkpoint', script)


if __name__ == "__main__":
    unittest.main()
