from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "public_eval" / "verify_model_lineage.py"
)
SPEC = importlib.util.spec_from_file_location("verify_model_lineage", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyModelLineageTest(unittest.TestCase):
    def _model(self, root: Path, weight: bytes, export_path: str) -> Path:
        root.mkdir()
        (root / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(weight)
        (root / "export_manifest.json").write_text(
            '{"source":"' + export_path + '"}', encoding="utf-8"
        )
        return root

    def _merged_model(self, root: Path, weight: bytes, base_model: Path) -> Path:
        model = self._model(root, weight, str(base_model))
        (model / "export_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "care-opd-qzbc-standalone-v1",
                    "artifact_type": "merged_huggingface_checkpoint",
                    "base_model": str(base_model),
                    "adapter_merged": True,
                    "runtime_dependencies": {"peft_adapter": False},
                }
            ),
            encoding="utf-8",
        )
        return model

    def test_warm_start_fingerprint_ignores_path_bearing_export_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            left = self._model(base / "left", b"same-weights", "/run/left")
            right = self._model(base / "right", b"same-weights", "/run/right")
            self.assertEqual(
                MODULE.fingerprint_directory(left)["model_fingerprint"],
                MODULE.fingerprint_directory(right)["model_fingerprint"],
            )

    def test_weight_change_changes_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            left = self._model(base / "left", b"weights-a", "/run/left")
            right = self._model(base / "right", b"weights-b", "/run/right")
            self.assertNotEqual(
                MODULE.fingerprint_directory(left)["model_fingerprint"],
                MODULE.fingerprint_directory(right)["model_fingerprint"],
            )

    def test_weight_index_is_part_of_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fingerprints = []
            for name, tensor_name in (("left", "tensor.a"), ("right", "tensor.b")):
                root = base / name
                root.mkdir()
                (root / "config.json").write_text("{}", encoding="utf-8")
                (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
                (root / "model-00001-of-00001.safetensors").write_bytes(b"same")
                (root / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": {},
                            "weight_map": {
                                tensor_name: "model-00001-of-00001.safetensors"
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                fingerprints.append(MODULE.fingerprint_directory(root)["model_fingerprint"])
            self.assertNotEqual(*fingerprints)

    def test_adapter_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._model(Path(temporary) / "model", b"weights", "/base")
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adapter"):
                MODULE.fingerprint_directory(root)

    def test_declared_inventory_must_close_its_model_fingerprint(self) -> None:
        inventory = [
            {"name": "model.safetensors", "bytes": 7, "sha256": "0" * 64},
            {"name": "config.json", "bytes": 2, "sha256": "1" * 64},
        ]
        fingerprint = MODULE.inventory_fingerprint(inventory)
        self.assertEqual(fingerprint, MODULE.canonical_hash(MODULE.normalize_inventory(inventory)))
        changed = [dict(item) for item in inventory]
        changed[0]["bytes"] = 8
        self.assertNotEqual(fingerprint, MODULE.inventory_fingerprint(changed))

    def test_suite_entry_rejects_tampered_declared_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._model(Path(temporary) / "frozen", b"weights", "/raw")
            actual = MODULE.fingerprint_directory(root)
            entry = {
                "label": "frozen",
                "canonical_path": str(root.resolve()),
                "inventory": actual["inventory"],
                "model_fingerprint": "f" * 64,
            }
            report, errors = MODULE.audit_suite_entry(
                entry, label="frozen", rehash=False
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(any("does not close" in error for error in errors))

    def test_export_chain_reaches_fixed_warm_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            warm = self._merged_model(base / "warm", b"warm", base / "raw")
            round_one = self._merged_model(base / "round_one", b"one", warm)
            final = self._merged_model(base / "final", b"two", round_one)
            trace = MODULE.trace_export_chain(final, warm)
            self.assertEqual(trace["status"], "PASS")
            self.assertTrue(trace["reached_common_warm_start"])
            self.assertEqual(len(trace["chain"]), 3)

    def test_origin_reference_requires_revision_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "origin.json"
            reference.write_text(
                json.dumps({"model_fingerprint": "a" * 64}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "revision"):
                MODULE._origin_reference(reference)

    def test_origin_reference_rejects_mutable_branch_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "origin.json"
            reference.write_text(
                json.dumps(
                    {
                        "model_fingerprint": "a" * 64,
                        "revision": "main",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "40- or 64-hex"):
                MODULE._origin_reference(reference)

    def test_run_contract_is_recomputed_instead_of_string_matched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = self._model(base / "model", b"weights", "/raw")
            model_fp = MODULE.fingerprint_directory(model)["model_fingerprint"]
            entry = {
                "canonical_path": str(model.resolve()),
                "model_fingerprint": model_fp,
            }
            run_root = base / "eval" / "official" / "frozen"
            run_root.mkdir(parents=True)
            prompt = {"template": "fixture"}
            generation = {"do_sample": False}
            fingerprints = {
                "dataset_fingerprint": "a" * 64,
                "prompt_fingerprint": MODULE.canonical_hash(prompt),
                "generation_fingerprint": MODULE.canonical_hash(generation),
                "model_fingerprint": model_fp,
            }
            manifest = {
                "protocol_version": "fixture-v1",
                "model_label": "frozen",
                "model_path": str(model.resolve()),
                "prompt_mode": "official",
                "num_workers": 1,
                "devices": ["cpu"],
                "fingerprints": fingerprints,
                "evaluator_sha256": "b" * 64,
                "dataset": {"sha256": "a" * 64},
                "prompt": prompt,
                "generation": generation,
                "model": entry,
            }
            manifest["run_fingerprint"] = MODULE.canonical_hash(
                MODULE._run_contract(manifest)
            )
            (run_root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            completion = {
                "status": "complete",
                "rows": 1,
                "run_fingerprint": manifest["run_fingerprint"],
                "fingerprints": fingerprints,
            }
            summary = {
                "status": "complete",
                "run_fingerprint": manifest["run_fingerprint"],
                "fingerprints": fingerprints,
            }
            (run_root / "COMPLETE.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            (run_root / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            row = {
                "id": "fixture-1",
                "run_fingerprint": manifest["run_fingerprint"],
                **fingerprints,
            }
            (run_root / "predictions.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            report, errors = MODULE.audit_run(
                run_root,
                entry=entry,
                label="frozen",
                prompt_mode="official",
            )
            self.assertEqual(report["status"], "PASS", errors)

            manifest["num_workers"] = 2
            (run_root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report, errors = MODULE.audit_run(
                run_root,
                entry=entry,
                label="frozen",
                prompt_mode="official",
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(any("does not close" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
