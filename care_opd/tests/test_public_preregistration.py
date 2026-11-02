from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "public_eval" / "preregistration.py"
SPEC = importlib.util.spec_from_file_location("public_preregistration", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicPreregistrationTest(unittest.TestCase):
    def make_lock(self, root: Path) -> Path:
        target = root / "public_eval" / "protocol.txt"
        target.parent.mkdir(parents=True)
        target.write_text("fixed\n", encoding="utf-8")
        contract = {
            "schema_version": MODULE.SCHEMA_VERSION,
            "scope": "test",
            "timing_disclosure": {"test_predictions_observed": False},
            "fixed_model_order": list(MODULE.MODEL_ORDER),
            "model_suite": {
                "suite_fingerprint": MODULE.canonical_hash(
                    [(label, "0" * 64) for label in MODULE.MODEL_ORDER]
                ),
                "models": [
                    {"label": label, "model_fingerprint": "0" * 64}
                    for label in MODULE.MODEL_ORDER
                ],
            },
            "locked_files": [
                {
                    "path": "public_eval/protocol.txt",
                    "bytes": target.stat().st_size,
                    "sha256": MODULE.sha256_file(target),
                }
            ],
        }
        payload = {
            **contract,
            "contract_fingerprint": MODULE.canonical_hash(contract),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        path = root / "public_eval" / "PREREGISTRATION_LOCK.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_valid_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_lock(root)
            result = MODULE.validate_static_lock(
                path,
                project_root=root,
                required_paths=("public_eval/protocol.txt",),
                timing_key="test_predictions_observed",
            )
            self.assertEqual(result["schema_version"], MODULE.SCHEMA_VERSION)
            self.assertEqual(result["locked_files"], 1)

    def test_modified_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_lock(root)
            (root / "public_eval" / "protocol.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE.validate_static_lock(path, project_root=root)

    def test_unrelated_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "lock.json"
            path.write_text('{"hello":"world"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                MODULE.validate_static_lock(path, project_root=root)

    def test_locked_auxiliary_model_is_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_lock(root)
            scorer = root / "bert-scorer"
            scorer.mkdir()
            weight = scorer / "model.safetensors"
            weight.write_bytes(b"fixed scorer")
            row = {
                "name": weight.name,
                "bytes": weight.stat().st_size,
                "sha256": MODULE.sha256_file(weight),
            }
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["auxiliary_models"] = {
                "mtcmb_bert_scorer": {
                    "status": "LOCKED",
                    "canonical_path": str(scorer.resolve()),
                    "inventory": [row],
                    "artifact_fingerprint": MODULE.canonical_hash([row]),
                }
            }
            contract = {
                key: value
                for key, value in payload.items()
                if key not in {"created_at_utc", "contract_fingerprint"}
            }
            payload["contract_fingerprint"] = MODULE.canonical_hash(contract)
            path.write_text(json.dumps(payload), encoding="utf-8")

            MODULE.validate_static_lock(path, project_root=root)
            weight.write_bytes(b"tampered scorer")
            with self.assertRaisesRegex(ValueError, "auxiliary artifact changed"):
                MODULE.validate_static_lock(path, project_root=root)


if __name__ == "__main__":
    unittest.main()
