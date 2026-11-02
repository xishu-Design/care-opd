import json
import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for item in (HERE, PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

# The audited functions are tokenizer-only.  Keep this unit test runnable on
# CPU-only packaging hosts where torch is intentionally absent; production
# training still imports the real torch module and is covered on the GPU host.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.long = object()
    torch_stub.float32 = object()
    torch_stub.bool = object()
    sys.modules["torch"] = torch_stub

from audit_token_matching import (
    _write_outputs,
    analyze,
)
import contract
import plan_experiment


class CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        result = {"input_ids": [ord(character) + 10 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


def row(identifier, start, end):
    return {
        "id": identifier,
        "variant": "source_outage",
        "task_type": "doctor_ai_chat",
        "risk_level": "high",
        "rollout": '{"summary":"abcdef","warnings":[]}',
        "teacher_prompt": "teacher",
        "repair_spans": [
            {"start": start, "end": end, "risk_weight": 4.0}
        ],
        "risk_weight": 4.0,
        "host_repair_verified": True,
        "host_verification_confidence": 1.0,
    }


class TokenMatchedControlTest(unittest.TestCase):
    def _write_formal_audit(self, root, data, tokenizer_reference="student"):
        summary, records = analyze(
            [row("a", 12, 18)],
            tokenizer=CharacterTokenizer(),
            seed=20260815,
            max_length=4096,
            max_response_tokens=1200,
            add_eos=True,
        )
        summary["source"] = {
            "path": str(data.resolve()),
            "sha256": contract._sha256(data),
            "rows": 1,
        }
        summary["tokenizer"] = tokenizer_reference
        summary["tokenizer_sha256"] = "a" * 64
        summary["implementation"] = {
            label: {
                "path": str(path),
                "sha256": contract._sha256(path),
                "bytes": path.stat().st_size,
            }
            for label, path in contract.IMPLEMENTATION_PATHS.items()
        }
        audit = root / "token_matching_audit.json"
        audit.write_text(json.dumps(summary), encoding="utf-8")
        return audit

    def test_analysis_is_exact_and_deterministic(self):
        rows = [row("a", 12, 18), row("b", 2, 8)]
        first, records = analyze(
            rows,
            tokenizer=CharacterTokenizer(),
            seed=17,
            max_length=128,
            max_response_tokens=80,
            add_eos=True,
        )
        second, _ = analyze(
            rows,
            tokenizer=CharacterTokenizer(),
            seed=17,
            max_length=128,
            max_response_tokens=80,
            add_eos=True,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "COMPLETE")
        self.assertEqual(first["overall"]["exact_count_match_rate"], 1.0)
        self.assertEqual(
            first["overall"]["exact_weight_multiset_match_rate"], 1.0
        )
        self.assertEqual(len(records), 2)

    def test_outputs_are_parseable(self):
        summary, records = analyze(
            [row("a", 12, 18)],
            tokenizer=CharacterTokenizer(),
            seed=17,
            max_length=128,
            max_response_tokens=80,
            add_eos=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_outputs(root, summary, records)
            parsed = json.loads(
                (root / "token_matching_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(parsed["schema_version"], summary["schema_version"])
            self.assertTrue((root / "token_matching_audit.csv").is_file())
            self.assertTrue((root / "token_matching_examples.csv").is_file())
            self.assertTrue((root / "token_matching_audit.md").is_file())

    def test_frozen_matrix_and_audit_binding_pass(self):
        matrix = contract.audit_matrix(HERE / "token_matched_matrix.json")
        self.assertEqual(matrix["status"], "PASS")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "full_selected.jsonl"
            data.write_text(json.dumps(row("a", 12, 18)) + "\n", encoding="utf-8")
            audit = self._write_formal_audit(root, data)
            report = contract.audit_token_report(
                audit, data_path=data, tokenizer_reference="student"
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["active_tokens_per_arm"], 6)

    def test_plan_embeds_contract_and_token_audit_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            views = root / "views"
            views.mkdir()
            data = views / "full_selected.jsonl"
            data.write_text(json.dumps(row("a", 12, 18)) + "\n", encoding="utf-8")
            audit = self._write_formal_audit(root, data)
            config = root / "config.yaml"
            config.write_text("{}\n", encoding="utf-8")
            args = argparse.Namespace(
                matrix=HERE / "token_matched_matrix.json",
                token_audit=audit,
                views_dir=views,
                train_config=config,
                student_model="student",
                teacher_model="teacher",
                reference_model="student",
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
            self.assertEqual(plan["token_matching_audit"]["status"], "PASS")
            self.assertEqual(len(commands), 2)
            for command in commands:
                self.assertEqual(command[command.index("--max_length") + 1], "4096")
                self.assertEqual(
                    command[command.index("--max_response_tokens") + 1], "1200"
                )


if __name__ == "__main__":
    unittest.main()
