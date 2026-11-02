from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from care_opd.qzbc_bridge import (
    _validate_report,
    compare_reports,
    prepare_bundle,
    run_comparison,
)
from care_opd.qzbc_contract import canonical_response
from care_opd.qzbc_native_eval import evaluate_paired


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _checkpoint(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen-test",
                "architectures": ["QwenTestForCausalLM"],
                "vocab_size": 100,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fake-full-model")
    return path


def _qzbc(root: Path) -> Path:
    path = root / "qzbc"
    _write(
        path / "backend/app/core/config.py",
        "\n".join(
            (
                "llm_provider = 'ollama'",
                "llm_allow_fallback = False",
                "llm_temperature = 0.2",
                "llm_max_output_tokens = 1200",
                "ollama_base_url = ''",
                "ollama_model = ''",
                "ollama_num_ctx = 32768",
                "openai_compatible_base_url = ''",
                "openai_compatible_model = ''",
            )
        ),
    )
    _write(
        path / "backend/app/llm/providers/ollama_provider.py",
        "class OllamaProvider: pass\n",
    )
    _write(
        path / "backend/app/llm/providers/openai_compatible_provider.py",
        "class OpenAICompatibleProvider: pass\n",
    )
    _write(
        path / "backend/app/llm/experiments/runner.py",
        "class LLMExperimentRunner: pass\n",
    )
    _write(
        path / "backend/app/api/routes/llm.py",
        '@router.post("/experiments/run")\ndef run(): pass\n',
    )
    _write(path / "backend/.env.example", "LLM_PROVIDER=ollama\n")
    case_root = path / "backend/app/llm/experiments/cases"
    for filename in (
        "doctor_review_cases.json",
        "patient_education_cases.json",
        "student_analysis_cases.json",
        "quiz_generation_cases.json",
    ):
        _write(
            case_root / filename,
            json.dumps([{"case_id": f"{filename}-{index}"} for index in range(3)]),
        )
    return path


def _report(model: str, passed: tuple[bool, bool]) -> dict:
    items = []
    for index, outcome in enumerate(passed):
        items.append(
            {
                "case_id": f"case-{index}",
                "role": "student",
                "task_type": "student_analysis",
                "passed": outcome,
                "provider": "ollama",
                "model": model,
                "latency_ms": 10 + index,
                "fallback_used": False,
            }
        )
    success = sum(passed)
    return {
        "provider": "ollama",
        "model": model,
        "suite": "full",
        "total_cases": 2,
        "success": success,
        "failed": 2 - success,
        "fallback_count": 0,
        "parse_success_rate": 1.0,
        "schema_valid_rate": 1.0,
        "safety_pass_rate": success / 2,
        "avg_latency_ms": 10,
        "items": items,
    }


class QzbcBridgeTest(unittest.TestCase):
    def test_prepare_generates_non_mutating_qzbc_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _checkpoint(root, "base")
            trained = _checkpoint(root, "trained")
            qzbc = _qzbc(root)
            before = sorted(
                path.relative_to(qzbc).as_posix()
                for path in qzbc.rglob("*")
                if path.is_file()
            )
            output = root / "bundle"
            manifest = prepare_bundle(
                base_checkpoint=base,
                trained_checkpoint=trained,
                qzbc_root=qzbc,
                output_dir=output,
                base_model_name="qzbc-base",
                trained_model_name="qzbc-trained",
            )
            after = sorted(
                path.relative_to(qzbc).as_posix()
                for path in qzbc.rglob("*")
                if path.is_file()
            )

            self.assertEqual(before, after)
            self.assertEqual(
                manifest["qzbc_contract"]["full_suite_cases"],
                12,
            )
            env = (output / "qzbc.ollama.env").read_text(encoding="utf-8")
            self.assertIn("LLM_PROVIDER=ollama", env)
            self.assertIn("LLM_ALLOW_FALLBACK=false", env)
            self.assertIn("OLLAMA_MODEL=qzbc-base:latest", env)
            register = (
                output / "ollama/register_models.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("ollama", register)
            self.assertIn("qzbc-base", register)
            self.assertIn("qzbc-trained", register)
            convert = (
                output / "ollama/convert_hf_to_gguf.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("convert_hf_to_gguf.py", convert)
            self.assertIn(str(base), convert)
            self.assertIn(str(trained), convert)

    def test_report_comparison_is_paired_by_qzbc_case(self) -> None:
        comparison = compare_reports(
            _report("base", (False, True)),
            _report("trained", (True, True)),
        )
        self.assertEqual(
            comparison["paired_outcomes"]["trained_wins"],
            1,
        )
        self.assertEqual(
            comparison["paired_outcomes"]["trained_losses"],
            0,
        )
        self.assertEqual(len(comparison["case_pairs"]), 2)

    def test_fallback_is_rejected_by_default(self) -> None:
        report = _report("base", (True, True))
        report["fallback_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "fallback"):
            _validate_report(
                report,
                model="base",
                provider="ollama",
                allow_fallback=False,
            )

    def test_runner_switches_only_model_and_writes_outputs(self) -> None:
        reports = {
            "base:latest": _report("base:latest", (False, True)),
            "trained:latest": _report("trained:latest", (True, True)),
        }

        def fake_run(
            qzbc_url,
            *,
            token,
            provider,
            model,
            suite,
            timeout,
        ):
            self.assertEqual(qzbc_url, "http://qzbc")
            self.assertEqual(token, "token")
            self.assertEqual(provider, "ollama")
            self.assertEqual(suite, "full")
            self.assertGreater(timeout, 0)
            return reports[model]

        with tempfile.TemporaryDirectory() as directory, patch(
            "care_opd.qzbc_bridge.run_qzbc_experiment",
            side_effect=fake_run,
        ) as mocked:
            output = Path(directory) / "comparison.json"
            result = run_comparison(
                qzbc_url="http://qzbc",
                base_model="base",
                trained_model="trained",
                output=output,
                token="token",
            )
            self.assertEqual(mocked.call_count, 2)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".cases.csv").is_file())
            self.assertEqual(
                result["paired_outcomes"]["trained_wins"],
                1,
            )

    def test_native_heldout_uses_identical_messages_and_assessor(self) -> None:
        response = {
            "title": "结果",
            "summary": "占位",
            "sections": [],
            "warnings": [],
            "next_actions": [],
            "disclaimer": "",
        }
        messages = [
            {"role": "system", "content": "qzbc system"},
            {"role": "user", "content": "qzbc payload"},
        ]
        row = {
            "id": "heldout-1",
            "variant": "clean",
            "request": {
                "role": "student",
                "task_type": "student_analysis",
                "input_text": "解释",
                "constraints": {},
                "evidence_packet": {},
            },
            "student_messages": messages,
            "verification": {"must_contain": ["已核验"]},
        }
        seen = []

        def fake_chat(**kwargs):
            seen.append((kwargs["model"], kwargs["messages"]))
            value = dict(response)
            if kwargs["model"] == "trained:latest":
                value["summary"] = "已核验"
            return canonical_response(value), 10, {
                "input_tokens": 2,
                "output_tokens": 3,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "heldout.jsonl"
            requests.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output = root / "native.json"
            with patch(
                "care_opd.qzbc_native_eval.check_ollama_models",
                return_value=("base:latest", "trained:latest"),
            ), patch(
                "care_opd.qzbc_native_eval._ollama_chat",
                side_effect=fake_chat,
            ):
                result = evaluate_paired(
                    request_file=requests,
                    output=output,
                    ollama_base_url="http://ollama",
                    base_model="base",
                    trained_model="trained",
                )
            self.assertEqual(seen[0][1], messages)
            self.assertEqual(seen[1][1], messages)
            self.assertEqual(
                result["paired_outcomes"]["trained_wins"],
                1,
            )
            self.assertEqual(result["base_report"]["passed"], 0)
            self.assertEqual(result["trained_report"]["passed"], 1)
            self.assertTrue(output.with_suffix(".cases.csv").is_file())


if __name__ == "__main__":
    unittest.main()
