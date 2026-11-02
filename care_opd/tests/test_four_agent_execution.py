from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from competition_evidence import execute_four_agent_eval as four
from competition_evidence.three_role_eval_common import canonical_json_sha256


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _task() -> dict:
    message = "请根据同一病例材料回答。"
    return {
        "task_id": "T-1",
        "family_token": "F-1",
        "role": "doctor",
        "variant": "clean",
        "arm": "role_aware",
        "source_text_private": "去标识化病例材料",
        "paired_case_material_sha256": canonical_json_sha256("病例材料"),
        "system_prompt_private": "你是医生端智能体。",
        "user_message_private": message,
    }


def _example_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "competition_evidence"
        / "four_agent_endpoints.free.example.json"
    )


def _args(task_path: Path, output_dir: Path, *, execute: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        input=task_path,
        config=_example_config(),
        output_dir=output_dir,
        execute=execute,
        resume=False,
        retry_failures=False,
        isolated_eval_ack="",
        external_memory_ack="",
        remote_data_transfer_ack="",
    )


def test_example_config_has_one_project_and_three_named_free_api_agents() -> None:
    config = json.loads(_example_config().read_text(encoding="utf-8"))
    systems = four.validate_config(config)
    assert len(systems) == 4
    assert sum(row["kind"] == four.PROJECT_KIND for row in systems) == 1
    external = {row["provider_id"]: row for row in systems if row["kind"] == four.EXTERNAL_KIND}
    assert set(external) == {"cloudflare_workers_ai", "groq", "cerebras"}
    assert external["cloudflare_workers_ai"]["base_url_template"] == (
        "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1"
    )
    assert external["groq"]["base_url"] == "https://api.groq.com/openai/v1"
    assert external["cerebras"]["base_url"] == "https://api.cerebras.ai/v1"


def test_config_rejects_wrong_system_count_and_inline_api_key() -> None:
    config = json.loads(_example_config().read_text(encoding="utf-8"))
    config["systems"] = config["systems"][:3]
    with pytest.raises(ValueError, match="exactly four"):
        four.validate_config(config)

    config = json.loads(_example_config().read_text(encoding="utf-8"))
    config["systems"][1]["api_key"] = "must-not-be-here"
    with pytest.raises(ValueError, match="inline secret"):
        four.validate_config(config)


def _local_vllm_config() -> dict:
    config = json.loads(_example_config().read_text(encoding="utf-8"))
    for index, row in enumerate(config["systems"][1:], start=1):
        row.pop("base_url_template", None)
        row["base_url"] = f"http://127.0.0.1:{18100 + index}/v1"
        row.pop("api_key_env", None)
        row["api_key_required"] = False
        row["chat_template_kwargs"] = (
            {"enable_thinking": False} if index == 1 else {}
        )
    return config


def test_local_vllm_config_allows_no_key_and_preserves_template_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _local_vllm_config()
    systems = four.validate_config(config)
    generation = four._generation(config)
    monkeypatch.setenv("SHENNONG_BACKEND_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SHENNONG_BACKEND_MODEL", "shennong-care-opd")
    monkeypatch.setenv("CARE_THREE_ROLE_PROVISION_SECRET", "p" * 40)
    monkeypatch.setenv("CARE_THREE_ROLE_BACKEND_EVAL_KEY", "e" * 40)
    cli = argparse.Namespace(
        execute=True,
        resume=False,
        retry_failures=False,
        isolated_eval_ack=four.engine.ISOLATED_ACK,
        external_memory_ack=four.engine.MEMORY_ACK,
        remote_data_transfer_ack="",
    )
    namespaces = [
        four._engine_args(
            input_path=tmp_path / "tasks.private.jsonl",
            output_path=tmp_path / str(row["system_id"]) / "responses.private.jsonl",
            row=row,
            generation=generation,
            cli=cli,
        )
        for row in systems
    ]
    assert namespaces[1].api_key_env == ""
    assert namespaces[1].chat_template_kwargs_json == {"enable_thinking": False}
    # Static authorization/key validation must complete without touching any
    # of the four loopback endpoints.
    four._static_execution_preflight(systems, namespaces, cli)


def test_plan_only_validates_all_four_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks.private.jsonl"
    _write_jsonl(tasks, [_task()])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan-only attempted network access")

    monkeypatch.setattr(four.engine, "_post_json", forbidden)
    monkeypatch.setattr(four.engine, "_get_json", forbidden)
    result = four.run(_args(tasks, tmp_path / "out"))
    assert result["status"] == "PLAN_ONLY"
    assert len(result["systems"]) == 4
    assert all(row["planned_task_count"] == 1 for row in result["systems"])
    assert not any(
        (tmp_path / "out" / row["system_id"] / "responses.private.jsonl").exists()
        for row in result["systems"]
    )


def test_execute_fails_for_missing_remote_key_before_first_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = tmp_path / "tasks.private.jsonl"
    _write_jsonl(tasks, [_task()])
    args = _args(tasks, tmp_path / "out", execute=True)
    args.isolated_eval_ack = four.engine.ISOLATED_ACK
    args.external_memory_ack = four.engine.MEMORY_ACK
    args.remote_data_transfer_ack = four.engine.REMOTE_ACK
    monkeypatch.setenv("SHENNONG_BACKEND_PROVIDER", "test-provider")
    monkeypatch.setenv("SHENNONG_BACKEND_MODEL", "test-model")
    monkeypatch.setenv("CARE_THREE_ROLE_PROVISION_SECRET", "p" * 40)
    monkeypatch.setenv("CARE_THREE_ROLE_BACKEND_EVAL_KEY", "e" * 40)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("network was reached before static preflight completed")

    monkeypatch.setattr(four.engine, "_post_json", forbidden)
    monkeypatch.setattr(four.engine, "_get_json", forbidden)
    with pytest.raises(ValueError, match="CLOUDFLARE_API_TOKEN"):
        four.run(args)
    assert calls == 0
