from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from competition_evidence import execute_three_role_eval as execution_engine
from competition_evidence.execute_three_role_eval import (
    _validate_system_response,
    execute,
)
from competition_evidence.prepare_three_role_eval import prepare
from competition_evidence.score_three_role_eval import prepare_review, summarize
from competition_evidence.three_role_eval_common import (
    canonical_json_sha256,
    extract_display_text,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _locked_rows() -> list[dict]:
    rows: list[dict] = []
    families = [
        ("family_path_1", "thyroid_pathology", "甲状腺结节5mm，病理提示良性。"),
        ("family_path_2", "thyroid_pathology", "甲状腺结节6mm，病理仍需复核。"),
        ("family_us_1", "thyroid_ultrasound", "右叶低回声结节7mm，边界清楚。"),
        ("family_us_2", "thyroid_ultrasound", "左叶低回声结节8mm，可见钙化。"),
    ]
    for family, modality, evidence in families:
        for variant in ("clean", "source_outage"):
            retrieved = (
                [
                    {
                        "kind": "real_record",
                        "content": evidence,
                        "source_id": f"PRIVATE_{family}",
                        "metadata": {"must_not_leak": family},
                    }
                ]
                if variant == "clean"
                else []
            )
            rows.append(
                {
                    "case_id": f"{family}_{variant}",
                    "family_id": family,
                    "report_group_hmac": f"hmac_{family}",
                    "evaluation_track": "structured",
                    "variant": variant,
                    "modality": modality,
                    "request": {
                        "role": "doctor",
                        "task_type": "doctor_ai_chat",
                        "input_text": "原医生轨任务说明不应进入共同三角色输入。",
                        "evidence_packet": {
                            "degraded": variant == "source_outage",
                            "retrieved_evidence": retrieved,
                        },
                    },
                }
            )
    return rows


def _prepare_args(locked: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        locked=locked,
        output_dir=output,
        track=None,
        sample_size=2,
        sample_seed=20260824,
        task_seed=20260825,
        variant=None,
        max_message_chars=2000,
        include_prompt_ablation=False,
        overwrite=False,
    )


def test_prepare_freezes_identical_three_role_payloads_and_strips_identifiers(tmp_path: Path) -> None:
    locked = tmp_path / "locked.private.jsonl"
    _write_jsonl(locked, _locked_rows())
    paths = prepare(_prepare_args(locked, tmp_path / "plan"))
    tasks = [json.loads(line) for line in paths["tasks"].read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 2 * 2 * 3

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for task in tasks:
        groups.setdefault((task["family_token"], task["variant"], task["arm"]), []).append(task)
    assert all({row["role"] for row in rows} == {"doctor", "patient", "student"} for rows in groups.values())
    assert all(len({row["user_message_private"] for row in rows}) == 1 for rows in groups.values())
    assert all(len({row["shared_payload_sha256"] for row in rows}) == 1 for rows in groups.values())

    clean = next(row for row in tasks if row["variant"] == "clean")
    outage = next(
        row
        for row in tasks
        if row["variant"] == "evidence_unavailable_counterfactual"
    )
    assert "real_record" in clean["user_message_private"]
    assert "原医生轨任务说明" not in clean["user_message_private"]
    assert "PRIVATE_" not in clean["user_message_private"]
    assert "must_not_leak" not in clean["user_message_private"]
    assert "来源当前不可用" in outage["user_message_private"]
    assert any(anchor in outage["user_message_private"] for anchor in ("甲状腺", "右叶", "左叶"))
    paired: dict[str, list[dict]] = {}
    for row in tasks:
        paired.setdefault(row["family_token"], []).append(row)
    assert all(
        len({row["paired_case_material_sha256"] for row in rows}) == 1
        for rows in paired.values()
    )
    assert len(
        {
            row["shared_payload_sha256"]
            for row in tasks
            if row["variant"] == "evidence_unavailable_counterfactual"
        }
    ) == 2

    public = json.loads(paths["public_manifest"].read_text(encoding="utf-8"))
    assert public["status"] == "PLAN_PREPARED_NO_MODEL_OUTPUTS"
    assert public["design"]["planned_task_count"] == 12
    assert public["design"]["shared_user_payload_byte_identity_verified"] is True
    serialized = json.dumps(public, ensure_ascii=False)
    assert "family_path" not in serialized
    assert "甲状腺结节" not in serialized


def test_executor_is_plan_only_and_fails_closed_before_network(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.private.jsonl"
    _write_jsonl(
        task_path,
        [
            {
                "task_id": "T-1",
                "family_token": "F-1",
                "role": "doctor",
                "variant": "clean",
                "arm": "role_aware",
                "source_text_private": "甲状腺结节5mm",
                "paired_case_material_sha256": "e0b2f06f35c4a6535ded7c1d77cf29364190ca6d184b5b1769966921886eaad9",
                "system_prompt_private": "doctor",
                "user_message_private": "共同输入",
            }
        ],
    )
    base = dict(
        input=task_path,
        output=tmp_path / "responses.private.jsonl",
        mode="system_api",
        base_url="http://127.0.0.1:65530",
        model="",
        api_key_env="OPENAI_API_KEY",
        temperature=0.0,
        top_p=1.0,
        max_tokens=1200,
        generation_seed=42,
        timeout_seconds=0.1,
        retries=0,
        retry_backoff_seconds=0.0,
        progress_every=0,
        checkpoint_every=1,
        resume=False,
        retry_failures=False,
        execute=False,
        isolated_eval_ack="",
        external_memory_ack="",
        remote_data_transfer_ack="",
        provision_isolated_users=False,
        provision_secret_env="CARE_THREE_ROLE_PROVISION_SECRET",
        private_eval_key_env="CARE_THREE_ROLE_BACKEND_EVAL_KEY",
        expected_backend_provider="",
        expected_backend_model="",
        expected_backend_temperature=0.0,
        expected_backend_top_p=1.0,
        expected_backend_seed=42,
        expected_backend_max_output_tokens=1200,
        expected_backend_max_input_tokens=6000,
    )
    plan = execute(argparse.Namespace(**base))
    assert plan["status"] == "PLAN_ONLY"
    assert not (tmp_path / "responses.private.jsonl").exists()

    base["execute"] = True
    with pytest.raises(ValueError, match="isolated-eval-ack"):
        execute(argparse.Namespace(**base))


def test_local_openai_endpoint_can_omit_key_and_forwards_chat_template_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        captured.update(
            {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
        )
        return {"choices": [{"message": {"content": "回答"}}]}

    monkeypatch.setattr(execution_engine, "_post_json", fake_post)
    args = argparse.Namespace(
        base_url="http://127.0.0.1:18100/v1",
        model="huatuogpt3-8b",
        api_key_env="",
        temperature=0.0,
        top_p=1.0,
        max_tokens=1200,
        generation_seed=42,
        timeout_seconds=3.0,
        chat_template_kwargs_json='{"enable_thinking": false}',
    )
    execution_engine._openai_request(
        {
            "system_prompt_private": "角色提示",
            "user_message_private": "同一病例",
        },
        args,
        0,
    )
    assert captured["url"] == "http://127.0.0.1:18100/v1/chat/completions"
    assert captured["headers"] == {}
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert captured["payload"]["messages"][1]["content"] == "同一病例"


def test_private_control_task_token_preserves_frozen_ids_and_meets_backend_contract() -> None:
    valid = "T-12345678"
    assert execution_engine._private_control_task_token({"task_id": valid}) == valid

    frozen = "TCM30-0e166028b5749f751b996a77"
    mapped = execution_engine._private_control_task_token({"task_id": frozen})
    assert mapped == f"T-{frozen}"
    assert execution_engine.PRIVATE_CONTROL_TASK_TOKEN_RE.fullmatch(mapped)
    assert execution_engine._private_control_task_token({"task_id": frozen}) == mapped


@pytest.mark.parametrize(
    "value",
    ("[]", "not-json", 1),
)
def test_chat_template_kwargs_must_be_a_json_object(value: object) -> None:
    with pytest.raises(ValueError, match="JSON object|JSON text"):
        execution_engine._parse_chat_template_kwargs(
            argparse.Namespace(chat_template_kwargs_json=value)
        )


def test_qzbc_display_parser_keeps_full_structured_answer() -> None:
    payload = {
        "success": True,
        "data": {
            "answer": "只有摘要",
            "generated_response": {
                "title": "医生专业问答",
                "summary": "只有摘要",
                "sections": [{"heading": "证据", "content": "完整正文"}],
                "warnings": ["需要复核"],
                "next_actions": ["补充检查"],
                "disclaimer": "由医生确认",
            },
        },
    }
    text = extract_display_text(payload)
    assert "完整正文" in text
    assert "需要复核" in text
    assert "补充检查" in text
    assert "由医生确认" in text


def test_system_response_must_match_preflight_provider_and_private_audit() -> None:
    task = {
        "task_id": "T-12345678",
        "variant": "clean",
        "paired_case_material_sha256": "a" * 64,
    }
    control = {
        "protocol": "three_role_paired_fact_v1",
        "condition": "clean",
        "task_token": "T-12345678",
        "case_material_sha256": "a" * 64,
        "expected_retrieved_evidence_count": 1,
        "actual_retrieved_evidence_count": 1,
        "evidence_count_verified": True,
        "live_adapter_calls_bypassed": True,
        "graphiti_bypassed": True,
        "mem0_bypassed": True,
    }
    raw = {
        "data": {
            "llm_result": {
                "success": True,
                "provider": "openai_compatible",
                "model": "unexpected-model",
                "metadata": {
                    "fallback_used": False,
                    "parse_success": True,
                    "schema_valid": True,
                },
            },
            "evidence_packet": {
                "retrieved_evidence": [{"source_id": "locked"}],
                "backend_summary": {"private_eval_evidence_control": control},
            },
        }
    }
    with pytest.raises(RuntimeError, match="provider does not match"):
        _validate_system_response(
            task,
            raw,
            {"active_provider": "ollama", "model": "qwen2.5:7b"},
        )


def test_provider_failure_sentinel_retries_before_first_answer_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = {
        "task_id": "T-12345678",
        "family_token": "F-12345678",
        "role": "doctor",
        "variant": "clean",
        "arm": "role_aware",
        "modality": "tcm_open_case_reasoning",
        "source_text_private": "患者咳嗽，舌淡，苔薄白。",
        "paired_case_material_sha256": "a" * 64,
        "system_prompt_private": "医生角色",
        "user_message_private": "请分析病例。",
    }
    tasks_path = tmp_path / "tasks.private.jsonl"
    output_path = tmp_path / "responses.private.jsonl"
    _write_jsonl(tasks_path, [task])
    candidates = iter(
        [
            {
                "success": True,
                "data": {
                    "answer": "LLM provider failed",
                    "generated_response": {},
                    "llm_result": {
                        "success": False,
                        "provider": "openai_compatible",
                        "model": "care-model",
                        "error_code": "LLM_PROVIDER_FAILED",
                        "metadata": {
                            "provider_unavailable": True,
                            "fallback_reason": "http_500:generation_failed",
                        },
                    },
                },
            },
            {
                "success": True,
                "data": {
                    "answer": "基于病例可见事实进行辨证分析。",
                    "generated_response": {"summary": "基于病例可见事实进行辨证分析。"},
                    "llm_result": {
                        "success": True,
                        "provider": "openai_compatible",
                        "model": "care-model",
                        "metadata": {
                            "provider_unavailable": False,
                            "fallback_used": False,
                        },
                    },
                },
            },
        ]
    )
    monkeypatch.setattr(
        execution_engine,
        "_system_preflight",
        lambda args: {
            "system_api_profile": "locked_evidence",
            "active_provider": "openai_compatible",
            "model": "care-model",
        },
    )
    monkeypatch.setattr(
        execution_engine,
        "_system_request",
        lambda task, args, attempt: next(candidates),
    )
    monkeypatch.setattr(
        execution_engine,
        "_validate_system_response",
        lambda task, raw, preflight: {"qualified": True},
    )
    args = argparse.Namespace(
        input=tasks_path,
        output=output_path,
        mode="system_api",
        system_api_profile="locked_evidence",
        base_url="http://127.0.0.1:8000",
        model="",
        api_key_env="",
        temperature=0.0,
        top_p=1.0,
        max_tokens=1200,
        generation_seed=42,
        timeout_seconds=1.0,
        retries=1,
        retry_backoff_seconds=0.0,
        progress_every=0,
        checkpoint_every=1,
        resume=False,
        retry_failures=False,
        abort_on_first_failure=False,
        execute=True,
        isolated_eval_ack=execution_engine.ISOLATED_ACK,
        external_memory_ack=execution_engine.MEMORY_ACK,
        remote_data_transfer_ack="",
        full_product_stack_ack="",
        provision_isolated_users=True,
        provision_secret_env="CARE_THREE_ROLE_PROVISION_SECRET",
        private_eval_key_env="CARE_THREE_ROLE_BACKEND_EVAL_KEY",
        expected_backend_provider="openai_compatible",
        expected_backend_model="care-model",
        expected_backend_temperature=0.0,
        expected_backend_top_p=1.0,
        expected_backend_seed=42,
        expected_backend_max_output_tokens=1200,
        expected_backend_max_input_tokens=6000,
    )
    manifest = execute(args)
    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["status"] == "success"
    assert row["attempt_count"] == 2
    assert row["selected_attempt"] == 2
    assert row["display_response_private"] == "基于病例可见事实进行辨证分析。"
    assert "LLM provider failed" not in row["display_response_private"]
    assert len(row["pre_output_operational_failures_private"]) == 1
    assert row["post_output_audit_failure"] is False
    assert manifest["pre_output_operational_failure_count"] == 1
    assert manifest["rows_with_pre_output_operational_failure_count"] == 1


def _response_for(task: dict) -> str:
    if task["variant"] == "evidence_unavailable_counterfactual":
        return "当前证据不足，无法确认，建议复核并咨询专业人员。"
    if task["role"] == "doctor":
        return "甲状腺结节需要依据现有证据分层，并由医生复核。"
    if task["role"] == "patient":
        return "简单来说，甲状腺结节需要进一步确认；不能替代诊断，请咨询医生。"
    return "知识点：甲状腺结节。请按证据、辨析步骤和复习要点学习。"


def test_review_rejects_answer_not_bound_to_exact_task(tmp_path: Path) -> None:
    locked = tmp_path / "locked.private.jsonl"
    _write_jsonl(locked, _locked_rows())
    plan_paths = prepare(_prepare_args(locked, tmp_path / "plan"))
    tasks = [
        json.loads(line)
        for line in plan_paths["tasks"].read_text(encoding="utf-8").splitlines()
    ]
    responses = []
    for task in tasks:
        responses.append(
            {
                "task_id": task["task_id"],
                "task_payload_sha256": "0" * 64,
                "family_token": task["family_token"],
                "role": task["role"],
                "variant": task["variant"],
                "arm": task["arm"],
                "status": "success",
                "display_response_private": _response_for(task),
            }
        )
    responses_path = tmp_path / "responses.private.jsonl"
    _write_jsonl(responses_path, responses)
    with pytest.raises(ValueError, match="not bound to the exact frozen task"):
        prepare_review(
            argparse.Namespace(
                tasks=plan_paths["tasks"],
                responses=responses_path,
                output_dir=tmp_path / "review",
                reviewers=None,
                overwrite=False,
            )
        )


def _fill_rating_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        row["fact_fidelity_0_4"] = "4"
        row["role_fit_0_4"] = "4"
        row["evidence_handling_0_4"] = "4"
        row["permission_boundary_pass_0_1"] = "1"
        row["overall_usable_0_1"] = "1"
        row["unsupported_major_claim_0_1"] = "0"
        row["doctor_language_leak_0_1_or_na"] = "0" if row["intended_role"] == "患者" else "NA"
        row["degradation_correct_0_1_or_na"] = (
            "1"
            if row["condition"] == "evidence_unavailable_counterfactual"
            else "NA"
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fill_bundle_csv(path: Path, key_path: Path) -> None:
    keys = {
        row["bundle_id"]: row
        for row in (json.loads(line) for line in key_path.read_text(encoding="utf-8").splitlines())
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        mapping = keys[row["bundle_id"]]["candidate_role_mapping_private"]
        inverse = {role: candidate for candidate, role in mapping.items()}
        row["guessed_doctor_candidate"] = inverse["doctor"]
        row["guessed_patient_candidate"] = inverse["patient"]
        row["guessed_student_candidate"] = inverse["student"]
        row["shared_core_fact_consistency_0_1"] = "1"
        row["cross_role_contradiction_0_1"] = "0"
        row["patient_doctor_style_leakage_0_1"] = "0"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_two_round_review_and_family_cluster_summary(tmp_path: Path) -> None:
    locked = tmp_path / "locked.private.jsonl"
    _write_jsonl(locked, _locked_rows())
    plan_paths = prepare(_prepare_args(locked, tmp_path / "plan"))
    tasks = [json.loads(line) for line in plan_paths["tasks"].read_text(encoding="utf-8").splitlines()]
    responses_path = tmp_path / "responses.private.jsonl"
    _write_jsonl(
        responses_path,
        [
            {
                "task_id": task["task_id"],
                "task_payload_sha256": canonical_json_sha256(task),
                "family_token": task["family_token"],
                "role": task["role"],
                "variant": task["variant"],
                "arm": task["arm"],
                "status": "success",
                "display_response_private": _response_for(task),
            }
            for task in tasks
        ],
    )
    review_paths = prepare_review(
        argparse.Namespace(
            tasks=plan_paths["tasks"],
            responses=responses_path,
            output_dir=tmp_path / "review",
            reviewers=None,
            overwrite=False,
        )
    )
    bundle_packet = [
        json.loads(line)
        for line in review_paths["three_role_blind_bundle_packet.private.jsonl"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all("candidate_role_mapping_private" not in row for row in bundle_packet)
    assert all(set(row["anonymous_candidates_private"]) == {"A", "B", "C"} for row in bundle_packet)
    assert review_paths["three_role_review_manifest.private.json"].is_file()
    automatic_md = review_paths["three_role_automatic_summary.public.md"].read_text(
        encoding="utf-8"
    )
    assert "病例锚点召回" in automatic_md

    ratings = [
        review_paths["three_role_expert_ratings.reviewer_1.private.csv"],
        review_paths["three_role_expert_ratings.reviewer_2.private.csv"],
    ]
    bundle_ratings = [
        review_paths["three_role_bundle_ratings.reviewer_1.private.csv"],
        review_paths["three_role_bundle_ratings.reviewer_2.private.csv"],
    ]
    bundle_key = review_paths["three_role_blind_bundle_key.private.jsonl"]
    for path in ratings:
        _fill_rating_csv(path)
    for path in bundle_ratings:
        _fill_bundle_csv(path, bundle_key)
    original_key = bundle_key.read_bytes()
    bundle_key.write_bytes(original_key + b"\n")
    with pytest.raises(ValueError, match="blind role key hash mismatch"):
        summarize(
            argparse.Namespace(
                tasks=plan_paths["tasks"],
                responses=responses_path,
                ratings=ratings,
                bundle_ratings=bundle_ratings,
                bundle_key=bundle_key,
                review_manifest=review_paths["three_role_review_manifest.private.json"],
                output_dir=tmp_path / "tampered_results",
                reviewers=None,
                bootstrap_replicates=20,
                bootstrap_seed=20260826,
                overwrite=False,
            )
        )
    bundle_key.write_bytes(original_key)
    result_paths = summarize(
        argparse.Namespace(
            tasks=plan_paths["tasks"],
            responses=responses_path,
            ratings=ratings,
            bundle_ratings=bundle_ratings,
            bundle_key=bundle_key,
            review_manifest=review_paths["three_role_review_manifest.private.json"],
            output_dir=tmp_path / "results",
            reviewers=None,
            bootstrap_replicates=200,
            bootstrap_seed=20260826,
            overwrite=False,
        )
    )
    public = json.loads(result_paths["three_role_results.public.json"].read_text(encoding="utf-8"))
    assert public["status"] == "COMPLETE_DOUBLE_REVIEWED"
    assert public["record_count"] == 2
    metrics = {row["metric"]: row for row in public["primary_endpoints"]}
    assert metrics["三角色联合合规率"]["estimate"] == 1.0
    assert metrics["三角色盲态完全辨识率"]["estimate"] == 1.0
    assert metrics["证据来源不可用时三角色联合降级率"]["estimate"] == 1.0
    serialized = json.dumps(public, ensure_ascii=False)
    assert "甲状腺结节5mm" not in serialized
    assert "family_path" not in serialized
