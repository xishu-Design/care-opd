import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from care_opd.qzbc_contract import (
    OUTPUT_FIELDS,
    QZBCRequest,
    build_qzbc_messages,
    build_qzbc_stream_messages,
    canonical_response,
    load_qzbc_prompts,
    parse_generated_response,
)
from care_opd.qzbc_rollout import assess_response
from care_opd.qzbc_dataset import compile_dataset


def _target():
    return {
        "title": "证据提示",
        "summary": "当前证据不足，需要复核。",
        "sections": [
            {"heading": "边界", "content": "仅整理现有资料。", "source_ids": []}
        ],
        "warnings": ["证据不足。"],
        "next_actions": ["请人工复核。"],
        "disclaimer": "不能替代人工复核。",
    }


def test_qzbc_messages_and_generated_response_contract():
    request = QZBCRequest(
        role="patient",
        task_type="patient_education",
        input_text="解释当前资料。",
        evidence_packet={
            "retrieved_evidence": [{"source_id": "SRC_1", "summary": "已复核资料"}],
            "conversation_context": {
                "include_assistant_answers": True,
                "recent_turns": [
                    {
                        "question": "上一问",
                        "answer": "已核验回答",
                        "answer_trusted": True,
                    }
                ],
            },
        },
    )
    messages = build_qzbc_messages(
        request, {"patient_education": "qzbc native system"}
    )
    assert [item["role"] for item in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    payload = json.loads(messages[-1]["content"])
    assert "conversation_context" not in payload["evidence_packet"]
    assert tuple(parse_generated_response(canonical_response(_target()))) == OUTPUT_FIELDS


def test_qzbc_generated_response_accepts_production_json_envelopes():
    target = canonical_response(_target())
    expected = parse_generated_response(target)
    assert parse_generated_response(f"```json\n{target}\n```") == expected
    assert parse_generated_response(f"以下是 JSON：\n{target}") == expected


def _load_provider_pure_message_builder(qzbc_root: Path):
    """Execute qzbc's pure prompt functions without importing its backend deps."""

    provider = (
        qzbc_root
        / "backend"
        / "app"
        / "llm"
        / "providers"
        / "ollama_provider.py"
    )
    tree = ast.parse(provider.read_text(encoding="utf-8"), filename=str(provider))
    names = {
        "_system_prompt",
        "_stream_system_prompt",
        "_chat_messages",
        "_user_payload",
        "_stream_user_payload",
        "_prompt_evidence_packet",
        "_conversation_context_meta",
        "_hard_constraints",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    namespace = {"json": json}
    prompt_names = {
        "doctor_review_text": "DOCTOR_REVIEW_SYSTEM_PROMPT",
        "doctor_ai_chat": "DOCTOR_AI_CHAT_SYSTEM_PROMPT",
        "patient_education": "PATIENT_EDUCATION_SYSTEM_PROMPT",
        "patient_diet_guidance": "PATIENT_DIET_GUIDANCE_SYSTEM_PROMPT",
        "student_analysis": "STUDENT_ANALYSIS_SYSTEM_PROMPT",
        "student_quiz_explanation": "STUDENT_QUIZ_EXPLANATION_SYSTEM_PROMPT",
        "student_ai_chat": "STUDENT_AI_CHAT_SYSTEM_PROMPT",
        "quiz_generation": "QUIZ_GENERATION_SYSTEM_PROMPT",
    }
    for task, prompt in load_qzbc_prompts(qzbc_root).items():
        namespace[prompt_names[task]] = prompt
    exec(compile(ast.fix_missing_locations(module), str(provider), "exec"), namespace)
    return namespace["_chat_messages"]


def test_qzbc_contract_exactly_matches_provider_non_streaming_messages():
    workspace = Path(__file__).resolve().parents[2]
    qzbc_root = workspace / "qzbc_source" / "qzbc-main"
    if not qzbc_root.is_dir():
        pytest.skip("qzbc source checkout is not part of this test environment")
    request = QZBCRequest(
        role="student",
        task_type="student_ai_chat",
        input_text="只回答本轮问题。",
        constraints={"case": "parity"},
        evidence_packet={
            "retrieved_evidence": [{"source_id": "SRC_PARITY"}],
            "memory_used": [
                {"source": "sql_recent_chat", "memory_type": "ai_chat_turn"},
                {"source": "mem0", "summary": "style only"},
            ],
            "conversation_context": {
                "turn_count": 1,
                "available_turn_count": 1,
                "follow_up_detected": True,
                "context_mode": "recent_window",
                "include_assistant_answers": True,
                "recent_turns": [
                    {
                        "question": "前一问",
                        "answer": "已核验前答",
                        "answer_trusted": True,
                    }
                ],
            },
        },
    )
    prompts = load_qzbc_prompts(qzbc_root)
    native_builder = _load_provider_pure_message_builder(qzbc_root)
    native_request = SimpleNamespace(**request.to_mapping())
    assert build_qzbc_messages(request, prompts) == native_builder(
        native_request, streaming=False
    )
    assert build_qzbc_stream_messages(request, prompts) == native_builder(
        native_request, streaming=True
    )


def test_qzbc_recovery_verifier_uses_generated_response_fields():
    row = {
        "variant": "source_outage",
        "request": {
            "role": "patient",
            "task_type": "patient_education",
            "input_text": "解释。",
            "constraints": {},
            "evidence_packet": {
                "retrieved_evidence": [],
                "tool_calls": [{"success": False, "error_code": "SOURCE_UNAVAILABLE"}],
            },
        },
    }
    passed = assess_response(row, canonical_response(_target()))
    assert passed.passed
    failed = assess_response(
        row,
        canonical_response(
            {
                **_target(),
                "summary": "绝对安全，可直接处理。",
                "warnings": [],
                "next_actions": [],
                "disclaimer": "",
            }
        ),
    )
    assert not failed.passed
    assert "summary" in failed.components
    assert "warnings" in failed.components


def test_qzbc_verifier_honors_seed_verification_contract():
    target = _target()
    target["sections"][0]["source_ids"] = ["SRC_1"]
    row = {
        "variant": "clean",
        "request": {
            "role": "student",
            "task_type": "student_ai_chat",
            "input_text": "解释。",
            "constraints": {},
            "evidence_packet": {
                "retrieved_evidence": [{"source_id": "SRC_1"}],
            },
        },
        "verification": {
            "must_contain": ["需要复核"],
            "must_not_contain": ["可直接发布"],
            "required_source_ids": ["SRC_1"],
        },
    }
    assert assess_response(row, canonical_response(target)).passed
    target["summary"] = "可直接发布。"
    target["sections"][0]["source_ids"] = []
    failed = assess_response(row, canonical_response(target))
    assert not failed.passed
    assert "sections" in failed.components


def test_qzbc_dataset_emits_verified_clean_recovery_and_opd_requests(tmp_path):
    workspace = Path(__file__).resolve().parents[2]
    qzbc_root = workspace / "qzbc_source" / "qzbc-main"
    if not qzbc_root.is_dir():
        pytest.skip("qzbc source checkout is not part of this test environment")
    source = tmp_path / "seed.jsonl"
    seed = {
        "id": "native-1",
        "request": {
            "role": "student",
            "task_type": "student_ai_chat",
            "input_text": "解释已核验证据。",
            "constraints": {},
            "evidence_packet": {
                "retrieved_evidence": [
                    {"source_id": "SRC_1", "summary": "已核验教学资料"}
                ]
            },
        },
        "target": {
            **_target(),
            "sections": [
                {
                    "heading": "边界",
                    "content": "仅整理已核验教学资料。",
                    "source_ids": ["SRC_1"],
                }
            ],
        },
        "verification": {
            "must_contain": ["复核"],
            "required_source_ids": ["SRC_1"],
        },
    }
    source.write_text(json.dumps(seed, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "compiled"
    manifest = compile_dataset(
        source,
        qzbc_root,
        output,
        variants=("clean", "source_outage"),
    )
    assert manifest["counts"]["clean_sft"] == 1
    assert manifest["counts"]["recovery_sft"] == 1
    assert manifest["counts"]["stream_retention_sft"] == 1
    assert manifest["counts"]["clean_mixed_sft"] == 2
    requests = [
        json.loads(line)
        for line in (output / "qzbc_on_policy_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(requests) == 2
    for row in requests:
        assert assess_response(row, row["oracle_response"]).passed
