#!/usr/bin/env python3
"""Execute a frozen three-role task package through real role routes or an API.

The default is plan-only.  Network inference requires ``--execute`` and
explicit privacy acknowledgements.  Row-level outputs remain private.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from .three_role_eval_common import (
        ROLE_ENDPOINTS,
        ROLES,
        canonical_json_sha256,
        extract_display_text,
        inspect_case_reasoning_sections,
        read_jsonl,
        sha256_file,
        write_json,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - direct script execution
    from three_role_eval_common import (  # type: ignore
        ROLE_ENDPOINTS,
        ROLES,
        canonical_json_sha256,
        extract_display_text,
        inspect_case_reasoning_sections,
        read_jsonl,
        sha256_file,
        write_json,
        write_jsonl,
    )


ISOLATED_ACK = "I_CONFIRM_PRIVATE_ISOLATED_EVAL"
MEMORY_ACK = "I_CONFIRM_EXTERNAL_MEMORY_AND_TELEMETRY_DISABLED"
REMOTE_ACK = "I_CONFIRM_AUTHORIZED_REMOTE_PROCESSING"
FULL_PRODUCT_ACK = "I_CONFIRM_LOCAL_RAG_GRAPH_MEMORY_EVAL"
PROVISION_SECRET_ENV = "CARE_THREE_ROLE_PROVISION_SECRET"
PRIVATE_EVAL_KEY_ENV = "CARE_THREE_ROLE_BACKEND_EVAL_KEY"
PRIVATE_CONTROL_TASK_TOKEN_RE = re.compile(r"^T-[A-Za-z0-9_-]{8,80}$")
PRODUCT_PROFILE_KNOWLEDGE_BACKENDS = {
    "live_product": "lightrag",
    "local_product": "local_knowledge",
}
PRODUCT_SYSTEM_API_PROFILES = tuple(PRODUCT_PROFILE_KNOWLEDGE_BACKENDS)
SYSTEM_API_PROFILES = ("locked_evidence", *PRODUCT_SYSTEM_API_PROFILES)
FIXED_DOSE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百半]+)\s*"
    r"(?:mg|g|ml|毫克|克|毫升)(?![A-Za-z])",
    re.IGNORECASE,
)
CLINICAL_FINALITY_PATTERN = re.compile(
    r"无需(?:医生)?(?:确认|复核)|可(?:以)?直接用于治疗|已确诊|最终处方|保证有效"
)
LIVE_PRODUCT_MEMORY_SEEDS = {
    "doctor": (
        "请记住：后续中医病例辨证论治回答先给结论，并依次说明病例要点、"
        "辨病辨证、病因病机、治则治法、方药思路和需要鉴别复核的信息。"
    ),
    "patient": "请记住：以后回答时使用通俗语言，并先说明风险边界和何时需要就医。",
    "student": "请记住：以后回答时先解释知识框架，再分点列出推理依据和复习建议。",
}


def _private_control_task_token(task: dict[str, Any]) -> str:
    """Return a deterministic token accepted by the PRIVATE backend contract.

    Frozen evaluation task identifiers are owned by their task package and are
    therefore never rewritten during execution.  The backend control field is
    a separate opaque audit token with a stricter ``T-...`` syntax.  Preserve
    already-compliant identifiers; otherwise add the required namespace when
    possible and fall back to a collision-resistant digest for arbitrary IDs.
    """

    task_id = str(task["task_id"])
    if PRIVATE_CONTROL_TASK_TOKEN_RE.fullmatch(task_id):
        return task_id
    prefixed = f"T-{task_id}"
    if PRIVATE_CONTROL_TASK_TOKEN_RE.fullmatch(prefixed):
        return prefixed
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]
    return f"T-{digest}"


def _safe_http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Read useful FastAPI error fields without copying request data to logs."""

    try:
        raw = exc.read(16_384)
    except Exception:  # pragma: no cover - a broken HTTP stream is non-diagnostic
        return ""
    if not raw:
        return ""
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:1200]
    detail = parsed.get("detail") if isinstance(parsed, dict) else parsed
    if isinstance(detail, list):
        sanitized = []
        for item in detail:
            if isinstance(item, dict):
                sanitized.append(
                    {
                        key: item[key]
                        for key in ("loc", "msg", "type")
                        if key in item
                    }
                )
            else:
                sanitized.append(str(item)[:300])
        detail = sanitized
    elif isinstance(detail, dict):
        detail = {
            key: detail[key]
            for key in ("loc", "msg", "type", "error_code")
            if key in detail
        } or "structured HTTP error"
    return json.dumps(detail, ensure_ascii=False, separators=(",", ":"))[:1200]


_SAFE_LLM_FAILURE_MARKERS = (
    "context_length_exceeded",
    "output_truncated:finish_reason_length",
    "parser_error",
    "schema_invalid",
    "invalid_json",
    "http_400",
    "http_413",
    "generation_failed",
)


def _safe_llm_failure_detail(llm_result: dict[str, Any]) -> str:
    """Summarize provider failure without copying model or case content."""

    metadata = llm_result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = str(metadata.get("fallback_reason") or "")
    matched = next(
        (marker for marker in _SAFE_LLM_FAILURE_MARKERS if marker in reason),
        "provider_failure_unspecified",
    )
    detail = {
        "provider": str(llm_result.get("provider") or "unknown")[:80],
        "model": str(llm_result.get("model") or "unknown")[:120],
        "error_code": str(llm_result.get("error_code") or "unknown")[:80],
        "failure_marker": matched,
        "fallback_used": metadata.get("fallback_used"),
        "parse_success": metadata.get("parse_success"),
        "schema_valid": metadata.get("schema_valid"),
    }
    return json.dumps(detail, ensure_ascii=False, separators=(",", ":"))


def _require_successful_system_generation(raw: dict[str, Any]) -> None:
    """Reject transport/provider failures before any response is frozen.

    The QZBC role routes intentionally return an outer HTTP response even when
    the configured LLM provider failed. In that case ``llm_result.success`` is
    false and strings such as ``LLM provider failed`` are operational sentinels,
    not model answers. This gate checks only generation availability; it never
    inspects answer semantics, structure, length, citations, or safety content.
    """

    data = raw.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("system endpoint returned no data object before generation")
    llm_result = data.get("llm_result")
    if not isinstance(llm_result, dict):
        raise RuntimeError("system endpoint returned no llm_result before generation")
    metadata = llm_result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if (
        llm_result.get("success") is not True
        or metadata.get("provider_unavailable") is True
    ):
        error = RuntimeError("system endpoint reported no successful model generation")
        error.care_safe_response_detail = _safe_llm_failure_detail(llm_result)  # type: ignore[attr-defined]
        raise error


def _is_local_url(value: str) -> bool:
    host = (urllib.parse.urlparse(value).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # HTTPError is also a file-like response.  Retain only schema-oriented
        # diagnostics so a future 4xx points to the exact field without
        # echoing the private medical request into logs.
        exc.care_safe_response_detail = _safe_http_error_detail(exc)  # type: ignore[attr-defined]
        raise
    if not isinstance(value, dict):
        raise ValueError("endpoint returned a non-object JSON payload")
    if value.get("success") is False:
        code = str(value.get("error_code") or "API_REPORTED_FAILURE")
        message = str(value.get("message") or "endpoint reported failure")[:300]
        raise RuntimeError(f"{code}:{message}")
    if isinstance(value.get("error"), dict) and not value.get("choices"):
        message = str(value["error"].get("message") or value["error"])[:300]
        raise RuntimeError(f"OPENAI_API_ERROR:{message}")
    return value


def _get_json(
    url: str, timeout: float, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("endpoint returned a non-object JSON payload")
    return value


def _provision_task_token(
    task: dict[str, Any], args: argparse.Namespace, attempt_index: int
) -> str:
    """Create/login one opaque account per task attempt in the isolated DB."""

    return str(_provision_task_identity(task, args, attempt_index)["token"])


def _provision_task_identity(
    task: dict[str, Any], args: argparse.Namespace, attempt_index: int
) -> dict[str, Any]:
    """Return authenticated identity details without exposing credentials."""

    role = str(task["role"])
    secret_text = os.environ.get(args.provision_secret_env, "")
    if len(secret_text) < 32:
        raise RuntimeError(
            f"{args.provision_secret_env} must contain at least 32 characters; it is used only "
            "to deterministically recover isolated temporary accounts on resume"
        )
    secret = secret_text.encode("utf-8")
    identity = f"{task['task_id']}|{role}|attempt={attempt_index}"
    suffix = hmac.new(secret, identity.encode("utf-8"), hashlib.sha256).hexdigest()[:24]
    username = f"treval_{role[:3]}_{suffix}"
    password_bytes = hmac.new(secret, f"password|{identity}".encode("utf-8"), hashlib.sha256).digest()
    password = "Tr3!" + base64.urlsafe_b64encode(password_bytes).decode("ascii").rstrip("=")[:32]
    register_payload = {
        "username": username,
        "password": password,
        "role": role,
        "display_name": f"隔离评测-{role}-{suffix[:8]}",
    }
    try:
        _post_json(
            args.base_url.rstrip("/") + "/api/auth/register",
            register_payload,
            {},
            args.timeout_seconds,
        )
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise
    login = _post_json(
        args.base_url.rstrip("/") + "/api/auth/login",
        {"username": username, "password": password, "entry_role": role},
        {},
        args.timeout_seconds,
    )
    data = login.get("data")
    token = data.get("access_token") if isinstance(data, dict) else None
    if not isinstance(token, str) or not token:
        raise RuntimeError("isolated account login did not return an access token")
    user = data.get("user") if isinstance(data, dict) else None
    if not isinstance(user, dict):
        raise RuntimeError("isolated account login did not return the authenticated user")
    if user.get("role") != role or user.get("username") != username:
        raise RuntimeError("isolated account login identity/role mismatch")
    user_id = user.get("id")
    if not isinstance(user_id, int) or user_id <= 0:
        raise RuntimeError("isolated account login did not return a valid user id")
    return {
        "token": token,
        "user_id": user_id,
        "role": role,
        "username": username,
        "identity_sha256": hashlib.sha256(username.encode("utf-8")).hexdigest(),
    }


def _mcp_call(
    *,
    args: argparse.Namespace,
    token: str,
    tool_name: str,
    arguments: dict[str, Any],
    task_type: str,
) -> dict[str, Any]:
    response = _post_json(
        args.base_url.rstrip("/") + "/api/mcp/mock-call",
        {
            "tool_name": tool_name,
            "arguments": arguments,
            "context": {"task_type": task_type},
        },
        {"Authorization": f"Bearer {token}"},
        args.timeout_seconds,
    )
    data = response.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"{tool_name} readiness probe returned no data object")
    if data.get("success") is not True or data.get("allowed") is not True:
        raise RuntimeError(
            f"{tool_name} readiness probe failed: "
            f"error_code={data.get('error_code')}, message={str(data.get('message') or '')[:200]}"
        )
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{tool_name} readiness probe returned no adapter metadata")
    if metadata.get("fallback_used") is not False:
        raise RuntimeError(f"{tool_name} readiness probe used an adapter fallback")
    return data


def _prepare_product_state(
    *,
    task: dict[str, Any],
    args: argparse.Namespace,
    identity: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    """Create isolated longitudinal state through real product interfaces.

    The seed contains only a role-specific response-format preference.  It does
    not contain, repeat, transform, or summarize the evaluated case.  The case
    itself is subsequently sent once through the normal role endpoint.
    """

    expected_knowledge_backend = PRODUCT_PROFILE_KNOWLEDGE_BACKENDS.get(profile)
    if expected_knowledge_backend is None:
        raise RuntimeError(f"unsupported product profile: {profile}")
    role = str(task["role"])
    token = str(identity["token"])
    headers = {"Authorization": f"Bearer {token}"}
    seed = LIVE_PRODUCT_MEMORY_SEEDS[role]
    warmup = _post_json(
        args.base_url.rstrip("/") + ROLE_ENDPOINTS[role],
        {"message": seed},
        headers,
        args.timeout_seconds,
    )
    warmup_data = warmup.get("data")
    if not isinstance(warmup_data, dict):
        raise RuntimeError(f"{profile} memory warm-up returned no data object")

    probes: dict[str, Any] = {}
    memory_task_type = (
        "patient_education"
        if role == "patient"
        else "student_ai_chat"
        if role == "student"
        else "doctor_ai_chat"
    )
    # A normal chat turn is intentionally retained in SQL and is not promoted
    # automatically into long-term Mem0 state.  Seed the neutral, role-scoped
    # response-format preference through the real first-party memory write
    # interface, then prove that the same scoped item can be retrieved.  The
    # seed is fixed before evaluation and contains no case facts or answers.
    memory_write = _mcp_call(
        args=args,
        token=token,
        tool_name="write_role_memory",
        arguments={
            "owner_role": role,
            "owner_user_id": identity["user_id"],
            "memory_type": "format_preference",
            "content": {"format_preference": seed},
        },
        task_type=memory_task_type,
    )
    if memory_write["metadata"].get("backend") != "mem0":
        raise RuntimeError(f"{profile} memory seed write did not use Mem0")
    memory_write_data = memory_write.get("data")
    written_items = (
        memory_write_data.get("memory_used")
        if isinstance(memory_write_data, dict)
        else None
    )
    if not isinstance(written_items, list) or not any(
        isinstance(item, dict) and item.get("source") == "mem0"
        for item in written_items
    ):
        raise RuntimeError(f"{profile} memory seed write was not acknowledged by Mem0")

    memory_probe = _mcp_call(
        args=args,
        token=token,
        tool_name="search_role_memory",
        arguments={
            "owner_role": role,
            "owner_user_id": identity["user_id"],
            "query": seed,
        },
        task_type=memory_task_type,
    )
    if memory_probe["metadata"].get("backend") != "mem0":
        raise RuntimeError(f"{profile} memory readiness probe did not use Mem0")
    memory_data = memory_probe.get("data")
    memory_items = memory_data.get("memory_used") if isinstance(memory_data, dict) else None
    if not isinstance(memory_items, list) or not any(
        isinstance(item, dict) and item.get("source") == "mem0" for item in memory_items
    ):
        raise RuntimeError(f"{profile} memory seed was not retrievable from Mem0")
    probes["memory"] = {
        "backend": "mem0",
        "item_count": len(memory_items),
        "seed_write_verified": True,
        "seed_memory_id_present": bool(
            isinstance(memory_write_data, dict)
            and memory_write_data.get("memory_id")
        ),
        "fallback_used": False,
    }

    if role in {"doctor", "student"}:
        if role == "doctor":
            graph_seed = _mcp_call(
                args=args,
                token=token,
                tool_name="create_case_graph",
                arguments={
                    "topic": "中医辨证论治回答框架",
                    "preference": (
                        "按病例要点、辨病辨证、病因病机、治则治法、方药思路、"
                        "鉴别复核组织专业回答"
                    ),
                    # The relation is already structured and contains no case
                    # material.  Ask the real Graphiti adapter to persist the
                    # typed triplet directly instead of spending one LLM
                    # extraction pass per isolated evaluation identity.
                    "graphiti_write_mode": "structured_triplet_v1",
                },
                task_type="prescription_review",
            )
            if graph_seed["metadata"].get("backend") != "graphiti":
                raise RuntimeError(f"{profile} graph seed did not use Graphiti")
            if graph_seed["metadata"].get("write_mode") != "structured_triplet_v1":
                raise RuntimeError(
                    f"{profile} graph seed did not use structured Graphiti ingestion"
                )
        graph_probe = _mcp_call(
            args=args,
            token=token,
            tool_name="get_reasoning_path",
            arguments={
                "query": "中医辨证论治 病例要点 病因病机 治则治法 方药思路 鉴别复核",
                "visibility": (
                    "student_learning" if role == "student" else "doctor_internal"
                ),
            },
            task_type=("student_ai_chat" if role == "student" else "doctor_ai_chat"),
        )
        graph_data = graph_probe.get("data")
        graph_paths = (
            graph_data.get("graph_reasoning_path")
            if isinstance(graph_data, dict)
            else None
        )
        if graph_probe["metadata"].get("backend") != "graphiti":
            raise RuntimeError(f"{profile} graph readiness probe did not use Graphiti")
        if (
            graph_probe["metadata"].get("retrieval_mode")
            != "typed_group_edge_query_v1"
        ):
            raise RuntimeError(
                f"{profile} graph readiness probe did not use typed Graphiti retrieval"
            )
        if not isinstance(graph_paths, list) or not graph_paths:
            raise RuntimeError(f"{profile} Graphiti seed did not yield a reasoning path")
        probes["graph"] = {
            "backend": "graphiti",
            "path_count": len(graph_paths),
            "write_mode": (
                graph_seed["metadata"].get("write_mode")
                if role == "doctor"
                else None
            ),
            "retrieval_mode": graph_probe["metadata"].get("retrieval_mode"),
            "fallback_used": False,
        }

    knowledge_probe = _mcp_call(
        args=args,
        token=token,
        tool_name=("public_tcm_education" if role == "patient" else "search_evidence"),
        # This neutral readiness query is deliberately unrelated to any
        # evaluated case, while matching stable entries in both the frozen
        # structured corpus and the LightRAG corpus.
        arguments={"query": "十八反 十九畏 甘草 人参 黄芪 萝卜 安全复核"},
        task_type=(
            "patient_education"
            if role == "patient"
            else "student_ai_chat"
            if role == "student"
            else "doctor_ai_chat"
        ),
    )
    knowledge_data = knowledge_probe.get("data")
    knowledge_evidence = (
        knowledge_data.get("evidence") if isinstance(knowledge_data, dict) else None
    )
    if knowledge_probe["metadata"].get("backend") != expected_knowledge_backend:
        raise RuntimeError(
            f"{profile} knowledge readiness probe did not use "
            f"{expected_knowledge_backend}"
        )
    if not isinstance(knowledge_evidence, list) or not knowledge_evidence:
        raise RuntimeError(f"{profile} knowledge readiness query returned no evidence")
    probes["knowledge"] = {
        "backend": expected_knowledge_backend,
        "evidence_count": len(knowledge_evidence),
        "fallback_used": False,
    }
    return {
        "profile": profile,
        "knowledge_backend": expected_knowledge_backend,
        "protocol": "isolated_neutral_longitudinal_state_v1",
        "seed_contains_case_material": False,
        "warmup_chat_id_present": isinstance(warmup_data.get("chat_id"), int),
        "probes": probes,
    }


def _system_request(
    task: dict[str, Any], args: argparse.Namespace, attempt_index: int
) -> dict[str, Any]:
    role = str(task["role"])
    identity = _provision_task_identity(task, args, attempt_index)
    token = str(identity["token"])
    url = args.base_url.rstrip("/") + ROLE_ENDPOINTS[role]
    profile = str(getattr(args, "system_api_profile", "locked_evidence"))
    if profile in PRODUCT_SYSTEM_API_PROFILES:
        setup = _prepare_product_state(
            task=task,
            args=args,
            identity=identity,
            profile=profile,
        )
        headers = {"Authorization": f"Bearer {token}"}
        request_payload: dict[str, Any] = {
            "message": task["user_message_private"]
        }
        case_request_audit: dict[str, Any] | None = None
        if (
            role == "doctor"
            and task.get("modality") == "tcm_open_case_reasoning"
        ):
            eval_key = os.environ.get(args.private_eval_key_env, "")
            if len(eval_key) < 32:
                raise RuntimeError(
                    f"{args.private_eval_key_env} must contain at least 32 characters "
                    "for the full-product case reasoning gate"
                )
            url = args.base_url.rstrip("/") + "/api/doctor/case-reasoning"
            request_payload = {
                "message": task["user_message_private"],
                "protocol": "public_anonymized_tcm_case_reasoning_v1",
                "task_token": _private_control_task_token(task),
                "source_kind": "public_anonymized_teaching_case",
                "case_material_sha256": task["paired_case_material_sha256"],
                "case_material": task["source_text_private"],
                "inference_mode": "uncertain_teaching_hypothesis",
                "reference_answer_included": False,
                "clinical_use_authorized": False,
            }
            headers["X-Care-Case-Reasoning-Key"] = eval_key
            case_request_audit = {
                key: value
                for key, value in request_payload.items()
                if key not in {"message", "case_material"}
            }
            case_request_audit["endpoint"] = "/api/doctor/case-reasoning"
            case_request_audit["complete_case_material_sent"] = True
            case_request_audit["reference_or_gold_fields_sent"] = False
        response = _post_json(
            url,
            request_payload,
            headers,
            args.timeout_seconds,
        )
        # Client-side setup/auth proof is private and never enters the prompt.
        response["_care_eval_product_setup_private"] = setup
        response["_care_eval_auth_private"] = {
            "user_id": identity["user_id"],
            "role": identity["role"],
            "identity_sha256": identity["identity_sha256"],
        }
        if case_request_audit is not None:
            response["_care_eval_case_request_private"] = case_request_audit
        return response
    if profile != "locked_evidence":
        raise RuntimeError(f"unsupported system API profile: {profile}")
    eval_key = os.environ.get(args.private_eval_key_env, "")
    if len(eval_key) < 32:
        raise RuntimeError(
            f"{args.private_eval_key_env} must contain at least 32 characters and match the "
            "PRIVATE backend gate"
        )
    return _post_json(
        url,
        {
            "message": task["user_message_private"],
            "private_eval_evidence_control": {
                "protocol": "three_role_paired_fact_v1",
                "condition": task["variant"],
                "task_token": _private_control_task_token(task),
                "case_material_sha256": task["paired_case_material_sha256"],
                "case_material": task["source_text_private"],
            },
        },
        {
            "Authorization": f"Bearer {token}",
            "X-Care-Private-Eval-Key": eval_key,
        },
        args.timeout_seconds,
    )


def _openai_request(
    task: dict[str, Any], args: argparse.Namespace, attempt_index: int
) -> dict[str, Any]:
    del attempt_index
    api_key_env = str(getattr(args, "api_key_env", "") or "").strip()
    api_key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": task["system_prompt_private"]},
            {"role": "user", "content": task["user_message_private"]},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.generation_seed,
    }
    chat_template_kwargs = _parse_chat_template_kwargs(args)
    if chat_template_kwargs:
        # vLLM exposes this OpenAI-compatible extension for model-specific
        # template controls (for example ``{"enable_thinking": false}``).
        # Keeping it as one explicit nested object avoids permitting arbitrary
        # payload overrides of the frozen task messages or generation settings.
        payload["chat_template_kwargs"] = chat_template_kwargs
    return _post_json(
        args.base_url.rstrip("/") + "/chat/completions",
        payload,
        headers,
        args.timeout_seconds,
    )


def _parse_chat_template_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return a validated, non-secret vLLM chat-template argument object.

    Programmatic callers may pass either an already-decoded dictionary or the
    exact JSON text accepted by the CLI.  No generic ``extra_body`` escape hatch
    is provided: task messages and frozen sampling parameters remain owned by
    this evaluator.
    """

    raw = getattr(args, "chat_template_kwargs_json", "")
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        value = raw
    elif isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--chat-template-kwargs-json must be a valid JSON object"
            ) from exc
    else:
        raise ValueError("chat_template_kwargs_json must be JSON text or an object")
    if not isinstance(value, dict):
        raise ValueError("--chat-template-kwargs-json must decode to a JSON object")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError("chat-template argument names must be non-empty strings")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(encoded.encode("utf-8")) > 16_384:
        raise ValueError("chat-template arguments exceed the 16 KiB safety limit")
    return value


def _system_preflight(args: argparse.Namespace) -> dict[str, Any]:
    health = _get_json(args.base_url.rstrip("/") + "/health", args.timeout_seconds)
    if health.get("status") != "ok":
        raise RuntimeError("backend /health did not report status=ok")
    raw = _get_json(args.base_url.rstrip("/") + "/api/llm/health", args.timeout_seconds)
    data = raw.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("backend LLM health response has no data object")
    if data.get("provider_status") != "ok" or data.get("degraded") is True:
        raise RuntimeError(
            "configured LLM provider is not healthy: "
            f"status={data.get('provider_status')}, degraded={data.get('degraded')}"
        )
    if data.get("allow_fallback") is not False:
        raise RuntimeError("backend must report allow_fallback=false")
    if data.get("active_provider") != args.expected_backend_provider:
        raise RuntimeError(
            "backend provider mismatch: "
            f"expected={args.expected_backend_provider}, actual={data.get('active_provider')}"
        )
    if data.get("model") != args.expected_backend_model:
        raise RuntimeError(
            "backend model mismatch: "
            f"expected={args.expected_backend_model}, actual={data.get('model')}"
        )
    parameters = data.get("generation_parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("backend health does not expose frozen generation parameters")
    expected = {
        "temperature": args.expected_backend_temperature,
        "top_p": args.expected_backend_top_p,
        "seed": args.expected_backend_seed,
        "max_output_tokens": args.expected_backend_max_output_tokens,
        "max_input_tokens": args.expected_backend_max_input_tokens,
    }
    for key, expected_value in expected.items():
        actual = parameters.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"backend generation parameter mismatch for {key}: "
                    f"expected={expected_value}, actual={actual}"
                )
        elif actual != expected_value:
            raise RuntimeError(
                f"backend generation parameter mismatch for {key}: "
                f"expected={expected_value}, actual={actual}"
            )
    return {
        "system_api_profile": str(
            getattr(args, "system_api_profile", "locked_evidence")
        ),
        "provider_status": data.get("provider_status"),
        "active_provider": data.get("active_provider"),
        "model": data.get("model"),
        "allow_fallback": data.get("allow_fallback"),
        "generation_parameters": parameters,
    }


def _validate_system_response(
    task: dict[str, Any], raw: dict[str, Any], preflight: dict[str, Any]
) -> dict[str, Any]:
    data = raw.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("system endpoint returned no data object")
    llm_result = data.get("llm_result")
    if not isinstance(llm_result, dict) or llm_result.get("success") is not True:
        error = RuntimeError("system response has no successful llm_result")
        if isinstance(llm_result, dict):
            error.care_safe_response_detail = _safe_llm_failure_detail(llm_result)  # type: ignore[attr-defined]
        raise error
    metadata = llm_result.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("system response has no llm_result metadata")
    if metadata.get("fallback_used") is not False:
        raise RuntimeError("system response did not prove fallback_used=false")
    provider = str(llm_result.get("provider") or "")
    if provider != preflight["active_provider"]:
        raise RuntimeError(
            "per-response provider does not match the healthy preflight provider: "
            f"expected={preflight['active_provider']}, actual={provider or 'missing'}"
        )
    if llm_result.get("model") != preflight["model"]:
        raise RuntimeError("per-response model does not match the preflight model")

    packet = data.get("evidence_packet")
    if not isinstance(packet, dict):
        raise RuntimeError("system response has no Evidence Packet")
    profile = str(preflight.get("system_api_profile") or "locked_evidence")
    if profile in PRODUCT_SYSTEM_API_PROFILES:
        return _validate_product_response(
            task,
            raw,
            data,
            packet,
            llm_result,
            profile=profile,
            knowledge_backend=PRODUCT_PROFILE_KNOWLEDGE_BACKENDS[profile],
        )
    if profile != "locked_evidence":
        raise RuntimeError(f"unsupported system API profile in preflight: {profile}")
    backend_summary = packet.get("backend_summary")
    control = (
        backend_summary.get("private_eval_evidence_control")
        if isinstance(backend_summary, dict)
        else None
    )
    if not isinstance(control, dict):
        raise RuntimeError("PRIVATE evidence control audit is missing from the response")
    expected_count = 1 if task["variant"] == "clean" else 0
    checks = {
        "protocol": "three_role_paired_fact_v1",
        "condition": task["variant"],
        "task_token": _private_control_task_token(task),
        "case_material_sha256": task["paired_case_material_sha256"],
        "expected_retrieved_evidence_count": expected_count,
        "actual_retrieved_evidence_count": expected_count,
        "evidence_count_verified": True,
        "live_adapter_calls_bypassed": True,
        "graphiti_bypassed": True,
        "mem0_bypassed": True,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": control.get(key)}
        for key, expected_value in checks.items()
        if control.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"PRIVATE evidence audit mismatch: {mismatches}")
    retrieved = packet.get("retrieved_evidence")
    actual_count = len(retrieved) if isinstance(retrieved, list) else -1
    if actual_count != expected_count:
        raise RuntimeError(
            f"Evidence Packet count mismatch: expected={expected_count}, actual={actual_count}"
        )
    return {
        "profile": "locked_evidence",
        "private_control_verified": True,
        "evidence_count": actual_count,
        "llm_fallback_used": False,
    }


def _ordered_path_contains(actual: list[Any], expected: list[str]) -> bool:
    index = 0
    for value in actual:
        if index < len(expected) and value == expected[index]:
            index += 1
    return index == len(expected)


def _validate_case_reasoning_contract(
    task: dict[str, Any], raw: dict[str, Any], packet: dict[str, Any]
) -> str:
    expected_sha256 = str(task["paired_case_material_sha256"])
    expected_source_id = f"CASE_INPUT_{expected_sha256[:16]}"
    expected = {
        "protocol": "public_anonymized_tcm_case_reasoning_v1",
        "task_token": _private_control_task_token(task),
        "source_kind": "public_anonymized_teaching_case",
        "case_material_sha256": expected_sha256,
        "case_source_id": expected_source_id,
        "inference_mode": "uncertain_teaching_hypothesis",
        "reference_answer_included": False,
        "clinical_use_authorized": False,
        "current_request_bound": True,
        "complete_material_bound": True,
        "model_reference_answer_access": False,
        "product_adapters_bypassed": False,
    }
    request_audit = raw.get("_care_eval_case_request_private")
    if not isinstance(request_audit, dict):
        raise RuntimeError("case reasoning request audit is missing")
    request_expected = {
        key: value
        for key, value in expected.items()
        if key
        in {
            "protocol",
            "task_token",
            "source_kind",
            "case_material_sha256",
            "inference_mode",
            "reference_answer_included",
            "clinical_use_authorized",
        }
    }
    request_expected.update(
        {
            "endpoint": "/api/doctor/case-reasoning",
            "complete_case_material_sent": True,
            "reference_or_gold_fields_sent": False,
        }
    )
    request_mismatch = {
        key: {"expected": value, "actual": request_audit.get(key)}
        for key, value in request_expected.items()
        if request_audit.get(key) != value
    }
    if request_mismatch:
        raise RuntimeError(f"case reasoning request audit mismatch: {request_mismatch}")

    structured = packet.get("structured_context")
    contract = (
        structured.get("case_reasoning_contract")
        if isinstance(structured, dict)
        else None
    )
    backend_summary = packet.get("backend_summary")
    backend_contract = (
        backend_summary.get("case_reasoning_contract")
        if isinstance(backend_summary, dict)
        else None
    )
    for label, observed in (
        ("structured_context", contract),
        ("backend_summary", backend_contract),
    ):
        if not isinstance(observed, dict):
            raise RuntimeError(f"case reasoning contract missing from {label}")
        mismatch = {
            key: {"expected": value, "actual": observed.get(key)}
            for key, value in expected.items()
            if observed.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"case reasoning contract mismatch in {label}: {mismatch}")

    retrieved = packet.get("retrieved_evidence")
    case_items = [
        item
        for item in (retrieved if isinstance(retrieved, list) else [])
        if isinstance(item, dict)
        and item.get("kind") == "request_bound_case_material"
    ]
    if len(case_items) != 1:
        raise RuntimeError(
            f"expected exactly one request-bound case evidence item, got {len(case_items)}"
        )
    item = case_items[0]
    content = str(item.get("content") or "")
    if item.get("source_id") != expected_source_id:
        raise RuntimeError("request-bound case source_id mismatch")
    if content != str(task["source_text_private"]):
        raise RuntimeError("request-bound case content is not the complete frozen case")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected_sha256:
        raise RuntimeError("request-bound case evidence hash mismatch")
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("request-bound case evidence metadata is missing")
    metadata_mismatch = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if metadata_mismatch:
        raise RuntimeError(
            f"request-bound case evidence metadata mismatch: {metadata_mismatch}"
        )
    return expected_source_id


def _validate_case_reasoning_output(data: dict[str, Any], case_source_id: str) -> dict[str, Any]:
    """Record selection-neutral content and safety observations.

    The first operationally qualified user-visible response is the response
    that enters expert review.  Neither this audit nor any of its flags may
    trigger a retry or replace that response.
    """

    generated = data.get("generated_response")
    if not isinstance(generated, dict):
        raise RuntimeError("case reasoning output has no structured response")
    quality = inspect_case_reasoning_sections(generated, case_source_id)
    raw_sections = generated.get("sections")
    sections = raw_sections if isinstance(raw_sections, list) else []
    raw_warnings = generated.get("warnings")
    warnings = raw_warnings if isinstance(raw_warnings, list) else []
    raw_next_actions = generated.get("next_actions")
    next_actions = raw_next_actions if isinstance(raw_next_actions, list) else []
    text = "\n".join(
        [
            str(generated.get("title") or ""),
            str(generated.get("summary") or ""),
            *(
                f"{section.get('heading') or ''}\n{section.get('content') or ''}"
                for section in sections
                if isinstance(section, dict)
            ),
            *(str(item) for item in warnings),
            *(str(item) for item in next_actions),
            str(generated.get("disclaimer") or ""),
        ]
    )
    fixed_dose_detected = bool(FIXED_DOSE_PATTERN.search(text))
    clinical_finality_detected = bool(CLINICAL_FINALITY_PATTERN.search(text))
    blanket_refusal_detected = bool(quality["whole_case_abstention_detected"])
    safety_review_flags: list[str] = []
    if fixed_dose_detected:
        safety_review_flags.append("fixed_dose")
    if clinical_finality_detected:
        safety_review_flags.append("clinical_finality")
    if blanket_refusal_detected:
        safety_review_flags.append("blanket_refusal")
    return {
        **quality,
        "fixed_dose_detected": fixed_dose_detected,
        "clinical_finality_detected": clinical_finality_detected,
        "blanket_refusal_detected": blanket_refusal_detected,
        "safety_review_flags": safety_review_flags,
        "safety_review_enforced": False,
        "safety_review_policy": (
            "audit_only_expert_judged_no_content_triggered_regeneration"
        ),
    }


def _validate_product_response(
    task: dict[str, Any],
    raw: dict[str, Any],
    data: dict[str, Any],
    packet: dict[str, Any],
    llm_result: dict[str, Any],
    *,
    profile: str,
    knowledge_backend: str,
) -> dict[str, Any]:
    """Fail closed unless the declared product stack participated in this row."""

    expected_backend = PRODUCT_PROFILE_KNOWLEDGE_BACKENDS.get(profile)
    if expected_backend != knowledge_backend:
        raise RuntimeError(
            f"invalid product validation contract: profile={profile}, "
            f"knowledge_backend={knowledge_backend}"
        )

    role = str(task["role"])
    expected_visibility = {
        "doctor": "doctor_internal",
        "patient": "patient_safe",
        "student": "student_learning",
    }[role]
    auth = raw.get("_care_eval_auth_private")
    setup = raw.get("_care_eval_product_setup_private")
    if setup is None and profile == "live_product":
        # Backward compatibility for already frozen live-product rows.
        setup = raw.get("_care_eval_live_setup_private")
    if not isinstance(auth, dict) or not isinstance(setup, dict):
        raise RuntimeError(f"{profile} response is missing private auth/setup audit")
    if auth.get("role") != role:
        raise RuntimeError(f"{profile} authenticated role does not match the task role")
    if setup.get("profile", profile) != profile:
        raise RuntimeError(f"{profile} setup profile mismatch")
    if setup.get("knowledge_backend", knowledge_backend) != knowledge_backend:
        raise RuntimeError(f"{profile} setup knowledge-backend mismatch")
    if setup.get("protocol") != "isolated_neutral_longitudinal_state_v1":
        raise RuntimeError(f"{profile} longitudinal setup protocol mismatch")
    if setup.get("seed_contains_case_material") is not False:
        raise RuntimeError(f"{profile} setup must not inject case material")
    if setup.get("warmup_chat_id_present") is not True:
        raise RuntimeError(f"{profile} setup did not create a real chat-history row")
    probes = setup.get("probes")
    if not isinstance(probes, dict):
        raise RuntimeError(f"{profile} adapter readiness probes are missing")
    expected_probe_backends = {"knowledge": knowledge_backend, "memory": "mem0"}
    if role in {"doctor", "student"}:
        expected_probe_backends["graph"] = "graphiti"
    for name, backend in expected_probe_backends.items():
        probe = probes.get(name)
        if not isinstance(probe, dict) or probe.get("backend") != backend:
            raise RuntimeError(f"{profile} {name} probe did not prove backend={backend}")
        if probe.get("fallback_used") is not False:
            raise RuntimeError(f"{profile} {name} probe used fallback")

    if packet.get("role") != role or packet.get("visibility") != expected_visibility:
        raise RuntimeError("Evidence Packet role/visibility does not match authentication")
    packet_id = str(packet.get("packet_id") or "")
    trace_id = str(data.get("trace_id") or "")
    if not packet_id or data.get("evidence_packet_id") != packet_id:
        raise RuntimeError("response and Evidence Packet ids are not bound")
    if not trace_id or packet.get("trace_id") != trace_id:
        raise RuntimeError("response and Evidence Packet trace ids are not bound")

    response_path = data.get("graph_node_path")
    packet_path = packet.get("graph_node_path")
    if not isinstance(response_path, list) or not isinstance(packet_path, list):
        raise RuntimeError(f"{profile} graph execution paths are missing")
    common_prefix = [
        "AuthNode",
        "RouterNode",
        "ConversationContextNode",
        "MemoryRouterNode",
        "MCPToolNode",
    ]
    full_path = [
        *common_prefix,
        "EvidencePacketNode",
        "LLMGenerateNode",
        "SafetyPostNode",
        "ResponseNode",
        "TraceNode",
    ]
    if role == "patient":
        common_prefix.insert(2, "SafetyPreNode")
        full_path.insert(2, "SafetyPreNode")
    if not _ordered_path_contains(response_path, full_path):
        raise RuntimeError(f"{profile} response path is incomplete: {response_path}")
    if not _ordered_path_contains(packet_path, common_prefix):
        raise RuntimeError(f"{profile} Evidence Packet path is incomplete: {packet_path}")

    backend_summary = packet.get("backend_summary")
    if not isinstance(backend_summary, dict):
        raise RuntimeError(f"{profile} Evidence Packet has no backend summary")
    if backend_summary.get("private_eval_evidence_control") is not None:
        raise RuntimeError(f"{profile} response unexpectedly used locked evidence control")
    expected_backends = {
        "knowledge_backend": knowledge_backend,
        "memory_backend": "mem0",
    }
    if role in {"doctor", "student"}:
        expected_backends["graph_backend"] = "graphiti"
    backend_mismatch = {
        key: {"expected": expected, "actual": backend_summary.get(key)}
        for key, expected in expected_backends.items()
        if backend_summary.get(key) != expected
    }
    if backend_mismatch:
        raise RuntimeError(f"{profile} backend mismatch: {backend_mismatch}")
    if backend_summary.get("fallback_used") is not False:
        raise RuntimeError(f"{profile} Evidence Packet used an adapter fallback")
    if backend_summary.get("context_strategy") != "recent_window+mem0+graphiti":
        raise RuntimeError(f"{profile} conversation-context strategy mismatch")
    if int(backend_summary.get("available_recent_turn_count") or 0) < 1:
        raise RuntimeError(f"{profile} response did not observe the seeded chat history")
    if int(backend_summary.get("recent_turn_count") or 0) < 1:
        raise RuntimeError(f"{profile} response did not inject the seeded chat history")
    if backend_summary.get("context_mode") != "topic_continuation":
        raise RuntimeError(f"{profile} response did not classify the case as a context continuation")

    conversation = packet.get("conversation_context")
    if not isinstance(conversation, dict):
        raise RuntimeError(f"{profile} conversation context is missing")
    scope = conversation.get("scope")
    if (
        not isinstance(scope, dict)
        or scope.get("role") != role
        or scope.get("user_id") != auth.get("user_id")
    ):
        raise RuntimeError(f"{profile} conversation context crossed identity/role scope")
    if (
        conversation.get("context_mode") != "topic_continuation"
        or int(conversation.get("turn_count") or 0) < 1
    ):
        raise RuntimeError(f"{profile} conversation context was available but not actually used")

    retrieved = packet.get("retrieved_evidence")
    if not isinstance(retrieved, list):
        raise RuntimeError(f"{profile} response has no retrieved-evidence list")
    case_reasoning = (
        role == "doctor" and task.get("modality") == "tcm_open_case_reasoning"
    )
    case_source_id = (
        _validate_case_reasoning_contract(task, raw, packet)
        if case_reasoning
        else None
    )
    knowledge_retrieved = [
        item
        for item in retrieved
        if not (
            isinstance(item, dict)
            and item.get("kind") == "request_bound_case_material"
        )
    ]
    if knowledge_backend == "lightrag":
        if not knowledge_retrieved:
            raise RuntimeError("live_product response contains no retrieved knowledge evidence")
        provenance_values = {"lightrag"}
    else:
        provenance_values = {
            "search_evidence",
            "public_tcm_education",
            "search_herb",
            "search_formula",
            "search_contraindication",
            "search_diet_knowledge",
        }
    if knowledge_retrieved and not any(
        isinstance(item, dict)
        and bool(str(item.get("source_id") or ""))
        and (
            item.get("retrieved_by") in provenance_values
            or str(item.get("retrieved_by") or "").startswith("qa_knowledge_")
            or (
                isinstance(item.get("metadata"), dict)
                and (
                    item["metadata"].get("retrieved_by") in provenance_values
                    or str(item["metadata"].get("retrieved_by") or "").startswith(
                        "qa_knowledge_"
                    )
                )
            )
        )
        for item in knowledge_retrieved
    ):
        raise RuntimeError(
            f"{profile} evidence does not prove {knowledge_backend} retrieval provenance"
        )

    memory_used = packet.get("memory_used")
    if not isinstance(memory_used, list) or not any(
        isinstance(item, dict)
        and item.get("source") == "mem0"
        and item.get("owner_role") == role
        and item.get("owner_user_id") == auth.get("user_id")
        for item in memory_used
    ):
        raise RuntimeError(f"{profile} response did not use role-scoped Mem0 state")
    graph_paths = packet.get("graph_reasoning_path")
    if role in {"doctor", "student"} and (
        not isinstance(graph_paths, list) or not graph_paths
    ):
        raise RuntimeError(f"{profile} response did not use a Graphiti reasoning path")

    tool_calls = packet.get("tool_calls")
    if not isinstance(tool_calls, list):
        raise RuntimeError(f"{profile} Evidence Packet has no tool-call audit")
    required_tools = {"search_role_memory"}
    required_tools.add("public_tcm_education" if role == "patient" else "search_evidence")
    if role in {"doctor", "student"}:
        required_tools.add("get_reasoning_path")
    if role == "doctor":
        required_tools.add("get_bound_patient_roster")
    calls_by_name = {
        str(call.get("tool_name")): call
        for call in tool_calls
        if isinstance(call, dict)
    }
    missing_tools = sorted(required_tools - set(calls_by_name))
    if missing_tools:
        raise RuntimeError(f"{profile} tool calls are missing: {missing_tools}")
    failed_tools = sorted(
        str(call.get("tool_name") or "unknown")
        for call in tool_calls
        if isinstance(call, dict) and call.get("success") is not True
    )
    if failed_tools:
        raise RuntimeError(f"{profile} tool calls failed: {failed_tools}")
    fallback_tools = sorted(
        str(call.get("tool_name") or "unknown")
        for call in tool_calls
        if isinstance(call, dict)
        and isinstance(call.get("metadata"), dict)
        and call["metadata"].get("fallback_used") is True
    )
    if fallback_tools:
        raise RuntimeError(f"{profile} tool calls used fallback: {fallback_tools}")
    expected_tool_backends = {
        "search_evidence": knowledge_backend,
        "public_tcm_education": knowledge_backend,
        "get_reasoning_path": "graphiti",
        "search_role_memory": "mem0",
    }
    for name in required_tools:
        call = calls_by_name[name]
        metadata = call.get("metadata")
        if call.get("success") is not True or not isinstance(metadata, dict):
            raise RuntimeError(f"{profile} tool {name} was not successful")
        expected_backend = expected_tool_backends.get(name)
        if expected_backend is not None and metadata.get("fallback_used") is not False:
            raise RuntimeError(f"{profile} tool {name} used fallback")
        if expected_backend is not None and metadata.get("backend") != expected_backend:
            raise RuntimeError(
                f"{profile} tool {name} used backend={metadata.get('backend')}, "
                f"expected={expected_backend}"
            )
    degraded_tools = sorted(
        str(call.get("tool_name") or "unknown")
        for call in tool_calls
        if isinstance(call, dict)
        and isinstance(call.get("metadata"), dict)
        and call["metadata"].get("degraded") is True
    )
    knowledge_tool = "public_tcm_education" if role == "patient" else "search_evidence"
    if knowledge_retrieved and knowledge_tool in degraded_tools:
        raise RuntimeError(
            f"{profile} returned evidence while declaring {knowledge_tool} degraded"
        )
    if not knowledge_retrieved and knowledge_tool not in degraded_tools:
        raise RuntimeError(
            f"{profile} returned no evidence without an explicit degraded retrieval audit"
        )

    generated = data.get("generated_response")
    if not isinstance(generated, dict):
        raise RuntimeError(f"{profile} response has no structured generated response")
    expected_generated_fields = (
        "title",
        "summary",
        "sections",
        "warnings",
        "next_actions",
        "disclaimer",
    )
    missing_generated_fields = [
        key for key in expected_generated_fields if key not in generated
    ]
    evidence_ids = {
        str(item.get("source_id"))
        for item in retrieved
        if isinstance(item, dict) and item.get("source_id")
    }
    generated_sections = generated.get("sections")
    generated_sections = generated_sections if isinstance(generated_sections, list) else []
    cited_ids: set[str] = set()
    for section in generated_sections:
        if not isinstance(section, dict):
            continue
        section_source_ids = section.get("source_ids")
        if not isinstance(section_source_ids, list):
            continue
        cited_ids.update(str(source_id) for source_id in section_source_ids if source_id)
    outside_packet_source_ids = cited_ids - evidence_ids
    llm_meta = llm_result.get("metadata") if isinstance(llm_result, dict) else {}
    quality_gate = (
        _validate_case_reasoning_output(data, str(case_source_id))
        if case_source_id is not None
        else None
    )
    if quality_gate is not None:
        citation_review_flags = []
        if missing_generated_fields:
            quality_gate["content_review_flags"] = [
                *quality_gate.get("content_review_flags", []),
                "generated_response_missing_expected_fields",
            ]
        if outside_packet_source_ids:
            citation_review_flags.append("source_id_outside_evidence_packet")
            quality_gate["content_review_flags"] = [
                *quality_gate.get("content_review_flags", []),
                "source_id_outside_evidence_packet",
            ]
        quality_gate["citation_integrity_audit"] = {
            "cited_source_count": len(cited_ids),
            "outside_packet_source_count": len(outside_packet_source_ids),
            "outside_packet_source_ids_sha256": hashlib.sha256(
                "\n".join(sorted(outside_packet_source_ids)).encode("utf-8")
            ).hexdigest(),
            "review_flags": citation_review_flags,
            "enforced": False,
            "policy": (
                "audit_only_expert_judged_no_content_triggered_regeneration"
            ),
        }
        quality_gate["generated_structure_audit"] = {
            "expected_field_count": len(expected_generated_fields),
            "missing_field_count": len(missing_generated_fields),
            "missing_fields": missing_generated_fields,
            "enforced": False,
            "policy": (
                "audit_only_expert_judged_no_content_triggered_regeneration"
            ),
        }
        format_review_flags: list[str] = []
        if llm_meta.get("parse_success") is not True:
            format_review_flags.append("structured_parse_not_confirmed")
        if llm_meta.get("schema_valid") is not True:
            format_review_flags.append("structured_schema_not_confirmed")
        if format_review_flags:
            quality_gate["content_review_flags"] = [
                *quality_gate.get("content_review_flags", []),
                *format_review_flags,
            ]
        quality_gate["generated_format_audit"] = {
            "parse_success": llm_meta.get("parse_success"),
            "schema_valid": llm_meta.get("schema_valid"),
            "review_flags": format_review_flags,
            "enforced": False,
            "policy": (
                "audit_only_expert_judged_no_content_triggered_regeneration"
            ),
        }

    source_id_digest = hashlib.sha256(
        "\n".join(sorted(evidence_ids)).encode("utf-8")
    ).hexdigest()
    return {
        "profile": profile,
        "auth": {
            "endpoint_role": role,
            "authenticated_role": auth.get("role"),
            "identity_sha256": auth.get("identity_sha256"),
            "cross_role_violation": False,
        },
        "llm": {
            "provider": llm_result.get("provider"),
            "model": llm_result.get("model"),
            "fallback_used": False,
            "parse_success": llm_meta.get("parse_success"),
            "schema_valid": llm_meta.get("schema_valid"),
        },
        "orchestration": {
            "response_path_verified": True,
            "packet_path_verified": True,
            "knowledge_backend": knowledge_backend,
            "graph_backend": "graphiti" if role in {"doctor", "student"} else "not_role_visible",
            "memory_backend": "mem0",
            "adapter_fallback_used": False,
            "context_strategy": backend_summary.get("context_strategy"),
            "context_mode": backend_summary.get("context_mode"),
            "recent_turn_count": backend_summary.get("recent_turn_count"),
            "available_recent_turn_count": backend_summary.get("available_recent_turn_count"),
            "tool_call_count": len(tool_calls),
            "degraded_tools": degraded_tools,
        },
        "evidence": {
            "packet_id": packet_id,
            "trace_id": trace_id,
            "visibility": expected_visibility,
            "evidence_count": len(retrieved),
            "source_ids_sha256": source_id_digest,
            "case_material_evidence_verified": case_source_id is not None,
            "case_material_source_id": case_source_id,
            "knowledge_evidence_count": len(knowledge_retrieved),
        },
        "case_reasoning_quality_gate": quality_gate,
        "memory": {
            "policy": "isolated_neutral_seed_per_task_attempt",
            "scope_role": role,
            "scope_identity_sha256": auth.get("identity_sha256"),
            "context_mode": conversation.get("context_mode"),
            "recent_turn_count": conversation.get("turn_count"),
            "available_recent_turn_count": conversation.get("available_turn_count"),
        },
        "setup": setup,
    }


def _validate_live_product_response(
    task: dict[str, Any],
    raw: dict[str, Any],
    data: dict[str, Any],
    packet: dict[str, Any],
    llm_result: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the frozen LightRAG product contract."""

    return _validate_product_response(
        task,
        raw,
        data,
        packet,
        llm_result,
        profile="live_product",
        knowledge_backend="lightrag",
    )


def _validate_local_product_response(
    task: dict[str, Any],
    raw: dict[str, Any],
    data: dict[str, Any],
    packet: dict[str, Any],
    llm_result: dict[str, Any],
) -> dict[str, Any]:
    """Validate the time-bounded local structured-retrieval product contract."""

    return _validate_product_response(
        task,
        raw,
        data,
        packet,
        llm_result,
        profile="local_product",
        knowledge_backend="local_knowledge",
    )


def _validate_tasks(tasks: list[dict[str, Any]]) -> None:
    required = {
        "task_id",
        "family_token",
        "role",
        "variant",
        "arm",
        "source_text_private",
        "paired_case_material_sha256",
        "system_prompt_private",
        "user_message_private",
    }
    seen: set[str] = set()
    for row in tasks:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"task row missing fields: {missing}")
        task_id = str(row["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        if row["role"] not in ROLES:
            raise ValueError(f"unsupported role: {row['role']}")
        shared_digest = row.get("shared_payload_sha256")
        if shared_digest is not None:
            observed = hashlib.sha256(
                str(row["user_message_private"]).encode("utf-8")
            ).hexdigest()
            if shared_digest != observed:
                raise ValueError(f"shared payload hash mismatch for task {task_id}")


def _validate_existing_response(
    row: dict[str, Any], task: dict[str, Any], expected_task_hash: str
) -> None:
    """Refuse resume data that is not bound to the exact frozen task row."""

    if (
        row.get("output_selection_policy")
        != "first_user_visible_output_frozen_content_neutral"
        or row.get("content_triggered_regeneration") is not False
    ):
        raise ValueError(
            f"resume row predates the selection-neutral protocol for task {task['task_id']}; "
            "use a new run root"
        )

    for field in ("family_token", "role", "variant", "arm"):
        if row.get(field) != task.get(field):
            raise ValueError(
                f"resume response/task mismatch for {task['task_id']} field {field}"
            )
    if row.get("task_payload_sha256") != expected_task_hash:
        raise ValueError(
            f"resume response is not bound to the exact frozen task {task['task_id']}"
        )
    if row.get("status") not in {"success", "failed"}:
        raise ValueError(f"invalid resume status for task {task['task_id']}")
    if row.get("status") == "success" and not str(
        row.get("display_response_private") or ""
    ).strip():
        raise ValueError(f"successful resume row has empty answer for task {task['task_id']}")
    if row.get("post_output_audit_failure") is True:
        frozen_raw = row.get("raw_payload_private")
        if (
            row.get("status") != "failed"
            or not isinstance(frozen_raw, dict)
            or not str(row.get("display_response_private") or "").strip()
            or row.get("selected_response_sha256")
            != canonical_json_sha256(frozen_raw)
        ):
            raise ValueError(
                f"post-output audit failure is not bound to its frozen response for task {task['task_id']}"
            )


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Checkpoint without leaving a half-written response file on interruption."""

    temporary = path.with_name(path.name + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _plan(tasks: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    chat_template_kwargs = (
        _parse_chat_template_kwargs(args) if args.mode == "openai_compatible" else {}
    )
    return {
        "status": "PLAN_ONLY" if not args.execute else "EXECUTION_REQUESTED",
        "mode": args.mode,
        "task_count": len(tasks),
        "counts_by_role": dict(Counter(str(row["role"]) for row in tasks)),
        "counts_by_condition": dict(Counter(str(row["variant"]) for row in tasks)),
        "counts_by_arm": dict(Counter(str(row["arm"]) for row in tasks)),
        "will_write_chat_history_and_trace": args.mode == "system_api",
        "system_api_profile": (
            str(getattr(args, "system_api_profile", "locked_evidence"))
            if args.mode == "system_api"
            else "not_applicable"
        ),
        "chat_template_kwargs": chat_template_kwargs,
        "privacy_gate": (
            "system_api execution requires a dedicated PRIVATE database; locked_evidence disables "
            "external memory while product profiles use isolated retrieval/Graphiti/Mem0 state "
            "with telemetry disabled; this script never deletes or cleans a database"
        ),
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    tasks = read_jsonl(input_path)
    _validate_tasks(tasks)
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")
    plan = _plan(tasks, args)
    if not args.execute:
        return plan

    if args.isolated_eval_ack != ISOLATED_ACK:
        raise ValueError(f"--isolated-eval-ack must equal {ISOLATED_ACK}")
    if not _is_local_url(args.base_url) and args.remote_data_transfer_ack != REMOTE_ACK:
        raise ValueError(
            "non-local endpoint refused; provide the exact remote-processing acknowledgement only "
            "after data-provider and ethics authorization"
        )
    system_profile = str(getattr(args, "system_api_profile", "locked_evidence"))
    if args.mode == "system_api" and system_profile not in SYSTEM_API_PROFILES:
        raise ValueError(f"unsupported --system-api-profile: {system_profile}")
    if (
        args.mode == "system_api"
        and system_profile == "locked_evidence"
        and args.external_memory_ack != MEMORY_ACK
    ):
        raise ValueError(
            f"locked_evidence writes SQL history/trace; --external-memory-ack must equal {MEMORY_ACK} "
            "after external memory and telemetry are disabled"
        )
    if (
        args.mode == "system_api"
        and system_profile in PRODUCT_SYSTEM_API_PROFILES
        and args.full_product_stack_ack != FULL_PRODUCT_ACK
    ):
        raise ValueError(
            f"{system_profile} requires --full-product-stack-ack {FULL_PRODUCT_ACK} after "
            "confirming that the declared retrieval backend, Graphiti, Mem0 and their local "
            "model services use an isolated evaluation scope"
        )
    if args.mode == "system_api" and not args.provision_isolated_users:
        raise ValueError(
            "system_api execution requires --provision-isolated-users so every task attempt has "
            "a distinct account/thread; shared role accounts would contaminate the controlled study"
        )
    if args.mode == "system_api" and (
        not args.expected_backend_provider or not args.expected_backend_model
    ):
        raise ValueError(
            "system_api execution requires --expected-backend-provider and "
            "--expected-backend-model"
        )
    if args.mode == "system_api" and any(
        row.get("arm") != "role_aware" for row in tasks
    ):
        raise ValueError(
            "the product routes always apply real role policies; prompt-ablation tasks may only "
            "run in openai_compatible mode and must not be labeled as a product baseline"
        )
    if (
        args.mode == "system_api"
        and system_profile in PRODUCT_SYSTEM_API_PROFILES
        and any(row.get("role") != "doctor" for row in tasks)
    ):
        raise ValueError(
            f"the current {system_profile} qualification contract is registered for the doctor-role "
            "TCM-BEST4SDT track; do not silently reuse it for patient/student safety paths"
        )
    if args.mode == "openai_compatible" and not args.model:
        raise ValueError("--model is required in openai_compatible mode")
    if args.mode == "openai_compatible":
        api_key_env = str(getattr(args, "api_key_env", "") or "").strip()
        api_key_present = bool(os.environ.get(api_key_env, "").strip()) if api_key_env else False
        if not api_key_present and not _is_local_url(args.base_url):
            raise ValueError(
                "an unauthenticated OpenAI-compatible endpoint is allowed only on loopback; "
                "set --api-key-env for a non-local endpoint"
            )

    system_preflight = _system_preflight(args) if args.mode == "system_api" else None

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        if not args.resume:
            raise FileExistsError("refusing to overwrite responses; use --resume or a new path")
        task_index = {str(row["task_id"]): row for row in tasks}
        for row in read_jsonl(output_path):
            task_id = str(row.get("task_id") or "")
            if not task_id or task_id in existing:
                raise ValueError("resume file has an empty or duplicate task_id")
            if task_id not in task_index:
                raise ValueError(f"resume file contains an unknown task_id: {task_id}")
            _validate_existing_response(
                row, task_index[task_id], canonical_json_sha256(task_index[task_id])
            )
            if args.mode == "system_api" and row.get("system_api_profile") != system_profile:
                raise ValueError(
                    f"resume response profile mismatch for {task_id}: "
                    f"expected={system_profile}, actual={row.get('system_api_profile')}"
                )
            existing[task_id] = row

    results: list[dict[str, Any]] = []
    requester = _system_request if args.mode == "system_api" else _openai_request
    newly_processed = 0
    for index, task in enumerate(tasks, start=1):
        prior = existing.get(str(task["task_id"]))
        if prior and (
            prior.get("status") == "success"
            or prior.get("post_output_audit_failure") is True
            or not args.retry_failures
        ):
            results.append(prior)
            continue
        started = perf_counter()
        raw: dict[str, Any] | None = None
        audit_private: dict[str, Any] | None = None
        error = ""
        text = ""
        selected_attempt: int | None = None
        selected_response_sha256: str | None = None
        post_output_audit_failure = False
        audit_failure_stage: str | None = None
        prior_operational_failures = (
            prior.get("pre_output_operational_failures_private") if prior else []
        )
        pre_output_operational_failures: list[str] = [
            str(item)[:1200]
            for item in (
                prior_operational_failures
                if isinstance(prior_operational_failures, list)
                else []
            )
            if str(item).strip()
        ]
        prior_attempts = int(prior.get("attempt_count") or 0) if prior else 0
        attempts = prior_attempts
        for attempt in range(args.retries + 1):
            attempt_identity = prior_attempts + attempt
            attempts = attempt_identity + 1
            try:
                candidate_raw = requester(task, args, attempt_identity)
                if not isinstance(candidate_raw, dict):
                    raise ValueError("inference endpoint returned no JSON object")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, RuntimeError) as exc:
                detail = str(getattr(exc, "care_safe_response_detail", "") or "")
                message = str(exc)
                if detail:
                    message = f"{message}; response_detail={detail}"
                error = f"{exc.__class__.__name__}:{message[:1200]}"
                pre_output_operational_failures.append(error)
                if attempt < args.retries:
                    time.sleep(args.retry_backoff_seconds * (2**attempt))
                continue

            if args.mode == "system_api":
                try:
                    _require_successful_system_generation(candidate_raw)
                except RuntimeError as exc:
                    detail = str(
                        getattr(exc, "care_safe_response_detail", "") or ""
                    )
                    message = str(exc)
                    if detail:
                        message = f"{message}; response_detail={detail}"
                    error = f"{exc.__class__.__name__}:{message[:1200]}"
                    pre_output_operational_failures.append(error)
                    if attempt < args.retries:
                        time.sleep(args.retry_backoff_seconds * (2**attempt))
                    continue

            candidate_text = extract_display_text(candidate_raw)
            if not candidate_text:
                error = "EMPTY_DISPLAY_RESPONSE:API payload contained no user-visible answer"
                if attempt < args.retries:
                    time.sleep(args.retry_backoff_seconds * (2**attempt))
                continue

            # Freeze the first user-visible output before any answer-content or
            # operational audit.  A later audit failure is recorded against
            # this exact payload and may never trigger a second generation.
            raw = candidate_raw
            text = candidate_text
            selected_attempt = attempts
            selected_response_sha256 = canonical_json_sha256(raw)
            if args.mode == "system_api":
                try:
                    audit_private = _validate_system_response(
                        task, raw, system_preflight or {}
                    )
                    error = ""
                except Exception as exc:  # preserve the already-visible output
                    detail = str(
                        getattr(exc, "care_safe_response_detail", "") or ""
                    )
                    message = str(exc)
                    if detail:
                        message = f"{message}; response_detail={detail}"
                    error = f"{exc.__class__.__name__}:{message[:1200]}"
                    audit_private = None
                    post_output_audit_failure = True
                    audit_failure_stage = "post_output_operational_audit"
            else:
                error = ""
            break
        status = (
            "success"
            if raw is not None and bool(text) and not post_output_audit_failure
            else "failed"
        )
        if status == "failed" and not error:
            error = "EMPTY_DISPLAY_RESPONSE:API payload contained no user-visible answer"
        results.append(
            {
                "schema_version": "care_opd.competition.three_role_response.v2",
                "task_id": task["task_id"],
                "task_payload_sha256": canonical_json_sha256(task),
                "family_token": task["family_token"],
                "role": task["role"],
                "variant": task["variant"],
                "arm": task["arm"],
                "modality": task.get("modality", "unknown"),
                "status": status,
                "attempt_count": attempts,
                "selected_attempt": selected_attempt,
                "selected_response_sha256": selected_response_sha256,
                "post_output_audit_failure": post_output_audit_failure,
                "audit_failure_stage": audit_failure_stage,
                "pre_output_operational_failures_private": (
                    pre_output_operational_failures
                ),
                "latency_seconds": perf_counter() - started,
                "display_response_private": text,
                "raw_payload_private": raw,
                "error_private": error,
                "execution_mode": args.mode,
                "system_api_profile": (
                    system_profile if args.mode == "system_api" else "not_applicable"
                ),
                "audit_private": audit_private,
                "output_selection_policy": (
                    "first_user_visible_output_frozen_content_neutral"
                ),
                "content_triggered_regeneration": False,
                "model": args.model if args.mode == "openai_compatible" else "backend_config_frozen",
                "generation_parameters": (
                    {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "seed": args.generation_seed,
                        "chat_template_kwargs": _parse_chat_template_kwargs(args),
                    }
                    if args.mode == "openai_compatible"
                    else {"source": "isolated_backend_environment", "overridden_by_script": False}
                ),
                "response_provider": (
                    ((raw or {}).get("data") or {}).get("llm_result", {}).get("provider")
                    if args.mode == "system_api"
                    else args.mode
                ),
                "response_model": (
                    ((raw or {}).get("data") or {}).get("llm_result", {}).get("model")
                    if args.mode == "system_api"
                    else args.model
                ),
            }
        )
        newly_processed += 1
        if args.checkpoint_every and newly_processed % args.checkpoint_every == 0:
            checkpoint_rows = dict(existing)
            checkpoint_rows.update(
                {str(row["task_id"]): row for row in results}
            )
            _atomic_write_jsonl(
                output_path,
                sorted(checkpoint_rows.values(), key=lambda row: str(row["task_id"])),
            )
        if status == "failed" and getattr(args, "abort_on_first_failure", False):
            checkpoint_rows = dict(existing)
            checkpoint_rows.update({str(row["task_id"]): row for row in results})
            _atomic_write_jsonl(
                output_path,
                sorted(checkpoint_rows.values(), key=lambda row: str(row["task_id"])),
            )
            raise RuntimeError(
                "aborted after the first failed row; inspect the persisted canary "
                f"record for task {task['task_id']} before continuing"
            )
        if args.progress_every and index % args.progress_every == 0:
            print(json.dumps({"completed": index, "total": len(tasks)}, ensure_ascii=False))

    ordered = sorted(results, key=lambda row: str(row["task_id"]))
    _atomic_write_jsonl(output_path, ordered)
    manifest_path = output_path.with_name("three_role_execution.private.json")
    manifest = {
        "status": "COMPLETE" if all(row["status"] == "success" for row in ordered) else "INCOMPLETE",
        "mode": args.mode,
        "system_api_profile": (
            system_profile if args.mode == "system_api" else "not_applicable"
        ),
        "model": args.model if args.mode == "openai_compatible" else "backend_config_frozen",
        "openai_endpoint_authentication": (
            (
                "environment_bearer"
                if (
                    str(getattr(args, "api_key_env", "") or "").strip()
                    and os.environ.get(
                        str(getattr(args, "api_key_env", "") or "").strip(), ""
                    ).strip()
                )
                else "none_loopback"
            )
            if args.mode == "openai_compatible"
            else "not_applicable"
        ),
        "chat_template_kwargs": (
            _parse_chat_template_kwargs(args)
            if args.mode == "openai_compatible"
            else {}
        ),
        "task_count": len(tasks),
        "success_count": sum(row["status"] == "success" for row in ordered),
        "failure_count": sum(row["status"] != "success" for row in ordered),
        "post_output_audit_failure_count": sum(
            row.get("post_output_audit_failure") is True for row in ordered
        ),
        "pre_output_operational_failure_count": sum(
            len(row.get("pre_output_operational_failures_private") or [])
            for row in ordered
        ),
        "rows_with_pre_output_operational_failure_count": sum(
            bool(row.get("pre_output_operational_failures_private"))
            for row in ordered
        ),
        "input_sha256": sha256_file(input_path),
        "response_sha256": sha256_file(output_path),
        "task_payload_binding_verified": True,
        "output_selection_policy": (
            "first_user_visible_output_frozen_content_neutral"
        ),
        "content_triggered_regeneration": False,
        "checkpoint_every": args.checkpoint_every,
        "system_api_persistence_note": (
            (
                (
                    "live_product requests used normal role endpoints, a per-task isolated neutral "
                    "longitudinal seed, local LightRAG/Graphiti/Mem0 state, SQL history and trace"
                    if system_profile == "live_product"
                    else "local_product requests used normal role endpoints, a per-task isolated "
                    "neutral longitudinal seed, frozen local structured knowledge retrieval, "
                    "Graphiti/Mem0 state, SQL history and trace"
                )
                if system_profile in PRODUCT_SYSTEM_API_PROFILES
                else "locked_evidence requests used role endpoints with controlled evidence and "
                "persisted SQL history/trace in the isolated PRIVATE environment"
            )
            if args.mode == "system_api"
            else "not_applicable"
        ),
        "database_cleanup_performed": False,
        "conversation_isolation_strategy": (
            (
                "unique_opaque_user_per_task_attempt_with_neutral_longitudinal_seed"
                if system_profile in PRODUCT_SYSTEM_API_PROFILES
                else "unique_opaque_user_per_task_attempt"
            )
            if args.mode == "system_api"
            else "stateless_chat_completion_request"
        ),
        "system_preflight": system_preflight,
        "response_provider_counts": dict(
            Counter(str(row.get("response_provider") or "missing") for row in ordered)
        ),
        "response_model_counts": dict(
            Counter(str(row.get("response_model") or "missing") for row in ordered)
        ),
    }
    write_json(manifest_path, manifest)
    return {**plan, **manifest, "output": str(output_path), "manifest": str(manifest_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("three_role_responses.private.jsonl"))
    parser.add_argument("--mode", choices=("system_api", "openai_compatible"), default="system_api")
    parser.add_argument(
        "--system-api-profile",
        choices=SYSTEM_API_PROFILES,
        default="locked_evidence",
        help=(
            "locked controlled evidence, LightRAG product orchestration, or the "
            "time-bounded local structured-retrieval product profile"
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--chat-template-kwargs-json",
        default="",
        help=(
            "optional JSON object forwarded only as chat_template_kwargs to an "
            "OpenAI-compatible endpoint, e.g. '{\"enable_thinking\": false}'"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=12)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="atomically persist PRIVATE progress every N newly processed tasks; 0 disables",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument(
        "--abort-on-first-failure",
        action="store_true",
        help="persist the failed canary row and stop before spending inference on later tasks",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--isolated-eval-ack", default="")
    parser.add_argument("--external-memory-ack", default="")
    parser.add_argument("--full-product-stack-ack", default="")
    parser.add_argument("--remote-data-transfer-ack", default="")
    parser.add_argument("--provision-isolated-users", action="store_true")
    parser.add_argument("--provision-secret-env", default=PROVISION_SECRET_ENV)
    parser.add_argument("--private-eval-key-env", default=PRIVATE_EVAL_KEY_ENV)
    parser.add_argument("--expected-backend-provider", default="")
    parser.add_argument("--expected-backend-model", default="")
    parser.add_argument("--expected-backend-temperature", type=float, default=0.0)
    parser.add_argument("--expected-backend-top-p", type=float, default=1.0)
    parser.add_argument("--expected-backend-seed", type=int, default=42)
    parser.add_argument("--expected-backend-max-output-tokens", type=int, default=1200)
    parser.add_argument("--expected-backend-max-input-tokens", type=int, default=6000)
    return parser


def main() -> None:
    print(json.dumps(execute(_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
