"""Fail-closed evidence qualification for the PRIVATE three-role study.

This module deliberately has no effect on ordinary product requests.  The
control plane is accepted only when all of the following hold at request time:

* the feature gate is explicitly enabled;
* the backend is marked as ``competition_eval`` and uses a non-default DB;
* Graphiti, Mem0, conversation carry-over and model fallback are disabled;
* the caller is loopback and proves possession of a dedicated secret; and
* the supplied case material matches both its digest and the shared message.

The secret is never returned or written to an audit record.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
from typing import Any

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.schemas.agent import PrivateEvalEvidenceControlRequest


ISOLATED_DATABASE_ACK = "I_CONFIRM_PRIVATE_ISOLATED_DATABASE"
PRIVATE_EVAL_PROTOCOL = "three_role_paired_fact_v1"


def authorize_private_evidence_control(
    *,
    control: PrivateEvalEvidenceControlRequest | None,
    message: str,
    supplied_key: str | None,
    client_host: str | None,
) -> dict[str, Any] | None:
    """Validate and minimize the caller control before it enters AgentState."""

    if control is None:
        if supplied_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="private evaluation key supplied without an evidence control payload",
            )
        return None

    settings = get_settings()
    if not settings.private_eval_evidence_control_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="private evaluation evidence control is disabled",
        )
    _assert_private_runtime(settings)
    if not _is_loopback(client_host):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="private evaluation evidence control is loopback-only",
        )
    if not supplied_key or not hmac.compare_digest(
        supplied_key, settings.private_eval_evidence_control_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid private evaluation key",
        )

    material = control.case_material.strip()
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(digest, control.case_material_sha256):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="private evaluation case material digest mismatch",
        )
    if material not in message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="private evaluation case material is not present verbatim in the shared message",
        )

    expected_count = 1 if control.condition == "clean" else 0
    return {
        "protocol": PRIVATE_EVAL_PROTOCOL,
        "condition": control.condition,
        "task_token": control.task_token,
        "case_material_sha256": control.case_material_sha256,
        "case_material": material,
        "expected_retrieved_evidence_count": expected_count,
        "authorization": "loopback_secret_and_runtime_gate",
    }


def _assert_private_runtime(settings: Any) -> None:
    failures: list[str] = []
    if settings.app_env.lower() != "competition_eval":
        failures.append("APP_ENV_must_be_competition_eval")
    database_url = settings.database_url.strip().lower().replace("\\", "/")
    if not database_url.startswith("sqlite:///") or not database_url.endswith(
        ".private.db"
    ):
        failures.append("DATABASE_URL_must_be_an_explicit_dot_private_sqlite_db")
    if settings.private_eval_isolated_database_ack != ISOLATED_DATABASE_ACK:
        failures.append("isolated_database_ack_missing")
    if len(settings.private_eval_evidence_control_secret) < 32:
        failures.append("private_eval_secret_too_short")
    if settings.graph_backend != "persistent_graph":
        failures.append("GRAPH_BACKEND_must_be_persistent_graph")
    if settings.memory_backend != "persistent_memory":
        failures.append("MEMORY_BACKEND_must_be_persistent_memory")
    if settings.graphiti_ingest_chat_history:
        failures.append("GRAPHITI_INGEST_CHAT_HISTORY_must_be_false")
    if settings.mem0_ingest_chat_history:
        failures.append("MEM0_INGEST_CHAT_HISTORY_must_be_false")
    if settings.conversation_context_enabled:
        failures.append("CONVERSATION_CONTEXT_ENABLED_must_be_false")
    if settings.lightrag_enabled:
        failures.append("LIGHTRAG_ENABLED_must_be_false")
    if settings.qa_knowledge_enabled:
        failures.append("QA_KNOWLEDGE_ENABLED_must_be_false")
    if settings.allow_adapter_fallback:
        failures.append("ALLOW_ADAPTER_FALLBACK_must_be_false")
    if settings.llm_allow_fallback:
        failures.append("LLM_ALLOW_FALLBACK_must_be_false")
    if settings.llm_provider not in {"ollama", "openai_compatible"}:
        failures.append("LLM_PROVIDER_must_be_a_real_provider")
    for variable in (
        "GRAPHITI_TELEMETRY_ENABLED",
        "MEM0_TELEMETRY",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
    ):
        if os.environ.get(variable, "").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            failures.append(f"{variable}_must_be_explicitly_false")
    if os.environ.get("DO_NOT_TRACK", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        failures.append("DO_NOT_TRACK_must_be_explicitly_true")
    if os.environ.get("LANGSMITH_API_KEY", "").strip():
        failures.append("LANGSMITH_API_KEY_must_be_unset")
    if failures:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "private evaluation runtime gate failed", "checks": failures},
        )


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"
