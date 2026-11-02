"""Authorize request-bound teaching-case evidence on the full product stack."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
from typing import Any

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.schemas.agent import DoctorCaseReasoningRequest


ISOLATED_DATABASE_ACK = "I_CONFIRM_PRIVATE_ISOLATED_DATABASE"


def authorize_public_teaching_case(
    *,
    payload: DoctorCaseReasoningRequest,
    message: str,
    supplied_key: str | None,
    client_host: str | None,
) -> dict[str, Any]:
    settings = get_settings()
    failures: list[str] = []
    if not settings.case_reasoning_evidence_control_enabled:
        failures.append("case_reasoning_evidence_control_disabled")
    if settings.app_env.lower() != "competition_eval":
        failures.append("APP_ENV_must_be_competition_eval")
    database_url = settings.database_url.strip().lower().replace("\\", "/")
    if not database_url.startswith("sqlite:///") or not database_url.endswith(
        ".private.db"
    ):
        failures.append("DATABASE_URL_must_be_an_explicit_dot_private_sqlite_db")
    if settings.case_reasoning_isolated_database_ack != ISOLATED_DATABASE_ACK:
        failures.append("isolated_database_ack_missing")
    if len(settings.case_reasoning_evidence_control_secret) < 32:
        failures.append("case_reasoning_secret_too_short")
    if settings.graph_backend != "graphiti":
        failures.append("GRAPH_BACKEND_must_be_graphiti")
    if settings.memory_backend != "mem0":
        failures.append("MEMORY_BACKEND_must_be_mem0")
    if settings.knowledge_backend not in {"local_knowledge", "lightrag"}:
        failures.append("KNOWLEDGE_BACKEND_must_be_full_product")
    if settings.allow_adapter_fallback:
        failures.append("ALLOW_ADAPTER_FALLBACK_must_be_false")
    if settings.llm_allow_fallback:
        failures.append("LLM_ALLOW_FALLBACK_must_be_false")
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
    if failures:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"case_reasoning_runtime_gate_failed": failures},
        )
    if not _is_loopback(client_host):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="case reasoning evidence control is loopback-only",
        )
    if not supplied_key or not hmac.compare_digest(
        supplied_key, settings.case_reasoning_evidence_control_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid case reasoning evidence key",
        )

    material = payload.case_material
    observed_sha256 = hashlib.sha256(material.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(observed_sha256, payload.case_material_sha256):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="case_material_sha256 mismatch",
        )
    if material not in message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="message is not bound to the complete case_material",
        )
    context = payload.model_dump(mode="json")
    context.pop("message", None)
    context["authorization"] = "loopback_secret_and_full_product_runtime_gate"
    return context


def _is_loopback(value: str | None) -> bool:
    if not value:
        return False
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return value.lower() == "localhost"
