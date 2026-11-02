#!/usr/bin/env python3
"""Run one frozen three-role task package through exactly four Agent systems.

The default is PLAN_ONLY and performs no network request.  Real execution is
deliberately guarded because the frozen package may contain de-identified real
medical records.  API credentials are read only from environment variables;
inline secrets in the configuration are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

try:
    from . import execute_three_role_eval as engine
    from .three_role_eval_common import sha256_file, write_json
except ImportError:  # pragma: no cover - direct script execution
    import execute_three_role_eval as engine  # type: ignore
    from three_role_eval_common import sha256_file, write_json  # type: ignore


SCHEMA_VERSION = "care_opd.competition.four_agent_execution.v1"
PROJECT_KIND = "project_agent"
EXTERNAL_KIND = "external_api_agent"
ALLOWED_MODES = {"system_api", "openai_compatible"}
SYSTEM_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
ENV_TEMPLATE_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]{1,127})\}")
INLINE_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "access_token",
    "secret",
    "headers",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def _reject_inline_secrets(value: Any, trail: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in INLINE_SECRET_KEYS:
                raise ValueError(
                    f"inline secret-bearing field is forbidden: {trail}.{key_text}; "
                    "store credentials in an environment variable and name it with api_key_env"
                )
            _reject_inline_secrets(child, f"{trail}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_inline_secrets(child, f"{trail}[{index}]")


def validate_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    _reject_inline_secrets(config)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config schema_version must equal {SCHEMA_VERSION}")
    systems = config.get("systems")
    if not isinstance(systems, list) or len(systems) != 4:
        raise ValueError("configuration must contain exactly four systems")
    if not all(isinstance(row, dict) for row in systems):
        raise ValueError("each system configuration must be an object")

    typed_systems = [dict(row) for row in systems]
    ids = [str(row.get("system_id") or "") for row in typed_systems]
    if len(set(ids)) != 4 or any(not SYSTEM_ID_RE.fullmatch(value) for value in ids):
        raise ValueError("four unique lower_snake_case system_id values are required")
    primary_id = str(config.get("primary_system_id") or "")
    kinds = [str(row.get("kind") or "") for row in typed_systems]
    if kinds.count(PROJECT_KIND) != 1 or kinds.count(EXTERNAL_KIND) != 3:
        raise ValueError("require one project_agent and exactly three external_api_agent systems")
    project = next(row for row in typed_systems if row.get("kind") == PROJECT_KIND)
    if project.get("system_id") != primary_id:
        raise ValueError("primary_system_id must identify the sole project_agent")

    providers: set[str] = set()
    for row in typed_systems:
        mode = str(row.get("mode") or "")
        if mode not in ALLOWED_MODES:
            raise ValueError(f"unsupported mode for {row['system_id']}: {mode}")
        base_url = str(row.get("base_url") or row.get("base_url_template") or "").strip()
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"valid http(s) base_url or base_url_template required for {row['system_id']}"
            )
        if row["kind"] == PROJECT_KIND:
            if mode != "system_api":
                raise ValueError("project_agent must use system_api mode")
            for field in (
                "expected_backend_provider_env",
                "expected_backend_model_env",
                "provision_secret_env",
                "private_eval_key_env",
            ):
                if not str(row.get(field) or "").strip():
                    raise ValueError(f"{row['system_id']} is missing {field}")
        else:
            if mode != "openai_compatible":
                raise ValueError("external_api_agent must use openai_compatible mode")
            for field in ("provider_id", "model"):
                if not str(row.get(field) or "").strip():
                    raise ValueError(f"{row['system_id']} is missing {field}")
            api_key_required = row.get("api_key_required")
            if api_key_required is not None and not isinstance(api_key_required, bool):
                raise ValueError(
                    f"{row['system_id']} api_key_required must be true or false"
                )
            if api_key_required is True and not str(row.get("api_key_env") or "").strip():
                raise ValueError(
                    f"{row['system_id']} requires api_key_env when api_key_required=true"
                )
            chat_template_kwargs = row.get("chat_template_kwargs", {})
            if not isinstance(chat_template_kwargs, dict):
                raise ValueError(
                    f"{row['system_id']} chat_template_kwargs must be an object"
                )
            provider_id = str(row["provider_id"])
            if provider_id in providers:
                raise ValueError(f"duplicate external provider_id: {provider_id}")
            providers.add(provider_id)
    return typed_systems


def _is_local_url(value: str) -> bool:
    host = (urllib.parse.urlparse(value).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _generation(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("generation") or {}
    if not isinstance(raw, dict):
        raise ValueError("generation must be an object")
    values = {
        "temperature": float(raw.get("temperature", 0.0)),
        "top_p": float(raw.get("top_p", 1.0)),
        "max_tokens": int(raw.get("max_tokens", 1200)),
        "seed": int(raw.get("seed", 42)),
        "timeout_seconds": float(raw.get("timeout_seconds", 120.0)),
        "retries": int(raw.get("retries", 2)),
        "retry_backoff_seconds": float(raw.get("retry_backoff_seconds", 1.0)),
        "checkpoint_every": int(raw.get("checkpoint_every", 1)),
        "progress_every": int(raw.get("progress_every", 12)),
    }
    if values["max_tokens"] <= 0 or values["timeout_seconds"] <= 0:
        raise ValueError("max_tokens and timeout_seconds must be positive")
    if values["retries"] < 0 or values["checkpoint_every"] < 0:
        raise ValueError("retries and checkpoint_every cannot be negative")
    return values


def _backend_value(row: dict[str, Any], field: str) -> str:
    env_name = str(row.get(field + "_env") or "").strip()
    return os.environ.get(env_name, "").strip() if env_name else str(row.get(field) or "").strip()


def _base_url(row: dict[str, Any]) -> str:
    value = str(row.get("base_url") or row.get("base_url_template") or "").strip()
    return ENV_TEMPLATE_RE.sub(
        lambda match: os.environ.get(match.group(1), match.group(0)).strip(), value
    )


def _engine_args(
    *,
    input_path: Path,
    output_path: Path,
    row: dict[str, Any],
    generation: dict[str, Any],
    cli: argparse.Namespace,
) -> argparse.Namespace:
    is_project = row["kind"] == PROJECT_KIND
    backend_generation = row.get("expected_backend_generation") or {}
    if not isinstance(backend_generation, dict):
        raise ValueError("expected_backend_generation must be an object")
    return argparse.Namespace(
        input=input_path,
        output=output_path,
        mode=row["mode"],
        base_url=_base_url(row),
        model="" if is_project else str(row["model"]),
        api_key_env=(
            ""
            if is_project
            else str(row.get("api_key_env") or "").strip()
        ),
        chat_template_kwargs_json=(
            {}
            if is_project
            else dict(row.get("chat_template_kwargs") or {})
        ),
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        max_tokens=generation["max_tokens"],
        generation_seed=generation["seed"],
        timeout_seconds=generation["timeout_seconds"],
        retries=generation["retries"],
        retry_backoff_seconds=generation["retry_backoff_seconds"],
        progress_every=generation["progress_every"],
        checkpoint_every=generation["checkpoint_every"],
        resume=bool(cli.resume),
        retry_failures=bool(cli.retry_failures),
        execute=bool(cli.execute),
        isolated_eval_ack=str(cli.isolated_eval_ack),
        external_memory_ack=str(cli.external_memory_ack),
        remote_data_transfer_ack=str(cli.remote_data_transfer_ack),
        provision_isolated_users=is_project,
        provision_secret_env=str(
            row.get("provision_secret_env") or engine.PROVISION_SECRET_ENV
        ),
        private_eval_key_env=str(
            row.get("private_eval_key_env") or engine.PRIVATE_EVAL_KEY_ENV
        ),
        expected_backend_provider=(
            _backend_value(row, "expected_backend_provider") if is_project else ""
        ),
        expected_backend_model=(
            _backend_value(row, "expected_backend_model") if is_project else ""
        ),
        expected_backend_temperature=float(backend_generation.get("temperature", 0.0)),
        expected_backend_top_p=float(backend_generation.get("top_p", 1.0)),
        expected_backend_seed=int(backend_generation.get("seed", 42)),
        expected_backend_max_output_tokens=int(
            backend_generation.get("max_output_tokens", 1200)
        ),
        expected_backend_max_input_tokens=int(
            backend_generation.get("max_input_tokens", 6000)
        ),
    )


def _static_execution_preflight(
    systems: list[dict[str, Any]],
    namespaces: list[argparse.Namespace],
    cli: argparse.Namespace,
) -> None:
    """Fail before the first request when authorization or credentials are absent."""

    if cli.isolated_eval_ack != engine.ISOLATED_ACK:
        raise ValueError(f"--isolated-eval-ack must equal {engine.ISOLATED_ACK}")
    for row, ns in zip(systems, namespaces, strict=True):
        unresolved = ENV_TEMPLATE_RE.findall(ns.base_url)
        if unresolved:
            raise ValueError(
                f"missing endpoint environment variable(s) for {row['system_id']}: "
                + ", ".join(sorted(unresolved))
            )
        if ns.output.exists() and not cli.resume:
            raise FileExistsError(
                f"response output already exists for {row['system_id']}; use --resume: {ns.output}"
            )
        if not _is_local_url(ns.base_url) and cli.remote_data_transfer_ack != engine.REMOTE_ACK:
            raise ValueError(
                "remote APIs are blocked until --remote-data-transfer-ack equals "
                f"{engine.REMOTE_ACK} after data-provider/ethics authorization"
            )
        if row["kind"] == PROJECT_KIND:
            if cli.external_memory_ack != engine.MEMORY_ACK:
                raise ValueError(
                    f"--external-memory-ack must equal {engine.MEMORY_ACK} for the project Agent"
                )
            if not ns.expected_backend_provider or not ns.expected_backend_model:
                raise ValueError(
                    "project Agent backend identity is missing; set the configured provider/model "
                    "environment variables"
                )
            for env_name in (ns.provision_secret_env, ns.private_eval_key_env):
                if len(os.environ.get(env_name, "")) < 32:
                    raise ValueError(f"{env_name} must be set to at least 32 characters")
        else:
            # Loopback vLLM commonly runs without an API key.  Remote endpoints
            # remain authenticated by default; an explicit true setting also
            # lets operators require auth on a loopback reverse proxy.
            api_key_required = row.get("api_key_required")
            if api_key_required is None:
                api_key_required = not _is_local_url(ns.base_url)
            if not _is_local_url(ns.base_url):
                api_key_required = True
            if api_key_required and not ns.api_key_env:
                raise ValueError(
                    f"api_key_env is required for authenticated endpoint {row['system_id']}"
                )
            key = os.environ.get(ns.api_key_env, "").strip() if ns.api_key_env else ""
            if api_key_required and not key:
                raise ValueError(
                    f"missing API credential for {row['system_id']}: set {ns.api_key_env}"
                )


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config = _load_json(config_path)
    systems = validate_config(config)
    generation = _generation(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    namespaces = [
        _engine_args(
            input_path=input_path,
            output_path=output_dir / str(row["system_id"]) / "responses.private.jsonl",
            row=row,
            generation=generation,
            cli=args,
        )
        for row in systems
    ]

    # Validate the frozen task package without any endpoint access.
    plan_rows: list[dict[str, Any]] = []
    for row, ns in zip(systems, namespaces, strict=True):
        plan_ns = argparse.Namespace(**vars(ns))
        plan_ns.execute = False
        plan = engine.execute(plan_ns)
        plan_rows.append(
            {
                "system_id": row["system_id"],
                "kind": row["kind"],
                "provider_id": row.get("provider_id", "project_backend"),
                "mode": row["mode"],
                "model": row.get("model", "backend_identity_verified_at_runtime"),
                "authentication": (
                    "project_backend_session"
                    if row["kind"] == PROJECT_KIND
                    else (
                        "environment_bearer"
                        if ns.api_key_env
                        else "none_loopback"
                    )
                ),
                "api_key_env": (
                    ns.api_key_env if ns.api_key_env else "not_applicable"
                ),
                "chat_template_kwargs": getattr(
                    ns, "chat_template_kwargs_json", {}
                ),
                "planned_task_count": plan["task_count"],
                "output": str(ns.output),
            }
        )

    manifest_path = output_dir / "four_agent_execution.private.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PLAN_ONLY" if not args.execute else "EXECUTION_REQUESTED",
        "primary_system_id": config["primary_system_id"],
        "comparison_scope": (
            "end_to_end_agent_expert_preference; not CARE-OPD causal attribution and not "
            "clinical efficacy"
        ),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "config": str(config_path),
        "config_sha256": _canonical_hash(config),
        "generation": generation,
        "systems": plan_rows,
        "secrets_embedded_in_manifest": False,
        "remote_processing_authorization_required": any(
            not _is_local_url(ns.base_url) for ns in namespaces
        ),
    }
    if not args.execute:
        write_json(manifest_path, manifest)
        return {**manifest, "manifest": str(manifest_path)}

    _static_execution_preflight(systems, namespaces, args)
    results: list[dict[str, Any]] = []
    for row, ns in zip(systems, namespaces, strict=True):
        try:
            result = engine.execute(ns)
            results.append(
                {
                    "system_id": row["system_id"],
                    "status": result.get("status", "UNKNOWN"),
                    "success_count": result.get("success_count", 0),
                    "failure_count": result.get("failure_count", 0),
                    "response_sha256": result.get("response_sha256", ""),
                    "output": result.get("output", str(ns.output)),
                    "engine_manifest": result.get("manifest", ""),
                }
            )
        except Exception as exc:  # preserve other systems and make resume possible
            results.append(
                {
                    "system_id": row["system_id"],
                    "status": "EXECUTION_ERROR",
                    "error_private": f"{exc.__class__.__name__}:{str(exc)[:500]}",
                    "output": str(ns.output),
                }
            )

    manifest["systems"] = results
    manifest["status"] = (
        "COMPLETE"
        if all(row.get("status") == "COMPLETE" for row in results)
        else "INCOMPLETE"
    )
    write_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--isolated-eval-ack", default="")
    parser.add_argument("--external-memory-ack", default="")
    parser.add_argument("--remote-data-transfer-ack", default="")
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
