"""Prepare and run a non-mutating qzbc base-vs-trained model comparison.

qzbc already exposes a provider experiment endpoint over fixed doctor,
patient, student, and quiz scenarios.  This module generates deployment
artifacts that match qzbc's real environment contract, then invokes that
endpoint twice while changing only the Ollama model name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import stat
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


QZBC_EXPERIMENT_PATH = "/api/llm/experiments/run"
QZBC_LOGIN_PATH = "/api/auth/login"
QZBC_CASE_FILES = (
    "doctor_review_cases.json",
    "patient_education_cases.json",
    "student_analysis_cases.json",
    "quiz_generation_cases.json",
)
REPORT_METRICS = (
    "success",
    "failed",
    "fallback_count",
    "parse_success_rate",
    "schema_valid_rate",
    "safety_pass_rate",
    "avg_latency_ms",
)


@dataclass(frozen=True)
class CheckpointInfo:
    label: str
    path: Path
    model_type: str
    architectures: tuple[str, ...]
    signature: Mapping[str, Any]
    weight_files: tuple[str, ...]
    weight_bytes: int
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["architectures"] = list(self.architectures)
        payload["weight_files"] = list(self.weight_files)
        payload["signature"] = dict(self.signature)
        return payload


@dataclass(frozen=True)
class QzbcContract:
    root: Path
    provider_names: tuple[str, ...]
    experiment_path: str
    full_suite_cases: int
    case_files: tuple[str, ...]
    source_fingerprints: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["root"] = str(self.root)
        payload["provider_names"] = list(self.provider_names)
        payload["case_files"] = list(self.case_files)
        payload["source_fingerprints"] = dict(self.source_fingerprints)
        return payload


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _architecture_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, Mapping) else config


def inspect_checkpoint(path: Path, label: str) -> CheckpointInfo:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} checkpoint directory not found: {resolved}")
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{label} checkpoint lacks config.json: {resolved}")
    config = _json_object(config_path)
    architecture = _architecture_config(config)
    model_type = str(
        architecture.get("model_type") or config.get("model_type") or ""
    ).strip()
    if not model_type:
        raise ValueError(f"{label} config.json lacks model_type")

    weight_paths = sorted(
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in resolved.glob(pattern)
        if not path.name.startswith("adapter_")
    )
    if not weight_paths:
        raise ValueError(
            f"{label} is not a merged/full HF checkpoint: no model weights in {resolved}"
        )
    tokenizer_files = tuple(
        name
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
        )
        if (resolved / name).is_file()
    )
    if not tokenizer_files:
        raise ValueError(f"{label} checkpoint has no tokenizer assets: {resolved}")

    raw_architectures = config.get("architectures", ())
    architectures = (
        tuple(str(value) for value in raw_architectures)
        if isinstance(raw_architectures, list)
        else ()
    )
    signature_keys = (
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    )
    signature = {
        key: architecture.get(key, config.get(key))
        for key in signature_keys
        if architecture.get(key, config.get(key)) is not None
    }
    descriptors = [
        {
            "name": weight.relative_to(resolved).as_posix(),
            "size": weight.stat().st_size,
        }
        for weight in weight_paths
    ]
    fingerprint_payload = {
        "config_sha256": _sha256_file(config_path),
        "weights": descriptors,
        "tokenizer_files": list(tokenizer_files),
    }
    fingerprint = _sha256_bytes(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return CheckpointInfo(
        label=label,
        path=resolved,
        model_type=model_type,
        architectures=architectures,
        signature=signature,
        weight_files=tuple(item["name"] for item in descriptors),
        weight_bytes=sum(item["size"] for item in descriptors),
        fingerprint=fingerprint,
    )


def ensure_compatible(base: CheckpointInfo, trained: CheckpointInfo) -> None:
    mismatches = {
        key: (base.signature.get(key), trained.signature.get(key))
        for key in sorted(set(base.signature) | set(trained.signature))
        if base.signature.get(key) != trained.signature.get(key)
    }
    if mismatches:
        raise ValueError(
            "base and trained checkpoints are not architecture-compatible: "
            f"{mismatches}"
        )


def inspect_qzbc(root: Path) -> QzbcContract:
    resolved = root.expanduser().resolve()
    backend = resolved / "backend" / "app"
    paths = {
        "config": backend / "core" / "config.py",
        "ollama_provider": backend / "llm" / "providers" / "ollama_provider.py",
        "openai_provider": (
            backend / "llm" / "providers" / "openai_compatible_provider.py"
        ),
        "experiment_runner": backend / "llm" / "experiments" / "runner.py",
        "llm_routes": backend / "api" / "routes" / "llm.py",
        "env_example": resolved / "backend" / ".env.example",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"qzbc contract files missing: {missing}")

    config_text = paths["config"].read_text(encoding="utf-8")
    required_settings = (
        "llm_provider",
        "llm_allow_fallback",
        "llm_temperature",
        "llm_max_output_tokens",
        "ollama_base_url",
        "ollama_model",
        "ollama_num_ctx",
        "openai_compatible_base_url",
        "openai_compatible_model",
    )
    absent = [field for field in required_settings if field not in config_text]
    if absent:
        raise ValueError(f"unrecognized qzbc settings contract; absent: {absent}")

    route_text = paths["llm_routes"].read_text(encoding="utf-8")
    if '@router.post("/experiments/run")' not in route_text:
        raise ValueError("qzbc lacks POST /api/llm/experiments/run")
    runner_text = paths["experiment_runner"].read_text(encoding="utf-8")
    if "LLMExperimentRunner" not in runner_text:
        raise ValueError("qzbc LLMExperimentRunner contract not found")

    case_root = backend / "llm" / "experiments" / "cases"
    full_suite_cases = 0
    for filename in QZBC_CASE_FILES:
        payload = json.loads((case_root / filename).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"qzbc experiment case file is not a list: {filename}")
        full_suite_cases += len(payload)

    return QzbcContract(
        root=resolved,
        provider_names=("ollama", "openai_compatible"),
        experiment_path=QZBC_EXPERIMENT_PATH,
        full_suite_cases=full_suite_cases,
        case_files=QZBC_CASE_FILES,
        source_fingerprints={
            name: _sha256_file(path) for name, path in paths.items()
        },
    )


def _atomic_text(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".partial",
        delete=False,
        newline="\n",
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _shell(value: str | Path) -> str:
    return shlex.quote(str(value))


def _env_text(values: Mapping[str, str]) -> str:
    return "".join(f"{key}={_shell(value)}\n" for key, value in values.items())


def _ollama_model_name(value: str) -> str:
    name = value.strip()
    if not name or any(character.isspace() for character in name):
        raise ValueError(f"invalid Ollama model name: {value!r}")
    return name if ":" in name else f"{name}:latest"


def _modelfile(gguf: Path, *, num_ctx: int, temperature: float) -> str:
    return (
        f"FROM {gguf.as_posix()}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        f"PARAMETER temperature {temperature}\n"
        "PARAMETER seed 42\n"
    )


def _vllm_script(
    checkpoint: CheckpointInfo,
    *,
    model_name: str,
    port: int,
    num_ctx: int,
    trust_remote_code: bool,
) -> str:
    trust_flag = " \\\n  --trust-remote-code" if trust_remote_code else ""
    return (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n\n"
        'VLLM_BIN="${VLLM_BIN:-vllm}"\n'
        'VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"\n'
        'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"\n'
        f'"${{VLLM_BIN}}" serve {_shell(checkpoint.path)} \\\n'
        f"  --served-model-name {_shell(model_name)} \\\n"
        "  --host 127.0.0.1 \\\n"
        f"  --port {port} \\\n"
        '  --dtype "${VLLM_DTYPE}" \\\n'
        f"  --max-model-len {num_ctx} \\\n"
        '  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"'
        f"{trust_flag} \\\n"
        '  "$@"\n'
    )


def prepare_bundle(
    *,
    base_checkpoint: Path,
    trained_checkpoint: Path,
    qzbc_root: Path,
    output_dir: Path,
    base_model_name: str,
    trained_model_name: str,
    ollama_base_url: str = "http://127.0.0.1:11434",
    openai_base_port: int = 8101,
    openai_trained_port: int = 8102,
    num_ctx: int = 32768,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
    trust_remote_code: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output} is not empty; pass --overwrite to refresh generated files"
        )
    base = inspect_checkpoint(base_checkpoint, "base")
    trained = inspect_checkpoint(trained_checkpoint, "trained")
    ensure_compatible(base, trained)
    qzbc = inspect_qzbc(qzbc_root)
    base_model_name = _ollama_model_name(base_model_name)
    trained_model_name = _ollama_model_name(trained_model_name)
    if base_model_name == trained_model_name:
        raise ValueError("base and trained Ollama model names must differ")

    ollama_dir = output / "ollama"
    gguf_dir = ollama_dir / "gguf"
    base_gguf = gguf_dir / "base-f16.gguf"
    trained_gguf = gguf_dir / "trained-f16.gguf"
    _atomic_text(
        ollama_dir / "Modelfile.base",
        _modelfile(base_gguf, num_ctx=num_ctx, temperature=temperature),
    )
    _atomic_text(
        ollama_dir / "Modelfile.trained",
        _modelfile(trained_gguf, num_ctx=num_ctx, temperature=temperature),
    )
    convert_script = (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n\n"
        'PYTHON_BIN="${PYTHON_BIN:-python3}"\n'
        'LLAMA_CPP_DIR="${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to llama.cpp}"\n'
        'CONVERTER="${LLAMA_CPP_CONVERT:-${LLAMA_CPP_DIR}/convert_hf_to_gguf.py}"\n'
        f"mkdir -p -- {_shell(gguf_dir)}\n"
        f'"${{PYTHON_BIN}}" "${{CONVERTER}}" {_shell(base.path)} \\\n'
        f"  --outfile {_shell(base_gguf)} --outtype f16\n"
        f'"${{PYTHON_BIN}}" "${{CONVERTER}}" {_shell(trained.path)} \\\n'
        f"  --outfile {_shell(trained_gguf)} --outtype f16\n"
        f"test -s {_shell(base_gguf)}\n"
        f"test -s {_shell(trained_gguf)}\n"
    )
    _atomic_text(
        ollama_dir / "convert_hf_to_gguf.sh",
        convert_script,
        executable=True,
    )
    register_script = (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n\n"
        f"PROJECT_ROOT={_shell(Path(__file__).resolve().parents[2])}\n"
        'PYTHON_BIN="${PYTHON_BIN:-python3}"\n'
        'OLLAMA_BIN="${OLLAMA_BIN:-ollama}"\n'
        f"OLLAMA_BASE_URL={_shell(ollama_base_url)}\n"
        'export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"\n'
        f"test -s {_shell(base_gguf)}\n"
        f"test -s {_shell(trained_gguf)}\n"
        f'"${{OLLAMA_BIN}}" create {_shell(base_model_name)} '
        f"-f {_shell(ollama_dir / 'Modelfile.base')}\n"
        f'"${{OLLAMA_BIN}}" create {_shell(trained_model_name)} '
        f"-f {_shell(ollama_dir / 'Modelfile.trained')}\n"
        '"${PYTHON_BIN}" -m care_opd.qzbc_bridge check-ollama \\\n'
        '  --ollama-base-url "${OLLAMA_BASE_URL}" \\\n'
        f"  --model {_shell(base_model_name)} \\\n"
        f"  --model {_shell(trained_model_name)}\n"
    )
    _atomic_text(
        ollama_dir / "register_models.sh",
        register_script,
        executable=True,
    )

    qzbc_env = {
        "LLM_ENABLED": "true",
        "LLM_PROVIDER": "ollama",
        "LLM_ALLOW_FALLBACK": "false",
        "LLM_TEMPERATURE": str(temperature),
        "LLM_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "OLLAMA_BASE_URL": ollama_base_url,
        "OLLAMA_MODEL": base_model_name,
        "OLLAMA_NUM_CTX": str(num_ctx),
    }
    _atomic_text(output / "qzbc.ollama.env", _env_text(qzbc_env))

    openai_dir = output / "openai_compatible"
    for label, checkpoint, model_name, port in (
        ("base", base, base_model_name, openai_base_port),
        ("trained", trained, trained_model_name, openai_trained_port),
    ):
        _atomic_text(
            openai_dir / f"launch_{label}_vllm.sh",
            _vllm_script(
                checkpoint,
                model_name=model_name,
                port=port,
                num_ctx=num_ctx,
                trust_remote_code=trust_remote_code,
            ),
            executable=True,
        )
        _atomic_text(
            openai_dir / f"qzbc.{label}.env",
            _env_text(
                {
                    "LLM_ENABLED": "true",
                    "LLM_PROVIDER": "openai_compatible",
                    "LLM_ALLOW_FALLBACK": "false",
                    "LLM_TEMPERATURE": str(temperature),
                    "LLM_MAX_OUTPUT_TOKENS": str(max_output_tokens),
                    "OPENAI_COMPATIBLE_BASE_URL": (
                        f"http://127.0.0.1:{port}/v1"
                    ),
                    "OPENAI_COMPATIBLE_API_KEY": "",
                    "OPENAI_COMPATIBLE_MODEL": model_name,
                }
            ),
        )

    compare_script = (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n\n"
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"PROJECT_ROOT={_shell(Path(__file__).resolve().parents[2])}\n"
        'PYTHON_BIN="${PYTHON_BIN:-python3}"\n'
        'QZBC_URL="${QZBC_URL:-http://127.0.0.1:8000}"\n'
        'OUTPUT="${QZBC_AB_OUTPUT:-${SCRIPT_DIR}/comparison.json}"\n'
        'export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"\n'
        'exec "${PYTHON_BIN}" -m care_opd.qzbc_bridge compare \\\n'
        '  --qzbc-url "${QZBC_URL}" \\\n'
        f"  --base-model {_shell(base_model_name)} \\\n"
        f"  --trained-model {_shell(trained_model_name)} \\\n"
        '  --suite "${QZBC_SUITE:-full}" \\\n'
        '  --output "${OUTPUT}" "$@"\n'
    )
    _atomic_text(output / "compare_ollama.sh", compare_script, executable=True)

    manifest = {
        "schema_version": "care-opd-qzbc-bridge-v1",
        "base": base.as_dict(),
        "trained": trained.as_dict(),
        "qzbc_contract": qzbc.as_dict(),
        "model_names": {
            "base": base_model_name,
            "trained": trained_model_name,
        },
        "qzbc_environment": qzbc_env,
        "comparison_contract": {
            "endpoint": QZBC_EXPERIMENT_PATH,
            "suite": "full",
            "full_suite_cases": qzbc.full_suite_cases,
            "only_changed_field": "model",
            "fallback_allowed": False,
        },
        "operational_notes": [
            "restart qzbc after changing environment files because settings are cached",
            "comparison API model override does not require a restart between arms",
            "do not use qzbc scripts that hard-code qwen2.5:7b for deployment checks",
            "LightRAG has an independent model configuration and is not changed",
        ],
        "ollama_note": (
            "Qwen-family merged HF checkpoints are converted to GGUF before "
            "Ollama registration; direct Safetensors import is not assumed."
        ),
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def check_ollama_models(
    ollama_base_url: str,
    models: Sequence[str],
    *,
    timeout: float = 10.0,
) -> tuple[str, ...]:
    request = urllib.request.Request(
        _url(ollama_base_url, "/api/tags"),
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Ollama /api/tags check failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Ollama /api/tags returned a non-object response")
    available = tuple(
        str(item.get("name") or item.get("model") or "")
        for item in payload.get("models", ())
        if isinstance(item, Mapping)
    )
    missing = [
        _ollama_model_name(model)
        for model in models
        if _ollama_model_name(model) not in available
    ]
    if missing:
        raise RuntimeError(
            f"Ollama models missing from /api/tags: {missing}; "
            f"available={list(available)}"
        )
    return available


def _http_json(
    url: str,
    *,
    payload: Mapping[str, Any],
    token: Optional[str] = None,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"qzbc HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"qzbc request failed: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("qzbc returned a non-object response")
    if raw.get("success") is not True or not isinstance(raw.get("data"), dict):
        raise RuntimeError(f"qzbc returned an error response: {raw}")
    return dict(raw["data"])


def login(
    qzbc_url: str,
    *,
    username: str,
    password: str,
    timeout: float,
) -> str:
    response = _http_json(
        _url(qzbc_url, QZBC_LOGIN_PATH),
        payload={
            "username": username,
            "password": password,
            "entry_role": "doctor",
        },
        timeout=timeout,
    )
    token = str(response.get("access_token") or "")
    if not token:
        raise RuntimeError("qzbc login response has no access_token")
    return token


def run_qzbc_experiment(
    qzbc_url: str,
    *,
    token: str,
    provider: str,
    model: str,
    suite: str,
    timeout: float,
) -> dict[str, Any]:
    return _http_json(
        _url(qzbc_url, QZBC_EXPERIMENT_PATH),
        payload={"provider": provider, "model": model, "suite": suite},
        token=token,
        timeout=timeout,
    )


def _validate_report(
    report: Mapping[str, Any],
    *,
    model: str,
    provider: str,
    allow_fallback: bool,
) -> None:
    if report.get("model") != model or report.get("provider") != provider:
        raise RuntimeError(
            "qzbc report identity mismatch: "
            f"expected {provider}/{model}, got "
            f"{report.get('provider')}/{report.get('model')}"
        )
    fallback_count = int(report.get("fallback_count") or 0)
    if fallback_count and not allow_fallback:
        raise RuntimeError(
            f"{model} used qzbc fallback in {fallback_count} cases; "
            "comparison aborted"
        )
    mismatched_cases = [
        str(item.get("case_id") or "unknown")
        for item in report.get("items", ())
        if isinstance(item, Mapping)
        and (
            item.get("provider") != provider
            or item.get("model") != model
        )
    ]
    if mismatched_cases and not allow_fallback:
        raise RuntimeError(
            f"{model} was not the actual provider model for cases "
            f"{mismatched_cases}; comparison aborted"
        )


def _exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def compare_reports(
    base: Mapping[str, Any],
    trained: Mapping[str, Any],
) -> dict[str, Any]:
    base_items = {
        str(item["case_id"]): item for item in base.get("items", [])
    }
    trained_items = {
        str(item["case_id"]): item for item in trained.get("items", [])
    }
    if set(base_items) != set(trained_items):
        raise ValueError(
            "qzbc scenario sets differ between base and trained reports"
        )
    pairs: list[dict[str, Any]] = []
    wins = losses = ties = 0
    for case_id in sorted(base_items):
        before = base_items[case_id]
        after = trained_items[case_id]
        base_passed = bool(before.get("passed"))
        trained_passed = bool(after.get("passed"))
        delta = int(trained_passed) - int(base_passed)
        wins += int(delta > 0)
        losses += int(delta < 0)
        ties += int(delta == 0)
        pairs.append(
            {
                "case_id": case_id,
                "role": after.get("role"),
                "task_type": after.get("task_type"),
                "base_passed": base_passed,
                "trained_passed": trained_passed,
                "delta": delta,
                "base_fallback": bool(before.get("fallback_used")),
                "trained_fallback": bool(after.get("fallback_used")),
                "base_latency_ms": int(before.get("latency_ms") or 0),
                "trained_latency_ms": int(after.get("latency_ms") or 0),
            }
        )
    metric_delta = {
        metric: float(trained.get(metric, 0.0) or 0.0)
        - float(base.get(metric, 0.0) or 0.0)
        for metric in REPORT_METRICS
    }
    return {
        "schema_version": "care-opd-qzbc-comparison-v1",
        "provider": base.get("provider"),
        "suite": base.get("suite"),
        "base_model": base.get("model"),
        "trained_model": trained.get("model"),
        "total_cases": len(pairs),
        "paired_outcomes": {
            "trained_wins": wins,
            "trained_losses": losses,
            "ties": ties,
            "exact_mcnemar_p_value": _exact_mcnemar(wins, losses),
        },
        "metric_delta_trained_minus_base": metric_delta,
        "base_report": dict(base),
        "trained_report": dict(trained),
        "case_pairs": pairs,
    }


def run_comparison(
    *,
    qzbc_url: str,
    base_model: str,
    trained_model: str,
    output: Path,
    suite: str = "full",
    provider: str = "ollama",
    token: Optional[str] = None,
    username: str = "doctor_demo",
    password: Optional[str] = None,
    timeout: float = 1800.0,
    allow_fallback: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = output.expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite")
    if provider == "ollama":
        base_model = _ollama_model_name(base_model)
        trained_model = _ollama_model_name(trained_model)
    if base_model == trained_model:
        raise ValueError("base and trained model names must differ")
    resolved_token = token
    if not resolved_token:
        if not password:
            raise ValueError(
                "set QZBC_TOKEN or QZBC_PASSWORD, or pass a token/password"
            )
        resolved_token = login(
            qzbc_url,
            username=username,
            password=password,
            timeout=timeout,
        )
    base = run_qzbc_experiment(
        qzbc_url,
        token=resolved_token,
        provider=provider,
        model=base_model,
        suite=suite,
        timeout=timeout,
    )
    _validate_report(
        base,
        model=base_model,
        provider=provider,
        allow_fallback=allow_fallback,
    )
    trained = run_qzbc_experiment(
        qzbc_url,
        token=resolved_token,
        provider=provider,
        model=trained_model,
        suite=suite,
        timeout=timeout,
    )
    _validate_report(
        trained,
        model=trained_model,
        provider=provider,
        allow_fallback=allow_fallback,
    )
    comparison = compare_reports(base, trained)
    comparison["qzbc_url"] = qzbc_url
    comparison["fallback_allowed"] = allow_fallback
    _atomic_json(target, comparison)
    csv_path = target.with_suffix(".cases.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        rows = comparison["case_pairs"]
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["case_id"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base-checkpoint", type=Path, required=True)
    prepare.add_argument("--trained-checkpoint", type=Path, required=True)
    prepare.add_argument("--qzbc-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--base-model-name", default="care-opd-base")
    prepare.add_argument("--trained-model-name", default="care-opd-trained")
    prepare.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
    )
    prepare.add_argument("--openai-base-port", type=int, default=8101)
    prepare.add_argument("--openai-trained-port", type=int, default=8102)
    prepare.add_argument("--num-ctx", type=int, default=32768)
    prepare.add_argument("--temperature", type=float, default=0.2)
    prepare.add_argument("--max-output-tokens", type=int, default=1200)
    prepare.add_argument("--trust-remote-code", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument(
        "--qzbc-url",
        default="http://127.0.0.1:8000",
    )
    compare.add_argument("--base-model", required=True)
    compare.add_argument("--trained-model", required=True)
    compare.add_argument(
        "--provider",
        choices=("ollama", "openai_compatible"),
        default="ollama",
    )
    compare.add_argument("--suite", choices=("quick", "full"), default="full")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--username", default="doctor_demo")
    compare.add_argument("--token-env", default="QZBC_TOKEN")
    compare.add_argument("--password-env", default="QZBC_PASSWORD")
    compare.add_argument("--timeout", type=float, default=1800.0)
    compare.add_argument("--allow-fallback", action="store_true")
    compare.add_argument("--overwrite", action="store_true")

    check_ollama = subparsers.add_parser("check-ollama")
    check_ollama.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
    )
    check_ollama.add_argument("--model", action="append", required=True)
    check_ollama.add_argument("--timeout", type=float, default=10.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    namespace = _build_parser().parse_args(argv)
    if namespace.command == "prepare":
        manifest = prepare_bundle(
            base_checkpoint=namespace.base_checkpoint,
            trained_checkpoint=namespace.trained_checkpoint,
            qzbc_root=namespace.qzbc_root,
            output_dir=namespace.output_dir,
            base_model_name=namespace.base_model_name,
            trained_model_name=namespace.trained_model_name,
            ollama_base_url=namespace.ollama_base_url,
            openai_base_port=namespace.openai_base_port,
            openai_trained_port=namespace.openai_trained_port,
            num_ctx=namespace.num_ctx,
            temperature=namespace.temperature,
            max_output_tokens=namespace.max_output_tokens,
            trust_remote_code=namespace.trust_remote_code,
            overwrite=namespace.overwrite,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if namespace.command == "check-ollama":
        available = check_ollama_models(
            namespace.ollama_base_url,
            namespace.model,
            timeout=namespace.timeout,
        )
        print(json.dumps({"available": available}, ensure_ascii=False, indent=2))
        return 0

    token = os.environ.get(namespace.token_env)
    password = os.environ.get(namespace.password_env)
    comparison = run_comparison(
        qzbc_url=namespace.qzbc_url,
        base_model=namespace.base_model,
        trained_model=namespace.trained_model,
        output=namespace.output,
        suite=namespace.suite,
        provider=namespace.provider,
        token=token,
        username=namespace.username,
        password=password,
        timeout=namespace.timeout,
        allow_fallback=namespace.allow_fallback,
        overwrite=namespace.overwrite,
    )
    print(json.dumps(comparison["paired_outcomes"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
