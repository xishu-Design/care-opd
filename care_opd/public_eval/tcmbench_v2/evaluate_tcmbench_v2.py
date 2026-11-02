#!/usr/bin/env python3
"""Four-GPU, resumable evaluator for the TCMBench V2 robustness track.

The primary estimand is the semantically paired SDT -> SDT_reverse / SDT_shuffle
change.  Other public tasks are preserved as contextual diagnostics; known
annotation/prompt defects are never averaged into the primary robustness claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import re
import signal
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
TCMEVAL_RUNNER = HERE.parent / "tcmeval_pa"
PUBLIC_EVAL_ROOT = HERE.parent
UNIFIED_METRIC_SCHEMA = HERE.parent / "UNIFIED_METRIC_SCHEMA.json"
if str(TCMEVAL_RUNNER) not in sys.path:
    sys.path.insert(0, str(TCMEVAL_RUNNER))
if str(PUBLIC_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLIC_EVAL_ROOT))

from evaluate_tcmeval_pa import (  # type: ignore  # noqa: E402
    _load_suite_entry,
    freeze_suite,
    validate_frozen_model,
)
from protocol import (  # noqa: E402
    ALL_13_TASKS,
    DATA_COMMIT,
    PIPELINE_REFERENCE_COMMIT,
    PRIMARY_VALID_TASKS,
    PROTOCOL_VERSION,
    ROBUSTNESS_TASKS,
    TASK_TO_FILE,
    build_sdt_pair_manifest,
    canonical_hash,
    exact_mcnemar,
    inspect_known_data_issues,
    load_episodes,
    load_overlap_exclusions,
    metric_block,
    sha256_file,
    verify_snapshot,
)
from runtime_identity import runtime_identity  # noqa: E402


ANSWER_ONLY_SYSTEM = (
    "你正在完成中医选择题稳健性测评。只输出一个答案，不要解释。"
    "严格使用格式 <eoa>A</eoa>，其中 A 替换为 A、B、C、D 或 E。"
)
ANSWER_ONLY_NUMERIC_SYSTEM = (
    "根据给定答案判断它对应问题列表中的哪一项。只输出编号，不要解释。"
    "严格使用格式 <eoa>[编号]</eoa>。"
)
PROTOCOLS = ("answer_only", "reconstructed_official_cot")
PREREGISTRATION_SCHEMA = "care-opd-public-generalization-preregistration-lock-v1"
ROOT_PREREGISTRATION_SCHEMA = "care-opd-public-preregistration-lock-v1"
EXPECTED_MODEL_ORDER = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)
CLUSTER_BOOTSTRAP_REPLICATES = 2000
CLUSTER_BOOTSTRAP_SEED = 20260808
NONINFERIORITY_MARGIN = -0.02
FORMAL_NUM_WORKERS = 4
FORMAL_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")


def analysis_spec() -> dict[str, Any]:
    """Frozen inferential contract for the public paired track."""

    return {
        "schema_version": "care-opd-tcmbench-cluster-analysis-v1",
        "primary_effectiveness": "perturbed_task_success_rate",
        "primary_robustness": "conditional_retention",
        "regression_rate": "1 - conditional_retention",
        "cluster_unit": "case_index",
        "cluster_scope": "all semantic subquestions and both perturbations within a clean case",
        "bootstrap": {
            "method": "nonparametric pairs-cluster percentile bootstrap",
            "replicates": CLUSTER_BOOTSTRAP_REPLICATES,
            "seed": CLUSTER_BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "noninferiority_margin_absolute": NONINFERIORITY_MARGIN,
        "diagnostic_only": [
            "robustness_drop_pp",
            "prediction_consistency",
            "worst_perturbation_tsr",
            "unclustered_exact_mcnemar",
        ],
    }


def unified_metric_schema_identity() -> dict[str, str]:
    payload = json.loads(UNIFIED_METRIC_SCHEMA.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "care-opd-unified-effectiveness-robustness-v1":
        raise ValueError(f"unexpected unified metric schema: {UNIFIED_METRIC_SCHEMA}")
    required = set(payload.get("required_fields") or {})
    expected = {
        "clean_task_success_rate", "perturbed_task_success_rate", "conditional_retention",
        "regression_rate", "robustness_drop_pp", "prediction_consistency",
        "worst_perturbation_tsr", "clean_correct_perturbed_wrong",
        "clean_wrong_perturbed_correct", "paired_units",
    }
    if required != expected:
        raise ValueError(f"unified metric fields changed: {sorted(required)}")
    return {
        "schema_version": str(payload["schema_version"]),
        "sha256": sha256_file(UNIFIED_METRIC_SCHEMA),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized),
    )
    return len(materialized)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def freeze_preregistration_lock(
    path: Path,
    *,
    model_suite_fingerprint: str,
    benchmark_contract: Mapping[str, Any],
    code_revision: str,
) -> dict[str, Any]:
    """Create-once lock before predictions; an existing lock is never overwritten."""

    protocol_document = HERE.parent / "PUBLIC_GENERALIZATION_PREREGISTRATION.md"
    if not protocol_document.is_file():
        raise FileNotFoundError(protocol_document)
    if not code_revision.strip():
        raise ValueError("code_revision must be non-empty")
    payload = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "created_at_utc": utc_now(),
        "protocol_document_sha256": sha256_file(protocol_document),
        "code_revision": code_revision.strip(),
        "model_suite_fingerprint": model_suite_fingerprint,
        "benchmark": "tcmbench_v2",
        "benchmark_contract": dict(benchmark_contract),
        "benchmark_contract_fingerprint": canonical_hash(benchmark_contract),
        "amendment_policy": "create a new versioned lock; never overwrite this file",
    }
    if path.exists():
        return validate_preregistration_lock(
            path,
            model_suite_fingerprint=model_suite_fingerprint,
            benchmark_contract=benchmark_contract,
        )
    atomic_json(path, payload)
    return validate_preregistration_lock(
        path,
        model_suite_fingerprint=model_suite_fingerprint,
        benchmark_contract=benchmark_contract,
    )


def validate_preregistration_lock(
    path: Path,
    *,
    model_suite_fingerprint: str,
    benchmark_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the immutable lock matches every current contract input."""

    resolved = path.resolve(strict=True)
    payload = _load_json_object(resolved)
    schema = payload.get("schema_version")
    if schema == ROOT_PREREGISTRATION_SCHEMA:
        embedded_fingerprint = str(payload.get("contract_fingerprint") or "")
        contract_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"created_at_utc", "contract_fingerprint"}
        }
        if not embedded_fingerprint or canonical_hash(contract_payload) != embedded_fingerprint:
            raise ValueError("root preregistration contract fingerprint does not close")
        if tuple(payload.get("fixed_model_order") or ()) != EXPECTED_MODEL_ORDER:
            raise ValueError("root preregistration fixed model order changed")
        locked_suite = payload.get("model_suite")
        if not isinstance(locked_suite, Mapping):
            raise ValueError("root preregistration model suite is absent")
        if locked_suite.get("suite_fingerprint") != model_suite_fingerprint:
            raise ValueError("run model suite differs from root preregistration")
        disclosure = payload.get("timing_disclosure")
        if not isinstance(disclosure, Mapping) or disclosure.get(
            "tcmbench_v2_predictions_observed"
        ) is not False:
            raise ValueError("TCMBench timing disclosure is absent or not preregistered")
        project_root = HERE.parents[1]
        locked_files = payload.get("locked_files")
        if not isinstance(locked_files, list) or not locked_files:
            raise ValueError("root preregistration has no locked file inventory")
        locked_by_path = {
            str(item.get("path")): item
            for item in locked_files
            if isinstance(item, Mapping)
        }
        required_locked = {
            "public_eval/UNIFIED_METRIC_SCHEMA.json",
            "public_eval/PUBLIC_GENERALIZATION_PREREGISTRATION.md",
            "public_eval/tcmbench_v2/protocol.py",
            "public_eval/tcmbench_v2/evaluate_tcmbench_v2.py",
            "public_eval/tcmbench_v2/run_existing_gpu4_5.sh",
            "public_eval/tcmbench_v2/overlap_exclusions.json",
        }
        if not required_locked.issubset(locked_by_path):
            raise ValueError(
                "root preregistration is missing TCMBench files: "
                + repr(sorted(required_locked - set(locked_by_path)))
            )
        for relative, item in locked_by_path.items():
            target = project_root / relative
            if not target.is_file():
                raise FileNotFoundError(f"preregistered file is missing: {target}")
            if int(item.get("bytes", -1)) != target.stat().st_size:
                raise ValueError(f"preregistered file size changed: {relative}")
            if str(item.get("sha256") or "") != sha256_file(target):
                raise ValueError(f"preregistered file hash changed: {relative}")
        return {
            "schema_version": ROOT_PREREGISTRATION_SCHEMA,
            "lock_kind": "shared_static_protocol_lock",
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "contract_fingerprint": embedded_fingerprint,
            "model_suite_fingerprint_run_binding": model_suite_fingerprint,
            "benchmark_contract_fingerprint_run_binding": canonical_hash(
                benchmark_contract
            ),
        }
    if schema != PREREGISTRATION_SCHEMA:
        raise ValueError(f"unexpected preregistration schema in {resolved}")
    protocol_document = HERE.parent / "PUBLIC_GENERALIZATION_PREREGISTRATION.md"
    current_document_sha = sha256_file(protocol_document)
    if payload.get("protocol_document_sha256") != current_document_sha:
        raise ValueError("preregistration protocol document changed after lock creation")
    if payload.get("model_suite_fingerprint") != model_suite_fingerprint:
        raise ValueError("preregistration model suite fingerprint mismatch")
    embedded_contract = payload.get("benchmark_contract")
    if not isinstance(embedded_contract, Mapping):
        raise ValueError("preregistration benchmark_contract is missing")
    embedded_fp = canonical_hash(embedded_contract)
    current_fp = canonical_hash(benchmark_contract)
    if payload.get("benchmark_contract_fingerprint") != embedded_fp:
        raise ValueError("preregistration embedded benchmark contract does not close")
    if embedded_fp != current_fp:
        raise ValueError("current benchmark contract differs from preregistration lock")
    if not str(payload.get("code_revision", "")).strip():
        raise ValueError("preregistration code_revision is empty")
    return {
        "schema_version": PREREGISTRATION_SCHEMA,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "protocol_document_sha256": current_document_sha,
        "code_revision": str(payload["code_revision"]),
        "model_suite_fingerprint": model_suite_fingerprint,
        "benchmark_contract_fingerprint": current_fp,
    }


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_letter(raw: str, protocol: str) -> dict[str, Any]:
    normalized = raw.strip().upper().translate(str.maketrans("ＡＢＣＤＥ", "ABCDE"))
    strict_answer_only = re.fullmatch(r"<EOA>\s*([A-E])\s*</EOA>", normalized)
    strict_official = re.fullmatch(
        r"(?:[\s\S]*?【答案】\s*[:：]?\s*)?([A-E])\s*<EOA>", normalized
    )
    strict_match = strict_answer_only if protocol == "answer_only" else strict_official
    parsed: str | None = strict_match.group(1) if strict_match else None
    parse_route = "strict" if parsed else None
    if parsed is None:
        tagged = re.findall(r"<EOA>\s*([A-E])\s*</EOA>", normalized)
        official = re.findall(r"【答案】\s*[:：]?\s*([A-E])(?=\s*<EOA>)", normalized)
        if tagged:
            parsed, parse_route = tagged[-1], "tagged_lenient"
        elif official:
            parsed, parse_route = official[-1], "official_lenient"
        elif re.fullmatch(r"[A-E]", normalized):
            parsed, parse_route = normalized, "bare_letter"
    return {
        "prediction": parsed,
        "parse_ok": parsed is not None,
        "strict_format": strict_match is not None,
        "parse_route": parse_route,
    }


def parse_numeric(raw: str, protocol: str, option_count: int) -> dict[str, Any]:
    normalized = raw.strip().translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    strict = re.fullmatch(r"<eoa>\s*\[(\d+)\]\s*</eoa>", normalized, re.IGNORECASE)
    parsed: str | None = strict.group(1) if strict else None
    route = "strict" if parsed else None
    if parsed is None:
        tagged = re.findall(r"<eoa>\s*\[?(\d+)\]?\s*</eoa>", normalized, re.IGNORECASE)
        if tagged:
            parsed, route = tagged[-1], "tagged_lenient"
        elif re.fullmatch(r"\[?\d+\]?", normalized):
            parsed, route = normalized.strip("[]"), "bare_number"
    if parsed is not None and not (1 <= int(parsed) <= option_count):
        parsed, route = None, "out_of_range"
    return {
        "prediction": parsed,
        "parse_ok": parsed is not None,
        "strict_format": strict is not None,
        "parse_route": route,
    }


def protocol_spec(name: str) -> dict[str, Any]:
    if name not in PROTOCOLS:
        raise ValueError(name)
    if name == "answer_only":
        return {
            "name": name,
            "official_prompt": False,
            "directly_comparable_to_published_official_numbers": False,
            "reason": "adapted answer-only protocol for economical, reliable paired robustness scoring",
            "sequential_chat": True,
            "enable_thinking": False,
            "do_sample": False,
            "letter_output_contract": "<eoa>A</eoa>",
            "numeric_output_contract": "<eoa>[N]</eoa>",
        }
    return {
        "name": name,
        "official_prompt": True,
        "directly_comparable_to_published_official_numbers": False,
        "reason": (
            "exact public prompt text and sequential chat are reconstructed locally, "
            "but the original API stack/version and known official dispatch bugs are not reproduced"
        ),
        "sequential_chat": True,
        "enable_thinking": False,
        "do_sample": False,
        "letter_output_contract": "public 【答案】 ... <eoa> contract",
        "numeric_output_contract": "public <eoa>[N]</eoa> contract",
    }


def initial_messages(episode: Mapping[str, Any], protocol: str) -> list[dict[str, str]]:
    if protocol == "answer_only":
        if episode["kind"] == "numeric":
            system = ANSWER_ONLY_NUMERIC_SYSTEM
            user = (
                f"给定答案：{episode['option']}\n"
                f"问题列表共有 {episode['option_count']} 项：\n"
                f"{episode['items'][0]['user_text']}"
            )
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        messages = [{"role": "system", "content": ANSWER_ONLY_SYSTEM}]
    else:
        system = str(episode["prefix_prompt"])
        if episode["kind"] == "numeric":
            system = system.replace("<answer>", str(episode["option"]))
            system = system.replace("<option_num>", str(episode["option_count"]))
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": str(episode["items"][0]["user_text"])},
            ]
        messages = [{"role": "system", "content": system}]
    if episode["kind"] == "sequential":
        messages.append({"role": "user", "content": str(episode["share_content"])})
    else:
        messages.append({"role": "user", "content": str(episode["items"][0]["user_text"])})
    return messages


def generation_spec(protocol: str, dtype: str, max_new_tokens: int) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "enable_thinking": False,
        "seed": 42,
    }


def recover_shard(path: Path, expected: Mapping[int, str], run_fp: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_bytes().splitlines()
    rows = []
    corrupt_tail = False
    for line_index, line in enumerate(lines):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if line_index != len(lines) - 1:
                raise ValueError(f"corrupt non-terminal shard line: {path}:{line_index + 1}")
            corrupt_tail = True
            break
        index = int(row.get("index", -1))
        if expected.get(index) != str(row.get("episode_id")):
            raise ValueError(f"unexpected episode in shard {path}: {index}")
        if row.get("run_fingerprint") != run_fp:
            raise ValueError(f"stale run fingerprint in {path}: {index}")
        if any(int(existing["index"]) == index for existing in rows):
            raise ValueError(f"duplicate episode {index} in {path}")
        rows.append(row)
    if corrupt_tail:
        atomic_jsonl(path, rows)
    return rows


def _generate_one(model: Any, tokenizer: Any, torch: Any, messages: list[dict[str, str]],
                  target_device: Any, max_new_tokens: int) -> tuple[str, float, int]:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "tokenizer chat template lacks enable_thinking=False; refusing protocol-changing fallback"
        ) from exc
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(target_device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, time.perf_counter() - started, int(new_tokens.numel())


def worker(worker_id: int, device: str, episodes: Sequence[dict[str, Any]], shard_path: Path,
           completion_path: Path, error_path: Path, model_entry: Mapping[str, Any],
           protocol: str, dtype: str, max_new_tokens: int, run_fp: str,
           fingerprints: Mapping[str, str]) -> None:
    try:
        project_root = HERE.parents[1]
        sys.path.insert(0, str(project_root / "src"))
        from care_opd.models import ModelArguments, load_causal_lm, load_tokenizer
        import torch

        expected = {int(row["index"]): str(row["episode_id"]) for row in episodes}
        existing = recover_shard(shard_path, expected, run_fp)
        completed = {int(row["index"]) for row in existing}
        remaining = [row for row in episodes if int(row["index"]) not in completed]
        if remaining:
            model_path = str(model_entry["canonical_path"])
            print(
                f"[worker] worker={worker_id} device={device} loading={model_path} "
                f"remaining={len(remaining)}", flush=True,
            )
            model_args = ModelArguments(
                student_model_name_or_path=model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
            tokenizer = load_tokenizer(model_args)
            model = load_causal_lm(model_path, model_args, trainable=False)
            target_device = torch.device(device)
            model.to(target_device)
            model.eval()
            model.config.use_cache = True
            print(f"[worker] worker={worker_id} device={device} model_loaded", flush=True)

            for position, episode in enumerate(remaining, 1):
                messages = initial_messages(episode, protocol)
                predictions = []
                for item_position, item in enumerate(episode["items"]):
                    if episode["kind"] == "sequential":
                        messages.append({"role": "user", "content": str(item["user_text"])})
                    raw, latency, generated_tokens = _generate_one(
                        model, tokenizer, torch, messages, target_device, max_new_tokens
                    )
                    if episode["kind"] == "numeric":
                        parsed = parse_numeric(raw, protocol, int(episode["option_count"]))
                    else:
                        parsed = parse_letter(raw, protocol)
                    predictions.append(
                        {
                            "item_id": item["item_id"],
                            "semantic_id": item["semantic_id"],
                            "presentation_index": item["presentation_index"],
                            "gold": item["gold"],
                            "raw_output": raw,
                            **parsed,
                            "canonical_prediction": (
                                parsed["prediction"] if parsed["parse_ok"]
                                else "INVALID:" + canonical_hash(raw.strip())[:16]
                            ),
                            "correct": bool(parsed["parse_ok"] and parsed["prediction"] == item["gold"]),
                            "latency_seconds": latency,
                            "generated_tokens": generated_tokens,
                            "context_messages": len(messages),
                        }
                    )
                    if episode["kind"] == "sequential":
                        messages.append({"role": "assistant", "content": raw})
                row = {
                    "index": episode["index"],
                    "episode_id": episode["episode_id"],
                    "task": episode["task"],
                    "level": episode["level"],
                    "relative_path": episode["relative_path"],
                    "source_row_index": episode["source_row_index"],
                    "case_index": episode["case_index"],
                    "kind": episode["kind"],
                    "protocol": protocol,
                    "predictions": predictions,
                    "case_all_correct": all(item["correct"] for item in predictions),
                    "case_strict_format": all(item["strict_format"] for item in predictions),
                    "worker_id": worker_id,
                    "device": device,
                    "run_fingerprint": run_fp,
                    **fingerprints,
                }
                append_jsonl(shard_path, row)
                if position == 1 or position % 10 == 0 or position == len(remaining):
                    print(
                        f"[progress] worker={worker_id} device={device} "
                        f"episode={len(completed) + position}/{len(episodes)} "
                        f"task={episode['task']} id={episode['episode_id']}", flush=True,
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows = recover_shard(shard_path, expected, run_fp)
        if len(rows) != len(episodes):
            raise RuntimeError(f"worker {worker_id} produced {len(rows)}/{len(episodes)} episodes")
        atomic_json(
            completion_path,
            {"status": "complete", "worker_id": worker_id, "episodes": len(rows),
             "run_fingerprint": run_fp, "completed_at_utc": utc_now()},
        )
        error_path.unlink(missing_ok=True)
    except BaseException as exc:
        atomic_json(
            error_path,
            {"worker_id": worker_id, "error_type": type(exc).__name__, "error": str(exc),
             "traceback": traceback.format_exc()},
        )
        raise


def select_tasks(track: str, tasks: str | None) -> tuple[str, ...]:
    if tasks:
        selected = tuple(value.strip() for value in tasks.split(",") if value.strip())
    elif track == "robustness":
        selected = ROBUSTNESS_TASKS
    elif track == "all13":
        selected = ALL_13_TASKS
    elif track == "primary_sdt":
        selected = PRIMARY_VALID_TASKS
    else:
        raise ValueError(track)
    unknown = [task for task in selected if task not in TASK_TO_FILE or task == "herb_predict"]
    if unknown:
        raise ValueError(f"unsupported/evaluation-excluded task(s): {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate tasks")
    return selected


def benchmark_contract(
    *,
    tasks: Sequence[str],
    source_fingerprint: str,
    task_manifest_fingerprint: str,
    pair_manifest_fingerprint: str | None,
    overlap_manifest_fingerprint: str,
    prompt: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    """Inputs frozen before generation, excluding model-specific/runtime topology fields."""

    protocol_source = HERE / "protocol.py"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "official_data_commit": DATA_COMMIT,
        "historical_pipeline_reference_commit": PIPELINE_REFERENCE_COMMIT,
        "selected_tasks": list(tasks),
        "source_fingerprint": source_fingerprint,
        "task_manifest_fingerprint": task_manifest_fingerprint,
        "pair_manifest_fingerprint": pair_manifest_fingerprint,
        "overlap_manifest_fingerprint": overlap_manifest_fingerprint,
        "prompt": dict(prompt),
        "generation": dict(generation),
        "unified_metric_schema": unified_metric_schema_identity(),
        "analysis": analysis_spec(),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_source_sha256": sha256_file(protocol_source),
    }


def _prepared_contract(
    *,
    data_root: Path,
    suite: Path,
    protocol: str,
    track: str,
    tasks_arg: str | None,
    overlap_exclusions: Path,
    dtype: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Recompute the exact suite and benchmark identities used by lock/evaluation."""

    tasks = select_tasks(track, tasks_arg)
    episodes, task_manifest = load_episodes(data_root.resolve(), tasks)
    pair_manifest = build_sdt_pair_manifest(episodes)
    overlap = load_overlap_exclusions(overlap_exclusions)
    source_inventory = [
        {"path": TASK_TO_FILE[task], "sha256": sha256_file(data_root / TASK_TO_FILE[task])}
        for task in tasks
    ]
    contract = benchmark_contract(
        tasks=tasks,
        source_fingerprint=canonical_hash(source_inventory),
        task_manifest_fingerprint=str(task_manifest["manifest_fingerprint"]),
        pair_manifest_fingerprint=pair_manifest.get("pair_manifest_fingerprint"),
        overlap_manifest_fingerprint=str(overlap["fingerprint"]),
        prompt=protocol_spec(protocol),
        generation=generation_spec(protocol, dtype, max_new_tokens),
    )
    suite_payload = _load_json_object(suite.resolve(strict=True))
    models = suite_payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("model suite has no models")
    observed_order = tuple(str(item.get("label") or "") for item in models)
    if observed_order != EXPECTED_MODEL_ORDER:
        raise ValueError(
            f"model order changed: expected={EXPECTED_MODEL_ORDER}, observed={observed_order}"
        )
    recomputed_suite_fp = canonical_hash(
        [(item.get("label"), item.get("model_fingerprint")) for item in models]
    )
    if suite_payload.get("suite_fingerprint") != recomputed_suite_fp:
        raise ValueError("model suite fingerprint does not close")
    return contract, suite_payload, {
        "tasks": tasks,
        "episodes": episodes,
        "task_manifest": task_manifest,
        "pair_manifest": pair_manifest,
        "overlap": overlap,
        "source_inventory": source_inventory,
    }


def freeze_lock(args: argparse.Namespace) -> dict[str, Any]:
    contract, suite, _prepared = _prepared_contract(
        data_root=args.data_root,
        suite=args.suite,
        protocol=args.protocol,
        track=args.track,
        tasks_arg=args.tasks,
        overlap_exclusions=args.overlap_exclusions,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    identity = freeze_preregistration_lock(
        args.output,
        model_suite_fingerprint=str(suite["suite_fingerprint"]),
        benchmark_contract=contract,
        code_revision=args.code_revision,
    )
    print(json.dumps(identity, ensure_ascii=False, indent=2), flush=True)
    return identity


def inspect_data(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    snapshot = verify_snapshot(data_root, include_herb=True)
    tasks = select_tasks(args.track, args.tasks)
    episodes, task_manifest = load_episodes(data_root, tasks)
    pair_manifest = build_sdt_pair_manifest(episodes)
    issues = inspect_known_data_issues(data_root)
    overlap = load_overlap_exclusions(args.overlap_exclusions)
    loaded_ids = {item["item_id"].replace("\\", "/") for row in episodes for item in row["items"]}
    overlap_in_track = sorted(loaded_ids & overlap["item_ids"])
    pair_summary = {
        key: value for key, value in pair_manifest.items() if key not in {"pairs", "cases"}
    }
    result = {
        "status": "valid",
        "snapshot": snapshot,
        "selected_tasks": list(tasks),
        "episodes": len(episodes),
        "items": sum(len(row["items"]) for row in episodes),
        "task_manifest_fingerprint": task_manifest["manifest_fingerprint"],
        "pair_manifest": pair_summary,
        "known_data_issues": issues,
        "overlap_exclusions_manifest_sha256": overlap["fingerprint"],
        "overlap_excluded_items_in_selected_track": len(overlap_in_track),
        "overlap_excluded_item_ids_in_selected_track": overlap_in_track,
        "primary_claim_task_set": list(PRIMARY_VALID_TASKS),
        "unified_metric_schema": unified_metric_schema_identity(),
        "diagnostic_only_tasks": ["DDST", "DDST_hard", "SDDT", "SDDT_hard", "herb_predict"],
    }
    if args.pair_manifest_output:
        atomic_json(args.pair_manifest_output, pair_manifest)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _flatten(episode_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for episode in episode_rows:
        for item in episode["predictions"]:
            result.append(
                {
                    "episode_id": episode["episode_id"],
                    "task": episode["task"],
                    "level": episode["level"],
                    "case_index": episode["case_index"],
                    "kind": episode["kind"],
                    **item,
                }
            )
    return result


def _expected_result_keys(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[tuple[str, str]]]:
    episode_ids = [str(episode["episode_id"]) for episode in episodes]
    item_keys = [
        (str(episode["episode_id"]), str(item["item_id"]))
        for episode in episodes
        for item in episode["items"]
    ]
    return episode_ids, item_keys


def _completed_valid(
    run_dir: Path,
    run_fp: str,
    *,
    episodes: Sequence[Mapping[str, Any]],
    task_manifest_expected: Mapping[str, Any],
    pair_manifest_expected: Mapping[str, Any],
) -> bool:
    complete = run_dir / "COMPLETE.json"
    predictions = run_dir / "predictions.jsonl"
    episodes_path = run_dir / "episodes.jsonl"
    run_manifest = run_dir / "run_manifest.json"
    task_manifest = run_dir / "task_manifest.json"
    pair_manifest = run_dir / "pair_manifest.json"
    artifacts = {
        "predictions.jsonl": predictions,
        "episodes.jsonl": episodes_path,
        "run_manifest.json": run_manifest,
        "task_manifest.json": task_manifest,
        "pair_manifest.json": pair_manifest,
    }
    if not complete.is_file() or not all(path.is_file() for path in artifacts.values()):
        return False
    try:
        marker = json.loads(complete.read_text(encoding="utf-8"))
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        episode_rows = _read_jsonl(episodes_path)
        prediction_rows = _read_jsonl(predictions)
        task_manifest_actual = json.loads(task_manifest.read_text(encoding="utf-8"))
        pair_manifest_actual = json.loads(pair_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    try:
        expected_flat = _flatten(episode_rows)
        expected_episode_ids, expected_item_keys = _expected_result_keys(episodes)
        observed_episode_ids = [str(row.get("episode_id")) for row in episode_rows]
        observed_item_keys = [
            (str(row.get("episode_id")), str(row.get("item_id")))
            for row in prediction_rows
        ]
        artifact_hashes = marker.get("artifact_sha256")
        return bool(
            marker.get("status") == "complete"
            and marker.get("run_fingerprint") == run_fp
            and manifest.get("run_fingerprint") == run_fp
            and marker.get("episodes") == len(episode_rows)
            and marker.get("items") == len(prediction_rows)
            and manifest.get("episodes") == len(expected_episode_ids)
            and manifest.get("items") == len(expected_item_keys)
            and observed_episode_ids == expected_episode_ids
            and observed_item_keys == expected_item_keys
            and prediction_rows == expected_flat
            and task_manifest_actual == task_manifest_expected
            and pair_manifest_actual == pair_manifest_expected
            and manifest.get("task_manifest_fingerprint")
            == task_manifest_expected.get("manifest_fingerprint")
            and manifest.get("pair_manifest_fingerprint")
            == pair_manifest_expected.get("pair_manifest_fingerprint")
            and isinstance(artifact_hashes, Mapping)
            and all(
                artifact_hashes.get(name) == sha256_file(path)
                for name, path in artifacts.items()
            )
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    contract, suite_payload, prepared = _prepared_contract(
        data_root=args.data_root,
        suite=args.suite,
        protocol=args.protocol,
        track=args.track,
        tasks_arg=args.tasks,
        overlap_exclusions=args.overlap_exclusions,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    tasks = prepared["tasks"]
    episodes = prepared["episodes"]
    task_manifest = prepared["task_manifest"]
    pair_manifest = prepared["pair_manifest"]
    overlap = prepared["overlap"]
    source_inventory = prepared["source_inventory"]
    model_entry = _load_suite_entry(args.suite.resolve(), args.model_label)
    validate_frozen_model(model_entry)
    preregistration = validate_preregistration_lock(
        args.preregistration_lock,
        model_suite_fingerprint=str(suite_payload["suite_fingerprint"]),
        benchmark_contract=contract,
    )
    source_fp = str(contract["source_fingerprint"])
    prompt = contract["prompt"]
    generation = contract["generation"]
    run_core = {
        "protocol_version": PROTOCOL_VERSION,
        "official_data_commit": DATA_COMMIT,
        "historical_pipeline_reference_commit": PIPELINE_REFERENCE_COMMIT,
        "model_label": args.model_label,
        "model_fingerprint": model_entry["model_fingerprint"],
        "source_fingerprint": source_fp,
        "task_manifest_fingerprint": task_manifest["manifest_fingerprint"],
        "pair_manifest_fingerprint": pair_manifest.get("pair_manifest_fingerprint"),
        "overlap_manifest_fingerprint": overlap["fingerprint"],
        "prompt": prompt,
        "generation": generation,
        "unified_metric_schema": unified_metric_schema_identity(),
        "analysis": analysis_spec(),
        "benchmark_contract_fingerprint": canonical_hash(contract),
        "preregistration_lock": preregistration,
        "num_workers": args.num_workers,
        "devices": list(args.devices),
        "runtime_identity": runtime_identity(),
    }
    run_fp = canonical_hash(run_core)
    run_dir = args.output_root.resolve() / args.protocol / args.model_label
    run_dir.mkdir(parents=True, exist_ok=True)
    if _completed_valid(
        run_dir,
        run_fp,
        episodes=episodes,
        task_manifest_expected=task_manifest,
        pair_manifest_expected=pair_manifest,
    ):
        print(f"[resume] already complete: {run_dir}", flush=True)
        return json.loads((run_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    existing_manifest = run_dir / "run_manifest.json"
    if existing_manifest.is_file():
        old = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if old.get("run_fingerprint") != run_fp:
            raise ValueError(f"run directory has a different protocol fingerprint: {run_dir}")
    else:
        atomic_json(
            existing_manifest,
            {**run_core, "run_fingerprint": run_fp, "created_at_utc": utc_now(),
             "model_path": model_entry["canonical_path"], "source_inventory": source_inventory,
             "tasks": list(tasks), "episodes": len(episodes),
             "items": sum(len(row["items"]) for row in episodes)},
        )
    for manifest_name, expected_payload in (
        ("task_manifest.json", task_manifest),
        ("pair_manifest.json", pair_manifest),
    ):
        manifest_path = run_dir / manifest_name
        if not manifest_path.is_file():
            atomic_json(manifest_path, expected_payload)
            continue
        try:
            actual_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"corrupt {manifest_name} in resumable run; use a new output root: {exc}"
            ) from exc
        if actual_payload != expected_payload:
            raise ValueError(
                f"stale {manifest_name} in resumable run; use a new output root"
            )

    shard_root = run_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    partitions = [episodes[index::args.num_workers] for index in range(args.num_workers)]
    fingerprints = {
        "model_fingerprint": str(model_entry["model_fingerprint"]),
        "source_fingerprint": source_fp,
        "task_manifest_fingerprint": str(task_manifest["manifest_fingerprint"]),
    }
    context = mp.get_context("spawn")
    processes = []
    for worker_id, (device, partition) in enumerate(zip(args.devices, partitions, strict=True)):
        process = context.Process(
            target=worker,
            args=(
                worker_id, device, partition,
                shard_root / f"worker_{worker_id:02d}.jsonl",
                shard_root / f"worker_{worker_id:02d}.complete.json",
                shard_root / f"worker_{worker_id:02d}.error.json",
                model_entry, args.protocol, args.dtype, args.max_new_tokens, run_fp, fingerprints,
            ),
            name=f"tcmbench-{args.model_label}-{worker_id}",
        )
        process.start()
        processes.append(process)

    interrupted = False
    previous_handlers = {}
    def stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        for child in processes:
            if child.is_alive():
                child.terminate()
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, stop)
    try:
        for process in processes:
            process.join()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if interrupted:
        raise KeyboardInterrupt("evaluation interrupted; completed shard rows are resumable")
    failed = [process for process in processes if process.exitcode != 0]
    if failed:
        messages = []
        for process in failed:
            worker_id = processes.index(process)
            error = shard_root / f"worker_{worker_id:02d}.error.json"
            messages.append(error.read_text(encoding="utf-8") if error.is_file() else str(process.exitcode))
        raise RuntimeError("worker failure(s): " + " | ".join(messages))

    all_rows = []
    for worker_id, partition in enumerate(partitions):
        expected = {int(row["index"]): str(row["episode_id"]) for row in partition}
        all_rows.extend(
            recover_shard(shard_root / f"worker_{worker_id:02d}.jsonl", expected, run_fp)
        )
    all_rows.sort(key=lambda row: int(row["index"]))
    if len(all_rows) != len(episodes):
        raise RuntimeError("consolidated episode count mismatch")
    episode_path = run_dir / "episodes.jsonl"
    prediction_path = run_dir / "predictions.jsonl"
    atomic_jsonl(episode_path, all_rows)
    flat = _flatten(all_rows)
    atomic_jsonl(prediction_path, flat)
    marker = {
        "status": "complete", "completed_at_utc": utc_now(), "model_label": args.model_label,
        "protocol": args.protocol, "tasks": list(tasks), "episodes": len(all_rows),
        "items": len(flat), "run_fingerprint": run_fp,
        "preregistration_lock_sha256": preregistration["sha256"],
        "episodes_sha256": sha256_file(episode_path),
        "predictions_sha256": sha256_file(prediction_path),
        "artifact_sha256": {
            "episodes.jsonl": sha256_file(episode_path),
            "predictions.jsonl": sha256_file(prediction_path),
            "run_manifest.json": sha256_file(run_dir / "run_manifest.json"),
            "task_manifest.json": sha256_file(run_dir / "task_manifest.json"),
            "pair_manifest.json": sha256_file(run_dir / "pair_manifest.json"),
        },
    }
    atomic_json(run_dir / "COMPLETE.json", marker)
    print(json.dumps(marker, ensure_ascii=False, indent=2), flush=True)
    return marker


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _case_metric(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item["episode_id"])].append(item)
    total = len(groups)
    exact = sum(all(bool(item["correct"]) for item in rows) for rows in groups.values())
    return {"n": total, "all_correct": exact, "all_correct_rate": exact / total if total else None}


def _paired_stats(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(pair_rows)
    clean_correct = sum(bool(row["clean_correct"]) for row in pair_rows)
    pert_correct = sum(bool(row["perturbed_correct"]) for row in pair_rows)
    retained = sum(bool(row["clean_correct"] and row["perturbed_correct"]) for row in pair_rows)
    consistent = sum(bool(row["answer_consistent"]) for row in pair_rows)
    clean_only = sum(bool(row["clean_correct"] and not row["perturbed_correct"]) for row in pair_rows)
    pert_only = sum(bool(not row["clean_correct"] and row["perturbed_correct"]) for row in pair_rows)
    exploratory_p = exact_mcnemar(clean_only, pert_only)
    return {
        "n": total,
        "clean_accuracy": clean_correct / total if total else None,
        "perturbed_accuracy": pert_correct / total if total else None,
        "clean_to_perturbed_drop": (clean_correct - pert_correct) / total if total else None,
        "paired_answer_consistency": consistent / total if total else None,
        "clean_correct_retention": retained / clean_correct if clean_correct else None,
        "clean_correct": clean_correct,
        "perturbed_correct": pert_correct,
        "clean_only_correct": clean_only,
        "perturbed_only_correct": pert_only,
        "exploratory_unclustered_mcnemar": {
            "two_sided_exact_p": exploratory_p,
            "unit_treated_as_independent": "semantic clean/perturbed pair",
            "claim_status": "exploratory_only; repeated case clusters invalidate primary pair-independent inference",
        },
    }


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _primary_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    stats = _paired_stats(rows)
    retention = stats["clean_correct_retention"]
    return {
        "clean_task_success_rate": stats["clean_accuracy"],
        "perturbed_task_success_rate": stats["perturbed_accuracy"],
        "conditional_retention": retention,
        "regression_rate": 1.0 - retention if retention is not None else None,
    }


def _interval_block(estimate: float | None, values: Sequence[float]) -> dict[str, Any]:
    return {
        "estimate": estimate,
        "ci95_lower": _percentile(values, 0.025),
        "ci95_upper": _percentile(values, 0.975),
        "bootstrap_replicates_used": len(values),
    }


def _case_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
    replicates: int = CLUSTER_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Percentile CIs with all subquestions/perturbations in a case resampled together."""

    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["case_index"])].append(row)
    cluster_ids = sorted(groups)
    point = _primary_values(rows)
    draws: dict[str, list[float]] = {key: [] for key in point}
    rng = random.Random(seed)
    if cluster_ids:
        for _ in range(replicates):
            sampled: list[Mapping[str, Any]] = []
            for _cluster in cluster_ids:
                sampled.extend(groups[rng.choice(cluster_ids)])
            values = _primary_values(sampled)
            for key, value in values.items():
                if value is not None:
                    draws[key].append(float(value))
    return {
        "method": "nonparametric pairs-cluster percentile bootstrap",
        "cluster_unit": "case_index",
        "clusters": len(cluster_ids),
        "paired_rows": len(rows),
        "seed": seed,
        "replicates_requested": replicates,
        "metrics": {
            key: _interval_block(value, draws[key]) for key, value in point.items()
        },
    }


def _case_cluster_bootstrap_delta(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    seed: int = CLUSTER_BOOTSTRAP_SEED,
    replicates: int = CLUSTER_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """CARE-minus-baseline primary deltas with case-level paired resampling."""

    groups: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for care, baseline in pairs:
        care_case = int(care["case_index"])
        if care_case != int(baseline["case_index"]):
            raise ValueError("CARE/baseline case_index mismatch")
        groups[care_case].append((care, baseline))
    cluster_ids = sorted(groups)

    def differences(
        selected: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ) -> dict[str, float | None]:
        care_values = _primary_values([care for care, _ in selected])
        baseline_values = _primary_values([baseline for _, baseline in selected])
        result: dict[str, float | None] = {}
        for key in care_values:
            left, right = care_values[key], baseline_values[key]
            result[key] = left - right if left is not None and right is not None else None
        return result

    point = differences(pairs)
    draws: dict[str, list[float]] = {key: [] for key in point}
    rng = random.Random(seed)
    if cluster_ids:
        for _ in range(replicates):
            sampled: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            for _cluster in cluster_ids:
                sampled.extend(groups[rng.choice(cluster_ids)])
            values = differences(sampled)
            for key, value in values.items():
                if value is not None:
                    draws[key].append(float(value))
    metrics = {}
    for key, estimate in point.items():
        block = _interval_block(estimate, draws[key])
        lower, upper = block["ci95_lower"], block["ci95_upper"]
        block["superiority_by_ci95"] = bool(lower is not None and lower > 0.0)
        block["difference_from_zero_by_ci95"] = bool(
            lower is not None and upper is not None and (lower > 0.0 or upper < 0.0)
        )
        if key in {"perturbed_task_success_rate", "conditional_retention"}:
            block["noninferiority_margin"] = NONINFERIORITY_MARGIN
            block["noninferiority_pass"] = bool(
                lower is not None and lower > NONINFERIORITY_MARGIN
            )
        metrics[f"{key}_delta_care_minus_baseline"] = block
    return {
        "method": "nonparametric paired case-cluster percentile bootstrap",
        "cluster_unit": "case_index",
        "clusters": len(cluster_ids),
        "paired_rows": len(pairs),
        "seed": seed,
        "replicates_requested": replicates,
        "metrics": metrics,
        "joint_primary_noninferiority_pass": all(
            metrics[f"{key}_delta_care_minus_baseline"]["noninferiority_pass"]
            for key in ("perturbed_task_success_rate", "conditional_retention")
        ),
    }


def summarize_model(run_dir: Path, pair_manifest: Mapping[str, Any], exclusions: set[str]) -> dict[str, Any]:
    items = _read_jsonl(run_dir / "predictions.jsonl")
    by_id = {str(item["item_id"]).replace("\\", "/"): item for item in items}
    if len(by_id) != len(items):
        raise ValueError(f"duplicate item predictions in {run_dir}")
    result: dict[str, Any] = {}
    for view, excluded in (("official_full", set()), ("overlap_clean", exclusions)):
        selected = [item for key, item in by_id.items() if key not in excluded]
        tasks = sorted({str(item["task"]) for item in selected})
        by_task = {task: metric_block([item for item in selected if item["task"] == task]) for task in tasks}
        task_accuracies = [value["accuracy"] for value in by_task.values() if value["n"]]
        result[view] = {
            "micro": metric_block(selected),
            "task_macro_accuracy": sum(task_accuracies) / len(task_accuracies) if task_accuracies else None,
            "by_task": by_task,
            "case_all_correct_by_task": {
                task: _case_metric([item for item in selected if item["task"] == task])
                for task in tasks
            },
            "excluded_items": len(items) - len(selected),
        }
        if pair_manifest.get("available"):
            paired_rows = []
            for pair in pair_manifest["pairs"]:
                clean_id = str(pair["clean_item_id"]).replace("\\", "/")
                pert_id = str(pair["perturbed_item_id"]).replace("\\", "/")
                if clean_id in excluded or pert_id in excluded:
                    continue
                clean, pert = by_id[clean_id], by_id[pert_id]
                paired_rows.append(
                    {
                        **pair,
                        "clean_prediction": clean["prediction"],
                        "perturbed_prediction": pert["prediction"],
                        "canonical_clean_prediction": clean.get(
                            "canonical_prediction",
                            clean["prediction"] if clean["parse_ok"]
                            else "INVALID:" + canonical_hash(str(clean["raw_output"]).strip())[:16],
                        ),
                        "canonical_perturbed_prediction": pert.get(
                            "canonical_prediction",
                            pert["prediction"] if pert["parse_ok"]
                            else "INVALID:" + canonical_hash(str(pert["raw_output"]).strip())[:16],
                        ),
                        "clean_correct": clean["correct"],
                        "perturbed_correct": pert["correct"],
                    }
                )
                paired_rows[-1]["answer_consistent"] = bool(
                    clean["parse_ok"]
                    and pert["parse_ok"]
                    and paired_rows[-1]["canonical_clean_prediction"]
                    == paired_rows[-1]["canonical_perturbed_prediction"]
                )
            pair_summary = {}
            for perturbation in ("SDT_reverse", "SDT_shuffle", "combined"):
                subset = paired_rows if perturbation == "combined" else [
                    row for row in paired_rows if row["perturbation"] == perturbation
                ]
                question_stats = _paired_stats(subset)
                case_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
                for row in subset:
                    case_groups[(str(row["perturbation"]), int(row["case_index"]))].append(row)
                case_rows = []
                for (name, case), rows in case_groups.items():
                    case_rows.append(
                        {
                            "perturbation": name,
                            "case_index": case,
                            "clean_correct": all(bool(row["clean_correct"]) for row in rows),
                            "perturbed_correct": all(bool(row["perturbed_correct"]) for row in rows),
                            "answer_consistent": all(bool(row["answer_consistent"]) for row in rows),
                        }
                    )
                pair_summary[perturbation] = {
                    "question_level": question_stats,
                    "question_level_case_cluster_bootstrap": _case_cluster_bootstrap(subset),
                    "case_all_correct_level": _paired_stats(case_rows),
                }
            result[view]["primary_sdt_robustness"] = pair_summary
            reverse = pair_summary["SDT_reverse"]["question_level"]
            shuffle = pair_summary["SDT_shuffle"]["question_level"]
            combined = pair_summary["combined"]["question_level"]
            clean_macro = (reverse["clean_accuracy"] + shuffle["clean_accuracy"]) / 2
            perturbed_macro = (
                reverse["perturbed_accuracy"] + shuffle["perturbed_accuracy"]
            ) / 2
            result[view]["public_paired_robustness_track"] = {
                "unit": "canonical-subquestion option exact (0/1)",
                "primary_effectiveness_metric": "perturbed_task_success_rate",
                "primary_robustness_metric": "conditional_retention",
                "regression_rate_relationship": "1 - conditional_retention",
                "clean_task_success_rate": combined["clean_accuracy"],
                "clean_task_success_rate_macro": clean_macro,
                "perturbed_task_success_rate": combined["perturbed_accuracy"],
                "perturbed_task_success_rate_macro": perturbed_macro,
                "conditional_retention": combined["clean_correct_retention"],
                "regression_rate": (
                    combined["clean_only_correct"] / combined["clean_correct"]
                    if combined["clean_correct"] else None
                ),
                "clean_correct_perturbed_wrong": combined["clean_only_correct"],
                "clean_wrong_perturbed_correct": combined["perturbed_only_correct"],
                "paired_units": combined["n"],
                "robustness_drop_pp": 100 * (
                    combined["clean_accuracy"] - combined["perturbed_accuracy"]
                ),
                "macro_drop_pp": 100 * (clean_macro - perturbed_macro),
                "prediction_consistency": combined["paired_answer_consistency"],
                "worst_perturbation_tsr": min(
                    reverse["perturbed_accuracy"], shuffle["perturbed_accuracy"]
                ),
                "primary_case_cluster_inference": pair_summary["combined"][
                    "question_level_case_cluster_bootstrap"
                ],
                "reverse": reverse,
                "shuffle": shuffle,
                "case_all_correct_secondary": pair_summary["combined"]["case_all_correct_level"],
            }
            if view == "official_full":
                atomic_jsonl(run_dir / "paired_predictions.jsonl", paired_rows)
    return result


def _care_comparisons(models: Mapping[str, Mapping[str, Any]], view: str) -> dict[str, Any]:
    if "care_opd" not in models:
        return {}
    care = models["care_opd"][view]
    output = {}
    for label, payload in models.items():
        if label == "care_opd":
            continue
        other = payload[view]
        care_robust = care.get("public_paired_robustness_track", {})
        other_robust = other.get("public_paired_robustness_track", {})
        output[label] = {
            "selected_micro_accuracy_delta": care["micro"]["accuracy"] - other["micro"]["accuracy"],
            "task_macro_accuracy_delta": care["task_macro_accuracy"] - other["task_macro_accuracy"],
            "perturbed_accuracy_delta": (
                care_robust.get("perturbed_task_success_rate") - other_robust.get("perturbed_task_success_rate")
                if care_robust.get("perturbed_task_success_rate") is not None and other_robust.get("perturbed_task_success_rate") is not None
                else None
            ),
            "robustness_drop_delta_care_minus_baseline": (
                care_robust.get("robustness_drop_pp") - other_robust.get("robustness_drop_pp")
                if care_robust.get("robustness_drop_pp") is not None and other_robust.get("robustness_drop_pp") is not None
                else None
            ),
            "answer_consistency_delta": (
                care_robust.get("prediction_consistency") - other_robust.get("prediction_consistency")
                if care_robust.get("prediction_consistency") is not None and other_robust.get("prediction_consistency") is not None
                else None
            ),
        }
    return output


def _binary_paired_comparison(care: Sequence[bool], baseline: Sequence[bool]) -> dict[str, Any]:
    if len(care) != len(baseline):
        raise ValueError("paired comparison length mismatch")
    care_only = sum(bool(a and not b) for a, b in zip(care, baseline, strict=True))
    baseline_only = sum(bool(b and not a) for a, b in zip(care, baseline, strict=True))
    return {
        "inference_status": "exploratory_unclustered_only",
        "warning": "Repeated case clusters violate the independent-pair assumption; use the case-cluster bootstrap for primary inference.",
        "paired_units": len(care),
        "care_success_rate": sum(care) / len(care) if care else None,
        "baseline_success_rate": sum(baseline) / len(baseline) if baseline else None,
        "care_only_success": care_only,
        "baseline_only_success": baseline_only,
        "exploratory_unclustered_mcnemar_exact_p": exact_mcnemar(care_only, baseline_only),
    }


def _main_pairwise_comparisons(output_root: Path, protocol: str, exclusions: set[str]) -> dict[str, Any]:
    care_path = output_root / protocol / "care_opd" / "paired_predictions.jsonl"
    if not care_path.is_file():
        return {}
    care_rows = _read_jsonl(care_path)
    key = lambda row: (
        str(row["perturbation"]), int(row["case_index"]), str(row["semantic_id"])
    )
    care_map = {key(row): row for row in care_rows}
    result = {}
    for baseline_label in ("clean_sft", "gkd_full"):
        baseline_path = output_root / protocol / baseline_label / "paired_predictions.jsonl"
        if not baseline_path.is_file():
            continue
        baseline_map = {key(row): row for row in _read_jsonl(baseline_path)}
        if care_map.keys() != baseline_map.keys():
            raise ValueError(f"paired prediction key mismatch: care_opd vs {baseline_label}")
        by_view = {}
        for view, excluded in (("official_full", set()), ("overlap_clean", exclusions)):
            pairs = [
                (care_map[item_key], baseline_map[item_key])
                for item_key in sorted(care_map)
                if str(care_map[item_key]["clean_item_id"]).replace("\\", "/") not in excluded
                and str(care_map[item_key]["perturbed_item_id"]).replace("\\", "/") not in excluded
            ]
            exploratory_effectiveness = _binary_paired_comparison(
                [bool(care["perturbed_correct"]) for care, _ in pairs],
                [bool(base["perturbed_correct"]) for _, base in pairs],
            )
            shared_clean_capable = [
                (care, base) for care, base in pairs
                if care["clean_correct"] and base["clean_correct"]
            ]
            exploratory_retention_on_shared_clean = _binary_paired_comparison(
                [bool(care["perturbed_correct"]) for care, _ in shared_clean_capable],
                [bool(base["perturbed_correct"]) for _, base in shared_clean_capable],
            )
            by_view[view] = {
                "primary_case_cluster_bootstrap": _case_cluster_bootstrap_delta(pairs),
                "exploratory_unclustered_exact_tests": {
                    "perturbed_success": exploratory_effectiveness,
                    "retention_on_shared_clean_correct_subset": (
                        exploratory_retention_on_shared_clean
                    ),
                },
                "retention_subset_note": (
                    "The primary cluster bootstrap compares each model's full conditional-retention "
                    "ratio with whole case_index clusters resampled together. The shared-clean-correct "
                    "McNemar view is retained only as an unclustered exploratory diagnostic."
                ),
            }
        result[f"care_opd_vs_{baseline_label}"] = by_view
    return result


def summarize_suite(args: argparse.Namespace) -> dict[str, Any]:
    contract, suite, prepared = _prepared_contract(
        data_root=args.data_root,
        suite=args.suite,
        protocol=args.protocol,
        track=args.track,
        tasks_arg=args.tasks,
        overlap_exclusions=args.overlap_exclusions,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    tasks = prepared["tasks"]
    task_manifest = prepared["task_manifest"]
    pair_manifest = prepared["pair_manifest"]
    overlap = prepared["overlap"]
    preregistration = validate_preregistration_lock(
        args.preregistration_lock,
        model_suite_fingerprint=str(suite["suite_fingerprint"]),
        benchmark_contract=contract,
    )
    models = {}
    run_manifests = {}
    for entry in suite["models"]:
        label = str(entry["label"])
        run_dir = args.output_root.resolve() / args.protocol / label
        run_manifest_path = run_dir / "run_manifest.json"
        if not run_manifest_path.is_file():
            if args.allow_partial_suite:
                continue
            raise FileNotFoundError(f"model evaluation incomplete: {run_manifest_path}")
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        expected_run_core = {
            "protocol_version": PROTOCOL_VERSION,
            "official_data_commit": DATA_COMMIT,
            "historical_pipeline_reference_commit": PIPELINE_REFERENCE_COMMIT,
            "model_label": label,
            "model_fingerprint": entry["model_fingerprint"],
            "source_fingerprint": contract["source_fingerprint"],
            "task_manifest_fingerprint": task_manifest["manifest_fingerprint"],
            "pair_manifest_fingerprint": pair_manifest.get(
                "pair_manifest_fingerprint"
            ),
            "overlap_manifest_fingerprint": overlap["fingerprint"],
            "prompt": contract["prompt"],
            "generation": contract["generation"],
            "unified_metric_schema": unified_metric_schema_identity(),
            "analysis": analysis_spec(),
            "benchmark_contract_fingerprint": canonical_hash(contract),
            "preregistration_lock": preregistration,
            "num_workers": manifest.get("num_workers"),
            "devices": manifest.get("devices"),
            "runtime_identity": manifest.get("runtime_identity"),
        }
        expected_run_fp = canonical_hash(expected_run_core)
        if manifest.get("run_fingerprint") != expected_run_fp:
            raise ValueError(f"run contract does not close for {label}")
        if manifest.get("model_path") != entry.get("canonical_path"):
            raise ValueError(f"model path mismatch for {label}")
        manifest_lock = manifest.get("preregistration_lock")
        if not isinstance(manifest_lock, Mapping) or manifest_lock.get(
            "sha256"
        ) != preregistration["sha256"]:
            raise ValueError(f"preregistration lock mismatch for {label}")
        if not _completed_valid(
            run_dir,
            expected_run_fp,
            episodes=prepared["episodes"],
            task_manifest_expected=task_manifest,
            pair_manifest_expected=pair_manifest,
        ):
            if args.allow_partial_suite:
                continue
            raise ValueError(
                f"model evaluation is incomplete or fails artifact/ID validation: {run_dir}"
            )
        models[label] = summarize_model(run_dir, pair_manifest, overlap["item_ids"])
        run_manifests[label] = {
            "run_fingerprint": manifest["run_fingerprint"],
            "model_fingerprint": manifest["model_fingerprint"],
            "model_path": manifest["model_path"],
        }
    result = {
        "schema_version": "care-opd-tcmbench-v2-suite-summary-v1",
        "created_at_utc": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "official_data_commit": DATA_COMMIT,
        "historical_pipeline_reference_commit": PIPELINE_REFERENCE_COMMIT,
        "prompt_protocol": protocol_spec(args.protocol),
        "suite_fingerprint": suite["suite_fingerprint"],
        "selected_tasks": list(tasks),
        "primary_valid_tasks": list(PRIMARY_VALID_TASKS),
        "primary_estimand": "semantic-paired SDT clean to reverse/shuffle robustness",
        "task_manifest_fingerprint": task_manifest["manifest_fingerprint"],
        "unified_metric_schema": unified_metric_schema_identity(),
        "pair_manifest_fingerprint": pair_manifest.get("pair_manifest_fingerprint"),
        "analysis": analysis_spec(),
        "preregistration_lock": preregistration,
        "overlap_exclusions_sha256": overlap["fingerprint"],
        "known_data_issues": inspect_known_data_issues(args.data_root),
        "models": models,
        "run_manifests": run_manifests,
        "care_comparisons": {
            view: _care_comparisons(models, view) for view in ("official_full", "overlap_clean")
        },
        "main_pairwise_comparisons": _main_pairwise_comparisons(
            args.output_root.resolve(), args.protocol, overlap["item_ids"]
        ),
        "claim_guardrail": (
            "Use primary_sdt_robustness for recovery/robustness transfer. "
            "DDST(_hard), SDDT(_hard), and herb_predict are diagnostic-only due to audited public-data defects."
        ),
    }
    summary_dir = args.output_root.resolve() / args.protocol
    atomic_json(summary_dir / "suite_summary.json", result)
    csv_rows = []
    for label, payload in models.items():
        for view in ("official_full", "overlap_clean"):
            overall = payload[view]
            robust = overall.get("public_paired_robustness_track", {})
            robust_case = overall.get("primary_sdt_robustness", {}).get("combined", {}).get("case_all_correct_level", {})
            csv_rows.append(
                {
                    "model": label,
                    "view": view,
                    "micro_n": overall["micro"]["n"],
                    "micro_accuracy": overall["micro"]["accuracy"],
                    "task_macro_accuracy": overall["task_macro_accuracy"],
                    "strict_format_rate": overall["micro"]["strict_format_rate"],
                    "clean_task_success_rate": robust.get("clean_task_success_rate"),
                    "clean_task_success_rate_macro": robust.get("clean_task_success_rate_macro"),
                    "perturbed_task_success_rate": robust.get("perturbed_task_success_rate"),
                    "perturbed_task_success_rate_macro": robust.get("perturbed_task_success_rate_macro"),
                    "conditional_retention": robust.get("conditional_retention"),
                    "regression_rate": robust.get("regression_rate"),
                    "clean_correct_perturbed_wrong": robust.get("clean_correct_perturbed_wrong"),
                    "clean_wrong_perturbed_correct": robust.get("clean_wrong_perturbed_correct"),
                    "paired_units": robust.get("paired_units"),
                    "robustness_drop_pp": robust.get("robustness_drop_pp"),
                    "macro_drop_pp": robust.get("macro_drop_pp"),
                    "prediction_consistency": robust.get("prediction_consistency"),
                    "worst_perturbation_tsr": robust.get("worst_perturbation_tsr"),
                    "sdt_pair_cases": robust_case.get("n"),
                    "sdt_case_clean_accuracy": robust_case.get("clean_accuracy"),
                    "sdt_case_perturbed_accuracy": robust_case.get("perturbed_accuracy"),
                    "sdt_case_answer_consistency": robust_case.get("paired_answer_consistency"),
                }
            )
    csv_path = summary_dir / "suite_summary.csv"
    if csv_rows:
        fd, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=csv_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, csv_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect-data", help="verify data, pairing, and known defects")
    inspect_parser.add_argument("--data-root", type=Path, required=True)
    inspect_parser.add_argument("--track", choices=("primary_sdt", "robustness", "all13"), default="robustness")
    inspect_parser.add_argument("--tasks")
    inspect_parser.add_argument("--overlap-exclusions", type=Path, default=HERE / "overlap_exclusions.json")
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.add_argument("--pair-manifest-output", type=Path)

    freeze_parser = commands.add_parser("freeze-suite", help="freeze six checkpoint fingerprints")
    freeze_parser.add_argument("--suite", type=Path, required=True)
    freeze_parser.add_argument("--model", action="append", required=True)

    lock_parser = commands.add_parser(
        "freeze-preregistration", help="create-once fail-closed benchmark contract lock"
    )
    lock_parser.add_argument("--data-root", type=Path, required=True)
    lock_parser.add_argument("--suite", type=Path, required=True)
    lock_parser.add_argument("--protocol", choices=PROTOCOLS, default="answer_only")
    lock_parser.add_argument(
        "--track", choices=("primary_sdt", "robustness", "all13"), default="robustness"
    )
    lock_parser.add_argument("--tasks")
    lock_parser.add_argument(
        "--overlap-exclusions", type=Path, default=HERE / "overlap_exclusions.json"
    )
    lock_parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    lock_parser.add_argument("--max-new-tokens", type=int, default=24)
    lock_parser.add_argument("--code-revision", required=True)
    lock_parser.add_argument("--output", type=Path, required=True)

    eval_parser = commands.add_parser("evaluate", help="evaluate one frozen suite model")
    eval_parser.add_argument("--data-root", type=Path, required=True)
    eval_parser.add_argument("--suite", type=Path, required=True)
    eval_parser.add_argument("--model-label", required=True)
    eval_parser.add_argument("--output-root", type=Path, required=True)
    eval_parser.add_argument("--protocol", choices=PROTOCOLS, default="answer_only")
    eval_parser.add_argument("--track", choices=("primary_sdt", "robustness", "all13"), default="robustness")
    eval_parser.add_argument("--tasks")
    eval_parser.add_argument("--overlap-exclusions", type=Path, default=HERE / "overlap_exclusions.json")
    eval_parser.add_argument("--preregistration-lock", type=Path, required=True)
    eval_parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    eval_parser.add_argument("--max-new-tokens", type=int, default=24)
    eval_parser.add_argument("--num-workers", type=int, default=FORMAL_NUM_WORKERS)
    eval_parser.add_argument("--devices", default=",".join(FORMAL_DEVICES))

    summary_parser = commands.add_parser("summarize-suite", help="score completed checkpoint runs")
    summary_parser.add_argument("--data-root", type=Path, required=True)
    summary_parser.add_argument("--suite", type=Path, required=True)
    summary_parser.add_argument("--output-root", type=Path, required=True)
    summary_parser.add_argument("--protocol", choices=PROTOCOLS, default="answer_only")
    summary_parser.add_argument("--track", choices=("primary_sdt", "robustness", "all13"), default="robustness")
    summary_parser.add_argument("--tasks")
    summary_parser.add_argument("--overlap-exclusions", type=Path, default=HERE / "overlap_exclusions.json")
    summary_parser.add_argument("--preregistration-lock", type=Path, required=True)
    summary_parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    summary_parser.add_argument("--max-new-tokens", type=int, default=24)
    summary_parser.add_argument("--allow-partial-suite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect-data":
        inspect_data(args)
    elif args.command == "freeze-suite":
        freeze_suite(args.suite, args.model)
    elif args.command == "freeze-preregistration":
        if args.max_new_tokens <= 0:
            raise ValueError("max-new-tokens must be positive")
        freeze_lock(args)
    elif args.command == "evaluate":
        args.devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
        if args.num_workers != FORMAL_NUM_WORKERS or args.devices != FORMAL_DEVICES:
            raise ValueError(
                "formal TCMBench runner requires four workers on logical "
                "cuda:0,cuda:1,cuda:2,cuda:3"
            )
        if args.max_new_tokens <= 0:
            raise ValueError("max-new-tokens must be positive")
        evaluate(args)
    elif args.command == "summarize-suite":
        summarize_suite(args)
    else:
        parser.error(str(args.command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
