#!/usr/bin/env python3
"""Paired controlled-robustness evaluation for TCMEval-PA v3.

This is deliberately separate from ``evaluate_tcmeval_pa.py`` and never writes
inside the completed official-evaluation root.  Each official ID is evaluated
twice with the same label-blind, no-thinking, greedy generation contract:

* clean: canonical option order;
* perturbed: a preregistered, ID-hashed, non-identity option permutation.

Perturbed predictions are mapped back to canonical labels before exact scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import signal
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import evaluate_tcmeval_pa as official_eval  # noqa: E402
from preregistration import validate_static_lock  # noqa: E402
from runtime_identity import runtime_identity  # noqa: E402


PROTOCOL_VERSION = "care-opd-tcmeval-pa-controlled-robustness-v1"
MAPPING_SCHEMA_VERSION = "tcmeval-pa-option-permutation-map-v1"
SUMMARY_SCHEMA_VERSION = "tcmeval-pa-controlled-summary-v1"
SUITE_SUMMARY_SCHEMA_VERSION = "tcmeval-pa-controlled-suite-summary-v1"
EXPECTED_LABELS = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)
PRIMARY_COMPARISONS = ("clean_sft", "gkd_full")
PERMUTATION_SALT = "TCMEval-PA-v3|controlled-option-permutation|20260808|v1"
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPLICATES = 10_000
NONINFERIORITY_MARGIN = -0.02
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
METRIC_SCHEMA_SOURCE = HERE.parent / "UNIFIED_METRIC_SCHEMA.json"
ANALYSIS_SPEC_SOURCE = HERE / "ANALYSIS_SPEC.json"
DEFAULT_PREREGISTRATION_LOCK = HERE.parent / "PREREGISTRATION_LOCK.json"
UNIFIED_FIELDS = (
    "clean_task_success_rate",
    "perturbed_task_success_rate",
    "conditional_retention",
    "regression_rate",
    "robustness_drop_pp",
    "prediction_consistency",
    "worst_perturbation_tsr",
    "clean_correct_perturbed_wrong",
    "clean_wrong_perturbed_correct",
    "paired_units",
)
TCMEVAL_CONTROLLED_PREREGISTERED_PATHS = (
    "public_eval/UNIFIED_METRIC_SCHEMA.json",
    "public_eval/PUBLIC_GENERALIZATION_PREREGISTRATION.md",
    "public_eval/tcmeval_pa/prepare_data.py",
    "public_eval/tcmeval_pa/evaluate_tcmeval_pa.py",
    "public_eval/tcmeval_pa/evaluate_controlled_robustness.py",
    "public_eval/tcmeval_pa/run_controlled_gpu4_5.sh",
    "public_eval/tcmeval_pa/ANALYSIS_SPEC.json",
    "public_eval/tcmeval_pa/data/TCMEval-PA.xlsx",
)


def _normalize_execution_topology(args: argparse.Namespace) -> None:
    """Validate and freeze the logical-to-physical worker topology."""

    devices = tuple(
        item.strip() for item in str(args.devices).split(",") if item.strip()
    )
    visible_devices = tuple(
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    )
    if args.num_workers < 1 or len(devices) != args.num_workers:
        raise ValueError("--num-workers must equal the number of --devices")
    if len(visible_devices) != args.num_workers:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must expose exactly one physical GPU per worker"
        )
    args.devices = devices
    args.visible_physical_devices = visible_devices


def _execution_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_workers": args.num_workers,
        "devices": list(args.devices),
        "visible_physical_devices": list(args.visible_physical_devices),
    }


class EvaluationInterrupted(BaseException):
    pass


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
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


def _validate_metric_schema(path: Path = METRIC_SCHEMA_SOURCE) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen metric schema: {path}")
    payload = _load_json(path)
    if tuple(payload.get("required_fields", {})) != UNIFIED_FIELDS:
        raise ValueError("root UNIFIED_METRIC_SCHEMA required_fields changed")
    if payload.get("primary_endpoints", {}).get("effectiveness") != (
        "perturbed_task_success_rate"
    ):
        raise ValueError("root unified effectiveness endpoint changed")
    return payload


def _validate_analysis_spec(path: Path = ANALYSIS_SPEC_SOURCE) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen TCMEval analysis spec: {path}")
    payload = _load_json(path)
    bootstrap = payload.get("confidence_interval", {})
    if bootstrap.get("replicates") != BOOTSTRAP_REPLICATES:
        raise ValueError("analysis spec bootstrap replicate count changed")
    if bootstrap.get("seed") != BOOTSTRAP_SEED:
        raise ValueError("analysis spec bootstrap seed changed")
    comparison = payload.get("comparison", {})
    if comparison.get("noninferiority_margin_absolute") != NONINFERIORITY_MARGIN:
        raise ValueError("analysis spec noninferiority margin changed")
    if comparison.get("joint_gate_endpoints") != [
        "perturbed_task_success_rate",
        "conditional_retention",
    ]:
        raise ValueError("analysis spec joint gate endpoints changed")
    return payload


def _validate_preregistration_lock(path: Path) -> tuple[Mapping[str, Any], str]:
    identity = validate_static_lock(
        path,
        project_root=HERE.parents[1],
        required_paths=TCMEVAL_CONTROLLED_PREREGISTERED_PATHS,
        timing_key="tcmeval_pa_controlled_predictions_observed",
        expected_model_order=EXPECTED_LABELS,
    )
    return identity, str(identity["sha256"])


def _assert_suite_preregistered(
    suite: Mapping[str, Any], lock_identity: Mapping[str, Any]
) -> None:
    if str(suite.get("suite_fingerprint") or "") != str(
        lock_identity.get("model_suite_fingerprint") or ""
    ):
        raise ValueError("model suite differs from preregistration lock")


def _freeze_schema_snapshots(output_root: Path) -> tuple[Path, str, Path, str]:
    _validate_metric_schema(METRIC_SCHEMA_SOURCE)
    _validate_analysis_spec(ANALYSIS_SPEC_SOURCE)
    schema_sha = official_eval.sha256_file(METRIC_SCHEMA_SOURCE)
    schema_target = output_root.resolve() / "UNIFIED_METRIC_SCHEMA.json"
    if schema_target.is_file():
        if official_eval.sha256_file(schema_target) != schema_sha:
            raise ValueError(
                f"metric schema snapshot differs in {output_root}; use a new output root"
            )
    else:
        _atomic_bytes(schema_target, METRIC_SCHEMA_SOURCE.read_bytes())
    analysis_sha = official_eval.sha256_file(ANALYSIS_SPEC_SOURCE)
    analysis_target = output_root.resolve() / "ANALYSIS_SPEC.json"
    if analysis_target.is_file():
        if official_eval.sha256_file(analysis_target) != analysis_sha:
            raise ValueError(
                f"analysis spec snapshot differs in {output_root}; use a new output root"
            )
    else:
        _atomic_bytes(analysis_target, ANALYSIS_SPEC_SOURCE.read_bytes())
    return schema_target, schema_sha, analysis_target, analysis_sha


def normalized_stem(question: str) -> str:
    return re.sub(r"\s+", "", question)


def normalized_stem_cluster(question: str) -> str:
    return official_eval.canonical_hash({"normalized_stem": normalized_stem(question)})


def _candidate_contents(question: official_eval.Question) -> Mapping[str, str]:
    pattern = re.compile(r"(?m)^[ \t]*([A-Z])[ \t]*[.．、:：][ \t]*")
    matches = list(pattern.finditer(question.candidate_answers))
    labels = tuple(match.group(1) for match in matches)
    if labels != question.allowed_options:
        raise ValueError(
            f"question {question.id} option-body labels {labels} differ from "
            f"inventory {question.allowed_options}"
        )
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(
            question.candidate_answers
        )
        content = question.candidate_answers[match.end() : stop].strip()
        if not content:
            raise ValueError(f"question {question.id} has an empty option {labels[index]}")
        result[labels[index]] = content
    return result


def candidate_answers_for_order(
    question: official_eval.Question,
    displayed_canonical_order: Sequence[str],
) -> str:
    allowed = tuple(question.allowed_options)
    order = tuple(displayed_canonical_order)
    if len(order) != len(allowed) or set(order) != set(allowed):
        raise ValueError(f"invalid permutation for question {question.id}: {order}")
    contents = _candidate_contents(question)
    return "\n".join(
        f"{display_label}. {contents[canonical_label]}"
        for display_label, canonical_label in zip(allowed, order, strict=True)
    )


def deterministic_nonidentity_order(question: official_eval.Question) -> tuple[str, ...]:
    labels = tuple(question.allowed_options)
    ranked = sorted(
        labels,
        key=lambda label: hashlib_digest(
            f"{PERMUTATION_SALT}|{question.id}|{label}".encode("utf-8")
        ),
    )
    order = tuple(ranked)
    if order == labels:
        rotation_digest = hashlib_digest(
            f"{PERMUTATION_SALT}|{question.id}|forced-rotation".encode("utf-8")
        )
        shift = 1 + int.from_bytes(rotation_digest[:8], "big") % (len(labels) - 1)
        order = labels[shift:] + labels[:shift]
    if order == labels:
        raise AssertionError(f"identity permutation survived for question {question.id}")
    return order


def hashlib_digest(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def mapping_item(question: official_eval.Question) -> Mapping[str, Any]:
    labels = tuple(question.allowed_options)
    order = deterministic_nonidentity_order(question)
    display_to_canonical = dict(zip(labels, order, strict=True))
    canonical_to_display = {canonical: display for display, canonical in display_to_canonical.items()}
    displayed_gold = tuple(sorted(canonical_to_display[item] for item in question.gold_options))
    clean_candidates = candidate_answers_for_order(question, labels)
    perturbed_candidates = candidate_answers_for_order(question, order)
    if clean_candidates == perturbed_candidates:
        raise ValueError(f"forced permutation was text-identical for question {question.id}")
    return {
        "index": question.index,
        "id": question.id,
        "allowed_options": list(labels),
        "canonical_gold_options": list(question.gold_options),
        "displayed_canonical_order": list(order),
        "display_to_canonical": display_to_canonical,
        "canonical_to_display": canonical_to_display,
        "perturbed_display_gold_options": list(displayed_gold),
        "clean_candidate_answers_sha256": hashlib_digest(
            clean_candidates.encode("utf-8")
        ).hex(),
        "perturbed_candidate_answers_sha256": hashlib_digest(
            perturbed_candidates.encode("utf-8")
        ).hex(),
    }


def freeze_mapping_manifest(
    data_path: Path,
    output_root: Path,
    preregistration_lock: Path,
) -> Mapping[str, Any]:
    questions = official_eval.load_questions(data_path)
    _, lock_sha = _validate_preregistration_lock(preregistration_lock)
    _, schema_sha, _, analysis_sha = _freeze_schema_snapshots(output_root)
    items = [mapping_item(question) for question in questions]
    semantic = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "dataset_sha256": official_eval.EXPECTED_DATA_SHA256,
        "permutation_spec": {
            "algorithm": "SHA256 rank by (salt, official ID, canonical label); "
            "deterministic nonzero rotation if the ranked order is identity",
            "salt": PERMUTATION_SALT,
            "forced_nonidentity": True,
            "score_space": "canonical option labels",
        },
        "paired_units": len(items),
        "items": items,
    }
    mapping_fingerprint = official_eval.canonical_hash(semantic)
    path = output_root.resolve() / "mapping_manifest.json"
    if path.is_file():
        existing = _load_json(path)
        for key, expected in semantic.items():
            if existing.get(key) != expected:
                raise ValueError(
                    f"frozen mapping differs at {key!r} in {path}; use a new output root"
                )
        if existing.get("mapping_fingerprint") != mapping_fingerprint:
            raise ValueError(f"invalid mapping_fingerprint in {path}")
        if existing.get("preregistration_lock_sha256") != lock_sha:
            raise ValueError(
                "preregistration lock changed after the mapping was frozen; use a new output root"
            )
        if existing.get("metric_schema_sha256") != schema_sha:
            raise ValueError("metric schema changed after the mapping was frozen")
        if existing.get("analysis_spec_sha256") != analysis_sha:
            raise ValueError("analysis spec changed after the mapping was frozen")
        return existing
    payload = {
        **semantic,
        "mapping_fingerprint": mapping_fingerprint,
        "preregistration_lock_path": str(preregistration_lock.resolve()),
        "preregistration_lock_sha256": lock_sha,
        "metric_schema_sha256": schema_sha,
        "analysis_spec_sha256": analysis_sha,
        "frozen_at_utc": official_eval.utc_now(),
    }
    official_eval.atomic_json(path, payload)
    return payload


def _question_variant(
    question: official_eval.Question,
    candidate_answers: str,
    gold_options: Sequence[str],
) -> official_eval.Question:
    return official_eval.Question(
        index=question.index,
        id=question.id,
        question=question.question,
        candidate_answers=candidate_answers,
        allowed_options=question.allowed_options,
        gold_options=tuple(gold_options),
        question_type=question.question_type,
        category=question.category,
        rule_id=question.rule_id,
        rule_summary=question.rule_summary,
    )


def _condition_questions(
    question: official_eval.Question, item: Mapping[str, Any]
) -> tuple[official_eval.Question, official_eval.Question]:
    labels = tuple(question.allowed_options)
    order = tuple(item["displayed_canonical_order"])
    clean = _question_variant(
        question,
        candidate_answers_for_order(question, labels),
        question.gold_options,
    )
    perturbed = _question_variant(
        question,
        candidate_answers_for_order(question, order),
        tuple(item["perturbed_display_gold_options"]),
    )
    return clean, perturbed


def map_display_prediction_to_canonical(
    parsed_options: Sequence[str], display_to_canonical: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(sorted(display_to_canonical[item] for item in parsed_options))


def generation_spec(dtype: str, max_new_tokens: int) -> Mapping[str, Any]:
    return {
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "enable_thinking": False,
        "seed_policy": "reset to 42 + canonical_index for each paired condition",
        "pair_order": "clean_then_perturbed",
    }


def _run_condition(
    *,
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    device: Any,
    question: official_eval.Question,
    canonical_gold: Sequence[str],
    display_to_canonical: Mapping[str, str],
    max_new_tokens: int,
) -> Mapping[str, Any]:
    messages = official_eval.prompt_messages(question, "label_blind")
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "tokenizer chat template does not accept enable_thinking=False; "
            "refusing a protocol-changing fallback"
        ) from exc
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch_module.manual_seed(42 + question.index)
    started = time.perf_counter()
    with torch_module.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = official_eval.parse_prediction(raw_output, question.allowed_options)
    canonical_prediction = map_display_prediction_to_canonical(
        parsed["parsed_options"], display_to_canonical
    )
    success = bool(
        parsed["parse_ok"] and canonical_prediction == tuple(canonical_gold)
    )
    return {
        "candidate_answers": question.candidate_answers,
        "messages": messages,
        "raw_output": raw_output,
        "display_prediction_options": list(parsed["parsed_options"]),
        "canonical_prediction_options": list(canonical_prediction),
        "invalid_letters": list(parsed["invalid_letters"]),
        "parse_ok": bool(parsed["parse_ok"]),
        "strict_format": bool(parsed["strict_format"]),
        "task_success": success,
        "strict_exact": bool(success and parsed["strict_format"]),
        "latency_seconds": time.perf_counter() - started,
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _recover_shard(
    path: Path, expected: Mapping[int, str], run_fingerprint: str
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw_lines = path.read_bytes().splitlines()
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    corrupted_last = False
    for line_index, line in enumerate(raw_lines):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if line_index != len(raw_lines) - 1:
                raise ValueError(f"corrupt non-terminal line in {path}")
            corrupted_last = True
            break
        index = int(row.get("index", -1))
        if expected.get(index) != str(row.get("id")):
            raise ValueError(f"unexpected pair row in {path}: index={index}")
        if row.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"stale pair row in {path}: index={index}")
        if index in seen:
            raise ValueError(f"duplicate index {index} in {path}")
        seen.add(index)
        rows.append(row)
    if corrupted_last:
        official_eval.atomic_jsonl(path, rows)
    return rows


def _worker(
    worker_id: int,
    device: str,
    questions: Sequence[official_eval.Question],
    mapping_by_id: Mapping[str, Mapping[str, Any]],
    shard_path: Path,
    completion_path: Path,
    error_path: Path,
    model_entry: Mapping[str, Any],
    max_new_tokens: int,
    dtype: str,
    run_fingerprint: str,
    fingerprints: Mapping[str, str],
) -> None:
    try:
        project_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(project_root / "src"))
        from care_opd.models import ModelArguments, load_causal_lm, load_tokenizer
        import torch

        expected = {question.index: question.id for question in questions}
        existing = _recover_shard(shard_path, expected, run_fingerprint)
        completed = {int(row["index"]) for row in existing}
        remaining = [question for question in questions if question.index not in completed]
        if remaining:
            model_path = str(model_entry["canonical_path"])
            print(
                f"[worker] worker={worker_id} device={device} status=loading_model "
                f"model={model_path} remaining_pairs={len(remaining)}",
                flush=True,
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
            if hasattr(model, "config"):
                model.config.use_cache = True
            print(
                f"[worker] worker={worker_id} device={device} status=model_loaded",
                flush=True,
            )
            for position, question in enumerate(remaining, 1):
                item = mapping_by_id[question.id]
                clean_question, perturbed_question = _condition_questions(question, item)
                identity_map = {label: label for label in question.allowed_options}
                clean = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    torch_module=torch,
                    device=target_device,
                    question=clean_question,
                    canonical_gold=question.gold_options,
                    display_to_canonical=identity_map,
                    max_new_tokens=max_new_tokens,
                )
                perturbed = _run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    torch_module=torch,
                    device=target_device,
                    question=perturbed_question,
                    canonical_gold=question.gold_options,
                    display_to_canonical=item["display_to_canonical"],
                    max_new_tokens=max_new_tokens,
                )
                clean_ok = bool(clean["task_success"])
                perturbed_ok = bool(perturbed["task_success"])
                if clean_ok and perturbed_ok:
                    flip = "both_correct"
                elif clean_ok:
                    flip = "clean_correct_to_wrong"
                elif perturbed_ok:
                    flip = "clean_wrong_to_correct"
                else:
                    flip = "both_wrong"
                consistency = bool(
                    clean["parse_ok"]
                    and perturbed["parse_ok"]
                    and clean["canonical_prediction_options"]
                    == perturbed["canonical_prediction_options"]
                )
                row = {
                    "index": question.index,
                    "id": question.id,
                    "question": question.question,
                    "canonical_candidate_answers": question.candidate_answers,
                    "canonical_gold_options": list(question.gold_options),
                    "question_type": question.question_type,
                    "category": question.category,
                    "rule_id": question.rule_id,
                    "rule_summary": question.rule_summary,
                    "normalized_stem_cluster": normalized_stem_cluster(question.question),
                    "displayed_canonical_order": item["displayed_canonical_order"],
                    "display_to_canonical": item["display_to_canonical"],
                    "perturbed_display_gold_options": item[
                        "perturbed_display_gold_options"
                    ],
                    "clean": clean,
                    "perturbed": perturbed,
                    "flip": flip,
                    "canonical_prediction_consistent": consistency,
                    "worker_id": worker_id,
                    "device": device,
                    "run_fingerprint": run_fingerprint,
                    **fingerprints,
                }
                _append_jsonl(shard_path, row)
                if position == 1 or position % 10 == 0 or position == len(remaining):
                    print(
                        f"[progress] worker={worker_id} device={device} "
                        f"pairs={len(completed) + position}/{len(questions)} id={question.id}",
                        flush=True,
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows = _recover_shard(shard_path, expected, run_fingerprint)
        if len(rows) != len(questions):
            raise RuntimeError(f"worker {worker_id} produced {len(rows)}/{len(questions)} pairs")
        official_eval.atomic_json(
            completion_path,
            {
                "status": "complete",
                "worker_id": worker_id,
                "paired_rows": len(rows),
                "run_fingerprint": run_fingerprint,
                "completed_at_utc": official_eval.utc_now(),
            },
        )
        error_path.unlink(missing_ok=True)
    except BaseException as exc:
        official_eval.atomic_json(
            error_path,
            {
                "worker_id": worker_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def unified_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    paired = len(rows)
    clean_correct = sum(bool(row["clean"]["task_success"]) for row in rows)
    perturbed_correct = sum(bool(row["perturbed"]["task_success"]) for row in rows)
    flips = Counter(str(row["flip"]) for row in rows)
    paired_state_counts = {
        "both_correct": flips["both_correct"],
        "clean_correct_to_wrong": flips["clean_correct_to_wrong"],
        "clean_wrong_to_correct": flips["clean_wrong_to_correct"],
        "both_wrong": flips["both_wrong"],
    }
    if sum(paired_state_counts.values()) != paired:
        raise ValueError("flip counts do not cover every paired unit")
    clean_tsr = clean_correct / paired if paired else None
    perturbed_tsr = perturbed_correct / paired if paired else None
    retained = paired_state_counts["both_correct"]
    regressed = paired_state_counts["clean_correct_to_wrong"]
    conditional_retention = retained / clean_correct if clean_correct else None
    regression_rate = regressed / clean_correct if clean_correct else None
    consistent = sum(bool(row["canonical_prediction_consistent"]) for row in rows)
    result = {
        "clean_task_success_rate": clean_tsr,
        "perturbed_task_success_rate": perturbed_tsr,
        "conditional_retention": conditional_retention,
        "regression_rate": regression_rate,
        "robustness_drop_pp": 100.0 * (clean_tsr - perturbed_tsr) if paired else None,
        "prediction_consistency": consistent / paired if paired else None,
        "worst_perturbation_tsr": perturbed_tsr if paired else None,
        "clean_correct_perturbed_wrong": regressed,
        "clean_wrong_perturbed_correct": paired_state_counts[
            "clean_wrong_to_correct"
        ],
        "paired_units": paired,
    }
    if tuple(result) != UNIFIED_FIELDS:
        raise AssertionError("unified metric field order drifted")
    if clean_correct and not math.isclose(
        float(conditional_retention) + float(regression_rate), 1.0
    ):
        raise AssertionError("retention and regression are not complementary")
    return result


def _cluster_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, tuple[int, ...]]:
    result: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    for row in rows:
        values = result[str(row["normalized_stem_cluster"])]
        clean = bool(row["clean"]["task_success"])
        perturbed = bool(row["perturbed"]["task_success"])
        values[0] += 1
        values[1] += int(clean)
        values[2] += int(perturbed)
        values[3] += int(clean and perturbed)
        values[4] += int(clean and not perturbed)
        values[5] += int(not clean and perturbed)
        values[6] += int(bool(row["canonical_prediction_consistent"]))
    return {key: tuple(value) for key, value in result.items()}


def _metrics_from_counts(counts: Sequence[int]) -> Mapping[str, float]:
    n, clean, perturbed, both, clean_to_wrong, _wrong_to_clean, _consistent = counts
    return {
        "clean_task_success_rate": clean / n,
        "perturbed_task_success_rate": perturbed / n,
        "conditional_retention": both / clean if clean else math.nan,
        "regression_rate": clean_to_wrong / clean if clean else math.nan,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(value for value in values if not math.isnan(value))
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Mapping[str, Any]:
    by_cluster = _cluster_counts(rows)
    clusters = sorted(by_cluster)
    if not clusters:
        raise ValueError("cannot bootstrap an empty result")
    samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(replicates):
        aggregate = [0] * 7
        for cluster in rng.choices(clusters, k=len(clusters)):
            for index, value in enumerate(by_cluster[cluster]):
                aggregate[index] += value
        for metric, value in _metrics_from_counts(aggregate).items():
            samples[metric].append(value)
    observed = unified_metrics(rows)
    return {
        "method": "normalized-stem cluster percentile bootstrap",
        "cluster_count": len(clusters),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "intervals": {
            metric: {
                "estimate": observed[metric],
                "low": _percentile(values, 0.025),
                "high": _percentile(values, 0.975),
            }
            for metric, values in samples.items()
        },
    }


def summarize_pairs(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    overall = unified_metrics(rows)
    state_counter = Counter(str(row["flip"]) for row in rows)
    paired_state_counts = {
        "both_correct": state_counter["both_correct"],
        "clean_correct_to_wrong": state_counter["clean_correct_to_wrong"],
        "clean_wrong_to_correct": state_counter["clean_wrong_to_correct"],
        "both_wrong": state_counter["both_wrong"],
    }
    by_type = {
        kind: unified_metrics([row for row in rows if row["question_type"] == kind])
        for kind in ("single", "multiple")
    }
    by_category = {
        category: unified_metrics([row for row in rows if row["category"] == category])
        for category in official_eval.EXPECTED_CATEGORY_COUNTS
    }
    format_diagnostics = {}
    for condition in ("clean", "perturbed"):
        format_diagnostics[condition] = {
            "parse_failure_count": sum(not row[condition]["parse_ok"] for row in rows),
            "parse_failure_rate": sum(not row[condition]["parse_ok"] for row in rows)
            / len(rows),
            "strict_format_count": sum(row[condition]["strict_format"] for row in rows),
            "strict_format_rate": sum(row[condition]["strict_format"] for row in rows)
            / len(rows),
        }
    return {
        "unified_metrics": overall,
        "paired_state_counts": paired_state_counts,
        "confidence_intervals_95": cluster_bootstrap_intervals(rows),
        "by_question_type": by_type,
        "by_category": by_category,
        "format_diagnostics": format_diagnostics,
    }


def validate_suite(path: Path) -> Mapping[str, Any]:
    suite = _load_json(path)
    entries = suite.get("models", [])
    labels = tuple(entry.get("label") for entry in entries)
    if labels != EXPECTED_LABELS:
        raise ValueError(f"expected model order {EXPECTED_LABELS}, got {labels}")
    expected_fingerprint = official_eval.canonical_hash(
        [(entry["label"], entry["model_fingerprint"]) for entry in entries]
    )
    if suite.get("suite_fingerprint") != expected_fingerprint:
        raise ValueError(f"invalid suite_fingerprint in {path}")
    for entry in entries:
        model_path = Path(str(entry["canonical_path"]))
        identity_files = official_eval.validate_model_directory(
            model_path, require_export_manifest=entry["label"] != "frozen"
        )
        current_names = {
            item.relative_to(model_path.resolve()).as_posix() for item in identity_files
        }
        frozen_names = {str(item["name"]) for item in entry.get("inventory", [])}
        if current_names != frozen_names:
            raise ValueError(
                f"model identity inventory changed for {entry['label']}: "
                f"current={sorted(current_names)}, frozen={sorted(frozen_names)}"
            )
        official_eval.validate_frozen_model(entry)
    return suite


def reuse_suite(source: Path, destination: Path) -> Mapping[str, Any]:
    suite = validate_suite(source)
    source_sha = official_eval.sha256_file(source)
    destination = destination.resolve()
    if destination.is_file():
        if official_eval.sha256_file(destination) != source_sha:
            raise ValueError(
                f"destination suite differs from source: {destination}; use a new root"
            )
    else:
        _atomic_bytes(destination, source.read_bytes())
    provenance = {
        "schema_version": "tcmeval-pa-controlled-suite-provenance-v1",
        "source_path": str(source.resolve()),
        "source_sha256": source_sha,
        "destination_path": str(destination),
        "destination_sha256": official_eval.sha256_file(destination),
        "suite_fingerprint": suite["suite_fingerprint"],
        "validation": (
            "labels, internal suite fingerprint, READY/export manifests, identity-file "
            "inventory, canonical paths, file sizes, and mtimes"
        ),
        "validated_at_utc": official_eval.utc_now(),
    }
    official_eval.atomic_json(destination.parent / "model_suite_source.json", provenance)
    return provenance


def _load_suite_entry(suite_path: Path, label: str) -> Mapping[str, Any]:
    suite = validate_suite(suite_path)
    matches = [entry for entry in suite["models"] if entry["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"model {label!r} is absent or duplicated")
    return matches[0]


def _completed_valid(
    output_dir: Path,
    questions: Sequence[official_eval.Question],
    run_manifest: Mapping[str, Any],
) -> bool:
    required = (
        output_dir / "COMPLETE.json",
        output_dir / "summary.json",
        output_dir / "paired_predictions.jsonl",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        completion = _load_json(required[0])
        summary = _load_json(required[1])
        rows = _read_jsonl(required[2])
    except (OSError, json.JSONDecodeError):
        return False
    expected_ids = {question.id for question in questions}
    actual_ids = [str(row.get("id")) for row in rows]
    fingerprints = run_manifest["fingerprints"]
    try:
        recomputed_summary = summarize_pairs(rows)
    except (KeyError, TypeError, ValueError):
        return False
    artifact_paths = {
        "paired_predictions.jsonl": required[2],
        "summary.json": required[1],
        "run_manifest.json": output_dir / "run_manifest.json",
    }
    artifact_hashes = completion.get("artifact_sha256")
    return bool(
        completion.get("status") == "complete"
        and completion.get("run_fingerprint") == run_manifest["run_fingerprint"]
        and summary.get("run_fingerprint") == run_manifest["run_fingerprint"]
        and summary.get("fingerprints") == fingerprints
        # JSON artifacts are written with sort_keys=True, so mapping order is
        # a serialization detail. Validate the exact field set, not insertion
        # order, while still rejecting missing or additional metrics.
        and isinstance(summary.get("unified_metrics"), Mapping)
        and set(summary["unified_metrics"]) == set(UNIFIED_FIELDS)
        and all(summary.get(key) == value for key, value in recomputed_summary.items())
        and summary.get("unified_metrics", {}).get("paired_units")
        == official_eval.EXPECTED_QUESTIONS
        and len(rows) == official_eval.EXPECTED_QUESTIONS
        and len(set(actual_ids)) == official_eval.EXPECTED_QUESTIONS
        and set(actual_ids) == expected_ids
        and all(row.get("run_fingerprint") == run_manifest["run_fingerprint"] for row in rows)
        and all(
            all(row.get(key) == value for key, value in fingerprints.items())
            for row in rows
        )
        and isinstance(artifact_hashes, Mapping)
        and all(
            artifact_hashes.get(name) == official_eval.sha256_file(path)
            for name, path in artifact_paths.items()
        )
    )


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    questions = official_eval.load_questions(args.data)
    suite = validate_suite(args.suite)
    model_entry = _load_suite_entry(args.suite, args.model_label)
    lock_identity, _ = _validate_preregistration_lock(args.preregistration_lock)
    _assert_suite_preregistered(suite, lock_identity)
    mapping = freeze_mapping_manifest(
        args.data, args.output_root, args.preregistration_lock
    )
    mapping_path = args.output_root.resolve() / "mapping_manifest.json"
    _, schema_sha, _, analysis_sha = _freeze_schema_snapshots(args.output_root)
    lock_identity, lock_sha = _validate_preregistration_lock(args.preregistration_lock)
    _assert_suite_preregistered(suite, lock_identity)
    p_spec = official_eval.prompt_spec("label_blind")
    g_spec = generation_spec(args.dtype, args.max_new_tokens)
    runtime = runtime_identity()
    fingerprints = {
        "dataset_fingerprint": official_eval.EXPECTED_DATA_SHA256,
        "suite_fingerprint": str(suite["suite_fingerprint"]),
        "prompt_fingerprint": official_eval.canonical_hash(p_spec),
        "generation_fingerprint": official_eval.canonical_hash(g_spec),
        "model_fingerprint": str(model_entry["model_fingerprint"]),
        "mapping_fingerprint": str(mapping["mapping_fingerprint"]),
        "mapping_manifest_sha256": official_eval.sha256_file(mapping_path),
        "metric_schema_sha256": schema_sha,
        "analysis_spec_sha256": analysis_sha,
        "preregistration_lock_sha256": lock_sha,
    }
    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "model_label": args.model_label,
        "model_path": model_entry["canonical_path"],
        "prompt_mode": "label_blind",
        "conditions": ["clean", "perturbed_nonidentity_option_permutation"],
        **_execution_contract(args),
        "fingerprints": fingerprints,
        "evaluator_sha256": official_eval.sha256_file(Path(__file__).resolve()),
        "runtime_identity": runtime,
    }
    run_fingerprint = official_eval.canonical_hash(contract)
    output_dir = args.output_root.resolve() / "paired_label_blind" / args.model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"run contract changed for {output_dir}; use a new output root"
            )
    else:
        manifest = {
            **contract,
            "run_fingerprint": run_fingerprint,
            "created_at_utc": official_eval.utc_now(),
            "dataset": {
                "path": str(args.data.resolve()),
                "sha256": official_eval.EXPECTED_DATA_SHA256,
                "questions": official_eval.EXPECTED_QUESTIONS,
                "normalized_stem_clusters": len(
                    {normalized_stem(question.question) for question in questions}
                ),
                "figshare_doi": official_eval.FIGSHARE_DOI,
                "tcmeval_repo_commit": official_eval.TCMEVAL_REPO_COMMIT,
            },
            "prompt": p_spec,
            "generation": g_spec,
            "mapping_manifest_path": str(mapping_path),
            "metric_schema_path": str(
                args.output_root.resolve() / "UNIFIED_METRIC_SCHEMA.json"
            ),
            "analysis_spec_path": str(
                args.output_root.resolve() / "ANALYSIS_SPEC.json"
            ),
            "preregistration_lock_path": str(args.preregistration_lock.resolve()),
            "model": model_entry,
        }
        official_eval.atomic_json(manifest_path, manifest)
    if _completed_valid(output_dir, questions, manifest):
        print(f"[skip] validated complete controlled result: {output_dir}", flush=True)
        return _load_json(output_dir / "summary.json")

    (output_dir / "COMPLETE.json").unlink(missing_ok=True)
    mapping_by_id = {str(item["id"]): item for item in mapping["items"]}
    if set(mapping_by_id) != {question.id for question in questions}:
        raise ValueError("mapping manifest does not cover the benchmark exactly")
    shard_root = output_dir / "shards"
    shard_root.mkdir(exist_ok=True)
    partitions = [questions[index :: args.num_workers] for index in range(args.num_workers)]
    context = mp.get_context("spawn")
    processes = []
    for worker_id, (device, partition) in enumerate(
        zip(args.devices, partitions, strict=True)
    ):
        error_path = shard_root / f"worker_{worker_id:02d}.error.json"
        error_path.unlink(missing_ok=True)
        process = context.Process(
            target=_worker,
            args=(
                worker_id,
                device,
                partition,
                mapping_by_id,
                shard_root / f"worker_{worker_id:02d}.jsonl",
                shard_root / f"worker_{worker_id:02d}.complete.json",
                error_path,
                model_entry,
                args.max_new_tokens,
                args.dtype,
                run_fingerprint,
                fingerprints,
            ),
            name=f"tcmeval-pa-controlled-{args.model_label}-{worker_id}",
        )
        processes.append(process)

    old_handlers: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise EvaluationInterrupted(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupt)
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            for process in processes:
                process.join(timeout=0.5)
        failures = [process for process in processes if process.exitcode != 0]
        if failures:
            diagnostics = []
            for process in failures:
                worker_id = processes.index(process)
                error_path = shard_root / f"worker_{worker_id:02d}.error.json"
                diagnostics.append(
                    error_path.read_text(encoding="utf-8")
                    if error_path.is_file()
                    else f"{process.name}: exitcode={process.exitcode}"
                )
            raise RuntimeError("worker failure(s): " + " | ".join(diagnostics))
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=30)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    merged: list[dict[str, Any]] = []
    for worker_id, partition in enumerate(partitions):
        completion = shard_root / f"worker_{worker_id:02d}.complete.json"
        if not completion.is_file():
            raise RuntimeError(f"missing worker completion sentinel: {completion}")
        expected = {question.index: question.id for question in partition}
        merged.extend(
            _recover_shard(
                shard_root / f"worker_{worker_id:02d}.jsonl",
                expected,
                run_fingerprint,
            )
        )
    merged.sort(key=lambda row: int(row["index"]))
    ids = [str(row["id"]) for row in merged]
    if (
        len(merged) != official_eval.EXPECTED_QUESTIONS
        or len(set(ids)) != official_eval.EXPECTED_QUESTIONS
    ):
        raise RuntimeError("merged paired predictions are incomplete or duplicated")
    official_eval.atomic_jsonl(output_dir / "paired_predictions.jsonl", merged)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "model_label": args.model_label,
        "protocol": "controlled_label_blind_paired",
        "run_fingerprint": run_fingerprint,
        "fingerprints": fingerprints,
        "completed_at_utc": official_eval.utc_now(),
        **summarize_pairs(merged),
    }
    official_eval.atomic_json(output_dir / "summary.json", summary)
    artifact_paths = {
        "paired_predictions.jsonl": output_dir / "paired_predictions.jsonl",
        "summary.json": output_dir / "summary.json",
        "run_manifest.json": output_dir / "run_manifest.json",
    }
    official_eval.atomic_json(
        output_dir / "COMPLETE.json",
        {
            "status": "complete",
            "paired_rows": official_eval.EXPECTED_QUESTIONS,
            "unique_ids": official_eval.EXPECTED_QUESTIONS,
            "run_fingerprint": run_fingerprint,
            "fingerprints": fingerprints,
            "artifact_sha256": {
                name: official_eval.sha256_file(path)
                for name, path in artifact_paths.items()
            },
            "completed_at_utc": official_eval.utc_now(),
        },
    )
    return summary


def _paired_comparison(
    care_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    baseline_label: str,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Mapping[str, Any]:
    care_by_id = {str(row["id"]): row for row in care_rows}
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    if set(care_by_id) != set(baseline_by_id):
        raise ValueError(f"CARE and {baseline_label} IDs differ")
    care_clusters = _cluster_counts(care_rows)
    baseline_clusters = _cluster_counts(baseline_rows)
    clusters = sorted(care_clusters)
    if clusters != sorted(baseline_clusters):
        raise ValueError(f"CARE and {baseline_label} normalized-stem clusters differ")
    samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    keys = (
        "perturbed_task_success_rate",
        "conditional_retention",
        "regression_rate",
    )
    for _ in range(replicates):
        care_aggregate = [0] * 7
        baseline_aggregate = [0] * 7
        for cluster in rng.choices(clusters, k=len(clusters)):
            for index, value in enumerate(care_clusters[cluster]):
                care_aggregate[index] += value
            for index, value in enumerate(baseline_clusters[cluster]):
                baseline_aggregate[index] += value
        care_metric = _metrics_from_counts(care_aggregate)
        baseline_metric = _metrics_from_counts(baseline_aggregate)
        for key in keys:
            samples[key].append(care_metric[key] - baseline_metric[key])
    care_observed = unified_metrics(care_rows)
    baseline_observed = unified_metrics(baseline_rows)
    intervals = {
        key: {
            "delta": care_observed[key] - baseline_observed[key],
            "low": _percentile(samples[key], 0.025),
            "high": _percentile(samples[key], 0.975),
        }
        for key in keys
    }
    gate_endpoints = (
        "perturbed_task_success_rate",
        "conditional_retention",
    )
    noninferiority = {
        key: {
            "margin": NONINFERIORITY_MARGIN,
            "ci95_low": intervals[key]["low"],
            "pass": bool(intervals[key]["low"] > NONINFERIORITY_MARGIN),
        }
        for key in gate_endpoints
    }
    mcnemar = official_eval._mcnemar_exact(
        {item: bool(row["perturbed"]["task_success"]) for item, row in care_by_id.items()},
        {
            item: bool(row["perturbed"]["task_success"])
            for item, row in baseline_by_id.items()
        },
    )
    return {
        "baseline": baseline_label,
        "effect_direction": (
            "CARE-minus-baseline; positive favors CARE except regression_rate, "
            "where negative favors CARE"
        ),
        "paired_units": len(care_by_id),
        "cluster_count": len(clusters),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "delta_and_95_ci": intervals,
        "noninferiority": {
            "margin_absolute": NONINFERIORITY_MARGIN,
            "endpoints": noninferiority,
            "joint_pass": all(item["pass"] for item in noninferiority.values()),
            "interpretation": (
                "Both CARE-minus-baseline paired cluster-bootstrap lower bounds must "
                "exceed -2 percentage points; passing is preservation, not superiority."
            ),
        },
        "exploratory_exact_mcnemar_perturbed_task_success_rate": mcnemar,
        "mcnemar_note": "Exploratory only; no multiple-comparison correction.",
    }


def _load_official_original_scores(
    official_root: Path,
    suite: Mapping[str, Any],
    questions: Sequence[official_eval.Question],
) -> list[Mapping[str, Any]]:
    official_root = official_root.resolve()
    records = []
    for entry in suite["models"]:
        label = str(entry["label"])
        model_root = official_root / label
        manifest_path = model_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing read-only official result: {manifest_path}")
        manifest = _load_json(manifest_path)
        if manifest.get("model_label") != label:
            raise ValueError(f"model label mismatch for {label}")
        if manifest.get("model_path") != entry.get("canonical_path"):
            raise ValueError(f"model path mismatch for {label}")
        if manifest.get("fingerprints", {}).get("model_fingerprint") != entry.get(
            "model_fingerprint"
        ):
            raise ValueError(f"model fingerprint mismatch for {label}")
        if manifest.get("prompt_mode") != "official":
            raise ValueError(f"{manifest_path} is not an official-prompt run")
        if manifest.get("fingerprints", {}).get("model_fingerprint") != entry[
            "model_fingerprint"
        ]:
            raise ValueError(f"official/controlled model fingerprint mismatch for {label}")
        if not official_eval._completed_valid(model_root, questions, manifest):
            raise ValueError(f"official result failed read-only validation for {label}")
        summary_path = model_root / "summary.json"
        summary = _load_json(summary_path)
        overall = summary["overall"]
        records.append(
            {
                "model": label,
                "protocol": "official_original",
                "n": overall["n"],
                "option_set_exact": overall["option_set_exact"],
                "option_set_accuracy": overall["option_set_accuracy"],
                "single_accuracy": summary["by_question_type"]["single"][
                    "option_set_accuracy"
                ],
                "multiple_accuracy": summary["by_question_type"]["multiple"][
                    "option_set_accuracy"
                ],
                "strict_format_rate": overall["strict_format_rate"],
                "parse_failure_rate": overall["parse_failure_rate"],
                "source_summary_sha256": official_eval.sha256_file(summary_path),
            }
        )
    return records


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        official_eval.atomic_text(path, "")
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    official_eval.atomic_text(path, stream.getvalue())


def _controlled_csv_record(label: str, summary: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = summary["unified_metrics"]
    ci = summary["confidence_intervals_95"]["intervals"]
    states = summary["paired_state_counts"]
    return {
        "model": label,
        "paired_units": metrics["paired_units"],
        "clean_task_success_rate": metrics["clean_task_success_rate"],
        "clean_task_success_rate_ci_low": ci["clean_task_success_rate"]["low"],
        "clean_task_success_rate_ci_high": ci["clean_task_success_rate"]["high"],
        "perturbed_task_success_rate": metrics["perturbed_task_success_rate"],
        "perturbed_task_success_rate_ci_low": ci[
            "perturbed_task_success_rate"
        ]["low"],
        "perturbed_task_success_rate_ci_high": ci[
            "perturbed_task_success_rate"
        ]["high"],
        "conditional_retention": metrics["conditional_retention"],
        "conditional_retention_ci_low": ci["conditional_retention"]["low"],
        "conditional_retention_ci_high": ci["conditional_retention"]["high"],
        "regression_rate": metrics["regression_rate"],
        "regression_rate_ci_low": ci["regression_rate"]["low"],
        "regression_rate_ci_high": ci["regression_rate"]["high"],
        "both_correct": states["both_correct"],
        "clean_correct_perturbed_wrong": metrics[
            "clean_correct_perturbed_wrong"
        ],
        "clean_wrong_perturbed_correct": metrics[
            "clean_wrong_perturbed_correct"
        ],
        "both_wrong": states["both_wrong"],
        "robustness_drop_pp": metrics["robustness_drop_pp"],
        "prediction_consistency": metrics["prediction_consistency"],
        "worst_perturbation_tsr": metrics["worst_perturbation_tsr"],
    }


def _write_runtime_readme(output_root: Path) -> None:
    value = """# TCMEval-PA controlled robustness v1

This directory is an independent, paired follow-up. It does not modify the
official TCMEval-PA result root. `official_original_scores.*` is a validated,
read-only import; it is never merged into the controlled metrics.

Both controlled conditions use one label-blind prompt, greedy decoding,
`enable_thinking=False`, and all 328 official IDs. The perturbed condition uses
the frozen `mapping_manifest.json`: every ID receives a deterministic,
non-identity option permutation. Predictions are mapped back to canonical
labels before exact option-set scoring.

Primary effectiveness: `perturbed_task_success_rate`. Primary robustness:
`conditional_retention` and `regression_rate`. `robustness_drop_pp`,
`prediction_consistency`, and the four paired state counts are diagnostics. Confidence intervals and CARE-vs-
Clean-SFT/GKD deltas use a fixed 10,000-replicate percentile bootstrap over
314 normalized-stem clusters. Exact McNemar tests are exploratory.

Important files:

* `UNIFIED_METRIC_SCHEMA.json`: frozen definitions and directions;
* `ANALYSIS_SPEC.json`: TCMEval-specific perturbation and bootstrap contract;
* `mapping_manifest.json`: all 328 fixed mappings and mapping fingerprint;
* `model_suite.json`: validated copy of the original six-model suite;
* `paired_label_blind/<model>/paired_predictions.jsonl`: item-level pairs;
* `suite_summary.csv`: controlled metrics only;
* `official_original_scores.csv`: original official-prompt metrics only;
* `paired_comparisons.csv`: CARE paired cluster-bootstrap comparisons.
"""
    official_eval.atomic_text(output_root.resolve() / "README.md", value)


def summarize_suite(args: argparse.Namespace) -> Mapping[str, Any]:
    questions = official_eval.load_questions(args.data)
    suite = validate_suite(args.suite)
    lock_identity, lock_sha = _validate_preregistration_lock(args.preregistration_lock)
    _assert_suite_preregistered(suite, lock_identity)
    _, schema_sha, _, analysis_sha = _freeze_schema_snapshots(args.output_root)
    mapping = freeze_mapping_manifest(
        args.data, args.output_root, args.preregistration_lock
    )
    controlled_records = []
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for entry in suite["models"]:
        label = str(entry["label"])
        model_root = args.output_root.resolve() / "paired_label_blind" / label
        manifest_path = model_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing controlled run: {manifest_path}")
        manifest = _load_json(manifest_path)
        if manifest.get("model_label") != label:
            raise ValueError(f"controlled model label mismatch for {label}")
        if manifest.get("model_path") != entry.get("canonical_path"):
            raise ValueError(f"controlled model path mismatch for {label}")
        if manifest.get("fingerprints", {}).get("model_fingerprint") != entry.get(
            "model_fingerprint"
        ):
            raise ValueError(f"controlled model fingerprint mismatch for {label}")
        if manifest.get("fingerprints", {}).get("preregistration_lock_sha256") != lock_sha:
            raise ValueError(f"preregistration lock mismatch for {label}")
        if manifest.get("fingerprints", {}).get("metric_schema_sha256") != schema_sha:
            raise ValueError(f"metric schema mismatch for {label}")
        if manifest.get("fingerprints", {}).get("analysis_spec_sha256") != analysis_sha:
            raise ValueError(f"analysis spec mismatch for {label}")
        if manifest.get("fingerprints", {}).get("mapping_fingerprint") != mapping[
            "mapping_fingerprint"
        ]:
            raise ValueError(f"mapping mismatch for {label}")
        if not _completed_valid(model_root, questions, manifest):
            raise ValueError(f"controlled result failed validation for {label}")
        summary = _load_json(model_root / "summary.json")
        rows = _read_jsonl(model_root / "paired_predictions.jsonl")
        rows_by_model[label] = rows
        controlled_records.append(_controlled_csv_record(label, summary))

    official_records = _load_official_original_scores(
        args.official_root, suite, questions
    )
    comparisons = [
        _paired_comparison(
            rows_by_model["care_opd"], rows_by_model[baseline], baseline
        )
        for baseline in PRIMARY_COMPARISONS
    ]
    output_root = args.output_root.resolve()
    _write_csv(output_root / "suite_summary.csv", controlled_records)
    _write_csv(output_root / "official_original_scores.csv", official_records)
    comparison_rows = []
    for comparison in comparisons:
        record: dict[str, Any] = {
            "baseline": comparison["baseline"],
            "paired_units": comparison["paired_units"],
            "cluster_count": comparison["cluster_count"],
        }
        for metric, interval in comparison["delta_and_95_ci"].items():
            record[f"{metric}_delta"] = interval["delta"]
            record[f"{metric}_ci_low"] = interval["low"]
            record[f"{metric}_ci_high"] = interval["high"]
        record["noninferiority_margin_absolute"] = comparison["noninferiority"][
            "margin_absolute"
        ]
        record["joint_primary_noninferiority_pass"] = comparison[
            "noninferiority"
        ]["joint_pass"]
        for metric, gate in comparison["noninferiority"]["endpoints"].items():
            record[f"{metric}_noninferiority_pass"] = gate["pass"]
        mcnemar = comparison[
            "exploratory_exact_mcnemar_perturbed_task_success_rate"
        ]
        record["care_correct_baseline_wrong"] = mcnemar[
            "care_correct_baseline_wrong"
        ]
        record["baseline_correct_care_wrong"] = mcnemar[
            "baseline_correct_care_wrong"
        ]
        record["mcnemar_two_sided_exact_p"] = mcnemar["two_sided_exact_p"]
        comparison_rows.append(record)
    _write_csv(output_root / "paired_comparisons.csv", comparison_rows)

    payload = {
        "schema_version": SUITE_SUMMARY_SCHEMA_VERSION,
        "benchmark": "TCMEval-PA v3",
        "dataset_sha256": official_eval.EXPECTED_DATA_SHA256,
        "suite_fingerprint": suite["suite_fingerprint"],
        "metric_schema_sha256": schema_sha,
        "analysis_spec_sha256": analysis_sha,
        "mapping_fingerprint": mapping["mapping_fingerprint"],
        "mapping_manifest_sha256": official_eval.sha256_file(
            output_root / "mapping_manifest.json"
        ),
        "preregistration_lock_sha256": lock_sha,
        "controlled_derived_results": {
            "protocol": "label_blind clean/permuted pair; no-think greedy",
            "primary_effectiveness": "perturbed_task_success_rate",
            "primary_robustness": ["conditional_retention", "regression_rate"],
            "diagnostics": [
                "robustness_drop_pp",
                "prediction_consistency",
                "worst_perturbation_tsr",
                "clean_correct_perturbed_wrong",
                "clean_wrong_perturbed_correct",
            ],
            "models": controlled_records,
            "care_paired_comparisons": comparisons,
        },
        "official_original_results": {
            "protocol": "official prompt; read-only validated import",
            "source_root": str(args.official_root.resolve()),
            "never_used_as_controlled_or_derived_score": True,
            "models": official_records,
        },
        "created_at_utc": official_eval.utc_now(),
    }
    official_eval.atomic_json(output_root / "suite_summary.json", payload)

    headers = list(controlled_records[0])
    markdown = [
        "# TCMEval-PA controlled robustness v1",
        "",
        "## Controlled derived results (label-blind paired protocol)",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for record in controlled_records:
        values = []
        for key in headers:
            value = record[key]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        markdown.append("| " + " | ".join(values) + " |")
    markdown.extend(
        [
            "",
            "Primary effectiveness is `perturbed_task_success_rate`; primary robustness is "
            "`conditional_retention` / `regression_rate`. Intervals are clustered by "
            "normalized stem (314 clusters), not by treating all 328 IDs as independent.",
            "The preregistered CARE-versus-clean-SFT/GKD joint preservation margin is "
            "-2 percentage points on perturbed TSR and conditional retention. Both paired "
            "cluster-bootstrap lower bounds must pass; passing is not superiority.",
            "",
            "## Official original results (read-only; separate protocol)",
            "",
            "These values are copied only after validating the original completion and "
            "fingerprints. They are not controlled scores and are not used to derive "
            "retention or regression.",
            "",
            "| model | n | exact | accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in official_records:
        markdown.append(
            f"| {record['model']} | {record['n']} | {record['option_set_exact']} | "
            f"{record['option_set_accuracy']:.6f} |"
        )
    official_eval.atomic_text(output_root / "suite_summary.md", "\n".join(markdown) + "\n")
    _write_runtime_readme(output_root)
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    suite_parser = commands.add_parser("reuse-suite")
    suite_parser.add_argument("--source", type=Path, required=True)
    suite_parser.add_argument("--destination", type=Path, required=True)

    mapping_parser = commands.add_parser("freeze-mapping")
    mapping_parser.add_argument("--data", type=Path, required=True)
    mapping_parser.add_argument("--output-root", type=Path, required=True)
    mapping_parser.add_argument(
        "--preregistration-lock",
        type=Path,
        default=DEFAULT_PREREGISTRATION_LOCK,
    )

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--data", type=Path, required=True)
    evaluate_parser.add_argument("--suite", type=Path, required=True)
    evaluate_parser.add_argument("--model-label", choices=EXPECTED_LABELS, required=True)
    evaluate_parser.add_argument("--output-root", type=Path, required=True)
    evaluate_parser.add_argument(
        "--preregistration-lock",
        type=Path,
        default=DEFAULT_PREREGISTRATION_LOCK,
    )
    evaluate_parser.add_argument("--dtype", default="bfloat16")
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=8)
    evaluate_parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    evaluate_parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))

    summary_parser = commands.add_parser("summarize-suite")
    summary_parser.add_argument("--data", type=Path, required=True)
    summary_parser.add_argument("--suite", type=Path, required=True)
    summary_parser.add_argument("--output-root", type=Path, required=True)
    summary_parser.add_argument("--official-root", type=Path, required=True)
    summary_parser.add_argument(
        "--preregistration-lock",
        type=Path,
        default=DEFAULT_PREREGISTRATION_LOCK,
    )
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "reuse-suite":
        result = reuse_suite(args.source, args.destination)
    elif args.command == "freeze-mapping":
        result = freeze_mapping_manifest(
            args.data, args.output_root, args.preregistration_lock
        )
    elif args.command == "evaluate":
        if args.max_new_tokens != 8:
            raise ValueError("formal controlled protocol fixes --max-new-tokens=8")
        _normalize_execution_topology(args)
        result = evaluate(args)
    elif args.command == "summarize-suite":
        result = summarize_suite(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
