#!/usr/bin/env python3
"""Contamination-aware, resumable six-checkpoint MTCMB evaluator.

Formal reporting is deliberately split into MTCMB-Full (in-source diagnostic),
an item-level overlap-clean view, and a strict task-family provenance-clean
Clean9 view.  TCM-SE-A is never silently approximated: its upstream GLM judge
requirement is recorded as not available in this all-local protocol.
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
import signal
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
PUBLIC_EVAL_ROOT = HERE.parent
PROJECT_ROOT = PUBLIC_EVAL_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PUBLIC_EVAL_ROOT))
import protocol  # noqa: E402
from preregistration import validate_static_lock  # noqa: E402
from runtime_identity import runtime_identity  # noqa: E402


DEFAULT_EXCLUSIONS = HERE / "overlap_exclusions.json"
DEFAULT_TASK_MANIFEST = HERE / "task_manifest.json"
DEFAULT_SCHEMA = PUBLIC_EVAL_ROOT / "UNIFIED_METRIC_SCHEMA.json"
DEFAULT_LOCK = PUBLIC_EVAL_ROOT / "PREREGISTRATION_LOCK.json"

REQUIRED_LOCK_PATHS = (
    "src/care_opd/models.py",
    "public_eval/create_preregistration_lock.py",
    "public_eval/GENERALIZATION_PROTOCOL.md",
    "public_eval/preregistration.py",
    "public_eval/runtime_identity.py",
    "public_eval/UNIFIED_METRIC_SCHEMA.json",
    "public_eval/PUBLIC_GENERALIZATION_PREREGISTRATION.md",
    "public_eval/mtcmb_clean/protocol.py",
    "public_eval/mtcmb_clean/prepare_data.py",
    "public_eval/mtcmb_clean/evaluate_mtcmb.py",
    "public_eval/mtcmb_clean/run_existing_gpu4_5.sh",
    "public_eval/mtcmb_clean/overlap_exclusions.json",
    "public_eval/mtcmb_clean/task_manifest.json",
)
METRIC_PACKAGE_VERSIONS = {
    "bert-score": "0.3.13",
    "jieba": "0.42.1",
    "rouge-chinese": "1.0.3",
    "sacrebleu": "2.5.1",
}
PAIR_BOOTSTRAP_SEED = 20260809
PAIR_BOOTSTRAP_REPLICATES = 10_000
NONINFERIORITY_MARGIN = -0.05
FORMAL_NUM_WORKERS = 4
FORMAL_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")


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


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized),
    )
    return len(materialized)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_spec() -> dict[str, Any]:
    return {
        "mode": "upstream_repository_zero_shot_with_official_remote_chat_system",
        "upstream_commit": protocol.UPSTREAM_COMMIT,
        "system_prompt": protocol.SYSTEM_PROMPT,
        "enable_thinking": False,
        "tcm_ft_exception": {
            "reason": "upstream builder references scripts/prompts/tcm_qa_points_output.yaml, absent at pinned commit",
            "fallback_version": protocol.FT_FALLBACK_PROMPT_VERSION,
        },
        "litdata_expansion": "one independent upstream-style call per annotation",
    }


def generation_spec(dtype: str, batch_size: int) -> dict[str, Any]:
    return {
        "dtype": dtype,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "batch_size": batch_size,
        "max_new_tokens": {
            "mcq": 8,
            "litdata_annotation": 128,
            "other_open_tasks": 768,
        },
    }


def validate_metric_packages() -> dict[str, str]:
    actual = {}
    for distribution, expected in METRIC_PACKAGE_VERSIONS.items():
        try:
            value = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"missing MTCMB metric package {distribution}=={expected}"
            ) from exc
        if value != expected:
            raise RuntimeError(
                f"MTCMB metric package drift: {distribution}=={value}, expected {expected}"
            )
        actual[distribution] = value
    return actual


def validate_protocol_assets(
    mtcmb_repo: Path,
    exclusions_path: Path,
    task_manifest_path: Path,
    schema_path: Path,
    *,
    verify_git: bool = True,
) -> dict[str, Any]:
    upstream = protocol.verify_upstream_repo(mtcmb_repo, verify_git=verify_git)
    exclusions = protocol.load_exclusions(exclusions_path)
    task_manifest = load_json(task_manifest_path.resolve(strict=True))
    if task_manifest.get("schema_version") != "care-opd-mtcmb-task-manifest-v1":
        raise ValueError("invalid MTCMB task manifest schema")
    if task_manifest.get("upstream", {}).get("commit") != protocol.UPSTREAM_COMMIT:
        raise ValueError("MTCMB task manifest commit changed")
    if task_manifest.get("protocol_version") != protocol.PROTOCOL_VERSION:
        raise ValueError("MTCMB task manifest protocol changed")
    expected_views = {
        "mtcmb_full": (protocol.EXPECTED_FULL_N, protocol.EXPECTED_OFFLINE11_N),
        "overlap_clean": (
            protocol.EXPECTED_OVERLAP_CLEAN_ALL12_N,
            protocol.EXPECTED_OVERLAP_CLEAN_OFFLINE11_N,
        ),
        "provenance_clean9": (
            protocol.EXPECTED_CLEAN9_N,
            protocol.EXPECTED_CLEAN9_N,
        ),
    }
    for view, (records, offline) in expected_views.items():
        value = task_manifest.get("views", {}).get(view, {})
        if (value.get("records"), value.get("offline_scored_records")) != (
            records,
            offline,
        ):
            raise ValueError(f"MTCMB task manifest view changed: {view}")
    task_values = task_manifest.get("tasks") or {}
    if {
        task: int(value.get("records", -1)) for task, value in task_values.items()
    } != protocol.TASK_COUNTS:
        raise ValueError("MTCMB task manifest task inventory changed")
    if [task for task, value in task_values.items() if value.get("clean9")] != list(
        protocol.CLEAN9_TASKS
    ):
        raise ValueError("MTCMB task manifest Clean9 inventory changed")
    controlled = task_manifest.get("controlled_robustness") or {}
    if (
        controlled.get("overlap_clean_mcq_pairs")
        != protocol.EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS
        or controlled.get("provenance_clean9_mcq_pairs")
        != protocol.EXPECTED_CLEAN9_MCQ_PAIRS
    ):
        raise ValueError("MTCMB controlled-robustness manifest changed")
    uncertainty = controlled.get("strict_clean9_rate_uncertainty") or {}
    if uncertainty.get("method") != "two-sided Wilson score interval" or uncertainty.get(
        "confidence_level"
    ) != 0.95:
        raise ValueError("MTCMB uncertainty contract changed")
    between_model = controlled.get("strict_clean9_between_model_noninferiority") or {}
    if (
        between_model.get("primary_baselines") != ["clean_sft", "gkd_full"]
        or between_model.get("endpoints")
        != ["perturbed_task_success_rate", "conditional_retention"]
        or between_model.get("replicates") != PAIR_BOOTSTRAP_REPLICATES
        or between_model.get("seed") != PAIR_BOOTSTRAP_SEED
        or between_model.get("margin_absolute") != NONINFERIORITY_MARGIN
    ):
        raise ValueError("MTCMB between-model noninferiority contract changed")
    if task_manifest.get("reporting", {}).get(
        "sample_micro_average_across_tasks_forbidden"
    ) is not True:
        raise ValueError("MTCMB cross-task aggregation prohibition changed")
    schema = load_json(schema_path.resolve(strict=True))
    required_metric_fields = set(schema.get("required_fields") or {})
    expected_metric_fields = set(protocol.unified_na())
    if required_metric_fields != expected_metric_fields:
        raise ValueError(
            f"unified metric field contract changed: {sorted(required_metric_fields)}"
        )
    return {
        "upstream": upstream,
        "exclusions_sha256": protocol.sha256_file(exclusions_path),
        "task_manifest_sha256": protocol.sha256_file(task_manifest_path),
        "unified_metric_schema_sha256": protocol.sha256_file(schema_path),
        "dataset_contract_fingerprint": protocol.dataset_contract_fingerprint(
            mtcmb_repo, exclusions_path
        ),
        "exclusions": exclusions,
    }


def _weight_files(model: Path) -> list[Path]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model / index_name
        if index_path.is_file():
            payload = load_json(index_path)
            names = sorted(set(payload.get("weight_map", {}).values()))
            files = [model / name for name in names]
            if not names or not all(path.is_file() for path in files):
                raise ValueError(f"invalid or incomplete weight index: {index_path}")
            return [index_path, *files]
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model / name
        if candidate.is_file():
            return [candidate]
    raise ValueError(f"no Hugging Face weight file or index found in {model}")


def validate_model_directory(path: Path, *, require_export_manifest: bool) -> list[Path]:
    path = path.resolve()
    required = [path / "config.json", path / "tokenizer_config.json"]
    if require_export_manifest:
        required.append(path / "export_manifest.json")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ValueError(f"model is not READY; missing {missing}")
    identity = list(required)
    for pattern in (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "chat_template*",
        "special_tokens_map.json",
        "added_tokens.json",
        "processor_config.json",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        identity.extend(sorted(path.glob(pattern)))
    identity.extend(_weight_files(path))
    return sorted(set(identity), key=lambda item: str(item.relative_to(path)))


def fingerprint_directory(path: Path, *, require_export_manifest: bool = False) -> dict[str, Any]:
    path = path.resolve(strict=True)
    files = validate_model_directory(path, require_export_manifest=require_export_manifest)
    inventory = []
    print(f"[fingerprint] path={path} files={len(files)}", flush=True)
    for index, file_path in enumerate(files, 1):
        stat = file_path.stat()
        relative = file_path.relative_to(path).as_posix()
        print(f"[fingerprint] {index}/{len(files)} hashing {relative}", flush=True)
        inventory.append(
            {
                "name": relative,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": protocol.sha256_file(file_path),
            }
        )
    content = [{key: row[key] for key in ("name", "bytes", "sha256")} for row in inventory]
    return {
        "canonical_path": str(path),
        "require_export_manifest": require_export_manifest,
        "model_fingerprint": protocol.canonical_hash(content),
        "inventory": inventory,
    }


def fingerprint_auxiliary_directory(path: Path) -> dict[str, Any]:
    """Use the same complete inventory contract as the root auxiliary-model lock."""

    root = path.resolve(strict=True)
    files = [
        item
        for item in root.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(root).parts
    ]
    if not files:
        raise ValueError(f"auxiliary scorer has no files: {root}")
    inventory = []
    for file_path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        stat = file_path.stat()
        inventory.append(
            {
                "name": file_path.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": protocol.sha256_file(file_path),
            }
        )
    content = [{key: row[key] for key in ("name", "bytes", "sha256")} for row in inventory]
    return {
        "canonical_path": str(root),
        "require_export_manifest": False,
        "model_fingerprint": protocol.canonical_hash(content),
        "inventory": inventory,
    }


def evaluation_assets_fingerprint(suite: Mapping[str, Any]) -> str:
    models = suite.get("models")
    bert = suite.get("bert_scorer")
    if not isinstance(models, list) or not isinstance(bert, Mapping):
        raise ValueError("MTCMB evaluation asset inventory is incomplete")
    return protocol.canonical_hash(
        [(str(entry["label"]), str(entry["model_fingerprint"])) for entry in models]
        + [("bert_scorer", str(bert["model_fingerprint"]))]
    )


def validate_frozen_directory(entry: Mapping[str, Any]) -> None:
    root = Path(str(entry["canonical_path"])).resolve()
    if str(root) != str(entry["canonical_path"]):
        raise ValueError(f"path is no longer canonical: {entry['canonical_path']}")
    for item in entry.get("inventory", []):
        path = root / str(item["name"])
        if not path.is_file():
            raise ValueError(f"frozen file disappeared: {path}")
        stat = path.stat()
        if stat.st_size != int(item["bytes"]) or stat.st_mtime_ns != int(item["mtime_ns"]):
            raise ValueError(f"frozen artifact changed: {path}")


def freeze_suite(suite_path: Path, model_specs: Sequence[str], bert_model: Path) -> Mapping[str, Any]:
    requested: list[tuple[str, Path]] = []
    for spec in model_specs:
        if "=" not in spec:
            raise ValueError(f"model spec must be LABEL=PATH: {spec!r}")
        label, raw_path = spec.split("=", 1)
        requested.append((label, Path(raw_path).resolve()))
    labels = tuple(label for label, _ in requested)
    if labels != protocol.MODEL_ORDER:
        raise ValueError(f"formal MTCMB suite order must be {protocol.MODEL_ORDER}, got {labels}")
    if len({path for _, path in requested}) != len(requested):
        raise ValueError("two MTCMB labels resolve to one model path")
    bert_model = bert_model.resolve()
    if suite_path.is_file():
        existing = load_json(suite_path)
        if existing.get("schema_version") != "care-opd-mtcmb-model-suite-v1":
            raise ValueError("existing MTCMB suite has an unsupported schema")
        old = [(entry["label"], Path(entry["canonical_path"])) for entry in existing["models"]]
        if old != requested or Path(existing["bert_scorer"]["canonical_path"]) != bert_model:
            raise ValueError("existing frozen suite differs; use a new output root")
        for entry in existing["models"]:
            validate_frozen_directory(entry)
        validate_frozen_directory(existing["bert_scorer"])
        if existing.get("suite_fingerprint") != _suite_model_fingerprint(existing):
            raise ValueError("existing MTCMB model suite fingerprint does not close")
        if existing.get("evaluation_assets_fingerprint") != evaluation_assets_fingerprint(
            existing
        ):
            raise ValueError("existing MTCMB evaluation-assets fingerprint does not close")
        return existing
    entries = []
    for label, path in requested:
        entry = fingerprint_directory(path, require_export_manifest=label != "frozen")
        entry["label"] = label
        entries.append(entry)
    bert_entry = fingerprint_auxiliary_directory(bert_model)
    payload = {
        "schema_version": "care-opd-mtcmb-model-suite-v1",
        "created_at_utc": utc_now(),
        "models": entries,
        "bert_scorer": bert_entry,
        "suite_fingerprint": protocol.canonical_hash(
            [(entry["label"], entry["model_fingerprint"]) for entry in entries]
        ),
    }
    payload["evaluation_assets_fingerprint"] = evaluation_assets_fingerprint(payload)
    atomic_json(suite_path, payload)
    return payload


def load_suite_entry(suite_path: Path, label: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    suite = load_json(suite_path.resolve(strict=True))
    if suite.get("schema_version") != "care-opd-mtcmb-model-suite-v1":
        raise ValueError("unsupported MTCMB suite schema")
    if tuple(entry["label"] for entry in suite["models"]) != protocol.MODEL_ORDER:
        raise ValueError("frozen model suite order changed")
    if suite.get("suite_fingerprint") != _suite_model_fingerprint(suite):
        raise ValueError("frozen MTCMB suite fingerprint does not close")
    if suite.get("evaluation_assets_fingerprint") != evaluation_assets_fingerprint(suite):
        raise ValueError("frozen MTCMB evaluation-assets fingerprint does not close")
    matches = [entry for entry in suite["models"] if entry["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"model {label!r} is absent or duplicated")
    validate_frozen_directory(matches[0])
    validate_frozen_directory(suite["bert_scorer"])
    return suite, matches[0], suite["bert_scorer"]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _recover_shard(
    path: Path, expected: Mapping[str, int], run_fingerprint: str
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw_lines = path.read_bytes().splitlines()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    corrupted_last = False
    for line_index, line in enumerate(raw_lines):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if line_index != len(raw_lines) - 1:
                raise ValueError(f"corrupt non-terminal line in {path}")
            corrupted_last = True
            break
        item_id = str(row.get("item_id") or "")
        if item_id not in expected or row.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"stale/unexpected row in {path}: {item_id}")
        if int(row.get("index", -1)) != int(expected[item_id]):
            raise ValueError(f"index mismatch for {item_id} in {path}")
        if item_id in seen:
            raise ValueError(f"duplicate item {item_id} in {path}")
        seen.add(item_id)
        rows.append(row)
    if corrupted_last:
        atomic_jsonl(path, rows)
    return rows


def _worker(
    worker_id: int,
    device: str,
    indexed_items: Sequence[tuple[int, protocol.InferenceItem]],
    shard_path: Path,
    completion_path: Path,
    error_path: Path,
    model_entry: Mapping[str, Any],
    dtype: str,
    batch_size: int,
    run_fingerprint: str,
    fingerprints: Mapping[str, str],
) -> None:
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from care_opd.models import ModelArguments, load_causal_lm, load_tokenizer
        import torch

        expected = {item.item_id: index for index, item in indexed_items}
        existing = _recover_shard(shard_path, expected, run_fingerprint)
        completed = {str(row["item_id"]) for row in existing}
        remaining = [(index, item) for index, item in indexed_items if item.item_id not in completed]
        if remaining:
            model_path = str(model_entry["canonical_path"])
            print(
                f"[worker] id={worker_id} device={device} loading={model_path} "
                f"remaining={len(remaining)} batch={batch_size}",
                flush=True,
            )
            model_args = ModelArguments(
                student_model_name_or_path=model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
            tokenizer = load_tokenizer(model_args)
            if tokenizer.pad_token_id is None:
                raise RuntimeError("tokenizer has no pad_token_id")
            tokenizer.padding_side = "left"
            model = load_causal_lm(model_path, model_args, trainable=False)
            target_device = torch.device(device)
            model.to(target_device)
            model.eval()
            if hasattr(model, "config"):
                model.config.use_cache = True
            print(f"[worker] id={worker_id} device={device} model_loaded", flush=True)

            cursor = 0
            generated_now = 0
            while cursor < len(remaining):
                token_budget = remaining[cursor][1].max_new_tokens
                end = cursor
                while (
                    end < len(remaining)
                    and end - cursor < batch_size
                    and remaining[end][1].max_new_tokens == token_budget
                ):
                    end += 1
                batch = remaining[cursor:end]
                rendered = []
                for _, item in batch:
                    messages = [
                        {"role": "system", "content": protocol.SYSTEM_PROMPT},
                        {"role": "user", "content": item.prompt},
                    ]
                    try:
                        rendered.append(
                            tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
                        )
                    except TypeError as exc:
                        raise RuntimeError(
                            "chat template lacks enable_thinking=False; refusing protocol fallback"
                        ) from exc
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                encoded = {key: value.to(target_device) for key, value in encoded.items()}
                max_new_tokens = token_budget
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
                latency = time.perf_counter() - started
                prompt_width = encoded["input_ids"].shape[1]
                for (global_index, item), sequence in zip(batch, generated, strict=True):
                    raw_output = tokenizer.decode(
                        sequence[prompt_width:], skip_special_tokens=True
                    ).strip()
                    parsed_prediction = ""
                    strict_format = False
                    if item.task in protocol.MCQ_TASKS:
                        labels = tuple(dict(item.canonical_to_displayed).values())
                        if not labels:
                            # Clean item: infer the inventory from the output contract.
                            labels = tuple("ABCD") if item.task == "TCM-SE-B" else tuple("ABCDE")
                        parsed_prediction = protocol.parse_official_choice(raw_output, labels)
                        _, strict_format = protocol.parse_choice(raw_output, labels)
                    row = {
                        "index": global_index,
                        "item_id": item.item_id,
                        "task": item.task,
                        "record_id": item.record_id,
                        "annotation_index": item.annotation_index,
                        "condition": item.condition,
                        "canonical_to_displayed": dict(item.canonical_to_displayed),
                        "prompt_sha256": hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
                        "raw_output": raw_output,
                        "parsed_prediction": parsed_prediction,
                        "strict_format": strict_format,
                        "batch_latency_seconds": latency,
                        "batch_size_actual": len(batch),
                        "worker_id": worker_id,
                        "device": device,
                        "run_fingerprint": run_fingerprint,
                        **fingerprints,
                    }
                    _append_jsonl(shard_path, row)
                generated_now += len(batch)
                done = len(completed) + generated_now
                if cursor == 0 or done % 100 <= len(batch) or done == len(indexed_items):
                    print(
                        f"[progress] worker={worker_id} {done}/{len(indexed_items)} "
                        f"last={batch[-1][1].item_id}",
                        flush=True,
                    )
                cursor = end
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows = _recover_shard(shard_path, expected, run_fingerprint)
        if len(rows) != len(indexed_items):
            raise RuntimeError(f"worker {worker_id} produced {len(rows)}/{len(indexed_items)}")
        atomic_json(
            completion_path,
            {
                "status": "complete",
                "worker_id": worker_id,
                "rows": len(rows),
                "run_fingerprint": run_fingerprint,
                "completed_at_utc": utc_now(),
            },
        )
        error_path.unlink(missing_ok=True)
    except BaseException as exc:
        atomic_json(
            error_path,
            {
                "worker_id": worker_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _task_lookup(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task, rows in records.items():
        for row in rows:
            key = (task, int(row["id"]))
            if key in result:
                raise ValueError(f"duplicate record key {key}")
            result[key] = row
    return result


def _prediction_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item_id = str(row["item_id"])
        if item_id in result:
            raise ValueError(f"duplicate prediction row {item_id}")
        result[item_id] = row
    return result


def _jieba_text(text: str) -> str:
    import jieba

    return " ".join(jieba.cut(str(text)))


def _lexical_scores(prediction: str, reference: str) -> dict[str, float]:
    import sacrebleu
    from rouge_chinese import Rouge

    if not str(prediction).strip() or not str(reference).strip():
        return {"rouge_1": 0.0, "rouge_l": 0.0, "bleu": 0.0}
    pred_tokens = _jieba_text(prediction)
    ref_tokens = _jieba_text(reference)
    try:
        rouge = Rouge().get_scores(pred_tokens, ref_tokens)[0]
        rouge_1 = float(rouge["rouge-1"]["f"])
        rouge_l = float(rouge["rouge-l"]["f"])
    except (ValueError, ZeroDivisionError):
        rouge_1 = 0.0
        rouge_l = 0.0
    bleu = float(sacrebleu.sentence_bleu(pred_tokens, [ref_tokens]).score)
    return {"rouge_1": rouge_1, "rouge_l": rouge_l, "bleu": bleu}


def _batch_bert_scores(
    pairs: Sequence[tuple[str, str, str]],
    bert_entry: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    if not pairs:
        return {}
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError(
            "MTCMB offline scoring requires bert-score, jieba, rouge-chinese and sacrebleu"
        ) from exc
    scorer = BERTScorer(
        model_type=str(bert_entry["canonical_path"]),
        # bert-score==0.3.13 maps the official model identifier
        # ``bert-base-chinese`` to layer 8.  Supplying the local path requires
        # the layer explicitly; using the architectural final layer (12)
        # would silently change the upstream metric.
        num_layers=8,
        batch_size=batch_size,
        nthreads=1,
        all_layers=False,
        idf=False,
        device=device,
        lang="zh",
        rescale_with_baseline=False,
    )
    identifiers = [identifier for identifier, _, _ in pairs]
    predictions = [prediction for _, prediction, _ in pairs]
    references = [reference for _, _, reference in pairs]
    _, _, f1 = scorer.score(predictions, references, batch_size=batch_size)
    values = [float(value) for value in f1.detach().cpu().tolist()]
    if len(values) != len(identifiers):
        raise RuntimeError("BERTScore output cardinality changed")
    return dict(zip(identifiers, values, strict=True))


def _pr_scores(gold_raw: Any, prediction_raw: Any) -> dict[str, float]:
    def upstream_list(value: Any) -> list[str]:
        import ast

        if isinstance(value, list):
            return [str(item).strip() for item in value]
        if not isinstance(value, str):
            return []
        raw = value.strip()
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except (ValueError, SyntaxError):
            pass
        parts = [item.strip() for item in raw.strip("[]").split(",") if item.strip()]
        return parts

    gold = set(upstream_list(gold_raw))
    pred = set(upstream_list(prediction_raw))
    intersection = gold & pred
    union = gold | pred
    jaccard = len(intersection) / len(union) if union else 0.0
    recall = len(intersection) / len(gold) if gold else 0.0
    precision = len(intersection) / len(pred) if pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = max(len(gold), len(pred))
    size_agreement = 1.0 - abs(len(gold) - len(pred)) / denominator if denominator else 0.0
    return {
        "Jaccard": jaccard,
        "Recall": recall,
        "Precision": precision,
        "F1_Score": f1,
        "Avg_Herb_Num_Match": size_agreement,
        "Task2_Score": (jaccard + f1 + size_agreement) / 3.0,
    }


def _msdd_scores(
    gold: Mapping[str, Any], raw: str, vocab: Mapping[str, Sequence[str]]
) -> tuple[float, bool, dict[str, list[str]]]:
    pred = protocol.extract_msdd_prediction(raw, vocab)
    parts = []
    exact = True
    serialized: dict[str, list[str]] = {}
    for key in ("证型", "疾病"):
        gold_set = {
            value.strip() for value in str(gold.get(key) or "").split("|") if value.strip()
        }
        pred_set = set(pred[key])
        parts.append(len(pred_set & gold_set) / len(gold_set) if gold_set else 0.0)
        exact = exact and pred_set == gold_set
        serialized[key] = sorted(pred_set)
    return sum(parts) / len(parts), exact, serialized


def expected_score_record_keys(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, ...]:
    keys = tuple(
        protocol.record_key(task, row["id"])
        for task in sorted(protocol.OFFLINE11_TASKS)
        for row in records[task]
    )
    if len(keys) != protocol.EXPECTED_OFFLINE11_N or len(set(keys)) != len(keys):
        raise ValueError("MTCMB expected score record-key inventory changed")
    return keys


def expected_controlled_pair_keys(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    exclusions: Mapping[str, Any],
) -> tuple[str, ...]:
    excluded = protocol.overlap_excluded_ids(exclusions)
    keys = tuple(
        protocol.record_key(task, row["id"])
        for task in sorted(protocol.MCQ_TASKS)
        for row in records[task]
        if protocol.record_key(task, row["id"]) not in excluded
        and protocol.record_key(task, row["id"])
        not in protocol.CONTROLLED_PERTURBATION_EXCLUDED_IDS
    )
    if len(keys) != protocol.EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS or len(set(keys)) != len(
        keys
    ):
        raise ValueError("MTCMB expected controlled-pair key inventory changed")
    return keys


def aggregate_scored_rows(
    scored: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute every persisted official/derived aggregate from scored rows."""

    view_expected = {
        "mtcmb_full": (protocol.EXPECTED_FULL_N, protocol.EXPECTED_OFFLINE11_N),
        "overlap_clean": (
            protocol.EXPECTED_OVERLAP_CLEAN_ALL12_N,
            protocol.EXPECTED_OVERLAP_CLEAN_OFFLINE11_N,
        ),
        "provenance_clean9": (protocol.EXPECTED_CLEAN9_N, protocol.EXPECTED_CLEAN9_N),
    }
    views: dict[str, Any] = {}
    for view, (declared_n, offline_n) in view_expected.items():
        subset = [row for row in scored if view in row["views"]]
        if len(subset) != offline_n:
            raise ValueError(f"{view} offline count {len(subset)} != {offline_n}")
        by_task: dict[str, Any] = {}
        for task in sorted({str(row["task"]) for row in subset}):
            task_rows = [row for row in subset if row["task"] == task]
            precise = sum(float(row["official_score_0_100"]) for row in task_rows) / len(
                task_rows
            )
            repository = precise
            if task in {
                "TCM-FT",
                "TCMeEE",
                "TCM-CHGD",
                "TCM-LitData",
                "TCM-Diagnosis",
                "TCM-PR",
                "TCM-FRD",
            }:
                repository = float(math.floor(precise))
            eligible = [row for row in task_rows if row["binary_task_success_eligible"]]
            by_task[task] = {
                "n": len(task_rows),
                "precise_official_score_0_100": precise,
                "upstream_repository_aggregation_score_0_100": repository,
                "binary_task_success_rate": (
                    sum(bool(row["binary_task_success"]) for row in eligible) / len(eligible)
                    if eligible
                    else None
                ),
                "strict_format_rate": (
                    sum(bool(row["strict_format"]) for row in task_rows) / len(task_rows)
                    if task in protocol.MCQ_TASKS
                    else None
                ),
                "parse_failure_rate": (
                    sum(bool(row["parse_failure"]) for row in task_rows) / len(task_rows)
                    if task in protocol.MCQ_TASKS
                    else None
                ),
            }
        eligible_rows = [row for row in subset if row["binary_task_success_eligible"]]
        exact_by_task = {
            task: {
                "eligible_records": value["n"],
                "clean_task_success_rate": value["binary_task_success_rate"],
            }
            for task, value in by_task.items()
            if task in protocol.EXACT_TASKS
        }
        views[view] = {
            "declared_records_including_remote_judge": declared_n,
            "offline_scored_records": offline_n,
            "offline_tasks": len(by_task),
            "remote_judge_task": {
                "task": "TCM-SE-A",
                "records": 50 if view != "provenance_clean9" else 0,
                "score": None,
                "reason": "upstream requires remote GLM-4-Air-250414 judge",
            },
            "task_scores": by_task,
            "descriptive_equal_task_macro_precise_0_100": sum(
                value["precise_official_score_0_100"] for value in by_task.values()
            )
            / len(by_task),
            "descriptive_equal_task_macro_repository_0_100": sum(
                value["upstream_repository_aggregation_score_0_100"]
                for value in by_task.values()
            )
            / len(by_task),
            "natural_exact_endpoints_by_task": {
                "tasks": exact_by_task,
                "eligible_records_descriptive_only": len(eligible_rows),
                "scope_note": (
                    "Task-specific endpoints only. They are not pooled into a sample-weighted "
                    "cross-task effectiveness score."
                ),
            },
        }
    controlled = {}
    for view in ("overlap_clean_mcq", "provenance_clean9_mcq"):
        subset = [row for row in pairs if view in row["views"]]
        controlled[view] = {
            **protocol.controlled_metrics(subset),
            "rate_estimates_95ci": protocol.controlled_rate_intervals(subset),
            "tasks": sorted({row["task"] for row in subset}),
            "status": "derived_not_official",
            "scope_note": (
                "The strict Clean9 view contains only 50 TCM-SE-B pairs."
                if view == "provenance_clean9_mcq"
                else "Direct, exact-stem and preregistered near overlaps removed, but "
                "ED-A/ED-B task-family exposure remains."
            ),
        }
    return {"views": views, "controlled_robustness": controlled}


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    exclusions: Mapping[str, Any],
    bert_entry: Mapping[str, Any],
    *,
    bert_device: str,
    bert_batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_item = _prediction_lookup(predictions)
    lookup = _task_lookup(records)
    excluded = protocol.overlap_excluded_ids(exclusions)
    vocab = protocol.msdd_vocab(records["TCM-MSDD"])

    bert_pairs: list[tuple[str, str, str]] = []
    for task in sorted(protocol.BERT_TASKS):
        for row in records[task]:
            rid = int(row["id"])
            prediction = by_item[f"clean/{task}:{rid}"]["raw_output"]
            if task == "TCM-FT":
                reference = str(row.get("points") or row.get("answer") or "").strip()
                candidate = str(prediction).strip()
            else:
                reference = json.dumps(row["answer"], ensure_ascii=False)
                candidate = json.dumps(str(prediction), ensure_ascii=False)
            bert_pairs.append((f"{task}:{rid}", candidate, reference))
    bert_scores = _batch_bert_scores(
        bert_pairs, bert_entry, device=bert_device, batch_size=bert_batch_size
    )

    scored: list[dict[str, Any]] = []
    for task in sorted(protocol.OFFLINE11_TASKS):
        for row in records[task]:
            rid = int(row["id"])
            key = protocol.record_key(task, rid)
            binary_success: bool | None = None
            strict_format: bool | None = None
            parse_failure: bool | None = None
            components: dict[str, Any]
            if task == "TCM-LitData":
                annotation_scores = []
                annotation_predictions = []
                for annotation_index, annotation in enumerate(row["annotations"]):
                    raw = str(
                        by_item[f"clean/{task}:{rid}/annotation:{annotation_index}"][
                            "raw_output"
                        ]
                    )
                    annotation_predictions.append(raw)
                    annotation_scores.append(_lexical_scores(raw, str(annotation["A"])))
                n_annotations = len(annotation_scores)
                components = {
                    metric: sum(value[metric] for value in annotation_scores) / n_annotations
                    for metric in ("rouge_1", "rouge_l", "bleu")
                }
                official_score = (
                    ((components["rouge_1"] + components["rouge_l"]) / 2.0)
                    + components["bleu"] / 100.0
                ) / 2.0 * 100.0
                raw_prediction: Any = annotation_predictions
            else:
                prediction_row = by_item[f"clean/{task}:{rid}"]
                raw = str(prediction_row["raw_output"])
                raw_prediction = raw
                if task in protocol.MCQ_TASKS:
                    labels, _ = protocol.parse_mcq_options(task, row)
                    parsed = protocol.parse_official_choice(raw, labels)
                    _, strict = protocol.parse_choice(raw, labels)
                    gold = str(row["answer"]).strip().upper()
                    binary_success = bool(parsed) and parsed == gold
                    strict_format = strict
                    parse_failure = not bool(parsed)
                    components = {
                        "prediction": parsed,
                        "gold": gold,
                        "accuracy": 1.0 if binary_success else 0.0,
                    }
                    official_score = components["accuracy"] * 100.0
                elif task == "TCM-FT":
                    components = {"bert": bert_scores[key]}
                    official_score = components["bert"] * 100.0
                elif task in {"TCMeEE", "TCM-CHGD", "TCM-Diagnosis", "TCM-FRD"}:
                    reference = json.dumps(row["answer"], ensure_ascii=False)
                    candidate = json.dumps(raw, ensure_ascii=False)
                    components = _lexical_scores(candidate, reference)
                    components["bert"] = bert_scores[key]
                    official_score = (
                        (components["rouge_1"] + components["rouge_l"]) / 2.0
                        + components["bert"]
                        + components["bleu"] / 100.0
                    ) / 3.0 * 100.0
                elif task == "TCM-MSDD":
                    partial, exact, parsed = _msdd_scores(row["answer"], raw, vocab)
                    binary_success = exact
                    components = {"partial_accuracy": partial, "parsed_labels": parsed}
                    official_score = partial * 100.0
                elif task == "TCM-PR":
                    components = _pr_scores(row["answer"], raw)
                    binary_success = protocol.exact_task_success(
                        task, row, raw, vocab=vocab
                    )
                    official_score = components["Task2_Score"] * 100.0
                else:
                    raise AssertionError(task)
            scored.append(
                {
                    "task": task,
                    "record_id": rid,
                    "record_key": key,
                    "views": list(protocol.record_views(task, rid, excluded)),
                    "raw_prediction": raw_prediction,
                    "official_components": components,
                    "official_score_0_100": official_score,
                    "binary_task_success_eligible": task in protocol.EXACT_TASKS,
                    "binary_task_success": binary_success,
                    "strict_format": strict_format,
                    "parse_failure": parse_failure,
                }
            )

    if len(scored) != protocol.EXPECTED_OFFLINE11_N:
        raise RuntimeError(f"scored record count changed: {len(scored)}")

    pairs: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction["condition"] != "option_permutation":
            continue
        task = str(prediction["task"])
        rid = int(prediction["record_id"])
        clean = by_item[f"clean/{task}:{rid}"]
        source = lookup[(task, rid)]
        labels, _ = protocol.parse_mcq_options(task, source)
        clean_prediction = protocol.parse_official_choice(str(clean["raw_output"]), labels)
        displayed_prediction = protocol.parse_official_choice(
            str(prediction["raw_output"]), labels
        )
        mapping = {str(k): str(v) for k, v in prediction["canonical_to_displayed"].items()}
        inverse = {displayed: canonical for canonical, displayed in mapping.items()}
        canonical_perturbed = inverse.get(displayed_prediction, "")
        gold = str(source["answer"]).strip().upper()
        row = {
            "task": task,
            "record_id": rid,
            "record_key": protocol.record_key(task, rid),
            "perturbation": "deterministic_nonidentity_option_permutation",
            "canonical_to_displayed": mapping,
            "clean_raw_output": clean["raw_output"],
            "perturbed_raw_output": prediction["raw_output"],
            "clean_canonical_prediction": clean_prediction,
            "clean_parse_ok": bool(clean_prediction),
            "perturbed_displayed_prediction": displayed_prediction,
            "perturbed_canonical_prediction": canonical_perturbed,
            "perturbed_parse_ok": bool(displayed_prediction) and bool(canonical_perturbed),
            "canonical_gold": gold,
            "clean_task_success": bool(clean_prediction) and clean_prediction == gold,
            "perturbed_task_success": bool(canonical_perturbed) and canonical_perturbed == gold,
            "views": ["overlap_clean_mcq"]
            + (["provenance_clean9_mcq"] if task == "TCM-SE-B" else []),
        }
        pairs.append(row)
    if len(pairs) != protocol.EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS:
        raise RuntimeError(f"controlled pair count changed: {len(pairs)}")
    if sum("provenance_clean9_mcq" in row["views"] for row in pairs) != protocol.EXPECTED_CLEAN9_MCQ_PAIRS:
        raise RuntimeError("strict Clean9 controlled pair count changed")

    return scored, pairs, aggregate_scored_rows(scored, pairs)


class EvaluationInterrupted(RuntimeError):
    """Raised after SIGINT/SIGTERM so worker shards remain resumable."""


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _suite_model_fingerprint(suite: Mapping[str, Any]) -> str:
    return protocol.canonical_hash(
        [(str(entry["label"]), str(entry["model_fingerprint"])) for entry in suite["models"]]
    )


def _assert_suite_matches_lock(
    suite: Mapping[str, Any], lock_identity: Mapping[str, Any]
) -> None:
    actual = _suite_model_fingerprint(suite)
    expected = str(lock_identity["model_suite_fingerprint"])
    if actual != expected:
        raise ValueError(
            "frozen MTCMB model suite does not match the preregistered six-model suite"
        )
    observed_assets = evaluation_assets_fingerprint(suite)
    if suite.get("evaluation_assets_fingerprint") != observed_assets:
        raise ValueError("MTCMB evaluation-assets fingerprint does not close")
    auxiliary = lock_identity.get("auxiliary_models")
    locked_bert = auxiliary.get("mtcmb_bert_scorer") if isinstance(auxiliary, Mapping) else None
    if not isinstance(locked_bert, Mapping) or locked_bert.get("status") != "LOCKED":
        raise ValueError("root preregistration lock has no frozen MTCMB BERT scorer")
    suite_bert = suite.get("bert_scorer")
    if not isinstance(suite_bert, Mapping):
        raise ValueError("MTCMB suite has no BERT scorer entry")
    if (
        str(suite_bert.get("canonical_path") or "")
        != str(locked_bert.get("canonical_path") or "")
        or str(suite_bert.get("model_fingerprint") or "")
        != str(locked_bert.get("artifact_fingerprint") or "")
    ):
        raise ValueError(
            "MTCMB BERT scorer differs between the root lock and evaluation suite"
        )


def _write_or_validate_inference_manifest(
    path: Path, items: Sequence[protocol.InferenceItem]
) -> str:
    expected = [item.as_dict() for item in items]
    if path.is_file():
        current = _jsonl_rows(path)
        if current != expected:
            raise ValueError(
                f"existing inference manifest differs from the locked protocol: {path}"
            )
    else:
        atomic_jsonl(path, expected)
    return protocol.sha256_file(path)


def _completed_valid(
    output_dir: Path,
    *,
    run_fingerprint: str,
    expected_item_ids: set[str],
    expected_score_keys: Sequence[str],
    expected_pair_keys: Sequence[str],
    fingerprints: Mapping[str, str],
) -> bool:
    required = {
        "predictions.jsonl": output_dir / "predictions.jsonl",
        "record_scores.jsonl": output_dir / "record_scores.jsonl",
        "controlled_robustness_pairs.jsonl": output_dir
        / "controlled_robustness_pairs.jsonl",
        "summary.json": output_dir / "summary.json",
        "run_manifest.json": output_dir / "run_manifest.json",
    }
    completion_path = output_dir / "COMPLETE.json"
    if not completion_path.is_file() or not all(path.is_file() for path in required.values()):
        return False
    try:
        completion = load_json(completion_path)
        predictions = _jsonl_rows(required["predictions.jsonl"])
        scores = _jsonl_rows(required["record_scores.jsonl"])
        pairs = _jsonl_rows(required["controlled_robustness_pairs.jsonl"])
        summary = load_json(required["summary.json"])
        manifest = load_json(required["run_manifest.json"])
        score_keys = [str(row.get("record_key") or "") for row in scores]
        pair_keys = [str(row.get("record_key") or "") for row in pairs]
        if (
            len(score_keys) != len(set(score_keys))
            or score_keys != list(expected_score_keys)
            or len(pair_keys) != len(set(pair_keys))
            or pair_keys != list(expected_pair_keys)
        ):
            return False
        recomputed = aggregate_scored_rows(scores, pairs)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    artifact_hashes = completion.get("artifact_sha256")
    hashes_close = isinstance(artifact_hashes, Mapping) and all(
        artifact_hashes.get(name) == protocol.sha256_file(path)
        for name, path in required.items()
    )
    ids = [str(row.get("item_id") or "") for row in predictions]
    return bool(
        completion.get("status") == "complete"
        and completion.get("run_fingerprint") == run_fingerprint
        and manifest.get("run_fingerprint") == run_fingerprint
        and summary.get("run_fingerprint") == run_fingerprint
        and summary.get("fingerprints") == dict(fingerprints)
        and summary.get("official_results") == recomputed["views"]
        and summary.get("derived_controlled_robustness")
        == recomputed["controlled_robustness"]
        and len(ids) == len(expected_item_ids)
        and len(set(ids)) == len(ids)
        and set(ids) == expected_item_ids
        and len(scores) == len(expected_score_keys)
        and len(pairs) == len(expected_pair_keys)
        and all(
            all(row.get(key) == value for key, value in fingerprints.items())
            for row in predictions
        )
        and hashes_close
    )


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.num_workers != FORMAL_NUM_WORKERS or tuple(args.devices) != FORMAL_DEVICES:
        raise ValueError(
            "formal MTCMB protocol is frozen to four workers on logical "
            "cuda:0,cuda:1,cuda:2,cuda:3"
        )
    assets = validate_protocol_assets(
        args.mtcmb_repo,
        args.exclusions,
        args.task_manifest,
        args.schema,
        verify_git=not args.skip_git_check,
    )
    metric_runtime = validate_metric_packages()
    inference_runtime = runtime_identity()
    lock_identity = validate_static_lock(
        args.preregistration_lock,
        project_root=PROJECT_ROOT,
        required_paths=REQUIRED_LOCK_PATHS,
        timing_key="mtcmb_clean_predictions_observed",
        expected_model_order=protocol.MODEL_ORDER,
    )
    suite, model_entry, bert_entry = load_suite_entry(args.suite, args.model_label)
    _assert_suite_matches_lock(suite, lock_identity)
    records = protocol.load_all_records(args.mtcmb_repo)
    items = protocol.build_inference_items(
        args.mtcmb_repo, records, assets["exclusions"]
    )
    score_key_contract = expected_score_record_keys(records)
    pair_key_contract = expected_controlled_pair_keys(records, assets["exclusions"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    inference_manifest = args.output_root / "inference_items.jsonl"
    inference_manifest_sha = _write_or_validate_inference_manifest(
        inference_manifest, items
    )
    p_spec = prompt_spec()
    g_spec = generation_spec(args.dtype, args.batch_size)
    analysis_spec = {
        "official_scoring": "pinned upstream formulas; precise and repository aggregation both retained",
        "official_and_derived_separate": True,
        "full_claim_scope": "in-source diagnostic only",
        "overlap_clean_claim_scope": (
            "result-blind direct provenance UNION exact normalized stem UNION "
            "preregistered near-match exclusions removed (1,186 records); "
            "ED-A/ED-B family exposure remains"
        ),
        "provenance_clean_claim_scope": "Clean9 excludes exposed families and remote judge",
        "controlled_robustness": "deterministic nonidentity option permutation",
        "bert_scoring": {
            "device": args.bert_device,
            "batch_size": args.bert_batch_size,
            "lang": "zh",
            "num_layers": 8,
            "rescale_with_baseline": False,
        },
        "soft_generation_binary_threshold": None,
        "sample_micro_average_across_tasks": None,
        "strict_clean9_rate_uncertainty": {
            "scope": "50 paired TCM-SE-B records",
            "endpoints": [
                "clean_task_success_rate",
                "perturbed_task_success_rate",
                "conditional_retention",
                "regression_rate",
            ],
            "method": "two-sided Wilson score interval",
            "confidence_level": 0.95,
            "reports_counts_estimate_percent_and_ci": True,
        },
        "primary_between_model_noninferiority": {
            "contrast": "care_opd minus clean_sft",
            "secondary_contrast": "care_opd minus gkd_full",
            "endpoints": [
                "perturbed_task_success_rate",
                "conditional_retention",
            ],
            "margin_absolute": NONINFERIORITY_MARGIN,
            "method": "paired item bootstrap percentile interval",
            "replicates": PAIR_BOOTSTRAP_REPLICATES,
            "seed": PAIR_BOOTSTRAP_SEED,
            "joint_rule": "both 95% CI lower bounds must exceed -0.05",
        },
    }
    fingerprints = {
        "dataset_contract_fingerprint": str(assets["dataset_contract_fingerprint"]),
        "inference_manifest_sha256": inference_manifest_sha,
        "prompt_fingerprint": protocol.canonical_hash(p_spec),
        "generation_fingerprint": protocol.canonical_hash(g_spec),
        "analysis_fingerprint": protocol.canonical_hash(analysis_spec),
        "task_manifest_sha256": str(assets["task_manifest_sha256"]),
        "unified_metric_schema_sha256": str(assets["unified_metric_schema_sha256"]),
        "preregistration_lock_sha256": str(lock_identity["sha256"]),
        "model_suite_fingerprint": str(suite["suite_fingerprint"]),
        "model_fingerprint": str(model_entry["model_fingerprint"]),
        "bert_scorer_fingerprint": str(bert_entry["model_fingerprint"]),
        "metric_runtime_fingerprint": protocol.canonical_hash(metric_runtime),
        "runtime_identity_fingerprint": protocol.canonical_hash(inference_runtime),
    }
    contract = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "model_label": args.model_label,
        "model_path": model_entry["canonical_path"],
        "workers": args.num_workers,
        "devices": list(args.devices),
        "fingerprints": fingerprints,
        "evaluator_sha256": protocol.sha256_file(Path(__file__).resolve()),
    }
    run_fingerprint = protocol.canonical_hash(contract)
    output_dir = args.output_root.resolve() / "official" / args.model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        run_manifest = load_json(manifest_path)
        if run_manifest.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"run contract changed for {output_dir}; select a fresh output root"
            )
    else:
        run_manifest = {
            **contract,
            "schema_version": "care-opd-mtcmb-run-manifest-v1",
            "run_fingerprint": run_fingerprint,
            "created_at_utc": utc_now(),
            "dataset": {
                "mtcmb_repo": str(args.mtcmb_repo.resolve()),
                "upstream": assets["upstream"],
                "full_records": protocol.EXPECTED_FULL_N,
                "offline11_records": protocol.EXPECTED_OFFLINE11_N,
                "overlap_clean_offline11_records": protocol.EXPECTED_OVERLAP_CLEAN_OFFLINE11_N,
                "provenance_clean9_records": protocol.EXPECTED_CLEAN9_N,
                "clean_inference_calls": protocol.EXPECTED_CLEAN_INFERENCE_CALLS,
                "controlled_inference_calls": protocol.EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS,
                "inference_manifest": str(inference_manifest.resolve()),
            },
            "prompt": p_spec,
            "generation": g_spec,
            "analysis": analysis_spec,
            "preregistration": lock_identity,
            "model": model_entry,
            "bert_scorer": bert_entry,
            "metric_runtime": metric_runtime,
            "runtime_identity": inference_runtime,
        }
        atomic_json(manifest_path, run_manifest)

    expected_item_ids = {item.item_id for item in items}
    if _completed_valid(
        output_dir,
        run_fingerprint=run_fingerprint,
        expected_item_ids=expected_item_ids,
        expected_score_keys=score_key_contract,
        expected_pair_keys=pair_key_contract,
        fingerprints=fingerprints,
    ):
        print(f"[skip] validated complete MTCMB result: {output_dir}", flush=True)
        return load_json(output_dir / "summary.json")

    (output_dir / "COMPLETE.json").unlink(missing_ok=True)
    shard_root = output_dir / "shards"
    shard_root.mkdir(exist_ok=True)
    indexed = list(enumerate(items))
    partitions = [indexed[index :: args.num_workers] for index in range(args.num_workers)]
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
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
                shard_root / f"worker_{worker_id:02d}.jsonl",
                shard_root / f"worker_{worker_id:02d}.complete.json",
                error_path,
                model_entry,
                args.dtype,
                args.batch_size,
                run_fingerprint,
                fingerprints,
            ),
            name=f"mtcmb-{args.model_label}-{worker_id}",
        )
        processes.append(process)

    old_handlers: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise EvaluationInterrupted(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupted)
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
            raise RuntimeError("MTCMB worker failure(s): " + " | ".join(diagnostics))
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
        expected = {item.item_id: index for index, item in partition}
        completion_path = shard_root / f"worker_{worker_id:02d}.complete.json"
        if not completion_path.is_file():
            raise RuntimeError(f"missing worker completion sentinel: {completion_path}")
        merged.extend(
            _recover_shard(
                shard_root / f"worker_{worker_id:02d}.jsonl",
                expected,
                run_fingerprint,
            )
        )
    merged.sort(key=lambda row: int(row["index"]))
    actual_ids = [str(row["item_id"]) for row in merged]
    if (
        len(actual_ids) != len(items)
        or len(set(actual_ids)) != len(items)
        or set(actual_ids) != expected_item_ids
    ):
        raise RuntimeError("merged MTCMB predictions are incomplete or duplicated")
    predictions_path = output_dir / "predictions.jsonl"
    atomic_jsonl(predictions_path, merged)

    # Scoring starts only after all generation workers have exited, so the
    # frozen local BERT scorer can safely reuse logical cuda:0.
    record_scores, pairs, aggregate = score_predictions(
        merged,
        records,
        assets["exclusions"],
        bert_entry,
        bert_device=args.bert_device,
        bert_batch_size=args.bert_batch_size,
    )
    scores_path = output_dir / "record_scores.jsonl"
    pairs_path = output_dir / "controlled_robustness_pairs.jsonl"
    atomic_jsonl(scores_path, record_scores)
    atomic_jsonl(pairs_path, pairs)
    summary = {
        "schema_version": "care-opd-mtcmb-summary-v1",
        "status": "complete",
        "model_label": args.model_label,
        "run_fingerprint": run_fingerprint,
        "fingerprints": fingerprints,
        "completed_at_utc": utc_now(),
        "official_results": aggregate["views"],
        "derived_controlled_robustness": aggregate["controlled_robustness"],
        "reporting_warning": (
            "MTCMB-Full is an in-source diagnostic because 1,062 unique official records "
            "were directly exposed by formal training seeds; only provenance_clean9 supports "
            "the supplemental task-family generalization interpretation."
        ),
    }
    summary_path = output_dir / "summary.json"
    atomic_json(summary_path, summary)
    artifacts = {
        "predictions.jsonl": predictions_path,
        "record_scores.jsonl": scores_path,
        "controlled_robustness_pairs.jsonl": pairs_path,
        "summary.json": summary_path,
        "run_manifest.json": manifest_path,
    }
    atomic_json(
        output_dir / "COMPLETE.json",
        {
            "status": "complete",
            "run_fingerprint": run_fingerprint,
            "prediction_rows": len(merged),
            "official_offline_records": len(record_scores),
            "controlled_pairs": len(pairs),
            "fingerprints": fingerprints,
            "artifact_sha256": {
                name: protocol.sha256_file(path) for name, path in artifacts.items()
            },
            "completed_at_utc": utc_now(),
        },
    )
    return summary


def _mcnemar_exact(
    care: Mapping[str, bool], baseline: Mapping[str, bool]
) -> dict[str, Any]:
    if set(care) != set(baseline):
        raise ValueError("McNemar maps are not fully paired")
    ids = sorted(care)
    care_only = sum(bool(care[key]) and not bool(baseline[key]) for key in ids)
    base_only = sum(bool(baseline[key]) and not bool(care[key]) for key in ids)
    discordant = care_only + base_only
    if not discordant:
        p_value = 1.0
    else:
        lower = min(care_only, base_only)
        tail = sum(math.comb(discordant, value) for value in range(lower + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "paired_units": len(ids),
        "care_correct_baseline_wrong": care_only,
        "baseline_correct_care_wrong": base_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_robustness_noninferiority(
    care_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    baseline_label: str,
    *,
    replicates: int = PAIR_BOOTSTRAP_REPLICATES,
    seed: int = PAIR_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired CARE-minus-baseline CI on the strict 50-item Clean9 track."""

    care = {str(row["record_key"]): dict(row) for row in care_rows}
    baseline = {str(row["record_key"]): dict(row) for row in baseline_rows}
    if set(care) != set(baseline) or len(care) != protocol.EXPECTED_CLEAN9_MCQ_PAIRS:
        raise ValueError(f"CARE and {baseline_label} strict Clean9 pair IDs differ")
    keys = sorted(care)
    endpoints = (
        "perturbed_task_success_rate",
        "conditional_retention",
    )
    care_observed = protocol.controlled_metrics([care[key] for key in keys])
    baseline_observed = protocol.controlled_metrics([baseline[key] for key in keys])
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {key: [] for key in endpoints}
    for _ in range(replicates):
        drawn = rng.choices(keys, k=len(keys))
        care_metric = protocol.controlled_metrics([care[key] for key in drawn])
        baseline_metric = protocol.controlled_metrics([baseline[key] for key in drawn])
        for endpoint in endpoints:
            left = care_metric[endpoint]
            right = baseline_metric[endpoint]
            if left is not None and right is not None:
                samples[endpoint].append(float(left) - float(right))
    intervals: dict[str, Any] = {}
    for endpoint in endpoints:
        observed_left = care_observed[endpoint]
        observed_right = baseline_observed[endpoint]
        low = _percentile(samples[endpoint], 0.025) if samples[endpoint] else None
        high = _percentile(samples[endpoint], 0.975) if samples[endpoint] else None
        intervals[endpoint] = {
            "care_opd": observed_left,
            "baseline": observed_right,
            "delta": (
                float(observed_left) - float(observed_right)
                if observed_left is not None and observed_right is not None
                else None
            ),
            "ci95_low": low,
            "ci95_high": high,
            "replicates_used": len(samples[endpoint]),
            "margin": NONINFERIORITY_MARGIN,
            "noninferiority_pass": bool(
                low is not None and low > NONINFERIORITY_MARGIN
            ),
        }
    return {
        "baseline": baseline_label,
        "paired_units": len(keys),
        "method": "paired item bootstrap percentile interval",
        "seed": seed,
        "replicates": replicates,
        "margin_absolute": NONINFERIORITY_MARGIN,
        "endpoints": intervals,
        "joint_primary_noninferiority_pass": all(
            intervals[endpoint]["noninferiority_pass"] for endpoint in endpoints
        ),
        "interpretation": (
            "Both CARE-minus-baseline lower bounds must exceed -5 percentage points; "
            "passing means preserved/non-inferior on the small SE-B track, not superiority."
        ),
    }


def _main_table_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    unified_fields = tuple(protocol.unified_na())
    # Emit one complete six-model paired robustness block.  MTCMB's heterogeneous
    # official task metrics stay task-specific in the detailed supplement; they
    # are not collapsed into a fabricated sample-weighted cross-task endpoint.
    for record in records:
        metrics = {field: record[field] for field in unified_fields}
        if set(metrics) != set(unified_fields):
            raise AssertionError("unified metric fields changed")
        rows.append(
            {
                "benchmark": "MTCMB",
                "model": record["model"],
                "track": "provenance_clean9_option_permutation",
                "track_provenance": "Public filtered / derived option permutation",
                "official_or_derived": "derived",
                "claim_scope": "strict task-family provenance-clean TCM-SE-B pairs",
                "clean_n": record["paired_units"],
                "perturbed_n": record["paired_units"],
                "effectiveness_display_key": "MTCMB Clean9 SE-B perturbed TSR",
                "robustness_display_key": "MTCMB Clean9 SE-B retention",
                "unified_metrics": metrics,
            }
        )
    return rows


def summarize_suite(args: argparse.Namespace) -> Mapping[str, Any]:
    assets = validate_protocol_assets(
        args.mtcmb_repo,
        args.exclusions,
        args.task_manifest,
        args.schema,
        verify_git=not args.skip_git_check,
    )
    lock_identity = validate_static_lock(
        args.preregistration_lock,
        project_root=PROJECT_ROOT,
        required_paths=REQUIRED_LOCK_PATHS,
        timing_key="mtcmb_clean_predictions_observed",
        expected_model_order=protocol.MODEL_ORDER,
    )
    suite = load_json(args.suite.resolve(strict=True))
    _assert_suite_matches_lock(suite, lock_identity)
    dataset_records = protocol.load_all_records(args.mtcmb_repo)
    score_key_contract = expected_score_record_keys(dataset_records)
    pair_key_contract = expected_controlled_pair_keys(
        dataset_records, assets["exclusions"]
    )
    inference_items = _jsonl_rows(args.output_root / "inference_items.jsonl")
    expected_item_ids = {str(row["item_id"]) for row in inference_items}
    if len(inference_items) != (
        protocol.EXPECTED_CLEAN_INFERENCE_CALLS
        + protocol.EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS
    ) or len(expected_item_ids) != len(inference_items):
        raise ValueError("suite inference manifest has invalid cardinality")

    records: list[dict[str, Any]] = []
    exact_maps: dict[str, dict[str, dict[str, bool]]] = {}
    perturbed_maps: dict[str, dict[str, bool]] = {}
    strict_pair_rows: dict[str, list[dict[str, Any]]] = {}
    for entry in suite["models"]:
        label = str(entry["label"])
        output_dir = args.output_root.resolve() / "official" / label
        manifest = load_json(output_dir / "run_manifest.json")
        run_fingerprint = str(manifest["run_fingerprint"])
        fingerprints = manifest["fingerprints"]
        if manifest.get("model_label") != label:
            raise ValueError(f"model label mismatch in MTCMB manifest for {label}")
        if fingerprints.get("preregistration_lock_sha256") != lock_identity["sha256"]:
            raise ValueError(f"preregistration mismatch for {label}")
        if fingerprints.get("model_suite_fingerprint") != suite["suite_fingerprint"]:
            raise ValueError(f"six-model suite fingerprint mismatch for {label}")
        if fingerprints.get("model_fingerprint") != entry["model_fingerprint"]:
            raise ValueError(f"checkpoint fingerprint mismatch for {label}")
        if not _completed_valid(
            output_dir,
            run_fingerprint=run_fingerprint,
            expected_item_ids=expected_item_ids,
            expected_score_keys=score_key_contract,
            expected_pair_keys=pair_key_contract,
            fingerprints=fingerprints,
        ):
            raise ValueError(f"incomplete or invalid MTCMB result for {label}")
        summary = load_json(output_dir / "summary.json")
        scores = _jsonl_rows(output_dir / "record_scores.jsonl")
        pair_rows = _jsonl_rows(output_dir / "controlled_robustness_pairs.jsonl")
        full = summary["official_results"]["mtcmb_full"]
        overlap = summary["official_results"]["overlap_clean"]
        clean9 = summary["official_results"]["provenance_clean9"]
        strict_robustness = summary["derived_controlled_robustness"][
            "provenance_clean9_mcq"
        ]
        overlap_robustness = summary["derived_controlled_robustness"][
            "overlap_clean_mcq"
        ]
        row = {
            "model": label,
            "mtcmb_full_declared_n": full["declared_records_including_remote_judge"],
            "mtcmb_full_offline_n": full["offline_scored_records"],
            "mtcmb_full_offline11_equal_task_macro_0_100": full[
                "descriptive_equal_task_macro_precise_0_100"
            ],
            "overlap_clean_declared_n": overlap["declared_records_including_remote_judge"],
            "overlap_clean_offline_n": overlap["offline_scored_records"],
            "overlap_clean_offline11_equal_task_macro_0_100": overlap[
                "descriptive_equal_task_macro_precise_0_100"
            ],
            "provenance_clean9_n": clean9["offline_scored_records"],
            "provenance_clean9_equal_task_macro_0_100": clean9[
                "descriptive_equal_task_macro_precise_0_100"
            ],
            "clean_task_success_rate": strict_robustness["clean_task_success_rate"],
            "perturbed_task_success_rate": strict_robustness[
                "perturbed_task_success_rate"
            ],
            "conditional_retention": strict_robustness["conditional_retention"],
            "regression_rate": strict_robustness["regression_rate"],
            "robustness_drop_pp": strict_robustness["robustness_drop_pp"],
            "prediction_consistency": strict_robustness["prediction_consistency"],
            "worst_perturbation_tsr": strict_robustness["worst_perturbation_tsr"],
            "clean_correct_perturbed_wrong": strict_robustness[
                "clean_correct_perturbed_wrong"
            ],
            "clean_wrong_perturbed_correct": strict_robustness[
                "clean_wrong_perturbed_correct"
            ],
            "paired_units": strict_robustness["paired_units"],
            "provenance_clean9_option_permutation_rate_estimates_95ci": strict_robustness[
                "rate_estimates_95ci"
            ],
            "overlap_clean_mcq_paired_units": overlap_robustness["paired_units"],
            "overlap_clean_mcq_perturbed_task_success_rate": overlap_robustness[
                "perturbed_task_success_rate"
            ],
        }
        records.append(row)
        exact_maps[label] = {
            task: {
                str(score["record_key"]): bool(score["binary_task_success"])
                for score in scores
                if "provenance_clean9" in score["views"]
                and bool(score["binary_task_success_eligible"])
                and score["task"] == task
            }
            for task in ("TCM-MSDD", "TCM-PR", "TCM-SE-B")
        }
        expected_exact_counts = {"TCM-MSDD": 100, "TCM-PR": 100, "TCM-SE-B": 50}
        actual_exact_counts = {
            task: len(values) for task, values in exact_maps[label].items()
        }
        if actual_exact_counts != expected_exact_counts:
            raise ValueError(
                f"task-specific exact endpoint inventory changed for {label}: "
                f"{actual_exact_counts}"
            )
        perturbed_maps[label] = {
            str(pair["record_key"]): bool(pair["perturbed_task_success"])
            for pair in pair_rows
            if "provenance_clean9_mcq" in pair["views"]
        }
        if len(perturbed_maps[label]) != protocol.EXPECTED_CLEAN9_MCQ_PAIRS:
            raise ValueError(f"strict MTCMB robustness pairs changed for {label}")
        strict_pair_rows[label] = sorted(
            (
                dict(pair)
                for pair in pair_rows
                if "provenance_clean9_mcq" in pair["views"]
            ),
            key=lambda pair: str(pair["record_key"]),
        )
        if len(strict_pair_rows[label]) != protocol.EXPECTED_CLEAN9_MCQ_PAIRS:
            raise ValueError(f"strict MTCMB paired rows changed for {label}")

    if tuple(record["model"] for record in records) != protocol.MODEL_ORDER:
        raise ValueError("suite completion order differs from preregistered model order")
    comparisons = []
    care = exact_maps["care_opd"]
    care_perturbed = perturbed_maps["care_opd"]
    for label in protocol.MODEL_ORDER:
        if label == "care_opd":
            continue
        for scope, care_map, baseline_map in (
            *(
                (
                    f"provenance_clean9_{task}_official_exact",
                    care[task],
                    exact_maps[label][task],
                )
                for task in ("TCM-MSDD", "TCM-PR", "TCM-SE-B")
            ),
            (
                "provenance_clean9_option_permutation_perturbed_50",
                care_perturbed,
                perturbed_maps[label],
            ),
        ):
            care_rate = sum(care_map.values()) / len(care_map)
            baseline_rate = sum(baseline_map.values()) / len(baseline_map)
            comparisons.append(
                {
                    "baseline": label,
                    "scope": scope,
                    "care_rate": care_rate,
                    "baseline_rate": baseline_rate,
                    "care_minus_baseline_pp": 100.0 * (care_rate - baseline_rate),
                    "mcnemar": _mcnemar_exact(care_map, baseline_map),
                }
            )

    paired_noninferiority = [
        _paired_robustness_noninferiority(
            strict_pair_rows["care_opd"], strict_pair_rows[baseline], baseline
        )
        for baseline in ("clean_sft", "gkd_full")
    ]

    payload = {
        "schema_version": "care-opd-mtcmb-suite-summary-v1",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "suite_fingerprint": suite["suite_fingerprint"],
        "preregistration_lock_sha256": lock_identity["sha256"],
        "dataset_contract_fingerprint": assets["dataset_contract_fingerprint"],
        "official_results_scope": {
            "mtcmb_full": "in-source diagnostic only",
            "overlap_clean": (
                "result-blind direct/exact-stem/preregistered-near item overlap removed; "
                "family exposure remains"
            ),
            "provenance_clean9": "strict task-family provenance-clean supplement",
        },
        "models": records,
        "main_table_records": _main_table_records(records),
        "paired_comparisons_vs_care_opd": comparisons,
        "primary_paired_noninferiority": paired_noninferiority,
        "remote_judge_exclusion": {
            "task": "TCM-SE-A",
            "records": 50,
            "reason": "official protocol requires remote GLM-4-Air-250414 judge",
        },
        "derived_robustness_scope": (
            "Only deterministic semantic-preserving option permutations are scored; "
            "the strict Clean9 endpoint contains 50 TCM-SE-B pairs."
        ),
        "created_at_utc": utc_now(),
    }
    official_root = args.output_root.resolve() / "official"
    atomic_json(official_root / "suite_summary.json", payload)
    csv_records = [
        {
            **row,
            "provenance_clean9_option_permutation_rate_estimates_95ci": json.dumps(
                row["provenance_clean9_option_permutation_rate_estimates_95ci"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for row in records
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(csv_records[0]))
    writer.writeheader()
    writer.writerows(csv_records)
    atomic_text(official_root / "suite_summary.csv", stream.getvalue())
    if comparisons:
        comparison_rows = [
            {
                "baseline": row["baseline"],
                "scope": row["scope"],
                "care_rate": row["care_rate"],
                "baseline_rate": row["baseline_rate"],
                "care_minus_baseline_pp": row["care_minus_baseline_pp"],
                **row["mcnemar"],
            }
            for row in comparisons
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
        atomic_text(official_root / "paired_comparisons.csv", stream.getvalue())
    gate_rows = []
    for comparison in paired_noninferiority:
        record: dict[str, Any] = {
            "baseline": comparison["baseline"],
            "paired_units": comparison["paired_units"],
            "margin_absolute": comparison["margin_absolute"],
            "joint_primary_noninferiority_pass": comparison[
                "joint_primary_noninferiority_pass"
            ],
        }
        for endpoint, values in comparison["endpoints"].items():
            for key in (
                "care_opd",
                "baseline",
                "delta",
                "ci95_low",
                "ci95_high",
                "replicates_used",
                "noninferiority_pass",
            ):
                record[f"{endpoint}_{key}"] = values[key]
        gate_rows.append(record)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(gate_rows[0]))
    writer.writeheader()
    writer.writerows(gate_rows)
    atomic_text(official_root / "primary_noninferiority.csv", stream.getvalue())

    markdown = [
        "# MTCMB contamination-aware evaluation",
        "",
        "MTCMB-Full is an **in-source diagnostic**, not an independent generalization set.",
        "The strict generalization supplement is task-family provenance-clean Clean9.",
        "The SE-B CARE-vs-clean-SFT/GKD preservation gate uses a preregistered -5 pp "
        "margin on both perturbed TSR and conditional retention; both paired-bootstrap "
        "lower bounds must pass, and passing is not a superiority claim.",
        "",
        "| model | Full offline11 macro | overlap-clean offline11 macro | Clean9 descriptive macro | SE-B clean TSR | SE-B perturbed TSR | SE-B retention | SE-B regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        def pct(value: Any) -> str:
            return "N/A" if value is None else f"{100.0 * float(value):.2f}%"

        markdown.append(
            f"| {row['model']} | "
            f"{row['mtcmb_full_offline11_equal_task_macro_0_100']:.2f} | "
            f"{row['overlap_clean_offline11_equal_task_macro_0_100']:.2f} | "
            f"{row['provenance_clean9_equal_task_macro_0_100']:.2f} | "
            f"{pct(row['clean_task_success_rate'])} | "
            f"{pct(row['perturbed_task_success_rate'])} | "
            f"{pct(row['conditional_retention'])} | "
            f"{pct(row['regression_rate'])} |"
        )
    markdown.extend(
        [
            "",
            "## Strict Clean9 SE-B paired endpoint uncertainty",
            "",
            "Wilson two-sided 95% confidence intervals; every cell reports numerator/denominator, percentage, and CI.",
            "",
            "| model | clean TSR | perturbed TSR | conditional retention | regression |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in records:
        intervals = row["provenance_clean9_option_permutation_rate_estimates_95ci"]

        def ci_cell(name: str) -> str:
            value = intervals[name]
            if value["estimate"] is None:
                return f"{value['numerator']}/{value['denominator']} (N/A; 95% CI N/A)"
            low, high = value["ci95_percent"]
            return (
                f"{value['numerator']}/{value['denominator']} "
                f"({value['estimate_percent']:.2f}%; 95% CI {low:.2f}–{high:.2f}%)"
            )

        markdown.append(
            f"| {row['model']} | {ci_cell('clean_task_success_rate')} | "
            f"{ci_cell('perturbed_task_success_rate')} | "
            f"{ci_cell('conditional_retention')} | {ci_cell('regression_rate')} |"
        )
    markdown.extend(
        [
            "",
            "Official task metrics and derived controlled robustness remain separate. "
            "TCM-SE-A is excluded from local aggregates because its official score uses a remote judge.",
            "",
        ]
    )
    atomic_text(official_root / "suite_summary.md", "\n".join(markdown))
    return payload


def inspect_data(args: argparse.Namespace) -> Mapping[str, Any]:
    assets = validate_protocol_assets(
        args.mtcmb_repo,
        args.exclusions,
        args.task_manifest,
        args.schema,
        verify_git=not args.skip_git_check,
    )
    records = protocol.load_all_records(args.mtcmb_repo)
    items = protocol.build_inference_items(
        args.mtcmb_repo, records, assets["exclusions"]
    )
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "upstream": assets["upstream"],
        "training_exposure": {
            "uses_with_repetition": protocol.EXPECTED_EXPOSED_WITH_REPETITION,
            "unique_records": protocol.EXPECTED_EXPOSED_UNIQUE,
            "unique_by_task": protocol.EXPECTED_EXPOSED_BY_TASK,
        },
        "views": {
            "mtcmb_full": protocol.EXPECTED_FULL_N,
            "mtcmb_full_offline11": protocol.EXPECTED_OFFLINE11_N,
            "overlap_clean_all12": protocol.EXPECTED_OVERLAP_CLEAN_ALL12_N,
            "overlap_clean_offline11": protocol.EXPECTED_OVERLAP_CLEAN_OFFLINE11_N,
            "provenance_clean9": protocol.EXPECTED_CLEAN9_N,
        },
        "inference_calls": {
            "clean": sum(item.condition == "clean" for item in items),
            "controlled_option_permutation": sum(
                item.condition == "option_permutation" for item in items
            ),
            "total_per_model": len(items),
        },
        "controlled_pairs": {
            "overlap_clean_mcq": protocol.EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS,
            "provenance_clean9_mcq": protocol.EXPECTED_CLEAN9_MCQ_PAIRS,
        },
        "excluded_remote_judge": "TCM-SE-A (50 records)",
        "dataset_contract_fingerprint": assets["dataset_contract_fingerprint"],
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    def data_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--mtcmb-repo", type=Path, required=True)
        command.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
        command.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
        command.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
        command.add_argument(
            "--skip-git-check",
            action="store_true",
            help="tests only; hashes and counts remain fail-closed",
        )

    inspect_parser = commands.add_parser("inspect-data")
    data_arguments(inspect_parser)

    freeze_parser = commands.add_parser("freeze-suite")
    freeze_parser.add_argument("--suite", type=Path, required=True)
    freeze_parser.add_argument("--model", action="append", required=True, dest="models")
    freeze_parser.add_argument("--bert-model", type=Path, required=True)

    eval_parser = commands.add_parser("evaluate")
    data_arguments(eval_parser)
    eval_parser.add_argument("--suite", type=Path, required=True)
    eval_parser.add_argument("--model-label", choices=protocol.MODEL_ORDER, required=True)
    eval_parser.add_argument("--output-root", type=Path, required=True)
    eval_parser.add_argument("--dtype", default="bfloat16")
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--num-workers", type=int, default=FORMAL_NUM_WORKERS)
    eval_parser.add_argument("--devices", default=",".join(FORMAL_DEVICES))
    eval_parser.add_argument("--bert-device", default="cuda:0")
    eval_parser.add_argument("--bert-batch-size", type=int, default=32)
    eval_parser.add_argument("--preregistration-lock", type=Path, default=DEFAULT_LOCK)

    summary_parser = commands.add_parser("summarize-suite")
    data_arguments(summary_parser)
    summary_parser.add_argument("--suite", type=Path, required=True)
    summary_parser.add_argument("--output-root", type=Path, required=True)
    summary_parser.add_argument("--preregistration-lock", type=Path, default=DEFAULT_LOCK)
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if hasattr(args, "devices"):
        args.devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
        if len(args.devices) != args.num_workers:
            raise ValueError("--devices must supply exactly one logical device per worker")
    if args.command == "inspect-data":
        result = inspect_data(args)
    elif args.command == "freeze-suite":
        result = freeze_suite(args.suite, args.models, args.bert_model)
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "summarize-suite":
        result = summarize_suite(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
