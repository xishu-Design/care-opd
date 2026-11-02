#!/usr/bin/env python3
"""Single-GPU, result-blind deployment-cost audit for CARE-OPD.

The audit deliberately bypasses the training-only verifier/teacher pipeline.
Each checkpoint receives the same messages and exactly one ``model.generate``
invocation per measured request.  This makes deployment cost, rather than the
cost of constructing OPD training examples, the object being measured.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_LABELS = ("clean_sft", "gkd_full", "care_opd")
EXPECTED_VARIANTS = ("bad_history", "clean", "evidence_conflict", "source_outage")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, matching NumPy's default definition."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _selection_score(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{item_id}".encode("utf-8")).hexdigest()


def select_result_blind_sample(
    request_rows: Sequence[Mapping[str, Any]],
    family_clean_ids: Sequence[str],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a variant-balanced sample using IDs only, never model results."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    allowed = {str(item_id) for item_id in family_clean_ids}
    if len(allowed) != len(family_clean_ids):
        raise ValueError("family-clean request IDs contain duplicates")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in request_rows:
        item_id = str(row.get("id") or "")
        if not item_id or item_id in by_id:
            raise ValueError(f"empty or duplicate request ID: {item_id!r}")
        by_id[item_id] = row
    missing = allowed - set(by_id)
    if missing:
        raise ValueError(f"{len(missing)} family-clean IDs are absent from requests")

    pools: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item_id in allowed:
        row = by_id[item_id]
        pools[str(row.get("variant") or "unknown")].append(row)
    variants = tuple(variant for variant in EXPECTED_VARIANTS if variant in pools)
    extras = tuple(sorted(set(pools) - set(variants)))
    variants = variants + extras
    if not variants:
        raise ValueError("family-clean pool is empty")

    base, remainder = divmod(sample_size, len(variants))
    quotas = {
        variant: base + int(index < remainder)
        for index, variant in enumerate(variants)
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for variant in variants:
        ranked = sorted(
            pools[variant],
            key=lambda row: (
                _selection_score(seed, str(row["id"])),
                str(row["id"]),
            ),
        )
        take = quotas[variant]
        if len(ranked) < take:
            raise ValueError(
                f"variant {variant!r} has {len(ranked)} rows but quota is {take}"
            )
        for raw in ranked[:take]:
            row = dict(raw)
            row["selection_hash"] = _selection_score(seed, str(row["id"]))
            selected.append(row)
            selected_ids.add(str(row["id"]))
    if len(selected) != sample_size or len(selected_ids) != sample_size:
        raise AssertionError("result-blind sample did not close")
    # A fixed hash order also makes warm-up and measured order identical by model.
    return sorted(selected, key=lambda row: (row["selection_hash"], str(row["id"])))


def summarize_observations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no measured observations")
    generation = [float(row["generation_seconds"]) for row in rows]
    end_to_end = [float(row["end_to_end_seconds"]) for row in rows]
    output_tokens = [int(row["output_tokens"]) for row in rows]
    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    total_generation_seconds = sum(generation)
    total_end_to_end_seconds = sum(end_to_end)
    total_output_tokens = sum(output_tokens)
    return {
        "requests": len(rows),
        "generation_latency_seconds": {
            "mean": statistics.fmean(generation),
            "p50": percentile(generation, 0.50),
            "p95": percentile(generation, 0.95),
        },
        "end_to_end_latency_seconds": {
            "mean": statistics.fmean(end_to_end),
            "p50": percentile(end_to_end, 0.50),
            "p95": percentile(end_to_end, 0.95),
        },
        "generated_tokens_per_second": (
            total_output_tokens / total_generation_seconds
            if total_generation_seconds > 0
            else None
        ),
        "output_tokens": {
            "total": total_output_tokens,
            "mean": statistics.fmean(output_tokens),
            "p50": percentile(output_tokens, 0.50),
            "p95": percentile(output_tokens, 0.95),
        },
        "prompt_tokens": {
            "total": sum(prompt_tokens),
            "mean": statistics.fmean(prompt_tokens),
        },
        "total_generation_seconds": total_generation_seconds,
        "total_end_to_end_seconds": total_end_to_end_seconds,
        "variant_counts": dict(sorted(Counter(str(row["variant"]) for row in rows).items())),
    }


def _render_messages(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    """Match the formal QZBC no-thinking chat-template boundary."""

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer lacks apply_chat_template; refusing protocol fallback")
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise ValueError(
            "tokenizer chat template lacks enable_thinking=False; refusing fallback"
        ) from exc


def _model_disk_identity(raw_suite_entry: Mapping[str, Any]) -> dict[str, Any]:
    inventory = raw_suite_entry.get("inventory")
    if not isinstance(inventory, list):
        raise ValueError("model suite entry lacks inventory")
    weight_rows = [
        row
        for row in inventory
        if Path(str(row.get("name") or "")).suffix.lower() in WEIGHT_SUFFIXES
    ]
    return {
        "checkpoint_files": len(inventory),
        "checkpoint_bytes": sum(int(row.get("bytes", 0)) for row in inventory),
        "weight_files": len(weight_rows),
        "weight_bytes": sum(int(row.get("bytes", 0)) for row in weight_rows),
        "weight_gib": sum(int(row.get("bytes", 0)) for row in weight_rows) / 2**30,
    }


def _load_model_and_tokenizer(model_path: str, dtype: str) -> tuple[Any, Any]:
    from care_opd.models import ModelArguments, load_causal_lm, load_tokenizer

    model_args = ModelArguments(
        student_model_name_or_path=model_path,
        tokenizer_name_or_path=model_path,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    tokenizer = load_tokenizer(model_args)
    model = load_causal_lm(model_path, model_args, trainable=False)
    return model, tokenizer


def _generate_once(
    *,
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    device: Any,
    max_new_tokens: int,
    measure: bool,
) -> dict[str, Any] | None:
    import torch

    begin_e2e = time.perf_counter()
    messages = row.get("student_messages")
    if not isinstance(messages, list):
        raise ValueError(f"request {row.get('id')}: student_messages is absent")
    prompt = _render_messages(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])
    torch.cuda.synchronize(device)
    begin_generation = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - begin_generation
    new_tokens = output[0, prompt_tokens:]
    output_tokens = int(new_tokens.numel())
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    end_to_end_seconds = time.perf_counter() - begin_e2e
    if not measure:
        return None
    return {
        "id": str(row["id"]),
        "variant": str(row.get("variant") or "unknown"),
        "selection_hash": str(row["selection_hash"]),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "generation_seconds": generation_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "generated_tokens_per_second": (
            output_tokens / generation_seconds if generation_seconds > 0 else None
        ),
        # Store a digest, not response text: the efficiency endpoint never reads
        # correctness and therefore cannot influence result-blind selection.
        "output_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
        "terminated_before_cap": output_tokens < max_new_tokens,
    }


def _runtime_identity(torch: Any, device: Any) -> dict[str, Any]:
    import transformers

    props = torch.cuda.get_device_properties(device)
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "device_name": props.name,
        "device_total_memory_bytes": int(props.total_memory),
        "visible_cuda_devices": torch.cuda.device_count(),
    }


def _architecture_identity(model: Any) -> dict[str, Any]:
    config = model.config
    architectures = getattr(config, "architectures", None) or []
    return {
        "model_class": type(model).__name__,
        "config_model_type": str(getattr(config, "model_type", "")),
        "config_architectures": [str(item) for item in architectures],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "config_sha256": canonical_hash(config.to_dict()),
    }


def evaluate_one_model(
    *,
    label: str,
    model_entry: Mapping[str, Any],
    raw_suite_entry: Mapping[str, Any],
    sample: Sequence[Mapping[str, Any]],
    output_root: Path,
    device_name: str,
    dtype: str,
    max_new_tokens: int,
    warmup_requests: int,
    seed: int,
    sample_fingerprint: str,
    rerun: bool,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    model_root = output_root / label
    complete_path = model_root / "COMPLETE.json"
    generation_contract = {
        "sample_fingerprint": sample_fingerprint,
        "model_label": label,
        "model_fingerprint": str(model_entry["model_fingerprint"]),
        "device": device_name,
        "dtype": dtype,
        "batch_size": 1,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "enable_thinking": False,
        "seed": seed,
        "warmup_requests": warmup_requests,
        "measured_requests": len(sample),
        "inference_boundary": "one_hf_checkpoint_one_generate_call_per_request",
    }
    run_fingerprint = canonical_hash(generation_contract)
    if complete_path.is_file() and not rerun:
        complete = read_json(complete_path)
        if complete.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"{label}: completed contract differs; use a new output root or --rerun"
            )
        metrics_path = model_root / "metrics.json"
        if sha256_file(metrics_path) != complete.get("metrics_sha256"):
            raise ValueError(f"{label}: completed metrics digest changed")
        print(f"[skip] validated efficiency result: {label}", flush=True)
        return read_json(metrics_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the deployment efficiency audit")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device")
    torch.cuda.set_device(device)
    set_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    print(f"[load] {label}: {model_entry['canonical_path']}", flush=True)
    load_start = time.perf_counter()
    model, tokenizer = _load_model_and_tokenizer(
        str(model_entry["canonical_path"]), dtype
    )
    model.to(device)
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    architecture = _architecture_identity(model)
    disk = _model_disk_identity(raw_suite_entry)
    loaded_allocated = int(torch.cuda.memory_allocated(device))
    loaded_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    try:
        warmup = list(sample[:warmup_requests])
        for index, row in enumerate(warmup, 1):
            print(f"[warmup] {label} {index}/{len(warmup)}", flush=True)
            _generate_once(
                model=model,
                tokenizer=tokenizer,
                row=row,
                device=device,
                max_new_tokens=max_new_tokens,
                measure=False,
            )
        # Reset after warm-up. Current model allocations remain the lower bound
        # of peak allocation, so the metric includes resident deployment memory.
        torch.cuda.reset_peak_memory_stats(device)
        observations: list[dict[str, Any]] = []
        for index, row in enumerate(sample, 1):
            observation = _generate_once(
                model=model,
                tokenizer=tokenizer,
                row=row,
                device=device,
                max_new_tokens=max_new_tokens,
                measure=True,
            )
            assert observation is not None
            observations.append(observation)
            if index == 1 or index % 10 == 0 or index == len(sample):
                print(f"[measure] {label} {index}/{len(sample)}", flush=True)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        summary = summarize_observations(observations)
        metrics: dict[str, Any] = {
            "schema_version": "care-opd-efficiency-model-v1",
            "status": "COMPLETE",
            "label": label,
            "run_fingerprint": run_fingerprint,
            "generation_contract": generation_contract,
            "model": {
                "canonical_path": str(model_entry["canonical_path"]),
                "actual_model_fingerprint": str(model_entry["model_fingerprint"]),
                **architecture,
                **disk,
                "load_seconds": load_seconds,
            },
            "runtime": _runtime_identity(torch, device),
            "memory": {
                "loaded_allocated_bytes": loaded_allocated,
                "loaded_reserved_bytes": loaded_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "inference_increment_peak_allocated_bytes": max(
                    0, peak_allocated - loaded_allocated
                ),
            },
            "measurements": summary,
            "deployment_call_audit": {
                "loaded_hf_checkpoints_at_once": 1,
                "measured_generation_calls": len(sample),
                "warmup_generation_calls": len(warmup),
                "agent_loop_calls": 0,
                "verifier_calls": 0,
                "teacher_model_calls": 0,
                "retrieval_or_tool_calls": 0,
                "scope": (
                    "Audit-script execution path only; one local HF checkpoint is "
                    "loaded and model.generate is called directly."
                ),
            },
        }
        model_root.mkdir(parents=True, exist_ok=True)
        observation_path = model_root / "per_request.jsonl"
        metrics_path = model_root / "metrics.json"
        atomic_jsonl(observation_path, observations)
        atomic_json(metrics_path, metrics)
        atomic_json(
            complete_path,
            {
                "schema_version": "care-opd-efficiency-complete-v1",
                "status": "COMPLETE",
                "run_fingerprint": run_fingerprint,
                "metrics_sha256": sha256_file(metrics_path),
                "per_request_sha256": sha256_file(observation_path),
            },
        )
        return metrics
    finally:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _write_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "label",
        "requests",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "generated_tokens_per_s",
        "mean_output_tokens",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "parameter_count",
        "weight_gib",
        "model_fingerprint",
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for result in results:
                measurements = result["measurements"]
                memory = result["memory"]
                model = result["model"]
                latency = measurements["generation_latency_seconds"]
                writer.writerow(
                    {
                        "label": result["label"],
                        "requests": measurements["requests"],
                        "latency_mean_s": latency["mean"],
                        "latency_p50_s": latency["p50"],
                        "latency_p95_s": latency["p95"],
                        "generated_tokens_per_s": measurements["generated_tokens_per_second"],
                        "mean_output_tokens": measurements["output_tokens"]["mean"],
                        "peak_allocated_gib": memory["peak_allocated_bytes"] / 2**30,
                        "peak_reserved_gib": memory["peak_reserved_bytes"] / 2**30,
                        "parameter_count": model["parameter_count"],
                        "weight_gib": model["weight_gib"],
                        "model_fingerprint": model["actual_model_fingerprint"],
                    }
                )
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    rows = summary["models"]
    lines = [
        "# CARE-OPD deployment efficiency audit",
        "",
        "All checkpoints are measured sequentially on the same logical GPU with the "
        "same result-blind QZBC family-clean sample, no-thinking chat template, greedy "
        "decoding, batch size 1, precision, and output cap.",
        "",
        "| Model | n | Latency mean / p50 / p95 (s) | Generated tok/s | Output tok (mean) | Peak allocated / reserved (GiB) | Params | Weight GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        measured = row["measurements"]
        latency = measured["generation_latency_seconds"]
        memory = row["memory"]
        model = row["model"]
        lines.append(
            "| {label} | {n} | {mean:.4f} / {p50:.4f} / {p95:.4f} | "
            "{tps:.2f} | {out:.1f} | {alloc:.3f} / {reserved:.3f} | "
            "{params:,} | {weight:.3f} |".format(
                label=row["label"],
                n=measured["requests"],
                mean=latency["mean"],
                p50=latency["p50"],
                p95=latency["p95"],
                tps=measured["generated_tokens_per_second"],
                out=measured["output_tokens"]["mean"],
                alloc=memory["peak_allocated_bytes"] / 2**30,
                reserved=memory["peak_reserved_bytes"] / 2**30,
                params=model["parameter_count"],
                weight=model["weight_gib"],
            )
        )
    lines.extend(
        [
            "",
            "## Deployment-boundary audit",
            "",
            "For every model, the audit loads exactly one local Hugging Face checkpoint "
            "and invokes `model.generate` once per measured request. It invokes no Agent "
            "loop, verifier, teacher model, retriever, or external tool. Consequently, any "
            "training-time CARE machinery is absent from the measured inference path.",
            "",
            "## Claim boundary",
            "",
            "This is a single-GPU descriptive deployment audit. Output length can change "
            "wall-clock latency, so latency, generated-token throughput, and output-token "
            "counts are reported together. It is not a task-success experiment and it does "
            "not establish statistical equivalence by itself.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_labels(raw: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("--labels must contain distinct nonempty labels")
    return labels


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Import the formal suite verifier only for the actual GPU run. Pure helper
    # functions and unit tests remain torch/transformers independent.
    from public_eval.qzbc_grouped.protocol import validate_model_suite, suite_entry

    labels = _parse_labels(args.labels)
    suite = validate_model_suite(args.suite, rehash_labels=labels)
    audit = read_json(args.data_audit)
    family_view = audit.get("family_clean_view")
    if not isinstance(family_view, Mapping):
        raise ValueError("DATA_AUDIT.json lacks family_clean_view")
    family_clean_ids = family_view.get("request_ids")
    if not isinstance(family_clean_ids, list):
        raise ValueError("DATA_AUDIT family_clean_view lacks request_ids")
    requests = read_jsonl(args.requests)
    sample = select_result_blind_sample(
        requests,
        [str(item) for item in family_clean_ids],
        sample_size=args.sample_size,
        seed=args.seed,
    )
    sample_payload = {
        "schema_version": "care-opd-efficiency-sample-v1",
        "selection": (
            "Within each variant, take the smallest SHA256(seed + NUL + request_id); "
            "selection never reads model outputs."
        ),
        "seed": args.seed,
        "sample_size": len(sample),
        "family_clean_pool_size": len(family_clean_ids),
        "variant_counts": dict(sorted(Counter(str(row["variant"]) for row in sample).items())),
        "request_ids": [str(row["id"]) for row in sample],
        "selection_hashes": [str(row["selection_hash"]) for row in sample],
        "requests_sha256": sha256_file(args.requests),
        "data_audit_sha256": sha256_file(args.data_audit),
    }
    sample_payload["sample_fingerprint"] = canonical_hash(sample_payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample_manifest.json"
    if sample_path.is_file() and read_json(sample_path) != sample_payload:
        raise ValueError("existing sample manifest differs; use a new output directory")
    atomic_json(sample_path, sample_payload)

    raw_by_label = {
        str(row.get("label")): row for row in suite["payload"].get("models", [])
    }
    results: list[dict[str, Any]] = []
    for label in labels:
        if label not in raw_by_label:
            raise ValueError(f"model suite lacks raw entry for {label}")
        results.append(
            evaluate_one_model(
                label=label,
                model_entry=suite_entry(suite, label),
                raw_suite_entry=raw_by_label[label],
                sample=sample,
                output_root=args.output_dir,
                device_name=args.device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                warmup_requests=args.warmup_requests,
                seed=args.seed,
                sample_fingerprint=str(sample_payload["sample_fingerprint"]),
                rerun=args.rerun,
            )
        )
    summary: dict[str, Any] = {
        "schema_version": "care-opd-efficiency-summary-v1",
        "status": "COMPLETE",
        "suite_fingerprint": suite["suite_fingerprint"],
        "sample_fingerprint": sample_payload["sample_fingerprint"],
        "labels": list(labels),
        "models": results,
        "deployment_protocol": {
            "sequential_same_process_and_device": True,
            "one_checkpoint_loaded_at_a_time": True,
            "same_request_order": True,
            "same_generation_contract": True,
            "care_inference_extra_agent_or_model_calls": 0,
        },
    }
    atomic_json(args.output_dir / "efficiency_metrics.json", summary)
    _write_csv(args.output_dir / "efficiency_metrics.csv", results)
    _write_markdown(args.output_dir / "efficiency_audit.md", summary)
    atomic_json(
        args.output_dir / "COMPLETE.json",
        {
            "schema_version": "care-opd-efficiency-suite-complete-v1",
            "status": "COMPLETE",
            "suite_fingerprint": suite["suite_fingerprint"],
            "sample_fingerprint": sample_payload["sample_fingerprint"],
            "metrics_sha256": sha256_file(args.output_dir / "efficiency_metrics.json"),
            "csv_sha256": sha256_file(args.output_dir / "efficiency_metrics.csv"),
            "report_sha256": sha256_file(args.output_dir / "efficiency_audit.md"),
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-measure completed model directories under the same contract.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup_requests < 0 or args.warmup_requests > args.sample_size:
        raise ValueError("warmup_requests must be in [0, sample_size]")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    run(args)
    print(f"EFFICIENCY_AUDIT=COMPLETE: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
