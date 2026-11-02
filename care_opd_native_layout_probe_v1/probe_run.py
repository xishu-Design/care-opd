#!/usr/bin/env python3
"""Frozen four-model layout diagnostic; prepare/status/score are CPU only."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil
import sys
import time

import probe_core as p


def import_runtime():
    folder = p.runtime_dir().resolve()
    for name in ("common", "evaluate"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve().parent != folder:
            raise RuntimeError("runtime module collision: " + name)
    sys.path.insert(0, str(folder))
    return importlib.import_module("evaluate")


def preflight(work, full_hash=False, runtime=None):
    manifest, tasks, _, prompts = p.load_work(work)
    if shutil.disk_usage(work).free < 128 * 1024 * 1024:
        raise RuntimeError("less than 128 MiB free on output filesystem")
    runtime = runtime or import_runtime()
    config = {"models": manifest["models"], "generation": manifest["generation"]}
    runtime.check_stack(config)  # four text models only; never load the VLM
    env = runtime.environment()
    software = {k: v for k, v in env.items() if k not in {
        "cuda_visible_devices", "gpu_uuid", "gpu_total_memory_bytes"}}
    if software != manifest["source_software_environment"]:
        delta = {k: {"source": manifest["source_software_environment"].get(k), "current": software.get(k)}
                 for k in set(software) | set(manifest["source_software_environment"])
                 if software.get(k) != manifest["source_software_environment"].get(k)}
        raise RuntimeError("software/GPU-class drift from cached control: " + p.c.canonical(delta))
    previous_path = Path(work) / "preflight.json"
    previous = p.c.read_json(previous_path) if previous_path.exists() else None
    assets, counts = {}, {}
    from transformers import AutoConfig, AutoTokenizer
    for arm in p.ARMS:
        old = (previous or {}).get("assets", {}).get(arm, manifest["model_assets"][arm])
        asset = runtime.fingerprint_model(Path(manifest["models"][arm]), previous=old, full=full_hash)
        if asset["sha256"] != manifest["model_assets"][arm]["sha256"]:
            raise RuntimeError("checkpoint/tokenizer changed since original run: " + arm)
        assets[arm] = asset
        tokenizer = AutoTokenizer.from_pretrained(manifest["models"][arm], local_files_only=True, trust_remote_code=False)
        cfg = AutoConfig.from_pretrained(manifest["models"][arm], local_files_only=True, trust_remote_code=False)
        limit = getattr(getattr(cfg, "text_config", cfg), "max_position_embeddings", None)
        lengths = []
        for row in prompts.values():
            rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
            count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if count > manifest["generation"]["max_input_tokens"]:
                raise RuntimeError("adapted input too long; no truncation allowed")
            if isinstance(limit, int) and count + manifest["generation"]["max_new_tokens"] > limit:
                raise RuntimeError("adapted input/output exceeds model context")
            lengths.append(count)
        counts[arm] = {"min": min(lengths), "max": max(lengths), "n": len(lengths)}
        del tokenizer
    frozen = {"protocol_sha256": manifest["protocol_sha256"], "software": software,
              "model_hashes": {a: x["sha256"] for a, x in assets.items()}, "prompt_tokens": counts}
    if previous and previous["frozen"] != frozen:
        raise RuntimeError("preflight drift on resume")
    if not previous:
        p.c.json_once(previous_path, {"frozen": frozen, "assets": assets})
    p.c.append_row(Path(work) / "device_history.jsonl", {"timestamp_unix": time.time(), "pid": os.getpid(),
                    "environment": env, "full_hash_requested": full_hash,
                    "protocol_sha256": manifest["protocol_sha256"]})
    print("PREFLIGHT_PASS", p.c.canonical(counts), flush=True)
    return runtime


def run(work, full_hash=False, runtime=None, skip_preflight=False):
    manifest, tasks, obs, prompts = p.load_work(work)
    # skip_preflight exists only for the CPU fake-runtime tests, never a CLI flag.
    if skip_preflight and runtime is None:
        raise ValueError("test runtime required")
    runtime = runtime if skip_preflight else preflight(work, full_hash, runtime)
    for arm in p.ARMS:
        path = Path(work) / f"responses/{arm}.jsonl"
        rows = p.load_new_rows(work, arm, require_complete=False)
        old = p.c.index_rows(p.c.read_jsonl(Path(work) / f"baseline/{arm}.jsonl"))
        generator = None
        try:
            for task in tasks:
                ident = task["task_id"]
                if ident in rows:
                    continue
                try:
                    if generator is None:
                        generator = runtime.HFGenerator(manifest["models"][arm], manifest["generation"], visual=False)
                    result = generator.generate(prompts[ident]["messages"], None, manifest["generation"]["max_new_tokens"])
                    if (not isinstance(result.get("raw_text"), str)
                            or result.get("finish_reason") not in ("stop", "length")
                            or result.get("image_seen") is not False):
                        raise RuntimeError("runtime returned invalid text generation record")
                    row = {**result, **p.new_binding(manifest, task, prompts, obs, arm),
                           "paired_baseline_row_sha256": old[ident]["row_sha256"],
                           "paired_baseline_input_sha256": old[ident]["input_sha256"],
                           "paired_baseline_protocol_sha256": old[ident]["protocol_sha256"],
                           "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
                    row["row_sha256"] = p.c.digest(row)
                    # First output persisted before any scoring. Empty/invalid/length
                    # outputs are completed and will never be content-retried.
                    p.c.append_row(path, row)
                    rows[ident] = row
                    print(f"PROGRESS arm={arm} saved={len(rows)}/{len(tasks)} finish={row['finish_reason']}", flush=True)
                except Exception as exc:
                    p.c.append_row(Path(work) / "infrastructure_errors.jsonl", {
                        "arm": arm, "task_id": ident, "timestamp_unix": time.time(),
                        "error": f"{type(exc).__name__}: {exc}", "content_retry": False})
                    raise
        finally:
            if generator is not None:
                generator.close()
    print("TEXT_GENERATION_COMPLETE", len(tasks) * len(p.ARMS), flush=True)


def status(work):
    manifest, tasks, _, _ = p.load_work(work)
    print(f"DEVELOPMENT_DIAGNOSTIC blocks={manifest['blocks']} tasks_per_arm={len(tasks)}")
    for arm in p.ARMS:
        print(f"{arm}: {len(p.load_new_rows(work, arm, require_complete=False))}/{len(tasks)}")
    lock = Path(work) / ".boundary.lock"
    if lock.exists():
        print("LOCK_PRESENT", lock.read_text(encoding="utf-8"))
    print("Counts only. No model efficacy or acceptance claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "check", "preflight", "run", "resume", "status", "score", "all"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU must be nonnegative")
    if args.phase in ("prepare", "all") and args.source is None:
        parser.error("prepare/all requires --source")
    os.umask(0o077)
    work = args.work.expanduser().resolve()
    if args.phase == "status":
        status(work)
        return
    # Prevent source overwrite even before taking a lock/making a directory.
    if args.source:
        src = args.source.expanduser().resolve()
        if work == src or work.is_relative_to(src) or src.is_relative_to(work):
            parser.error("work and source must be disjoint")
    with p.c.run_lock(work):
        if args.phase in ("prepare", "all"):
            p.prepare(args.source, work, args.blocks)
        if args.phase == "check":
            m, *_ = p.load_work(work)
            print("STATIC_CHECK_PASS", m["protocol_sha256"], "gold_opened=false GPU_calls=0")
        if args.phase in ("preflight", "run", "resume", "all"):
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1",
                              TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            if args.phase == "preflight":
                preflight(work, args.full_hash)
            else:
                run(work, args.full_hash)
        if args.phase in ("score", "run", "resume", "all"):
            from probe_score import score
            print(p.c.canonical(score(work)), flush=True)


if __name__ == "__main__":
    main()
