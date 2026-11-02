#!/usr/bin/env python3
"""One frozen visual experiment, coordinated across physical GPUs 4 and 5.

The parent holds the existing exclusive run-root lock for the entire job.
Spawned workers write disjoint arms; all text arms share the very same completed
observation file. No training, sample filtering, content retries or score changes.
"""
from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

import evaluate
from common import ARMS, append_row, file_sha256, load_tasks, run_lock


GPU_PLAN = {"4": ("direct", "gkd_full"),
            "5": ("clean_sft", "tropd_full", "care_opd")}
EXTRA_SOURCES = ("gpu45.py", "run_visual_transfer_gpu45.sh")


def fleet_environment(environments: dict[str, dict]) -> dict:
    if set(environments) != set(GPU_PLAN):
        raise ValueError("GPU45 requires environments for physical GPUs 4 and 5")
    arms = ["vlm_direct" if arm == "direct" else arm
            for sequence in GPU_PLAN.values() for arm in sequence]
    if sorted(arms) != sorted(ARMS):
        raise ValueError("GPU45 plan must include each answer arm exactly once")
    for gpu, info in environments.items():
        if info.get("cuda_visible_devices") != gpu:
            raise ValueError(f"physical GPU selection differs for GPU {gpu}")
    same = ("python", "cuda_runtime", "torch", "transformers", "Pillow", "accelerate",
            "gpu_name", "gpu_total_memory_bytes")
    for key in same:
        if key not in environments["4"] or environments["4"][key] != environments["5"].get(key):
            raise ValueError(f"GPU45 environment mismatch: {key}")
    uuid4, uuid5 = (environments[gpu].get("gpu_uuid") for gpu in ("4", "5"))
    if uuid4 not in (None, "unavailable") and uuid4 == uuid5:
        raise ValueError("GPU45 resolved both physical indices to the same GPU UUID")
    return {"schema_version": "care_opd.visual_transfer.gpu45_environment.v1",
            "devices": environments,
            "schedule": {"observe": "4", "parallel": {g: list(v) for g, v in GPU_PLAN.items()}},
            "coordination": "parent_exclusive_root_lock_spawned_disjoint_arm_workers"}


def _select_gpu(gpu: str) -> None:
    # Called in a fresh spawn process, before importing torch or initializing CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _inspect(config: dict, gpu: str, connection) -> None:
    try:
        _select_gpu(gpu)
        evaluate.check_stack(config)
        connection.send({"ok": True, "environment": evaluate.environment()})
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def inspect_environment(config: dict, gpu: str) -> dict:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_inspect, args=(config, gpu, sender))
    process.start()
    sender.close()
    try:
        try:
            result = receiver.recv()
        except EOFError as exc:
            raise RuntimeError(f"GPU {gpu} environment probe exited without a result") from exc
    finally:
        receiver.close()
        process.join()
    if process.exitcode != 0 or not result.get("ok"):
        raise RuntimeError(f"GPU {gpu} preflight failed: {result.get('error', process.exitcode)}")
    return result["environment"]


def verify_worker(config: dict, manifest: dict, gpu: str) -> None:
    expected = manifest["environment"]["devices"][gpu]
    if evaluate.environment() != expected:
        raise ValueError(f"GPU {gpu} environment differs from the frozen manifest")
    source_hashes = evaluate.source_fingerprints()
    for name in EXTRA_SOURCES:
        source_hashes[name] = file_sha256(Path(__file__).resolve().parent / name)
    if source_hashes != manifest["code_sha256"]:
        raise ValueError("GPU45 worker source code differs from frozen manifest")
    for name, path in config["models"].items():
        asset = evaluate.fingerprint_model(Path(path), previous=manifest["model_assets"][name])
        if asset["sha256"] != manifest["model_fingerprints"][name]:
            raise ValueError(f"GPU45 checkpoint content changed for {name}")


def _worker(config: dict, manifest: dict, tasks: list[dict], gpu: str,
            stages: tuple[str, ...], log_path: str) -> None:
    _select_gpu(gpu)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                print(f"WORKER_START physical_gpu={gpu} pid={os.getpid()} stages={stages}", flush=True)
                verify_worker(config, manifest, gpu)
                evaluate.verify_images(tasks)
                for stage in stages:
                    if stage in ("observe", "direct"):
                        evaluate.run_phase(config, manifest, tasks, stage)
                    else:
                        evaluate.run_phase(config, manifest, tasks, "generate", stage)
                verify_worker(config, manifest, gpu)
                print(f"WORKER_COMPLETE physical_gpu={gpu} stages={stages}", flush=True)
            except BaseException:
                traceback.print_exc()
                raise


def run_workers(config: dict, manifest: dict, tasks: list[dict], plan: dict[str, tuple[str, ...]],
                *, stage_label: str) -> None:
    context = mp.get_context("spawn")
    root = Path(config["run_root"])
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        for gpu, stages in plan.items():
            log = logs / f"gpu{gpu}.{stage_label}.log"
            child = context.Process(target=_worker, args=(config, manifest, tasks, gpu, stages, str(log)))
            child.start()
            processes.append((gpu, child))
            append_row(root / "runtime" / "workers.jsonl", {
                "event": "started", "parent_pid": os.getpid(), "pid": child.pid,
                "physical_gpu": gpu, "stages": list(stages), "log": str(log)})
            print(f"GPU45_WORKER gpu={gpu} pid={child.pid} log={log}", flush=True)
    finally:
        # Do not release the parent root lock while a child is still writing.
        # A failure in one lane never kills the unrelated work in another lane.
        interrupted = False
        for gpu, child in processes:
            while True:
                try:
                    child.join()
                    break
                except KeyboardInterrupt:
                    interrupted = True
        # All children must be joined BEFORE fallible audit writes (disk full,
        # permissions, etc.). Otherwise an audit error could release the root
        # lock while a later child was still appending model outputs.
        for gpu, child in processes:
            append_row(root / "runtime" / "workers.jsonl", {
                "event": "exited", "pid": child.pid, "physical_gpu": gpu,
                "exitcode": child.exitcode})
        if interrupted:
            raise KeyboardInterrupt
    failures = [(gpu, child.exitcode) for gpu, child in processes if child.exitcode != 0]
    if failures:
        raise RuntimeError(f"GPU45 stages failed: {failures}; inspect lane logs and resume unchanged")


def execute(config: dict, phase: str, *, full_hash: bool = False) -> None:
    # One root lock excludes legacy all/observe/generate/preflight commands too.
    with run_lock(Path(config["run_root"])):
        tasks, selection = load_tasks(Path(config["tasks_dir"]))
        evaluate.verify_images(tasks)
        environments = {gpu: inspect_environment(config, gpu) for gpu in GPU_PLAN}
        manifest = evaluate.bind_run(
            config, tasks, selection, full_hash=full_hash,
            execution_environment=fleet_environment(environments), extra_sources=EXTRA_SOURCES)
        print(f"GPU45_PREFLIGHT=PASS tasks={len(tasks)} protocol={manifest['protocol_sha256']}", flush=True)
        if phase == "preflight":
            return
        if phase != "all":
            raise ValueError("GPU45 phase must be preflight or all")
        run_workers(config, manifest, tasks, {"4": ("observe",)}, stage_label="observe")
        run_workers(config, manifest, tasks, GPU_PLAN, stage_label="answers")
        import score_results
        report = score_results.score_results(tasks_dir=Path(config["tasks_dir"]),
                                            run_root=Path(config["run_root"]),
                                            output=Path(config["run_root"]) / "summary")
        if report["status"] != "COMPLETE":
            raise RuntimeError("GPU45 scoring incomplete; retain all results and inspect summary")
        print(f"GPU45_COMPLETE summary={config['run_root']}/summary", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "all"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    execute(evaluate.configuration(args.config), args.phase, full_hash=args.full_hash)


if __name__ == "__main__":
    main()
