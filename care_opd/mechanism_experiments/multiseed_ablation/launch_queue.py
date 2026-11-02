#!/usr/bin/env python3
"""Execute a frozen multi-seed training plan through a bounded GPU queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, BinaryIO, Sequence

try:
    from .common import (
        QUEUE_STATUS_SCHEMA,
        TRAIN_SUITE_SCHEMA,
        atomic_json,
        load_json_object,
        require_keys,
        verify_file_binding,
    )
except ImportError:
    from common import (
        QUEUE_STATUS_SCHEMA,
        TRAIN_SUITE_SCHEMA,
        atomic_json,
        load_json_object,
        require_keys,
        verify_file_binding,
    )


def _load_jobs(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite = load_json_object(plan_path, schema=TRAIN_SUITE_SCHEMA)
    seeds = suite.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise ValueError("multi-seed plan must contain at least two seeds")
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for index, entry in enumerate(seeds):
        if not isinstance(entry, dict):
            raise ValueError(f"seeds[{index}] must be an object")
        require_keys(entry, ("training_seed", "plan"), where=f"seeds[{index}]")
        seed = entry["training_seed"]
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seeds[{index}].training_seed must be a non-negative integer")
        _, nested = verify_file_binding(
            entry["plan"], where=f"seeds[{index}].plan", schema="care_opd.causal_ablation_plan.v1"
        )
        if nested.get("budget_contract", {}).get("seed") != seed:
            raise ValueError(f"seeds[{index}]: nested training seed does not match suite binding")
        for arm in nested.get("arms", []):
            if not isinstance(arm, dict):
                raise ValueError(f"seed {seed}: arm must be an object")
            require_keys(arm, ("name", "output_dir", "command"), where=f"seed {seed} arm")
            name, command = arm["name"], arm["command"]
            if not isinstance(name, str) or not name:
                raise ValueError(f"seed {seed}: invalid arm name")
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                raise ValueError(f"seed {seed} arm {name}: command must be a non-empty string list")
            if "--nproc_per_node=1" not in command:
                raise ValueError("GPU queue requires plans built with --nproc-per-node 1")
            key = (seed, name)
            if key in seen:
                raise ValueError(f"duplicate queued job {key}")
            seen.add(key)
            jobs.append(
                {
                    "training_seed": seed,
                    "arm": name,
                    "output_dir": str(arm["output_dir"]),
                    "command": command,
                }
            )
    return suite, jobs


def _is_complete(job: dict[str, Any], *, expected_steps: int) -> bool:
    final = Path(job["output_dir"]) / "final"
    metadata_path = final / "run_metadata.json"
    adapter_config = final / "adapter_config.json"
    adapter_weights = final / "adapter_model.safetensors"
    metrics_path = Path(job["output_dir"]) / "metrics.jsonl"
    if not (
        metadata_path.is_file()
        and adapter_config.is_file()
        and adapter_config.stat().st_size > 0
        and adapter_weights.is_file()
        and adapter_weights.stat().st_size > 0
        and metrics_path.is_file()
    ):
        return False
    metadata = load_json_object(metadata_path)
    identity_matches = (
        metadata.get("global_step") == expected_steps
        and metadata.get("training", {}).get("seed") == job["training_seed"]
        and metadata.get("training", {}).get("max_steps") == expected_steps
    )
    if not identity_matches:
        return False
    complete = False
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{metrics_path}:{line_number}: invalid JSON") from exc
            if row.get("event") == "complete" and row.get("global_step") == expected_steps:
                complete = True
    return complete


def _validate_existing_identity(job: dict[str, Any], *, expected_steps: int) -> None:
    """Reject a final artifact from a different seed/budget before resuming."""

    metadata_path = Path(job["output_dir"]) / "final" / "run_metadata.json"
    if not metadata_path.is_file():
        return
    metadata = load_json_object(metadata_path)
    training = metadata.get("training")
    actual_seed = training.get("seed") if isinstance(training, dict) else None
    actual_steps = training.get("max_steps") if isinstance(training, dict) else None
    if actual_seed != job["training_seed"] or actual_steps != expected_steps:
        raise ValueError(
            f"contaminated output root {job['output_dir']}: final metadata has "
            f"seed={actual_seed!r}, max_steps={actual_steps!r}; expected "
            f"seed={job['training_seed']!r}, max_steps={expected_steps!r}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="skip only final adapters whose metadata matches planned seed and max_steps",
    )
    parser.add_argument("--execute", action="store_true", help="required to start subprocesses")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite, all_jobs = _load_jobs(args.plan.resolve())
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain distinct non-empty device IDs")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be positive")
    limit = min(args.max_parallel, len(gpus))
    expected_steps = suite["common_inputs"]["max_steps"]
    if not isinstance(expected_steps, int) or expected_steps < 1:
        raise ValueError("plan common_inputs.max_steps must be positive")
    for job in all_jobs:
        _validate_existing_identity(job, expected_steps=expected_steps)
    skipped = [job for job in all_jobs if args.skip_completed and _is_complete(job, expected_steps=expected_steps)]
    pending = [job for job in all_jobs if job not in skipped]
    print(
        f"PLAN_ONLY={not args.execute} jobs={len(all_jobs)} queued={len(pending)} "
        f"skipped_complete={len(skipped)} gpus={','.join(gpus)} max_parallel={limit}"
    )
    if not args.execute:
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    status_path = (args.status or (args.log_dir.parent / "queue_status.json")).resolve()
    states: dict[str, dict[str, Any]] = {}
    for job in all_jobs:
        key = f"seed_{job['training_seed']}/{job['arm']}"
        states[key] = {
            "training_seed": job["training_seed"],
            "arm": job["arm"],
            "state": "skipped_complete" if job in skipped else "pending",
            "gpu": None,
            "pid": None,
            "exit_code": None,
            "artifact_contract_passed": True if job in skipped else None,
        }

    def write_status() -> None:
        atomic_json(
            status_path,
            {
                "schema_version": QUEUE_STATUS_SCHEMA,
                "plan": str(args.plan.resolve()),
                "jobs": states,
                "terminal": all(item["state"] in {"completed", "failed", "skipped_complete"} for item in states.values()),
            },
        )

    write_status()
    repo_root = Path(__file__).resolve().parents[2]
    free_gpus = list(gpus[:limit])
    running: list[tuple[subprocess.Popen[bytes], dict[str, Any], str, BinaryIO]] = []
    failures: list[str] = []
    while pending or running:
        while pending and free_gpus and len(running) < limit:
            job = pending.pop(0)
            gpu = free_gpus.pop(0)
            key = f"seed_{job['training_seed']}/{job['arm']}"
            job_log_dir = args.log_dir / f"seed_{job['training_seed']}"
            job_log_dir.mkdir(parents=True, exist_ok=True)
            log_handle = (job_log_dir / f"{job['arm']}.log").open("wb")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            project_paths = os.pathsep.join((str(repo_root / "src"), str(repo_root)))
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = project_paths + (os.pathsep + existing if existing else "")
            process = subprocess.Popen(
                job["command"], cwd=repo_root, env=env, stdout=log_handle, stderr=subprocess.STDOUT
            )
            states[key].update(state="running", gpu=gpu, pid=process.pid)
            write_status()
            running.append((process, job, gpu, log_handle))
            print(f"START job={key} physical_gpu={gpu} pid={process.pid}", flush=True)
        completed = [item for item in running if item[0].poll() is not None]
        if not completed:
            time.sleep(0.5)
            continue
        for process, job, gpu, log_handle in completed:
            running.remove((process, job, gpu, log_handle))
            log_handle.close()
            free_gpus.append(gpu)
            code = int(process.returncode or 0)
            key = f"seed_{job['training_seed']}/{job['arm']}"
            artifact_ok = code == 0 and _is_complete(job, expected_steps=expected_steps)
            states[key].update(
                state="completed" if artifact_ok else "failed",
                exit_code=code,
                artifact_contract_passed=artifact_ok,
            )
            write_status()
            print(
                f"DONE job={key} physical_gpu={gpu} exit={code} artifact_contract={artifact_ok}",
                flush=True,
            )
            if not artifact_ok:
                reason = str(code) if code != 0 else "missing_or_mismatched_final_artifact"
                failures.append(f"{key}:{reason}")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
