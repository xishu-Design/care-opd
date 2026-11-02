#!/usr/bin/env python3
"""Run plan arms concurrently, assigning one physical GPU per arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="required to start subprocesses")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("plan contains no arms")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or args.max_parallel < 1:
        raise ValueError("at least one GPU and positive --max-parallel are required")
    limit = min(args.max_parallel, len(gpus))
    for arm in arms:
        command = arm.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("every arm must contain a string command list")
        if "--nproc_per_node=1" not in command:
            raise ValueError("parallel single-GPU launcher requires a plan built with --nproc-per-node 1")
    print(f"PLAN_ONLY={not args.execute} arms={len(arms)} gpus={','.join(gpus)} max_parallel={limit}")
    if not args.execute:
        return 0
    args.log_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    pending = list(arms)
    running: list[tuple[subprocess.Popen[bytes], str, str, object]] = []
    free_gpus = list(gpus[:limit])
    failures: list[str] = []
    while pending or running:
        while pending and free_gpus and len(running) < limit:
            arm = pending.pop(0)
            gpu = free_gpus.pop(0)
            label = str(arm["name"])
            log_handle = (args.log_dir / f"{label}.log").open("wb")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            existing_pythonpath = env.get("PYTHONPATH", "")
            project_paths = os.pathsep.join((str(repo_root / "src"), str(repo_root)))
            env["PYTHONPATH"] = (
                project_paths + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
            )
            process = subprocess.Popen(
                arm["command"], cwd=repo_root, env=env, stdout=log_handle, stderr=subprocess.STDOUT
            )
            running.append((process, label, gpu, log_handle))
            print(f"START arm={label} physical_gpu={gpu} pid={process.pid}", flush=True)
        completed = [item for item in running if item[0].poll() is not None]
        if not completed:
            time.sleep(0.5)
            continue
        for process, label, gpu, log_handle in completed:
            running.remove((process, label, gpu, log_handle))
            code = int(process.returncode or 0)
            log_handle.close()
            free_gpus.append(gpu)
            print(f"DONE arm={label} physical_gpu={gpu} exit={code}", flush=True)
            if code != 0:
                failures.append(f"{label}:{code}")
    if failures:
        raise SystemExit("failed arms: " + ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
