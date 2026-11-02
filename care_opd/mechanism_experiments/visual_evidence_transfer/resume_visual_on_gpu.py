#!/usr/bin/env python3
"""Resume a missing CARE tail on an explicitly audited replacement GPU.

The original manifest retains its original device history. Migration is an
append-only runtime event, not a new prompt/protocol or a content retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import uuid

import common
import evaluate

EXTRA_SOURCES = ("gpu45.py", "run_visual_transfer_gpu45.sh")
DEVICE_ID_FIELDS = {"cuda_visible_devices", "gpu_uuid"}


def validate_device(original: dict, actual: dict, gpu: str) -> None:
    if actual.get("cuda_visible_devices") != gpu:
        raise ValueError("replacement GPU differs from the requested physical index")
    required = {"python", "cuda_runtime", "torch", "transformers", "Pillow", "accelerate",
                "gpu_name", "gpu_total_memory_bytes", *DEVICE_ID_FIELDS}
    if not required <= set(original) or set(actual) != set(original):
        raise ValueError("missing or changed environment fields")
    for key in set(original) - DEVICE_ID_FIELDS:
        if original[key] != actual[key]:
            raise ValueError(f"environment differs beyond device identity: {key}")


def verify_existing(config: dict, manifest: dict, tasks: list[dict]) -> dict:
    """Validate all saved rows before writes/model load; gold is not read."""
    root = Path(config["run_root"])
    expected = {task["task_id"] for task in tasks}
    observations = common.unique_rows(root / "observations" / "observations.jsonl")
    if set(observations) != expected:
        raise ValueError("shared observations must be complete for CARE-only resume")
    paths = {"observations": root / "observations" / "observations.jsonl"}
    paths.update({arm: root / "arms" / arm / "responses.jsonl" for arm in common.ARMS})
    snapshot = {}
    for label, path in paths.items():
        rows = common.unique_rows(path)
        if set(rows) - expected:
            raise ValueError(f"unknown saved task in {label}")
        if label != "care_opd" and set(rows) != expected:
            raise ValueError(f"{label} incomplete; this entry resumes only CARE")
        if not path.is_file():
            raise ValueError(f"missing saved output: {path}")
        if label == "care_opd" and path.stat().st_size:
            with path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise ValueError("CARE JSONL has no terminal newline; preserve and inspect before appending")
        for task in tasks:
            row = rows.get(task["task_id"])
            if row is None:
                continue
            observing = label == "observations"
            direct = label == "vlm_direct"
            obs = None if observing or direct else observations[task["task_id"]]
            messages = (common.observation_messages(task) if observing else
                        common.answer_messages(task, obs, direct=direct))
            model = "vlm" if observing or direct else label
            checks = {
                "status": "completed", "arm": label,
                "protocol_sha256": manifest["protocol_sha256"],
                "model_fingerprint": manifest["model_fingerprints"][model],
                "input_sha256": common.input_digest(messages, task["image_sha256"] if observing or direct else None),
                "observation_sha256": common.digest(obs) if obs else None,
                **{key: task[key] for key in ("family_id", "cluster_id", "condition", "image_sha256")},
            }
            for key, value in checks.items():
                if row.get(key) != value:
                    raise ValueError(f"saved {label}/{task['task_id']} differs: {key}")
        snapshot[label] = {"path": str(path), "bytes": path.stat().st_size,
                           "sha256": common.file_sha256(path), "task_ids": sorted(rows)}
    return snapshot


def verify_retained(snapshot: dict) -> None:
    for label, before in snapshot.items():
        path = Path(before["path"])
        with path.open("rb") as handle:
            prefix = handle.read(before["bytes"])
        if hashlib.sha256(prefix).hexdigest() != before["sha256"]:
            raise ValueError(f"original output prefix changed: {label}")
        if label != "care_opd" and path.stat().st_size != before["bytes"]:
            raise ValueError(f"completed output changed: {label}")


def execute(config: dict, gpu: str, phase: str, *, full_hash: bool = False) -> None:
    if not gpu.isdigit() or phase not in ("preflight", "resume"):
        raise ValueError("require one numeric physical GPU and preflight/resume")
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    root = Path(config["run_root"])
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("resume requires an existing frozen run manifest")
    with common.run_lock(root):
        manifest_hash = common.file_sha256(manifest_path)
        manifest = common.read_json(manifest_path)
        if common.digest({k: v for k, v in manifest.items() if k != "protocol_sha256"}) != manifest.get("protocol_sha256"):
            raise ValueError("original manifest hash closure failed")
        fleet = manifest["environment"]
        if (fleet.get("schema_version") != "care_opd.visual_transfer.gpu45_environment.v1"
                or "care_opd" not in fleet.get("schedule", {}).get("parallel", {}).get("5", [])):
            raise ValueError("expected original GPU45 CARE lane on physical GPU5")
        tasks, selection = common.load_tasks(Path(config["tasks_dir"]))
        evaluate.verify_images(tasks)
        # Verify the ORIGINAL experiment. Actual device is checked and recorded
        # separately; it is never represented as the old physical GPU.
        evaluate.bind_run(config, tasks, selection, full_hash=full_hash,
                          execution_environment=fleet, extra_sources=EXTRA_SOURCES)
        evaluate.check_stack(config)
        actual = evaluate.environment()
        validate_device(fleet["devices"]["5"], actual, gpu)
        before = verify_existing(config, manifest, tasks)
        missing = sorted({t["task_id"] for t in tasks} - set(before["care_opd"]["task_ids"]))
        print(f"GPU_MIGRATION_PREFLIGHT=PASS gpu={gpu} care_saved={len(before['care_opd']['task_ids'])} care_missing={len(missing)}", flush=True)
        if phase == "preflight":
            return
        receipt = root / "runtime" / "device_migrations.jsonl"
        event_id = str(uuid.uuid4())

        def event(status: str, **extra):
            common.append_row(receipt, {
                "schema_version": "care_opd.visual_transfer.device_migration.v1",
                "event_id": event_id, "event": status, "pid": os.getpid(),
                "utc": datetime.now(timezone.utc).isoformat(), **extra})

        event("started", original_manifest_sha256=manifest_hash,
              original_protocol_sha256=manifest["protocol_sha256"], original_gpu="5", physical_gpu=gpu,
              original_environment=fleet["devices"]["5"], actual_environment=actual,
              resume_entry_sha256=common.file_sha256(Path(__file__)),
              frozen_code_sha256=manifest["code_sha256"], retained_outputs=before,
              missing_task_ids=missing, arm="care_opd",
              policy="device_identity_only_original_protocol_and_first_outputs_retained",
              timing_comparison="mixed_physical_devices_not_single_device_timing")
        try:
            evaluate.run_phase(config, manifest, tasks, "generate", "care_opd")
            verify_retained(before)
            if common.file_sha256(manifest_path) != manifest_hash:
                raise ValueError("original manifest was modified")
            evaluate.bind_run(config, tasks, selection, full_hash=full_hash,
                              execution_environment=fleet, extra_sources=EXTRA_SOURCES)
            validate_device(fleet["devices"]["5"], evaluate.environment(), gpu)
            after = verify_existing(config, manifest, tasks)
            if len(after["care_opd"]["task_ids"]) != len(tasks):
                raise ValueError("CARE still incomplete after resume")
            import score_results
            report = score_results.score_results(tasks_dir=Path(config["tasks_dir"]),
                                                 run_root=root, output=root / "summary")
            if report["status"] != "COMPLETE":
                raise ValueError("scoring is incomplete; retain outputs")
            event("completed", physical_gpu=gpu, outputs=after, summary=str(root / "summary"))
            print(f"GPU_MIGRATION_COMPLETE gpu={gpu} summary={root / 'summary'} audit={receipt}", flush=True)
        except BaseException as exc:
            event("failed", physical_gpu=gpu, error=f"{type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "resume"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    expected = args.source_dir.resolve()
    import score_results
    if any(Path(module.__file__).resolve().parent != expected for module in (common, evaluate, score_results)):
        parser.error("set PYTHONPATH to the ORIGINAL frozen visual source directory")
    execute(evaluate.configuration(args.config), args.gpu, args.phase, full_hash=args.full_hash)


if __name__ == "__main__":
    main()
