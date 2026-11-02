#!/usr/bin/env python3
"""Frozen four-arm GPU runner, status and CPU scoring. No training or index build."""
from __future__ import annotations

import argparse
import csv
import importlib
import io
import os
from pathlib import Path
import sys
import time

import boundary_common as c

ROOT = Path(__file__).resolve().parent
GENERATION = {
    "seed": 20260917, "max_new_tokens": 1024, "observation_max_new_tokens": 512,
    "max_input_tokens": 6144, "max_pixels": 1024 * 28 * 28, "min_pixels": 256 * 28 * 28,
    "dtype": "bfloat16", "attention": "sdpa",
}
DEFAULT_MODELS = {
    "care_opd": "/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model",
    "clean_sft": "/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged",
    "gkd_full": "/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/gkd_full/final_model",
    "tropd_full": "/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/tropd_full/final_model",
}


def code_hashes():
    return {name: c.hash_file(ROOT / name) for name in (
        "boundary_common.py", "boundary_prepare.py", "boundary_fetch.py",
        "boundary_run.py", "boundary_score.py",
    )}


def load_prepared(work):
    work = Path(work)
    manifest = c.read_json(work / "selection_manifest.json")
    if manifest["schema_version"] != "care_opd.multimedia_boundary.selection.v1":
        raise ValueError("unsupported selection protocol")
    if manifest.get("evaluation_use") != "development_pilot":
        raise ValueError("this runner is for a development pilot, not a relabelled final test")
    for name, sha in manifest["files"].items():
        # Gold bytes are checked, but gold annotations are NOT parsed here.
        if c.hash_file(c.safe_path(work, name)) != sha:
            raise ValueError("prepared input changed: " + name)
    if not {"tasks.jsonl", "gold.private.jsonl"}.issubset(manifest["files"]):
        raise ValueError("missing required data fingerprints")
    for name, sha in manifest["image_files"].items():
        if c.hash_file(c.safe_path(work, name)) != sha:
            raise ValueError("image changed: " + name)
    tasks = c.read_jsonl(work / "tasks.jsonl")
    c.index_rows(tasks)
    if len(tasks) != manifest["task_count"] or not tasks:
        raise ValueError("task count mismatch/empty")
    blocks = {}
    for task in tasks:
        if any(k in task for k in ("answers", "gold", "reference_answer", "attack_target")):
            raise ValueError("private labels in runtime task")
        request = task["request"]
        if set(request) != {"questions", "requested_records", "record_context"}:
            raise ValueError("unknown request field")
        if set(task["sources"]) != {"a", "b"} or not isinstance(request["record_context"], str):
            raise ValueError("invalid task sources/context")
        for name in ("questions", "requested_records"):
            if set(request[name]) != {"a", "b"} or any(
                not isinstance(v, str) or not v for v in request[name].values()
            ):
                raise ValueError("invalid runtime " + name)
        for side in ("a", "b"):
            source = task["sources"][side]
            if set(source) != {"source_id", "record_id", "image_path", "image_sha256"}:
                raise ValueError("unknown runtime source field")
            if source["source_id"] != "image:" + side:
                raise ValueError("source identity mismatch")
            if manifest["image_files"].get(source["image_path"]) != source["image_sha256"]:
                raise ValueError("task image is not bound to selection manifest")
            expected_match = not (side == "a" and task["condition"] == "scope_mismatch")
            if (source["record_id"] == request["requested_records"][side]) != expected_match:
                raise ValueError("incorrect scope intervention")
        block = blocks.setdefault(task["block_id"], {})
        if task["condition"] in block or task["condition"] not in c.CONDITIONS:
            raise ValueError("duplicate/unknown block condition")
        block[task["condition"]] = task
    if len(blocks) != manifest["blocks"]:
        raise ValueError("independent block count mismatch")
    for block in blocks.values():
        if set(block) != set(c.CONDITIONS):
            raise ValueError("incomplete block")
        clean = block["clean"]
        for task in block.values():
            request, clean_request = task["request"], clean["request"]
            if (request["questions"] != clean_request["questions"]
                    or request["record_context"] != clean_request["record_context"]
                    or request["requested_records"]["b"] != clean_request["requested_records"]["b"]
                    or task["sources"]["b"] != clean["sources"]["b"]):
                raise ValueError("question/context/B changed inside a paired block")
            if (task["condition"] != "valid_update"
                    and request["requested_records"]["a"] != clean_request["requested_records"]["a"]):
                raise ValueError("unexpected requested record change")
        if clean["sources"]["a"]["image_sha256"] != block["scope_mismatch"]["sources"]["a"]["image_sha256"]:
            raise ValueError("scope-only intervention must not alter pixels")
        if clean["sources"]["a"]["image_sha256"] == block["valid_update"]["sources"]["a"]["image_sha256"]:
            raise ValueError("valid update must change image")
    return tasks, manifest


def find_runtime(explicit=None):
    candidates = [Path(explicit)] if explicit else [
        ROOT / "runtime",
        ROOT.parent / "care_opd/mechanism_experiments/visual_evidence_transfer",
    ]
    for candidate in candidates:
        if all((candidate / n).is_file() for n in ("evaluate.py", "common.py")):
            return candidate.resolve()
    raise FileNotFoundError("bundled HF runtime missing; re-extract the complete archive")


def find_vlm(value=None):
    if value and (Path(value).expanduser() / "config.json").is_file():
        return str(Path(value).expanduser().resolve())
    if value:
        print("VLM_PATH_NOT_FOUND", value, flush=True)
    cache = Path("/opt/care-data/model_cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct")
    ref = cache / "refs/main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        if len(revision) == 40 and all(ch in "0123456789abcdef" for ch in revision):
            candidate = cache / "snapshots" / revision
            if (candidate / "config.json").is_file():
                return str(candidate.resolve())
    candidates = [p for p in (cache / "snapshots").glob("*") if (p / "config.json").is_file()]
    if len(candidates) == 1:
        return str(candidates[0].resolve())
    raise FileNotFoundError("cannot unambiguously resolve Qwen2.5-VL; use init --vlm /actual/snapshot/path")


def init(work, source_config=None, runtime_dir=None, vlm=None):
    load_prepared(work)
    source = c.read_json(source_config) if source_config else {}
    models = dict(source.get("models", DEFAULT_MODELS))
    if not set(c.ARMS).issubset(models):
        raise ValueError("source config must contain all four text checkpoints")
    if vlm and not (Path(vlm).expanduser() / "config.json").is_file():
        raise FileNotFoundError("explicit --vlm path is not a checkpoint: " + vlm)
    models["vlm"] = find_vlm(vlm or models.get("vlm"))
    models = {name: str(Path(models[name]).expanduser().resolve()) for name in ("vlm", *c.ARMS)}
    for name, model in models.items():
        if not (Path(model) / "config.json").is_file():
            raise FileNotFoundError(f"missing model config: {name}: {model}")
    runtime = find_runtime(runtime_dir)
    config = {
        "schema_version": c.VERSION + ".config", "models": models, "generation": GENERATION,
        "runtime_dir": str(runtime), "runtime_files": {n: c.hash_file(runtime / n) for n in ("evaluate.py", "common.py")},
        "code_sha256": code_hashes(), "training": False,
        "selection_manifest_sha256": c.hash_file(work / "selection_manifest.json"),
        "selection_policy": "first_output_no_content_retry",
        "expected_model_fingerprints": source.get("expected_model_fingerprints", {}),
    }
    c.json_once(work / "config.json", config)
    print("CONFIG_READY", work / "config.json", flush=True)
    for name, model in models.items():
        print("MODEL", name, model, flush=True)
    return config


def load_config(work):
    config = c.read_json(work / "config.json")
    if config["schema_version"] != c.VERSION + ".config" or config["code_sha256"] != code_hashes():
        raise ValueError("experiment code changed after init; preserve run and use a new work root")
    if config["selection_manifest_sha256"] != c.hash_file(work / "selection_manifest.json"):
        raise ValueError("selection manifest changed")
    if config["generation"] != GENERATION:
        raise ValueError("generation settings changed; this package has one frozen shared budget")
    for name, sha in config["runtime_files"].items():
        if c.hash_file(Path(config["runtime_dir"]) / name) != sha:
            raise ValueError("HF runtime changed: " + name)
    return config


def import_runtime(config):
    folder = str(Path(config["runtime_dir"]).resolve())
    # Vendored evaluate.py imports its sibling common.py. No module named
    # common is used by this adapter; fail closed on an unexpected collision.
    for name in ("common", "evaluate"):
        loaded = sys.modules.get(name)
        if loaded is not None and str(Path(loaded.__file__).resolve().parent) != folder:
            raise RuntimeError("unexpected Python module collision: " + name)
    sys.path.insert(0, folder)
    return importlib.import_module("evaluate")


def load_manifest(work):
    load_config(work)
    manifest = c.read_json(work / "run_manifest.json")
    if c.digest({k: v for k, v in manifest.items() if k != "protocol_sha256"}) != manifest["protocol_sha256"]:
        raise ValueError("runtime manifest digest mismatch")
    if manifest["config_sha256"] != c.hash_file(work / "config.json"):
        raise ValueError("runtime/config mismatch")
    return manifest


def bind_runtime(work, config, full_hash=False, runtime=None):
    runtime = runtime or import_runtime(config)
    runtime.check_stack(config)
    environment = runtime.environment()
    software = {k: v for k, v in environment.items() if k not in {
        "cuda_visible_devices", "gpu_uuid", "gpu_total_memory_bytes"
    }}
    path = work / "run_manifest.json"
    previous = load_manifest(work) if path.exists() else None
    base = {"schema_version": c.VERSION + ".run", "config_sha256": c.hash_file(work / "config.json"),
            "software_environment": software, "code_sha256": code_hashes(),
            "first_output_policy": "immutable_no_content_retry", "training": False}
    if previous:
        for key, value in base.items():
            if previous.get(key) != value:
                raise ValueError("runtime binding changed: " + key)
    elif (work / "responses").exists() or (work / "observations.jsonl").exists():
        raise ValueError("unbound inference outputs exist; choose a fresh root")
    assets = {}
    for name, model in config["models"].items():
        old = previous["model_assets"][name] if previous else None
        asset = runtime.fingerprint_model(Path(model), previous=old, full=full_hash)
        if old and asset["sha256"] != old["sha256"]:
            raise ValueError("checkpoint changed: " + name)
        expected = config.get("expected_model_fingerprints", {}).get(name)
        if expected and expected != asset["sha256"]:
            raise ValueError("checkpoint differs from source experiment: " + name)
        assets[name] = asset
    if previous is None:
        base["model_assets"] = assets
        base["initial_environment"] = environment
        base["protocol_sha256"] = c.digest(base)
        c.json_once(path, base)
        previous = base
    c.append_row(work / "device_history.jsonl", {"pid": os.getpid(), "timestamp_unix": time.time(),
                 "environment": environment, "protocol_sha256": previous["protocol_sha256"]})
    print("PREFLIGHT_PASS", previous["protocol_sha256"], flush=True)
    return runtime, previous


def binding(ident, model_hash, input_hash, manifest):
    return {"task_id": ident, "status": "completed", "protocol_sha256": manifest["protocol_sha256"],
            "model_fingerprint": model_hash, "input_sha256": input_hash}


def seal_row(row):
    return {**row, "row_sha256": c.digest(row)}


def checked_ledger(path, expected):
    rows = c.ledger(path)
    if set(rows) - set(expected):
        raise ValueError("unknown ledger IDs: " + str(path))
    for ident, row in rows.items():
        for key, value in expected[ident].items():
            if row.get(key) != value:
                raise ValueError(f"ledger binding mismatch: {ident}/{key}")
        if not isinstance(row.get("raw_text"), str) or row.get("finish_reason") not in {"stop", "length", "rule"}:
            raise ValueError("missing raw text/finish reason: " + ident)
        if row.get("row_sha256") != c.digest({k: v for k, v in row.items() if k != "row_sha256"}):
            raise ValueError("saved first output changed: " + ident)
    return rows


def observation_inventory(tasks, manifest):
    specs = {}
    for task in tasks:
        for side in ("a", "b"):
            oid, messages, source = c.observation_spec(task, side)
            specs.setdefault(oid, (messages, source))
    expected = {oid: {**binding(oid, manifest["model_assets"]["vlm"]["sha256"], oid, manifest),
                      "image_seen": True} for oid in specs}
    return specs, expected


def response_bindings(tasks, arm, observations, manifest):
    model_hash = (manifest["model_assets"][arm]["sha256"] if arm in c.ARMS
                  else c.digest({"baseline": arm, "code": manifest["code_sha256"]}))
    expected, messages, per_task_obs = {}, {}, {}
    for task in tasks:
        if not all(c.observation_spec(task, side)[0] in observations for side in ("a", "b")):
            continue
        obs = c.task_observations(task, observations)
        msgs = c.answer_messages(task, obs)
        ident = task["task_id"]
        expected[ident] = {**binding(ident, model_hash, c.digest(msgs), manifest),
                           "observation_sha256": c.digest(obs), "arm": arm,
                           "block_id": task["block_id"], "condition": task["condition"]}
        messages[ident], per_task_obs[ident] = msgs, obs
    return expected, messages, per_task_obs


def run(work, limit_blocks=0, full_hash=False, runtime=None):
    tasks, _ = load_prepared(work)
    config = load_config(work)
    runtime, manifest = bind_runtime(work, config, full_hash, runtime=runtime)
    specs, obs_bindings = observation_inventory(tasks, manifest)
    blocks = list(dict.fromkeys(t["block_id"] for t in tasks))
    if limit_blocks:
        blocks = blocks[:limit_blocks]
    selected = [t for t in tasks if t["block_id"] in blocks]
    needed = {c.observation_spec(t, side)[0] for t in selected for side in ("a", "b")}
    obs_path = work / "observations.jsonl"
    observations = checked_ledger(obs_path, obs_bindings)
    generator = None
    try:
        for oid, (messages, source) in specs.items():
            if oid not in needed or oid in observations:
                continue
            try:
                if generator is None:
                    generator = runtime.HFGenerator(config["models"]["vlm"], config["generation"], visual=True)
                result = generator.generate(messages, str(c.safe_path(work, source["image_path"])),
                                            config["generation"]["observation_max_new_tokens"])
                if result.get("image_seen") is not True:
                    raise RuntimeError("VLM did not receive real pixels")
                if not isinstance(result.get("raw_text"), str) or result.get("finish_reason") not in {"stop", "length"}:
                    raise RuntimeError("runtime returned no complete generation record")
                row = seal_row({**result, **obs_bindings[oid]})
                c.append_row(obs_path, row)
                observations[oid] = row
                print(f"OBS_PROGRESS saved={len(observations)}/{len(specs)}", flush=True)
            except Exception as exc:
                c.append_row(work / "infrastructure_errors.jsonl", {"stage": "observe", "id": oid,
                             "error": f"{type(exc).__name__}: {exc}"})
                raise
    finally:
        if generator is not None:
            generator.close()
    for arm in (*c.ARMS, *c.BASELINES):
        expected, messages, per_task_obs = response_bindings(tasks, arm, observations, manifest)
        path = work / "responses" / arm / "responses.jsonl"
        existing = checked_ledger(path, expected)
        generator = None
        try:
            for task in selected:
                ident = task["task_id"]
                if ident in existing:
                    continue
                try:
                    if arm in c.ARMS:
                        if generator is None:
                            generator = runtime.HFGenerator(config["models"][arm], config["generation"], visual=False)
                        result = generator.generate(messages[ident], None, config["generation"]["max_new_tokens"])
                        if not isinstance(result.get("raw_text"), str) or result.get("finish_reason") not in {"stop", "length"}:
                            raise RuntimeError("runtime returned no complete generation record")
                    else:
                        result = {"raw_text": c.rule_response(task, per_task_obs[ident], arm == "always_abstain"),
                                  "finish_reason": "rule"}
                    row = seal_row({**result, **expected[ident],
                                    "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
                    # Persist BEFORE any answer-quality scoring. Empty, invalid and
                    # truncated outputs remain completed first outputs on resume.
                    c.append_row(path, row)
                    existing[ident] = row
                    print(f"GENERATE arm={arm} saved={len(existing)}/{len(tasks)} finish={row['finish_reason']}", flush=True)
                except Exception as exc:
                    c.append_row(work / "infrastructure_errors.jsonl", {"stage": "generate", "arm": arm,
                                 "task_id": ident, "error": f"{type(exc).__name__}: {exc}"})
                    raise
        finally:
            if generator is not None:
                generator.close()
    print("RUN_PHASE_COMPLETE", "smoke" if limit_blocks else "all", flush=True)


def csv_once(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: c.canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    c.write_once(path, stream.getvalue().encode("utf-8-sig"))


def flatten_metrics(value, prefix=()):
    if isinstance(value, dict):
        if {"count", "denominator", "rate"}.issubset(value):
            return [{"metric": "/".join(prefix), **value}]
        return [row for k, child in value.items() if k != "rows"
                for row in flatten_metrics(child, (*prefix, str(k)))]
    return []


def score(work):
    from boundary_score import score_rows
    tasks, selection = load_prepared(work)
    manifest = load_manifest(work)
    gold = c.index_rows(c.read_jsonl(work / "gold.private.jsonl"))
    specs, obs_bindings = observation_inventory(tasks, manifest)
    observations = checked_ledger(work / "observations.jsonl", obs_bindings)
    if set(observations) != set(specs):
        raise ValueError("shared observations incomplete; no final score produced")
    responses, hashes = {}, {}
    for arm in (*c.ARMS, *c.BASELINES):
        expected, _, _ = response_bindings(tasks, arm, observations, manifest)
        path = work / "responses" / arm / "responses.jsonl"
        rows = checked_ledger(path, expected)
        if set(rows) != {t["task_id"] for t in tasks}:
            raise ValueError(f"{arm}: {len(rows)}/{len(tasks)} responses; no final score produced")
        responses[arm] = list(rows.values())
        hashes[arm] = c.hash_file(path)
    per_task_obs = {t["task_id"]: c.task_observations(t, observations) for t in tasks}
    report = score_rows(tasks, gold, responses, per_task_obs)
    report["selection_manifest_sha256"] = c.hash_file(work / "selection_manifest.json")
    report["evaluation_use"] = "development_pilot"
    report["formal_claim_allowed"] = False
    report["selection_limitations"] = selection.get("limitations", [])
    fingerprint = {"response_files": hashes, "observations": c.hash_file(work / "observations.jsonl"),
                   "gold": c.hash_file(work / "gold.private.jsonl"), "code": code_hashes(),
                   "protocol": manifest["protocol_sha256"]}
    out = work / "summary" / c.digest(fingerprint)[:16]
    c.json_once(out / "metrics.json", report)
    c.json_once(out / "scoring_manifest.json", fingerprint)
    csv_once(out / "metrics.csv", flatten_metrics(report.get("arms", report)))
    csv_once(out / "frontend_metrics.csv", flatten_metrics(report.get("frontend", {})))
    for arm, views in report["arms"].items():
        for view, values in views.items():
            csv_once(out / f"{arm}_{view}_rows.private.csv", values["rows"])
    differences = report.get("paired_differences", [])
    if isinstance(differences, list):
        csv_once(out / "paired_effects.csv", differences)
    else:
        c.json_once(out / "paired_effects.json", differences)
    lines = ["# Multimedia boundary development pilot", "",
             f"Prepared blocks (bootstrap units): {selection['blocks']}; tasks per arm: {len(tasks)}.",
             "First outputs are frozen. No training and no expert-rating claim.", "",
             "metrics.csv uses explicit numerator/denominator/rate. Zero denominators are null.",
             "Strict and fence-normalized views are separate. Format failures are not dropped.",
             "Use joint useful-and-boundary-correct performance, not refusal rate alone.", "",
             "## Main summary (semantic view; counts over tasks, not independent blocks)", "",
             "| Arm | Joint useful + boundary correct | Normal accuracy | Normal coverage | Accuracy among answered |",
             "| --- | --- | --- | --- | --- |"]
    for arm, views in report["arms"].items():
        fields = []
        for name in ("joint_visual_safe_useful", "normal_accuracy", "normal_coverage", "normal_selective_accuracy"):
            value = views["semantic"]["overall"][name]
            display = "NA" if value["rate"] is None else f"{value['rate']:.2%}"
            fields.append(f"{display} ({value['count']}/{value['denominator']})")
        lines.append("| " + " | ".join([arm, *fields]) + " |")
    lines.extend(["",
             "## Limits", "",
             "- Source outage removes one whole chart, not a medically meaningful region.",
             "- Scope mismatch uses constructed host metadata; compare the rule baseline.",
             "- This pilot does not establish native multimodal training or clinical utility."])
    c.write_once(out / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    print("SUMMARY_READY", out, flush=True)
    return report


def status(work):
    tasks, selection = load_prepared(work)
    count = len({c.observation_spec(t, s)[0] for t in tasks for s in ("a", "b")})
    obs = c.ledger(work / "observations.jsonl")
    print(f"blocks={selection['blocks']} tasks={len(tasks)} observations={len(obs)}/{count}")
    for arm in (*c.ARMS, *c.BASELINES):
        rows = c.ledger(work / "responses" / arm / "responses.jsonl")
        print(f"{arm}: {len(rows)}/{len(tasks)}")
    path = work / ".boundary.lock"
    if path.exists():
        print("LOCK", path.read_text(encoding="utf-8"))
    print("Counts only, not scientific success. Do not score while a writer is active.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "preflight", "run", "status", "score"))
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--vlm")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--limit-blocks", type=int, default=0)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or args.limit_blocks < 0:
        parser.error("GPU and limit-blocks must be nonnegative")
    os.umask(0o077)
    work = args.work.expanduser().resolve()
    if args.command == "status":
        status(work)
        return
    with c.run_lock(work):
        if args.command == "init":
            init(work, args.source_config, args.runtime_dir, args.vlm)
        elif args.command == "score":
            score(work)
        else:
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1",
                              TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            if args.command == "preflight":
                load_prepared(work)
                bind_runtime(work, load_config(work), args.full_hash)
            else:
                run(work, args.limit_blocks, args.full_hash)


if __name__ == "__main__":
    main()
