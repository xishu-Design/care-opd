"""CPU-only frozen evidence-layout probe. No labels are opened by prepare/run.

This is a post-hoc development diagnostic, not native multimodal training.
The system instructions, observation strings, output contract and scorer stay fixed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "_vendor"
if not VENDOR.is_dir():
    VENDOR = ROOT.parent / "care_opd_multimedia_boundary_v1"
sys.path.insert(0, str(VENDOR))
import boundary_common as c

VERSION = "care_opd.native_layout_probe.v1"
VARIANT = "evidence_packet_layout_v1"
SELECTION_SEED = 20260918
ARMS = c.ARMS


def runtime_dir():
    folder = ROOT / "runtime"
    if not folder.is_dir():
        folder = ROOT.parent / "care_opd/mechanism_experiments/visual_evidence_transfer"
    return folder


def package_hashes():
    names = ("probe_core.py", "probe_run.py", "probe_score.py")
    result = {name: c.hash_file(ROOT / name) for name in names}
    for name in ("boundary_common.py", "boundary_score.py"):
        result["_vendor/" + name] = c.hash_file(VENDOR / name)
    for name in ("common.py", "evaluate.py"):
        result["runtime/" + name] = c.hash_file(runtime_dir() / name)
    return result


def selected_block_ids(tasks, blocks=3):
    ids = {t["block_id"] for t in tasks}
    if isinstance(blocks, bool) or not 1 <= blocks <= len(ids):
        raise ValueError("blocks must be between 1 and available complete blocks")
    return sorted(ids, key=lambda x: hashlib.sha256(
        f"{SELECTION_SEED}:{x}".encode()).hexdigest())[:blocks]


def adapter_messages(task, observations):
    """Move evidence only. No new role, native task label, verified flag or schema.

    This deliberately does NOT claim full QZBC contract compatibility. Invoking
    student_analysis would change task semantics and introduce a second treatment.
    """
    old = c.answer_messages(task, observations)
    payload = c.strict_json(old[1]["content"])
    entries = []
    for side in ("a", "b"):
        source = payload["evidence"][side]
        entries.append({"source_id": source["source_id"],
                        "record_id": source["record_id"],
                        "summary": source["observation"], "origin": source["origin"]})
    adapted = {"questions": payload["questions"],
               "requested_records": payload["requested_records"],
               "record_context": payload["record_context"],
               "evidence_packet": {"retrieved_evidence": entries}}
    messages = [{"role": "system", "content": old[0]["content"]},
                {"role": "user", "content": c.canonical(adapted)}]
    assert roundtrip_payload(messages) == payload, "lossy evidence transformation"
    return messages


def roundtrip_payload(messages):
    payload = c.strict_json(messages[1]["content"])
    evidence = {}
    for side, item in zip(("a", "b"), payload["evidence_packet"]["retrieved_evidence"]):
        evidence[side] = {"source_id": item["source_id"], "record_id": item["record_id"],
                          "observation": item["summary"], "origin": item["origin"]}
    return {"questions": payload["questions"], "requested_records": payload["requested_records"],
            "record_context": payload["record_context"], "evidence": evidence}


def check_seal(row):
    if row.get("row_sha256") != c.digest({k: v for k, v in row.items() if k != "row_sha256"}):
        raise ValueError("row seal mismatch: " + str(row.get("task_id")))
    if row.get("status") != "completed" or not isinstance(row.get("raw_text"), str):
        raise ValueError("incomplete first-output row")
    if row.get("finish_reason") not in ("stop", "length"):
        raise ValueError("invalid finish reason")


def check_fields(row, expected):
    check_seal(row)
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"row binding mismatch: {row.get('task_id')}/{key}")


def validate_tasks(tasks, selection):
    c.index_rows(tasks)
    if len(tasks) != selection["task_count"]:
        raise ValueError("source task count mismatch")
    grouped = {}
    for task in tasks:
        if set(task) != {"task_id", "block_id", "condition", "dataset", "request", "sources"}:
            raise ValueError("unexpected task fields (possible label contamination)")
        req = task["request"]
        if set(req) != {"questions", "requested_records", "record_context"}:
            raise ValueError("unexpected request fields")
        if not isinstance(req["record_context"], str) or set(task["sources"]) != {"a", "b"}:
            raise ValueError("invalid request/sources")
        for key in ("questions", "requested_records"):
            if set(req[key]) != {"a", "b"} or not all(isinstance(x, str) and x for x in req[key].values()):
                raise ValueError("invalid question/record ids")
        for side, source in task["sources"].items():
            if set(source) != {"source_id", "record_id", "image_path", "image_sha256"}:
                raise ValueError("unexpected source fields")
            if source["source_id"] != "image:" + side:
                raise ValueError("source id changed")
            if selection["image_files"].get(source["image_path"]) != source["image_sha256"]:
                raise ValueError("image binding mismatch")
        block = grouped.setdefault(task["block_id"], set())
        if task["condition"] in block:
            raise ValueError("duplicate condition")
        block.add(task["condition"])
    if len(grouped) != selection["blocks"] or any(x != set(c.CONDITIONS) for x in grouped.values()):
        raise ValueError("incomplete six-condition blocks")


def inspect_source(source):
    """Read only cached runtime materials. Gold is NOT opened, even for hashing."""
    source = Path(source).resolve()
    files = ["config.json", "selection_manifest.json", "run_manifest.json", "tasks.jsonl",
             "observations.jsonl", *[f"responses/{a}/responses.jsonl" for a in ARMS]]
    before = {f: c.hash_file(source / f) for f in files}
    config = c.read_json(source / "config.json")
    selection = c.read_json(source / "selection_manifest.json")
    run = c.read_json(source / "run_manifest.json")
    if selection.get("evaluation_use") != "development_pilot":
        raise ValueError("only the frozen boundary development pilot is supported")
    if config.get("schema_version") != "care_opd.multimedia_boundary.v1.config":
        raise ValueError("unsupported original run config")
    if run["protocol_sha256"] != c.digest({k: v for k, v in run.items() if k != "protocol_sha256"}):
        raise ValueError("original protocol digest mismatch")
    if run["config_sha256"] != before["config.json"] or config["selection_manifest_sha256"] != before["selection_manifest.json"]:
        raise ValueError("original config/selection binding mismatch")
    if selection["files"]["tasks.jsonl"] != before["tasks.jsonl"]:
        raise ValueError("source tasks changed")
    for name in ("boundary_common.py", "boundary_score.py"):
        if config["code_sha256"][name] != c.hash_file(VENDOR / name):
            raise ValueError("original adapter/scorer version differs: " + name)
    if run["code_sha256"] != config["code_sha256"]:
        raise ValueError("source run/code mismatch")
    for name in ("common.py", "evaluate.py"):
        if config["runtime_files"][name] != c.hash_file(runtime_dir() / name):
            raise ValueError("original inference runtime differs: " + name)
    tasks = c.read_jsonl(source / "tasks.jsonl")
    validate_tasks(tasks, selection)
    observations = c.index_rows(c.read_jsonl(source / "observations.jsonl"))
    needed = {c.observation_spec(t, s)[0] for t in tasks for s in ("a", "b")}
    if set(observations) != needed:
        raise ValueError("observation cache incomplete or contains unknown rows")
    for ident, row in observations.items():
        check_fields(row, {"task_id": ident, "input_sha256": ident, "image_seen": True,
                          "model_fingerprint": run["model_assets"]["vlm"]["sha256"],
                          "protocol_sha256": run["protocol_sha256"]})
    baseline = {}
    for arm in ARMS:
        asset = run["model_assets"][arm]
        if asset["sha256"] != c.digest(asset["files"]):
            raise ValueError("model asset manifest mismatch")
        declared = config.get("expected_model_fingerprints", {}).get(arm)
        if declared and declared != asset["sha256"]:
            raise ValueError("original expected model mismatch")
        baseline[arm] = c.index_rows(c.read_jsonl(source / f"responses/{arm}/responses.jsonl"))
        if set(baseline[arm]) != {t["task_id"] for t in tasks}:
            raise ValueError("old arm incomplete: " + arm)
        for task in tasks:
            obs = c.task_observations(task, observations)
            check_fields(baseline[arm][task["task_id"]], {
                "task_id": task["task_id"], "arm": arm, "block_id": task["block_id"],
                "condition": task["condition"], "input_sha256": c.digest(c.answer_messages(task, obs)),
                "observation_sha256": c.digest(obs), "model_fingerprint": asset["sha256"],
                "protocol_sha256": run["protocol_sha256"]})
    if any(c.hash_file(source / f) != sha for f, sha in before.items()):
        raise ValueError("source changed while being inspected; stop the original writer")
    return config, selection, run, tasks, observations, baseline, before


def prepare(source, work, blocks=3):
    source, work = Path(source).resolve(), Path(work).resolve()
    if source == work or work.is_relative_to(source) or source.is_relative_to(work):
        raise ValueError("new work and original source must be disjoint directories")
    if (work / "experiment.json").exists():
        previous, *_ = load_work(work)
        if previous["source_root"] != str(source) or previous["blocks"] != blocks:
            raise ValueError("frozen selection differs; do not change an existing run")
        print("PREPARE_ALREADY_FROZEN", previous["task_count"], flush=True)
        return previous
    if (work / "responses").exists() or (work / "preflight.json").exists():
        raise ValueError("unbound inference outputs exist; use a fresh work directory")
    config, selection, run, tasks, observations, baseline, hashes = inspect_source(source)
    ids = selected_block_ids(tasks, blocks)
    chosen = sorted([t for t in tasks if t["block_id"] in ids],
                    key=lambda t: (ids.index(t["block_id"]), c.CONDITIONS.index(t["condition"])))
    needed = {c.observation_spec(t, s)[0] for t in chosen for s in ("a", "b")}
    obs = {k: observations[k] for k in sorted(needed)}
    prompts = []
    for task in chosen:
        messages = adapter_messages(task, c.task_observations(task, obs))
        prompts.append({"task_id": task["task_id"], "messages": messages, "input_sha256": c.digest(messages)})
    c.jsonl_once(work / "tasks.jsonl", chosen)
    c.jsonl_once(work / "observations.jsonl", obs.values())
    c.jsonl_once(work / "prompts.jsonl", prompts)
    for arm in ARMS:
        c.jsonl_once(work / f"baseline/{arm}.jsonl", [baseline[arm][t["task_id"]] for t in chosen])
    names = ["tasks.jsonl", "observations.jsonl", "prompts.jsonl", *[f"baseline/{a}.jsonl" for a in ARMS]]
    manifest = {"schema_version": VERSION, "variant": VARIANT,
                "evaluation_use": "post_hoc_development_diagnostic", "formal_claim_allowed": False,
                "source_root": str(source), "source_hashes": hashes,
                "gold_sha256": selection["files"]["gold.private.jsonl"],
                "gold_opened_during_prepare_or_inference": False,
                "source_protocol_sha256": run["protocol_sha256"],
                "selected_blocks": ids, "blocks": blocks, "task_count": len(chosen),
                "selection_seed": SELECTION_SEED, "selection_policy": "sha256_seed_block_id_no_response_or_gold_selection",
                "variant_policy": "same_system_rules_raw_observations_generation_and_scoring_evidence_layout_only",
                "source_software_environment": run["software_environment"],
                "models": {a: config["models"][a] for a in ARMS},
                "model_assets": {a: run["model_assets"][a] for a in ARMS},
                "generation": copy.deepcopy(config["generation"]),
                "files": {n: c.hash_file(work / n) for n in names}, "package_sha256": package_hashes(),
                "training": False, "new_visual_inference": False,
                "limitations": ["Previously inspected development data; not independent confirmation.",
                                "Only native evidence layout, not complete QZBC task/system contract.",
                                "No normalization or repair of VLM strings; no new local Protect training.",
                                "Old layout uses cached earlier generations; new layout is a later execution."]}
    manifest["protocol_sha256"] = c.digest(manifest)
    c.json_once(work / "experiment.json", manifest)
    load_work(work)
    print(f"PREPARE_PASS blocks={blocks} tasks={len(chosen)} new_text_generations={len(chosen)*4} new_visual=0", flush=True)
    return manifest


def load_work(work):
    work = Path(work).resolve()
    m = c.read_json(work / "experiment.json")
    if m["schema_version"] != VERSION or m["protocol_sha256"] != c.digest({k: v for k, v in m.items() if k != "protocol_sha256"}):
        raise ValueError("experiment manifest changed")
    if m["package_sha256"] != package_hashes():
        raise ValueError("probe code changed; do not mix script versions")
    for root, files in ((work, m["files"]), (Path(m["source_root"]), m["source_hashes"])):
        for name, sha in files.items():
            if c.hash_file(c.safe_path(root, name)) != sha:
                raise ValueError("frozen file changed: " + name)
    tasks = c.read_jsonl(work / "tasks.jsonl")
    obs = c.index_rows(c.read_jsonl(work / "observations.jsonl"))
    prompts = c.index_rows(c.read_jsonl(work / "prompts.jsonl"))
    if len(tasks) != m["task_count"] or set(prompts) != {t["task_id"] for t in tasks}:
        raise ValueError("prepared prompt count mismatch")
    for task in tasks:
        expected = adapter_messages(task, c.task_observations(task, obs))
        saved = prompts[task["task_id"]]
        if saved["messages"] != expected or saved["input_sha256"] != c.digest(expected):
            raise ValueError("frozen prompt differs from adapter")
    return m, tasks, obs, prompts


def new_binding(manifest, task, prompts, obs, arm):
    return {"task_id": task["task_id"], "status": "completed", "arm": arm, "variant": VARIANT,
            "block_id": task["block_id"], "condition": task["condition"],
            "protocol_sha256": manifest["protocol_sha256"],
            "input_sha256": prompts[task["task_id"]]["input_sha256"],
            "observation_sha256": c.digest(c.task_observations(task, obs)),
            "model_fingerprint": manifest["model_assets"][arm]["sha256"], "image_seen": False}


def load_new_rows(work, arm, require_complete=True):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    m, tasks, obs, prompts = load_work(work)
    expected = {t["task_id"]: new_binding(m, t, prompts, obs, arm) for t in tasks}
    rows = c.ledger(Path(work) / f"responses/{arm}.jsonl")
    baseline = c.index_rows(c.read_jsonl(Path(work) / f"baseline/{arm}.jsonl"))
    if set(rows) - set(expected):
        raise ValueError("unknown response IDs")
    for ident, row in rows.items():
        check_fields(row, {**expected[ident],
                          "paired_baseline_row_sha256": baseline[ident]["row_sha256"],
                          "paired_baseline_input_sha256": baseline[ident]["input_sha256"],
                          "paired_baseline_protocol_sha256": baseline[ident]["protocol_sha256"]})
    if require_complete and set(rows) != set(expected):
        raise ValueError(f"{arm} incomplete: {len(rows)}/{len(tasks)}; no partial score")
    return rows
