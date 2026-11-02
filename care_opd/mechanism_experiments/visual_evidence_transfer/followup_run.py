"""Isolated validation follow-up; never modify or select V4 model outputs."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

import common
import evaluate
import followup_attributes
import followup_audit
import followup_locality

SCHEMA = "care_opd.visual_transfer.followup.v1"
CODE_NAMES = ("common.py", "evaluate.py", "prepare_slake.py", "score_results.py",
              "followup_audit.py", "followup_attributes.py", "followup_locality.py", "followup_run.py",
              "run_followup_v1.sh")


def code_hashes() -> dict:
    return {name: common.file_sha256(Path(__file__).parent / name) for name in CODE_NAMES}


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"frozen output differs: {path}; preserve it and choose a fresh work root")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def json_once(path: Path, value) -> None:
    write_once(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def jsonl_once(path: Path, rows) -> None:
    write_once(path, ("".join(common.canonical(row) + "\n" for row in rows)).encode("utf-8"))


def csv_once(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: common.canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    write_once(path, stream.getvalue().encode("utf-8-sig"))


def metric_table(value: dict, prefix: tuple[str, ...] = ()) -> list[dict]:
    """Flatten explicit count/denominator/rate records for auditable CSV export."""
    if {"count", "denominator", "rate"}.issubset(value):
        return [{"metric": "/".join(prefix), **value}]
    result = []
    for key, child in value.items():
        if isinstance(child, dict):
            result.extend(metric_table(child, (*prefix, str(key))))
    return result


def isolated_output(work: Path, source: dict) -> None:
    work = work.resolve()
    for key in ("tasks_dir", "run_root"):
        old = Path(source["config"][key]).resolve()
        if work == old or work in old.parents or old in work.parents:
            raise ValueError(f"new follow-up output must be separate from source {key}: {old}")


def export_audit(source: dict, work: Path) -> dict:
    isolated_output(work, source)
    audit = followup_audit.build_audit(source)
    root = work / "source_audit"
    json_once(root / "audit.private.json", audit)
    jsonl_once(root / "raw_rows.private.jsonl", audit["row_records"])
    csv_once(root / "usefulness_and_errors.csv", audit["per_arm_condition"])
    csv_once(root / "availability_rules.csv", audit["rule_baselines"])
    updates = audit.get("valid_image_update", [])
    csv_once(root / "valid_image_update.csv", updates)
    lines = ["# V4 首次输出审计", "", "本报告只重算既有输出，不重新生成、不修复或筛选答案。",
             "", "## 正常证据下的有效回答", "",
             "准确率以所有题为分母；覆盖率只计 YES/NO；选择性准确率只以 YES/NO 为分母，0 分母记为 null。",
             "UNKNOWN 与 ABSTAIN 分开，格式无效及截断不从分母删除。详见 usefulness_and_errors.csv。",
             "", "## 有效换图", "",
             "仅使用问题不变、图像与真实相反标注同步更新的配对；不是身份错配检测，也不以拒答为正确响应。",
             "详见 valid_image_update.csv，旧图和新图都正确是核心端点，单纯标签翻转不是成功。",
             "", "## 边界", "", *["- " + str(x) for x in audit["limitations"]]]
    write_once(root / "README.md", ("\n".join(lines) + "\n").encode("utf-8"))
    print(f"SOURCE_AUDIT_COMPLETE output={root}", flush=True)
    return audit


def prepare(config_path: Path, work: Path, seed: int = 20260917, source: dict | None = None) -> dict:
    source = source if source is not None else followup_audit.load_source(config_path)
    isolated_output(work, source)
    export_audit(source, work)
    source = followup_attributes.attach_attributes(source)
    prepared = followup_locality.prepare_locality(source, seed=seed)
    tasks, gold = prepared["tasks"], prepared["gold_by_id"]
    if not tasks:
        raise ValueError("no eligible disjoint A/B/C blocks; do not invent labels or relax isolation after results")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    jsonl_once(work / "tasks.jsonl", tasks)
    jsonl_once(work / "gold.private.jsonl", [gold[task["task_id"]] for task in tasks])
    json_once(work / "selection_manifest.json", prepared["selection"])
    json_once(work / "attribute_provenance.private.json", source["text_attribute_provenance"])
    for arm in common.TEXT_ARMS:
        rows = followup_locality.independent_baseline(source, tasks, arm=arm)
        jsonl_once(work / "baselines" / f"independent_{arm}.jsonl", rows)
    names = ["tasks.jsonl", "gold.private.jsonl", "selection_manifest.json", "attribute_provenance.private.json"]
    names += [f"baselines/independent_{arm}.jsonl" for arm in common.TEXT_ARMS]
    config = {
        "schema_version": SCHEMA, "evaluation_use": "validation_development_diagnostic",
        "source_config": str(config_path.resolve()), "source_binding": source["binding"],
        "models": {arm: source["config"]["models"][arm] for arm in common.TEXT_ARMS},
        "expected_model_fingerprints": {arm: source["manifest"]["model_fingerprints"][arm] for arm in common.TEXT_ARMS},
        "generation": {**source["config"]["generation"], "max_new_tokens": 1024},
        "generation_budget_reason": "five output sections instead of two; same 1024 limit for all four arms, fixed before generation",
        "files": {name: common.file_sha256(work / name) for name in names},
        "code_sha256": code_hashes(), "task_count": len(tasks), "seed": seed,
        "training": False, "selection_policy": "first_output_no_content_retry",
        "scope": "cached_frozen_VLM_observations_plus_independent_supplied_text_record_not_native_multimodal_training",
        "prior_test_outputs_inspected": True,
        "formal_claim_allowed": False,
    }
    json_once(work / "experiment.json", config)
    print(f"FOLLOWUP_PREPARED tasks={len(tasks)} arms={len(common.TEXT_ARMS)} work={work}", flush=True)
    return config


def load_work(work: Path) -> tuple[dict, list[dict]]:
    config = common.read_json(work / "experiment.json")
    if config.get("schema_version") != SCHEMA or config.get("formal_claim_allowed") is not False:
        raise ValueError("unsupported follow-up configuration")
    if config["code_sha256"] != code_hashes():
        raise ValueError("follow-up source changed after prepare; never mix protocol versions")
    for name, expected in config["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe prepared filename")
        if common.file_sha256(work / name) != expected:
            raise ValueError(f"prepared input changed: {name}")
    # Gold bytes are hashed above but are NOT parsed by inference.
    tasks = list(common.unique_rows(work / "tasks.jsonl").values())
    if len(tasks) != config["task_count"]:
        raise ValueError("prepared task count differs")
    for task in tasks:
        followup_locality.locality_messages(task)
    return config, tasks


def bind_runtime(work: Path, config: dict, full_hash: bool = False) -> dict:
    path = work / "run_manifest.json"
    previous = common.read_json(path) if path.exists() else None
    if previous is None and (work / "arms").exists():
        raise ValueError("unbound inference outputs exist; choose a fresh work root")
    evaluate.check_stack(config)
    base = {"schema_version": SCHEMA + ".run", "experiment_sha256": common.file_sha256(work / "experiment.json"),
            "environment": evaluate.environment(), "code_sha256": code_hashes(),
            "arms": list(common.TEXT_ARMS), "task_count": config["task_count"],
            "selection_policy": "first_output_no_content_retry", "training": False}
    if previous:
        for key, value in base.items():
            if previous.get(key) != value:
                raise ValueError(f"runtime {key} changed; keep completed first outputs and investigate")
    assets = {}
    for arm, model in config["models"].items():
        old = previous["model_assets"][arm] if previous else None
        assets[arm] = evaluate.fingerprint_model(Path(model), previous=old, full=full_hash)
        if assets[arm]["sha256"] != config["expected_model_fingerprints"][arm]:
            raise ValueError(f"model differs from frozen V4 checkpoint: {arm}")
    if previous:
        return previous
    base["model_assets"] = assets
    base["model_fingerprints"] = {arm: asset["sha256"] for arm, asset in assets.items()}
    base["protocol_sha256"] = common.digest(base)
    json_once(path, base)
    return base


def row_binding(task: dict, arm: str, manifest: dict) -> dict:
    return {"task_id": task["task_id"], "arm": arm, "status": "completed",
            "protocol_sha256": manifest["protocol_sha256"],
            "model_fingerprint": manifest["model_fingerprints"][arm],
            "input_sha256": common.digest(followup_locality.locality_messages(task)),
            "task_sha256": common.digest(task)}


def checked_rows(work: Path, tasks: list[dict], arm: str, manifest: dict) -> dict:
    ledger = work / "arms" / arm / "responses.jsonl"
    if ledger.exists() and ledger.stat().st_size:
        with ledger.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                raise ValueError(f"response ledger has an unterminated final row: {ledger}; preserve bytes and investigate")
    rows = common.unique_rows(ledger)
    expected = {task["task_id"]: task for task in tasks}
    if set(rows) - set(expected):
        raise ValueError(f"unknown response tasks in {arm}")
    for ident, row in rows.items():
        for key, value in row_binding(expected[ident], arm, manifest).items():
            if row.get(key) != value:
                raise ValueError(f"frozen row mismatch {arm}/{ident}/{key}")
        if not isinstance(row.get("raw_text"), str) or not isinstance(row.get("finish_reason"), str):
            raise ValueError("completed response must preserve raw text and finish reason")
    return rows


def generate_arm(work: Path, config: dict, tasks: list[dict], manifest: dict, arm: str,
                 generator_factory=None) -> None:
    existing = checked_rows(work, tasks, arm, manifest)
    factory = generator_factory or evaluate.HFGenerator
    generator = None
    output = work / "arms" / arm / "responses.jsonl"
    try:
        for index, task in enumerate(tasks, 1):
            if task["task_id"] in existing:
                continue
            binding = row_binding(task, arm, manifest)
            try:
                if generator is None:
                    generator = factory(config["models"][arm], config["generation"], visual=False)
                result = generator.generate(followup_locality.locality_messages(task), None,
                                            config["generation"]["max_new_tokens"])
                row = {**result, **binding, "block_id": task["block_id"], "condition": task["condition"]}
                # Persist the first output before any quality/answer scoring.
                common.append_row(output, row)
                print(f"FOLLOWUP_PROGRESS arm={arm} row={index}/{len(tasks)} finish={row['finish_reason']}", flush=True)
            except Exception as exc:
                common.append_row(output.parent / "infrastructure_errors.jsonl",
                                  {**binding, "status": "infra_error", "error": f"{type(exc).__name__}: {exc}"})
                raise RuntimeError(f"technical failure at {arm}/{task['task_id']}; completed rows stay frozen") from exc
    finally:
        if generator is not None:
            generator.close()


def score(work: Path) -> dict:
    config, tasks = load_work(work)
    manifest = common.read_json(work / "run_manifest.json")
    if manifest["experiment_sha256"] != common.file_sha256(work / "experiment.json"):
        raise ValueError("runtime was bound to a different experiment")
    stored = manifest["protocol_sha256"]
    if common.digest({k: v for k, v in manifest.items() if k != "protocol_sha256"}) != stored:
        raise ValueError("runtime manifest fingerprint mismatch")
    responses = {arm: list(checked_rows(work, tasks, arm, manifest).values()) for arm in common.TEXT_ARMS}
    if any(len(rows) != len(tasks) for rows in responses.values()):
        raise ValueError("follow-up incomplete; run status/resume before final scoring")
    gold = common.unique_rows(work / "gold.private.jsonl")
    report = followup_locality.score_locality(tasks, gold, responses)
    baseline_rows = {"independent_" + arm: common.read_jsonl(work / "baselines" / f"independent_{arm}.jsonl")
                     for arm in common.TEXT_ARMS}
    baseline = followup_locality.score_locality(tasks, gold, baseline_rows)
    fingerprint = common.digest({arm: common.file_sha256(work / "arms" / arm / "responses.jsonl") for arm in common.TEXT_ARMS})
    output = work / "summary" / fingerprint[:16]
    json_once(output / "locality_metrics.json", report)
    json_once(output / "independent_visual_only_baselines.json", baseline)
    csv_once(output / "locality_metrics.csv", metric_table(report))
    csv_once(output / "independent_visual_only_baselines.csv", metric_table(baseline))
    for arm, values in report.get("arms", {}).items():
        csv_once(output / (arm + "_rows.private.csv"), values.get("row_scores", []))
        protection = []
        for condition, data in values.get("protected_b", {}).items():
            protection.append({"arm": arm, "condition": condition, "field": "visual_b", **data})
        for condition, data in values.get("protected_text_attributes", {}).items():
            for field, measures in data.items():
                if isinstance(measures, dict):
                    protection.append({"arm": arm, "condition": condition, "field": field, **measures})
        csv_once(output / (arm + "_protection.csv"), protection)
    json_once(output / "scoring_manifest.json", {"schema_version": SCHEMA + ".score", "responses_sha256": fingerprint,
               "experiment_sha256": manifest["experiment_sha256"], "protocol_sha256": stored,
               "GPU_experiment_completed": True, "evaluation_use": config["evaluation_use"],
               "baselines_have_text_attributes": False,
               "baseline_cost_policy": "reuses separate old per-question inference; not equal compute or a zero-cost model"})
    # Export all list-of-record tables without depending on optional metric names.
    for name, values in report.items():
        if isinstance(values, list) and values and all(isinstance(x, dict) for x in values):
            csv_once(output / (name + ".csv"), values)
    write_once(output / "README.md", ("# 多字段局部更新与保护开发集试验\n\n"
        "四个后端共享相同观察、文本属性、提示和1024-token预算，保留首个输出。\n\n"
        "A为变化字段，B为独立视觉保护字段，C提供两个原始标注形成的文本属性；8字符代码只是辅助控制。\n\n"
        "文本属性是已给出的独立记录事实，不是从像素推断的新结论。各原始图像簇不跨块复用。\n\n"
        "独立拼装对照只可比较视觉字段；文本属性未推理不能计作错误或正确，不能比较全字段严格成功率。\n\n"
        "当前V4 validation样本很小，报告计数/分母和描述性比例，不作显著性或TMM主实验优势结论。\n\n"
        "本轮不训练模型，不等价于Protect训练项的因果消融，也不是患者身份错配检测。\n").encode("utf-8"))
    print(f"FOLLOWUP_SCORE_COMPLETE output={output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare", "preflight", "run", "score", "status"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    os.umask(0o077)
    if args.command in ("audit", "prepare"):
        if args.source_config is None:
            parser.error("--source-config is required for audit/prepare")
        source = followup_audit.load_source(args.source_config)
        isolated_output(work, source)
        with common.run_lock(work):
            if args.command == "audit":
                export_audit(source, work)
            else:
                prepare(args.source_config, work, args.seed, source)
        return
    if args.command == "status":
        config, tasks = load_work(work)
        counts = {arm: len(common.unique_rows(work / "arms" / arm / "responses.jsonl")) for arm in common.TEXT_ARMS}
        print(json.dumps({"expected_each_arm": len(tasks), "saved": counts,
                          "lock_exists": (work / ".execution.lock").exists()}, indent=2))
        return
    with common.run_lock(work):
        if args.command == "score":
            score(work)
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        config, tasks = load_work(work)
        manifest = bind_runtime(work, config, args.full_hash)
        print(f"FOLLOWUP_PREFLIGHT_PASS tasks={len(tasks)} protocol={manifest['protocol_sha256']}", flush=True)
        if args.command == "run":
            for arm in common.TEXT_ARMS:
                generate_arm(work, config, tasks, manifest, arm)
            score(work)


if __name__ == "__main__":
    main()
