"""Read-only diagnostics for a complete V4 validation run; no inference or writes."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import common
import evaluate
import score_results as scorer

SCHEMA = "care_opd.visual_transfer.followup_audit.v1"
RULES = ("always_abstain", "missing_abstain_else_yes", "missing_abstain_else_no")
FROZEN_CODE = ("common.py", "evaluate.py", "prepare_slake.py", "score_results.py")


class SourceAuditError(ValueError):
    """A source run cannot be safely used for the bounded follow-up."""


def _snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for role, path in sorted(paths.items()):
        if not path.exists():
            result[role] = {"exists": False}
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise SourceAuditError(f"source input is not a file: {path}")
        before = resolved.stat()
        sha = scorer._sha256_file(resolved)
        after = resolved.stat()
        signature = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        if signature(before) != signature(after) or path.resolve(strict=True) != resolved:
            raise SourceAuditError(f"source input changed while hashing: {path}")
        result[role] = {"exists": True, "bytes": after.st_size, "sha256": sha,
                        "_resolved": str(resolved), "_stat": signature(after)}
    return result


def _input_paths(config_path: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    tasks_dir, run_root = Path(config["tasks_dir"]), Path(config["run_root"])
    paths = {"config": config_path, "manifest": run_root / "run_manifest.json"}
    for name in ("tasks.jsonl", "gold.private.jsonl", "selection_manifest.json"):
        paths["bundle/" + name] = tasks_dir / name
    paths["observations"] = run_root / "observations" / "observations.jsonl"
    for arm in common.ARMS:
        paths["responses/" + arm] = run_root / "arms" / arm / "responses.jsonl"
    directories = {"observations": run_root / "observations"}
    directories.update({"arms/" + arm: run_root / "arms" / arm for arm in common.ARMS})
    for role, directory in directories.items():
        for name in ("infrastructure_errors.jsonl", "infra_attempts.jsonl"):
            paths["optional/" + role + "/" + name] = directory / name
    for name in ("device_migrations.jsonl", "runtime/device_migrations.jsonl"):
        paths["optional/" + name] = run_root / name
    for name in (*FROZEN_CODE, Path(__file__).name):
        paths["code/" + name] = Path(__file__).resolve().parent / name
    return paths


def load_source(config_path: Path) -> dict[str, Any]:
    """Validate complete source inputs without GPU/model loading or output writes."""
    config_path = Path(config_path).expanduser().resolve(strict=True)
    config_before = _snapshot({"config": config_path})["config"]
    config = evaluate.configuration(config_path)
    if config.get("protocol_version") != "care_opd.visual_transfer.v4":
        raise SourceAuditError("follow-up requires protocol V4, not an older source root")
    if config.get("evaluation_use") != "validation":
        raise SourceAuditError("follow-up requires evaluation_use=validation")
    paths = _input_paths(config_path, config)
    before = _snapshot(paths)
    if before["config"] != config_before:
        raise SourceAuditError("source config changed during discovery")
    for role, state in before.items():
        if not role.startswith("optional/") and not state["exists"]:
            raise SourceAuditError(f"missing required source input: {role}")
    tasks_dir, run_root = Path(config["tasks_dir"]), Path(config["run_root"])
    for task in scorer._load_jsonl(tasks_dir / "tasks.jsonl"):
        if task.get("image_path"):
            image_path = Path(task["image_path"])
            if not image_path.is_absolute():
                image_path = tasks_dir / image_path
            paths["images/" + str(task["task_id"])] = image_path
    before.update(_snapshot({role: path for role, path in paths.items() if role.startswith("images/")}))
    tasks, task_by_id, gold_by_id, hashes, bundle_audit = scorer._validate_bundle(tasks_dir)
    if not tasks:
        raise SourceAuditError("empty validation bundle")
    selection = scorer._load_json(tasks_dir / "selection_manifest.json")
    evaluate.validate_evaluation_use(config, selection)
    manifest = scorer._validate_run_manifest(run_root, hashes)
    if manifest.get("protocol_version") != "care_opd.visual_transfer.v4":
        raise SourceAuditError("source manifest is not protocol V4")
    if manifest.get("config") != config:
        raise SourceAuditError("source config differs from the frozen run manifest")
    if manifest.get("task_count") != len(tasks):
        raise SourceAuditError("source manifest task_count differs from the bundle")
    for name in FROZEN_CODE:
        if manifest.get("code_sha256", {}).get(name) != before["code/" + name]["sha256"]:
            raise SourceAuditError(f"current frozen adapter differs from source run: {name}")
    scores_by_arm, responses, arm_diagnostics = {}, {}, []
    for arm in common.ARMS:
        scores, diagnostic = scorer._load_arm(arm, run_root, tasks, task_by_id, gold_by_id, manifest)
        if not diagnostic["complete_for_inference"] or diagnostic["response_rows"] != len(tasks):
            raise SourceAuditError(f"source arm is incomplete: {arm}")
        scores_by_arm[arm] = scores
        arm_diagnostics.append(diagnostic)
        responses[arm] = {row["task_id"]: row for row in scorer._load_jsonl(paths["responses/" + arm])}
    observation_diagnostics = scorer._check_observation_identity(scores_by_arm, run_root, task_by_id, manifest)
    observations = {row["task_id"]: row for row in scorer._load_jsonl(paths["observations"])}
    if set(observations) != set(task_by_id) or observation_diagnostics["unresolved_infra_tasks"]:
        raise SourceAuditError("source shared observations are incomplete")
    for task_id, observation in observations.items():
        if not isinstance(observation.get("raw_text"), str) or not isinstance(observation.get("finish_reason"), str):
            raise SourceAuditError(f"observation text/finish reason is invalid: {task_id}")
    after = _snapshot(paths)
    if before != after:
        changed = sorted(role for role in paths if before.get(role) != after.get(role))
        raise SourceAuditError(f"source inputs changed during audit: {changed}")
    binding = {"source_protocol_sha256": manifest["protocol_sha256"],
               "source_manifest_sha256": before["manifest"]["sha256"],
               "protocol_version": manifest["protocol_version"], "evaluation_use": "validation",
               "files": {role: {key: value for key, value in state.items() if not key.startswith("_")}
                         for role, state in sorted(before.items())}}
    binding["fingerprint"] = common.digest(binding)
    return {"config": config, "manifest": manifest, "tasks": tasks, "task_by_id": task_by_id,
            "gold_by_id": gold_by_id, "observations": observations, "responses": responses,
            "scores_by_arm": scores_by_arm, "binding": binding, "selection": selection,
            "bundle_audit": bundle_audit, "arm_diagnostics": arm_diagnostics,
            "observation_diagnostics": observation_diagnostics}


def _rule_prediction(rule: str, task: Mapping[str, Any]) -> str:
    """Predict from input availability only; never accept a gold object."""
    if rule not in RULES:
        raise ValueError(f"unknown diagnostic rule: {rule}")
    if rule == "always_abstain" or not task.get("image_path"):
        return "ABSTAIN"
    return "YES" if rule == "missing_abstain_else_yes" else "NO"


def image_updates(source: Mapping[str, Any], records: list[dict]) -> list[dict]:
    families = {}
    for task in source["tasks"]:
        families.setdefault(task["family_id"], {})[task["condition"]] = task
    pairs = []
    for family, variants in families.items():
        if not {"clean", "swapped_image"}.issubset(variants):
            continue
        old, new = variants["clean"], variants["swapped_image"]
        old_gold = source["gold_by_id"][old["task_id"]]["answer"]
        new_gold = source["gold_by_id"][new["task_id"]]["answer"]
        if (old["question"] != new["question"] or {old_gold, new_gold} != {"YES", "NO"}
                or old["image_sha256"] == new["image_sha256"]):
            raise SourceAuditError(f"invalid genuinely relabelled image replacement: {family}")
        pairs.append((old["task_id"], new["task_id"]))
    output = []
    for arm in common.ARMS:
        indexed = {row["task_id"]: row for row in records if row["arm"] == arm}
        rows = [(indexed[a], indexed[b]) for a, b in pairs]
        old_correct = sum(a["answer_correct"] for a, b in rows)
        both = sum(a["answer_correct"] and b["answer_correct"] for a, b in rows)
        flips = [(a, b) for a, b in rows if a["parsed_answer"] in {"YES", "NO"}
                 and b["parsed_answer"] in {"YES", "NO"} and a["parsed_answer"] != b["parsed_answer"]]
        output.append({"arm": arm, "paired_families": len(rows), "both_correct_count": both,
                       "both_correct_rate": scorer._rate(both, len(rows)),
                       "old_correct_count": old_correct, "new_correct_count": sum(b["answer_correct"] for a, b in rows),
                       "correct_update_given_old_correct": scorer._rate(both, old_correct),
                       "binary_flip_count": len(flips),
                       "wrong_flip_count": sum(not (a["answer_correct"] and b["answer_correct"]) for a, b in flips),
                       "new_abstain_count": sum(b["parsed_answer"] == "ABSTAIN" for a, b in rows),
                       "new_unknown_count": sum(b["parsed_answer"] is None for a, b in rows),
                       "unit": "paired_family_descriptive_not_independent_condition_rows"})
    return output


def build_audit(source: Mapping[str, Any]) -> dict[str, Any]:
    """Describe every frozen output without making a pixel/medical judgement."""
    records = []
    for arm in common.ARMS:
        for score in source["scores_by_arm"][arm]:
            task_id = score["task_id"]
            task, gold = source["task_by_id"][task_id], source["gold_by_id"][task_id]
            raw = source["responses"][arm][task_id]
            observation = source["observations"][task_id]
            parsed, parse_error = scorer.parse_native_output(raw["raw_text"])
            sections = parsed.get("sections", []) if isinstance(parsed, dict) else []
            contents = [section.get("content") for section in sections
                        if isinstance(section, dict) and section.get("heading") == "visual_answer"] if isinstance(sections, list) else []
            answer = score["answer"]
            flags = {"parse_invalid": not score["parse_valid"], "schema_invalid": not score["format_valid"],
                     "unrecognized_answer": answer is None, "truncated": score["truncated"],
                     "source_invalid": not score["source_valid"], "context_inexact": not score["context_exact"],
                     "content_incorrect": not score["answer_correct"],
                     "wrong_yes_no": answer in scorer.VISIBLE_ANSWERS and not score["answer_correct"],
                     "abstain_against_visible_gold": answer == "ABSTAIN" and gold["answer"] in scorer.VISIBLE_ANSWERS,
                     "content_correct_contract_failure": score["answer_correct"] and not score["strict_contract_success"]}
            records.append({"arm": arm, "task_id": task_id, "family_id": task["family_id"],
                "cluster_id": task["cluster_id"], "condition": task["condition"],
                "question": task["question"], "image_sha256": task.get("image_sha256"),
                "observation_text": observation["raw_text"], "observation_used_by_arm": arm != "vlm_direct",
                "observation_finish_reason": observation["finish_reason"],
                "observation_truncated": observation["finish_reason"].strip().lower() in scorer.TRUNCATED_FINISH_REASONS,
                "raw_response": raw["raw_text"], "parsed_answer": answer,
                "answer_category": answer if answer is not None else "UNRECOGNIZED",
                "declared_visual_answer_contents": contents, "parse_error": parse_error,
                "reference_answer": gold["answer"], "original_reference_answer": gold["original_answer"],
                "answer_correct": score["answer_correct"], "strict_contract_success": score["strict_contract_success"],
                "format_valid": score["format_valid"], "context_exact": score["context_exact"],
                "source_valid": score["source_valid"], "finish_reason": raw["finish_reason"], "flags": flags,
                "pixel_grounding": "unknown_not_reviewed", "observation_medical_correctness": "unknown_not_reviewed"})
    summaries = []
    for arm in common.ARMS:
        for condition in (*common.CONDITIONS, "all"):
            rows = [row for row in records if row["arm"] == arm and (condition == "all" or row["condition"] == condition)]
            counts = {"arm": arm, "condition": condition, "total": len(rows)}
            for answer in ("YES", "NO", "ABSTAIN", "UNRECOGNIZED"):
                counts["answer_" + answer.lower()] = sum(row["answer_category"] == answer for row in rows)
            for name in ("answer_correct", "strict_contract_success", "format_valid", "context_exact", "source_valid", "observation_truncated"):
                counts[name + "_count"] = sum(bool(row[name]) for row in rows)
            for name in records[0]["flags"]:
                counts[name + "_count"] = sum(bool(row["flags"][name]) for row in rows)
            answered = [row for row in rows if row["parsed_answer"] in {"YES", "NO"}]
            visible_answered = [row for row in answered if row["condition"] != "no_image"]
            counts.update({"answer_accuracy": scorer._rate(counts["answer_correct_count"], len(rows)),
                           "answered_count": len(answered), "answer_coverage": scorer._rate(len(answered), len(rows)),
                           "answered_correct_count": sum(row["answer_correct"] for row in visible_answered),
                           "selective_accuracy_denominator": len(visible_answered),
                           "selective_accuracy": scorer._rate(sum(row["answer_correct"] for row in visible_answered), len(visible_answered)),
                           "selective_accuracy_scope": "visible_conditions_only_no_missing_evidence_guesses",
                           "abstain_rate": scorer._rate(counts["answer_abstain"], len(rows)),
                           "unknown_rate": scorer._rate(counts["answer_unrecognized"], len(rows))})
            guesses = [row for row in answered if row["condition"] == "no_image"]
            counts["missing_evidence_guesses"] = len(guesses)
            counts["original_answer_accuracy_among_guesses"] = scorer._rate(
                sum(row["parsed_answer"] == row["original_reference_answer"] for row in guesses), len(guesses))
            summaries.append(counts)
    rules = []
    for rule in RULES:
        predictions = {task["task_id"]: _rule_prediction(rule, task) for task in source["tasks"]}
        for condition in (*common.CONDITIONS, "all"):
            tasks = [task for task in source["tasks"] if condition == "all" or task["condition"] == condition]
            correct = sum(predictions[task["task_id"]] == source["gold_by_id"][task["task_id"]]["answer"] for task in tasks)
            rules.append({"rule": rule, "condition": condition, "total": len(tasks),
                          "answer_correct_count": correct, "answer_accuracy": scorer._rate(correct, len(tasks)),
                          "prediction_uses_gold": False, "metric_scope": "answer_accuracy_only"})
    return {"schema_version": SCHEMA, "status": "COMPLETE", "diagnostic_only": True,
            "binding": source["binding"], "per_arm_condition": summaries, "row_records": records,
            "valid_image_update": image_updates(source, records), "rule_baselines": rules,
            "limitations": [
                "Reference agreement is not an independent medical or pixel-grounding adjudication.",
                "Error flags overlap and do not assign a unique causal failure category.",
                "Unknown answers are not counted as ABSTAIN; the original scorer is unchanged.",
                "Shared observations shown for VLM-direct are not inputs consumed by that arm.",
                "Availability-only baselines diagnose a simple rule, not clinical ability.",
                "No outputs are repaired, retried, discarded or selected; all source rows remain in denominators.",
                "V4 validation after prior test inspection is not an unseen-test confirmation.",
                "Existing run model identities are manifest-validated, not rehashed in the read-only audit.",
                "Paired families may share image clusters; counts here are descriptive, not independent-trial p-values.",
            ]}
