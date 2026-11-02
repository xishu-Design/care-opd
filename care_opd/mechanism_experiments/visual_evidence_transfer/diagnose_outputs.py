#!/usr/bin/env python3
"""Read-only, gold-free runtime diagnostics; not a scorer or selection gate.

Usage: python diagnose_outputs.py --config /path/to/config.json
Only tasks.jsonl and the six first-output ledgers are read. No model, image,
gold, manifest, or infrastructure-error ledger is loaded. Output is aggregate
JSON on stdout, never raw model text, task contents, hashes, or private labels.
"""
from __future__ import annotations

import sys

# Importing the parser must not create __pycache__ in an immutable run package.
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re

try:
    from .score_results import parse_native_output
except ImportError:
    from score_results import parse_native_output


ARMS = ("vlm_direct", "clean_sft", "gkd_full", "tropd_full", "care_opd")
CONDITIONS = ("clean", "no_image", "swapped_image")
ANSWERS = ("YES", "NO", "ABSTAIN", "invalid")
UNABLE_PREFIXES = ("无法", "无法确定", "不能", "未提供图像", "unable", "Unable", "cannot", "Cannot", "I cannot")
OUTER_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n[\s\S]*?\r?\n```\Z", re.I)
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
MAX_LINE_BYTES = 4 * 1024 * 1024
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_WARNING_DETAILS = 200


class Warnings:
    def __init__(self):
        self.counts = Counter()
        self.details = []

    def add(self, source, code, line=None):
        self.counts[code] += 1
        if len(self.details) < MAX_WARNING_DETAILS:
            item = {"source": source, "code": code}
            if line is not None:
                item["line"] = line
            self.details.append(item)

    def report(self):
        return {"counts": dict(self.counts), "details": self.details,
                "details_truncated": sum(self.counts.values()) > len(self.details)}


def read_snapshot(path, source, warnings):
    """Bound reads at the initial file size; never count an unterminated tail.

    Even syntactically valid JSON without a final newline is an uncommitted
    trailing row. Appends, rewrites, invalid lines and bounds prevent a complete
    snapshot declaration. Errors report line numbers, not contents or paths.
    """
    state = {"exists": False, "partial_trailing_line": False,
             "changed_during_read": False, "read_limit_reached": False,
             "invalid_lines": 0, "parsed_rows": 0, "readable_snapshot": False}
    rows = []
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            state["exists"] = True
            remaining = min(before.st_size, MAX_FILE_BYTES)
            state["read_limit_reached"] = before.st_size > MAX_FILE_BYTES
            number = 0
            while remaining:
                number += 1
                line = handle.readline(min(remaining, MAX_LINE_BYTES + 1))
                remaining -= len(line)
                if not line:
                    state["changed_during_read"] = True
                    break
                if len(line) > MAX_LINE_BYTES:
                    state["read_limit_reached"] = True
                    break
                if not line.endswith(b"\n"):
                    state["partial_trailing_line"] = True
                    warnings.add(source, "partial_trailing_line", number)
                    break
                try:
                    text = line.decode("utf-8-sig" if number == 1 else "utf-8")
                    if not text.strip():
                        continue
                    row = json.loads(text, object_pairs_hook=_unique_keys)
                    if not isinstance(row, dict):
                        raise ValueError("not_object")
                except (UnicodeError, ValueError, RecursionError):
                    state["invalid_lines"] += 1
                    warnings.add(source, "invalid_jsonl_line", number)
                    continue
                rows.append((number, row))
            after = os.fstat(handle.fileno())
            state["changed_during_read"] |= (before.st_size, before.st_mtime_ns) != (
                after.st_size, after.st_mtime_ns)
    except FileNotFoundError:
        warnings.add(source, "missing_file")
    except OSError:
        warnings.add(source, "unreadable_file")
        return rows, state
    if state["changed_during_read"]:
        warnings.add(source, "changed_during_read")
    if state["read_limit_reached"]:
        warnings.add(source, "read_limit_reached")
    state["parsed_rows"] = len(rows)
    state["readable_snapshot"] = (state["exists"] and not any(state[key] for key in (
        "partial_trailing_line", "changed_during_read", "read_limit_reached", "invalid_lines")))
    return rows, state


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _nonempty(value):
    return isinstance(value, str) and bool(value)


def _load_tasks(path, warnings):
    rows, state = read_snapshot(path, "tasks", warnings)
    tasks = {}
    rejected = 0
    for line, row in rows:
        task_id = row.get("task_id")
        if not _nonempty(task_id):
            warnings.add("tasks", "invalid_task_id", line)
            rejected += 1
        elif task_id in tasks:
            warnings.add("tasks", "duplicate_task_id", line)
            rejected += 1
        elif (row.get("condition") not in CONDITIONS
              or not _nonempty(row.get("family_id"))
              or not _nonempty(row.get("context_value"))):
            warnings.add("tasks", "invalid_task_metadata", line)
            rejected += 1
        else:
            # Keep only the public fields needed for aggregate diagnostics.
            tasks[task_id] = {key: row[key] for key in ("condition", "family_id", "context_value")}
    state["rejected_rows"] = rejected
    state["valid_task_snapshot"] = state["readable_snapshot"] and not rejected and bool(tasks)
    if not tasks:
        warnings.add("tasks", "no_valid_tasks")
    return tasks, state


def _load_first_rows(path, source, tasks, warnings):
    rows, state = read_snapshot(path, source, warnings)
    first = {}
    seen = set()
    counts = Counter()
    for line, row in rows:
        task_id = row.get("task_id")
        if not _nonempty(task_id):
            code = "invalid_task_id"
        elif task_id in seen:
            code = "duplicate_task_id"
        elif task_id not in tasks:
            seen.add(task_id)
            code = "unknown_task_id"
        else:
            seen.add(task_id)
            first[task_id] = row
            if row.get("condition") != tasks[task_id]["condition"]:
                counts["condition_mismatch"] += 1
                warnings.add(source, "condition_mismatch", line)
            if row.get("status") != "completed":
                counts["non_completed_first_row"] += 1
                warnings.add(source, "non_completed_first_row", line)
            continue
        counts[code] += 1
        warnings.add(source, code, line)
    completed = {key: value for key, value in first.items() if value.get("status") == "completed"}
    state.update({"first_rows": len(first), "first_completions": len(completed),
                  "missing_first_completions": len(tasks) - len(completed),
                  "row_warnings": dict(counts)})
    state["complete"] = bool(tasks) and state["readable_snapshot"] and not counts and len(completed) == len(tasks)
    return completed, state


def _numbers(rows, key):
    values = [row.get(key) for row in rows]
    valid = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and value >= 0]
    return {"n": len(valid), "missing_or_invalid": len(values) - len(valid),
            "mean": sum(valid) / len(valid) if valid else None,
            "min": min(valid) if valid else None, "max": max(valid) if valid else None}


def _native_diagnostic(raw, context):
    try:
        parsed, error = parse_native_output(raw)
    except RecursionError:
        parsed, error = None, "json_nesting_limit"
    sections = parsed.get("sections") if isinstance(parsed, dict) else None
    sections = sections if isinstance(sections, list) else []
    visual = [item for item in sections if isinstance(item, dict) and item.get("heading") == "visual_answer"]
    contexts = [item for item in sections if isinstance(item, dict) and item.get("heading") == "context_value"]
    answer = visual[0].get("content") if len(visual) == 1 else None
    answer = answer if isinstance(answer, str) and answer in ANSWERS[:3] else "invalid"
    expected = ["image:current"] if answer in ("YES", "NO") else []
    visual_error = len(visual) != 1 or visual[0].get("source_ids") != expected
    context_error = len(contexts) != 1 or contexts[0].get("source_ids") != ["text:context"]
    return {"answer": answer, "parse_error": error is not None,
            "outer_markdown_fence": isinstance(raw, str) and bool(OUTER_FENCE.fullmatch(raw.strip())),
            "visual_source_ids_error": visual_error, "context_source_ids_error": context_error,
            "source_ids_error": visual_error or context_error,
            "context_exact": len(contexts) == 1 and contexts[0].get("content") == context}


def _answer_summary(tasks, rows):
    result = {}
    for condition in CONDITIONS:
        ids = [key for key, task in tasks.items() if task["condition"] == condition]
        diagnostics = [_native_diagnostic(rows[key].get("raw_text"), tasks[key]["context_value"])
                       for key in ids if key in rows]
        counts = Counter(item["answer"] for item in diagnostics)
        exact = sum(item["context_exact"] for item in diagnostics)
        result[condition] = {
            "expected": len(ids), "first_completions": len(diagnostics),
            "missing_first_completions": len(ids) - len(diagnostics),
            "final_answer_distribution": {answer: counts[answer] for answer in ANSWERS},
            **{key + "_count": sum(item[key] for item in diagnostics) for key in (
                "parse_error", "outer_markdown_fence", "visual_source_ids_error", "context_source_ids_error", "source_ids_error")},
            "context_field": {"exact_count": exact, "completed_denominator": len(diagnostics),
                              "accuracy_among_first_completions": exact / len(diagnostics) if diagnostics else None,
                              "expected_denominator": len(ids),
                              "accuracy_over_expected_tasks": exact / len(ids) if ids else None},
        }
    return result


def _observation_summary(tasks, rows):
    result = {}
    for condition in CONDITIONS:
        ids = [key for key, task in tasks.items() if task["condition"] == condition]
        selected = [rows[key] for key in ids if key in rows]
        texts = [row["raw_text"] for row in selected if isinstance(row.get("raw_text"), str)]
        reasons = Counter()
        for row in selected:
            reason = row.get("finish_reason")
            reasons[reason if isinstance(reason, str) and reason in (
                "stop", "length", "max_length", "max_tokens", "not_applicable") else "missing_or_other"] += 1
        result[condition] = {
            "expected": len(ids), "first_completions": len(selected),
            "missing_first_completions": len(ids) - len(selected),
            "finish_reason_counts": dict(reasons),
            "length_finish_count": sum(reasons[key] for key in ("length", "max_length", "max_tokens")),
            "raw_text_missing_or_invalid": len(selected) - len(texts),
            "text_length_characters": _numbers([{"length": len(value)} for value in texts], "length"),
            "completion_tokens": _numbers(selected, "completion_tokens"),
            "prompt_tokens": _numbers(selected, "prompt_tokens"),
            "literal_unable_prefix_count": sum(value.lstrip().startswith(UNABLE_PREFIXES) for value in texts),
            "literal_unable_prefix_counts": {prefix: sum(value.lstrip().startswith(prefix) for value in texts)
                                             for prefix in UNABLE_PREFIXES},
        }
    return result


def _pair_diagnostics(tasks, observations, arms, warnings):
    families = defaultdict(lambda: defaultdict(list))
    for key, task in tasks.items():
        families[task["family_id"]][task["condition"]].append(key)
    pairs = []
    ambiguous = 0
    for conditions in families.values():
        if len(conditions["clean"]) > 1 or len(conditions["swapped_image"]) > 1:
            ambiguous += 1
        elif len(conditions["clean"]) == len(conditions["swapped_image"]) == 1:
            pairs.append((conditions["clean"][0], conditions["swapped_image"][0]))
    if ambiguous:
        warnings.add("tasks", "ambiguous_clean_swapped_families")

    def compare(rows, field, valid):
        available = identical = 0
        for clean, swapped in pairs:
            left, right = rows.get(clean, {}).get(field), rows.get(swapped, {}).get(field)
            if valid(left) and valid(right):
                available += 1
                identical += left == right
        return {"expected_pairs": len(pairs), "comparable_pairs": available, "identical_pairs": identical,
                "different_pairs": available - identical, "missing_or_invalid_pairs": len(pairs) - available}

    valid_hash = lambda value: isinstance(value, str) and bool(SHA256.fullmatch(value))
    per_arm = {}
    for arm in ARMS[1:]:
        rows = arms[arm]
        hashes = [row.get("input_sha256") for row in rows.values()]
        valid = [value for value in hashes if valid_hash(value)]
        per_arm[arm] = {**compare(rows, "input_sha256", valid_hash),
                        "completed_rows_with_valid_input_hash": len(valid),
                        "missing_or_invalid_input_hash_count": len(hashes) - len(valid),
                        "distinct_input_hash_count": len(set(valid))}
    shared = equal = 0
    for key in tasks:
        values = [arms[arm].get(key, {}).get("input_sha256") for arm in ARMS[1:]]
        if all(valid_hash(value) for value in values):
            shared += 1
            equal += len(set(values)) == 1
    return {"ambiguous_family_count": ambiguous,
            "observation_text": compare(observations, "raw_text", lambda value: isinstance(value, str)),
            "per_text_arm_input_hash": per_arm,
            "cross_text_arm_same_task_input_hash": {"comparable_tasks": shared, "all_four_identical_tasks": equal,
                                                    "mismatched_tasks": shared - equal}}


def diagnose(config_path):
    """Return counts only. Relative paths follow the runtime's cwd semantics."""
    with Path(config_path).open(encoding="utf-8-sig") as handle:
        config = json.load(handle, object_pairs_hook=_unique_keys)
    if not isinstance(config, dict) or not all(_nonempty(config.get(key)) for key in ("tasks_dir", "run_root")):
        raise ValueError("config_requires_tasks_dir_and_run_root")
    task_root = Path(config["tasks_dir"]).expanduser().resolve()
    run_root = Path(config["run_root"]).expanduser().resolve()
    warnings = Warnings()
    tasks, task_state = _load_tasks(task_root / "tasks.jsonl", warnings)
    observations, observation_state = _load_first_rows(
        run_root / "observations" / "observations.jsonl", "observations", tasks, warnings)
    arms, arm_states = {}, {}
    for arm in ARMS:
        arms[arm], arm_states[arm] = _load_first_rows(
            run_root / "arms" / arm / "responses.jsonl", arm, tasks, warnings)
    pairs = _pair_diagnostics(tasks, observations, arms, warnings)
    complete = (task_state["valid_task_snapshot"] and observation_state["complete"]
                and all(state["complete"] for state in arm_states.values())
                and not pairs["ambiguous_family_count"])
    return {"schema_version": "care_opd.visual_transfer.runtime_diagnostics.v1",
            "diagnostic_only": True, "gold_used": False, "model_selection_gate": False,
            "first_output_policy": "first_row_per_task_only_never_select_duplicate_or_retry",
            "snapshot_complete": bool(complete),
            "completion_scope": "first-output ledger coverage only; not run validity or scientific success; files are not atomically snapshotted together",
            "literal_prefix_scope": "case-sensitive startswith after leading-whitespace removal; individual prefix counts may overlap; not calibrated uncertainty or refusal quality",
            "answer_scope": "extracts the sole visual_answer section; invalid means absent, ambiguous or non-YES/NO/ABSTAIN, not a full format score",
            "context_scope": "exact copy of public task context_value; no visual answer accuracy is computed",
            "tasks": task_state,
            "observations": {"ledger": observation_state, "by_condition": _observation_summary(tasks, observations)},
            "arms": {arm: {"ledger": arm_states[arm], "by_condition": _answer_summary(tasks, arms[arm])} for arm in ARMS},
            "clean_swapped_identity": pairs, "warnings": warnings.report()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.config)
    except (OSError, ValueError, RecursionError):
        # Do not echo exception strings: parse failures may include private data.
        print(json.dumps({"diagnostic_only": True, "gold_used": False, "model_selection_gate": False,
                          "snapshot_complete": False, "error": "configuration_unreadable_or_invalid"}))
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
