"""Score the frozen SLAKE visual-evidence transfer experiment.

The scorer is deliberately independent from model/runtime modules.  It consumes
only the frozen task/gold bundle, the run manifest, and one immutable JSONL per
arm.  Missing rows, infrastructure failures, malformed or truncated model
outputs remain in every relevant denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # Package import in tests; direct import when invoked as a script.
    from .common import answer_messages, input_digest, observation_messages
except ImportError:  # pragma: no cover - exercised by the CLI, not package tests.
    from common import answer_messages, input_digest, observation_messages


SCHEMA_VERSION = "care_opd.visual_transfer.metrics.v1"
RUN_SCHEMA_VERSION = "care_opd.visual_transfer.run.v1"
ARMS = ("vlm_direct", "clean_sft", "gkd_full", "tropd_full", "care_opd")
BASELINES = ("clean_sft", "gkd_full", "tropd_full")
CONDITIONS = ("clean", "no_image", "swapped_image")
ANSWERS = frozenset({"YES", "NO", "ABSTAIN"})
VISIBLE_ANSWERS = frozenset({"YES", "NO"})
OUTER_KEYS = frozenset(
    {"title", "summary", "sections", "warnings", "next_actions", "disclaimer"}
)
SECTION_KEYS = frozenset({"heading", "content", "source_ids"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\Z",
    re.IGNORECASE,
)
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_length", "max_tokens"})


class ScoringError(ValueError):
    """Raised when immutable inputs or response ledgers violate the protocol."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ScoringError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _json_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except ScoringError:
        raise
    except json.JSONDecodeError as exc:
        raise ScoringError(f"{label}: invalid JSON: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScoringError(f"missing required file: {path}")
    payload = _json_loads(path.read_text(encoding="utf-8-sig"), label=str(path))
    if not isinstance(payload, dict):
        raise ScoringError(f"{path}: expected a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ScoringError(f"missing required file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = _json_loads(line, label=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise ScoringError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    if not rows:
        raise ScoringError(f"{path}: JSONL is empty")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value.lower()))


def _require_text(row: Mapping[str, Any], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ScoringError(f"{label}: {key} must be a non-empty string")
    return value


def _require_number(row: Mapping[str, Any], key: str, label: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError(f"{label}: {key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ScoringError(f"{label}: {key} must be finite and non-negative")
    return number


def _require_count(row: Mapping[str, Any], key: str, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScoringError(f"{label}: {key} must be a non-negative integer")
    return value


def _validate_bundle(
    tasks_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, Any],
]:
    tasks_path = tasks_dir / "tasks.jsonl"
    gold_path = tasks_dir / "gold.private.jsonl"
    selection_path = tasks_dir / "selection_manifest.json"
    tasks = _load_jsonl(tasks_path)
    gold_rows = _load_jsonl(gold_path)
    selection = _load_json(selection_path)

    hashes = {
        "tasks_sha256": _sha256_file(tasks_path),
        "gold_sha256": _sha256_file(gold_path),
        "selection_manifest_sha256": _sha256_file(selection_path),
    }
    for field in ("tasks_sha256", "gold_sha256"):
        declared = selection.get(field)
        if declared != hashes[field]:
            raise ScoringError(
                f"selection_manifest.json {field} mismatch: "
                f"declared={declared!r} actual={hashes[field]}"
            )

    task_by_id: dict[str, dict[str, Any]] = {}
    family_tasks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    image_cluster_by_sha: dict[str, str] = {}
    for index, task in enumerate(tasks):
        label = f"tasks.jsonl row {index + 1}"
        task_id = _require_text(task, "task_id", label)
        if task_id in task_by_id:
            raise ScoringError(f"duplicate task_id in tasks.jsonl: {task_id}")
        family_id = _require_text(task, "family_id", label)
        cluster_id = _require_text(task, "cluster_id", label)
        condition = _require_text(task, "condition", label)
        if condition not in CONDITIONS:
            raise ScoringError(f"{label}: unknown condition {condition!r}")
        _require_text(task, "question", label)
        context = _require_text(task, "context_value", label)
        if len(context) != 8:
            raise ScoringError(f"{label}: context_value must contain exactly 8 characters")
        if condition in {"clean", "swapped_image"}:
            image_path = _require_text(task, "image_path", label)
            image_sha = task.get("image_sha256")
            if not _is_sha256(image_sha):
                raise ScoringError(f"{label}: image_sha256 must be a SHA-256 digest")
            # The frozen adapter records the resolved SLAKE asset path.  It is
            # intentionally allowed to live outside the small task bundle; the
            # content hash, rather than directory containment, binds the image.
            raw_image_path = Path(image_path)
            resolved = (
                raw_image_path.resolve()
                if raw_image_path.is_absolute()
                else (tasks_dir / raw_image_path).resolve()
            )
            if not resolved.is_file():
                raise ScoringError(f"{label}: missing image {resolved}")
            if _sha256_file(resolved) != str(image_sha).lower():
                raise ScoringError(f"{label}: image SHA-256 mismatch")
            previous_cluster = image_cluster_by_sha.setdefault(
                str(image_sha).lower(), cluster_id
            )
            if previous_cluster != cluster_id:
                raise ScoringError(
                    f"{label}: identical image bytes occur in different bootstrap "
                    f"components ({previous_cluster!r}, {cluster_id!r})"
                )
        else:
            if task.get("image_path") not in (None, ""):
                raise ScoringError(f"{label}: no_image task must not expose image_path")
            if task.get("image_sha256") not in (None, ""):
                raise ScoringError(f"{label}: no_image task must not expose image_sha256")
        if condition in family_tasks[family_id]:
            raise ScoringError(f"family {family_id}: duplicate {condition} task")
        task_by_id[task_id] = task
        family_tasks[family_id][condition] = task

    gold_by_id: dict[str, dict[str, Any]] = {}
    for index, gold in enumerate(gold_rows):
        label = f"gold.private.jsonl row {index + 1}"
        task_id = _require_text(gold, "task_id", label)
        if task_id in gold_by_id:
            raise ScoringError(f"duplicate task_id in gold.private.jsonl: {task_id}")
        answer = _require_text(gold, "answer", label)
        original = _require_text(gold, "original_answer", label)
        if answer not in ANSWERS or original not in VISIBLE_ANSWERS:
            raise ScoringError(f"{label}: invalid answer/original_answer")
        _require_text(gold, "family_id", label)
        _require_text(gold, "cluster_id", label)
        _require_text(gold, "condition", label)
        context = _require_text(gold, "context_value", label)
        if len(context) != 8:
            raise ScoringError(f"{label}: context_value must contain exactly 8 characters")
        gold_by_id[task_id] = gold

    if set(task_by_id) != set(gold_by_id):
        missing_gold = sorted(set(task_by_id) - set(gold_by_id))
        extra_gold = sorted(set(gold_by_id) - set(task_by_id))
        raise ScoringError(
            f"task/gold ID mismatch: missing_gold={missing_gold[:5]} "
            f"extra_gold={extra_gold[:5]}"
        )

    for task_id, task in task_by_id.items():
        gold = gold_by_id[task_id]
        for field in ("family_id", "cluster_id", "condition", "context_value"):
            if task.get(field) != gold.get(field):
                raise ScoringError(f"{task_id}: task/gold {field} mismatch")
        condition = str(task["condition"])
        if condition == "clean" and gold["answer"] != gold["original_answer"]:
            raise ScoringError(f"{task_id}: clean answer must equal original_answer")
        if condition == "no_image" and gold["answer"] != "ABSTAIN":
            raise ScoringError(f"{task_id}: no_image gold must be ABSTAIN")
        if condition == "swapped_image":
            if gold["answer"] == gold["original_answer"]:
                raise ScoringError(f"{task_id}: swapped gold must oppose original_answer")
            _require_text(gold, "counterpart_qid", f"gold {task_id}")

    for family_id, by_condition in family_tasks.items():
        # A swapped task exists only when the official split supplies the same
        # question on a different image with the opposite gold answer.  Clean
        # and no_image are mandatory; unmatched families must not receive an
        # invented counterfactual.
        if not {"clean", "no_image"}.issubset(by_condition) or not set(
            by_condition
        ).issubset(CONDITIONS):
            raise ScoringError(
                f"family {family_id}: expected clean/no_image and optional swapped_image, "
                f"found {sorted(by_condition)}"
            )
        clean, no_image = (by_condition[name] for name in ("clean", "no_image"))
        family_rows = list(by_condition.values())
        for field in ("question", "context_value", "cluster_id"):
            if len({row[field] for row in family_rows}) != 1:
                raise ScoringError(f"family {family_id}: inconsistent {field} across conditions")
        if "swapped_image" in by_condition:
            swapped = by_condition["swapped_image"]
            if clean["image_sha256"] == swapped["image_sha256"]:
                raise ScoringError(
                    f"family {family_id}: swapped image must differ from clean image"
                )
        family_gold = [gold_by_id[row["task_id"]] for row in family_rows]
        if len({row["original_answer"] for row in family_gold}) != 1:
            raise ScoringError(f"family {family_id}: inconsistent original_answer")

    families_by_cluster: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        families_by_cluster[str(task["cluster_id"])].add(str(task["family_id"]))
    cluster_family_counts = {
        cluster: len(families) for cluster, families in families_by_cluster.items()
    }
    largest_cluster = max(cluster_family_counts.values(), default=0)
    family_count = len(family_tasks)
    largest_fraction = largest_cluster / family_count if family_count else 0.0
    cluster_warnings: list[str] = []
    if family_count >= 8 and largest_fraction >= 0.25:
        cluster_warnings.append(
            "A bootstrap connected component contains at least 25% of selected "
            "families; confidence intervals may have low effective sample size."
        )
    bundle_audit = {
        "condition_counts": {
            condition: sum(task["condition"] == condition for task in tasks)
            for condition in CONDITIONS
        },
        "family_count": family_count,
        "matched_swapped_family_count": sum(
            "swapped_image" in states for states in family_tasks.values()
        ),
        "unmatched_family_count": sum(
            "swapped_image" not in states for states in family_tasks.values()
        ),
        "cluster_count": len(families_by_cluster),
        "cluster_family_counts": dict(sorted(cluster_family_counts.items())),
        "largest_cluster_family_count": largest_cluster,
        "largest_cluster_family_fraction": largest_fraction,
        "warnings": cluster_warnings,
    }
    return tasks, task_by_id, gold_by_id, hashes, bundle_audit


def _validate_run_manifest(
    run_root: Path,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    manifest = _load_json(run_root / "run_manifest.json")
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ScoringError("run_manifest.json: unsupported schema_version")
    if manifest.get("arms") != list(ARMS):
        raise ScoringError(f"run_manifest.json: arms must be exactly {list(ARMS)}")
    for field, actual in hashes.items():
        if manifest.get(field) != actual:
            raise ScoringError(
                f"run_manifest.json {field} mismatch: "
                f"declared={manifest.get(field)!r} actual={actual}"
            )
    protocol_sha = manifest.get("protocol_sha256")
    if not _is_sha256(protocol_sha):
        raise ScoringError("run_manifest.json: protocol_sha256 must be SHA-256")
    closed_manifest = {
        key: value for key, value in manifest.items() if key != "protocol_sha256"
    }
    if _digest(closed_manifest) != protocol_sha:
        raise ScoringError(
            "run_manifest.json: protocol_sha256 does not close over the manifest"
        )
    fingerprints = manifest.get("model_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ScoringError("run_manifest.json: model_fingerprints must be an object")
    for key in ("vlm", "clean_sft", "gkd_full", "tropd_full", "care_opd"):
        if not _is_sha256(fingerprints.get(key)):
            raise ScoringError(f"run_manifest.json: missing model fingerprint {key}")
    model_assets = manifest.get("model_assets")
    if not isinstance(model_assets, dict):
        raise ScoringError("run_manifest.json: model_assets must be an object")
    for key in ("vlm", "clean_sft", "gkd_full", "tropd_full", "care_opd"):
        asset = model_assets.get(key)
        if not isinstance(asset, dict) or asset.get("sha256") != fingerprints[key]:
            raise ScoringError(
                f"run_manifest.json: model_fingerprints.{key} does not match "
                "model_assets sha256"
            )
    return manifest


def parse_native_output(raw_text: str) -> tuple[Any | None, str | None]:
    """Parse native JSON, allowing only a complete outer Markdown fence."""

    if not isinstance(raw_text, str):
        return None, "raw_text_not_string"
    stripped = raw_text.strip()
    fence = FENCE_RE.fullmatch(stripped)
    if fence:
        stripped = fence.group("body").strip()
    elif stripped.startswith("```") or stripped.endswith("```"):
        return None, "incomplete_or_non_outer_markdown_fence"
    try:
        parsed = _json_loads(stripped, label="model output")
    except ScoringError as exc:
        return None, str(exc)
    return parsed, None


def _score_native_output(raw_text: str, gold: Mapping[str, Any]) -> dict[str, Any]:
    parsed, parse_error = parse_native_output(raw_text)
    answer: str | None = None
    context_content: str | None = None
    all_source_ids: list[str] = []
    visual_sources: Any = None
    context_sources: Any = None
    shape_valid = False

    if isinstance(parsed, dict):
        sections = parsed.get("sections")
        if isinstance(sections, list):
            visual_sections: list[dict[str, Any]] = []
            context_sections: list[dict[str, Any]] = []
            section_shapes_valid = True
            seen_headings: set[str] = set()
            for section in sections:
                if not isinstance(section, dict):
                    section_shapes_valid = False
                    continue
                if set(section) != set(SECTION_KEYS):
                    section_shapes_valid = False
                heading = section.get("heading")
                content = section.get("content")
                source_ids = section.get("source_ids")
                if not isinstance(heading, str) or not isinstance(content, str):
                    section_shapes_valid = False
                if not isinstance(source_ids, list) or not all(
                    isinstance(item, str) for item in source_ids
                ):
                    section_shapes_valid = False
                else:
                    all_source_ids.extend(source_ids)
                if isinstance(heading, str):
                    if heading in seen_headings:
                        section_shapes_valid = False
                    seen_headings.add(heading)
                if heading == "visual_answer":
                    visual_sections.append(section)
                elif heading == "context_value":
                    context_sections.append(section)
            if len(visual_sections) == 1:
                answer_value = visual_sections[0].get("content")
                if isinstance(answer_value, str) and answer_value in ANSWERS:
                    answer = answer_value
                visual_sources = visual_sections[0].get("source_ids")
            if len(context_sections) == 1:
                content_value = context_sections[0].get("content")
                if isinstance(content_value, str):
                    context_content = content_value
                context_sources = context_sections[0].get("source_ids")
            outer_types_valid = (
                isinstance(parsed.get("title"), str)
                and isinstance(parsed.get("summary"), str)
                and isinstance(parsed.get("disclaimer"), str)
                and isinstance(parsed.get("warnings"), list)
                and all(isinstance(item, str) for item in parsed.get("warnings", []))
                and isinstance(parsed.get("next_actions"), list)
                and all(isinstance(item, str) for item in parsed.get("next_actions", []))
            )
            shape_valid = (
                set(parsed) == set(OUTER_KEYS)
                and outer_types_valid
                and section_shapes_valid
                and len(sections) == 2
                and len(visual_sections) == 1
                and len(context_sections) == 1
            )

    condition = str(gold["condition"])
    expected_visual_sources = ["image:current"] if answer in VISIBLE_ANSWERS else []
    allowed_sources = (
        {"text:context"}
        if condition == "no_image"
        else {"text:context", "image:current"}
    )
    invented_source = any(source not in allowed_sources for source in all_source_ids)
    source_valid = (
        visual_sources == expected_visual_sources
        and context_sources == ["text:context"]
        and not invented_source
    )
    return {
        "parse_valid": parse_error is None,
        "parse_error": parse_error,
        "format_valid": bool(shape_valid),
        "answer": answer,
        "answer_correct": answer == gold["answer"],
        "answered": answer in VISIBLE_ANSWERS,
        "context_exact": context_content == gold["context_value"],
        "source_valid": bool(source_valid),
        "invented_source": bool(invented_source),
    }


def _validate_response_row(
    row: Mapping[str, Any],
    *,
    arm: str,
    task: Mapping[str, Any],
    gold: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    label = f"{arm}/{task_id}"
    for field in ("task_id", "family_id", "cluster_id", "condition"):
        if row.get(field) != task.get(field):
            raise ScoringError(f"{label}: response {field} does not match task")
    if "arm" in row and row.get("arm") != arm:
        raise ScoringError(f"{label}: response arm does not match its ledger")
    status = row.get("status")
    if status != "completed":
        raise ScoringError(
            f"{label}: responses.jsonl may contain only frozen completed outputs; "
            "infrastructure failures belong in infrastructure_errors.jsonl"
        )
    protocol_sha = row.get("protocol_sha256")
    if protocol_sha != manifest["protocol_sha256"]:
        raise ScoringError(f"{label}: protocol_sha256 mismatch")
    fingerprint_key = "vlm" if arm == "vlm_direct" else arm
    expected_fingerprint = manifest["model_fingerprints"][fingerprint_key]
    if row.get("model_fingerprint") != expected_fingerprint:
        raise ScoringError(f"{label}: model_fingerprint mismatch")
    if not _is_sha256(row.get("input_sha256")):
        raise ScoringError(f"{label}: invalid input_sha256")
    if row.get("image_sha256") != task.get("image_sha256"):
        raise ScoringError(f"{label}: image_sha256 does not match the frozen task")
    observation_sha = row.get("observation_sha256")
    if arm == "vlm_direct":
        if observation_sha is not None:
            raise ScoringError(f"{label}: vlm_direct observation_sha256 must be null")
    elif not _is_sha256(observation_sha):
        raise ScoringError(f"{label}: downstream observation_sha256 must be non-null SHA-256")
    if not isinstance(row.get("image_seen"), bool):
        raise ScoringError(f"{label}: image_seen must be boolean")
    expected_seen = arm == "vlm_direct" and task["condition"] != "no_image"
    if status == "completed" and row["image_seen"] is not expected_seen:
        raise ScoringError(f"{label}: image_seen contradicts condition")
    if not isinstance(row.get("raw_text"), str):
        raise ScoringError(f"{label}: raw_text must be a string")
    if not isinstance(row.get("finish_reason"), str):
        raise ScoringError(f"{label}: finish_reason must be a string")
    latency = _require_number(row, "latency_seconds", label)
    prompt_tokens = _require_count(row, "prompt_tokens", label)
    completion_tokens = _require_count(row, "completion_tokens", label)
    upstream_latency = _require_number(row, "upstream_latency_seconds", label)
    end_to_end_latency = _require_number(row, "end_to_end_latency_seconds", label)
    upstream_prompt_tokens = _require_count(row, "upstream_prompt_tokens", label)
    upstream_completion_tokens = _require_count(
        row, "upstream_completion_tokens", label
    )
    if not isinstance(row.get("upstream_image_seen"), bool):
        raise ScoringError(f"{label}: upstream_image_seen must be boolean")
    expected_upstream_seen = arm != "vlm_direct" and task["condition"] != "no_image"
    if status == "completed" and row["upstream_image_seen"] is not expected_upstream_seen:
        raise ScoringError(f"{label}: upstream_image_seen contradicts arm/condition")
    if not math.isclose(
        end_to_end_latency,
        latency + upstream_latency,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ScoringError(f"{label}: end_to_end_latency_seconds is inconsistent")

    if status == "completed":
        native = _score_native_output(str(row["raw_text"]), gold)
        truncated = row["finish_reason"].strip().lower() in TRUNCATED_FINISH_REASONS
    else:
        native = {
            "parse_valid": False,
            "parse_error": "infra_error",
            "format_valid": False,
            "answer": None,
            "answer_correct": False,
            "answered": False,
            "context_exact": False,
            "source_valid": False,
            "invented_source": False,
        }
        truncated = False
    strict_success = bool(
        status == "completed"
        and not truncated
        and native["format_valid"]
        and native["source_valid"]
        and native["answer_correct"]
        and native["context_exact"]
    )
    return {
        "task_id": task_id,
        "family_id": task["family_id"],
        "cluster_id": task["cluster_id"],
        "condition": task["condition"],
        "status": status,
        "input_sha256": row["input_sha256"],
        "observation_sha256": observation_sha,
        "image_seen": row["image_seen"],
        "upstream_image_seen": row["upstream_image_seen"],
        "latency_seconds": latency,
        "upstream_latency_seconds": upstream_latency,
        "end_to_end_latency_seconds": end_to_end_latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "upstream_prompt_tokens": upstream_prompt_tokens,
        "upstream_completion_tokens": upstream_completion_tokens,
        "truncated": truncated,
        "invalid_output": bool(
            status == "completed" and (truncated or not native["format_valid"])
        ),
        "strict_contract_success": strict_success,
        "original_answer": gold["original_answer"],
        **native,
    }


def _missing_score(task: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "family_id": task["family_id"],
        "cluster_id": task["cluster_id"],
        "condition": task["condition"],
        "status": "missing",
        "input_sha256": None,
        "observation_sha256": None,
        "image_seen": None,
        "upstream_image_seen": None,
        "latency_seconds": None,
        "upstream_latency_seconds": None,
        "end_to_end_latency_seconds": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "upstream_prompt_tokens": None,
        "upstream_completion_tokens": None,
        "truncated": False,
        "invalid_output": False,
        "strict_contract_success": False,
        "parse_valid": False,
        "parse_error": "missing_response",
        "format_valid": False,
        "answer": None,
        "answer_correct": False,
        "answered": False,
        "context_exact": False,
        "source_valid": False,
        "invented_source": False,
        "original_answer": gold["original_answer"],
    }


def _infra_score(task: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, Any]:
    score = _missing_score(task, gold)
    score["status"] = "infra_error"
    score["parse_error"] = "infra_error"
    return score


def _load_infra_ledgers(directory: Path, known_task_ids: set[str]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for filename in ("infrastructure_errors.jsonl", "infra_attempts.jsonl"):
        path = directory / filename
        if not path.is_file():
            continue
        for index, row in enumerate(_load_jsonl(path), 1):
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or task_id not in known_task_ids:
                raise ScoringError(f"{path}:{index}: unknown task_id {task_id!r}")
            attempts.append(row)
    return attempts


def _load_arm(
    arm: str,
    run_root: Path,
    tasks: Sequence[dict[str, Any]],
    task_by_id: Mapping[str, dict[str, Any]],
    gold_by_id: Mapping[str, dict[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = run_root / "arms" / arm / "responses.jsonl"
    directory = path.parent
    infra_attempts = _load_infra_ledgers(directory, set(task_by_id))
    unresolved_infra_ids = {str(row["task_id"]) for row in infra_attempts}
    if not path.is_file():
        scores = [
            (
                _infra_score(task, gold_by_id[task["task_id"]])
                if task["task_id"] in unresolved_infra_ids
                else _missing_score(task, gold_by_id[task["task_id"]])
            )
            for task in tasks
        ]
        return scores, {
            "arm": arm,
            "file_present": False,
            "response_rows": 0,
            "missing_rows": len(tasks),
            "infra_error_rows": sum(score["status"] == "infra_error" for score in scores),
            "historical_infra_attempts": len(infra_attempts),
            "complete_for_inference": False,
        }
    rows = _load_jsonl(path)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in task_by_id:
            raise ScoringError(f"{path}:{index + 1}: unknown task_id {task_id!r}")
        if task_id in rows_by_id:
            raise ScoringError(f"{path}: duplicate response task_id {task_id}")
        rows_by_id[task_id] = row
    scores: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["task_id"]
        if task_id not in rows_by_id:
            scores.append(
                _infra_score(task, gold_by_id[task_id])
                if task_id in unresolved_infra_ids
                else _missing_score(task, gold_by_id[task_id])
            )
        else:
            scores.append(
                _validate_response_row(
                    rows_by_id[task_id],
                    arm=arm,
                    task=task,
                    gold=gold_by_id[task_id],
                    manifest=manifest,
                )
            )
    missing = sum(score["status"] == "missing" for score in scores)
    infra = sum(score["status"] == "infra_error" for score in scores)
    return scores, {
        "arm": arm,
        "file_present": True,
        "response_rows": len(rows),
        "missing_rows": missing,
        "infra_error_rows": infra,
        "historical_infra_attempts": len(infra_attempts),
        "complete_for_inference": missing == 0 and infra == 0,
    }


def _check_observation_identity(
    scores_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    run_root: Path,
    task_by_id: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, int]:
    """Bind every result to the actual shared observation and frozen input."""

    observation_path = run_root / "observations" / "observations.jsonl"
    observation_infra = _load_infra_ledgers(
        observation_path.parent, set(task_by_id)
    )
    observations: dict[str, dict[str, Any]] = {}
    if observation_path.is_file():
        for index, observation in enumerate(_load_jsonl(observation_path), 1):
            task_id = observation.get("task_id")
            if not isinstance(task_id, str) or task_id not in task_by_id:
                raise ScoringError(
                    f"{observation_path}:{index}: unknown task_id {task_id!r}"
                )
            if task_id in observations:
                raise ScoringError(f"{observation_path}: duplicate task_id {task_id}")
            task = task_by_id[task_id]
            for field in ("task_id", "family_id", "cluster_id", "condition"):
                if observation.get(field) != task.get(field):
                    raise ScoringError(f"observation {task_id}: {field} mismatch")
            if observation.get("status") != "completed":
                raise ScoringError(f"observation {task_id}: non-completed row is not frozen output")
            if observation.get("protocol_sha256") != manifest["protocol_sha256"]:
                raise ScoringError(f"observation {task_id}: protocol mismatch")
            if observation.get("model_fingerprint") != manifest["model_fingerprints"]["vlm"]:
                raise ScoringError(f"observation {task_id}: VLM fingerprint mismatch")
            if observation.get("image_sha256") != task.get("image_sha256"):
                raise ScoringError(f"observation {task_id}: image SHA mismatch")
            expected_input = input_digest(
                observation_messages(dict(task)), task.get("image_sha256")
            )
            if observation.get("input_sha256") != expected_input:
                raise ScoringError(f"observation {task_id}: input SHA mismatch")
            expected_seen = task["condition"] != "no_image"
            if observation.get("image_seen") is not expected_seen:
                raise ScoringError(f"observation {task_id}: image_seen mismatch")
            observations[task_id] = observation

    indexed = {
        arm: {str(row["task_id"]): row for row in rows}
        for arm, rows in scores_by_arm.items()
    }
    for task_id, task in task_by_id.items():
        direct = indexed["vlm_direct"][task_id]
        if direct["status"] == "completed":
            expected_direct = input_digest(
                answer_messages(dict(task), None, direct=True), task.get("image_sha256")
            )
            if direct["input_sha256"] != expected_direct:
                raise ScoringError(f"{task_id}: vlm_direct input SHA mismatch")

        completed_downstream = [
            indexed[arm][task_id]
            for arm in ARMS[1:]
            if indexed[arm][task_id]["status"] == "completed"
        ]
        if not completed_downstream:
            continue
        if task_id not in observations:
            raise ScoringError(f"{task_id}: completed downstream row lacks shared observation")
        observation = observations[task_id]
        expected_observation_sha = _digest(observation)
        expected_input = input_digest(
            answer_messages(dict(task), observation, direct=False), None
        )
        for score in completed_downstream:
            if score["observation_sha256"] != expected_observation_sha:
                raise ScoringError(
                    f"{task_id}: downstream observation SHA does not bind actual observation"
                )
            if score["input_sha256"] != expected_input:
                raise ScoringError(f"{task_id}: downstream input SHA mismatch")
    unresolved = {
        str(row["task_id"]) for row in observation_infra
    } - set(observations)
    return {
        "historical_infra_attempts": len(observation_infra),
        "unresolved_infra_tasks": len(unresolved),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarize_rows(
    arm: str, condition: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    total = len(rows)
    completed = sum(row["status"] == "completed" for row in rows)
    infra = sum(row["status"] == "infra_error" for row in rows)
    missing = sum(row["status"] == "missing" for row in rows)
    strict = sum(bool(row["strict_contract_success"]) for row in rows)
    correct = sum(bool(row["answer_correct"]) for row in rows)
    answered_rows = [row for row in rows if row["answered"]]
    if condition == "no_image":
        selective_correct = sum(
            row["answer"] == row["original_answer"] for row in answered_rows
        )
        selective_definition = "original_answer_accuracy_among_answered_no_image"
    elif condition in CONDITIONS:
        selective_correct = sum(bool(row["answer_correct"]) for row in answered_rows)
        selective_definition = "condition_gold_accuracy_among_answered_visible_image"
    else:
        selective_correct = 0
        selective_definition = "not_aggregated_across_conditions"
    selective_denominator = len(answered_rows) if condition in CONDITIONS else 0
    latencies = [
        float(row["latency_seconds"])
        for row in rows
        if row["status"] == "completed" and row["latency_seconds"] is not None
    ]
    upstream_latencies = [
        float(row["upstream_latency_seconds"])
        for row in rows
        if row["status"] == "completed"
        and row["upstream_latency_seconds"] is not None
    ]
    end_to_end_latencies = [
        float(row["end_to_end_latency_seconds"])
        for row in rows
        if row["status"] == "completed"
        and row["end_to_end_latency_seconds"] is not None
    ]
    prompt_tokens = [
        int(row["prompt_tokens"])
        for row in rows
        if row["status"] == "completed" and row["prompt_tokens"] is not None
    ]
    completion_tokens = [
        int(row["completion_tokens"])
        for row in rows
        if row["status"] == "completed" and row["completion_tokens"] is not None
    ]
    end_to_end_prompt_tokens = [
        int(row["prompt_tokens"]) + int(row["upstream_prompt_tokens"])
        for row in rows
        if row["status"] == "completed"
        and row["prompt_tokens"] is not None
        and row["upstream_prompt_tokens"] is not None
    ]
    end_to_end_completion_tokens = [
        int(row["completion_tokens"]) + int(row["upstream_completion_tokens"])
        for row in rows
        if row["status"] == "completed"
        and row["completion_tokens"] is not None
        and row["upstream_completion_tokens"] is not None
    ]
    return {
        "arm": arm,
        "condition": condition,
        "total_expected": total,
        "completed_count": completed,
        "infra_error_count": infra,
        "missing_count": missing,
        "strict_contract_success_count": strict,
        "strict_contract_success": _rate(strict, total),
        "answer_correct_count": correct,
        "answer_accuracy": _rate(correct, total),
        "answered_count": len(answered_rows),
        "answered_rate": _rate(len(answered_rows), total),
        "selective_correct_count": selective_correct,
        "selective_denominator": selective_denominator,
        "selective_accuracy": _rate(selective_correct, selective_denominator),
        "selective_accuracy_definition": selective_definition,
        "context_exact_count": sum(bool(row["context_exact"]) for row in rows),
        "context_exact": _rate(sum(bool(row["context_exact"]) for row in rows), total),
        "format_valid_count": sum(bool(row["format_valid"]) for row in rows),
        "format_valid": _rate(sum(bool(row["format_valid"]) for row in rows), total),
        "invented_source_count": sum(bool(row["invented_source"]) for row in rows),
        "invented_source": _rate(sum(bool(row["invented_source"]) for row in rows), total),
        "invalid_output_count": sum(bool(row["invalid_output"]) for row in rows),
        "invalid_output": _rate(sum(bool(row["invalid_output"]) for row in rows), total),
        "truncated_count": sum(bool(row["truncated"]) for row in rows),
        "latency_completed_count": len(latencies),
        "latency_mean_seconds": statistics.fmean(latencies) if latencies else None,
        "latency_p50_seconds": _percentile(latencies, 0.50),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "upstream_latency_mean_seconds": (
            statistics.fmean(upstream_latencies) if upstream_latencies else None
        ),
        "end_to_end_latency_mean_seconds": (
            statistics.fmean(end_to_end_latencies) if end_to_end_latencies else None
        ),
        "end_to_end_latency_p50_seconds": _percentile(end_to_end_latencies, 0.50),
        "end_to_end_latency_p95_seconds": _percentile(end_to_end_latencies, 0.95),
        "prompt_tokens_total": sum(prompt_tokens),
        "prompt_tokens_mean": statistics.fmean(prompt_tokens) if prompt_tokens else None,
        "completion_tokens_total": sum(completion_tokens),
        "completion_tokens_mean": (
            statistics.fmean(completion_tokens) if completion_tokens else None
        ),
        "end_to_end_prompt_tokens_total": sum(end_to_end_prompt_tokens),
        "end_to_end_prompt_tokens_mean": (
            statistics.fmean(end_to_end_prompt_tokens)
            if end_to_end_prompt_tokens
            else None
        ),
        "end_to_end_completion_tokens_total": sum(end_to_end_completion_tokens),
        "end_to_end_completion_tokens_mean": (
            statistics.fmean(end_to_end_completion_tokens)
            if end_to_end_completion_tokens
            else None
        ),
    }


def _family_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_family[str(row["family_id"])][str(row["condition"])] = row
    clean_correct_families = [
        family
        for family, states in by_family.items()
        if bool(states["clean"]["answer_correct"])
    ]
    retained = sum(
        bool(by_family[family]["no_image"]["context_exact"])
        for family in clean_correct_families
    )
    pair_families = [
        family
        for family, states in by_family.items()
        if "clean" in states and "swapped_image" in states
    ]
    pair_correct = sum(
        bool(by_family[family]["clean"]["answer_correct"])
        and bool(by_family[family]["swapped_image"]["answer_correct"])
        for family in pair_families
    )
    pair_strict = sum(
        bool(by_family[family]["clean"]["strict_contract_success"])
        and bool(by_family[family]["swapped_image"]["strict_contract_success"])
        for family in pair_families
    )
    return {
        "context_conditional_retention_clean_correct_denominator": len(
            clean_correct_families
        ),
        "context_conditional_retention_clean_correct_numerator": retained,
        "context_conditional_retention_clean_correct": _rate(
            retained, len(clean_correct_families)
        ),
        "context_conditional_retention_definition": (
            "no_image context_exact among families whose clean visual answer was correct"
        ),
        "family_count": len(by_family),
        "clean_swapped_pair_eligible_count": len(pair_families),
        "clean_swapped_pair_ineligible_unmatched_count": len(by_family)
        - len(pair_families),
        "clean_swapped_pair_count": len(pair_families),
        "clean_swapped_both_answer_correct_count": pair_correct,
        "clean_swapped_both_answer_correct": _rate(pair_correct, len(pair_families)),
        "clean_swapped_both_strict_count": pair_strict,
        "clean_swapped_both_strict": _rate(pair_strict, len(pair_families)),
        "pair_consistency_definition": (
            "both actual family members correct; abstention, missing, and infra_error fail"
        ),
    }


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    return float(bool(row[metric]))


def paired_cluster_bootstrap(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    condition: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ScoringError("bootstrap replicates must be positive")
    candidate = {str(row["task_id"]): row for row in candidate_rows}
    baseline = {str(row["task_id"]): row for row in baseline_rows}
    if set(candidate) != set(baseline):
        raise ScoringError("paired bootstrap requires identical task IDs")
    task_ids = [
        task_id
        for task_id, row in candidate.items()
        if condition == "all" or row["condition"] == condition
    ]
    if not task_ids:
        return {
            "paired_rows": 0,
            "cluster_count": 0,
            "delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "bootstrap_unit": "cluster_id connected component",
            "estimable": False,
            "ci_estimable": False,
            "warnings": ["No task rows exist for this condition."],
        }
    by_cluster: dict[str, list[float]] = defaultdict(list)
    all_deltas: list[float] = []
    for task_id in task_ids:
        left, right = candidate[task_id], baseline[task_id]
        if left["cluster_id"] != right["cluster_id"]:
            raise ScoringError(f"{task_id}: paired rows disagree on cluster_id")
        delta = _metric_value(left, metric) - _metric_value(right, metric)
        by_cluster[str(left["cluster_id"])].append(delta)
        all_deltas.append(delta)
    clusters = sorted(by_cluster)
    point_delta = statistics.fmean(all_deltas)
    bootstrap_warnings: list[str] = []
    if len(clusters) < 10:
        bootstrap_warnings.append(
            "Fewer than 10 independent cluster_id components; interval precision is limited."
        )
    if len(clusters) < 2:
        bootstrap_warnings.append(
            "A single independent component cannot support a meaningful bootstrap interval."
        )
        return {
            "paired_rows": len(task_ids),
            "cluster_count": len(clusters),
            "delta": point_delta,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "bootstrap_unit": "cluster_id connected component",
            "estimable": True,
            "ci_estimable": False,
            "warnings": bootstrap_warnings,
        }
    derived_seed = int.from_bytes(
        hashlib.sha256(f"{seed}|{metric}|{condition}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = random.Random(derived_seed)
    distribution: list[float] = []
    for _ in range(replicates):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [value for cluster in sampled for value in by_cluster[cluster]]
        distribution.append(statistics.fmean(values))
    return {
        "paired_rows": len(task_ids),
        "cluster_count": len(clusters),
        "delta": point_delta,
        "ci95_low": _percentile(distribution, 0.025),
        "ci95_high": _percentile(distribution, 0.975),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_unit": "cluster_id connected component",
        "estimable": True,
        "ci_estimable": True,
        "warnings": bootstrap_warnings,
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _format_rate(value: Any) -> str:
    return "NA" if value is None else f"{100 * float(value):.2f}%"


def _write_outputs(output: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        output / "metrics.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    metric_rows = list(payload.get("per_arm_condition", []))
    metric_columns = list(metric_rows[0]) if metric_rows else ["arm", "condition"]
    _atomic_text(output / "metrics.csv", _csv_text(metric_rows, metric_columns))
    paired_rows = list(payload.get("paired_effects", []))
    paired_columns = list(paired_rows[0]) if paired_rows else [
        "candidate",
        "baseline",
        "condition",
        "metric",
        "delta",
        "ci95_low",
        "ci95_high",
    ]
    _atomic_text(output / "paired_effects.csv", _csv_text(paired_rows, paired_columns))

    lines = [
        "# Visual evidence transfer scoring summary",
        "",
        f"**Status: {payload['status']}**",
        "",
    ]
    if payload.get("protocol_version"):
        lines += [f"Protocol: {payload['protocol_version']}",
                  f"Evaluation use: {payload.get('evaluation_use', 'unspecified')}", ""]
    if payload.get("revision_disclosure", {}).get("prior_test_outputs_inspected"):
        lines += ["This adapter revision follows inspection of earlier test outputs. "
                  "A repeated test run is post-test protocol revision evidence, not an unseen-test confirmation. "
                  "Earlier outputs remain preserved; no best-version output selection is permitted.", ""]
    if payload["status"] != "COMPLETE":
        lines += [
            "No complete comparative claim is authorized from this run.",
            "",
            "## Diagnostics",
            "",
        ]
        for item in payload.get("diagnostics", []):
            lines.append(f"- {item}")
    else:
        lines += [
            "All expected arms and tasks completed. Invalid, malformed, and length-truncated "
            "model outputs remain denominator failures.",
            "",
            "| Arm | Condition | Strict contract | Answer accuracy | Answered | Context exact | Format valid | Invented source | Invalid output |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in metric_rows:
            if row["condition"] == "all":
                continue
            lines.append(
                "| {arm} | {condition} | {strict} | {answer} | {answered} | "
                "{context} | {format_valid} | {invented} | {invalid} |".format(
                    arm=row["arm"],
                    condition=row["condition"],
                    strict=_format_rate(row["strict_contract_success"]),
                    answer=_format_rate(row["answer_accuracy"]),
                    answered=_format_rate(row["answered_rate"]),
                    context=_format_rate(row["context_exact"]),
                    format_valid=_format_rate(row["format_valid"]),
                    invented=_format_rate(row["invented_source"]),
                    invalid=_format_rate(row["invalid_output"]),
                )
            )
        lines += [
            "",
            "Selective accuracy uses condition gold for visible-image conditions. For no_image "
            "it is reported separately as original-answer accuracy among non-abstaining guesses; "
            "it is not an abstention-success metric.",
        ]
    _atomic_text(output / "summary.md", "\n".join(lines) + "\n")


def _write_failure_outputs(output: Path, message: str) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "UNKNOWN",
        "diagnostics": [message],
        "per_arm_condition": [],
        "paired_effects": [],
    }
    _write_outputs(output, payload)


def score_results(
    *,
    tasks_dir: Path,
    run_root: Path,
    output: Path,
    bootstrap: int = 2000,
    seed: int = 20260915,
) -> dict[str, Any]:
    if bootstrap <= 0:
        raise ScoringError("--bootstrap must be positive")
    tasks, task_by_id, gold_by_id, hashes, bundle_audit = _validate_bundle(tasks_dir)
    manifest = _validate_run_manifest(run_root, hashes)
    scores_by_arm: dict[str, list[dict[str, Any]]] = {}
    arm_diagnostics: list[dict[str, Any]] = []
    for arm in ARMS:
        scores, diagnostic = _load_arm(
            arm, run_root, tasks, task_by_id, gold_by_id, manifest
        )
        scores_by_arm[arm] = scores
        arm_diagnostics.append(diagnostic)
    observation_diagnostics = _check_observation_identity(
        scores_by_arm, run_root, task_by_id, manifest
    )

    response_rows = sum(item["response_rows"] for item in arm_diagnostics)
    incomplete = [item for item in arm_diagnostics if not item["complete_for_inference"]]
    if response_rows == 0:
        status = "UNKNOWN"
    elif incomplete:
        status = "INCOMPLETE"
    else:
        status = "COMPLETE"
    diagnostics = [
        f"{item['arm']}: file_present={item['file_present']} "
        f"rows={item['response_rows']} missing={item['missing_rows']} "
        f"infra_error={item['infra_error_rows']}"
        for item in incomplete
    ]
    diagnostics.extend(bundle_audit["warnings"])

    summaries: list[dict[str, Any]] = []
    family_metrics: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = scores_by_arm[arm]
        for condition in CONDITIONS:
            summaries.append(
                _summarize_rows(
                    arm,
                    condition,
                    [row for row in arm_rows if row["condition"] == condition],
                )
            )
        summaries.append(_summarize_rows(arm, "all", arm_rows))
        family_metrics[arm] = _family_metrics(arm_rows)

    paired: list[dict[str, Any]] = []
    if status == "COMPLETE":
        metrics = (
            "strict_contract_success",
            "answer_correct",
            "answered",
            "context_exact",
            "format_valid",
            "invented_source",
            "invalid_output",
        )
        metric_labels = {
            "answer_correct": "answer_accuracy",
            "answered": "answered_rate",
        }
        for baseline in BASELINES:
            for condition in (*CONDITIONS, "all"):
                for metric in metrics:
                    contrast_seed = int.from_bytes(
                        hashlib.sha256(
                            f"{seed}|care_opd|{baseline}".encode("utf-8")
                        ).digest()[:4],
                        "big",
                    )
                    effect = paired_cluster_bootstrap(
                        scores_by_arm["care_opd"],
                        scores_by_arm[baseline],
                        metric=metric,
                        condition=condition,
                        replicates=bootstrap,
                        seed=contrast_seed,
                    )
                    paired.append(
                        {
                            "candidate": "care_opd",
                            "baseline": baseline,
                            "condition": condition,
                            "metric": metric_labels.get(metric, metric),
                            "direction": (
                                "lower_is_better"
                                if metric in {"invented_source", "invalid_output"}
                                else "higher_is_better_or_descriptive"
                            ),
                            **effect,
                        }
                    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "protocol_sha256": manifest["protocol_sha256"],
        "protocol_version": manifest.get("protocol_version"),
        "evaluation_use": manifest.get("config", {}).get("evaluation_use"),
        "revision_disclosure": manifest.get("revision_disclosure", {}),
        "tasks_sha256": hashes["tasks_sha256"],
        "gold_sha256": hashes["gold_sha256"],
        "selection_manifest_sha256": hashes["selection_manifest_sha256"],
        "expected_arms": list(ARMS),
        "task_count": len(tasks),
        "family_count": len({task["family_id"] for task in tasks}),
        "cluster_count": len({task["cluster_id"] for task in tasks}),
        "bundle_audit": bundle_audit,
        "bootstrap": {
            "method": "paired nonparametric cluster percentile bootstrap",
            "unit": "cluster_id connected component",
            "replicates": bootstrap,
            "seed": seed,
            "training_variance": False,
        },
        "metric_definitions": {
            "strict_contract_success": (
                "completed, non-truncated, fixed schema and legal sources, correct condition "
                "answer, and exact 8-character context probe"
            ),
            "answer_accuracy": "content answer correctness even if another schema field fails",
            "invalid_output": "completed output with malformed schema or length truncation",
            "infra_policy": "infra_error and missing rows remain denominator failures",
        },
        "arm_diagnostics": arm_diagnostics,
        "observation_diagnostics": observation_diagnostics,
        "diagnostics": diagnostics,
        "per_arm_condition": summaries,
        "per_arm_family_metrics": family_metrics,
        "paired_effects": paired,
    }
    _write_outputs(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser


def _resolve_cli_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    tasks_dir = args.tasks_dir
    run_root = args.run_root
    if args.config is not None:
        config = _load_json(args.config)
        if config.get("schema_version") != "care_opd.visual_transfer.config.v1":
            raise ScoringError("config: unsupported schema_version")
        configured_tasks = Path(_require_text(config, "tasks_dir", "config"))
        configured_run = Path(_require_text(config, "run_root", "config"))
        if tasks_dir is not None and tasks_dir.resolve() != configured_tasks.resolve():
            raise ScoringError("--tasks-dir differs from frozen config")
        if run_root is not None and run_root.resolve() != configured_run.resolve():
            raise ScoringError("--run-root differs from frozen config")
        tasks_dir, run_root = configured_tasks, configured_run
    if tasks_dir is None or run_root is None:
        raise ScoringError("provide --config or both --tasks-dir and --run-root")
    output = args.output if args.output is not None else run_root / "summary"
    return tasks_dir, run_root, output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fallback_output = args.output or (
        args.run_root / "summary" if args.run_root is not None else Path("visual_transfer_summary")
    )
    try:
        tasks_dir, run_root, output = _resolve_cli_paths(args)
        # If validation fails after config resolution, diagnostics belong next
        # to that configured run rather than an unrelated working directory.
        fallback_output = output
        result = score_results(
            tasks_dir=tasks_dir,
            run_root=run_root,
            output=output,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
    except ScoringError as exc:
        _write_failure_outputs(fallback_output, str(exc))
        print(f"SCORING_UNKNOWN: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
