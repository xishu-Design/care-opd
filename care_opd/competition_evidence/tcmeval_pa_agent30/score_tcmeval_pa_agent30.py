#!/usr/bin/env python3
"""Score four frozen systems on the fixed TCMEval-PA 30-question pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


SCHEMA_VERSION = "care_opd.competition.tcmeval_pa_agent30_score.v1"
PRIMARY_SYSTEM = "shennong_sanyan_full"
DEFAULT_SYSTEMS = (
    "shennong_sanyan_full",
    "huatuogpt3_local_medical_agent",
    "glm4_local_role_model",
    "internlm3_local_role_model",
)
SYSTEM_LABELS = {
    "shennong_sanyan_full": "神农三言完整项目",
    "huatuogpt3_local_medical_agent": "HuatuoGPT-3-8B 本地医疗模型基线",
    "glm4_local_role_model": "GLM-4-9B-0414 本地中文模型基线",
    "internlm3_local_role_model": "InternLM3-8B-Instruct 本地中文模型基线",
}

FULLWIDTH_TABLE = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪ：，；。",
    "ABCDEFGHIJ:,;.",
)
STRICT_MARKER = re.compile(
    r"(?im)^\s*最终选项\s*[:：]\s*"
    r"([A-JＡ-Ｊ](?:[\s,，、/;；和及]*[A-JＡ-Ｊ])*)\s*[。.]?\s*$"
)
INLINE_FINAL_MARKER = re.compile(
    r"(?i)最终选项\s*[:：]\s*"
    r"([A-JＡ-Ｊ](?:[\s,，、/;；和及]*[A-JＡ-Ｊ])*)"
)
FALLBACK_MARKER = re.compile(
    r"(?im)^\s*(?:最终答案|答案|选择)\s*[:：]\s*"
    r"([A-JＡ-Ｊ](?:[\s,，、/;；和及]*[A-JＡ-Ｊ])*)\s*[。.]?\s*$"
)


def _care_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_evaluator() -> Any:
    return _load_module(
        _care_root() / "public_eval" / "tcmeval_pa" / "evaluate_tcmeval_pa.py",
        "care_tcmeval_pa_evaluator_for_agent30_score",
    )


def _load_common() -> Any:
    return _load_module(
        _care_root() / "competition_evidence" / "three_role_eval_common.py",
        "care_three_role_common_for_agent30_score",
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _extract_letters(fragment: str, allowed: set[str]) -> tuple[str, ...] | None:
    normalized = fragment.translate(FULLWIDTH_TABLE).upper()
    letters = re.findall(r"[A-J]", normalized)
    if not letters or len(letters) != len(set(letters)):
        return None
    if not set(letters).issubset(allowed):
        return None
    return tuple(sorted(letters))


def parse_prediction(text: str, allowed_options: Iterable[str]) -> dict[str, Any]:
    """Use a frozen marker-first parser; never scan arbitrary reasoning letters."""

    allowed = {str(value).upper() for value in allowed_options}
    strict = list(STRICT_MARKER.finditer(text))
    if strict:
        value = _extract_letters(strict[-1].group(1), allowed)
        return {
            "prediction": value,
            "parse_method": "strict_final_option_marker" if value else "invalid_marker",
            "strict_marker": value is not None,
        }
    inline = list(INLINE_FINAL_MARKER.finditer(text))
    if inline:
        value = _extract_letters(inline[-1].group(1), allowed)
        return {
            "prediction": value,
            "parse_method": "inline_final_option_marker" if value else "invalid_marker",
            "strict_marker": False,
        }
    fallback = list(FALLBACK_MARKER.finditer(text))
    if fallback:
        value = _extract_letters(fallback[-1].group(1), allowed)
        return {
            "prediction": value,
            "parse_method": "fallback_answer_marker" if value else "invalid_marker",
            "strict_marker": False,
        }
    compact = re.sub(r"[\s,，、/;；。.]+", "", text.translate(FULLWIDTH_TABLE)).upper()
    if compact and re.fullmatch(r"[A-J]+", compact):
        value = _extract_letters(compact, allowed)
        return {
            "prediction": value,
            "parse_method": "option_only_response" if value else "invalid_option_only",
            "strict_marker": False,
        }
    return {"prediction": None, "parse_method": "parse_failure", "strict_marker": False}


def _wilson(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def _exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _parse_response_specs(args: argparse.Namespace) -> dict[str, Path]:
    if args.response:
        result: dict[str, Path] = {}
        for raw in args.response:
            if "=" not in raw:
                raise ValueError("--response must use SYSTEM_ID=/path/to/responses.private.jsonl")
            system_id, path_text = raw.split("=", 1)
            system_id = system_id.strip()
            if not system_id or system_id in result:
                raise ValueError(f"invalid or duplicate response system: {system_id!r}")
            result[system_id] = Path(path_text).expanduser().resolve()
        return result
    run_root = args.run_root.expanduser().resolve()
    return {
        system_id: run_root / system_id / "responses.private.jsonl"
        for system_id in DEFAULT_SYSTEMS
    }


def _slice_rows(system_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[("question_type", row["question_type"])].append(row)
        groups[("category", row["category"])].append(row)
    result: list[dict[str, Any]] = []
    for (dimension, value), items in sorted(groups.items()):
        correct = sum(bool(item["correct"]) for item in items)
        low, high = _wilson(correct, len(items))
        result.append(
            {
                "system_id": system_id,
                "system_label": SYSTEM_LABELS.get(system_id, system_id),
                "slice_dimension": dimension,
                "slice_value": value,
                "correct": correct,
                "total": len(items),
                "accuracy": correct / len(items),
                "wilson95_low": low,
                "wilson95_high": high,
            }
        )
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = _load_evaluator()
    common = _load_common()
    tasks_path = args.tasks.expanduser().resolve()
    manifest_path = args.selection_manifest.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    for path in (tasks_path, manifest_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tasks = common.read_jsonl(tasks_path)
    if len(tasks) != 30 or len({str(row.get("task_id")) for row in tasks}) != 30:
        raise ValueError("task package must contain exactly 30 unique task IDs")
    manifest = _read_object(manifest_path)
    if manifest.get("status") != "FROZEN_BEFORE_ANY_MODEL_GENERATION":
        raise ValueError("selection manifest was not frozen before generation")
    expected_tasks_sha = manifest.get("artifacts", {}).get("tasks_sha256")
    if expected_tasks_sha != common.sha256_file(tasks_path):
        raise ValueError("task file does not match the frozen selection manifest")

    questions = evaluator.load_questions(data_path)
    question_index = {question.id: question for question in questions}
    if evaluator.sha256_file(data_path) != manifest.get("benchmark", {}).get("dataset_sha256"):
        raise ValueError("dataset does not match selection manifest")

    task_index: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        benchmark_id = str(task.get("benchmark_id") or "")
        question = question_index.get(benchmark_id)
        if question is None:
            raise ValueError(f"unknown benchmark_id in task: {benchmark_id}")
        if "gold" in json.dumps(task, ensure_ascii=False).lower():
            raise ValueError(f"task unexpectedly contains a gold-like field: {task_id}")
        if list(question.allowed_options) != task.get("allowed_options"):
            raise ValueError(f"allowed option mismatch for task {task_id}")
        task_index[task_id] = task

    response_paths = _parse_response_specs(args)
    if PRIMARY_SYSTEM not in response_paths:
        raise ValueError(f"responses must include primary system {PRIMARY_SYSTEM}")

    item_rows: list[dict[str, Any]] = []
    response_hashes: dict[str, str] = {}
    for system_id, response_path in response_paths.items():
        if not response_path.is_file():
            raise FileNotFoundError(response_path)
        responses = common.read_jsonl(response_path)
        if len(responses) != 30:
            raise ValueError(f"{system_id}: expected 30 responses, got {len(responses)}")
        seen: set[str] = set()
        for response in responses:
            task_id = str(response.get("task_id") or "")
            if task_id in seen or task_id not in task_index:
                raise ValueError(f"{system_id}: duplicate or unknown task_id {task_id!r}")
            seen.add(task_id)
            task = task_index[task_id]
            if response.get("status") != "success":
                raise ValueError(f"{system_id}: non-success response for {task_id}")
            if response.get("task_payload_sha256") != common.canonical_json_sha256(task):
                raise ValueError(f"{system_id}: task hash mismatch for {task_id}")
            text = str(response.get("display_response_private") or "").strip()
            if not text:
                raise ValueError(f"{system_id}: empty response for {task_id}")
            question = question_index[str(task["benchmark_id"])]
            parsed = parse_prediction(text, question.allowed_options)
            prediction = parsed["prediction"]
            gold = tuple(sorted(question.gold_options))
            item_rows.append(
                {
                    "system_id": system_id,
                    "system_label": SYSTEM_LABELS.get(system_id, system_id),
                    "task_id": task_id,
                    "benchmark_id": question.id,
                    "category": question.category,
                    "question_type": question.question_type,
                    "prediction": "" if prediction is None else "".join(prediction),
                    "gold": "".join(gold),
                    "correct": prediction == gold,
                    "parsed": prediction is not None,
                    "strict_marker": bool(parsed["strict_marker"]),
                    "parse_method": parsed["parse_method"],
                    "latency_seconds": float(response.get("latency_seconds") or 0.0),
                }
            )
        if seen != set(task_index):
            raise ValueError(f"{system_id}: incomplete task coverage")
        response_hashes[system_id] = common.sha256_file(response_path)

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        by_system[row["system_id"]].append(row)

    summaries: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    for system_id in response_paths:
        rows = by_system[system_id]
        correct = sum(bool(row["correct"]) for row in rows)
        parsed_count = sum(bool(row["parsed"]) for row in rows)
        strict_count = sum(bool(row["strict_marker"]) for row in rows)
        low, high = _wilson(correct, len(rows))
        latencies = [float(row["latency_seconds"]) for row in rows]
        summaries.append(
            {
                "system_id": system_id,
                "system_label": SYSTEM_LABELS.get(system_id, system_id),
                "correct": correct,
                "total": len(rows),
                "accuracy": correct / len(rows),
                "accuracy_percent": 100.0 * correct / len(rows),
                "wilson95_low": low,
                "wilson95_high": high,
                "parsed_count": parsed_count,
                "parse_rate": parsed_count / len(rows),
                "strict_marker_count": strict_count,
                "strict_marker_rate": strict_count / len(rows),
                "mean_latency_seconds": mean(latencies),
                "median_latency_seconds": median(latencies),
                "parse_methods": dict(Counter(row["parse_method"] for row in rows)),
            }
        )
        slices.extend(_slice_rows(system_id, rows))

    primary_by_task = {row["task_id"]: row for row in by_system[PRIMARY_SYSTEM]}
    paired: list[dict[str, Any]] = []
    primary_accuracy = next(
        row["accuracy"] for row in summaries if row["system_id"] == PRIMARY_SYSTEM
    )
    for system_id in response_paths:
        if system_id == PRIMARY_SYSTEM:
            continue
        other_by_task = {row["task_id"]: row for row in by_system[system_id]}
        b = sum(
            bool(primary_by_task[task_id]["correct"])
            and not bool(other_by_task[task_id]["correct"])
            for task_id in task_index
        )
        c = sum(
            not bool(primary_by_task[task_id]["correct"])
            and bool(other_by_task[task_id]["correct"])
            for task_id in task_index
        )
        both_correct = sum(
            bool(primary_by_task[task_id]["correct"])
            and bool(other_by_task[task_id]["correct"])
            for task_id in task_index
        )
        both_wrong = 30 - b - c - both_correct
        other_accuracy = next(
            row["accuracy"] for row in summaries if row["system_id"] == system_id
        )
        paired.append(
            {
                "primary_system_id": PRIMARY_SYSTEM,
                "comparison_system_id": system_id,
                "comparison_system_label": SYSTEM_LABELS.get(system_id, system_id),
                "primary_only_correct": b,
                "comparison_only_correct": c,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "discordant": b + c,
                "accuracy_difference_percentage_points": 100.0
                * (primary_accuracy - other_accuracy),
                "exact_mcnemar_p_two_sided": _exact_mcnemar_p(b, c),
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "system_summary.csv",
        [
            "system_id",
            "system_label",
            "correct",
            "total",
            "accuracy",
            "accuracy_percent",
            "wilson95_low",
            "wilson95_high",
            "parsed_count",
            "parse_rate",
            "strict_marker_count",
            "strict_marker_rate",
            "mean_latency_seconds",
            "median_latency_seconds",
            "parse_methods",
        ],
        summaries,
    )
    _write_csv(
        output_dir / "slice_summary.csv",
        list(slices[0]) if slices else [],
        slices,
    )
    _write_csv(
        output_dir / "paired_vs_shennong.csv",
        list(paired[0]) if paired else [],
        paired,
    )
    _write_csv(
        output_dir / "item_scores.private.csv",
        list(item_rows[0]) if item_rows else [],
        item_rows,
    )

    public_summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "benchmark_label": "TCMEval-PA v3 固定30题先导子集",
        "reporting_boundary": (
            "非官方328题全量分数；公开中医处方审核客观题的小样本补充实验。"
            "完整项目比较不能单独归因CARE-OPD，也不能解释为临床疗效。"
        ),
        "sample_count": 30,
        "primary_metric": "option-set exact-match accuracy",
        "systems": summaries,
        "paired_vs_primary": paired,
        "slices": slices,
        "selection_manifest_sha256": common.sha256_file(manifest_path),
        "tasks_sha256": common.sha256_file(tasks_path),
    }
    _write_json(output_dir / "summary.public.json", public_summary)
    _write_json(
        output_dir / "score_manifest.private.json",
        {
            **public_summary,
            "data_sha256": common.sha256_file(data_path),
            "response_sha256_by_system": response_hashes,
            "item_score_file": "item_scores.private.csv",
        },
    )

    lines = [
        "# TCMEval-PA v3 固定30题先导子集结果",
        "",
        "> 该结果为固定30题的小样本补充实验，不是TCMEval-PA官方328题全量成绩；"
        "用于比较神农三言完整项目与三种冻结本地公开权重模型基线。",
        "",
        "| 系统 | 正确数 | 准确率 | Wilson 95% CI | 解析率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['system_label']} | {row['correct']}/{row['total']} | "
            f"{row['accuracy_percent']:.2f}% | "
            f"{100*row['wilson95_low']:.2f}%–{100*row['wilson95_high']:.2f}% | "
            f"{100*row['parse_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 与神农三言完整项目的逐题配对比较",
            "",
            "| 对照系统 | 仅神农三言正确 | 仅对照正确 | 准确率差（百分点） | exact McNemar p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['comparison_system_label']} | {row['primary_only_correct']} | "
            f"{row['comparison_only_correct']} | "
            f"{row['accuracy_difference_percentage_points']:+.2f} | "
            f"{row['exact_mcnemar_p_two_sided']:.4f} |"
        )
    lines.extend(
        [
            "",
            "30题的统计功效有限，每题对应3.33个百分点；应同时报告正确数、区间与逐题配对，"
            "不得据此声称排行榜或全面临床领先。公开题库也不能排除基础模型预训练期接触。",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "COMPLETE",
        "sample_count": 30,
        "system_count": len(summaries),
        "summary": str(output_dir / "summary.public.json"),
        "report": str(output_dir / "REPORT.md"),
    }


def parser() -> argparse.ArgumentParser:
    root = _care_root()
    default_task_root = (
        root / "runs" / "public_eval" / "tcmeval_pa_agent30_fixed_seed20260825"
    )
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--data",
        type=Path,
        default=root / "public_eval" / "tcmeval_pa" / "data" / "TCMEval-PA.xlsx",
    )
    value.add_argument(
        "--tasks",
        type=Path,
        default=default_task_root / "tcmeval_pa_agent30.tasks.jsonl",
    )
    value.add_argument(
        "--selection-manifest",
        type=Path,
        default=default_task_root / "tcmeval_pa_agent30.selection_manifest.json",
    )
    value.add_argument(
        "--run-root",
        type=Path,
        default=default_task_root / "four_system_runs",
    )
    value.add_argument(
        "--response",
        action="append",
        default=[],
        help="repeat SYSTEM_ID=/path/to/responses.private.jsonl; defaults to four run-root systems",
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        default=default_task_root / "scored",
    )
    return value


def main() -> None:
    print(json.dumps(score(parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
