#!/usr/bin/env python3
"""Freeze a deterministic 30-question TCMEval-PA pilot for four-system inference.

The benchmark gold labels never enter the generated task file.  Scoring loads
them independently from the checksum-pinned public workbook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "care_opd.competition.tcmeval_pa_agent30.v1"
SAMPLE_SEED = "20260825"
SELECTION_ALGORITHM = "sha256_rank_within_fixed_category_type_strata_v1"
SAMPLE_LABEL = "TCMEval-PA v3 固定30题先导子集"

# These quotas are fixed before any model output.  They closely match the full
# 328-question margins (297 single / 31 multiple; 6 / 86 / 236 categories)
# while keeping every major category and both question types represented.
STRATUM_QUOTAS: dict[tuple[str, str], int] = {
    ("基础概念", "single"): 1,
    ("处方规范性", "single"): 7,
    ("处方规范性", "multiple"): 1,
    ("处方适宜性", "single"): 19,
    ("处方适宜性", "multiple"): 2,
}

BASELINE_SYSTEM_PROMPT = (
    "你是中医药处方审核场景中的医生角色助手。请基于题面和选项独立判断，"
    "给出必要且简明的专业依据；证据不足时应明确保留判断。不要虚构题面之外的"
    "患者事实、检查结果或指南条文。最后必须按照用户要求输出最终选项。"
)


def _care_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_evaluator() -> Any:
    path = _care_root() / "public_eval" / "tcmeval_pa" / "evaluate_tcmeval_pa.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("care_tcmeval_pa_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_frozen(path: Path, payload: bytes) -> None:
    """Create a lock artifact once; identical reruns are harmless."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to replace a different frozen artifact: {path}"
            )
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _rank(namespace: str, question_id: str) -> str:
    return _sha256_text(
        f"tcmeval-pa-v3|agent30|{SELECTION_ALGORITHM}|{SAMPLE_SEED}|"
        f"{namespace}|{question_id}"
    )


def _select(questions: list[Any]) -> list[Any]:
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for question in questions:
        grouped[(question.category, question.question_type)].append(question)

    selected: list[Any] = []
    for stratum, quota in STRATUM_QUOTAS.items():
        candidates = sorted(
            grouped.get(stratum, []), key=lambda item: (_rank("select", item.id), item.id)
        )
        if len(candidates) < quota:
            raise ValueError(
                f"stratum {stratum!r} has {len(candidates)} rows, requires {quota}"
            )
        selected.extend(candidates[:quota])

    if len(selected) != 30 or len({item.id for item in selected}) != 30:
        raise RuntimeError("fixed selection did not produce 30 unique questions")
    # Mix strata deterministically so model state/order cannot align with a category block.
    return sorted(selected, key=lambda item: (_rank("task-order", item.id), item.id))


def _source_text(question: Any) -> str:
    return f"题目：{question.question}\n选项：\n{question.candidate_answers}".strip()


def _user_message(source_text: str) -> str:
    return (
        "请完成以下中药处方审核题。请先给出必要且简明的判断依据，再在最后一行"
        "严格按“最终选项：X”作答；如有多个选项，按字母升序连续书写，例如"
        "“最终选项：AC”。答案可能包含一个或多个选项，不要输出题目中不存在的字母。\n\n"
        + source_text
    )


def _task(question: Any, data_sha256: str) -> dict[str, Any]:
    source = _source_text(question)
    message = _user_message(source)
    opaque = _rank("task-id", question.id)[:24]
    family = _rank("family-id", question.id)[:24]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": f"TCM30-{opaque}",
        "family_token": f"TCM-F-{family}",
        "role": "doctor",
        "variant": "clean",
        "arm": "role_aware",
        "modality": "tcm_prescription_audit_mcq",
        "benchmark": "TCMEval-PA",
        "benchmark_version": "v3",
        "benchmark_id": question.id,
        "benchmark_row_index": question.index,
        "benchmark_category": question.category,
        "benchmark_question_type": question.question_type,
        "allowed_options": list(question.allowed_options),
        "dataset_sha256": data_sha256,
        "source_text_private": source,
        "paired_case_material_sha256": _sha256_text(source),
        "shared_payload_sha256": _sha256_text(message),
        "system_prompt_private": BASELINE_SYSTEM_PROMPT,
        "user_message_private": message,
        # Deliberately no answer/gold/rule summary/explanation fields.
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = _load_evaluator()
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    questions = evaluator.load_questions(data_path)
    data_sha256 = evaluator.sha256_file(data_path)
    if data_sha256 != evaluator.EXPECTED_DATA_SHA256:
        raise ValueError("dataset checksum differs from the pinned TCMEval-PA v3 file")
    selected = _select(questions)
    tasks = [_task(question, data_sha256) for question in selected]
    task_bytes = _jsonl_bytes(tasks)
    tasks_sha256 = hashlib.sha256(task_bytes).hexdigest()

    category_counts = Counter(item.category for item in selected)
    type_counts = Counter(item.question_type for item in selected)
    stratum_counts = Counter((item.category, item.question_type) for item in selected)
    selection_rows = [
        {
            "benchmark_id": item.id,
            "benchmark_row_index": item.index,
            "category": item.category,
            "question_type": item.question_type,
            "selection_rank_sha256": _rank("select", item.id),
            "task_order_rank_sha256": _rank("task-order", item.id),
        }
        for item in selected
    ]
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_ANY_MODEL_GENERATION",
        "sample_label": SAMPLE_LABEL,
        "reporting_boundary": (
            "固定30题先导子集，非TCMEval-PA官方328题全量成绩；仅作为完整项目与"
            "三种冻结本地公开权重模型基线的中医处方审核补充证据。"
        ),
        "performance_based_selection": False,
        "benchmark": {
            "name": "TCMEval-PA",
            "version": "v3",
            "figshare_doi": evaluator.FIGSHARE_DOI,
            "upstream_commit": evaluator.TCMEVAL_REPO_COMMIT,
            "license": "CC BY 4.0",
            "dataset_sha256": data_sha256,
            "full_question_count": len(questions),
        },
        "selection": {
            "seed": SAMPLE_SEED,
            "algorithm": SELECTION_ALGORITHM,
            "sample_count": len(selected),
            "fixed_quotas": {
                f"{category}|{question_type}": quota
                for (category, question_type), quota in STRATUM_QUOTAS.items()
            },
            "category_counts": dict(sorted(category_counts.items())),
            "question_type_counts": dict(sorted(type_counts.items())),
            "stratum_counts": {
                f"{category}|{question_type}": count
                for (category, question_type), count in sorted(stratum_counts.items())
            },
            "selected_rows_in_task_order": selection_rows,
            "selected_id_digest": _canonical_hash([item.id for item in selected]),
        },
        "task_protocol": {
            "role": "doctor",
            "condition": "clean",
            "shared_user_payload_across_four_systems": True,
            "gold_labels_excluded_from_task_file": True,
            "temperature": 0,
            "top_p": 1.0,
            "seed": 42,
            "four_system_scope": (
                "神农三言完整项目对比三种冻结本地公开权重模型基线；非CARE-OPD因果消融"
            ),
        },
        "artifacts": {
            "tasks_file": "tcmeval_pa_agent30.tasks.jsonl",
            "tasks_sha256": tasks_sha256,
            "preparation_script": script_path.name,
            "preparation_logic_version": SELECTION_ALGORITHM,
        },
    }

    tasks_path = output_dir / "tcmeval_pa_agent30.tasks.jsonl"
    manifest_path = output_dir / "tcmeval_pa_agent30.selection_manifest.json"
    if args.run_root:
        run_root = args.run_root.expanduser().resolve()
        existing = list(run_root.glob("*/responses.private.jsonl")) if run_root.exists() else []
        if existing and not tasks_path.exists():
            raise RuntimeError(
                "model responses already exist but the frozen task artifact is absent; refuse to "
                "create a post-hoc selection"
            )
    _write_frozen(tasks_path, task_bytes)
    _write_frozen(manifest_path, _json_bytes(manifest))
    return {
        "status": "FROZEN_BEFORE_ANY_MODEL_GENERATION",
        "sample_label": SAMPLE_LABEL,
        "task_count": len(tasks),
        "tasks": str(tasks_path),
        "tasks_sha256": tasks_sha256,
        "selection_manifest": str(manifest_path),
        "category_counts": dict(category_counts),
        "question_type_counts": dict(type_counts),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--data",
        type=Path,
        default=_care_root() / "public_eval" / "tcmeval_pa" / "data" / "TCMEval-PA.xlsx",
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _care_root()
            / "runs"
            / "public_eval"
            / "tcmeval_pa_agent30_fixed_seed20260825"
        ),
    )
    value.add_argument(
        "--run-root",
        type=Path,
        help="optional four-system run root used to detect a post-hoc selection attempt",
    )
    return value


def main() -> None:
    print(json.dumps(prepare(parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
