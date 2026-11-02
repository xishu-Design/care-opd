#!/usr/bin/env python3
"""Formal six-checkpoint evaluator for the public CMB-Exam test set.

The primary endpoint is exact equality of parsed option sets.  This evaluator
is independent of the QZBC GeneratedResponse parser and verifier.  It always
generates all 11,200 official test items once and reports both the official
full set and the predeclared overlap-clean subset (11,196 items).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import signal
import sys
import tempfile
import time
import traceback
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import prepare_data as data_contract  # noqa: E402
from preregistration import validate_static_lock  # noqa: E402
from runtime_identity import runtime_identity  # noqa: E402

UNIFIED_METRIC_SCHEMA_PATH = HERE.parent / "UNIFIED_METRIC_SCHEMA.json"
DEFAULT_PREREGISTRATION_LOCK_PATH = HERE.parent / "PREREGISTRATION_LOCK.json"
EXPECTED_UNIFIED_METRIC_SCHEMA_SHA256 = (
    "3c901d158dc60db2f96264d9c206491e4551b4222fbd6b4aeb3946bc3ac431e5"
)


EXPECTED_QUESTIONS = 11200
EXPECTED_CLEAN_QUESTIONS = 11196
EXPECTED_TYPE_COUNTS = {"single": 10010, "multiple": 1190}
EXPECTED_GOLD_CARDINALITY_COUNTS = {"single": 10035, "multiple": 1165}
EXPECTED_EXAM_TYPE_COUNTS = {
    "医师考试": 2000,
    "护理考试": 1600,
    "药师考试": 3200,
    "医技考试": 1200,
    "专业知识考试": 1600,
    "医学考研": 1600,
}
EXPECTED_EXAM_CLASSES = 28
EXPECTED_CLASS_SIZE = 400
EXPECTED_RAW_C_TYPE_MISMATCHES = 11
EXPECTED_UNIQUE_NORMALIZED_STEMS = 11098
PROTOCOL_VERSION = "care-opd-cmb-exam-v1"
FORMAL_NUM_WORKERS = 4
FORMAL_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
MODEL_ORDER = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)
CMB_PREREGISTERED_PATHS = (
    "public_eval/UNIFIED_METRIC_SCHEMA.json",
    "public_eval/PUBLIC_GENERALIZATION_PREREGISTRATION.md",
    "public_eval/cmb_exam/prepare_data.py",
    "public_eval/cmb_exam/evaluate_cmb_exam.py",
    "public_eval/cmb_exam/run_existing_gpu4_5.sh",
    "public_eval/cmb_exam/data/controlled_robustness_v1.json",
    "public_eval/cmb_exam/data/overlap_exclusions.json",
)

SYSTEM_PROMPT = "你是一个人工智能助手。"
USER_TEMPLATE = (
    "以下是中国{exam_type}中{exam_class}考试的一道{question_type_label}，"
    "不需要做任何分析和解释，直接输出答案选项。\n"
    "{question}\n{candidate_answers}"
)


@dataclass(frozen=True)
class Question:
    index: int
    id: str
    question: str
    candidate_answers: str
    option_texts: tuple[str, ...]
    allowed_options: tuple[str, ...]
    gold_options: tuple[str, ...]
    question_type: str
    question_type_label: str
    raw_question_type: str
    gold_cardinality_type: str
    exam_type: str
    exam_class: str
    exam_subject: str
    overlap_excluded: bool
    canonical_id: str
    perturbation: str
    canonical_to_displayed: tuple[tuple[str, str], ...]

    @property
    def class_key(self) -> str:
        return f"{self.exam_type}/{self.exam_class}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in materialized
        ),
    )
    return len(materialized)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _type_name(value: str) -> str:
    mapping = {"单项选择题": "single", "多项选择题": "multiple"}
    if value not in mapping:
        raise ValueError(f"unsupported official CMB question_type: {value!r}")
    return mapping[value]


def _options(value: Any, item_id: int) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"CMB ID {item_id} option must be an object")
    nonempty = [
        (str(label).strip().upper(), str(text or "").strip())
        for label, text in value.items()
        if str(text or "").strip()
    ]
    labels = [label for label, _ in nonempty]
    if len(labels) != len(set(labels)) or not 2 <= len(labels) <= 10:
        raise ValueError(f"invalid option inventory for CMB ID {item_id}: {labels}")
    expected = [chr(ord("A") + index) for index in range(len(labels))]
    if labels != expected:
        raise ValueError(
            f"non-contiguous/insertion-ordered options for CMB ID {item_id}: {labels}"
        )
    lines = []
    texts = []
    for label, text in nonempty:
        lines.append(f"{label}. {text}")
        texts.append(text)
    return tuple(labels), "\n".join(lines), tuple(texts)


def _gold(value: Any, allowed: Sequence[str], item_id: int) -> tuple[str, ...]:
    letters = re.findall(r"[A-Z]", unicodedata.normalize("NFKC", str(value)).upper())
    if not letters or len(letters) != len(set(letters)):
        raise ValueError(f"invalid CMB gold answer for ID {item_id}: {value!r}")
    result = tuple(sorted(letters))
    if not set(result).issubset(set(allowed)):
        raise ValueError(f"gold answer outside option inventory for CMB ID {item_id}")
    return result


def load_questions(data_dir: Path) -> list[Question]:
    """Load and strictly cross-check the official question and answer files."""
    data_dir = data_dir.resolve()
    data_contract.prepare(
        data_dir,
        question_source=None,
        answer_source=None,
        overlap_report=None,
        verify_only=True,
    )
    question_path = data_dir / data_contract.QUESTION_FILENAME
    answer_path = data_dir / data_contract.ANSWER_FILENAME
    exclusion_path = data_dir / data_contract.EXCLUSION_FILENAME
    raw_questions = _load_json(question_path)
    raw_answers = _load_json(answer_path)
    exclusion = _load_json(exclusion_path)
    excluded_ids = {int(value) for value in exclusion["excluded_ids"]}
    if len(excluded_ids) != data_contract.EXPECTED_EXCLUSIONS:
        raise ValueError("invalid overlap exclusion cardinality")

    questions_by_id: dict[int, Mapping[str, Any]] = {}
    answers_by_id: dict[int, Mapping[str, Any]] = {}
    for row in raw_questions:
        item_id = int(row["id"])
        if item_id in questions_by_id:
            raise ValueError(f"duplicate question ID: {item_id}")
        questions_by_id[item_id] = row
    for row in raw_answers:
        item_id = int(row["id"])
        if item_id in answers_by_id:
            raise ValueError(f"duplicate answer ID: {item_id}")
        answers_by_id[item_id] = row
    expected_ids = set(range(1, EXPECTED_QUESTIONS + 1))
    if set(questions_by_id) != expected_ids or set(answers_by_id) != expected_ids:
        raise ValueError("CMB question/answer IDs do not each cover 1..11200")

    result = []
    raw_type_mismatches = []
    for index, item_id in enumerate(sorted(expected_ids)):
        question_row = questions_by_id[item_id]
        answer_row = answers_by_id[item_id]
        for field in ("exam_type", "exam_class", "exam_subject"):
            if str(question_row.get(field)) != str(answer_row.get(field)):
                raise ValueError(f"CMB ID {item_id} metadata mismatch in {field}")
        raw_type = str(question_row.get("question_type") or "").strip()
        official_type_label = str(answer_row.get("question_type") or "").strip()
        if raw_type != official_type_label:
            raw_type_mismatches.append((item_id, raw_type, official_type_label))
            if raw_type != "C型选择题" or official_type_label != "单项选择题":
                raise ValueError(f"unexpected CMB ID {item_id} question_type mismatch")
        allowed, candidates, option_texts = _options(question_row.get("option"), item_id)
        gold = _gold(answer_row.get("answer"), allowed, item_id)
        result.append(
            Question(
                index=index,
                id=str(item_id),
                question=str(question_row.get("question") or "").strip(),
                candidate_answers=candidates,
                option_texts=option_texts,
                allowed_options=allowed,
                gold_options=gold,
                question_type=_type_name(official_type_label),
                question_type_label=official_type_label,
                raw_question_type=raw_type,
                gold_cardinality_type="single" if len(gold) == 1 else "multiple",
                exam_type=str(answer_row["exam_type"]).strip(),
                exam_class=str(answer_row["exam_class"]).strip(),
                exam_subject=str(answer_row["exam_subject"]).strip(),
                overlap_excluded=item_id in excluded_ids,
                canonical_id=str(item_id),
                perturbation="clean",
                canonical_to_displayed=tuple((label, label) for label in allowed),
            )
        )
    if len(raw_type_mismatches) != EXPECTED_RAW_C_TYPE_MISMATCHES:
        raise ValueError(f"unexpected raw C-type mismatch count: {len(raw_type_mismatches)}")
    _validate_dataset_counts(result)
    return result


def load_evaluation_items(data_dir: Path) -> list[Question]:
    clean = load_questions(data_dir)
    clean_by_id = {question.id: question for question in clean}
    robustness = _load_json(
        data_dir.resolve() / data_contract.ROBUSTNESS_MANIFEST_FILENAME
    )
    perturbed: list[Question] = []
    for offset, entry in enumerate(robustness["entries"]):
        canonical_id = str(entry["canonical_id"])
        source = clean_by_id[canonical_id]
        if source.overlap_excluded:
            raise ValueError(f"robustness entry uses excluded ID {canonical_id}")
        mapping = {
            str(old): str(new)
            for old, new in entry["canonical_to_perturbed"].items()
        }
        if set(mapping) != set(source.allowed_options) or set(mapping.values()) != set(
            source.allowed_options
        ):
            raise ValueError(f"invalid robustness permutation for CMB ID {canonical_id}")
        displayed_text: dict[str, str] = {}
        for canonical_label, text in zip(
            source.allowed_options, source.option_texts, strict=True
        ):
            displayed_text[mapping[canonical_label]] = text
        candidate_answers = "\n".join(
            f"{label}. {displayed_text[label]}" for label in source.allowed_options
        )
        perturbed.append(
            Question(
                index=EXPECTED_QUESTIONS + offset,
                id=str(entry["perturbed_id"]),
                question=source.question,
                candidate_answers=candidate_answers,
                option_texts=tuple(displayed_text[label] for label in source.allowed_options),
                allowed_options=source.allowed_options,
                gold_options=tuple(sorted(mapping[label] for label in source.gold_options)),
                question_type=source.question_type,
                question_type_label=source.question_type_label,
                raw_question_type=source.raw_question_type,
                gold_cardinality_type=source.gold_cardinality_type,
                exam_type=source.exam_type,
                exam_class=source.exam_class,
                exam_subject=source.exam_subject,
                overlap_excluded=False,
                canonical_id=canonical_id,
                perturbation="option_permutation_v1",
                canonical_to_displayed=tuple(sorted(mapping.items())),
            )
        )
    if len(perturbed) != data_contract.EXPECTED_ROBUSTNESS_ITEMS:
        raise ValueError(f"unexpected robustness item count: {len(perturbed)}")
    all_items = clean + perturbed
    if len({item.id for item in all_items}) != len(all_items):
        raise ValueError("clean and perturbed evaluation IDs are not unique")
    return all_items


def _validate_dataset_counts(questions: Sequence[Question]) -> None:
    if len(questions) != EXPECTED_QUESTIONS or len({row.id for row in questions}) != EXPECTED_QUESTIONS:
        raise ValueError("CMB dataset cardinality/ID contract failed")
    type_counts = dict(Counter(row.question_type for row in questions))
    gold_counts = dict(Counter(row.gold_cardinality_type for row in questions))
    exam_counts = dict(Counter(row.exam_type for row in questions))
    class_counts = Counter(row.class_key for row in questions)
    clean_count = sum(not row.overlap_excluded for row in questions)
    normalized_stems = {data_contract.normalized_stem(row.question) for row in questions}
    if type_counts != EXPECTED_TYPE_COUNTS:
        raise ValueError(f"unexpected official question-type counts: {type_counts}")
    if gold_counts != EXPECTED_GOLD_CARDINALITY_COUNTS:
        raise ValueError(f"unexpected gold-cardinality counts: {gold_counts}")
    if exam_counts != EXPECTED_EXAM_TYPE_COUNTS:
        raise ValueError(f"unexpected exam_type counts: {exam_counts}")
    if len(class_counts) != EXPECTED_EXAM_CLASSES or set(class_counts.values()) != {EXPECTED_CLASS_SIZE}:
        raise ValueError(f"unexpected 28-class contract: {dict(class_counts)}")
    if clean_count != EXPECTED_CLEAN_QUESTIONS:
        raise ValueError(f"unexpected overlap-clean count: {clean_count}")
    if len(normalized_stems) != EXPECTED_UNIQUE_NORMALIZED_STEMS:
        raise ValueError(f"unexpected unique normalized stem count: {len(normalized_stems)}")


def prompt_spec() -> Mapping[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_mode": "official_answer_only_zero_shot",
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "question_type_source": "public CMB answer-file metadata (never gold cardinality)",
        "few_shot": 0,
        "chain_of_thought": False,
        "chat_template": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def prompt_messages(question: Question) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                exam_type=question.exam_type,
                exam_class=question.exam_class,
                question_type_label=question.question_type_label,
                question=question.question,
                candidate_answers=question.candidate_answers,
            ),
        },
    ]


def parse_prediction(raw_output: str, allowed: Sequence[str]) -> Mapping[str, Any]:
    normalized = unicodedata.normalize("NFKC", raw_output).strip().upper()
    all_letters = re.findall(r"[A-Z]", normalized)
    allowed_set = set(allowed)
    valid = tuple(sorted(set(letter for letter in all_letters if letter in allowed_set)))
    invalid = tuple(sorted(set(letter for letter in all_letters if letter not in allowed_set)))
    parse_ok = bool(valid) and not invalid
    strict_shape = re.fullmatch(r"[A-Z](?:\s*[、,，]?\s*[A-Z])*", normalized)
    strict_format = bool(strict_shape) and parse_ok and len(all_letters) == len(set(all_letters))
    if strict_format and tuple(all_letters) != tuple(sorted(all_letters)):
        strict_format = False
    return {
        "parsed_options": list(valid),
        "invalid_letters": list(invalid),
        "parse_ok": parse_ok,
        "strict_format": strict_format,
    }


def generation_spec(dtype: str, max_new_tokens: int, batch_size: int) -> Mapping[str, Any]:
    return {
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "batch_size_per_worker": batch_size,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "enable_thinking": False,
        "seed": 42,
    }


def analysis_spec() -> Mapping[str, Any]:
    actual_schema_sha = sha256_file(UNIFIED_METRIC_SCHEMA_PATH)
    if actual_schema_sha != EXPECTED_UNIFIED_METRIC_SCHEMA_SHA256:
        raise ValueError(
            "UNIFIED_METRIC_SCHEMA.json changed; create a new protocol version and output root"
        )
    return {
        "primary_metric": "option-set exact accuracy",
        "subsets": ["official_full", "overlap_clean"],
        "mcnemar": "two-sided exact paired item-level",
        "paired_item_bootstrap": {
            "replicates": 2000,
            "seed": 20260808,
            "interval": "percentile_95",
        },
        "paired_normalized_stem_cluster_bootstrap": {
            "replicates": 2000,
            "seed": 20260808,
            "interval": "percentile_95",
            "cluster": "QZBC-overlap-audit normalized stem",
        },
        "preservation_margin_accuracy_points": -0.01,
        "preservation_references": ["clean_sft", "gkd_full"],
        "unified_metric_schema": {
            "schema_version": "care-opd-unified-effectiveness-robustness-v1",
            "sha256": actual_schema_sha,
        },
        "controlled_robustness_v1": {
            "official_track": False,
            "n": data_contract.EXPECTED_ROBUSTNESS_ITEMS,
            "selection_seed": data_contract.ROBUSTNESS_SEED,
            "perturbation": "option permutation with label-remapped gold",
            "metrics": [
                "clean_task_success_rate",
                "perturbed_task_success_rate",
                "conditional_retention",
                "robustness_drop_pp",
                "prediction_consistency",
                "worst_perturbation_tsr",
                "clean_correct_perturbed_wrong",
                "clean_wrong_perturbed_correct",
                "regression_rate",
            ],
        },
    }


def load_preregistration_lock(path: Path) -> Mapping[str, Any]:
    identity = validate_static_lock(
        path,
        project_root=HERE.parents[1],
        required_paths=CMB_PREREGISTERED_PATHS,
        timing_key="cmb_exam_predictions_observed",
    )
    return {
        **identity,
        "preregistration_lock_sha256": identity["sha256"],
    }


def assert_suite_matches_preregistration(
    suite_path: Path, lock: Mapping[str, Any]
) -> Mapping[str, Any]:
    suite = _load_json(suite_path.resolve(strict=True))
    models = suite.get("models")
    if not isinstance(models, list):
        raise ValueError("model suite has no models")
    labels = tuple(str(item.get("label") or "") for item in models)
    if labels != MODEL_ORDER:
        raise ValueError(f"model suite order changed: {labels}")
    fingerprint = canonical_hash(
        [
            (str(item.get("label") or ""), str(item.get("model_fingerprint") or ""))
            for item in models
        ]
    )
    if suite.get("suite_fingerprint") != fingerprint:
        raise ValueError("model suite fingerprint does not close")
    if lock.get("model_suite_fingerprint") != fingerprint:
        raise ValueError("model suite differs from preregistration lock")
    return suite


def locked_analysis_spec(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        **analysis_spec(),
        "preregistration_lock": {
            "schema_version": lock.get("schema_version"),
            "preregistration_lock_sha256": lock[
                "preregistration_lock_sha256"
            ],
        },
    }


def dataset_fingerprint(data_dir: Path) -> str:
    manifest = _load_json(data_dir.resolve() / data_contract.DATA_MANIFEST_FILENAME)
    value = str(manifest.get("dataset_fingerprint") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid CMB data manifest fingerprint")
    return value


def _weight_files(model: Path) -> list[Path]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model / index_name
        if index_path.is_file():
            payload = _load_json(index_path)
            names = sorted(set(payload.get("weight_map", {}).values()))
            files = [model / name for name in names]
            if not names or not all(path.is_file() for path in files):
                raise ValueError(f"invalid or incomplete weight index: {index_path}")
            return [index_path, *files]
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model / name
        if candidate.is_file():
            return [candidate]
    raise ValueError(f"no Hugging Face weight file or index found in {model}")


def validate_model_directory(path: Path, *, require_export_manifest: bool) -> list[Path]:
    path = path.resolve()
    required = [path / "config.json", path / "tokenizer_config.json"]
    if require_export_manifest:
        required.append(path / "export_manifest.json")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ValueError(f"model is not READY; missing {missing}")
    identity = list(required)
    for pattern in (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "chat_template*",
        "special_tokens_map.json",
        "added_tokens.json",
        "processor_config.json",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        identity.extend(sorted(path.glob(pattern)))
    identity.extend(_weight_files(path))
    return sorted(set(identity), key=lambda item: str(item.relative_to(path)))


def fingerprint_model(path: Path, *, require_export_manifest: bool) -> Mapping[str, Any]:
    path = path.resolve()
    files = validate_model_directory(path, require_export_manifest=require_export_manifest)
    inventory = []
    print(f"[fingerprint] model={path} files={len(files)}", flush=True)
    for file_index, file_path in enumerate(files, 1):
        stat = file_path.stat()
        relative_name = file_path.relative_to(path).as_posix()
        print(
            f"[fingerprint] file={file_index}/{len(files)} status=hashing "
            f"name={relative_name} bytes={stat.st_size}",
            flush=True,
        )
        file_sha256 = sha256_file(file_path)
        print(
            f"[fingerprint] file={file_index}/{len(files)} status=complete "
            f"name={relative_name} sha256={file_sha256}",
            flush=True,
        )
        inventory.append(
            {
                "name": relative_name,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256,
            }
        )
    content = [{key: row[key] for key in ("name", "bytes", "sha256")} for row in inventory]
    return {
        "canonical_path": str(path),
        "require_export_manifest": require_export_manifest,
        "model_fingerprint": canonical_hash(content),
        "inventory": inventory,
    }


def validate_frozen_model(entry: Mapping[str, Any]) -> None:
    path = Path(str(entry["canonical_path"])).resolve()
    if str(path) != str(entry["canonical_path"]):
        raise ValueError(f"model path is no longer canonical: {entry['canonical_path']}")
    for item in entry.get("inventory", []):
        file_path = path / str(item["name"])
        if not file_path.is_file():
            raise ValueError(f"frozen model file disappeared: {file_path}")
        stat = file_path.stat()
        if stat.st_size != int(item["bytes"]) or stat.st_mtime_ns != int(item["mtime_ns"]):
            raise ValueError(f"frozen model artifact changed: {file_path}")


def freeze_suite(suite_path: Path, model_specs: Sequence[str]) -> Mapping[str, Any]:
    requested: list[tuple[str, Path]] = []
    for spec in model_specs:
        if "=" not in spec:
            raise ValueError(f"model spec must be LABEL=PATH: {spec!r}")
        label, raw_path = spec.split("=", 1)
        if not re.fullmatch(r"[a-z0-9_]+", label):
            raise ValueError(f"invalid model label: {label!r}")
        requested.append((label, Path(raw_path).resolve()))
    labels = tuple(label for label, _ in requested)
    if labels != MODEL_ORDER:
        raise ValueError(f"formal CMB suite order must be {MODEL_ORDER}, got {labels}")
    if len({path for _, path in requested}) != len(requested):
        raise ValueError("two CMB suite labels resolve to the same model path")
    if suite_path.is_file():
        existing = _load_json(suite_path)
        old = [(item["label"], Path(item["canonical_path"])) for item in existing["models"]]
        if old != requested:
            raise ValueError(
                f"frozen suite {suite_path} differs from requested models; use a new output root"
            )
        for entry in existing["models"]:
            validate_frozen_model(entry)
        print(f"[suite] reuse frozen suite {suite_path}", flush=True)
        return existing
    entries = []
    for label, path in requested:
        entry = dict(fingerprint_model(path, require_export_manifest=label != "frozen"))
        entry["label"] = label
        entries.append(entry)
    payload = {
        "schema_version": "care-opd-cmb-model-suite-v1",
        "created_at_utc": utc_now(),
        "models": entries,
        "suite_fingerprint": canonical_hash(
            [(item["label"], item["model_fingerprint"]) for item in entries]
        ),
    }
    atomic_json(suite_path, payload)
    return payload


def _load_suite_entry(suite_path: Path, label: str) -> Mapping[str, Any]:
    suite = _load_json(suite_path)
    matches = [entry for entry in suite["models"] if entry["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"model {label!r} not found exactly once in {suite_path}")
    validate_frozen_model(matches[0])
    return matches[0]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _recover_shard(
    path: Path, expected: Mapping[int, str], run_fingerprint: str
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw_lines = path.read_bytes().splitlines()
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    corrupted_last = False
    for line_index, line in enumerate(raw_lines):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if line_index != len(raw_lines) - 1:
                raise ValueError(f"corrupt non-terminal line in {path}")
            corrupted_last = True
            break
        index = int(row.get("index", -1))
        if expected.get(index) != str(row.get("id")) or row.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"stale or unexpected row in {path}: index={index}")
        if index in seen:
            raise ValueError(f"duplicate index {index} in {path}")
        seen.add(index)
        rows.append(row)
    if corrupted_last:
        atomic_jsonl(path, rows)
    return rows


def _worker(
    worker_id: int,
    device: str,
    questions: Sequence[Question],
    shard_path: Path,
    completion_path: Path,
    error_path: Path,
    model_entry: Mapping[str, Any],
    dtype: str,
    max_new_tokens: int,
    batch_size: int,
    run_fingerprint: str,
    fingerprints: Mapping[str, str],
) -> None:
    try:
        project_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(project_root / "src"))
        from care_opd.models import ModelArguments, load_causal_lm, load_tokenizer
        import torch

        expected = {question.index: question.id for question in questions}
        existing = _recover_shard(shard_path, expected, run_fingerprint)
        completed = {int(row["index"]) for row in existing}
        remaining = [question for question in questions if question.index not in completed]
        if remaining:
            model_path = str(model_entry["canonical_path"])
            print(
                f"[worker] worker={worker_id} device={device} status=loading_model "
                f"model={model_path} remaining={len(remaining)} batch_size={batch_size}",
                flush=True,
            )
            model_args = ModelArguments(
                student_model_name_or_path=model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
            tokenizer = load_tokenizer(model_args)
            if tokenizer.pad_token_id is None:
                raise RuntimeError("tokenizer has no pad_token_id; refusing a batching fallback")
            tokenizer.padding_side = "left"
            model = load_causal_lm(model_path, model_args, trainable=False)
            target_device = torch.device(device)
            model.to(target_device)
            model.eval()
            if hasattr(model, "config"):
                model.config.use_cache = True
            print(
                f"[worker] worker={worker_id} device={device} status=model_loaded",
                flush=True,
            )
            for start in range(0, len(remaining), batch_size):
                batch = remaining[start : start + batch_size]
                messages_batch = [prompt_messages(question) for question in batch]
                rendered = []
                for messages in messages_batch:
                    try:
                        rendered.append(
                            tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
                        )
                    except TypeError as exc:
                        raise RuntimeError(
                            "tokenizer chat template does not accept enable_thinking=False; "
                            "refusing a protocol-changing fallback"
                        ) from exc
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                encoded = {key: value.to(target_device) for key, value in encoded.items()}
                torch.manual_seed(42 + batch[0].index)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                batch_latency = time.perf_counter() - started
                prompt_width = encoded["input_ids"].shape[1]
                for question, messages, sequence in zip(
                    batch, messages_batch, generated, strict=True
                ):
                    raw_output = tokenizer.decode(
                        sequence[prompt_width:], skip_special_tokens=True
                    ).strip()
                    parsed = parse_prediction(raw_output, question.allowed_options)
                    canonical_to_displayed = dict(question.canonical_to_displayed)
                    displayed_to_canonical = {
                        displayed: canonical
                        for canonical, displayed in canonical_to_displayed.items()
                    }
                    canonical_parsed = sorted(
                        displayed_to_canonical[label]
                        for label in parsed["parsed_options"]
                    )
                    canonical_gold = sorted(
                        displayed_to_canonical[label] for label in question.gold_options
                    )
                    exact = bool(
                        parsed["parse_ok"]
                        and tuple(parsed["parsed_options"]) == question.gold_options
                    )
                    row = {
                        "index": question.index,
                        "id": question.id,
                        "question": question.question,
                        "candidate_answers": question.candidate_answers,
                        "option_texts": list(question.option_texts),
                        "allowed_options": list(question.allowed_options),
                        "gold_options": list(question.gold_options),
                        "question_type": question.question_type,
                        "question_type_label": question.question_type_label,
                        "raw_question_type": question.raw_question_type,
                        "gold_cardinality_type": question.gold_cardinality_type,
                        "exam_type": question.exam_type,
                        "exam_class": question.exam_class,
                        "exam_subject": question.exam_subject,
                        "class_key": question.class_key,
                        "normalized_stem_sha256": hashlib.sha256(
                            data_contract.normalized_stem(question.question).encode("utf-8")
                        ).hexdigest(),
                        "overlap_excluded": question.overlap_excluded,
                        "canonical_id": question.canonical_id,
                        "perturbation": question.perturbation,
                        "canonical_to_displayed": canonical_to_displayed,
                        "canonical_gold_options": canonical_gold,
                        "messages": messages,
                        "raw_output": raw_output,
                        **parsed,
                        "canonical_parsed_options": canonical_parsed,
                        "option_set_exact": exact,
                        "strict_exact": bool(exact and parsed["strict_format"]),
                        "batch_latency_seconds": batch_latency,
                        "batch_size_actual": len(batch),
                        "worker_id": worker_id,
                        "device": device,
                        "run_fingerprint": run_fingerprint,
                        **fingerprints,
                    }
                    _append_jsonl(shard_path, row)
                done = len(completed) + min(start + len(batch), len(remaining))
                if start == 0 or done % 100 <= len(batch) or done == len(questions):
                    print(
                        f"[progress] worker={worker_id} device={device} "
                        f"step={done}/{len(questions)} last_id={batch[-1].id}",
                        flush=True,
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows = _recover_shard(shard_path, expected, run_fingerprint)
        if len(rows) != len(questions):
            raise RuntimeError(f"worker {worker_id} produced {len(rows)}/{len(questions)} rows")
        atomic_json(
            completion_path,
            {
                "status": "complete",
                "worker_id": worker_id,
                "rows": len(rows),
                "run_fingerprint": run_fingerprint,
                "completed_at_utc": utc_now(),
            },
        )
        error_path.unlink(missing_ok=True)
    except BaseException as exc:
        atomic_json(
            error_path,
            {
                "worker_id": worker_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _wilson(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    total = len(rows)
    exact = sum(bool(row["option_set_exact"]) for row in rows)
    strict_exact = sum(bool(row["strict_exact"]) for row in rows)
    strict_format = sum(bool(row["strict_format"]) for row in rows)
    parse_failures = sum(not bool(row["parse_ok"]) for row in rows)
    lower, upper = _wilson(exact, total) if total else (None, None)
    return {
        "n": total,
        "option_set_exact": exact,
        "option_set_accuracy": exact / total if total else None,
        "option_set_accuracy_wilson95_lower": lower,
        "option_set_accuracy_wilson95_upper": upper,
        "strict_exact": strict_exact,
        "strict_exact_accuracy": strict_exact / total if total else None,
        "strict_format": strict_format,
        "strict_format_rate": strict_format / total if total else None,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / total if total else None,
    }


def _slice_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    types = ("single", "multiple")
    exam_types = sorted({str(row["exam_type"]) for row in rows})
    class_keys = sorted({str(row["class_key"]) for row in rows})
    subjects = sorted({str(row["exam_subject"]) for row in rows})
    return {
        "overall": _metric_block(rows),
        "by_question_type": {
            value: _metric_block([row for row in rows if row["question_type"] == value])
            for value in types
        },
        "by_exam_type": {
            value: _metric_block([row for row in rows if row["exam_type"] == value])
            for value in exam_types
        },
        "by_exam_class": {
            value: _metric_block([row for row in rows if row["class_key"] == value])
            for value in class_keys
        },
        "by_exam_subject": {
            value: _metric_block([row for row in rows if row["exam_subject"] == value])
            for value in subjects
        },
    }


def _robustness_metric(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Mapping[str, Any]:
    total = len(pairs)
    clean_correct = sum(bool(clean["option_set_exact"]) for clean, _ in pairs)
    perturbed_correct = sum(bool(perturbed["option_set_exact"]) for _, perturbed in pairs)
    retained = sum(
        bool(clean["option_set_exact"]) and bool(perturbed["option_set_exact"])
        for clean, perturbed in pairs
    )
    clean_correct_perturbed_wrong = sum(
        bool(clean["option_set_exact"]) and not bool(perturbed["option_set_exact"])
        for clean, perturbed in pairs
    )
    clean_wrong_perturbed_correct = sum(
        not bool(clean["option_set_exact"]) and bool(perturbed["option_set_exact"])
        for clean, perturbed in pairs
    )
    consistent = sum(
        bool(clean["parse_ok"])
        and bool(perturbed["parse_ok"])
        and tuple(clean["canonical_parsed_options"])
        == tuple(perturbed["canonical_parsed_options"])
        for clean, perturbed in pairs
    )
    clean_rate = clean_correct / total if total else None
    perturbed_rate = perturbed_correct / total if total else None
    retention = retained / clean_correct if clean_correct else None
    return {
        "n": total,
        "paired_units": total,
        "clean_correct": clean_correct,
        "perturbed_correct": perturbed_correct,
        "clean_task_success_rate": clean_rate,
        "perturbed_task_success_rate": perturbed_rate,
        "conditional_retention": retention,
        "clean_correct_perturbed_wrong": clean_correct_perturbed_wrong,
        "clean_wrong_perturbed_correct": clean_wrong_perturbed_correct,
        "regression_rate": 1.0 - retention if retention is not None else None,
        "robustness_drop_pp": 100.0 * (clean_rate - perturbed_rate) if total else None,
        "prediction_consistency": consistent / total if total else None,
        "worst_perturbation_tsr": perturbed_rate,
    }


def _robustness_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    clean_by_id = {
        str(row["canonical_id"]): row
        for row in rows
        if row["perturbation"] == "clean"
    }
    perturbed_by_id = {
        str(row["canonical_id"]): row
        for row in rows
        if row["perturbation"] == "option_permutation_v1"
    }
    if len(perturbed_by_id) != data_contract.EXPECTED_ROBUSTNESS_ITEMS:
        raise ValueError("controlled robustness predictions are incomplete")
    pairs = [(clean_by_id[item], perturbed_by_id[item]) for item in sorted(perturbed_by_id)]
    exam_types = sorted({str(clean["exam_type"]) for clean, _ in pairs})
    class_keys = sorted({str(clean["class_key"]) for clean, _ in pairs})
    return {
        "official_track": False,
        "track": "controlled_robustness_v1",
        "perturbation": "deterministic option permutation with gold-label remapping",
        "overall": _robustness_metric(pairs),
        "confidence_intervals_95": _robustness_cluster_bootstrap(pairs),
        "by_exam_type": {
            value: _robustness_metric(
                [(clean, perturbed) for clean, perturbed in pairs if clean["exam_type"] == value]
            )
            for value in exam_types
        },
        "by_exam_class": {
            value: _robustness_metric(
                [(clean, perturbed) for clean, perturbed in pairs if clean["class_key"] == value]
            )
            for value in class_keys
        },
    }


def _robustness_cluster_bootstrap(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> Mapping[str, Any]:
    """Within-model CIs with duplicate normalized stems resampled together."""

    spec = analysis_spec()["paired_normalized_stem_cluster_bootstrap"]
    replicates = int(spec["replicates"])
    seed = int(spec["seed"])
    # n, clean-correct, perturbed-correct, both-correct, regressed
    clusters: dict[str, list[int]] = {}
    for clean, perturbed in pairs:
        key = str(clean["normalized_stem_sha256"])
        counts = clusters.setdefault(key, [0, 0, 0, 0, 0])
        clean_ok = bool(clean["option_set_exact"])
        perturbed_ok = bool(perturbed["option_set_exact"])
        counts[0] += 1
        counts[1] += int(clean_ok)
        counts[2] += int(perturbed_ok)
        counts[3] += int(clean_ok and perturbed_ok)
        counts[4] += int(clean_ok and not perturbed_ok)
    cluster_ids = sorted(clusters)
    observed = [sum(values[index] for values in clusters.values()) for index in range(5)]

    def values(counts: Sequence[int]) -> Mapping[str, float | None]:
        n, clean, perturbed, both, regressed = counts
        return {
            "clean_task_success_rate": clean / n if n else None,
            "perturbed_task_success_rate": perturbed / n if n else None,
            "conditional_retention": both / clean if clean else None,
            "regression_rate": regressed / clean if clean else None,
        }

    point = values(observed)
    draws: dict[str, list[float]] = {key: [] for key in point}
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled = [0, 0, 0, 0, 0]
        for _cluster in cluster_ids:
            chosen = clusters[rng.choice(cluster_ids)]
            for index, value in enumerate(chosen):
                sampled[index] += value
        for key, value in values(sampled).items():
            if value is not None:
                draws[key].append(float(value))

    def percentile(values_: Sequence[float], probability: float) -> float | None:
        if not values_:
            return None
        ordered = sorted(values_)
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    numerators_denominators = {
        "clean_task_success_rate": (observed[1], observed[0]),
        "perturbed_task_success_rate": (observed[2], observed[0]),
        "conditional_retention": (observed[3], observed[1]),
        "regression_rate": (observed[4], observed[1]),
    }
    return {
        "method": "normalized-stem cluster percentile bootstrap",
        "cluster_count": len(cluster_ids),
        "replicates": replicates,
        "seed": seed,
        "metrics": {
            key: {
                "numerator": numerators_denominators[key][0],
                "denominator": numerators_denominators[key][1],
                "estimate": estimate,
                "ci95_lower": percentile(draws[key], 0.025),
                "ci95_upper": percentile(draws[key], 0.975),
            }
            for key, estimate in point.items()
        },
    }


def _robustness_pair_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    clean_by_id = {
        str(row["canonical_id"]): row
        for row in rows
        if row["perturbation"] == "clean"
    }
    perturbed_by_id = {
        str(row["canonical_id"]): row
        for row in rows
        if row["perturbation"] == "option_permutation_v1"
    }
    result = []
    for canonical_id in sorted(perturbed_by_id, key=int):
        clean = clean_by_id[canonical_id]
        perturbed = perturbed_by_id[canonical_id]
        clean_success = bool(clean["option_set_exact"])
        perturbed_success = bool(perturbed["option_set_exact"])
        consistent = bool(clean["parse_ok"]) and bool(perturbed["parse_ok"]) and (
            tuple(clean["canonical_parsed_options"])
            == tuple(perturbed["canonical_parsed_options"])
        )
        result.append(
            {
                "canonical_id": canonical_id,
                "perturbed_id": perturbed["id"],
                "exam_type": clean["exam_type"],
                "exam_class": clean["exam_class"],
                "class_key": clean["class_key"],
                "perturbation": "option_permutation_v1",
                "canonical_to_displayed": perturbed["canonical_to_displayed"],
                "canonical_gold_options": clean["canonical_gold_options"],
                "clean_canonical_prediction": clean["canonical_parsed_options"],
                "perturbed_canonical_prediction": perturbed[
                    "canonical_parsed_options"
                ],
                "clean_task_success": clean_success,
                "perturbed_task_success": perturbed_success,
                "regression": clean_success and not perturbed_success,
                "improvement": not clean_success and perturbed_success,
                "prediction_consistent": consistent,
                "clean_raw_output": clean["raw_output"],
                "perturbed_raw_output": perturbed["raw_output"],
                "run_fingerprint": clean["run_fingerprint"],
                "dataset_fingerprint": clean["dataset_fingerprint"],
                "model_fingerprint": clean["model_fingerprint"],
            }
        )
    if len(result) != data_contract.EXPECTED_ROBUSTNESS_ITEMS:
        raise ValueError("controlled robustness pair output is incomplete")
    return result


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    full = [row for row in rows if row["perturbation"] == "clean"]
    clean = [row for row in full if not bool(row["overlap_excluded"])]
    if len(full) != EXPECTED_QUESTIONS or len(clean) != EXPECTED_CLEAN_QUESTIONS:
        raise ValueError(f"invalid full/clean result cardinality: {len(full)}/{len(clean)}")
    return {
        "official_full": _slice_summary(full),
        "overlap_clean": _slice_summary(clean),
        "overlap_excluded": _metric_block(
            [row for row in full if bool(row["overlap_excluded"])]
        ),
        "controlled_robustness_v1": _robustness_summary(rows),
    }


class EvaluationInterrupted(BaseException):
    pass


def _completed_valid(
    output_dir: Path,
    questions: Sequence[Question],
    run_manifest: Mapping[str, Any],
) -> bool:
    sentinel = output_dir / "COMPLETE.json"
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.jsonl"
    official_path = output_dir / "official_predictions.jsonl"
    robustness_path = output_dir / "controlled_robustness_predictions.jsonl"
    pairs_path = output_dir / "controlled_robustness_pairs.jsonl"
    if not all(
        path.is_file()
        for path in (
            sentinel,
            summary_path,
            predictions_path,
            official_path,
            robustness_path,
            pairs_path,
        )
    ):
        return False
    try:
        completion = _load_json(sentinel)
        summary = _load_json(summary_path)
        rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
        official_rows = [
            json.loads(line)
            for line in official_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        robustness_rows = [
            json.loads(line)
            for line in robustness_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pair_rows = [
            json.loads(line)
            for line in pairs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    expected_ids = {question.id for question in questions}
    actual_ids = [str(row.get("id")) for row in rows]
    fingerprints = run_manifest["fingerprints"]
    expected_official = [row for row in rows if row["perturbation"] == "clean"]
    expected_robustness = [row for row in rows if row["perturbation"] != "clean"]
    try:
        expected_pairs = _robustness_pair_rows(rows)
        recomputed_summary = summarize_rows(rows)
    except (KeyError, TypeError, ValueError):
        return False
    artifact_paths = {
        "predictions.jsonl": predictions_path,
        "official_predictions.jsonl": official_path,
        "controlled_robustness_predictions.jsonl": robustness_path,
        "controlled_robustness_pairs.jsonl": pairs_path,
        "summary.json": summary_path,
        "run_manifest.json": output_dir / "run_manifest.json",
    }
    artifact_hashes = completion.get("artifact_sha256")
    hashes_valid = isinstance(artifact_hashes, Mapping) and all(
        artifact_hashes.get(name) == sha256_file(path)
        for name, path in artifact_paths.items()
    )
    return bool(
        completion.get("status") == "complete"
        and completion.get("run_fingerprint") == run_manifest["run_fingerprint"]
        and summary.get("run_fingerprint") == run_manifest["run_fingerprint"]
        and summary.get("fingerprints") == fingerprints
        and len(rows) == len(questions)
        and len(set(actual_ids)) == len(questions)
        and set(actual_ids) == expected_ids
        and len(official_rows) == EXPECTED_QUESTIONS
        and len(robustness_rows) == data_contract.EXPECTED_ROBUSTNESS_ITEMS
        and len(pair_rows) == data_contract.EXPECTED_ROBUSTNESS_ITEMS
        and official_rows == expected_official
        and robustness_rows == expected_robustness
        and pair_rows == expected_pairs
        and all(summary.get(key) == value for key, value in recomputed_summary.items())
        and hashes_valid
        and all(
            all(row.get(key) == value for key, value in fingerprints.items())
            for row in rows
        )
    )


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    questions = load_evaluation_items(args.data_dir)
    model_entry = _load_suite_entry(args.suite, args.model_label)
    data_manifest = _load_json(
        args.data_dir.resolve() / data_contract.DATA_MANIFEST_FILENAME
    )
    p_spec = prompt_spec()
    g_spec = generation_spec(args.dtype, args.max_new_tokens, args.batch_size)
    preregistration_lock = load_preregistration_lock(args.preregistration_lock)
    assert_suite_matches_preregistration(args.suite, preregistration_lock)
    a_spec = locked_analysis_spec(preregistration_lock)
    runtime = runtime_identity()
    fingerprints = {
        "dataset_fingerprint": dataset_fingerprint(args.data_dir),
        "robustness_set_fingerprint": data_manifest["controlled_robustness_v1"][
            "robustness_set_fingerprint"
        ],
        "prompt_fingerprint": canonical_hash(p_spec),
        "generation_fingerprint": canonical_hash(g_spec),
        "analysis_fingerprint": canonical_hash(a_spec),
        "unified_metric_schema_fingerprint": EXPECTED_UNIFIED_METRIC_SCHEMA_SHA256,
        "preregistration_lock_sha256": preregistration_lock[
            "preregistration_lock_sha256"
        ],
        "model_fingerprint": str(model_entry["model_fingerprint"]),
    }
    code_fingerprint = sha256_file(Path(__file__).resolve())
    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "model_label": args.model_label,
        "model_path": model_entry["canonical_path"],
        "prompt_mode": "official_answer_only_zero_shot",
        "num_workers": args.num_workers,
        "devices": list(args.devices),
        "fingerprints": fingerprints,
        "evaluator_sha256": code_fingerprint,
        "preregistration_lock_sha256": preregistration_lock[
            "preregistration_lock_sha256"
        ],
        "runtime_identity": runtime,
    }
    run_fingerprint = canonical_hash(contract)
    output_dir = args.output_root.resolve() / "official" / args.model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run_manifest.json"
    if run_manifest_path.is_file():
        existing = _load_json(run_manifest_path)
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"run contract changed for {output_dir}; use a new output root instead of mixing results"
            )
        run_manifest = existing
    else:
        exclusion = _load_json(args.data_dir.resolve() / data_contract.EXCLUSION_FILENAME)
        run_manifest = {
            **contract,
            "run_fingerprint": run_fingerprint,
            "created_at_utc": utc_now(),
            "dataset": {
                "data_dir": str(args.data_dir.resolve()),
                "dataset_fingerprint": fingerprints["dataset_fingerprint"],
                "questions": EXPECTED_QUESTIONS,
                "overlap_clean_questions": EXPECTED_CLEAN_QUESTIONS,
                "controlled_robustness_questions": data_contract.EXPECTED_ROBUSTNESS_ITEMS,
                "total_inference_items": len(questions),
                "question_sha256": data_contract.EXPECTED_QUESTION_SHA256,
                "answer_sha256": data_contract.EXPECTED_ANSWER_SHA256,
                "overlap_report_sha256": data_contract.EXPECTED_OVERLAP_REPORT_SHA256,
                "excluded_ids": exclusion["excluded_ids"],
                "controlled_robustness_v1": data_manifest["controlled_robustness_v1"],
                "source": data_manifest["source"],
            },
            "prompt": p_spec,
            "generation": g_spec,
            "analysis": a_spec,
            "preregistration_lock": preregistration_lock,
            "model": model_entry,
        }
        atomic_json(run_manifest_path, run_manifest)
    if _completed_valid(output_dir, questions, run_manifest):
        print(f"[skip] validated complete result: {output_dir}", flush=True)
        return _load_json(output_dir / "summary.json")

    (output_dir / "COMPLETE.json").unlink(missing_ok=True)
    shard_root = output_dir / "shards"
    shard_root.mkdir(exist_ok=True)
    partitions = [questions[index :: args.num_workers] for index in range(args.num_workers)]
    context = mp.get_context("spawn")
    processes = []
    for worker_id, (device, partition) in enumerate(
        zip(args.devices, partitions, strict=True)
    ):
        error_path = shard_root / f"worker_{worker_id:02d}.error.json"
        error_path.unlink(missing_ok=True)
        process = context.Process(
            target=_worker,
            args=(
                worker_id,
                device,
                partition,
                shard_root / f"worker_{worker_id:02d}.jsonl",
                shard_root / f"worker_{worker_id:02d}.complete.json",
                error_path,
                model_entry,
                args.dtype,
                args.max_new_tokens,
                args.batch_size,
                run_fingerprint,
                fingerprints,
            ),
            name=f"cmb-exam-{args.model_label}-{worker_id}",
        )
        processes.append(process)

    old_handlers: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise EvaluationInterrupted(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupt)
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            for process in processes:
                process.join(timeout=0.5)
        failures = [process for process in processes if process.exitcode != 0]
        if failures:
            diagnostics = []
            for process in failures:
                worker_id = processes.index(process)
                error_path = shard_root / f"worker_{worker_id:02d}.error.json"
                diagnostics.append(
                    error_path.read_text(encoding="utf-8")
                    if error_path.is_file()
                    else f"{process.name}: exitcode={process.exitcode}"
                )
            raise RuntimeError("worker failure(s): " + " | ".join(diagnostics))
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=30)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    merged: list[dict[str, Any]] = []
    for worker_id, partition in enumerate(partitions):
        expected = {question.index: question.id for question in partition}
        completion = shard_root / f"worker_{worker_id:02d}.complete.json"
        if not completion.is_file():
            raise RuntimeError(f"missing worker completion sentinel: {completion}")
        merged.extend(
            _recover_shard(
                shard_root / f"worker_{worker_id:02d}.jsonl",
                expected,
                run_fingerprint,
            )
        )
    merged.sort(key=lambda row: int(row["index"]))
    ids = [str(row["id"]) for row in merged]
    if len(merged) != len(questions) or len(set(ids)) != len(questions):
        raise RuntimeError("merged predictions are incomplete or contain duplicate IDs")
    atomic_jsonl(output_dir / "predictions.jsonl", merged)
    atomic_jsonl(
        output_dir / "official_predictions.jsonl",
        [row for row in merged if row["perturbation"] == "clean"],
    )
    atomic_jsonl(
        output_dir / "controlled_robustness_predictions.jsonl",
        [row for row in merged if row["perturbation"] != "clean"],
    )
    atomic_jsonl(
        output_dir / "controlled_robustness_pairs.jsonl",
        _robustness_pair_rows(merged),
    )
    summary = {
        "schema_version": "care-opd-cmb-summary-v1",
        "status": "complete",
        "model_label": args.model_label,
        "prompt_mode": "official_answer_only_zero_shot",
        "run_fingerprint": run_fingerprint,
        "fingerprints": fingerprints,
        "completed_at_utc": utc_now(),
        **summarize_rows(merged),
    }
    atomic_json(output_dir / "summary.json", summary)
    artifact_paths = {
        "predictions.jsonl": output_dir / "predictions.jsonl",
        "official_predictions.jsonl": output_dir / "official_predictions.jsonl",
        "controlled_robustness_predictions.jsonl": output_dir
        / "controlled_robustness_predictions.jsonl",
        "controlled_robustness_pairs.jsonl": output_dir
        / "controlled_robustness_pairs.jsonl",
        "summary.json": output_dir / "summary.json",
        "run_manifest.json": output_dir / "run_manifest.json",
    }
    atomic_json(
        output_dir / "COMPLETE.json",
        {
            "status": "complete",
            "rows": len(questions),
            "unique_ids": len(questions),
            "official_rows": EXPECTED_QUESTIONS,
            "overlap_clean_rows": EXPECTED_CLEAN_QUESTIONS,
            "controlled_robustness_rows": data_contract.EXPECTED_ROBUSTNESS_ITEMS,
            "run_fingerprint": run_fingerprint,
            "fingerprints": fingerprints,
            "artifact_sha256": {
                name: sha256_file(path) for name, path in artifact_paths.items()
            },
            "completed_at_utc": utc_now(),
        },
    )
    return summary


def _mcnemar_exact(
    care: Mapping[str, bool], base: Mapping[str, bool], ids: Sequence[str]
) -> Mapping[str, Any]:
    matched = [item for item in ids if item in care and item in base]
    care_only = sum(care[item] and not base[item] for item in matched)
    base_only = sum(base[item] and not care[item] for item in matched)
    discordant = care_only + base_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(care_only, base_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "matched_questions": len(matched),
        "care_correct_baseline_wrong": care_only,
        "baseline_correct_care_wrong": base_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _derived_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _paired_bootstrap(
    care: Mapping[str, bool],
    base: Mapping[str, bool],
    stem_by_id: Mapping[str, str],
    ids: Sequence[str],
    *,
    comparison_label: str,
) -> Mapping[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for deterministic paired bootstrap analysis") from exc

    spec = analysis_spec()
    boot = spec["paired_item_bootstrap"]
    reps = int(boot["replicates"])
    base_seed = int(boot["seed"])
    seed = _derived_seed(base_seed, comparison_label)
    matched = [item for item in ids if item in care and item in base]
    if len(matched) != len(ids):
        raise ValueError("paired bootstrap input is not fully matched")
    deltas = np.asarray(
        [int(bool(care[item])) - int(bool(base[item])) for item in matched],
        dtype=np.int8,
    )
    n = len(matched)
    care_accuracy = sum(bool(care[item]) for item in matched) / n
    base_accuracy = sum(bool(base[item]) for item in matched) / n
    point = float(deltas.sum() / n)

    # Resampling items is exactly multinomial resampling of the three possible
    # paired deltas; this avoids allocating a reps x 11,200 index matrix.
    counts = np.asarray(
        [(deltas == -1).sum(), (deltas == 0).sum(), (deltas == 1).sum()],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=reps)
    item_differences = (draws[:, 2] - draws[:, 0]) / n
    item_lower, item_upper = np.quantile(item_differences, [0.025, 0.975]).tolist()

    cluster_delta: dict[str, int] = {}
    cluster_size: dict[str, int] = {}
    for item, delta in zip(matched, deltas.tolist(), strict=True):
        stem = stem_by_id[item]
        cluster_delta[stem] = cluster_delta.get(stem, 0) + int(delta)
        cluster_size[stem] = cluster_size.get(stem, 0) + 1
    stems = sorted(cluster_delta)
    delta_array = np.asarray([cluster_delta[stem] for stem in stems], dtype=np.int32)
    size_array = np.asarray([cluster_size[stem] for stem in stems], dtype=np.int32)
    cluster_count = len(stems)
    cluster_rng = np.random.default_rng(_derived_seed(seed, "stem-cluster"))
    cluster_differences = np.empty(reps, dtype=np.float64)
    chunk = 50
    for start in range(0, reps, chunk):
        width = min(chunk, reps - start)
        sampled = cluster_rng.integers(0, cluster_count, size=(width, cluster_count))
        numerators = delta_array[sampled].sum(axis=1)
        denominators = size_array[sampled].sum(axis=1)
        cluster_differences[start : start + width] = numerators / denominators
    cluster_lower, cluster_upper = np.quantile(
        cluster_differences, [0.025, 0.975]
    ).tolist()
    margin = float(spec["preservation_margin_accuracy_points"])
    return {
        "n": n,
        "normalized_stem_clusters": cluster_count,
        "care_accuracy": care_accuracy,
        "baseline_accuracy": base_accuracy,
        "care_minus_baseline": point,
        "item_bootstrap": {
            "replicates": reps,
            "base_seed": base_seed,
            "derived_seed": seed,
            "ci95_lower": float(item_lower),
            "ci95_upper": float(item_upper),
        },
        "normalized_stem_cluster_bootstrap": {
            "replicates": reps,
            "base_seed": base_seed,
            "derived_seed": _derived_seed(seed, "stem-cluster"),
            "clusters": cluster_count,
            "ci95_lower": float(cluster_lower),
            "ci95_upper": float(cluster_upper),
        },
        "preservation_margin": margin,
        "preservation_gate_basis": "normalized_stem_cluster_bootstrap_ci95_lower",
        "preservation_gate_pass": bool(cluster_lower >= margin),
    }


def summarize_suite(args: argparse.Namespace) -> Mapping[str, Any]:
    suite = _load_json(args.suite)
    questions = load_questions(args.data_dir)
    evaluation_items = load_evaluation_items(args.data_dir)
    preregistration_lock = load_preregistration_lock(args.preregistration_lock)
    assert_suite_matches_preregistration(args.suite, preregistration_lock)
    expected_lock_sha = preregistration_lock["preregistration_lock_sha256"]
    official_root = args.output_root.resolve() / "official"
    question_by_id = {question.id: question for question in questions}
    full_ids = [question.id for question in questions]
    clean_ids = [question.id for question in questions if not question.overlap_excluded]
    records = []
    raw_by_model: dict[str, dict[str, bool]] = {}
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for entry in suite["models"]:
        label = str(entry["label"])
        model_output = official_root / label
        summary_path = model_output / "summary.json"
        predictions_path = model_output / "predictions.jsonl"
        run_manifest_path = model_output / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise FileNotFoundError(f"incomplete suite; missing {run_manifest_path}")
        run_manifest = _load_json(run_manifest_path)
        if run_manifest.get("model_label") != label:
            raise ValueError(f"model label mismatch for {label}")
        if run_manifest.get("model_path") != entry.get("canonical_path"):
            raise ValueError(f"model path mismatch for {label}")
        if run_manifest.get("fingerprints", {}).get("model_fingerprint") != entry.get(
            "model_fingerprint"
        ):
            raise ValueError(f"model fingerprint mismatch for {label}")
        if run_manifest.get("preregistration_lock_sha256") != expected_lock_sha:
            raise ValueError(
                f"preregistration lock mismatch for {label}; use the lock frozen before prediction"
            )
        if not _completed_valid(model_output, evaluation_items, run_manifest):
            raise ValueError(f"result failed completion/fingerprint validation for {label}")
        summary = _load_json(summary_path)
        rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
        ]
        expected_inference_items = EXPECTED_QUESTIONS + data_contract.EXPECTED_ROBUSTNESS_ITEMS
        if len(rows) != expected_inference_items or len({row["id"] for row in rows}) != expected_inference_items:
            raise ValueError(f"invalid result cardinality for {label}")
        full = summary["official_full"]
        clean = summary["overlap_clean"]
        full_overall = full["overall"]
        clean_overall = clean["overall"]
        robustness = summary["controlled_robustness_v1"]["overall"]
        robustness_ci = summary["controlled_robustness_v1"][
            "confidence_intervals_95"
        ]["metrics"]
        record: dict[str, Any] = {
            "model": label,
            "official_full_n": full_overall["n"],
            "official_full_accuracy": full_overall["option_set_accuracy"],
            "official_full_ci95_lower": full_overall["option_set_accuracy_wilson95_lower"],
            "official_full_ci95_upper": full_overall["option_set_accuracy_wilson95_upper"],
            "overlap_clean_n": clean_overall["n"],
            "overlap_clean_accuracy": clean_overall["option_set_accuracy"],
            "overlap_clean_ci95_lower": clean_overall["option_set_accuracy_wilson95_lower"],
            "overlap_clean_ci95_upper": clean_overall["option_set_accuracy_wilson95_upper"],
            "single_accuracy": clean["by_question_type"]["single"]["option_set_accuracy"],
            "multiple_accuracy": clean["by_question_type"]["multiple"]["option_set_accuracy"],
            "strict_format_rate": clean_overall["strict_format_rate"],
            "parse_failure_rate": clean_overall["parse_failure_rate"],
            "clean_task_success_rate": robustness["clean_task_success_rate"],
            "perturbed_task_success_rate": robustness["perturbed_task_success_rate"],
            "conditional_retention": robustness["conditional_retention"],
            "robustness_drop_pp": robustness["robustness_drop_pp"],
            "prediction_consistency": robustness["prediction_consistency"],
            "worst_perturbation_tsr": robustness["worst_perturbation_tsr"],
            "clean_correct_perturbed_wrong": robustness[
                "clean_correct_perturbed_wrong"
            ],
            "clean_wrong_perturbed_correct": robustness[
                "clean_wrong_perturbed_correct"
            ],
            "regression_rate": robustness["regression_rate"],
            "paired_units": robustness["paired_units"],
        }
        for metric_name, interval in robustness_ci.items():
            record[f"{metric_name}_numerator"] = interval["numerator"]
            record[f"{metric_name}_denominator"] = interval["denominator"]
            record[f"{metric_name}_ci95_lower"] = interval["ci95_lower"]
            record[f"{metric_name}_ci95_upper"] = interval["ci95_upper"]
        for exam_type in EXPECTED_EXAM_TYPE_COUNTS:
            record[f"{exam_type}_accuracy"] = clean["by_exam_type"][exam_type][
                "option_set_accuracy"
            ]
        records.append(record)
        raw_by_model[label] = {
            str(row["id"]): bool(row["option_set_exact"])
            for row in rows
            if row["perturbation"] == "clean"
        }
        rows_by_model[label] = rows
    if tuple(record["model"] for record in records) != MODEL_ORDER:
        raise ValueError("completed suite does not preserve the preregistered model order")
    if "care_opd" not in raw_by_model:
        raise ValueError("formal suite has no care_opd result")

    care = raw_by_model["care_opd"]
    care_rows = rows_by_model["care_opd"]
    stem_by_id = {
        str(row["id"]): data_contract.normalized_stem(question_by_id[str(row["id"])].question)
        for row in care_rows
        if row["perturbation"] == "clean"
    }
    comparisons = []
    for label in MODEL_ORDER:
        if label == "care_opd":
            continue
        subsets = {}
        for subset_name, subset_ids in (
            ("official_full", full_ids),
            ("overlap_clean", clean_ids),
        ):
            subsets[subset_name] = {
                "mcnemar": _mcnemar_exact(care, raw_by_model[label], subset_ids),
                "paired_accuracy_difference": _paired_bootstrap(
                    care,
                    raw_by_model[label],
                    stem_by_id,
                    subset_ids,
                    comparison_label=f"care_opd:{label}:{subset_name}",
                ),
            }
        comparisons.append({"baseline": label, "subsets": subsets})

    comparison_by_label = {row["baseline"]: row for row in comparisons}
    preservation_gates = {
        label: comparison_by_label[label]["subsets"]["overlap_clean"][
            "paired_accuracy_difference"
        ]
        for label in ("clean_sft", "gkd_full")
    }
    a_spec = locked_analysis_spec(preregistration_lock)
    payload = {
        "schema_version": "care-opd-cmb-suite-summary-v1",
        "prompt_mode": "official_answer_only_zero_shot",
        "dataset_fingerprint": dataset_fingerprint(args.data_dir),
        "suite_fingerprint": suite["suite_fingerprint"],
        "analysis": a_spec,
        "analysis_fingerprint": canonical_hash(a_spec),
        "unified_metric_schema_sha256": EXPECTED_UNIFIED_METRIC_SCHEMA_SHA256,
        "preregistration_lock_sha256": expected_lock_sha,
        "official_full_n": EXPECTED_QUESTIONS,
        "overlap_clean_n": EXPECTED_CLEAN_QUESTIONS,
        "models": records,
        "paired_comparisons_vs_care_opd": comparisons,
        "primary_overlap_clean_preservation_gates": preservation_gates,
        "paired_analysis_note": (
            "Exploratory paired item-level tests and fixed-seed percentile intervals without "
            "multiple-comparison correction. Because 11,200 IDs contain 11,098 unique "
            "overlap-audit-normalized stems, the stem-cluster bootstrap is the preregistered conservative "
            "basis for the -1 percentage-point preservation gate."
        ),
        "overlap_clean_note": (
            "Only four exact normalized-stem matches found by the pinned formal QZBC overlap "
            "audit are removed. This does not rule out semantic overlap or base-model "
            "pretraining exposure."
        ),
        "created_at_utc": utc_now(),
    }
    atomic_json(official_root / "suite_summary.json", payload)

    if records:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        atomic_text(official_root / "suite_summary.csv", stream.getvalue())

    compact_pairs = []
    for comparison in comparisons:
        clean_pair = comparison["subsets"]["overlap_clean"]
        difference = clean_pair["paired_accuracy_difference"]
        cluster = difference["normalized_stem_cluster_bootstrap"]
        compact_pairs.append(
            {
                "baseline": comparison["baseline"],
                "care_minus_baseline": difference["care_minus_baseline"],
                "cluster_ci95_lower": cluster["ci95_lower"],
                "cluster_ci95_upper": cluster["ci95_upper"],
                "mcnemar_p": clean_pair["mcnemar"]["two_sided_exact_p"],
                "minus_1pp_preservation_pass": difference["preservation_gate_pass"],
            }
        )
    headers = list(records[0]) if records else []
    markdown = ["# CMB-Exam — official answer-only zero-shot", ""]
    if records:
        display_headers = [
            "model",
            "official_full_n",
            "official_full_accuracy",
            "overlap_clean_n",
            "overlap_clean_accuracy",
            "single_accuracy",
            "multiple_accuracy",
            "strict_format_rate",
            "parse_failure_rate",
            "clean_task_success_rate",
            "perturbed_task_success_rate",
            "conditional_retention",
            "robustness_drop_pp",
            "prediction_consistency",
            "worst_perturbation_tsr",
            "regression_rate",
            "paired_units",
        ]
        markdown.extend(
            [
                "| " + " | ".join(display_headers) + " |",
                "|" + "|".join(["---"] + ["---:"] * (len(display_headers) - 1)) + "|",
            ]
        )
        for record in records:
            values = []
            for key in display_headers:
                value = record[key]
                if key == "robustness_drop_pp":
                    values.append(f"{value:.2f} pp")
                else:
                    values.append(
                        f"{100 * value:.2f}%" if isinstance(value, float) else str(value)
                    )
            markdown.append("| " + " | ".join(values) + " |")
    markdown.extend(
        [
            "",
            "## CARE-OPD paired overlap-clean differences",
            "",
            "| baseline | CARE - baseline | stem-cluster 95% CI | exact McNemar p | -1pp preservation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in compact_pairs:
        markdown.append(
            f"| {row['baseline']} | {100 * row['care_minus_baseline']:.2f} pp | "
            f"[{100 * row['cluster_ci95_lower']:.2f}, {100 * row['cluster_ci95_upper']:.2f}] pp | "
            f"{row['mcnemar_p']:.6g} | {'PASS' if row['minus_1pp_preservation_pass'] else 'FAIL'} |"
        )
    markdown.extend(
        [
            "",
            "Primary metric: exact equality of parsed option sets. The main generalization "
            "interpretation uses the 11,196-item overlap-clean subset; all 11,200 official "
            "test items remain reported for official-set comparability.",
            "",
        ]
    )
    atomic_text(official_root / "suite_summary.md", "\n".join(markdown))
    return payload


def inspect_data(args: argparse.Namespace) -> Mapping[str, Any]:
    questions = load_questions(args.data_dir)
    class_counts = Counter(question.class_key for question in questions)
    return {
        "data_dir": str(args.data_dir.resolve()),
        "dataset_fingerprint": dataset_fingerprint(args.data_dir),
        "questions": len(questions),
        "unique_ids": len({question.id for question in questions}),
        "overlap_clean_questions": sum(not question.overlap_excluded for question in questions),
        "overlap_excluded_ids": [
            int(question.id) for question in questions if question.overlap_excluded
        ],
        "official_question_types": dict(Counter(question.question_type for question in questions)),
        "gold_cardinality_types": dict(
            Counter(question.gold_cardinality_type for question in questions)
        ),
        "exam_types": dict(Counter(question.exam_type for question in questions)),
        "exam_classes": dict(class_counts),
        "exam_class_count": len(class_counts),
        "exam_subject_count": len({question.exam_subject for question in questions}),
        "raw_c_type_mismatches": sum(
            question.raw_question_type == "C型选择题" for question in questions
        ),
        "unique_normalized_stems": len(
            {data_contract.normalized_stem(question.question) for question in questions}
        ),
        "question_sha256": data_contract.EXPECTED_QUESTION_SHA256,
        "answer_sha256": data_contract.EXPECTED_ANSWER_SHA256,
        "overlap_report_sha256": data_contract.EXPECTED_OVERLAP_REPORT_SHA256,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect-data")
    inspect_parser.add_argument("--data-dir", type=Path, required=True)

    freeze_parser = commands.add_parser("freeze-suite")
    freeze_parser.add_argument("--suite", type=Path, required=True)
    freeze_parser.add_argument("--model", action="append", required=True, dest="models")

    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--data-dir", type=Path, required=True)
    eval_parser.add_argument("--suite", type=Path, required=True)
    eval_parser.add_argument("--model-label", choices=MODEL_ORDER, required=True)
    eval_parser.add_argument("--output-root", type=Path, required=True)
    eval_parser.add_argument("--dtype", default="bfloat16")
    eval_parser.add_argument("--max-new-tokens", type=int, default=8)
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--num-workers", type=int, default=FORMAL_NUM_WORKERS)
    eval_parser.add_argument("--devices", default=",".join(FORMAL_DEVICES))
    eval_parser.add_argument(
        "--preregistration-lock",
        type=Path,
        default=DEFAULT_PREREGISTRATION_LOCK_PATH,
    )

    summary_parser = commands.add_parser("summarize-suite")
    summary_parser.add_argument("--data-dir", type=Path, required=True)
    summary_parser.add_argument("--suite", type=Path, required=True)
    summary_parser.add_argument("--output-root", type=Path, required=True)
    summary_parser.add_argument(
        "--preregistration-lock",
        type=Path,
        default=DEFAULT_PREREGISTRATION_LOCK_PATH,
    )
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "inspect-data":
        result = inspect_data(args)
    elif args.command == "freeze-suite":
        result = freeze_suite(args.suite, args.models)
    elif args.command == "evaluate":
        if args.max_new_tokens != 8:
            raise ValueError("formal CMB-Exam protocol fixes --max-new-tokens=8")
        if args.batch_size != 8:
            raise ValueError("formal CMB-Exam protocol fixes --batch-size=8")
        devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
        if args.num_workers != FORMAL_NUM_WORKERS or devices != FORMAL_DEVICES:
            raise ValueError(
                "formal runner requires four workers on logical "
                "cuda:0,cuda:1,cuda:2,cuda:3"
            )
        args.devices = devices
        result = evaluate(args)
    elif args.command == "summarize-suite":
        result = summarize_suite(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
