#!/usr/bin/env python3
"""Production evaluator for the public TCMEval-PA v3 benchmark.

This module intentionally does not import or call the QZBC GeneratedResponse
parser/verifier.  TCMEval-PA is an independent multiple-choice benchmark and
is scored by exact equality of option sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import signal
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET


EXPECTED_DATA_SHA256 = "905cb28f145ba50f3b1c912265ff6649e0bb38f56adf6cd588d77ca768db92c1"
EXPECTED_QUESTIONS = 328
EXPECTED_TYPE_COUNTS = {"single": 297, "multiple": 31}
EXPECTED_CATEGORY_COUNTS = {"基础概念": 6, "处方规范性": 86, "处方适宜性": 236}
FIGSHARE_DOI = "10.6084/m9.figshare.29651261.v3"
TCMEVAL_REPO_COMMIT = "12473b50ce94ad2fd22b1e984b826b0a618c4407"
PROTOCOL_VERSION = "care-opd-tcmeval-pa-v1"

OFFICIAL_SYSTEM = {
    "single": "以下是关于中药处方审核的单选题，请根据规则选择正确的选项。仅输出选项即可。",
    "multiple": "以下是关于中药处方审核的多选题，请根据规则选择所有正确的选项。仅输出选项即可。",
}
LABEL_BLIND_SYSTEM = (
    "以下是关于中药处方审核的选择题，请根据规则选择所有正确选项。"
    "答案可能包含一个或多个选项。仅输出选项字母，不要解释。"
)
USER_TEMPLATE = "题目：{question}\n选项：{candidate_answers}"

XML_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XML_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class Question:
    index: int
    id: str
    question: str
    candidate_answers: str
    allowed_options: tuple[str, ...]
    gold_options: tuple[str, ...]
    question_type: str
    category: str
    rule_id: str
    rule_summary: str


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
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    value = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in materialized
    )
    atomic_text(path, value)
    return len(materialized)


def _cell_column(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"invalid XLSX cell reference: {reference!r}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def _xlsx_rows(path: Path) -> list[list[str]]:
    ns = {"m": XML_MAIN, "r": XML_REL, "pr": XML_PACKAGE_REL}
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relations.findall("pr:Relationship", ns)
        }
        sheets = workbook.findall("m:sheets/m:sheet", ns)
        if len(sheets) != 1 or sheets[0].attrib.get("name") != "单选-多选":
            names = [sheet.attrib.get("name") for sheet in sheets]
            raise ValueError(f"unexpected workbook sheets: {names}")
        relation_id = sheets[0].attrib[f"{{{XML_REL}}}id"]
        target = relation_targets[relation_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("m:si", ns):
                shared.append(
                    "".join(node.text or "" for node in item.iter(f"{{{XML_MAIN}}}t"))
                )

        sheet_root = ET.fromstring(archive.read(target))
        result: list[list[str]] = []
        for row in sheet_root.findall(".//m:sheetData/m:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                column = _cell_column(cell.attrib["r"])
                kind = cell.attrib.get("t")
                if kind == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{{{XML_MAIN}}}t")
                    )
                else:
                    node = cell.find("m:v", ns)
                    value = "" if node is None or node.text is None else node.text
                    if kind == "s" and value:
                        value = shared[int(value)]
                values[column] = value
            width = max(values, default=-1) + 1
            result.append([values.get(index, "") for index in range(width)])
    return result


def _normalize_id(value: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _option_inventory(candidate_answers: str) -> tuple[str, ...]:
    labels = re.findall(r"(?m)^\s*([A-Z])\s*[\.、:：)]", candidate_answers.upper())
    if not labels:
        raise ValueError(f"could not parse option labels from {candidate_answers!r}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate option labels in {candidate_answers!r}")
    expected = [chr(ord("A") + index) for index in range(len(labels))]
    if labels != expected:
        raise ValueError(f"non-contiguous option labels {labels}, expected {expected}")
    return tuple(labels)


def _gold_inventory(answer: str) -> tuple[str, ...]:
    labels = re.findall(r"[A-Z]", answer.upper())
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"invalid gold answer: {answer!r}")
    return tuple(sorted(labels))


def load_questions(path: Path) -> list[Question]:
    path = path.resolve()
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_DATA_SHA256:
        raise ValueError(
            f"TCMEval-PA checksum mismatch: expected {EXPECTED_DATA_SHA256}, got {actual_sha}"
        )
    rows = _xlsx_rows(path)
    if not rows:
        raise ValueError("empty workbook")
    required = (
        "ID",
        "Question",
        "Candidate Answers",
        "Answer",
        "Explanation",
        "Category",
        "Rule_ID",
        "Rule_summary",
    )
    headers = rows[0]
    missing = [name for name in required if name not in headers]
    if missing:
        raise ValueError(f"missing workbook columns: {missing}")
    positions = {name: headers.index(name) for name in required}
    questions: list[Question] = []
    for index, raw in enumerate(rows[1:]):
        value = lambda name: raw[positions[name]] if positions[name] < len(raw) else ""
        allowed = _option_inventory(value("Candidate Answers"))
        gold = _gold_inventory(value("Answer"))
        if not set(gold).issubset(allowed):
            raise ValueError(f"question {value('ID')} gold {gold} outside options {allowed}")
        questions.append(
            Question(
                index=index,
                id=_normalize_id(value("ID")),
                question=value("Question").strip(),
                candidate_answers=value("Candidate Answers").strip(),
                allowed_options=allowed,
                gold_options=gold,
                question_type="single" if len(gold) == 1 else "multiple",
                category=value("Category").strip(),
                rule_id=value("Rule_ID").strip(),
                rule_summary=value("Rule_summary").strip(),
            )
        )
    ids = [question.id for question in questions]
    if len(questions) != EXPECTED_QUESTIONS or len(set(ids)) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"expected {EXPECTED_QUESTIONS} rows with unique IDs; got "
            f"{len(questions)} rows and {len(set(ids))} IDs"
        )
    type_counts = Counter(question.question_type for question in questions)
    category_counts = Counter(question.category for question in questions)
    if dict(type_counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(f"unexpected single/multiple counts: {dict(type_counts)}")
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"unexpected category counts: {dict(category_counts)}")
    return questions


def prompt_spec(mode: str) -> Mapping[str, Any]:
    if mode == "official":
        systems: Any = OFFICIAL_SYSTEM
        reconstruction = True
        comparable = True
    elif mode == "label_blind":
        systems = LABEL_BLIND_SYSTEM
        reconstruction = False
        comparable = False
    else:
        raise ValueError(f"unsupported prompt mode: {mode}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_mode": mode,
        "system_prompts": systems,
        "user_template": USER_TEMPLATE,
        "chat_template": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "type_reconstructed_from_gold_cardinality": reconstruction,
        "directly_comparable_to_official_reference": comparable,
    }


def prompt_messages(question: Question, mode: str) -> list[dict[str, str]]:
    if mode == "official":
        system = OFFICIAL_SYSTEM[question.question_type]
    elif mode == "label_blind":
        system = LABEL_BLIND_SYSTEM
    else:
        raise ValueError(mode)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                question=question.question,
                candidate_answers=question.candidate_answers,
            ),
        },
    ]


def parse_prediction(raw_output: str, allowed: Sequence[str]) -> Mapping[str, Any]:
    normalized = raw_output.strip().upper().translate(
        str.maketrans("ＡＢＣＤＥＦＧＨＩＪ", "ABCDEFGHIJ")
    )
    all_letters = re.findall(r"[A-Z]", normalized)
    allowed_set = set(allowed)
    valid = tuple(sorted(set(letter for letter in all_letters if letter in allowed_set)))
    invalid = tuple(sorted(set(letter for letter in all_letters if letter not in allowed_set)))
    parse_ok = bool(valid) and not invalid
    format_shape = re.fullmatch(r"[A-Z](?:\s*[、,， ]?\s*[A-Z])*", normalized)
    strict_format = bool(format_shape) and parse_ok and len(all_letters) == len(set(all_letters))
    if strict_format and tuple(all_letters) != tuple(sorted(all_letters)):
        strict_format = False
    return {
        "parsed_options": list(valid),
        "invalid_letters": list(invalid),
        "parse_ok": parse_ok,
        "strict_format": strict_format,
    }


def generation_spec(dtype: str, max_new_tokens: int) -> Mapping[str, Any]:
    return {
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "enable_thinking": False,
        "seed": 42,
    }


def _weight_files(model: Path) -> list[Path]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model / index_name
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
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
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
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
    if len({label for label, _ in requested}) != len(requested):
        raise ValueError("duplicate model labels")
    if suite_path.is_file():
        existing = json.loads(suite_path.read_text(encoding="utf-8"))
        old = [(item["label"], Path(item["canonical_path"])) for item in existing["models"]]
        if old != requested:
            raise ValueError(
                f"frozen suite {suite_path} does not match requested models; use a new output root"
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
        "schema_version": "tcmeval-pa-model-suite-v1",
        "created_at_utc": utc_now(),
        "models": entries,
        "suite_fingerprint": canonical_hash(
            [(item["label"], item["model_fingerprint"]) for item in entries]
        ),
    }
    atomic_json(suite_path, payload)
    return payload


def _load_suite_entry(suite_path: Path, label: str) -> Mapping[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
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


def _recover_shard(path: Path, expected: Mapping[int, str], run_fp: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw_lines = path.read_bytes().splitlines()
    rows: list[dict[str, Any]] = []
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
        if expected.get(index) != str(row.get("id")) or row.get("run_fingerprint") != run_fp:
            raise ValueError(f"stale or unexpected row in {path}: index={index}")
        if any(existing["index"] == index for existing in rows):
            raise ValueError(f"duplicate index {index} in {path}")
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
    prompt_mode: str,
    dtype: str,
    max_new_tokens: int,
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
                f"model={model_path} remaining={len(remaining)}",
                flush=True,
            )
            model_args = ModelArguments(
                student_model_name_or_path=model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
            tokenizer = load_tokenizer(model_args)
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
            for position, question in enumerate(remaining, 1):
                messages = prompt_messages(question, prompt_mode)
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "tokenizer chat template does not accept enable_thinking=False; "
                        "refusing a protocol-changing fallback"
                    ) from exc
                encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
                encoded = {key: value.to(target_device) for key, value in encoded.items()}
                torch.manual_seed(42 + question.index)
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
                new_tokens = generated[0, encoded["input_ids"].shape[1] :]
                raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                parsed = parse_prediction(raw_output, question.allowed_options)
                exact = parsed["parse_ok"] and tuple(parsed["parsed_options"]) == question.gold_options
                row = {
                    "index": question.index,
                    "id": question.id,
                    "question": question.question,
                    "candidate_answers": question.candidate_answers,
                    "allowed_options": list(question.allowed_options),
                    "gold_options": list(question.gold_options),
                    "question_type": question.question_type,
                    "category": question.category,
                    "rule_id": question.rule_id,
                    "rule_summary": question.rule_summary,
                    "prompt_mode": prompt_mode,
                    "messages": messages,
                    "raw_output": raw_output,
                    **parsed,
                    "option_set_exact": bool(exact),
                    "strict_exact": bool(exact and parsed["strict_format"]),
                    "latency_seconds": time.perf_counter() - started,
                    "worker_id": worker_id,
                    "device": device,
                    "run_fingerprint": run_fingerprint,
                    **fingerprints,
                }
                _append_jsonl(shard_path, row)
                if position == 1 or position % 10 == 0 or position == len(remaining):
                    print(
                        f"[progress] worker={worker_id} device={device} "
                        f"step={len(completed) + position}/{len(questions)} id={question.id}",
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


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    total = len(rows)
    exact = sum(bool(row["option_set_exact"]) for row in rows)
    strict_exact = sum(bool(row["strict_exact"]) for row in rows)
    strict_format = sum(bool(row["strict_format"]) for row in rows)
    parse_failures = sum(not bool(row["parse_ok"]) for row in rows)
    return {
        "n": total,
        "option_set_exact": exact,
        "option_set_accuracy": exact / total if total else None,
        "strict_exact": strict_exact,
        "strict_exact_accuracy": strict_exact / total if total else None,
        "strict_format": strict_format,
        "strict_format_rate": strict_format / total if total else None,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / total if total else None,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_type = {
        kind: _metric_block([row for row in rows if row["question_type"] == kind])
        for kind in ("single", "multiple")
    }
    by_category = {
        category: _metric_block([row for row in rows if row["category"] == category])
        for category in EXPECTED_CATEGORY_COUNTS
    }
    rules = sorted({str(row["rule_id"]) for row in rows})
    by_rule = {
        rule: _metric_block([row for row in rows if row["rule_id"] == rule])
        for rule in rules
    }
    return {
        "overall": _metric_block(rows),
        "by_question_type": by_type,
        "by_category": by_category,
        "by_rule": by_rule,
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
    if not (sentinel.is_file() and summary_path.is_file() and predictions_path.is_file()):
        return False
    try:
        completion = json.loads(sentinel.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return False
    expected_ids = {question.id for question in questions}
    actual_ids = [str(row.get("id")) for row in rows]
    fingerprints = run_manifest["fingerprints"]
    return bool(
        completion.get("status") == "complete"
        and completion.get("run_fingerprint") == run_manifest["run_fingerprint"]
        and summary.get("run_fingerprint") == run_manifest["run_fingerprint"]
        and summary.get("fingerprints") == fingerprints
        and len(rows) == EXPECTED_QUESTIONS
        and len(set(actual_ids)) == EXPECTED_QUESTIONS
        and set(actual_ids) == expected_ids
        and all(row.get("dataset_fingerprint") == fingerprints["dataset_fingerprint"] for row in rows)
        and all(row.get("prompt_fingerprint") == fingerprints["prompt_fingerprint"] for row in rows)
        and all(row.get("generation_fingerprint") == fingerprints["generation_fingerprint"] for row in rows)
        and all(row.get("model_fingerprint") == fingerprints["model_fingerprint"] for row in rows)
    )


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    questions = load_questions(args.data)
    model_entry = _load_suite_entry(args.suite, args.model_label)
    p_spec = prompt_spec(args.prompt_mode)
    g_spec = generation_spec(args.dtype, args.max_new_tokens)
    fingerprints = {
        "dataset_fingerprint": EXPECTED_DATA_SHA256,
        "prompt_fingerprint": canonical_hash(p_spec),
        "generation_fingerprint": canonical_hash(g_spec),
        "model_fingerprint": str(model_entry["model_fingerprint"]),
    }
    code_fingerprint = sha256_file(Path(__file__).resolve())
    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "model_label": args.model_label,
        "model_path": model_entry["canonical_path"],
        "prompt_mode": args.prompt_mode,
        "num_workers": args.num_workers,
        "devices": list(args.devices),
        "fingerprints": fingerprints,
        "evaluator_sha256": code_fingerprint,
    }
    run_fingerprint = canonical_hash(contract)
    output_dir = args.output_root.resolve() / args.prompt_mode / args.model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run_manifest.json"
    if run_manifest_path.is_file():
        existing = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"run contract changed for {output_dir}; use a new output root instead of mixing results"
            )
        run_manifest = existing
    else:
        run_manifest = {
            **contract,
            "run_fingerprint": run_fingerprint,
            "created_at_utc": utc_now(),
            "dataset": {
                "path": str(args.data.resolve()),
                "sha256": EXPECTED_DATA_SHA256,
                "questions": EXPECTED_QUESTIONS,
                "figshare_doi": FIGSHARE_DOI,
                "tcmeval_repo_commit": TCMEVAL_REPO_COMMIT,
            },
            "prompt": p_spec,
            "generation": g_spec,
            "model": model_entry,
        }
        atomic_json(run_manifest_path, run_manifest)
    if _completed_valid(output_dir, questions, run_manifest):
        print(f"[skip] validated complete result: {output_dir}", flush=True)
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    (output_dir / "COMPLETE.json").unlink(missing_ok=True)
    shard_root = output_dir / "shards"
    shard_root.mkdir(exist_ok=True)
    partitions = [questions[index :: args.num_workers] for index in range(args.num_workers)]
    context = mp.get_context("spawn")
    processes = []
    for worker_id, (device, partition) in enumerate(zip(args.devices, partitions, strict=True)):
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
                args.prompt_mode,
                args.dtype,
                args.max_new_tokens,
                run_fingerprint,
                fingerprints,
            ),
            name=f"tcmeval-pa-{args.model_label}-{worker_id}",
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
                    error_path.read_text(encoding="utf-8") if error_path.is_file()
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
    ids = [row["id"] for row in merged]
    if len(merged) != EXPECTED_QUESTIONS or len(set(ids)) != EXPECTED_QUESTIONS:
        raise RuntimeError("merged predictions are incomplete or contain duplicate IDs")
    atomic_jsonl(output_dir / "predictions.jsonl", merged)
    summary = {
        "schema_version": "tcmeval-pa-summary-v1",
        "status": "complete",
        "model_label": args.model_label,
        "prompt_mode": args.prompt_mode,
        "run_fingerprint": run_fingerprint,
        "fingerprints": fingerprints,
        "completed_at_utc": utc_now(),
        **summarize_rows(merged),
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "COMPLETE.json",
        {
            "status": "complete",
            "rows": EXPECTED_QUESTIONS,
            "unique_ids": EXPECTED_QUESTIONS,
            "run_fingerprint": run_fingerprint,
            "fingerprints": fingerprints,
            "completed_at_utc": utc_now(),
        },
    )
    return summary


def _mcnemar_exact(care: Mapping[str, bool], base: Mapping[str, bool]) -> Mapping[str, Any]:
    ids = sorted(set(care) & set(base))
    care_only = sum(care[item] and not base[item] for item in ids)
    base_only = sum(base[item] and not care[item] for item in ids)
    discordant = care_only + base_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(care_only, base_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {
        "matched_questions": len(ids),
        "care_correct_baseline_wrong": care_only,
        "baseline_correct_care_wrong": base_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def summarize_suite(args: argparse.Namespace) -> Mapping[str, Any]:
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    questions = load_questions(args.data)
    mode_root = args.output_root.resolve() / args.prompt_mode
    records = []
    raw_by_model: dict[str, dict[str, bool]] = {}
    for entry in suite["models"]:
        label = entry["label"]
        model_output = mode_root / label
        summary_path = model_output / "summary.json"
        predictions_path = model_output / "predictions.jsonl"
        run_manifest_path = model_output / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise FileNotFoundError(f"incomplete suite; missing {run_manifest_path}")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if not _completed_valid(model_output, questions, run_manifest):
            raise ValueError(f"result failed completion/fingerprint validation for {label}")
        if not summary_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"incomplete suite; missing results for {label}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != EXPECTED_QUESTIONS or len({row["id"] for row in rows}) != EXPECTED_QUESTIONS:
            raise ValueError(f"invalid result cardinality for {label}")
        overall = summary["overall"]
        record = {
            "model": label,
            "n": overall["n"],
            "option_set_accuracy": overall["option_set_accuracy"],
            "strict_exact_accuracy": overall["strict_exact_accuracy"],
            "single_accuracy": summary["by_question_type"]["single"]["option_set_accuracy"],
            "multiple_accuracy": summary["by_question_type"]["multiple"]["option_set_accuracy"],
            "基础概念_accuracy": summary["by_category"]["基础概念"]["option_set_accuracy"],
            "处方规范性_accuracy": summary["by_category"]["处方规范性"]["option_set_accuracy"],
            "处方适宜性_accuracy": summary["by_category"]["处方适宜性"]["option_set_accuracy"],
            "strict_format_rate": overall["strict_format_rate"],
            "parse_failure_rate": overall["parse_failure_rate"],
        }
        records.append(record)
        raw_by_model[label] = {row["id"]: bool(row["option_set_exact"]) for row in rows}
    if "care_opd" not in raw_by_model:
        comparisons = []
    else:
        comparisons = [
            {"baseline": label, **_mcnemar_exact(raw_by_model["care_opd"], values)}
            for label, values in raw_by_model.items()
            if label != "care_opd"
        ]
    payload = {
        "schema_version": "tcmeval-pa-suite-summary-v1",
        "prompt_mode": args.prompt_mode,
        "dataset_sha256": EXPECTED_DATA_SHA256,
        "suite_fingerprint": suite["suite_fingerprint"],
        "models": records,
        "paired_mcnemar_vs_care_opd": comparisons,
        "paired_mcnemar_note": (
            "Exploratory item-level exact tests without multiple-comparison correction; "
            "the workbook has 328 IDs but 314 unique normalized stems."
        ),
        "created_at_utc": utc_now(),
    }
    atomic_json(mode_root / "suite_summary.json", payload)
    csv_path = mode_root / "suite_summary.csv"
    if records:
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        atomic_text(csv_path, stream.getvalue())
    headers = list(records[0]) if records else []
    markdown = [f"# TCMEval-PA v3 — {args.prompt_mode}", ""]
    if records:
        markdown.extend([
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
        ])
        for record in records:
            values = []
            for key in headers:
                value = record[key]
                values.append(f"{100 * value:.2f}%" if key.endswith(("accuracy", "rate")) else str(value))
            markdown.append("| " + " | ".join(values) + " |")
    markdown.extend([
        "",
        "Primary metric: option-set exact accuracy. `official` reconstructs single/multiple "
        "type from gold cardinality as disclosed in each run manifest; `label_blind` does not.",
        "",
    ])
    atomic_text(mode_root / "suite_summary.md", "\n".join(markdown))
    return payload


def inspect_data(args: argparse.Namespace) -> Mapping[str, Any]:
    questions = load_questions(args.data)
    return {
        "path": str(args.data.resolve()),
        "sha256": sha256_file(args.data),
        "questions": len(questions),
        "unique_ids": len({question.id for question in questions}),
        "question_types": dict(Counter(question.question_type for question in questions)),
        "categories": dict(Counter(question.category for question in questions)),
        "rules": len({question.rule_id for question in questions}),
        "unique_stems": len({re.sub(r"\s+", "", question.question) for question in questions}),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect-data")
    inspect_parser.add_argument("--data", type=Path, required=True)

    freeze_parser = commands.add_parser("freeze-suite")
    freeze_parser.add_argument("--suite", type=Path, required=True)
    freeze_parser.add_argument("--model", action="append", required=True, dest="models")

    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--data", type=Path, required=True)
    eval_parser.add_argument("--suite", type=Path, required=True)
    eval_parser.add_argument("--model-label", required=True)
    eval_parser.add_argument("--output-root", type=Path, required=True)
    eval_parser.add_argument("--prompt-mode", choices=("official", "label_blind"), default="official")
    eval_parser.add_argument("--dtype", default="bfloat16")
    eval_parser.add_argument("--max-new-tokens", type=int, default=8)
    eval_parser.add_argument("--num-workers", type=int, default=2)
    eval_parser.add_argument("--devices", default="cuda:0,cuda:1")

    summary_parser = commands.add_parser("summarize-suite")
    summary_parser.add_argument("--data", type=Path, required=True)
    summary_parser.add_argument("--suite", type=Path, required=True)
    summary_parser.add_argument("--output-root", type=Path, required=True)
    summary_parser.add_argument("--prompt-mode", choices=("official", "label_blind"), default="official")
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "inspect-data":
        result = inspect_data(args)
    elif args.command == "freeze-suite":
        result = freeze_suite(args.suite, args.models)
    elif args.command == "evaluate":
        if args.max_new_tokens != 8:
            raise ValueError("formal TCMEval-PA protocol fixes --max-new-tokens=8")
        devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
        if args.num_workers != 2 or len(devices) != args.num_workers:
            raise ValueError("formal runner requires exactly two workers/devices")
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
