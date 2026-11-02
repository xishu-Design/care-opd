#!/usr/bin/env python3
"""Freeze a deterministic 30-case TCM-BEST4SDT expert-preference track.

The model task file contains case stems only.  Upstream reference answers and
diagnosis labels are written to a separate PRIVATE artifact and are never
needed by the four-system inference runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "care_opd.competition.tcm_best4sdt_agent30.v1"
SAMPLE_SEED = "20260825"
SELECTION_ALGORITHM = "one_per_normalized_diagnosis_sha256_rank_v1"
TASK_ORDER_ALGORITHM = "sha256_task_order_v1"
SAMPLE_LABEL = "TCM-BEST4SDT 固定30例开放式专家盲审轨"
EXPECTED_SOURCE_SHA256 = (
    "d94711ac60db83e3a33c2c46b1d6efbf072565a22e8fd464f5f61b444860de59"
)
EXPECTED_SOURCE_MD5 = "36aac279a5ad17c32d2b2d1ff32c8b1d"
EXPECTED_SOURCE_ROWS = 300
FIGSHARE_ARTICLE_ID = 30615956
FIGSHARE_FILE_ID = 59596040
FIGSHARE_DOI = "10.6084/m9.figshare.30615956.v8"
SOURCE_URL = f"https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}"
SOURCE_LICENSE = "CC BY 4.0"
QZBC_TEXT_LIMIT = 2000
NEAR_JACCARD_THRESHOLD = 0.55
NEAR_CONTAINMENT_THRESHOLD = 0.80
NEAR_SEQUENCE_THRESHOLD = 0.85
NGRAM_SIZE = 8

SYSTEM_PROMPT = (
    "你是中医辨证论治场景中的医生角色助手。请只依据病例中可见事实进行专业分析，"
    "明确区分病例事实、辨证推断与仍需复核的信息；证据不足时说明不确定性，不虚构"
    "病例未提供的症状、检查或既往史。按用户指定结构作答。"
)

USER_PREFIX = """请对以下公开、已匿名化的中医教学病例进行辨证论治分析。请依次给出：
1. 病例要点（只复述与判断有关的已知事实）；
2. 中医辨病与辨证结论；
3. 病因病机及“症状/舌脉—判断”证据链；
4. 治则治法；
5. 方药思路与加减原则（不写固定剂量，不作为现实处方）；
6. 需要鉴别或进一步复核的信息。

病例："""


def _care_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _write_frozen(path: Path, payload: bytes) -> None:
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


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return "".join(character for character in value if character.isalnum())


def _ngrams(value: str, size: int = NGRAM_SIZE) -> set[str]:
    if len(value) < size:
        return {value} if value else set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def _rank(
    namespace: str,
    source_id: str,
    *,
    seed: str = SAMPLE_SEED,
    algorithm: str = SELECTION_ALGORITHM,
) -> str:
    return _sha256_text(
        f"tcm-best4sdt-v8|agent30|{algorithm}|{seed}|"
        f"{namespace}|{source_id}"
    )


def _iter_strings(value: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        if len(_normalize(value)) >= 20:
            yield prefix, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_strings(item, f"{prefix}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_strings(item, child)


def _load_local_corpus(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    source_meta: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        stable_source_label = path.name
        source_meta.append(
            {
                "source_file": stable_source_label,
                "sha256": _sha256_file(path),
            }
        )
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            uri = f"file:{path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA query_only = ON")
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if "qa_documents" not in tables:
                    raise ValueError(f"SQLite overlap source has no qa_documents table: {path}")
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(qa_documents)")
                }
                selected = [
                    name for name in ("title", "question", "answer") if name in columns
                ]
                if not selected:
                    raise ValueError(
                        f"qa_documents has no auditable text columns: {path}"
                    )
                roots = [
                    dict(row)
                    for row in connection.execute(
                        f"SELECT {', '.join(selected)} FROM qa_documents ORDER BY rowid"
                    )
                ]
            finally:
                connection.close()
        elif path.suffix.lower() == ".jsonl":
            roots: list[Any] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
                roots = list(loaded["items"])
            elif isinstance(loaded, list):
                roots = loaded
            else:
                roots = [loaded]
        for root_index, root in enumerate(roots):
            # Restrict the knowledge-base audit to actual payload/content fields.
            candidate = root
            if isinstance(root, dict) and "training_payload" in root:
                candidate = root["training_payload"]
            for field, text in _iter_strings(candidate):
                normalized = _normalize(text)
                if len(normalized) < 20:
                    continue
                documents.append(
                    {
                        "source": stable_source_label,
                        "record": root_index,
                        "field": field,
                        "normalized": normalized,
                        "grams": _ngrams(normalized),
                    }
                )
    return documents, source_meta


def _best_overlap(
    text: str,
    documents: list[dict[str, Any]],
    inverted: dict[str, list[int]],
) -> dict[str, Any]:
    normalized = _normalize(text)
    grams = _ngrams(normalized)
    exact_hash = _sha256_text(normalized)
    counts: Counter[int] = Counter()
    for gram in grams:
        counts.update(inverted.get(gram, ()))
    best: dict[str, Any] = {
        "status": "no_detected_overlap",
        "exact_match": False,
        "near_duplicate": False,
        "best_jaccard": 0.0,
        "best_containment": 0.0,
        "best_sequence_ratio": 0.0,
        "matched_source": None,
        "matched_record": None,
        "matched_field": None,
    }
    for index, common in counts.most_common(40):
        document = documents[index]
        other_grams = document["grams"]
        union = len(grams) + len(other_grams) - common
        jaccard = common / union if union else 1.0
        containment = common / min(len(grams), len(other_grams)) if grams and other_grams else 0.0
        # SequenceMatcher is reserved for plausible candidates to keep the audit fast.
        sequence = 0.0
        if jaccard >= 0.20 or containment >= 0.40:
            sequence = SequenceMatcher(None, normalized, document["normalized"]).ratio()
        exact = exact_hash == _sha256_text(document["normalized"])
        near = (
            exact
            or jaccard >= NEAR_JACCARD_THRESHOLD
            or containment >= NEAR_CONTAINMENT_THRESHOLD
            or sequence >= NEAR_SEQUENCE_THRESHOLD
        )
        score = (int(exact), int(near), jaccard, containment, sequence)
        prior = (
            int(best["exact_match"]),
            int(best["near_duplicate"]),
            best["best_jaccard"],
            best["best_containment"],
            best["best_sequence_ratio"],
        )
        if score > prior:
            best.update(
                {
                    "status": "exact_match" if exact else ("near_duplicate" if near else "similar_below_threshold"),
                    "exact_match": exact,
                    "near_duplicate": near,
                    "best_jaccard": round(jaccard, 6),
                    "best_containment": round(containment, 6),
                    "best_sequence_ratio": round(sequence, 6),
                    "matched_source": document["source"],
                    "matched_record": document["record"],
                    "matched_field": document["field"],
                }
            )
    return best


def _source_id(row: dict[str, Any]) -> str:
    value = row.get("id")
    if not isinstance(value, (str, int)) or str(value).strip() == "":
        raise ValueError("TCM_SDT row has no usable id")
    return str(value)


def _diagnosis(row: dict[str, Any]) -> str:
    value = row.get("中医疾病诊断")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TCM_SDT row {_source_id(row)} has no 中医疾病诊断")
    return value.strip()


def _task(
    row: dict[str, Any],
    source_sha256: str,
    *,
    selection_seed: str = SAMPLE_SEED,
) -> dict[str, Any]:
    source_id = _source_id(row)
    source_text = str(row["instruction"]).strip()
    user_message = USER_PREFIX + source_text
    if not 1 <= len(source_text) <= QZBC_TEXT_LIMIT:
        raise ValueError(f"source text length violates QZBC limit for id={source_id}")
    if not 1 <= len(user_message) <= QZBC_TEXT_LIMIT:
        raise ValueError(f"user message length violates QZBC limit for id={source_id}")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": f"BEST30-{_rank('task-id', source_id, seed=selection_seed)[:24]}",
        "family_token": f"BEST-F-{_rank('family-id', source_id, seed=selection_seed)[:24]}",
        "role": "doctor",
        "variant": "clean",
        "arm": "role_aware",
        "modality": "tcm_open_case_reasoning",
        "benchmark": "TCM-BEST4SDT",
        "benchmark_version": "Figshare v8",
        "benchmark_id": source_id,
        "dataset_sha256": source_sha256,
        "source_text_private": source_text,
        "paired_case_material_sha256": _sha256_text(source_text),
        "system_prompt_private": SYSTEM_PROMPT,
        "user_message_private": user_message,
        "shared_payload_sha256": _sha256_text(user_message),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selection_seed = str(getattr(args, "selection_seed", SAMPLE_SEED) or "").strip()
    sample_label = str(getattr(args, "sample_label", SAMPLE_LABEL) or "").strip()
    exclude_manifest_arg = getattr(args, "exclude_selection_manifest", None)
    if not selection_seed:
        raise ValueError("selection_seed cannot be empty")
    if not sample_label:
        raise ValueError("sample_label cannot be empty")
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    source_sha256 = _sha256_file(data_path)
    source_md5 = _md5_file(data_path)
    if source_sha256 != EXPECTED_SOURCE_SHA256 or source_md5 != EXPECTED_SOURCE_MD5:
        raise ValueError("TCM_SDT.json differs from the checksum-pinned Figshare v8 file")
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"expected {EXPECTED_SOURCE_ROWS} source rows")
    if len({_source_id(row) for row in rows}) != len(rows):
        raise ValueError("source IDs are duplicated")

    excluded_benchmark_ids: set[str] = set()
    excluded_diagnosis_hashes: set[str] = set()
    exclusion_metadata: dict[str, Any] | None = None
    if exclude_manifest_arg is not None:
        exclude_manifest_path = Path(exclude_manifest_arg).expanduser().resolve()
        if not exclude_manifest_path.is_file():
            raise FileNotFoundError(exclude_manifest_path)
        exclude_manifest = json.loads(exclude_manifest_path.read_text(encoding="utf-8"))
        if exclude_manifest.get("status") != "FROZEN_BEFORE_ANY_MODEL_GENERATION":
            raise ValueError("exclusion manifest is not a frozen pre-generation selection")
        excluded_rows = (exclude_manifest.get("selection") or {}).get(
            "selected_rows_in_task_order"
        )
        if not isinstance(excluded_rows, list) or not excluded_rows:
            raise ValueError("exclusion manifest has no selected rows")
        for item in excluded_rows:
            if not isinstance(item, dict):
                raise ValueError("exclusion manifest has a non-object selected row")
            benchmark_id = str(item.get("benchmark_id") or "").strip()
            diagnosis_hash = str(item.get("diagnosis_sha256") or "").strip()
            if not benchmark_id or not re.fullmatch(r"[0-9a-f]{64}", diagnosis_hash):
                raise ValueError("exclusion manifest selected row is incomplete")
            excluded_benchmark_ids.add(benchmark_id)
            excluded_diagnosis_hashes.add(diagnosis_hash)
        exclusion_metadata = {
            "policy": "exclude_prior_benchmark_id_and_normalized_diagnosis_sha256",
            "source_manifest_name": exclude_manifest_path.name,
            "source_manifest_sha256": _sha256_file(exclude_manifest_path),
            "excluded_benchmark_id_count": len(excluded_benchmark_ids),
            "excluded_normalized_diagnosis_count": len(excluded_diagnosis_hashes),
        }

    overlap_paths = list(
        args.overlap_source
        or [
            _care_root().parent
            / "qzbc_source"
            / "qzbc-main"
            / "backend"
            / "app"
            / "content"
            / "data"
            / "learning_cases.json"
        ]
    )
    documents, overlap_sources = _load_local_corpus(overlap_paths)
    inverted: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        for gram in document["grams"]:
            inverted[gram].append(index)

    audit_rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for row in rows:
        source_id = _source_id(row)
        instruction = row.get("instruction")
        output = row.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"invalid instruction for source id={source_id}")
        if not isinstance(output, list) or not output:
            raise ValueError(f"invalid reference output for source id={source_id}")
        diagnosis = _diagnosis(row)
        diagnosis_hash = _sha256_text(_normalize(diagnosis))
        user_message_length = len(USER_PREFIX + instruction.strip())
        overlap = _best_overlap(instruction, documents, inverted) if documents else {
            "status": "not_run_no_local_sources",
            "exact_match": False,
            "near_duplicate": False,
            "best_jaccard": None,
            "best_containment": None,
            "best_sequence_ratio": None,
            "matched_source": None,
            "matched_record": None,
            "matched_field": None,
        }
        within_limit = len(instruction.strip()) <= QZBC_TEXT_LIMIT and user_message_length <= QZBC_TEXT_LIMIT
        excluded_by_prior_selection = (
            source_id in excluded_benchmark_ids
            or diagnosis_hash in excluded_diagnosis_hashes
        )
        is_eligible = (
            within_limit
            and not overlap["near_duplicate"]
            and not excluded_by_prior_selection
        )
        audit_row = {
                "benchmark_id": source_id,
                "diagnosis_normalized_sha256": diagnosis_hash,
                "instruction_sha256": _sha256_text(instruction.strip()),
                "instruction_length_chars": len(instruction.strip()),
                "user_message_length_chars": user_message_length,
                "within_qzbc_2000_char_limits": within_limit,
                "eligible_before_sampling": is_eligible,
                **overlap,
            }
        if exclusion_metadata is not None:
            audit_row["excluded_by_prior_frozen_selection"] = (
                excluded_by_prior_selection
            )
        audit_rows.append(audit_row)
        if is_eligible:
            eligible.append(row)

    # Select one case per diagnosis first to maximize breadth without inspecting outputs.
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[_normalize(_diagnosis(row))].append(row)
    representatives: list[dict[str, Any]] = []
    for diagnosis_key, candidates in grouped.items():
        del diagnosis_key
        representatives.append(
            min(
                candidates,
                key=lambda row: (
                    _rank(
                        "within-diagnosis",
                        _source_id(row),
                        seed=selection_seed,
                    ),
                    _source_id(row),
                ),
            )
        )
    selected = sorted(
        representatives,
        key=lambda row: (
            _rank("select-diagnosis", _source_id(row), seed=selection_seed),
            _source_id(row),
        ),
    )[: args.sample_size]
    if len(selected) != args.sample_size:
        raise ValueError(
            f"only {len(selected)} eligible diagnosis representatives for sample_size={args.sample_size}"
        )
    selected = sorted(
        selected,
        key=lambda row: (
            _rank("task-order", _source_id(row), seed=selection_seed),
            _source_id(row),
        ),
    )
    if len({_diagnosis(row) for row in selected}) != len(selected):
        raise RuntimeError("selection did not preserve one case per exact diagnosis label")

    tasks = [
        _task(row, source_sha256, selection_seed=selection_seed)
        for row in selected
    ]
    task_by_source = {task["benchmark_id"]: task for task in tasks}
    references = [
        {
            "schema_version": SCHEMA_VERSION,
            "privacy": "PRIVATE_REFERENCE_NOT_FOR_MODEL_INPUT",
            "task_id": task_by_source[_source_id(row)]["task_id"],
            "benchmark_id": _source_id(row),
            "中医疾病诊断": _diagnosis(row),
            "reference_output": row["output"],
        }
        for row in selected
    ]
    task_payload = _jsonl_bytes(tasks)
    reference_payload = _jsonl_bytes(references)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED_BEFORE_MODEL_GENERATION",
        "scope": "TCM-BEST4SDT 300 case stems versus locally available project knowledge payloads",
        "boundary": (
            "This audit detects direct/near text reuse in the enumerated local files; it cannot "
            "prove absence from foundation-model pretraining or unavailable historical artifacts."
        ),
        "algorithm": {
            "normalization": "Unicode NFKC, lowercase, retain alphanumeric characters",
            "character_ngram_size": NGRAM_SIZE,
            "near_jaccard_threshold": NEAR_JACCARD_THRESHOLD,
            "near_containment_threshold": NEAR_CONTAINMENT_THRESHOLD,
            "near_sequence_threshold": NEAR_SEQUENCE_THRESHOLD,
        },
        "local_sources": overlap_sources,
        "local_string_count": len(documents),
        "source_case_count": len(rows),
        "eligible_case_count": len(eligible),
        "excluded_exact_or_near_duplicate_count": sum(not row["eligible_before_sampling"] and row["near_duplicate"] for row in audit_rows),
        "excluded_length_count": sum(not row["within_qzbc_2000_char_limits"] for row in audit_rows),
        "selected_benchmark_ids": [_source_id(row) for row in selected],
        "rows": audit_rows,
    }
    if exclusion_metadata is not None:
        audit["prior_frozen_selection_exclusion"] = exclusion_metadata
        audit["excluded_by_prior_frozen_selection_count"] = sum(
            row.get("excluded_by_prior_frozen_selection") is True
            for row in audit_rows
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_ANY_MODEL_GENERATION",
        "sample_label": sample_label,
        "performance_based_selection": False,
        "reporting_boundary": (
            "基于TCM-BEST4SDT病例构建的固定30例开放式专家盲审轨；非官方300例完整榜单，"
            "不单独归因CARE-OPD，也不等同于临床疗效。"
        ),
        "benchmark": {
            "name": "TCM-BEST4SDT",
            "component": "TCM_SDT.json",
            "figshare_article_id": FIGSHARE_ARTICLE_ID,
            "figshare_file_id": FIGSHARE_FILE_ID,
            "doi": FIGSHARE_DOI,
            "version": 8,
            "license": SOURCE_LICENSE,
            "download_url": SOURCE_URL,
            "source_sha256": source_sha256,
            "source_md5": source_md5,
            "source_row_count": len(rows),
        },
        "selection": {
            "seed": selection_seed,
            "algorithm": SELECTION_ALGORITHM,
            "task_order_algorithm": TASK_ORDER_ALGORITHM,
            "sample_count": len(selected),
            "eligible_case_count": len(eligible),
            "eligible_normalized_diagnosis_count": len(grouped),
            "one_case_per_normalized_diagnosis": True,
            "selected_rows_in_task_order": [
                {
                    "benchmark_id": _source_id(row),
                    "diagnosis_sha256": _sha256_text(_normalize(_diagnosis(row))),
                    "within_diagnosis_rank_sha256": _rank(
                        "within-diagnosis", _source_id(row), seed=selection_seed
                    ),
                    "selection_rank_sha256": _rank(
                        "select-diagnosis", _source_id(row), seed=selection_seed
                    ),
                    "task_order_rank_sha256": _rank(
                        "task-order", _source_id(row), seed=selection_seed
                    ),
                }
                for row in selected
            ],
            "selected_id_digest": _canonical_hash([_source_id(row) for row in selected]),
        },
        "task_protocol": {
            "role": "doctor",
            "condition": "clean",
            "modality": "tcm_open_case_reasoning",
            "shared_user_payload_across_four_systems": True,
            "reference_answers_excluded_from_task_file": True,
            "temperature": 0,
            "top_p": 1.0,
            "seed": 42,
            "max_output_tokens": 1200,
            "primary_human_endpoint": "safety-qualified expert preference share",
            "comparison_scope": (
                "神农三言完整项目对比三种冻结本地公开权重模型基线；非四个同构Agent，"
                "非CARE-OPD因果消融。"
            ),
        },
        "artifacts": {
            "tasks_file": "tcm_best4sdt_agent30.tasks.private.jsonl",
            "tasks_sha256": hashlib.sha256(task_payload).hexdigest(),
            "reference_file_private": "tcm_best4sdt_agent30.reference.private.jsonl",
            "reference_sha256_private": hashlib.sha256(reference_payload).hexdigest(),
            "overlap_audit_file": "tcm_best4sdt_agent30.overlap_audit.json",
        },
    }
    if exclusion_metadata is not None:
        manifest["selection"]["prior_frozen_selection_exclusion"] = (
            exclusion_metadata
        )

    tasks_path = output_dir / manifest["artifacts"]["tasks_file"]
    reference_path = output_dir / manifest["artifacts"]["reference_file_private"]
    audit_path = output_dir / manifest["artifacts"]["overlap_audit_file"]
    manifest_path = output_dir / "tcm_best4sdt_agent30.selection_manifest.json"
    if args.run_root:
        run_root = args.run_root.expanduser().resolve()
        existing = list(run_root.glob("*/responses.private.jsonl")) if run_root.exists() else []
        if existing and not tasks_path.exists():
            raise RuntimeError("responses exist but frozen tasks are absent; refusing post-hoc selection")
    _write_frozen(tasks_path, task_payload)
    _write_frozen(reference_path, reference_payload)
    _write_frozen(audit_path, _json_bytes(audit))
    _write_frozen(manifest_path, _json_bytes(manifest))
    return {
        "status": manifest["status"],
        "sample_label": sample_label,
        "task_count": len(tasks),
        "tasks": str(tasks_path),
        "tasks_sha256": manifest["artifacts"]["tasks_sha256"],
        "reference_private": str(reference_path),
        "overlap_audit": str(audit_path),
        "selection_manifest": str(manifest_path),
        "eligible_case_count": len(eligible),
    }


def build_parser() -> argparse.ArgumentParser:
    root = _care_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "public_eval" / "tcm_best4sdt" / "data" / "TCM_SDT.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "public_eval" / "tcm_best4sdt_agent30_fixed_seed20260825",
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--selection-seed", default=SAMPLE_SEED)
    parser.add_argument("--sample-label", default=SAMPLE_LABEL)
    parser.add_argument(
        "--exclude-selection-manifest",
        type=Path,
        help=(
            "optional frozen prior selection; excludes both its benchmark IDs "
            "and normalized-diagnosis hashes before ranking"
        ),
    )
    parser.add_argument(
        "--overlap-source",
        type=Path,
        action="append",
        default=None,
        help="repeatable local JSON/JSONL corpus used only for pre-generation overlap audit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
