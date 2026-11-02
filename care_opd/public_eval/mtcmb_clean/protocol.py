#!/usr/bin/env python3
"""Pure-data contract for the contamination-aware MTCMB evaluation.

The module intentionally has no torch/transformers/metric-package imports so
that dataset, overlap, prompt and scoring-contract checks remain CPU-only.
"""

from __future__ import annotations

import ast
import copy
import functools
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "care-opd-mtcmb-clean-v1"
UPSTREAM_COMMIT = "faffd813c67fe012cef74277a23270d5dd6db9b8"
UPSTREAM_URL = "https://github.com/Wayyuanyuan/MTCMB"

MODEL_ORDER = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)

TASK_COUNTS = {
    "TCM-CHGD": 100,
    "TCM-Diagnosis": 200,
    "TCM-ED-A": 1200,
    "TCM-ED-B": 4800,
    "TCM-FRD": 200,
    "TCM-FT": 100,
    "TCM-LitData": 100,
    "TCM-MSDD": 100,
    "TCM-PR": 100,
    "TCM-SE-A": 50,
    "TCM-SE-B": 50,
    "TCMeEE": 100,
}

DATA_SHA256 = {
    "TCM-CHGD.jsonl": "a3012c7ba34fd2ad53ee40232cccd8cae78c6340ae581802d001882535c92ba9",
    "TCM-Diagnosis.jsonl": "3fdf974ce2be8e6ddd347501d2bee4272895ac4970fea092a49d98149e003543",
    "TCM-ED-A.jsonl": "a1163c53ba443d23d96c938ab3dd8b79a96b99e28524b097f4f5856f3e177fba",
    "TCM-ED-B.jsonl": "0058097e300015ed64fba25912a384c0907008a9782c38e917f2f7249be9afb5",
    "TCM-FRD.jsonl": "faf05f86991edd7bec0763e780d767f16fc1fa0de9784a5df78508efe7eac16c",
    "TCM-FT.jsonl": "6d4cfe58ef4f3958bb9d7cfbae90959547bd0672ac779184183d494d19009048",
    "TCM-LitData.jsonl": "07328d959dd1107f95fcb323a88736f4cd2dc77271aa7061b9f66eff669f2e8d",
    "TCM-MSDD.jsonl": "075042dd71d3b98e3b7a41c09480a93606c45bf68ee7832a3a65dbe2ef0029cf",
    "TCM-PR.jsonl": "cc6cbbdfc3a095a6a0208fdb9f9605d7d3511d307c6c61534c461860ecebecf5",
    "TCM-SE-A.jsonl": "6e4d58c0276e137fa32051a7f45e001392ba328698f5b5890e97c934457d227a",
    "TCM-SE-B.jsonl": "ea4da92ec41a7bec45615bde5d09588c60ae7e79df2c65ca101fd1560b02759e",
    "TCMeEE.jsonl": "437c230f8073a8607b21ee4cca4dcabe00bce48a187d787f9077dd1cfb7846a9",
}

PROMPT_SOURCE_SHA256 = {
    "TCM_CHGD.py": "eee34277b31f5893704c37abb16b648bcd1ccdfd182bc6c0c5479570097fad7c",
    "TCM_Diagnosis.py": "244a5cdefbe25a2548be86f63a85086049f6c8a813c80eca131e4f26ae5b7aa5",
    "TCM_ED_A.py": "26e45cbfa469d9a9387be8686c8dddb0c72f2805ac79d9d87b32673c753d79a9",
    "TCM_ED_B.py": "e4d4f6e805c93da7acf796713e8203b582c4025d674552123abd6c7291ca4c07",
    "TCM_FRD.py": "b13cac14a7bd50809c14cb82361a3bec68e98ca90250d31f62772bc9b1e8e1c6",
    "TCM_FT_score.py": "a902a2cb7b861bda3975a69797e9b0af09ce2a6fe34b45746e771c09e8a39363",
    "TCM_LitData_score.py": "57f09690dab8873bb5bb1df78dcdbfc09fba571d288b8dcc4b278787188fd72c",
    "TCM_MSDD.py": "a49171c49d160f951687d4bba5da01def610e8ce950fe279dd5e15d73ebe00f7",
    "TCM_PR.py": "e0a7e388909035366b3267f95ac549847944d804fbbc8236aaad225006af612e",
    "TCM_SE_B.py": "1c6e980a7a6cc20e1c91aaad4bde552bb2663c647c12124f147f2f0680be5ccc",
    "TCMeEE_score.py": "7f24ade88520ad9ec6a51add5ca1cc7ad828c85206b1629088eb8a6192cb66c6",
}

PROMPT_SOURCE_BY_TASK = {
    "TCM-CHGD": "TCM_CHGD.py",
    "TCM-Diagnosis": "TCM_Diagnosis.py",
    "TCM-ED-A": "TCM_ED_A.py",
    "TCM-ED-B": "TCM_ED_B.py",
    "TCM-FRD": "TCM_FRD.py",
    "TCM-MSDD": "TCM_MSDD.py",
    "TCM-PR": "TCM_PR.py",
    "TCM-SE-B": "TCM_SE_B.py",
    "TCMeEE": "TCMeEE_score.py",
}

REMOTE_JUDGE_TASKS = frozenset({"TCM-SE-A"})
DIRECTLY_EXPOSED_FAMILIES = frozenset({"TCM-ED-A", "TCM-ED-B"})
CLEAN9_TASKS = (
    "TCM-FT",
    "TCMeEE",
    "TCM-CHGD",
    "TCM-LitData",
    "TCM-MSDD",
    "TCM-Diagnosis",
    "TCM-PR",
    "TCM-FRD",
    "TCM-SE-B",
)
OFFLINE11_TASKS = tuple(task for task in TASK_COUNTS if task not in REMOTE_JUDGE_TASKS)
MCQ_TASKS = frozenset({"TCM-ED-A", "TCM-ED-B", "TCM-SE-B"})
EXACT_TASKS = frozenset({"TCM-ED-A", "TCM-ED-B", "TCM-MSDD", "TCM-PR", "TCM-SE-B"})
BERT_TASKS = frozenset({"TCM-FT", "TCMeEE", "TCM-CHGD", "TCM-Diagnosis", "TCM-FRD"})

EXPECTED_FULL_N = 7100
EXPECTED_OFFLINE11_N = 7050
EXPECTED_EXPOSED_UNIQUE = 1062
EXPECTED_EXPOSED_WITH_REPETITION = 1200
EXPECTED_EXPOSED_BY_TASK = {"TCM-ED-A": 231, "TCM-ED-B": 831}
EXPECTED_OVERLAP_REPORT_SHA256 = "5f4e14b13427f12e6658cc1769687da857f66452e071b1e1eeabd6d4ac787968"
EXPECTED_OVERLAP_NONEMPTY_QUESTIONS = 6600
EXPECTED_EXACT_STEM_OVERLAP = 1096
EXPECTED_SAME_ITEM_OVERLAP = 1068
EXPECTED_VISIBLE_SEGMENT_OVERLAP = 1041
EXPECTED_NEAR_STEM_OVERLAP = 2
EXPECTED_OVERLAP_EXCLUDED = 1186
EXPECTED_OVERLAP_EXCLUDED_BY_TASK = {"TCM-ED-A": 249, "TCM-ED-B": 937}
EXPECTED_OVERLAP_ADDITIONAL = 124
EXPECTED_OVERLAP_CLEAN_ALL12_N = 5914
EXPECTED_OVERLAP_CLEAN_OFFLINE11_N = 5864
EXPECTED_CLEAN9_N = 1050
EXPECTED_LITDATA_CALLS = 306
EXPECTED_CLEAN_INFERENCE_CALLS = 7256
CONTROLLED_PERTURBATION_EXCLUDED_IDS = frozenset(
    {"TCM-ED-B:108", "TCM-ED-B:109"}
)
EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS = 4862
EXPECTED_CLEAN9_MCQ_PAIRS = 50

OVERLAP_NON_TEXT = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
OVERLAP_LEADING_NUMBER = re.compile(
    r"^\s*(?:\[?\d+\]?\s*[\.．、:)）]\s*|[（(]\d+[）)]\s*)+"
)
OVERLAP_OPTION_START = re.compile(r"\n\s*[AＡ][\.．、:：]\s*")
OVERLAP_OPTION_LINE = re.compile(r"(?:^|\n)\s*([A-EＡ-Ｅ])[\.．、:：]\s*([^\n]+)")
OVERLAP_OPTION_LABEL = re.compile(r"^\s*[A-EＡ-Ｅ][\.．、:：]?\s*")

SYSTEM_PROMPT = "你是一个专业中医医生，能够准确全面的解答中医问题。本次对话，均只采用中文提问和回答。"
REMOTE_CHAT_SOURCE_SHA256 = "9fe025c94916b167f01b68c886b36d66334aa3bfbccfd0ecb3e18eb73cca43d3"
FT_FALLBACK_PROMPT_VERSION = "mtcmb-ft-zero-shot-fallback-v1"
MSDD_ALIASES = {
    "证型": {
        "气虚血瘀": "气虚血瘀证",
        "痰瘀互结": "痰瘀互结证",
        "气阴两虚": "气阴两虚证",
        "气滞血瘀": "气滞血瘀证",
        "肝阳上亢": "肝阳上亢证",
        "阴虚阳亢": "阴虚阳亢证",
        "痰热蕴结": "痰热蕴结证",
        "痰湿痹阻": "痰湿痹阻证",
        "阳虚水停": "阳虚水停证",
        "肝肾阴虚": "肝肾阴虚证",
    },
    "疾病": {
        "胸痹心痛": "胸痹心痛病",
        "心衰": "心衰病",
        "眩晕": "眩晕病",
        "心悸": "心悸病",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return "".join(ch for ch in text if ch.isalnum())


def overlap_norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return OVERLAP_NON_TEXT.sub("", text)


def overlap_stem(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = OVERLAP_LEADING_NUMBER.sub("", text)
    found = OVERLAP_OPTION_START.search(text)
    if found:
        text = text[: found.start()]
    return overlap_norm(text)


def overlap_option_set(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        parts = value.values()
    elif isinstance(value, list):
        parts = value
    else:
        parts = [match.group(2) for match in OVERLAP_OPTION_LINE.finditer(str(value or ""))]
    return {
        overlap_norm(OVERLAP_OPTION_LABEL.sub("", str(part)))
        for part in parts
        if overlap_norm(OVERLAP_OPTION_LABEL.sub("", str(part)))
    }


def _overlap_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _overlap_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _overlap_strings(item)


def _overlap_ngrams(value: str, n: int = 13) -> set[str]:
    if len(value) < n:
        return {value} if value else set()
    return {value[index : index + n] for index in range(len(value) - n + 1)}


def compute_overlap_audit(
    records: Mapping[str, Sequence[Mapping[str, Any]]], train_seed_file: Path
) -> dict[str, Any]:
    """Recompute the result-blind audit encoded by the local overlap report."""

    train_rows = load_jsonl(train_seed_file.resolve(strict=True))
    if len(train_rows) != 9600:
        raise ValueError(f"formal training seed count changed: {len(train_rows)}")
    quiz: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        if row.get("metadata", {}).get("task_type") != "student_quiz_explanation":
            continue
        asset = next(
            (
                item
                for item in row.get("metadata", {}).get("curated_assets", [])
                if item.get("source_id") == "SRC_MTCMB"
            ),
            None,
        )
        if not asset:
            continue
        source_id = str(asset.get("source_record_id") or "")
        parse_source_record_id(source_id)
        constraints = row.get("request", {}).get("constraints", {})
        quiz.setdefault(
            source_id,
            {
                "question": str(constraints.get("question") or ""),
                "options": constraints.get("options") or [],
            },
        )

    official_rows = [
        {
            "id": record_key(task, row["id"]),
            "question": str(row.get("question") or ""),
            "options": row.get("options"),
        }
        for task, task_rows in records.items()
        for row in task_rows
    ]
    official_ids = {row["id"] for row in official_rows}
    direct = set(quiz) & official_ids
    train_stems: dict[str, list[str]] = defaultdict(list)
    train_ngrams: dict[str, set[str]] = {}
    inverted: dict[str, set[str]] = defaultdict(set)
    for source_id, row in quiz.items():
        value = overlap_stem(row["question"])
        if len(value) < 8:
            continue
        train_stems[value].append(source_id)
        grams = _overlap_ngrams(value)
        train_ngrams[source_id] = grams
        for gram in grams:
            inverted[gram].add(source_id)
    visible_segments = {
        overlap_norm(text)
        for row in train_rows
        for text in _overlap_strings(
            {"request": row.get("request"), "target": row.get("target")}
        )
        if len(overlap_norm(text)) >= 8
    }

    exact: set[str] = set()
    same_item: set[str] = set()
    segment: set[str] = set()
    near: set[str] = set()
    nonempty = 0
    for row in official_rows:
        raw = row["question"]
        value = overlap_stem(raw)
        raw_norm = overlap_norm(raw)
        if not value:
            continue
        nonempty += 1
        is_exact = value in train_stems
        if is_exact:
            exact.add(row["id"])
            options = overlap_option_set(row["options"])
            best = 0.0
            if len(options) >= 2:
                for source_id in train_stems[value]:
                    candidate = overlap_option_set(quiz[source_id]["options"])
                    union = options | candidate
                    best = max(best, len(options & candidate) / len(union) if union else 0.0)
            if best >= 0.8:
                same_item.add(row["id"])
        if raw_norm in visible_segments:
            segment.add(row["id"])

        grams = _overlap_ngrams(value)
        candidates: set[str] = set()
        for gram in grams:
            candidates.update(inverted.get(gram, ()))
        best_score: tuple[float, float] | None = None
        for source_id in candidates:
            other = train_ngrams[source_id]
            intersection = len(grams & other)
            union = len(grams | other)
            score = (
                intersection / union if union else 0.0,
                intersection / min(len(grams), len(other)) if grams and other else 0.0,
            )
            if best_score is None or score > best_score:
                best_score = score
        if (
            best_score
            and not is_exact
            and best_score[0] >= 0.8
            and best_score[1] >= 0.9
            and len(value) >= 16
        ):
            near.add(row["id"])

    overlap_excluded = direct | exact | near
    counts = {
        "nonempty_questions": nonempty,
        "direct_provenance": len(direct),
        "exact_normalized_question_matches": len(exact),
        "high_confidence_same_item_matches": len(same_item),
        "exact_visible_segment_matches": len(segment),
        "near_question_matches": len(near),
        "union_direct_exact_near": len(overlap_excluded),
    }
    expected = {
        "nonempty_questions": EXPECTED_OVERLAP_NONEMPTY_QUESTIONS,
        "direct_provenance": EXPECTED_EXPOSED_UNIQUE,
        "exact_normalized_question_matches": EXPECTED_EXACT_STEM_OVERLAP,
        "high_confidence_same_item_matches": EXPECTED_SAME_ITEM_OVERLAP,
        "exact_visible_segment_matches": EXPECTED_VISIBLE_SEGMENT_OVERLAP,
        "near_question_matches": EXPECTED_NEAR_STEM_OVERLAP,
        "union_direct_exact_near": EXPECTED_OVERLAP_EXCLUDED,
    }
    if counts != expected:
        raise ValueError(f"MTCMB overlap audit changed: {counts} != {expected}")
    by_task = Counter(parse_source_record_id(value)[0] for value in overlap_excluded)
    if dict(by_task) != EXPECTED_OVERLAP_EXCLUDED_BY_TASK:
        raise ValueError(f"MTCMB overlap union task distribution changed: {dict(by_task)}")
    additional = overlap_excluded - direct
    if len(additional) != EXPECTED_OVERLAP_ADDITIONAL:
        raise ValueError("MTCMB additional overlap exclusion count changed")
    if not segment <= overlap_excluded:
        raise ValueError("visible-segment overlaps are not covered by the exclusion union")
    return {
        "counts": counts,
        "direct_provenance_source_record_ids": sorted(direct),
        "additional_exact_or_near_record_ids": sorted(additional),
        "overlap_excluded_record_ids": sorted(overlap_excluded),
        "overlap_excluded_by_task": dict(sorted(by_task.items())),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def verify_upstream_repo(repo: Path, *, verify_git: bool = True) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    data_root = repo / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"MTCMB data directory is missing: {data_root}")
    if verify_git:
        head_path = repo / ".git"
        if not head_path.exists():
            raise ValueError(f"pinned MTCMB checkout must retain git metadata: {repo}")
        import subprocess

        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != UPSTREAM_COMMIT:
            raise ValueError(f"MTCMB HEAD {head} != pinned {UPSTREAM_COMMIT}")
    files: dict[str, Any] = {}
    total = 0
    for filename, expected_hash in DATA_SHA256.items():
        path = data_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"MTCMB hash mismatch for {filename}: {actual_hash}")
        rows = load_jsonl(path)
        task = path.stem
        if len(rows) != TASK_COUNTS[task]:
            raise ValueError(f"MTCMB count mismatch for {task}: {len(rows)}")
        ids = [int(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs in {filename}")
        files[filename] = {"sha256": actual_hash, "records": len(rows)}
        total += len(rows)
    if total != EXPECTED_FULL_N:
        raise ValueError(f"unexpected MTCMB total: {total}")
    prompt_files: dict[str, Any] = {}
    prompt_root = repo / "make_answer" / "works"
    for filename, expected_hash in PROMPT_SOURCE_SHA256.items():
        path = prompt_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"MTCMB prompt source hash mismatch for {filename}: {actual_hash}")
        prompt_files[filename] = {"sha256": actual_hash}
    remote_chat_source = repo / "make_answer" / "chat" / "remote" / "openai_api.py"
    if sha256_file(remote_chat_source) != REMOTE_CHAT_SOURCE_SHA256:
        raise ValueError("MTCMB official remote-chat system prompt source changed")
    if SYSTEM_PROMPT not in remote_chat_source.read_text(encoding="utf-8"):
        raise ValueError("configured MTCMB system prompt is not present in pinned source")
    litdata_rows = load_jsonl(data_root / "TCM-LitData.jsonl")
    litdata_calls = sum(len(row.get("annotations") or []) for row in litdata_rows)
    if litdata_calls != EXPECTED_LITDATA_CALLS:
        raise ValueError(f"unexpected TCM-LitData annotation calls: {litdata_calls}")
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "records": total,
        "task_counts": dict(sorted(TASK_COUNTS.items())),
        "litdata_annotation_calls": litdata_calls,
        "files": files,
        "prompt_sources": prompt_files,
        "remote_chat_source": {
            "path": "make_answer/chat/remote/openai_api.py",
            "sha256": REMOTE_CHAT_SOURCE_SHA256,
        },
    }


def load_all_records(repo: Path) -> dict[str, list[dict[str, Any]]]:
    data_root = repo.resolve() / "data"
    return {
        task: load_jsonl(data_root / f"{task}.jsonl")
        for task in sorted(TASK_COUNTS)
    }


def parse_source_record_id(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"(TCM-[A-Za-z-]+):(\d+)", str(value).strip())
    if not match:
        raise ValueError(f"invalid MTCMB source_record_id: {value!r}")
    return match.group(1), int(match.group(2))


def extract_train_exposure(train_seed_file: Path) -> dict[str, Any]:
    unique: set[str] = set()
    repeated: list[str] = []
    rows = 0
    with train_seed_file.resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            seed = json.loads(line)
            assets = seed.get("metadata", {}).get("curated_assets", [])
            for asset in assets:
                if asset.get("source_id") != "SRC_MTCMB":
                    continue
                source_record_id = str(asset.get("source_record_id") or "")
                parse_source_record_id(source_record_id)
                repeated.append(source_record_id)
                unique.add(source_record_id)
    by_task = Counter(parse_source_record_id(value)[0] for value in unique)
    if len(unique) != EXPECTED_EXPOSED_UNIQUE:
        raise ValueError(f"expected {EXPECTED_EXPOSED_UNIQUE} unique MTCMB records, got {len(unique)}")
    if len(repeated) != EXPECTED_EXPOSED_WITH_REPETITION:
        raise ValueError(
            f"expected {EXPECTED_EXPOSED_WITH_REPETITION} repeated MTCMB uses, got {len(repeated)}"
        )
    if dict(by_task) != EXPECTED_EXPOSED_BY_TASK:
        raise ValueError(f"unexpected MTCMB exposure families: {dict(by_task)}")
    return {
        "train_seed_file_sha256": sha256_file(train_seed_file),
        "train_seed_rows": rows,
        "source_id": "SRC_MTCMB",
        "uses_with_repetition": len(repeated),
        "unique_source_record_ids": len(unique),
        "unique_by_task": dict(sorted(by_task.items())),
        "source_record_ids": sorted(unique),
    }


def validate_exclusion_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != "care-opd-mtcmb-overlap-exclusions-v1":
        raise ValueError("unsupported MTCMB overlap exclusion schema")
    ids = list(manifest.get("source_record_ids") or [])
    if manifest.get("source_record_ids_scope") != "direct_provenance_only":
        raise ValueError("MTCMB direct-provenance source ID scope is not explicit")
    if len(ids) != EXPECTED_EXPOSED_UNIQUE or len(ids) != len(set(ids)):
        raise ValueError("invalid MTCMB direct-provenance cardinality")
    counts = Counter(parse_source_record_id(value)[0] for value in ids)
    if dict(counts) != EXPECTED_EXPOSED_BY_TASK:
        raise ValueError(f"invalid MTCMB direct provenance by task: {dict(counts)}")
    additional = list(manifest.get("overlap_additional_record_ids") or [])
    if len(additional) != EXPECTED_OVERLAP_ADDITIONAL or len(additional) != len(
        set(additional)
    ):
        raise ValueError("invalid MTCMB exact/near additional exclusion cardinality")
    if set(ids) & set(additional):
        raise ValueError("direct and additional MTCMB overlap exclusions are not disjoint")
    excluded = set(ids) | set(additional)
    if len(excluded) != EXPECTED_OVERLAP_EXCLUDED:
        raise ValueError("invalid MTCMB overlap-clean exclusion union cardinality")
    excluded_counts = Counter(
        parse_source_record_id(value)[0] for value in excluded
    )
    if dict(excluded_counts) != EXPECTED_OVERLAP_EXCLUDED_BY_TASK:
        raise ValueError(
            f"invalid MTCMB overlap-clean exclusions by task: {dict(excluded_counts)}"
        )
    if int(manifest.get("overlap_excluded_records", -1)) != EXPECTED_OVERLAP_EXCLUDED:
        raise ValueError("invalid MTCMB overlap exclusion count")
    if manifest.get("overlap_excluded_by_task") != EXPECTED_OVERLAP_EXCLUDED_BY_TASK:
        raise ValueError("invalid MTCMB overlap exclusion task distribution")
    audit = manifest.get("overlap_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("missing MTCMB overlap audit contract")
    if audit.get("source_overlap_report_sha256") != EXPECTED_OVERLAP_REPORT_SHA256:
        raise ValueError("invalid MTCMB source overlap report fingerprint")
    expected_audit_counts = {
        "nonempty_questions": EXPECTED_OVERLAP_NONEMPTY_QUESTIONS,
        "direct_provenance": EXPECTED_EXPOSED_UNIQUE,
        "exact_normalized_question_matches": EXPECTED_EXACT_STEM_OVERLAP,
        "high_confidence_same_item_matches": EXPECTED_SAME_ITEM_OVERLAP,
        "exact_visible_segment_matches": EXPECTED_VISIBLE_SEGMENT_OVERLAP,
        "near_question_matches": EXPECTED_NEAR_STEM_OVERLAP,
        "union_direct_exact_near": EXPECTED_OVERLAP_EXCLUDED,
    }
    if audit.get("counts") != expected_audit_counts:
        raise ValueError("invalid MTCMB overlap audit counts")
    if audit.get("result_blind_exclusion_rule") != "direct_provenance UNION exact_normalized_question_stem UNION preregistered_near_question_threshold":
        raise ValueError("invalid MTCMB result-blind overlap exclusion rule")
    if audit.get("exact_visible_segment_subset_of_union") is not True:
        raise ValueError("MTCMB visible-segment coverage is not asserted")
    if int(manifest.get("overlap_clean_all12_records", -1)) != EXPECTED_OVERLAP_CLEAN_ALL12_N:
        raise ValueError("invalid overlap-clean all12 count")
    if int(manifest.get("overlap_clean_offline11_records", -1)) != EXPECTED_OVERLAP_CLEAN_OFFLINE11_N:
        raise ValueError("invalid overlap-clean offline11 count")
    if list(manifest.get("clean9_tasks") or []) != list(CLEAN9_TASKS):
        raise ValueError("invalid Clean9 task definition")
    if int(manifest.get("clean9_records", -1)) != EXPECTED_CLEAN9_N:
        raise ValueError("invalid Clean9 count")
    controlled = manifest.get("controlled_option_permutation")
    if not isinstance(controlled, Mapping):
        raise ValueError("missing controlled option-permutation contract")
    if int(controlled.get("overlap_clean_mcq_pairs", -1)) != EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS:
        raise ValueError("invalid overlap-clean controlled pair count")
    if int(controlled.get("strict_clean9_mcq_pairs", -1)) != EXPECTED_CLEAN9_MCQ_PAIRS:
        raise ValueError("invalid Clean9 controlled pair count")
    if set(controlled.get("excluded_malformed_option_inventory_ids") or ()) != set(
        CONTROLLED_PERTURBATION_EXCLUDED_IDS
    ):
        raise ValueError("invalid malformed-option exclusion set")


def load_exclusions(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    validate_exclusion_manifest(value)
    return value


def overlap_excluded_ids(manifest: Mapping[str, Any]) -> set[str]:
    """Return the frozen result-blind union used by the overlap-clean view."""

    validate_exclusion_manifest(manifest)
    return set(manifest["source_record_ids"]) | set(
        manifest["overlap_additional_record_ids"]
    )


def record_key(task: str, record_id: Any) -> str:
    return f"{task}:{int(record_id)}"


def record_views(task: str, record_id: Any, excluded: set[str]) -> tuple[str, ...]:
    key = record_key(task, record_id)
    views = ["mtcmb_full"]
    if key not in excluded:
        views.append("overlap_clean")
    if task in CLEAN9_TASKS:
        views.append("provenance_clean9")
    return tuple(views)


def _restricted_fstring_value(node: ast.AST, row: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id == "data":
        return row
    if isinstance(node, ast.Subscript):
        owner = _restricted_fstring_value(node.value, row)
        key = _restricted_fstring_value(node.slice, row)
        if owner is not row or not isinstance(key, str):
            raise ValueError("unsupported upstream prompt subscript")
        return row[key]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
        and not node.keywords
    ):
        return str(_restricted_fstring_value(node.args[0], row))
    raise ValueError(f"unsupported upstream prompt expression: {ast.dump(node)}")


def _render_restricted_joined_string(node: ast.JoinedStr, row: Mapping[str, Any]) -> str:
    pieces = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            pieces.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            rendered = _restricted_fstring_value(value.value, row)
            if value.conversion == 114:
                rendered = repr(rendered)
            elif value.conversion == 97:
                rendered = ascii(rendered)
            elif value.conversion in (-1, 115):
                rendered = str(rendered)
            else:
                raise ValueError("unsupported upstream f-string conversion")
            if value.format_spec is not None:
                raise ValueError("upstream zero-shot prompt unexpectedly uses a format spec")
            pieces.append(rendered)
        else:
            raise ValueError("unsupported node in upstream zero-shot prompt")
    return "".join(pieces)


@functools.lru_cache(maxsize=None)
def _zero_shot_prompt_node(source_path_text: str, expected_hash: str) -> ast.JoinedStr:
    source_path = Path(source_path_text)
    if sha256_file(source_path) != expected_hash:
        raise ValueError(f"pinned prompt source changed: {source_path}")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "zero_shot_prompt"
            for target in node.targets
        ):
            candidates.append(node.value)
    if len(candidates) != 1 or not isinstance(candidates[0], ast.JoinedStr):
        raise RuntimeError(f"cannot uniquely locate upstream zero-shot prompt in {source_path}")
    return candidates[0]


def _capture_official_prompt(repo: Path, task: str, row: Mapping[str, Any]) -> str:
    if task == "TCM-FT":
        return (
            "请回答下面的中医知识问题。只输出采分要点，每条要点单独一行并以“- ”开头，"
            "不要输出推理过程、Markdown代码块或资料外内容。\n问题："
            + str(row.get("question") or "").strip()
        )

    filename = PROMPT_SOURCE_BY_TASK.get(task)
    if not filename:
        raise ValueError(f"no pinned zero-shot prompt source for {task}")
    source_path = repo.resolve() / "make_answer" / "works" / filename
    node = _zero_shot_prompt_node(str(source_path), PROMPT_SOURCE_SHA256[filename])
    return _render_restricted_joined_string(node, row)


def parse_mcq_options(task: str, row: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, str]]:
    if task == "TCM-ED-A":
        raw = row.get("options")
        if not isinstance(raw, Mapping):
            raise ValueError(f"{task}:{row.get('id')} options are not an object")
        labels = tuple(str(key).strip().upper() for key in raw)
        texts = {str(key).strip().upper(): str(value).strip() for key, value in raw.items()}
    elif task == "TCM-ED-B":
        raw = row.get("options")
        if not isinstance(raw, list):
            raise ValueError(f"{task}:{row.get('id')} options are not a list")
        pairs = []
        for position, value in enumerate(raw):
            match = re.match(r"\s*([A-Ea-e])[\.、:：\s]+(.*)\Z", str(value), re.S)
            if not match:
                raise ValueError(f"cannot parse {task}:{row.get('id')} option {value!r}")
            # The official data has two records (IDs 108 and 109) whose fourth
            # positional choice is mistyped as another "C".  The benchmark
            # answer space is positional A--E, so clean scoring still uses the
            # position; those two records are separately excluded from the
            # derived permutation assay.  Clean prompts retain upstream bytes.
            pairs.append((chr(ord("A") + position), match.group(2).strip()))
        labels = tuple(label for label, _ in pairs)
        texts = dict(pairs)
    elif task == "TCM-SE-B":
        raw = str(row.get("options") or "")
        matches = list(re.finditer(r"(?:^|\s)([A-Da-d])[\.、:：]\s*", raw))
        pairs = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            pairs.append((match.group(1).upper(), raw[match.end() : end].strip()))
        labels = tuple(label for label, _ in pairs)
        texts = dict(pairs)
    else:
        raise ValueError(f"not an MTCMB MCQ task: {task}")
    expected = tuple(chr(ord("A") + index) for index in range(len(labels)))
    if labels != expected or not 4 <= len(labels) <= 5 or any(not texts[label] for label in labels):
        raise ValueError(f"invalid option inventory for {task}:{row.get('id')}: {labels}")
    return labels, texts


def deterministic_permutation(task: str, record_id: Any, labels: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(labels)
    seed = hashlib.sha256(f"{PROTOCOL_VERSION}|option-permutation|{task}:{int(record_id)}".encode()).digest()
    ranked = tuple(sorted(labels, key=lambda label: hashlib.sha256(seed + label.encode()).digest()))
    if ranked == labels:
        ranked = labels[1:] + labels[:1]
    if set(ranked) != set(labels) or ranked == labels:
        raise AssertionError("failed to construct a nonidentity permutation")
    return ranked


def perturb_mcq_row(task: str, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    labels, texts = parse_mcq_options(task, row)
    order = deterministic_permutation(task, row["id"], labels)
    canonical_to_displayed = {
        canonical: labels[position] for position, canonical in enumerate(order)
    }
    displayed = {labels[position]: texts[canonical] for position, canonical in enumerate(order)}
    result = copy.deepcopy(dict(row))
    if task == "TCM-ED-A":
        result["options"] = displayed
    elif task == "TCM-ED-B":
        result["options"] = [f"{label}.{displayed[label]}" for label in labels]
    else:
        result["options"] = " ".join(f"{label}. {displayed[label]}" for label in labels)
    gold = str(row.get("answer") or "").strip().upper()
    if gold not in canonical_to_displayed:
        raise ValueError(f"invalid gold label for {task}:{row.get('id')}: {gold!r}")
    result["answer"] = canonical_to_displayed[gold]
    return result, canonical_to_displayed


@dataclass(frozen=True)
class InferenceItem:
    item_id: str
    task: str
    record_id: int
    annotation_index: int | None
    condition: str
    prompt: str
    max_new_tokens: int
    canonical_to_displayed: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "task": self.task,
            "record_id": self.record_id,
            "annotation_index": self.annotation_index,
            "condition": self.condition,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "canonical_to_displayed": dict(self.canonical_to_displayed),
        }


def build_inference_items(
    repo: Path,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    exclusions: Mapping[str, Any],
) -> list[InferenceItem]:
    excluded = overlap_excluded_ids(exclusions)
    items: list[InferenceItem] = []
    for task in sorted(OFFLINE11_TASKS):
        for row in records[task]:
            rid = int(row["id"])
            if task == "TCM-LitData":
                for annotation_index, annotation in enumerate(row["annotations"]):
                    prompt = (
                        f"这是一段经典文献，其内容是{row['text']}。\n"
                        "你只能摘取文献中的原文进行回答，每个回答只有1到2句话。"
                        f"{annotation['Q']}"
                    )
                    items.append(
                        InferenceItem(
                            item_id=f"clean/{task}:{rid}/annotation:{annotation_index}",
                            task=task,
                            record_id=rid,
                            annotation_index=annotation_index,
                            condition="clean",
                            prompt=prompt,
                            max_new_tokens=128,
                            canonical_to_displayed=(),
                        )
                    )
                continue
            max_tokens = 8 if task in MCQ_TASKS else 768
            items.append(
                InferenceItem(
                    item_id=f"clean/{task}:{rid}",
                    task=task,
                    record_id=rid,
                    annotation_index=None,
                    condition="clean",
                    prompt=_capture_official_prompt(repo, task, row),
                    max_new_tokens=max_tokens,
                    canonical_to_displayed=(),
                )
            )
            key = record_key(task, rid)
            if (
                task in MCQ_TASKS
                and key not in excluded
                and key not in CONTROLLED_PERTURBATION_EXCLUDED_IDS
            ):
                perturbed, mapping = perturb_mcq_row(task, row)
                items.append(
                    InferenceItem(
                        item_id=f"option_permutation/{task}:{rid}",
                        task=task,
                        record_id=rid,
                        annotation_index=None,
                        condition="option_permutation",
                        prompt=_capture_official_prompt(repo, task, perturbed),
                        max_new_tokens=8,
                        canonical_to_displayed=tuple(sorted(mapping.items())),
                    )
                )
    clean_calls = sum(item.condition == "clean" for item in items)
    pair_calls = sum(item.condition == "option_permutation" for item in items)
    if clean_calls != EXPECTED_CLEAN_INFERENCE_CALLS:
        raise ValueError(f"unexpected clean inference calls: {clean_calls}")
    if pair_calls != EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS:
        raise ValueError(f"unexpected controlled MCQ calls: {pair_calls}")
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("duplicate MTCMB inference item IDs")
    return items


def parse_choice(raw: str, allowed: Sequence[str]) -> tuple[str, bool]:
    allowed_set = set(allowed)
    text = unicodedata.normalize("NFKC", str(raw or "")).strip()
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.I).strip()
    strict = re.fullmatch(r"[\s\[\](){}<>《》【】'\"`]*([A-Ea-e])[\s\[\](){}<>《》【】'\"`\.。,:：;；!！?？]*", text)
    if strict and strict.group(1).upper() in allowed_set:
        return strict.group(1).upper(), True
    explicit = re.search(r"(?:答案|选择|选项|answer)\s*(?:是|为|[:：])?\s*([A-Ea-e])\b", text, re.I)
    if explicit and explicit.group(1).upper() in allowed_set:
        return explicit.group(1).upper(), False
    isolated = [match.upper() for match in re.findall(r"(?<![A-Za-z])([A-Ea-e])(?![A-Za-z])", text)]
    isolated = [value for value in isolated if value in allowed_set]
    return (isolated[-1], False) if isolated else ("", False)


def parse_official_choice(raw: str, allowed: Sequence[str]) -> str:
    """Reproduce the upstream make-answer parser (first allowed Latin letter)."""

    allowed_set = {str(value).upper() for value in allowed}
    text = str(raw or "")
    character_class = "".join(
        sorted({value for label in allowed_set for value in (label, label.lower())})
    )
    match = re.search(f"([{re.escape(character_class)}])", text)
    if not match:
        return ""
    value = match.group(1).upper()
    return value if value in allowed_set else ""


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value or "").strip()
        fence = re.search(r"```(?:json|python)?\s*([\s\S]*?)\s*```", text, re.I)
        if fence:
            text = fence.group(1).strip()
        raw_items = []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                raw_items = parsed
        except (ValueError, SyntaxError):
            pass
        if not raw_items:
            text = text.strip("[]")
            raw_items = re.split(r"[,，、;；\n]+", text)
    result = []
    for item in raw_items:
        normalized = unicodedata.normalize("NFKC", str(item)).strip(" \t\r\n'\"`*-•")
        if normalized:
            result.append(normalized)
    return result


def msdd_vocab(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = {"证型": set(), "疾病": set()}
    for row in rows:
        for key in values:
            for value in str(row["answer"].get(key) or "").split("|"):
                if value.strip():
                    values[key].add(value.strip())
    expected = {key: set(aliases.values()) for key, aliases in MSDD_ALIASES.items()}
    if values != expected:
        raise ValueError(f"MTCMB MSDD label vocabulary changed: {values}")
    return {key: tuple(sorted(group, key=len, reverse=True)) for key, group in values.items()}


def extract_msdd_prediction(raw: str, vocab: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    text = unicodedata.normalize("NFKC", str(raw or ""))
    result: dict[str, set[str]] = {}
    for key, terms in vocab.items():
        allowed = set(terms)
        result[key] = {
            canonical
            for alias, canonical in MSDD_ALIASES[key].items()
            if canonical in allowed and alias in text
        }
    return result


def exact_task_success(task: str, row: Mapping[str, Any], raw: str, *, vocab: Mapping[str, Sequence[str]]) -> bool:
    if task in MCQ_TASKS:
        allowed, _ = parse_mcq_options(task, row)
        pred = parse_official_choice(raw, allowed)
        return bool(pred) and pred == str(row.get("answer") or "").strip().upper()
    if task == "TCM-MSDD":
        pred = extract_msdd_prediction(raw, vocab)
        for key in ("证型", "疾病"):
            gold = {value.strip() for value in str(row["answer"].get(key) or "").split("|") if value.strip()}
            if pred[key] != gold:
                return False
        return True
    if task == "TCM-PR":
        gold = {normalized_text(value) for value in parse_list(row.get("answer"))}
        pred = {normalized_text(value) for value in parse_list(raw)}
        return bool(gold) and pred == gold
    raise ValueError(f"task has no preregistered binary exact endpoint: {task}")


def unified_na(*, clean_task_success_rate: float | None = None, paired_units: int = 0) -> dict[str, Any]:
    return {
        "clean_task_success_rate": clean_task_success_rate,
        "perturbed_task_success_rate": None,
        "conditional_retention": None,
        "regression_rate": None,
        "robustness_drop_pp": None,
        "prediction_consistency": None,
        "worst_perturbation_tsr": None,
        "clean_correct_perturbed_wrong": None,
        "clean_wrong_perturbed_correct": None,
        "paired_units": paired_units,
    }


def controlled_metrics(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(pairs)
    if not n:
        return unified_na(paired_units=0)
    clean = [bool(row["clean_task_success"]) for row in pairs]
    perturbed = [bool(row["perturbed_task_success"]) for row in pairs]
    regressions = sum(c and not p for c, p in zip(clean, perturbed, strict=True))
    improvements = sum(not c and p for c, p in zip(clean, perturbed, strict=True))
    clean_correct = sum(clean)
    clean_rate = sum(clean) / n
    perturbed_rate = sum(perturbed) / n
    return {
        "clean_task_success_rate": clean_rate,
        "perturbed_task_success_rate": perturbed_rate,
        "conditional_retention": (
            sum(c and p for c, p in zip(clean, perturbed, strict=True)) / clean_correct
            if clean_correct
            else None
        ),
        "regression_rate": regressions / clean_correct if clean_correct else None,
        "robustness_drop_pp": 100.0 * (clean_rate - perturbed_rate),
        "prediction_consistency": sum(
            bool(row.get("clean_parse_ok"))
            and bool(row.get("perturbed_parse_ok"))
            and bool(row.get("clean_canonical_prediction"))
            and bool(row.get("perturbed_canonical_prediction"))
            and row["clean_canonical_prediction"]
            == row["perturbed_canonical_prediction"]
            for row in pairs
        ) / n,
        "worst_perturbation_tsr": perturbed_rate,
        "clean_correct_perturbed_wrong": regressions,
        "clean_wrong_perturbed_correct": improvements,
        "paired_units": n,
    }


def controlled_rate_intervals(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pre-registered interval report for the four primary paired rates."""

    clean = [bool(row["clean_task_success"]) for row in pairs]
    perturbed = [bool(row["perturbed_task_success"]) for row in pairs]
    clean_correct = sum(clean)
    both_correct = sum(c and p for c, p in zip(clean, perturbed, strict=True))
    regressions = sum(c and not p for c, p in zip(clean, perturbed, strict=True))
    return {
        "method": "Wilson score interval",
        "confidence_level": 0.95,
        "z": 1.959963984540054,
        "clean_task_success_rate": wilson_rate(sum(clean), len(clean)),
        "perturbed_task_success_rate": wilson_rate(sum(perturbed), len(perturbed)),
        "conditional_retention": wilson_rate(both_correct, clean_correct),
        "regression_rate": wilson_rate(regressions, clean_correct),
    }


def wilson_rate(numerator: int, denominator: int) -> dict[str, Any]:
    """Return count, estimate, percentage, and a two-sided 95% Wilson CI."""

    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("invalid binomial numerator/denominator")
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "estimate": None,
            "estimate_percent": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci95_percent": [None, None],
        }
    z = 1.959963984540054
    estimate = numerator / denominator
    z2 = z * z
    scale = 1.0 + z2 / denominator
    center = (estimate + z2 / (2.0 * denominator)) / scale
    half = (
        z
        * (
            estimate * (1.0 - estimate) / denominator
            + z2 / (4.0 * denominator * denominator)
        )
        ** 0.5
        / scale
    )
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": estimate,
        "estimate_percent": 100.0 * estimate,
        "ci95_low": low,
        "ci95_high": high,
        "ci95_percent": [100.0 * low, 100.0 * high],
    }


def dataset_contract_fingerprint(repo: Path, exclusions_path: Path) -> str:
    return canonical_hash(
        {
            "protocol_version": PROTOCOL_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "data_sha256": DATA_SHA256,
            "exclusions_sha256": sha256_file(exclusions_path),
            "clean9_tasks": CLEAN9_TASKS,
            "ft_prompt_fallback": FT_FALLBACK_PROMPT_VERSION,
        }
    )
