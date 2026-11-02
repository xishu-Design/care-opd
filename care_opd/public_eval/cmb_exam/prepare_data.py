#!/usr/bin/env python3
"""Prepare and verify the checksum-pinned CMB-Exam public test bundle.

The overlap-clean exclusion manifest is derived from the existing formal
QZBC overlap report.  IDs are never copied from a hand-written list: every
reported example is resolved against the current CMB question file and its
normalized stem is revalidated before the manifest is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CMB_REPO_URL = "https://github.com/FreedomIntelligence/CMB"
CMB_REPO_COMMIT = "6c8ece46097dae736c6805dd3b831e1a38c08971"
CMB_PAPER = "https://arxiv.org/abs/2308.08833"
QUESTION_FILENAME = "CMB-test-choice-question-merge.json"
ANSWER_FILENAME = "CMB-test-choice-answer.json"
OVERLAP_REPORT_FILENAME = "qzbc_overlap_report.json"
EXCLUSION_FILENAME = "overlap_exclusions.json"
DATA_MANIFEST_FILENAME = "data_manifest.json"
ROBUSTNESS_MANIFEST_FILENAME = "controlled_robustness_v1.json"

EXPECTED_QUESTION_SHA256 = "f1902a61fff9b859318a1c5b3b19a0f889652610141ad18c4f541d9e9ec79497"
EXPECTED_ANSWER_SHA256 = "42130b20f7660a83dd92f3c005eb6a8ef1221c98d9b27d3351204143944b532f"
EXPECTED_OVERLAP_REPORT_SHA256 = "5f4e14b13427f12e6658cc1769687da857f66452e071b1e1eeabd6d4ac787968"
EXPECTED_SOURCE_ZIP_SHA256 = "c2e4288127fa5e6c03cc3659d893b0bbb19e699f74fdadce8e7fa10ac5bfdd94"
EXPECTED_QUESTIONS = 11200
EXPECTED_EXCLUSIONS = 4
EXPECTED_ROBUSTNESS_ITEMS = 2800
ROBUSTNESS_PER_CLASS = 100
ROBUSTNESS_SEED = 20260808
OVERLAP_BENCHMARK_KEY = "CMB_exam_test"

NON_TEXT = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
LEADING_NUMBER = re.compile(
    r"^\s*(?:\[?\d+\]?\s*[\.．、:)）]\s*|[（(]\d+[）)]\s*)+"
)
OPTION_START = re.compile(r"\n\s*[AＡ][\.．、:：]\s*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return NON_TEXT.sub("", text)


def normalized_stem(value: Any) -> str:
    """Exactly reproduce the normalization used by the formal overlap audit."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = LEADING_NUMBER.sub("", text)
    found = OPTION_START.search(text)
    if found:
        text = text[: found.start()]
    return norm(text)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_pinned(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}: {path}")


def _copy_pinned(source: Path, destination: Path, expected: str, label: str) -> None:
    source = source.resolve()
    destination = destination.resolve()
    _verify_pinned(source, expected, label)
    if destination.is_file():
        _verify_pinned(destination, expected, label)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        _verify_pinned(temporary, expected, label)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _cmb_overlap_entry(report: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        row
        for row in report.get("benchmarks", [])
        if isinstance(row, Mapping) and row.get("benchmark") == OVERLAP_BENCHMARK_KEY
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {OVERLAP_BENCHMARK_KEY!r} overlap entry, got {len(matches)}"
        )
    entry = matches[0]
    if int(entry.get("records", -1)) != EXPECTED_QUESTIONS:
        raise ValueError(f"overlap report CMB cardinality changed: {entry.get('records')}")
    if int(entry.get("exact_normalized_question_matches", -1)) != EXPECTED_EXCLUSIONS:
        raise ValueError(
            "overlap report exact-match count changed: "
            f"{entry.get('exact_normalized_question_matches')}"
        )
    examples = entry.get("exact_examples")
    if not isinstance(examples, list) or len(examples) != EXPECTED_EXCLUSIONS:
        raise ValueError(
            "the report does not contain every exact CMB example; refusing to infer missing IDs"
        )
    return entry


def build_exclusion_manifest(
    question_path: Path,
    report_path: Path,
) -> Mapping[str, Any]:
    _verify_pinned(question_path, EXPECTED_QUESTION_SHA256, "CMB question file")
    _verify_pinned(report_path, EXPECTED_OVERLAP_REPORT_SHA256, "QZBC overlap report")
    questions = _load_json(question_path)
    if not isinstance(questions, list) or len(questions) != EXPECTED_QUESTIONS:
        raise ValueError("invalid CMB question payload")
    by_id: dict[int, Mapping[str, Any]] = {}
    for row in questions:
        if not isinstance(row, Mapping):
            raise ValueError("CMB question row is not an object")
        item_id = int(row["id"])
        if item_id in by_id:
            raise ValueError(f"duplicate CMB ID: {item_id}")
        by_id[item_id] = row

    report = _load_json(report_path)
    if not isinstance(report, Mapping):
        raise ValueError("overlap report is not an object")
    entry = _cmb_overlap_entry(report)
    exclusions = []
    for example in entry["exact_examples"]:
        if not isinstance(example, Mapping):
            raise ValueError("invalid overlap example")
        match = re.fullmatch(r"exam:(\d+)", str(example.get("eval_id", "")))
        if match is None:
            raise ValueError(f"invalid CMB overlap eval_id: {example.get('eval_id')!r}")
        item_id = int(match.group(1))
        if item_id not in by_id:
            raise ValueError(f"overlap report references unknown CMB ID {item_id}")
        question = str(by_id[item_id].get("question") or "")
        report_text = str(example.get("text") or "")
        # The audit stores raw[:240].  Revalidate both that exact excerpt and
        # the normalization that actually determined overlap membership.
        if question[:240] != report_text:
            raise ValueError(f"overlap report text no longer matches CMB ID {item_id}")
        current_stem = normalized_stem(question)
        reported_stem = normalized_stem(report_text)
        if not current_stem or current_stem != reported_stem:
            raise ValueError(f"normalized overlap stem mismatch for CMB ID {item_id}")
        train_ids = example.get("train_ids")
        if not isinstance(train_ids, list) or not train_ids:
            raise ValueError(f"overlap CMB ID {item_id} has no matched training IDs")
        exclusions.append(
            {
                "id": item_id,
                "eval_id": f"exam:{item_id}",
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "normalized_stem_sha256": hashlib.sha256(current_stem.encode("utf-8")).hexdigest(),
                "matched_train_ids": [str(value) for value in train_ids],
                "overlap_kind": "exact_normalized_question",
            }
        )
    exclusions.sort(key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in exclusions]
    if len(ids) != EXPECTED_EXCLUSIONS or len(set(ids)) != EXPECTED_EXCLUSIONS:
        raise ValueError(f"invalid derived overlap exclusion IDs: {ids}")
    return {
        "schema_version": "care-opd-cmb-overlap-exclusions-v1",
        "benchmark": "CMB-Exam test",
        "derivation": "parsed from the CMB_exam_test exact_examples in the pinned QZBC overlap report",
        "normalization": "NFKC + leading-number removal + option-block removal + lowercase/non-text removal",
        "question_sha256": EXPECTED_QUESTION_SHA256,
        "overlap_report_sha256": EXPECTED_OVERLAP_REPORT_SHA256,
        "official_full_n": EXPECTED_QUESTIONS,
        "excluded_n": EXPECTED_EXCLUSIONS,
        "overlap_clean_n": EXPECTED_QUESTIONS - EXPECTED_EXCLUSIONS,
        "excluded_ids": ids,
        "exclusions": exclusions,
        "exclusion_set_fingerprint": canonical_hash(exclusions),
    }


def verify_exclusion_manifest(
    manifest: Mapping[str, Any], question_path: Path, report_path: Path
) -> None:
    expected = build_exclusion_manifest(question_path, report_path)
    stable_keys = (
        "schema_version",
        "benchmark",
        "derivation",
        "normalization",
        "question_sha256",
        "overlap_report_sha256",
        "official_full_n",
        "excluded_n",
        "overlap_clean_n",
        "excluded_ids",
        "exclusions",
        "exclusion_set_fingerprint",
    )
    actual_stable = {key: manifest.get(key) for key in stable_keys}
    expected_stable = {key: expected.get(key) for key in stable_keys}
    if actual_stable != expected_stable:
        raise ValueError("overlap exclusion manifest is stale or was not derived from the pinned report")


def build_robustness_manifest(
    question_path: Path, exclusion_manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Build a result-blind 100-per-class option-permutation track."""
    _verify_pinned(question_path, EXPECTED_QUESTION_SHA256, "CMB question file")
    questions = _load_json(question_path)
    excluded = {int(value) for value in exclusion_manifest["excluded_ids"]}
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for row in questions:
        item_id = int(row["id"])
        if item_id in excluded:
            continue
        class_key = f"{row['exam_type']}/{row['exam_class']}"
        by_class.setdefault(class_key, []).append(row)
    if len(by_class) != 28:
        raise ValueError(f"expected 28 CMB exam classes for robustness, got {len(by_class)}")

    entries = []
    class_counts = {}
    for class_key in sorted(by_class):
        candidates = sorted(
            by_class[class_key],
            key=lambda row: hashlib.sha256(
                f"select:{ROBUSTNESS_SEED}:{class_key}:{int(row['id'])}".encode("utf-8")
            ).hexdigest(),
        )
        selected = candidates[:ROBUSTNESS_PER_CLASS]
        if len(selected) != ROBUSTNESS_PER_CLASS:
            raise ValueError(f"not enough clean CMB items in class {class_key}")
        class_counts[class_key] = len(selected)
        for row in selected:
            item_id = int(row["id"])
            options = row.get("option")
            if not isinstance(options, Mapping):
                raise ValueError(f"CMB ID {item_id} options are not an object")
            canonical = [
                str(label).strip().upper()
                for label, text in options.items()
                if str(text or "").strip()
            ]
            permuted = sorted(
                canonical,
                key=lambda label: hashlib.sha256(
                    f"permute:{ROBUSTNESS_SEED}:{item_id}:{label}".encode("utf-8")
                ).hexdigest(),
            )
            if permuted == canonical:
                permuted = permuted[1:] + permuted[:1]
            mapping = dict(zip(canonical, permuted, strict=True))
            if set(mapping) != set(canonical) or set(mapping.values()) != set(canonical):
                raise ValueError(f"invalid option permutation for CMB ID {item_id}")
            if all(mapping[label] == label for label in canonical):
                raise ValueError(f"identity option permutation for CMB ID {item_id}")
            entries.append(
                {
                    "canonical_id": item_id,
                    "perturbed_id": f"perm:{item_id}",
                    "exam_type": str(row["exam_type"]),
                    "exam_class": str(row["exam_class"]),
                    "class_key": class_key,
                    "canonical_to_perturbed": mapping,
                    "selection_rank_sha256": hashlib.sha256(
                        f"select:{ROBUSTNESS_SEED}:{class_key}:{item_id}".encode("utf-8")
                    ).hexdigest(),
                }
            )
    entries.sort(key=lambda row: int(row["canonical_id"]))
    if len(entries) != EXPECTED_ROBUSTNESS_ITEMS:
        raise ValueError(f"unexpected robustness subset size: {len(entries)}")
    ids = [int(row["canonical_id"]) for row in entries]
    if len(set(ids)) != EXPECTED_ROBUSTNESS_ITEMS or set(ids) & excluded:
        raise ValueError("robustness subset contains duplicates or overlap-excluded IDs")
    return {
        "schema_version": "care-opd-cmb-controlled-robustness-v1",
        "benchmark": "CMB-Exam test / derived controlled robustness",
        "official_track": False,
        "perturbation": "deterministic option permutation with gold-label remapping",
        "semantic_invariant": "question and option text are unchanged; only option labels/order change",
        "selection": "100 result-blind items per each of 28 exam classes after exact-overlap exclusion",
        "seed": ROBUSTNESS_SEED,
        "items_per_class": ROBUSTNESS_PER_CLASS,
        "n": EXPECTED_ROBUSTNESS_ITEMS,
        "question_sha256": EXPECTED_QUESTION_SHA256,
        "overlap_exclusion_set_fingerprint": exclusion_manifest[
            "exclusion_set_fingerprint"
        ],
        "class_counts": class_counts,
        "canonical_ids": ids,
        "entries": entries,
        "robustness_set_fingerprint": canonical_hash(entries),
    }


def verify_robustness_manifest(
    manifest: Mapping[str, Any],
    question_path: Path,
    exclusion_manifest: Mapping[str, Any],
) -> None:
    expected = build_robustness_manifest(question_path, exclusion_manifest)
    if manifest != expected:
        raise ValueError("controlled robustness manifest is stale or not deterministically derived")


def _validate_question_answer_contract(question_path: Path, answer_path: Path) -> Mapping[str, Any]:
    questions = _load_json(question_path)
    answers = _load_json(answer_path)
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise ValueError("CMB test inputs must be JSON arrays")
    if len(questions) != EXPECTED_QUESTIONS or len(answers) != EXPECTED_QUESTIONS:
        raise ValueError(f"expected {EXPECTED_QUESTIONS} question and answer rows")
    question_ids = [int(row["id"]) for row in questions]
    answer_ids = [int(row["id"]) for row in answers]
    expected_ids = list(range(1, EXPECTED_QUESTIONS + 1))
    if sorted(question_ids) != expected_ids or sorted(answer_ids) != expected_ids:
        raise ValueError("CMB IDs must each cover 1..11200 exactly once")
    return {
        "question_rows": len(questions),
        "answer_rows": len(answers),
        "unique_question_ids": len(set(question_ids)),
        "unique_answer_ids": len(set(answer_ids)),
    }


def prepare(
    output_dir: Path,
    *,
    question_source: Optional[Path],
    answer_source: Optional[Path],
    overlap_report: Optional[Path],
    verify_only: bool,
) -> Mapping[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    question_path = output_dir / QUESTION_FILENAME
    answer_path = output_dir / ANSWER_FILENAME
    report_path = output_dir / OVERLAP_REPORT_FILENAME
    exclusion_path = output_dir / EXCLUSION_FILENAME
    robustness_path = output_dir / ROBUSTNESS_MANIFEST_FILENAME

    if verify_only:
        _verify_pinned(question_path, EXPECTED_QUESTION_SHA256, "CMB question file")
        _verify_pinned(answer_path, EXPECTED_ANSWER_SHA256, "CMB answer file")
        _verify_pinned(report_path, EXPECTED_OVERLAP_REPORT_SHA256, "QZBC overlap report")
        if not exclusion_path.is_file():
            raise FileNotFoundError(f"missing overlap exclusion manifest: {exclusion_path}")
        if not robustness_path.is_file():
            raise FileNotFoundError(f"missing robustness manifest: {robustness_path}")
    else:
        if question_source is None or answer_source is None or overlap_report is None:
            raise ValueError(
                "preparation requires --question-source, --answer-source, and --overlap-report"
            )
        _copy_pinned(question_source, question_path, EXPECTED_QUESTION_SHA256, "CMB question file")
        _copy_pinned(answer_source, answer_path, EXPECTED_ANSWER_SHA256, "CMB answer file")
        _copy_pinned(
            overlap_report, report_path, EXPECTED_OVERLAP_REPORT_SHA256, "QZBC overlap report"
        )
        atomic_json(exclusion_path, build_exclusion_manifest(question_path, report_path))

    contract = _validate_question_answer_contract(question_path, answer_path)
    exclusion = _load_json(exclusion_path)
    if not isinstance(exclusion, Mapping):
        raise ValueError("overlap exclusion manifest is not an object")
    verify_exclusion_manifest(exclusion, question_path, report_path)
    robustness = _load_json(robustness_path) if verify_only else build_robustness_manifest(
        question_path, exclusion
    )
    if not isinstance(robustness, Mapping):
        raise ValueError("controlled robustness manifest is not an object")
    if not verify_only:
        atomic_json(robustness_path, robustness)
    verify_robustness_manifest(robustness, question_path, exclusion)
    dataset_fingerprint = canonical_hash(
        {
            "question_sha256": EXPECTED_QUESTION_SHA256,
            "answer_sha256": EXPECTED_ANSWER_SHA256,
            "overlap_report_sha256": EXPECTED_OVERLAP_REPORT_SHA256,
            "exclusion_set_fingerprint": exclusion["exclusion_set_fingerprint"],
            "robustness_set_fingerprint": robustness["robustness_set_fingerprint"],
        }
    )
    manifest = {
        "schema_version": "care-opd-cmb-data-manifest-v1",
        "benchmark": "CMB-Exam test",
        "official_full_n": EXPECTED_QUESTIONS,
        "overlap_clean_n": EXPECTED_QUESTIONS - EXPECTED_EXCLUSIONS,
        "files": {
            QUESTION_FILENAME: {
                "path": str(question_path),
                "bytes": question_path.stat().st_size,
                "sha256": EXPECTED_QUESTION_SHA256,
            },
            ANSWER_FILENAME: {
                "path": str(answer_path),
                "bytes": answer_path.stat().st_size,
                "sha256": EXPECTED_ANSWER_SHA256,
            },
            OVERLAP_REPORT_FILENAME: {
                "path": str(report_path),
                "bytes": report_path.stat().st_size,
                "sha256": EXPECTED_OVERLAP_REPORT_SHA256,
            },
            EXCLUSION_FILENAME: {
                "path": str(exclusion_path),
                "bytes": exclusion_path.stat().st_size,
                "sha256": sha256_file(exclusion_path),
            },
            ROBUSTNESS_MANIFEST_FILENAME: {
                "path": str(robustness_path),
                "bytes": robustness_path.stat().st_size,
                "sha256": sha256_file(robustness_path),
            },
        },
        "dataset_fingerprint": dataset_fingerprint,
        "source": {
            "repository": CMB_REPO_URL,
            "repository_commit": CMB_REPO_COMMIT,
            "paper": CMB_PAPER,
            "license": "Apache-2.0",
            "question_origin": "data/CMB.zip -> CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json",
            "answer_origin": "data/CMB-test-choice-answer.json",
            "source_zip_sha256": EXPECTED_SOURCE_ZIP_SHA256,
        },
        "overlap_audit": {
            "report_sha256": EXPECTED_OVERLAP_REPORT_SHA256,
            "excluded_ids": exclusion["excluded_ids"],
            "exclusion_set_fingerprint": exclusion["exclusion_set_fingerprint"],
        },
        "controlled_robustness_v1": {
            "official_track": False,
            "n": robustness["n"],
            "seed": robustness["seed"],
            "robustness_set_fingerprint": robustness["robustness_set_fingerprint"],
        },
        "contract": contract,
        "verified_at_utc": utc_now(),
    }
    atomic_json(output_dir / DATA_MANIFEST_FILENAME, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--question-source", type=Path)
    result.add_argument("--answer-source", type=Path)
    result.add_argument("--overlap-report", type=Path)
    result.add_argument("--verify-only", action="store_true")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    manifest = prepare(
        args.output_dir,
        question_source=args.question_source,
        answer_source=args.answer_source,
        overlap_report=args.overlap_report,
        verify_only=args.verify_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
