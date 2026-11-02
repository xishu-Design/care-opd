#!/usr/bin/env python3
"""Verify pinned MTCMB data and freeze formal-training overlap exclusions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import protocol


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "overlap_exclusions.json"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_manifest(
    repo: Path,
    train_seed_file: Path,
    overlap_report: Path | None = None,
) -> dict[str, Any]:
    upstream = protocol.verify_upstream_repo(repo)
    exposure = protocol.extract_train_exposure(train_seed_file)
    source_ids = exposure.pop("source_record_ids")
    records = protocol.load_all_records(repo)
    audit = protocol.compute_overlap_audit(records, train_seed_file)
    if audit["direct_provenance_source_record_ids"] != source_ids:
        raise ValueError("direct provenance extraction disagrees with overlap audit")
    if overlap_report is not None:
        overlap_report = overlap_report.resolve(strict=True)
        actual_hash = protocol.sha256_file(overlap_report)
        if actual_hash != protocol.EXPECTED_OVERLAP_REPORT_SHA256:
            raise ValueError(
                f"overlap report hash {actual_hash} != pinned "
                f"{protocol.EXPECTED_OVERLAP_REPORT_SHA256}"
            )
    official_ids = {
        protocol.record_key(task, row["id"])
        for task, rows in records.items()
        for row in rows
    }
    unknown = set(source_ids) - official_ids
    if unknown:
        raise ValueError(f"formal training provenance references unknown MTCMB IDs: {sorted(unknown)}")
    manifest = {
        "schema_version": "care-opd-mtcmb-overlap-exclusions-v1",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "upstream_url": protocol.UPSTREAM_URL,
        "upstream_commit": protocol.UPSTREAM_COMMIT,
        "upstream_data_fingerprint": protocol.canonical_hash(upstream),
        "normalization": (
            "result-blind union of direct curated-asset provenance, exact Unicode-NFKC "
            "normalized question stems, and the preregistered 13-character n-gram near rule"
        ),
        "formal_train_exposure": exposure,
        "source_record_ids_scope": "direct_provenance_only",
        "source_record_ids": source_ids,
        "overlap_additional_record_ids": audit[
            "additional_exact_or_near_record_ids"
        ],
        "overlap_excluded_records": protocol.EXPECTED_OVERLAP_EXCLUDED,
        "overlap_excluded_by_task": audit["overlap_excluded_by_task"],
        "overlap_audit": {
            "source_overlap_report_sha256": protocol.EXPECTED_OVERLAP_REPORT_SHA256,
            "result_blind_exclusion_rule": (
                "direct_provenance UNION exact_normalized_question_stem UNION "
                "preregistered_near_question_threshold"
            ),
            "counts": audit["counts"],
            "exact_visible_segment_subset_of_union": True,
        },
        "mtcmb_full_records": protocol.EXPECTED_FULL_N,
        "offline11_records": protocol.EXPECTED_OFFLINE11_N,
        "overlap_clean_all12_records": protocol.EXPECTED_OVERLAP_CLEAN_ALL12_N,
        "overlap_clean_offline11_records": protocol.EXPECTED_OVERLAP_CLEAN_OFFLINE11_N,
        "directly_exposed_task_families": sorted(protocol.DIRECTLY_EXPOSED_FAMILIES),
        "clean9_tasks": list(protocol.CLEAN9_TASKS),
        "clean9_records": protocol.EXPECTED_CLEAN9_N,
        "remote_judge_excluded_from_offline_scoring": sorted(protocol.REMOTE_JUDGE_TASKS),
        "controlled_option_permutation": {
            "status": "derived_not_official",
            "algorithm": (
                "SHA256 rank of canonical labels keyed by protocol/task/id; rotate by one "
                "only when the rank is identity; consistently remap the gold label"
            ),
            "overlap_clean_mcq_pairs": protocol.EXPECTED_OVERLAP_CLEAN_MCQ_PAIRS,
            "strict_clean9_mcq_pairs": protocol.EXPECTED_CLEAN9_MCQ_PAIRS,
            "excluded_malformed_option_inventory_ids": sorted(
                protocol.CONTROLLED_PERTURBATION_EXCLUDED_IDS
            ),
        },
        "claim_scope": {
            "mtcmb_full": "in-source diagnostic only",
            "overlap_clean": (
                "supplementary result-blind item-decontaminated view; direct provenance, "
                "exact normalized stems and preregistered near matches are excluded; "
                "ED-A/ED-B family exposure remains"
            ),
            "provenance_clean9": (
                "supplementary strict task-family provenance-clean view; not proof against "
                "base-model pretraining exposure"
            ),
        },
    }
    protocol.validate_exclusion_manifest(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtcmb-repo", type=Path, required=True)
    parser.add_argument("--train-seed-file", type=Path, required=True)
    parser.add_argument(
        "--overlap-report",
        type=Path,
        help="optional pinned audit artifact; when supplied its SHA-256 is fail-closed",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    expected = build_manifest(
        args.mtcmb_repo,
        args.train_seed_file,
        args.overlap_report,
    )
    output = args.output.resolve()
    if args.verify_only:
        if not output.is_file():
            raise FileNotFoundError(output)
        actual = json.loads(output.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"frozen MTCMB exclusion manifest differs: {output}")
        status = "verified"
    else:
        if output.is_file():
            actual = json.loads(output.read_text(encoding="utf-8"))
            if actual != expected:
                raise RuntimeError(
                    f"refusing to overwrite a different exclusion manifest: {output}"
                )
        else:
            atomic_json(output, expected)
        status = "created_or_identical"
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "output_sha256": protocol.sha256_file(output),
                "full": protocol.EXPECTED_FULL_N,
                "overlap_clean_all12": protocol.EXPECTED_OVERLAP_CLEAN_ALL12_N,
                "overlap_clean_offline11": protocol.EXPECTED_OVERLAP_CLEAN_OFFLINE11_N,
                "clean9": protocol.EXPECTED_CLEAN9_N,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
