#!/usr/bin/env python3
"""Freeze the public-generalization protocol before formal model predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCKED_FILES = (
    "src/care_opd/models.py",
    "src/care_opd/qzbc_assets.py",
    "src/care_opd/qzbc_contract.py",
    "src/care_opd/qzbc_dataset.py",
    "src/care_opd/qzbc_rollout.py",
    "src/care_opd/qzbc_seed_builder.py",
    "src/care_opd/rollout.py",
    "src/care_opd/trainer_utils.py",
    "public_eval/create_preregistration_lock.py",
    "public_eval/README.md",
    "public_eval/GENERALIZATION_PROTOCOL.md",
    "public_eval/PUBLIC_GENERALIZATION_PREREGISTRATION.md",
    "public_eval/UNIFIED_METRIC_SCHEMA.json",
    "public_eval/preregistration.py",
    "public_eval/runtime_identity.py",
    "public_eval/summarize_internal_qzbc.py",
    "public_eval/summarize_generalization.py",
    "public_eval/run_generalization_gpu4_5.sh",
    "public_eval/verify_model_lineage.py",
    "public_eval/qzbc_grouped/ANALYSIS_SPEC.json",
    "public_eval/qzbc_grouped/protocol.py",
    "public_eval/qzbc_grouped/evaluate_qzbc_grouped.py",
    "public_eval/qzbc_grouped/run_existing_gpu4_5.sh",
    "public_eval/cmb_exam/prepare_data.py",
    "public_eval/cmb_exam/evaluate_cmb_exam.py",
    "public_eval/cmb_exam/run_existing_gpu4_5.sh",
    "public_eval/cmb_exam/data/controlled_robustness_v1.json",
    "public_eval/cmb_exam/data/CMB-test-choice-answer.json",
    "public_eval/cmb_exam/data/CMB-test-choice-question-merge.json",
    "public_eval/cmb_exam/data/overlap_exclusions.json",
    "public_eval/cmb_exam/data/qzbc_overlap_report.json",
    "public_eval/tcmbench_v2/protocol.py",
    "public_eval/tcmbench_v2/evaluate_tcmbench_v2.py",
    "public_eval/tcmbench_v2/run_existing_gpu4_5.sh",
    "public_eval/tcmbench_v2/overlap_exclusions.json",
    "public_eval/tcmeval_pa/evaluate_tcmeval_pa.py",
    "public_eval/tcmeval_pa/prepare_data.py",
    "public_eval/tcmeval_pa/evaluate_controlled_robustness.py",
    "public_eval/tcmeval_pa/run_controlled_gpu4_5.sh",
    "public_eval/tcmeval_pa/ANALYSIS_SPEC.json",
    "public_eval/tcmeval_pa/data/TCMEval-PA.xlsx",
    "public_eval/mtcmb_clean/protocol.py",
    "public_eval/mtcmb_clean/prepare_data.py",
    "public_eval/mtcmb_clean/evaluate_mtcmb.py",
    "public_eval/mtcmb_clean/run_existing_gpu4_5.sh",
    "public_eval/mtcmb_clean/overlap_exclusions.json",
    "public_eval/mtcmb_clean/task_manifest.json",
)
MODEL_ORDER = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)
IMMUTABLE_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def load_model_suite(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"model suite must be an object: {resolved}")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"model suite has no models: {resolved}")
    labels = tuple(str(item.get("label") or "") for item in models)
    if labels != MODEL_ORDER:
        raise ValueError(f"model suite order changed: {labels}")
    semantic = []
    locked_models = []
    for item in models:
        fingerprint = str(item.get("model_fingerprint") or "")
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise ValueError(f"invalid model fingerprint for {item.get('label')}")
        label = str(item["label"])
        semantic.append((label, fingerprint))
        locked_models.append(
            {
                "label": label,
                "model_fingerprint": fingerprint,
                "canonical_path": str(item.get("canonical_path") or item.get("model_path") or ""),
            }
        )
    expected = canonical_hash(semantic)
    if payload.get("suite_fingerprint") != expected:
        raise ValueError("model suite fingerprint does not close")
    return {
        "suite_fingerprint": expected,
        "models": locked_models,
    }


def fingerprint_auxiliary_model(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "NOT_LOCKED",
            "reason": "no auxiliary scorer path supplied",
        }
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"auxiliary scorer is not a directory: {root}")
    files = [
        item
        for item in root.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(root).parts
    ]
    if not files:
        raise ValueError(f"auxiliary scorer has no files: {root}")
    inventory = []
    for item in sorted(files, key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        inventory.append(
            {
                "name": relative,
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return {
        "status": "LOCKED",
        "role": "MTCMB local BERTScore scorer",
        "canonical_path": str(root),
        "artifact_fingerprint": canonical_hash(inventory),
        "inventory": inventory,
    }


def load_origin_reference(path: Path | None, model_suite: dict[str, Any]) -> dict[str, Any]:
    frozen_fingerprint = str(model_suite["models"][0]["model_fingerprint"])
    if path is None:
        return {
            "status": "NOT_VERIFIED",
            "reason": "no independent immutable origin reference supplied before evaluation",
            "frozen_model_fingerprint": frozen_fingerprint,
        }
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"origin reference must be an object: {resolved}")
    reference_fingerprint = str(payload.get("model_fingerprint") or "")
    revision = str(payload.get("revision") or "").strip().lower()
    if (
        IMMUTABLE_REVISION_RE.fullmatch(revision) is None
        or reference_fingerprint != frozen_fingerprint
    ):
        raise ValueError("origin reference does not match the frozen suite entry")
    return {
        "status": "REFERENCE_MATCHED",
        "claim_guardrail": (
            "A matching file is recorded, but its independence/provenance must be established "
            "outside this self-authored lock before calling frozen the raw upstream initial model."
        ),
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "revision": revision,
        "model_fingerprint": reference_fingerprint,
    }


def locked_contract(
    project_root: Path,
    *,
    model_suite_path: Path,
    origin_reference_path: Path | None,
    bert_model_path: Path | None,
) -> dict[str, Any]:
    files = []
    missing = []
    for relative in LOCKED_FILES:
        path = project_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if missing:
        raise FileNotFoundError(f"cannot freeze incomplete protocol; missing {missing}")
    model_suite = load_model_suite(model_suite_path)
    contract = {
        "schema_version": "care-opd-public-preregistration-lock-v1",
        "scope": (
            "internal QZBC paired recovery, TCMEval-PA controlled robustness, "
            "CMB-Exam, TCMBench V2, and MTCMB provenance-clean evaluation"
        ),
        "timing_disclosure": {
            "tcmeval_pa_official_clean_already_observed": True,
            "tcmeval_pa_controlled_predictions_observed": False,
            "cmb_exam_predictions_observed": False,
            "tcmbench_v2_predictions_observed": False,
            "mtcmb_clean_predictions_observed": False,
            "internal_qzbc_results_observed": True,
        },
        "fixed_model_order": list(MODEL_ORDER),
        "model_suite": model_suite,
        "auxiliary_models": {
            "mtcmb_bert_scorer": fingerprint_auxiliary_model(bert_model_path),
        },
        "origin_reference": load_origin_reference(origin_reference_path, model_suite),
        "source_versions": {
            "tcmeval_pa_figshare_doi": "10.6084/m9.figshare.29651261.v3",
            "tcmeval_pa_xlsx_sha256": "905cb28f145ba50f3b1c912265ff6649e0bb38f56adf6cd588d77ca768db92c1",
            "cmb_repository_commit": "6c8ece46097dae736c6805dd3b831e1a38c08971",
            "tcmbench_data_commit": "7e2d2326ff79da186c6fccaafa374cf1f77ac968",
            "mtcmb_repository_commit": "faffd813c67fe012cef74277a23270d5dd6db9b8",
        },
        "locked_files": files,
    }
    contract["contract_fingerprint"] = canonical_hash(contract)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="care_opd project root",
    )
    parser.add_argument(
        "--model-suite",
        type=Path,
        required=True,
        help="already frozen six-checkpoint model_suite.json, before public predictions",
    )
    parser.add_argument(
        "--origin-reference",
        type=Path,
        help="optional independently preserved upstream revision/fingerprint reference",
    )
    parser.add_argument(
        "--bert-model",
        type=Path,
        help="local frozen BERT scorer used by MTCMB official metrics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "PREREGISTRATION_LOCK.json",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve(strict=True)
    contract = locked_contract(
        project_root,
        model_suite_path=args.model_suite,
        origin_reference_path=args.origin_reference,
        bert_model_path=args.bert_model,
    )
    output = args.output.resolve()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        old_contract = {key: value for key, value in existing.items() if key != "created_at_utc"}
        if old_contract != contract:
            raise RuntimeError(
                f"existing lock differs from current protocol: {output}; "
                "use a new versioned lock/output root instead of overwriting it"
            )
        print(f"[preregistration] validated existing immutable lock: {output}")
        print(f"[preregistration] sha256={sha256_file(output)}")
        return 0
    payload = {
        **contract,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output, payload)
    print(f"[preregistration] created immutable lock: {output}")
    print(f"[preregistration] sha256={sha256_file(output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
