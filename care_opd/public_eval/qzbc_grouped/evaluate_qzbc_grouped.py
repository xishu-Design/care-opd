#!/usr/bin/env python3
"""Identity-bound six-model QZBC grouped evaluation.

This is a thin formal wrapper around :mod:`care_opd.qzbc_rollout`.  It does
not alter the Agent or verifier.  Its job is to bind a deterministic greedy
4,200-request run to exact checkpoint, data/split, evaluator and runtime
identities, publish hash-bearing completion markers, and summarize the paired
clean-to-recovery endpoints used by the unified main table.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PUBLIC_ROOT = HERE.parent
PROJECT_ROOT = PUBLIC_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(HERE), str(SRC_ROOT), str(PUBLIC_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from care_opd.qzbc_rollout import collect  # noqa: E402
from care_opd.rollout import GenerationArguments  # noqa: E402
from preregistration import validate_static_lock  # noqa: E402
from runtime_identity import runtime_identity  # noqa: E402
import summarize_internal_qzbc as paired_summary  # noqa: E402
from protocol import (  # noqa: E402
    EXPECTED_REQUEST_ROWS,
    EXPECTED_SEED,
    EXPECTED_VARIANTS,
    MODEL_ORDER,
    PROTOCOL_VERSION,
    artifact_inventory,
    atomic_json,
    atomic_text,
    audit_data_contract,
    canonical_hash,
    read_json,
    read_jsonl,
    sha256_file,
    suite_entry,
    validate_artifact_inventory,
    validate_frozen_data_audit_quick,
    validate_model_suite,
)


ANALYSIS_SPEC = HERE / "ANALYSIS_SPEC.json"
UNIFIED_SCHEMA = PUBLIC_ROOT / "UNIFIED_METRIC_SCHEMA.json"
REQUIRED_LOCKED_PATHS = (
    "src/care_opd/models.py",
    "src/care_opd/qzbc_contract.py",
    "src/care_opd/qzbc_assets.py",
    "src/care_opd/qzbc_dataset.py",
    "src/care_opd/qzbc_rollout.py",
    "src/care_opd/qzbc_seed_builder.py",
    "src/care_opd/rollout.py",
    "src/care_opd/trainer_utils.py",
    "public_eval/preregistration.py",
    "public_eval/runtime_identity.py",
    "public_eval/summarize_internal_qzbc.py",
    "public_eval/UNIFIED_METRIC_SCHEMA.json",
    "public_eval/qzbc_grouped/ANALYSIS_SPEC.json",
    "public_eval/qzbc_grouped/protocol.py",
    "public_eval/qzbc_grouped/evaluate_qzbc_grouped.py",
    "public_eval/qzbc_grouped/run_existing_gpu4_5.sh",
)
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
OUTPUT_ARTIFACTS = (
    "run_manifest.json",
    "qzbc_rollouts.jsonl",
    "qzbc_opd.jsonl",
    "qzbc_dpo.jsonl",
    "summary.json",
    "evaluation_summary.json",
)


def _normalize_execution_topology(args: argparse.Namespace) -> None:
    """Validate and freeze the logical-to-physical worker topology."""

    devices = tuple(
        item.strip() for item in str(args.devices).split(",") if item.strip()
    )
    visible_devices = tuple(
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    )
    if args.num_workers < 1 or len(devices) != args.num_workers:
        raise ValueError("--num-workers must equal the number of --devices")
    if len(visible_devices) != args.num_workers:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must expose exactly one physical GPU per worker"
        )
    args.devices = devices
    args.visible_physical_devices = visible_devices


def _execution_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_workers": args.num_workers,
        "devices": list(args.devices),
        "visible_physical_devices": list(args.visible_physical_devices),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _analysis_spec() -> dict[str, Any]:
    payload = read_json(ANALYSIS_SPEC)
    expected_generation = {
        "dtype": "bfloat16",
        "do_sample": False,
        "max_new_tokens": 1200,
        "seed": EXPECTED_SEED,
        "trust_remote_code": False,
    }
    if (
        payload.get("schema_version") != "care-opd-qzbc-grouped-analysis-v1"
        or tuple(payload.get("model_order") or ()) != MODEL_ORDER
        or tuple(payload.get("variants") or ()) != EXPECTED_VARIANTS
        or int(payload.get("expected_request_rows", -1)) != EXPECTED_REQUEST_ROWS
        or int(payload.get("expected_family_clean_request_rows", -1)) != 3900
        or int(payload.get("expected_family_clean_seed_rows", -1)) != 975
        or payload.get("family_clean_kept_seed_ids_fingerprint")
        != "f68e74e04fdc91f5ecaf96acaff9c9a52d00c224fd44e12c09fa2e0c413dcd4d"
        or payload.get("family_clean_excluded_seed_ids_fingerprint")
        != "dd129dbec60d2b93157ed0be4f833133af09dc8a25777e1b03a8f11d4348ef1b"
        or payload.get("family_clean_request_ids_fingerprint")
        != "9ab70bd2237ba505203717b5d44af0f651c109bac69085da77bcff5f63a4cd3a"
        or payload.get("family_clean_task_counts")
        != {
            "doctor_ai_chat": 137,
            "doctor_review_text": 0,
            "patient_diet_guidance": 150,
            "patient_education": 150,
            "quiz_generation": 142,
            "student_ai_chat": 106,
            "student_analysis": 140,
            "student_quiz_explanation": 150,
        }
        or int(payload.get("expected_seed", -1)) != EXPECTED_SEED
        or payload.get("generation") != expected_generation
    ):
        raise ValueError("QZBC grouped analysis specification changed")
    return payload


def _code_identity() -> dict[str, Any]:
    paths = (
        ANALYSIS_SPEC,
        HERE / "protocol.py",
        Path(__file__).resolve(),
        HERE / "run_existing_gpu4_5.sh",
        SRC_ROOT / "care_opd" / "qzbc_rollout.py",
        SRC_ROOT / "care_opd" / "qzbc_contract.py",
        SRC_ROOT / "care_opd" / "qzbc_assets.py",
        SRC_ROOT / "care_opd" / "qzbc_dataset.py",
        SRC_ROOT / "care_opd" / "qzbc_seed_builder.py",
        SRC_ROOT / "care_opd" / "rollout.py",
        SRC_ROOT / "care_opd" / "trainer_utils.py",
        PUBLIC_ROOT / "runtime_identity.py",
        PUBLIC_ROOT / "summarize_internal_qzbc.py",
        UNIFIED_SCHEMA,
    )
    inventory = [
        {
            "path": path.resolve(strict=True).relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {"inventory": inventory, "fingerprint": canonical_hash(inventory)}


def _static_identity(lock_path: Path) -> dict[str, Any]:
    return validate_static_lock(
        lock_path,
        project_root=PROJECT_ROOT,
        required_paths=REQUIRED_LOCKED_PATHS,
    )


def _compiled_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "train": args.compiled_train_manifest,
        "dev": args.compiled_dev_manifest,
        "test": args.compiled_test_manifest,
    }


def _recompute_data_audit(args: argparse.Namespace) -> dict[str, Any]:
    return audit_data_contract(
        qzbc_root=args.qzbc_root,
        requests_path=args.requests,
        seed_manifest_path=args.seed_manifest,
        compiled_manifests=_compiled_paths(args),
        expected_seed=EXPECTED_SEED,
        expected_request_rows=EXPECTED_REQUEST_ROWS,
    )


def _load_and_match_data_audit(
    args: argparse.Namespace, *, full_recompute: bool
) -> dict[str, Any]:
    prepared = read_json(args.prepared_marker)
    if prepared.get("status") != "PREPARED":
        raise ValueError("QZBC prepared marker is not complete")
    prepared_contract = {
        key: value
        for key, value in prepared.items()
        if key not in {"prepared_at_utc", "prepared_fingerprint"}
    }
    if canonical_hash(prepared_contract) != prepared.get("prepared_fingerprint"):
        raise ValueError("QZBC prepared marker fingerprint does not close")
    recorded_suite = prepared.get("model_suite") or {}
    if (
        str(args.suite.resolve(strict=True)) != str(recorded_suite.get("path") or "")
        or sha256_file(args.suite) != recorded_suite.get("sha256")
    ):
        raise ValueError("QZBC model suite differs from the prepared marker")
    recorded_lock = prepared.get("preregistration_lock") or {}
    if sha256_file(args.preregistration_lock) != recorded_lock.get("sha256"):
        raise ValueError("QZBC preregistration lock differs from the prepared marker")
    if sha256_file(ANALYSIS_SPEC) != prepared.get("analysis_spec_sha256"):
        raise ValueError("QZBC analysis specification differs from the prepared marker")
    recorded_audit = prepared.get("data_contract") or {}
    if (
        str(args.data_audit.resolve(strict=True)) != str(recorded_audit.get("path") or "")
        or sha256_file(args.data_audit) != recorded_audit.get("sha256")
    ):
        raise ValueError("QZBC data audit differs from the prepared marker")
    if not full_recompute:
        observed = validate_frozen_data_audit_quick(
            args.data_audit,
            requests_path=args.requests,
        )
    else:
        observed = _recompute_data_audit(args)
    frozen = read_json(args.data_audit)
    if frozen.get("data_contract_fingerprint") != observed["data_contract_fingerprint"]:
        raise ValueError(
            "formal data contract changed after preparation; use a new output root"
        )
    if frozen != observed:
        raise ValueError("formal data audit content changed without fingerprint change")
    return observed


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    _analysis_spec()
    lock = _static_identity(args.preregistration_lock)
    source_suite = validate_model_suite(args.source_suite)
    if source_suite["suite_fingerprint"] != lock["model_suite_fingerprint"]:
        raise ValueError("source model suite differs from the preregistration lock")
    destination = args.suite.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != args.source_suite.resolve(strict=True).read_bytes():
            raise ValueError(
                "destination suite differs from source; use a new output root"
            )
    else:
        atomic_text(
            destination,
            args.source_suite.resolve(strict=True).read_text(encoding="utf-8"),
        )
    copied_suite = validate_model_suite(destination)
    if copied_suite["sha256"] != source_suite["sha256"]:
        raise ValueError("model suite copy is not byte-identical")
    audit = _recompute_data_audit(args)
    audit_path = args.data_audit.resolve()
    if audit_path.is_file():
        if read_json(audit_path) != audit:
            raise ValueError(
                "data audit differs from an existing prepared contract; use a new output root"
            )
    else:
        atomic_json(audit_path, audit)
    prepared_contract = {
        "schema_version": "care-opd-qzbc-prepared-v1",
        "status": "PREPARED",
        "model_suite": {
            key: copied_suite[key]
            for key in ("path", "sha256", "suite_fingerprint", "models")
        },
        "data_contract": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "fingerprint": audit["data_contract_fingerprint"],
        },
        "preregistration_lock": lock,
        "analysis_spec_sha256": sha256_file(ANALYSIS_SPEC),
    }
    prepared_fingerprint = canonical_hash(prepared_contract)
    if args.prepared_marker.is_file():
        prepared = read_json(args.prepared_marker)
        existing_contract = {
            key: value
            for key, value in prepared.items()
            if key not in {"prepared_at_utc", "prepared_fingerprint"}
        }
        if (
            existing_contract != prepared_contract
            or prepared.get("prepared_fingerprint") != prepared_fingerprint
        ):
            raise ValueError(
                "prepared marker contract changed; use a new output root"
            )
    else:
        prepared = {
            **prepared_contract,
            "prepared_at_utc": utc_now(),
            "prepared_fingerprint": prepared_fingerprint,
        }
        atomic_json(args.prepared_marker, prepared)
    print(
        "[prepared] QZBC grouped data/model contract PASS "
        f"requests={audit['requests']['rows']} suite={copied_suite['suite_fingerprint']}",
        flush=True,
    )
    return prepared


def _expected_requests(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_REQUEST_ROWS:
        raise ValueError("formal QZBC request cardinality changed")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id") or "")
        if not item_id or item_id in by_id:
            raise ValueError("formal QZBC requests contain empty/duplicate IDs")
        by_id[item_id] = row
    return rows, by_id


def _validate_rollout_outputs(
    output_dir: Path,
    request_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = read_jsonl(output_dir / "qzbc_rollouts.jsonl")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id") or "")
        if not item_id or item_id in by_id:
            raise ValueError("rollouts contain empty/duplicate IDs")
        expected = request_by_id.get(item_id)
        if expected is None:
            raise ValueError(f"unexpected rollout ID: {item_id}")
        if str(row.get("variant") or "") != str(expected.get("variant") or ""):
            raise ValueError(f"rollout/request variant mismatch: {item_id}")
        by_id[item_id] = row
    if set(by_id) != set(request_by_id):
        missing = sorted(set(request_by_id) - set(by_id))[:5]
        raise ValueError(f"rollouts are incomplete; missing={missing}")
    raw_summary = read_json(output_dir / "summary.json")
    passed = sum(bool(row.get("passed")) for row in rows)
    if int(raw_summary.get("rollouts", -1)) != len(rows) or int(
        raw_summary.get("passed", -1)
    ) != passed:
        raise ValueError("qzbc_rollout summary does not match rollout rows")
    return {"rows": len(rows), "passed": passed, "failed": len(rows) - passed}


def completed_valid(
    output_dir: Path,
    *,
    expected_run_fingerprint: str,
    request_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    try:
        completion = read_json(output_dir / "COMPLETE.json")
        manifest = read_json(output_dir / "run_manifest.json")
        evaluation_summary = read_json(output_dir / "evaluation_summary.json")
        if (
            completion.get("status") != "COMPLETE"
            or completion.get("run_fingerprint") != expected_run_fingerprint
            or manifest.get("run_fingerprint") != expected_run_fingerprint
            or evaluation_summary.get("run_fingerprint") != expected_run_fingerprint
            or int(completion.get("rows", -1)) != len(request_by_id)
            or tuple(completion.get("artifact_names") or ()) != OUTPUT_ARTIFACTS
            or sorted(
                str(item.get("name") or "")
                for item in (completion.get("artifacts") or [])
                if isinstance(item, Mapping)
            )
            != sorted(OUTPUT_ARTIFACTS)
            or completion.get("artifacts_fingerprint")
            != canonical_hash(completion.get("artifacts"))
            or completion.get("model_fingerprint")
            != manifest.get("model_fingerprint")
            or completion.get("suite_fingerprint")
            != manifest.get("model_suite_fingerprint")
            or completion.get("data_contract_fingerprint")
            != manifest.get("data_contract_fingerprint")
            or not validate_artifact_inventory(output_dir, completion.get("artifacts"))
        ):
            return False
        _validate_rollout_outputs(output_dir, request_by_id)
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _run_contract(
    *,
    args: argparse.Namespace,
    suite: Mapping[str, Any],
    model: Mapping[str, Any],
    data_audit: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    runtime = runtime_identity()
    code = _code_identity()
    generation = {
        "dtype": "bfloat16",
        "do_sample": False,
        "max_new_tokens": 1200,
        "seed": EXPECTED_SEED,
        "trust_remote_code": False,
        "max_format_failure_fraction": 0.25,
        **_execution_contract(args),
    }
    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "model_label": args.model_label,
        "model_canonical_path": model["canonical_path"],
        "model_fingerprint": model["model_fingerprint"],
        "model_suite_fingerprint": suite["suite_fingerprint"],
        "model_suite_sha256": suite["sha256"],
        "requests_sha256": data_audit["requests"]["sha256"],
        "requests_rows": data_audit["requests"]["rows"],
        "data_contract_fingerprint": data_audit["data_contract_fingerprint"],
        "data_audit_sha256": sha256_file(args.data_audit),
        "prepared_marker_sha256": sha256_file(args.prepared_marker),
        "prepared_fingerprint": read_json(args.prepared_marker)["prepared_fingerprint"],
        "analysis_spec_sha256": sha256_file(ANALYSIS_SPEC),
        "unified_schema_sha256": sha256_file(UNIFIED_SCHEMA),
        "generation": generation,
        "code_identity": code,
        "runtime_identity": runtime,
        "runtime_fingerprint": canonical_hash(runtime),
        "preregistration_lock_sha256": lock["sha256"],
    }
    return contract, canonical_hash(contract)


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    _analysis_spec()
    if args.model_label not in MODEL_ORDER:
        raise ValueError(f"unexpected model label: {args.model_label}")
    lock = _static_identity(args.preregistration_lock)
    suite = validate_model_suite(args.suite, rehash_labels=(args.model_label,))
    if suite["suite_fingerprint"] != lock["model_suite_fingerprint"]:
        raise ValueError("model suite differs from preregistration")
    model = suite_entry(suite, args.model_label)
    data_audit = _load_and_match_data_audit(args, full_recompute=False)
    request_rows, request_by_id = _expected_requests(args.requests)
    contract, run_fingerprint = _run_contract(
        args=args, suite=suite, model=model, data_audit=data_audit, lock=lock
    )
    output_dir = args.output_root.resolve() / "formal" / args.model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"run contract changed for {output_dir}; use a new output root"
            )
    else:
        manifest = {
            "schema_version": "care-opd-qzbc-model-run-v1",
            **contract,
            "run_fingerprint": run_fingerprint,
            "created_at_utc": utc_now(),
            "model": model,
            "suite_path": suite["path"],
            "data_audit_path": str(args.data_audit.resolve(strict=True)),
        }
        atomic_json(manifest_path, manifest)
    if completed_valid(
        output_dir,
        expected_run_fingerprint=run_fingerprint,
        request_by_id=request_by_id,
    ):
        print(f"[skip] validated QZBC grouped result: {output_dir}", flush=True)
        return read_json(output_dir / "evaluation_summary.json")

    (output_dir / "COMPLETE.json").unlink(missing_ok=True)
    print(
        f"[evaluate] model={args.model_label} requests={len(request_rows)} "
        f"workers={args.num_workers} devices={','.join(args.devices)} "
        f"physical_gpus={','.join(args.visible_physical_devices)}",
        flush=True,
    )
    collect(
        args.requests.resolve(strict=True),
        output_dir,
        GenerationArguments(
            model=str(model["canonical_path"]),
            device=None,
            dtype="bfloat16",
            max_new_tokens=1200,
            temperature=1.0,
            top_p=1.0,
            do_sample=False,
            trust_remote_code=False,
        ),
        seed=EXPECTED_SEED,
        max_examples=-1,
        max_format_failure_fraction=0.25,
        num_workers=args.num_workers,
        devices=args.devices,
    )
    counts = _validate_rollout_outputs(output_dir, request_by_id)
    evaluation_summary = {
        "schema_version": "care-opd-qzbc-model-evaluation-v1",
        "status": "COMPLETE",
        "model_label": args.model_label,
        "run_fingerprint": run_fingerprint,
        "model_fingerprint": model["model_fingerprint"],
        "model_canonical_path": model["canonical_path"],
        "suite_fingerprint": suite["suite_fingerprint"],
        "data_contract_fingerprint": data_audit["data_contract_fingerprint"],
        "preregistration_lock_sha256": lock["sha256"],
        "completed_at_utc": utc_now(),
        **counts,
        "task_success_rate": counts["passed"] / counts["rows"],
    }
    atomic_json(output_dir / "evaluation_summary.json", evaluation_summary)
    artifact_paths = [output_dir / name for name in OUTPUT_ARTIFACTS]
    artifacts = artifact_inventory(artifact_paths, relative_to=output_dir)
    completion = {
        "schema_version": "care-opd-qzbc-complete-v1",
        "status": "COMPLETE",
        "model_label": args.model_label,
        "rows": counts["rows"],
        "unique_ids": counts["rows"],
        "run_fingerprint": run_fingerprint,
        "model_fingerprint": model["model_fingerprint"],
        "model_canonical_path": model["canonical_path"],
        "suite_fingerprint": suite["suite_fingerprint"],
        "data_contract_fingerprint": data_audit["data_contract_fingerprint"],
        "artifact_names": list(OUTPUT_ARTIFACTS),
        "artifacts": artifacts,
        "artifacts_fingerprint": canonical_hash(artifacts),
        "completed_at_utc": utc_now(),
    }
    atomic_json(output_dir / "COMPLETE.json", completion)
    if not completed_valid(
        output_dir,
        expected_run_fingerprint=run_fingerprint,
        request_by_id=request_by_id,
    ):
        (output_dir / "COMPLETE.json").unlink(missing_ok=True)
        raise RuntimeError("newly published QZBC grouped completion failed validation")
    return evaluation_summary


def _suite_model_payload(
    *,
    label: str,
    model_dir: Path,
    request_by_id: Mapping[str, Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    run_fp = str(run_manifest.get("run_fingerprint") or "")
    if not completed_valid(
        model_dir,
        expected_run_fingerprint=run_fp,
        request_by_id=request_by_id,
    ):
        raise ValueError(f"{label}: incomplete or stale formal result")
    pairs, payload = paired_summary._model_pairs(
        request_by_id,
        model_dir / "qzbc_rollouts.jsonl",
        clean_variant="clean",
    )
    completion = read_json(model_dir / "COMPLETE.json")
    payload["identity_binding"] = {
        "status": "PASS",
        "run_fingerprint": run_fp,
        "model_canonical_path": run_manifest["model_canonical_path"],
        "model_fingerprint": run_manifest["model_fingerprint"],
        "complete_path": str((model_dir / "COMPLETE.json").resolve()),
        "complete_sha256": sha256_file(model_dir / "COMPLETE.json"),
        "artifacts_fingerprint": completion["artifacts_fingerprint"],
    }
    return pairs, payload, completion


def _family_clean_payload(
    *,
    label: str,
    model_dir: Path,
    request_by_id: Mapping[str, Mapping[str, Any]],
    family_clean_request_ids: Sequence[str],
    derived_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    selected_ids = set(family_clean_request_ids)
    subset_requests = {
        item_id: request_by_id[item_id] for item_id in family_clean_request_ids
    }
    source_rows = read_jsonl(model_dir / "qzbc_rollouts.jsonl")
    source_by_id = {str(row.get("id") or ""): row for row in source_rows}
    if len(source_by_id) != len(source_rows) or not selected_ids.issubset(source_by_id):
        raise ValueError(f"{label}: cannot derive complete family-clean rollout view")
    derived_path = derived_root / f"{label}.jsonl"
    atomic_text(
        derived_path,
        "".join(
            json.dumps(source_by_id[item_id], ensure_ascii=False, sort_keys=True) + "\n"
            for item_id in family_clean_request_ids
        ),
    )
    pairs, payload = paired_summary._model_pairs(
        subset_requests,
        derived_path,
        clean_variant="clean",
    )
    payload["view_provenance"] = {
        "status": "PASS",
        "name": "family_clean_result_blind_sensitivity",
        "source_rollouts_path": str((model_dir / "qzbc_rollouts.jsonl").resolve()),
        "source_rollouts_sha256": sha256_file(model_dir / "qzbc_rollouts.jsonl"),
        "derived_rollouts_path": str(derived_path.resolve()),
        "derived_rollouts_sha256": sha256_file(derived_path),
        "request_ids_fingerprint": canonical_hash(list(family_clean_request_ids)),
        "selection_uses_model_outputs": False,
        "retrospective": True,
    }
    return pairs, payload, derived_path


def summarize_suite(args: argparse.Namespace) -> dict[str, Any]:
    _analysis_spec()
    lock = _static_identity(args.preregistration_lock)
    suite = validate_model_suite(args.suite)
    if suite["suite_fingerprint"] != lock["model_suite_fingerprint"]:
        raise ValueError("model suite differs from preregistration")
    data_audit = _load_and_match_data_audit(args, full_recompute=True)
    _request_rows, request_by_id = _expected_requests(args.requests)
    models: dict[str, Any] = {}
    full_entity_models: dict[str, Any] = {}
    pairs_by_model: dict[str, list[dict[str, Any]]] = {}
    completions: dict[str, Any] = {}
    formal_root = args.output_root.resolve() / "formal"
    (formal_root / "SUITE_COMPLETE.json").unlink(missing_ok=True)
    derived_root = formal_root / "family_clean_views"
    family_clean_request_ids = tuple(data_audit["family_clean_view"]["request_ids"])
    if (
        len(family_clean_request_ids) != 3900
        or canonical_hash(list(family_clean_request_ids))
        != data_audit["family_clean_view"]["request_ids_fingerprint"]
    ):
        raise ValueError("frozen family-clean request ID view does not close")
    derived_paths: list[Path] = []
    for label in MODEL_ORDER:
        model_dir = formal_root / label
        manifest = read_json(model_dir / "run_manifest.json")
        if manifest.get("model_label") != label:
            raise ValueError(f"{label}: run manifest label mismatch")
        entry = suite_entry(suite, label)
        if (
            manifest.get("model_fingerprint") != entry["model_fingerprint"]
            or manifest.get("model_canonical_path") != entry["canonical_path"]
            or manifest.get("model_suite_fingerprint") != suite["suite_fingerprint"]
            or manifest.get("data_contract_fingerprint")
            != data_audit["data_contract_fingerprint"]
            or manifest.get("preregistration_lock_sha256") != lock["sha256"]
            or manifest.get("prepared_marker_sha256")
            != sha256_file(args.prepared_marker)
            or manifest.get("prepared_fingerprint")
            != read_json(args.prepared_marker)["prepared_fingerprint"]
        ):
            raise ValueError(f"{label}: formal run identity differs from suite/data/lock")
        _full_pairs, full_payload, completion = _suite_model_payload(
            label=label,
            model_dir=model_dir,
            request_by_id=request_by_id,
            run_manifest=manifest,
        )
        pairs, payload, derived_path = _family_clean_payload(
            label=label,
            model_dir=model_dir,
            request_by_id=request_by_id,
            family_clean_request_ids=family_clean_request_ids,
            derived_root=derived_root,
        )
        models[label] = payload
        full_entity_models[label] = full_payload
        pairs_by_model[label] = pairs
        derived_paths.append(derived_path)
        completions[label] = {
            "path": str((model_dir / "COMPLETE.json").resolve()),
            "sha256": sha256_file(model_dir / "COMPLETE.json"),
            "run_fingerprint": completion["run_fingerprint"],
            "artifacts_fingerprint": completion["artifacts_fingerprint"],
        }

    comparisons = {
        f"care_opd_vs_{baseline}": paired_summary._bootstrap_delta(
            pairs_by_model["care_opd"], pairs_by_model[baseline]
        )
        for baseline in ("clean_sft", "gkd_full")
    }
    records = []
    for label in MODEL_ORDER:
        item = models[label]
        metric = item["unified_metrics"]
        diagnostics = item["diagnostics"]
        records.append(
            {
                "model": label,
                "benchmark": "QZBC grouped family-clean test",
                "track": "identity-bound family-clean paired clean→recovery variants",
                "track_provenance": (
                    "Internal formal / retrospective result-blind provenance-family filter"
                ),
                "official_or_derived": "internal_formal_derived",
                "evidence_role": "in-domain mechanism contamination-sensitivity",
                "clean_n": int(metric["clean_unique_units"]),
                "perturbed_n": int(metric["paired_units"]),
                "unified_metrics": metric,
                "full_view_n": int(full_entity_models[label]["diagnostics"]["rollouts"]),
                "full_view_tsr": float(
                    full_entity_models[label]["diagnostics"][
                        "overall_mixed_task_success_rate"
                    ]
                ),
                "effectiveness_display_key": "QZBC perturbed TSR",
                "robustness_display_key": "QZBC retention",
                "notes": (
                    "Formal checkpoint-bound rerun; main internal view uses a fixed result-blind "
                    "3900-request family-clean filter. It is retrospective and shares the same "
                    "generator/templates/tasks, so it is not a preregistered external or OOD test."
                ),
            }
        )
    result = {
        "schema_version": "care-opd-qzbc-grouped-suite-v1",
        "status": "COMPLETE",
        "created_at_utc": utc_now(),
        "suite_fingerprint": suite["suite_fingerprint"],
        "preregistration_lock": lock,
        "preregistration_lock_sha256": lock["sha256"],
        "track_provenance": (
            "Internal formal / identity-bound with retrospective result-blind family-clean view"
        ),
        "claim_scope": data_audit["claim_scope"],
        "checkpoint_identity_binding": {
            "status": "PASS_CURRENT_CHECKPOINT_BYTES",
            "model_suite": {
                key: suite[key]
                for key in ("path", "sha256", "suite_fingerprint", "models")
            },
            "model_completions": completions,
        },
        "split_integrity_binding": {
            "status": "PASS_CURRENT_SPLIT_AND_TEST_RECOMPILE",
            "data_audit_path": str(args.data_audit.resolve(strict=True)),
            "data_audit_sha256": sha256_file(args.data_audit),
            "data_contract_fingerprint": data_audit["data_contract_fingerprint"],
            "prepared_marker_path": str(args.prepared_marker.resolve(strict=True)),
            "prepared_marker_sha256": sha256_file(args.prepared_marker),
            "recomputed_disjointness": data_audit["recomputed_disjointness"],
            "timing_disclosure": (
                "The legacy internal outcomes were already observed before this identity-bound "
                "rerun; use it to establish artifact attribution, not as a blind new benchmark."
            ),
            "training_provenance": {
                "status": "CURRENT_DATA_AUDITED_TRAINING_BYTES_UNVERIFIED",
                "reason": (
                    "This evaluator audits current seed/compiled files and split membership. "
                    "It does not read a contemporaneous cryptographic training-input receipt, "
                    "so it cannot retroactively prove the exact bytes used by historical checkpoints."
                ),
            },
        },
        "family_clean_view": data_audit["family_clean_view"],
        "requests": data_audit["requests"],
        "unified_metric_schema": paired_summary._validate_schema(),
        "clean_variant": "clean",
        "models": models,
        "full_entity_view": {
            "status": "DIAGNOSTIC_SOURCE_FAMILY_OVERLAP",
            "models": full_entity_models,
        },
        "primary_pairwise_comparisons": comparisons,
        "main_table_records": records,
    }
    result["summary_fingerprint"] = canonical_hash(result)
    atomic_json(formal_root / "suite_summary.json", result)

    fieldnames = [
        "model",
        "episodes",
        "overall_mixed_task_success_rate",
        "clean_task_success_rate",
        "perturbed_task_success_rate",
        "conditional_retention",
        "regression_rate",
        "robustness_drop_pp",
        "prediction_consistency",
        "worst_perturbation_tsr",
        "clean_correct_perturbed_wrong",
        "clean_wrong_perturbed_correct",
        "paired_units",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for label in MODEL_ORDER:
        metric = models[label]["unified_metrics"]
        diagnostics = models[label]["diagnostics"]
        writer.writerow(
            {
                "model": label,
                "episodes": diagnostics["rollouts"],
                "overall_mixed_task_success_rate": diagnostics[
                    "overall_mixed_task_success_rate"
                ],
                **{key: metric.get(key) for key in fieldnames[3:]},
            }
        )
    atomic_text(formal_root / "suite_summary.csv", stream.getvalue())
    lines = [
        "# Identity-bound QZBC grouped evaluation",
        "",
        "Current entity-level split proof: **PASS**. Checkpoint/artifact binding: **PASS**.",
        "",
        "The table uses the retrospective result-blind 3,900-request family-clean view. "
        "The full 4,200-row entity view has source-family overlap and remains diagnostic.",
        "",
        "| Model | Mixed TSR | Clean TSR | Perturbed TSR | Retention | Regression | Worst perturbation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    pct = lambda value: "N/A" if value is None else f"{100.0 * float(value):.2f}%"
    for label in MODEL_ORDER:
        metric = models[label]["unified_metrics"]
        diagnostic = models[label]["diagnostics"]
        lines.append(
            f"| {label} | {pct(diagnostic['overall_mixed_task_success_rate'])} | "
            f"{pct(metric['clean_task_success_rate'])} | "
            f"{pct(metric['perturbed_task_success_rate'])} | "
            f"{pct(metric['conditional_retention'])} | {pct(metric['regression_rate'])} | "
            f"{pct(metric['worst_perturbation_tsr'])} |"
        )
    atomic_text(formal_root / "suite_summary.md", "\n".join(lines) + "\n")
    suite_artifacts = artifact_inventory(
        [
            formal_root / "suite_summary.json",
            formal_root / "suite_summary.csv",
            formal_root / "suite_summary.md",
            *derived_paths,
        ],
        relative_to=formal_root,
    )
    atomic_json(
        formal_root / "SUITE_COMPLETE.json",
        {
            "schema_version": "care-opd-qzbc-suite-complete-v1",
            "status": "COMPLETE",
            "suite_fingerprint": suite["suite_fingerprint"],
            "data_contract_fingerprint": data_audit["data_contract_fingerprint"],
            "preregistration_lock_sha256": lock["sha256"],
            "models": list(MODEL_ORDER),
            "family_clean_request_ids_fingerprint": data_audit["family_clean_view"][
                "request_ids_fingerprint"
            ],
            "artifacts": suite_artifacts,
            "artifacts_fingerprint": canonical_hash(suite_artifacts),
            "completed_at_utc": utc_now(),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qzbc-root", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--compiled-train-manifest", type=Path, required=True)
    parser.add_argument("--compiled-dev-manifest", type=Path, required=True)
    parser.add_argument("--compiled-test-manifest", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source-suite", type=Path, required=True)
    prepare_parser.add_argument("--suite", type=Path, required=True)
    prepare_parser.add_argument("--preregistration-lock", type=Path, required=True)
    prepare_parser.add_argument("--prepared-marker", type=Path, required=True)
    _add_data_args(prepare_parser)

    evaluate_parser = sub.add_parser("evaluate-model")
    evaluate_parser.add_argument("--suite", type=Path, required=True)
    evaluate_parser.add_argument("--preregistration-lock", type=Path, required=True)
    evaluate_parser.add_argument("--output-root", type=Path, required=True)
    evaluate_parser.add_argument("--prepared-marker", type=Path, required=True)
    evaluate_parser.add_argument("--model-label", choices=MODEL_ORDER, required=True)
    evaluate_parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    evaluate_parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    _add_data_args(evaluate_parser)

    summary_parser = sub.add_parser("summarize-suite")
    summary_parser.add_argument("--suite", type=Path, required=True)
    summary_parser.add_argument("--preregistration-lock", type=Path, required=True)
    summary_parser.add_argument("--output-root", type=Path, required=True)
    summary_parser.add_argument("--prepared-marker", type=Path, required=True)
    _add_data_args(summary_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "evaluate-model":
        _normalize_execution_topology(args)
        evaluate_model(args)
    elif args.command == "summarize-suite":
        summarize_suite(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
