#!/usr/bin/env python3
"""Integrity guard and deterministic summaries for the TROP-D backfill.

This module never trains a model and never constructs a replacement test set.
It binds a new TROP-D evaluation to the already frozen QZBC and real-thyroid
inputs, validates completed artifacts, and refuses cross-protocol aggregation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "care-opd-tropd-backfill-v1"
QZBC_LABELS = ("tropd_full", "care_opd")
REAL_TRACKS = {
    "structured": {
        "evaluation_track": "structured_evidence_faithfulness",
        "families": 65,
        "cases": 260,
        "gold_fields": 586,
    },
    "raw": {
        "evaluation_track": "raw_strict_extraction_challenge",
        "families": 54,
        "cases": 216,
        "gold_fields": 163,
    },
}
HASH_FIELDS = (
    "verifier_sha256",
    "evaluator_sha256",
    "care_opd_qzbc_contract_sha256",
)


def sha256_file(path: Path) -> str:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {resolved}")
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def immutable_json(path: Path, value: Mapping[str, Any], *, fingerprint_key: str) -> None:
    """Create a lock once, or require an existing lock to close identically."""

    if path.exists():
        existing = read_json(path)
        if existing.get(fingerprint_key) != value.get(fingerprint_key):
            raise ValueError(
                f"immutable lock changed at {path}; choose a new BACKFILL_ROOT"
            )
        return
    atomic_json(path, value)


def _install_project_imports(project_root: Path) -> None:
    import sys

    roots = (
        project_root / "src",
        project_root / "public_eval",
        project_root / "public_eval" / "qzbc_grouped",
    )
    for root in roots:
        text = str(root.resolve(strict=True))
        if text not in sys.path:
            sys.path.insert(0, text)


def _model_identity(model_path: Path, *, project_root: Path) -> dict[str, Any]:
    """Use the v5 evaluator's checkpoint fingerprint for both track families."""

    import sys

    evaluator_root = project_root / "mechanism_experiments" / "real_thyroid_external"
    text = str(evaluator_root.resolve(strict=True))
    if text not in sys.path:
        sys.path.insert(0, text)
    from thyroid_external_eval import fingerprint_local_model

    identity = fingerprint_local_model(model_path)
    return {
        "canonical_path": identity["canonical_path_private"],
        "model_fingerprint": identity["fingerprint"],
        "inventory": identity["files_private"],
        "file_count": identity["file_count"],
        "bytes": identity["bytes"],
        "fingerprint_scheme": "thyroid_external_eval.fingerprint_local_model",
    }


def _code_identity(paths: Sequence[Path], *, relative_to: Path) -> dict[str, Any]:
    inventory = []
    for path in paths:
        resolved = path.resolve(strict=True)
        inventory.append(
            {
                "path": resolved.relative_to(relative_to.resolve(strict=True)).as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {"inventory": inventory, "fingerprint": canonical_hash(inventory)}


def _validate_model_lock(path: Path) -> dict[str, Any]:
    lock = read_json(path)
    contract = {
        key: value
        for key, value in lock.items()
        if key not in {"contract_fingerprint", "created_at_utc"}
    }
    observed = canonical_hash(contract)
    if lock.get("contract_fingerprint") != observed:
        raise ValueError(f"model lock fingerprint does not close: {path}")
    if set(lock.get("models", {})) != set(QZBC_LABELS):
        raise ValueError("model lock must contain tropd_full then care_opd")
    for label in QZBC_LABELS:
        model = lock["models"][label]
        inventory = model.get("inventory")
        if not isinstance(inventory, list) or not inventory:
            raise ValueError(f"{label}: model lock inventory is absent")
        if canonical_hash(inventory) != model.get("model_fingerprint"):
            raise ValueError(f"{label}: model lock inventory fingerprint does not close")
    return lock


def prepare_model_lock(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve(strict=True)
    _install_project_imports(project_root)
    contract = {
        "schema_version": SCHEMA + ".model_contract",
        "fingerprint_scheme": "thyroid_external_eval.fingerprint_local_model",
        "models": {
            "tropd_full": _model_identity(args.tropd_model, project_root=project_root),
            "care_opd": _model_identity(args.care_model, project_root=project_root),
        },
    }
    payload = {
        **contract,
        "contract_fingerprint": canonical_hash(contract),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    immutable_json(args.output, payload, fingerprint_key="contract_fingerprint")
    _validate_model_lock(args.output)
    print(
        "MODEL_LOCK=PASS "
        f"tropd={payload['models']['tropd_full']['model_fingerprint']} "
        f"care={payload['models']['care_opd']['model_fingerprint']}"
    )
    return 0


def _validate_qzbc_lock(path: Path) -> dict[str, Any]:
    lock = read_json(path)
    contract = {
        key: value
        for key, value in lock.items()
        if key not in {"contract_fingerprint", "created_at_utc"}
    }
    if lock.get("contract_fingerprint") != canonical_hash(contract):
        raise ValueError(f"QZBC protocol lock fingerprint does not close: {path}")
    if set(lock.get("models", {})) != set(QZBC_LABELS):
        raise ValueError("QZBC protocol lock model set changed")
    for label in QZBC_LABELS:
        model = lock["models"][label]
        if canonical_hash(model.get("inventory")) != model.get("model_fingerprint"):
            raise ValueError(f"{label}: QZBC lock model inventory does not close")
    return lock


def _validate_data_audit_payload(audit: Mapping[str, Any]) -> None:
    deterministic = {
        key: value
        for key, value in audit.items()
        if key not in {"schema_version", "data_contract_fingerprint"}
    }
    if audit.get("schema_version") != "care-opd-qzbc-data-audit-v1" or audit.get(
        "data_contract_fingerprint"
    ) != canonical_hash(deterministic):
        raise ValueError("QZBC DATA_AUDIT fingerprint does not close")


def prepare_qzbc(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve(strict=True)
    _install_project_imports(project_root)
    from protocol import (
        EXPECTED_REQUEST_ROWS,
        EXPECTED_SEED,
        FORMAL_FAMILY_CLEAN_CONTRACT,
        audit_data_contract,
    )
    from runtime_identity import runtime_identity

    audit = audit_data_contract(
        qzbc_root=args.qzbc_root,
        requests_path=args.requests,
        seed_manifest_path=args.seed_manifest,
        compiled_manifests={
            "train": args.compiled_train_manifest,
            "dev": args.compiled_dev_manifest,
            "test": args.compiled_test_manifest,
        },
    )
    _validate_data_audit_payload(audit)
    if audit["requests"]["rows"] != EXPECTED_REQUEST_ROWS:
        raise ValueError("QZBC request count changed")
    if audit["family_clean_view"]["request_rows"] != int(
        FORMAL_FAMILY_CLEAN_CONTRACT["request_rows"]
    ):
        raise ValueError("QZBC family-clean view changed")

    model_lock = _validate_model_lock(args.model_lock)
    model_identities = model_lock["models"]
    code_paths = (
        Path(__file__),
        Path(__file__).with_name("run_tropd_backfill.sh"),
        project_root / "src" / "care_opd" / "qzbc_rollout.py",
        project_root / "src" / "care_opd" / "qzbc_contract.py",
        project_root / "src" / "care_opd" / "rollout.py",
        project_root / "public_eval" / "summarize_internal_qzbc.py",
        project_root / "public_eval" / "qzbc_grouped" / "protocol.py",
        project_root / "public_eval" / "qzbc_grouped" / "ANALYSIS_SPEC.json",
    )
    contract = {
        "schema_version": SCHEMA + ".qzbc_contract",
        "evaluation": "QZBC grouped identity-bound backfill",
        "view_policy": {
            "primary": "3900-request retrospective result-blind family-clean view",
            "diagnostic": "4200-request entity-held-out full view",
            "no_cross_view_mixing": True,
        },
        "data_contract_fingerprint": audit["data_contract_fingerprint"],
        "requests_sha256": audit["requests"]["sha256"],
        "requests_rows": audit["requests"]["rows"],
        "family_clean_request_ids_fingerprint": audit["family_clean_view"][
            "request_ids_fingerprint"
        ],
        "family_clean_request_rows": audit["family_clean_view"]["request_rows"],
        "models": model_identities,
        "generation": {
            "dtype": "bfloat16",
            "do_sample": False,
            "max_new_tokens": 1200,
            "seed": int(EXPECTED_SEED),
            "trust_remote_code": False,
            "max_examples": -1,
            "max_format_failure_fraction": 0.25,
            "num_workers": 1,
            "logical_device": args.logical_device,
            "physical_gpu": args.physical_gpu,
        },
        "code_identity": _code_identity(code_paths, relative_to=project_root),
        "runtime_identity": runtime_identity(),
    }
    if getattr(args, "durable_checkpoint", False):
        contract["checkpoint_policy"] = {
            "version": "qzbc_durable_candidates_v1",
            "persistence": "per_request_fsync_before_progress",
            "resume_selection": "all_first_outputs_retained_including_failures",
            "legacy_partial_import": False,
        }
    contract_fingerprint = canonical_hash(contract)
    payload = {
        **contract,
        "contract_fingerprint": contract_fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    immutable_json(
        args.output_dir / "protocol_lock.json",
        payload,
        fingerprint_key="contract_fingerprint",
    )
    audit_path = args.output_dir / "DATA_AUDIT.json"
    if audit_path.exists():
        existing = read_json(audit_path)
        _validate_data_audit_payload(existing)
        if existing.get("data_contract_fingerprint") != audit[
            "data_contract_fingerprint"
        ]:
            raise ValueError("existing QZBC DATA_AUDIT differs; use a new output root")
    else:
        atomic_json(audit_path, audit)
    print(
        "QZBC_PROTOCOL_LOCK=PASS "
        f"requests={audit['requests']['rows']} "
        f"family_clean={audit['family_clean_view']['request_rows']} "
        f"fingerprint={contract_fingerprint}"
    )
    return 0


def _request_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = str(row.get("id") or "")
        if not item_id or item_id in result:
            raise ValueError(f"empty or duplicate request id in {path}: {item_id!r}")
        result[item_id] = row
    return result


def _rollout_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = str(row.get("id") or "")
        if not item_id or item_id in result:
            raise ValueError(f"empty or duplicate rollout id in {path}: {item_id!r}")
        result[item_id] = row
    return result


def _qzbc_model_contract(
    lock: Mapping[str, Any], label: str, output_dir: Path
) -> dict[str, Any]:
    if label not in QZBC_LABELS:
        raise ValueError(f"unsupported QZBC label: {label}")
    model = lock["models"][label]
    contract = {
        "schema_version": SCHEMA + ".qzbc_model_run",
        "protocol_fingerprint": lock["contract_fingerprint"],
        "model_label": label,
        "model_canonical_path": model["canonical_path"],
        "model_fingerprint": model["model_fingerprint"],
        "requests_sha256": lock["requests_sha256"],
        "requests_rows": lock["requests_rows"],
        "generation": lock["generation"],
        "output_dir": str(output_dir.resolve()),
    }
    contract["run_fingerprint"] = canonical_hash(contract)
    return contract


def _validate_qzbc_rows(
    requests_path: Path, output_dir: Path, expected_rows: int
) -> dict[str, Any]:
    requests = _request_index(requests_path)
    rollouts_path = output_dir / "qzbc_rollouts.jsonl"
    rollouts = _rollout_index(rollouts_path)
    if set(requests) != set(rollouts):
        raise ValueError("QZBC request and rollout IDs differ")
    if len(rollouts) != expected_rows:
        raise ValueError(f"QZBC rollout count changed: {len(rollouts)} != {expected_rows}")
    for item_id, request in requests.items():
        if str(rollouts[item_id].get("variant") or "") != str(
            request.get("variant") or ""
        ):
            raise ValueError(f"{item_id}: rollout/request variant mismatch")
    summary = read_json(output_dir / "summary.json")
    if int(summary.get("rollouts", -1)) != len(rollouts):
        raise ValueError("QZBC summary rollout count mismatch")
    passed = sum(bool(row.get("passed")) for row in rollouts.values())
    if int(summary.get("passed", -1)) != passed:
        raise ValueError("QZBC summary pass count mismatch")
    return {
        "rows": len(rollouts),
        "passed": passed,
        "rollouts_sha256": sha256_file(rollouts_path),
        "summary_sha256": sha256_file(output_dir / "summary.json"),
    }


def finalize_qzbc_model(args: argparse.Namespace) -> int:
    lock = _validate_qzbc_lock(args.lock)
    if lock.get("requests_sha256") != sha256_file(args.requests):
        raise ValueError("QZBC request SHA-256 differs from protocol lock")
    counts = _validate_qzbc_rows(
        args.requests, args.output_dir, int(lock["requests_rows"])
    )
    contract = _qzbc_model_contract(lock, args.label, args.output_dir)
    manifest = {
        **contract,
        "status": "COMPLETE",
        "counts": counts,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_dir / "run_manifest.json", manifest)
    artifacts = []
    for name in ("qzbc_rollouts.jsonl", "qzbc_opd.jsonl", "qzbc_dpo.jsonl", "summary.json", "run_manifest.json"):
        path = args.output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append(
            {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    completion = {
        "schema_version": SCHEMA + ".qzbc_complete",
        "status": "COMPLETE",
        "model_label": args.label,
        "run_fingerprint": contract["run_fingerprint"],
        "protocol_fingerprint": lock["contract_fingerprint"],
        "artifacts": artifacts,
        "artifacts_fingerprint": canonical_hash(artifacts),
    }
    atomic_json(args.output_dir / "COMPLETE.json", completion)
    print(
        f"QZBC_MODEL_COMPLETE=PASS label={args.label} rows={counts['rows']} "
        f"passed={counts['passed']}"
    )
    return 0


def verify_qzbc_model(args: argparse.Namespace) -> int:
    lock = _validate_qzbc_lock(args.lock)
    manifest = read_json(args.output_dir / "run_manifest.json")
    completion = read_json(args.output_dir / "COMPLETE.json")
    expected = _qzbc_model_contract(lock, args.label, args.output_dir)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"QZBC manifest mismatch for {key}")
    counts = _validate_qzbc_rows(
        args.requests, args.output_dir, int(lock["requests_rows"])
    )
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("QZBC completion artifact inventory is absent")
    for item in artifacts:
        path = args.output_dir / str(item["name"])
        if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            raise ValueError(f"QZBC artifact changed: {path}")
    if completion.get("artifacts_fingerprint") != canonical_hash(artifacts):
        raise ValueError("QZBC completion artifact fingerprint does not close")
    if completion.get("run_fingerprint") != expected["run_fingerprint"]:
        raise ValueError("QZBC completion run fingerprint mismatch")
    print(
        f"QZBC_MODEL_VERIFY=PASS label={args.label} rows={counts['rows']} "
        f"passed={counts['passed']}"
    )
    return 0


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
    )


def summarize_qzbc(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve(strict=True)
    _install_project_imports(project_root)
    import summarize_internal_qzbc as paired

    lock = _validate_qzbc_lock(args.lock)
    audit = read_json(args.data_audit)
    _validate_data_audit_payload(audit)
    if audit.get("data_contract_fingerprint") != lock.get("data_contract_fingerprint"):
        raise ValueError("QZBC audit/lock fingerprint mismatch")
    requests = _request_index(args.requests)
    family_ids = tuple(audit["family_clean_view"]["request_ids"])
    if canonical_hash(list(family_ids)) != lock["family_clean_request_ids_fingerprint"]:
        raise ValueError("QZBC family-clean IDs differ from protocol lock")
    subset_requests = {item_id: requests[item_id] for item_id in family_ids}

    full_models: dict[str, Any] = {}
    family_models: dict[str, Any] = {}
    family_pairs: dict[str, list[dict[str, Any]]] = {}
    derived_root = args.output_dir / "family_clean_views"
    derived_root.mkdir(parents=True, exist_ok=True)
    for label in QZBC_LABELS:
        model_dir = args.eval_root / "formal" / label
        verify_args = argparse.Namespace(
            lock=args.lock,
            requests=args.requests,
            label=label,
            output_dir=model_dir,
        )
        verify_qzbc_model(verify_args)
        full_pairs, full_payload = paired._model_pairs(
            requests, model_dir / "qzbc_rollouts.jsonl", clean_variant="clean"
        )
        del full_pairs
        source = _rollout_index(model_dir / "qzbc_rollouts.jsonl")
        derived_path = derived_root / f"{label}.jsonl"
        _write_jsonl(derived_path, [source[item_id] for item_id in family_ids])
        pairs, payload = paired._model_pairs(
            subset_requests, derived_path, clean_variant="clean"
        )
        identity = lock["models"][label]
        binding = {
            "protocol_fingerprint": lock["contract_fingerprint"],
            "model_fingerprint": identity["model_fingerprint"],
            "source_rollouts_sha256": sha256_file(model_dir / "qzbc_rollouts.jsonl"),
            "derived_rollouts_sha256": sha256_file(derived_path),
        }
        full_payload["identity_binding"] = binding
        payload["identity_binding"] = binding
        full_models[label] = full_payload
        family_models[label] = payload
        family_pairs[label] = pairs

    comparison = paired._bootstrap_delta(
        family_pairs["tropd_full"], family_pairs["care_opd"]
    )
    # The reused helper historically names its first/second arguments
    # ``care_opd``/``baseline``.  Relabel those generic positions explicitly so
    # a TROP-D backfill cannot be misread as a CARE-OPD delta.
    comparison["candidate_label"] = "tropd_full"
    comparison["reference_label"] = "care_opd"
    for values in comparison.get("metrics", {}).values():
        values["candidate_value"] = values.pop("care_opd")
        values["reference_value"] = values.pop("baseline")
        values["delta_candidate_minus_reference"] = values.pop("delta")
    comparison["exploratory_unclustered_mcnemar_perturbed_success"][
        "a_label"
    ] = "tropd_full"
    comparison["exploratory_unclustered_mcnemar_perturbed_success"][
        "b_label"
    ] = "care_opd"
    result = {
        "schema_version": SCHEMA + ".qzbc_summary",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_fingerprint": lock["contract_fingerprint"],
        "requests": {
            "sha256": lock["requests_sha256"],
            "full_rows": lock["requests_rows"],
            "family_clean_rows": lock["family_clean_request_rows"],
        },
        "primary_view": {
            "name": "family_clean_result_blind_sensitivity",
            "provenance": audit["family_clean_view"]["provenance"],
            "models": family_models,
            "paired_comparisons": {"tropd_full_vs_care_opd": comparison},
        },
        "diagnostic_full_entity_view": {
            "warning": "Do not mix this 4200-row view with the 3900-row family-clean main view.",
            "models": full_models,
        },
    }
    result["summary_fingerprint"] = canonical_hash(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "suite_summary.json", result)

    stream = io.StringIO(newline="")
    fields = (
        "model",
        "episodes",
        "mixed_tsr",
        "clean_tsr",
        "perturbed_tsr",
        "conditional_retention",
        "regression_rate",
        "worst_perturbation_tsr",
    )
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for label in QZBC_LABELS:
        payload = family_models[label]
        metric = payload["unified_metrics"]
        writer.writerow(
            {
                "model": label,
                "episodes": payload["diagnostics"]["rollouts"],
                "mixed_tsr": payload["diagnostics"]["overall_mixed_task_success_rate"],
                "clean_tsr": metric["clean_task_success_rate"],
                "perturbed_tsr": metric["perturbed_task_success_rate"],
                "conditional_retention": metric["conditional_retention"],
                "regression_rate": metric["regression_rate"],
                "worst_perturbation_tsr": metric["worst_perturbation_tsr"],
            }
        )
    atomic_text(args.output_dir / "suite_summary.csv", stream.getvalue())
    print(
        "QZBC_SUMMARY=PASS "
        f"full={lock['requests_rows']} family_clean={lock['family_clean_request_rows']}"
    )
    return 0


def _real_stable_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "input",
        "evaluation_track",
        "challenge_mode",
        "generation",
        "verifier_sha256",
        "evaluator_sha256",
        "care_opd_qzbc_contract_sha256",
        "bootstrap",
    )
    missing = [key for key in keys if key not in manifest or manifest.get(key) is None]
    if missing:
        raise ValueError(f"real manifest lacks required stable fields: {missing}")
    if manifest.get("schema_version") != "care_opd.real_thyroid_external_eval.v4.run_contract":
        raise ValueError("real manifest schema is not the locked v5 evaluator contract")
    if not isinstance(manifest.get("input"), Mapping) or not isinstance(
        manifest.get("generation"), Mapping
    ) or not isinstance(manifest.get("bootstrap"), Mapping):
        raise ValueError("real manifest input/generation/bootstrap must be objects")
    input_contract = manifest["input"]
    for key in ("seed_sha256", "field_gold_sha256", "input_fingerprint"):
        value = str(input_contract.get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"real manifest has invalid input hash: {key}")
    for key in HASH_FIELDS:
        value = str(manifest.get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"real manifest has invalid code hash: {key}")
    return {key: manifest[key] for key in keys}


def inspect_real_inputs(args: argparse.Namespace) -> int:
    """Load the frozen v5 inputs through the evaluator's own contract code."""

    expected = REAL_TRACKS[args.track]
    project_root = args.project_root.resolve(strict=True)
    _install_project_imports(project_root)
    import sys

    evaluator_root = project_root / "mechanism_experiments" / "real_thyroid_external"
    sys.path.insert(0, str(evaluator_root.resolve(strict=True)))
    from thyroid_external_eval import load_locked_inputs

    rows, contract = load_locked_inputs(
        args.seeds,
        field_gold_path=args.field_gold,
        qzbc_root=args.qzbc_root,
    )
    if contract.get("evaluation_track") != expected["evaluation_track"]:
        raise ValueError(
            f"{args.track}: evaluation track changed: {contract.get('evaluation_track')}"
        )
    if int(contract.get("family_count", -1)) != expected["families"]:
        raise ValueError(f"{args.track}: family count changed")
    if int(contract.get("case_count", -1)) != expected["cases"] or len(rows) != expected[
        "cases"
    ]:
        raise ValueError(f"{args.track}: case count changed")
    print(
        f"REAL_INPUT_CONTRACT=PASS track={args.track} "
        f"families={expected['families']} cases={expected['cases']} "
        f"fingerprint={contract['input_fingerprint']}"
    )
    return 0


def compare_real_manifests(args: argparse.Namespace) -> int:
    left = read_json(args.left)
    right = read_json(args.right)
    left_contract = _real_stable_contract(left)
    right_contract = _real_stable_contract(right)
    differences = [
        key for key in left_contract if left_contract[key] != right_contract[key]
    ]
    payload = {
        "schema_version": SCHEMA + ".real_compatibility",
        "status": "PASS" if not differences else "INCOMPATIBLE",
        "left": {"path": str(args.left.resolve()), "sha256": sha256_file(args.left)},
        "right": {"path": str(args.right.resolve()), "sha256": sha256_file(args.right)},
        "stable_contract_fingerprint_left": canonical_hash(left_contract),
        "stable_contract_fingerprint_right": canonical_hash(right_contract),
        "different_fields": differences,
        "descriptive_side_by_side_allowed": not differences,
        "automatic_metric_splicing_performed": False,
        "note": (
            "A PASS permits protocol-matched interpretation. Paired deltas still require "
            "case-level outputs under the same family bootstrap; this helper never pastes "
            "numbers from incompatible summaries."
        ),
    }
    if args.output:
        atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not differences else 2


def verify_real(args: argparse.Namespace) -> int:
    expected = REAL_TRACKS[args.track]
    project_root = Path(__file__).resolve().parents[2]
    private_manifest_path = args.output_dir / "private" / "run_manifest.private.json"
    public_manifest_path = args.output_dir / "public" / "run_manifest.json"
    summary_path = args.output_dir / "public" / "summary.json"
    private_manifest = read_json(private_manifest_path)
    public_manifest = read_json(public_manifest_path)
    summary = read_json(summary_path)
    input_contract = private_manifest.get("input") or {}
    if input_contract.get("evaluation_track") != expected["evaluation_track"]:
        raise ValueError("real-thyroid evaluation track mismatch")
    if int(input_contract.get("family_count", -1)) != expected["families"]:
        raise ValueError("real-thyroid family count mismatch")
    if int(input_contract.get("case_count", -1)) != expected["cases"]:
        raise ValueError("real-thyroid case count mismatch")
    if input_contract.get("seed_sha256") != sha256_file(args.seeds):
        raise ValueError("real-thyroid seed SHA-256 mismatch")
    if input_contract.get("field_gold_sha256") != sha256_file(args.field_gold):
        raise ValueError("real-thyroid field-gold SHA-256 mismatch")
    models = private_manifest.get("models") or []
    labels = [str(item.get("label") or "") for item in models]
    if labels != ["tropd_full", "care_opd"]:
        raise ValueError(f"unexpected real-thyroid model order: {labels}")
    model_lock = _validate_model_lock(args.model_lock)
    expected_fingerprints = {
        label: model_lock["models"][label]["model_fingerprint"]
        for label in QZBC_LABELS
    }
    observed_fingerprints = {
        str(item["label"]): str(item.get("fingerprint") or "") for item in models
    }
    if observed_fingerprints != expected_fingerprints:
        raise ValueError("real-thyroid model fingerprints differ from the QZBC backfill lock")
    generation = private_manifest.get("generation") or {}
    expected_generation = {
        "do_sample": False,
        "max_input_tokens": 32768,
        "max_new_tokens": 2048,
        "seed": 42,
    }
    for key, value in expected_generation.items():
        if generation.get(key) != value:
            raise ValueError(f"real-thyroid generation setting changed: {key}")
    if private_manifest.get("reference_model") != "care_opd" or private_manifest.get(
        "reference_models"
    ) != ["care_opd"]:
        raise ValueError("real-thyroid reference model contract changed")
    if private_manifest.get("bootstrap") != {
        "replicates": 10000,
        "seed": 20260819,
    }:
        raise ValueError("real-thyroid bootstrap contract changed")
    current_hashes = {
        "evaluator_sha256": sha256_file(
            project_root
            / "mechanism_experiments"
            / "real_thyroid_external"
            / "thyroid_external_eval.py"
        ),
        "verifier_sha256": sha256_file(
            project_root
            / "mechanism_experiments"
            / "real_thyroid_external"
            / "thyroid_verifier.py"
        ),
        "care_opd_qzbc_contract_sha256": sha256_file(
            project_root / "src" / "care_opd" / "qzbc_contract.py"
        ),
    }
    for key, value in current_hashes.items():
        if private_manifest.get(key) != value:
            raise ValueError(f"real-thyroid current code differs from run manifest: {key}")
    if public_manifest.get("evaluation_track") != expected["evaluation_track"]:
        raise ValueError("public/private real-thyroid track mismatch")
    if int(public_manifest.get("case_count", -1)) != expected["cases"]:
        raise ValueError("public real-thyroid case count mismatch")
    if not isinstance(summary.get("models"), list) or len(summary["models"]) != 2:
        raise ValueError("real-thyroid summary is not a complete two-model result")
    report = {
        "schema_version": SCHEMA + ".real_verify",
        "status": "PASS",
        "track": args.track,
        "evaluation_track": expected["evaluation_track"],
        "families": expected["families"],
        "cases": expected["cases"],
        "seed_sha256": input_contract["seed_sha256"],
        "field_gold_sha256": input_contract["field_gold_sha256"],
        "run_fingerprint": private_manifest.get("run_fingerprint"),
        "summary_sha256": sha256_file(summary_path),
        "public_manifest_sha256": sha256_file(public_manifest_path),
        "private_manifest_sha256": sha256_file(private_manifest_path),
    }
    atomic_json(args.output_dir / "VERIFY.json", report)
    print(
        f"REAL_TRACK_VERIFY=PASS track={args.track} families={expected['families']} "
        f"cases={expected['cases']}"
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-qzbc")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--qzbc-root", type=Path, required=True)
    prepare.add_argument("--requests", type=Path, required=True)
    prepare.add_argument("--seed-manifest", type=Path, required=True)
    prepare.add_argument("--compiled-train-manifest", type=Path, required=True)
    prepare.add_argument("--compiled-dev-manifest", type=Path, required=True)
    prepare.add_argument("--compiled-test-manifest", type=Path, required=True)
    prepare.add_argument("--model-lock", type=Path, required=True)
    prepare.add_argument("--physical-gpu", required=True)
    prepare.add_argument("--logical-device", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--durable-checkpoint", action="store_true")
    prepare.set_defaults(handler=prepare_qzbc)

    models = sub.add_parser("prepare-model-lock")
    models.add_argument("--project-root", type=Path, required=True)
    models.add_argument("--tropd-model", type=Path, required=True)
    models.add_argument("--care-model", type=Path, required=True)
    models.add_argument("--output", type=Path, required=True)
    models.set_defaults(handler=prepare_model_lock)

    for name, handler in (
        ("finalize-qzbc-model", finalize_qzbc_model),
        ("verify-qzbc-model", verify_qzbc_model),
    ):
        model = sub.add_parser(name)
        model.add_argument("--lock", type=Path, required=True)
        model.add_argument("--requests", type=Path, required=True)
        model.add_argument("--label", choices=QZBC_LABELS, required=True)
        model.add_argument("--output-dir", type=Path, required=True)
        model.set_defaults(handler=handler)

    summary = sub.add_parser("summarize-qzbc")
    summary.add_argument("--project-root", type=Path, required=True)
    summary.add_argument("--lock", type=Path, required=True)
    summary.add_argument("--data-audit", type=Path, required=True)
    summary.add_argument("--requests", type=Path, required=True)
    summary.add_argument("--eval-root", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.set_defaults(handler=summarize_qzbc)

    compare = sub.add_parser("compare-real-manifests")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=compare_real_manifests)

    inspect = sub.add_parser("inspect-real-inputs")
    inspect.add_argument("--track", choices=tuple(REAL_TRACKS), required=True)
    inspect.add_argument("--project-root", type=Path, required=True)
    inspect.add_argument("--qzbc-root", type=Path, required=True)
    inspect.add_argument("--seeds", type=Path, required=True)
    inspect.add_argument("--field-gold", type=Path, required=True)
    inspect.set_defaults(handler=inspect_real_inputs)

    verify = sub.add_parser("verify-real")
    verify.add_argument("--track", choices=tuple(REAL_TRACKS), required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--seeds", type=Path, required=True)
    verify.add_argument("--field-gold", type=Path, required=True)
    verify.add_argument("--model-lock", type=Path, required=True)
    verify.set_defaults(handler=verify_real)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
