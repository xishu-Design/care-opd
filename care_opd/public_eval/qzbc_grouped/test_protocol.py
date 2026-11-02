from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import protocol
import evaluate_qzbc_grouped as evaluator


@contextmanager
def _raises(error_type, pattern: str):
    try:
        yield
    except error_type as exc:
        if not re.search(pattern, str(exc)):
            raise AssertionError(f"{exc!r} does not match {pattern!r}") from exc
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _seed(seed_id: str, split: str, asset, *, task: str = "student_analysis"):
    assets = [] if asset is None else [asset]
    return {
        "id": seed_id,
        "split": split,
        "request": {"input_text": f"request-{seed_id}"},
        "target": {"summary": f"target-{seed_id}"},
        "metadata": {"task_type": task, "curated_assets": assets},
    }


def _asset(source: str, kind: str, asset_id: str, record: str):
    return {
        "source_id": source,
        "asset_type": kind,
        "asset_id": asset_id,
        "source_record_id": record,
    }


def _compiled_id(seed_id: str, variant: str, contract: str) -> str:
    return hashlib.sha256(f"{seed_id}|{variant}|{contract}".encode()).hexdigest()[:24]


def _build_data_fixture(root: Path, *, exact_entity_overlap: bool = False):
    qzbc_root = root / "qzbc"
    contract_file = qzbc_root / "backend" / "contract.py"
    asset_file = qzbc_root / "backend" / "assets.json"
    contract_file.parent.mkdir(parents=True)
    contract_file.write_text("contract-v1", encoding="utf-8")
    asset_file.write_text("{}", encoding="utf-8")
    contract_files = {
        "backend/contract.py": protocol.sha256_file(contract_file),
    }
    contract_fp = protocol.canonical_hash(contract_files)

    train_asset = _asset("SRC_SHARED", "herb", "train-a", "REC_SHARED")
    test_shared_asset = (
        dict(train_asset)
        if exact_entity_overlap
        else _asset("SRC_SHARED", "formula", "test-b", "REC_SHARED")
    )
    split_rows = {
        "train": [_seed("train-1", "train", train_asset)],
        "dev": [_seed("dev-1", "dev", _asset("SRC_DEV", "herb", "dev-a", "REC_DEV"))],
        "test": [
            _seed("test-overlap", "test", test_shared_asset),
            _seed("test-clean", "test", _asset("SRC_CLEAN", "herb", "clean-a", "REC_CLEAN")),
            _seed("test-synthetic", "test", None),
        ],
    }
    seed_dir = root / "data" / "seeds"
    outputs = {}
    for split, rows in split_rows.items():
        path = seed_dir / f"seeds.{split}.jsonl"
        _write_jsonl(path, rows)
        outputs[split] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": protocol.sha256_file(path),
        }
    all_rows = [*split_rows["train"], *split_rows["dev"], *split_rows["test"]]
    all_path = seed_dir / "seeds.jsonl"
    _write_jsonl(all_path, all_rows)
    outputs["all"] = {
        "path": str(all_path.resolve()),
        "rows": len(all_rows),
        "sha256": protocol.sha256_file(all_path),
    }
    seed_manifest = {
        "schema_version": "care-opd-qzbc-seeds-v2",
        "seed": 42,
        "dev_ratio": 0.2,
        "test_ratio": 0.2,
        "split_counts": {
            split: {"student_analysis": len(rows)} for split, rows in split_rows.items()
        },
        "outputs": outputs,
        "qzbc_contract": {
            "contract_fingerprint": contract_fp,
            "files": contract_files,
        },
        "qzbc_grounding_assets": {
            "files": {
                "backend/assets.json": {
                    "bytes": asset_file.stat().st_size,
                    "sha256": protocol.sha256_file(asset_file),
                }
            }
        },
        "grounding_asset_split": {
            "strategy": "exact_asset_disjoint_deterministic_shuffle",
            "seed": 42,
            "overlap_check": "PASS",
        },
        "data_policy": {"knowledge_assets_are_exact_split_disjoint": True},
    }
    seed_manifest_path = seed_dir / "seeds.manifest.json"
    seed_manifest_path.write_text(json.dumps(seed_manifest), encoding="utf-8")

    compiled_manifests = {}
    test_requests = []
    for seed in split_rows["test"]:
        for variant in protocol.EXPECTED_VARIANTS:
            test_requests.append(
                {
                    "id": _compiled_id(seed["id"], variant, contract_fp),
                    "seed_id": seed["id"],
                    "variant": variant,
                    "contract_fingerprint": contract_fp,
                }
            )
    output_names = {
        "sft": "qzbc_sft.jsonl",
        "clean_sft": "qzbc_clean_sft.jsonl",
        "recovery_sft": "qzbc_recovery_sft.jsonl",
        "stream_retention_sft": "qzbc_stream_retention_sft.jsonl",
        "clean_mixed_sft": "qzbc_clean_mixed_sft.jsonl",
        "on_policy_requests": "qzbc_on_policy_requests.jsonl",
        "host_overridden_retention": "qzbc_host_overridden_retention.jsonl",
    }
    requests_path = None
    for split, seeds in split_rows.items():
        compiled_dir = root / "data" / "compiled" / split
        compiled_dir.mkdir(parents=True)
        counts = {}
        for key, filename in output_names.items():
            rows = test_requests if split == "test" and key == "on_policy_requests" else []
            path = compiled_dir / filename
            _write_jsonl(path, rows)
            counts[key] = len(rows)
            if split == "test" and key == "on_policy_requests":
                requests_path = path
        manifest = {
            "schema_version": "care-opd-qzbc-data-v1",
            "source": outputs[split]["path"],
            "source_rows": len(seeds),
            "variants": list(protocol.EXPECTED_VARIANTS),
            "counts": counts,
            "qzbc_contract": {"contract_fingerprint": contract_fp},
        }
        path = compiled_dir / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        compiled_manifests[split] = path
    assert requests_path is not None
    kept = sorted(("test-clean", "test-synthetic"))
    excluded = ["test-overlap"]
    family_request_ids = sorted(
        row["id"] for row in test_requests if row["seed_id"] in set(kept)
    )
    family_contract = {
        "seed_rows": 2,
        "request_rows": 8,
        "kept_seed_ids_fingerprint": protocol.canonical_hash(kept),
        "excluded_seed_ids_fingerprint": protocol.canonical_hash(excluded),
        "request_ids_fingerprint": protocol.canonical_hash(family_request_ids),
    }
    return {
        "qzbc_root": qzbc_root,
        "requests": requests_path,
        "seed_manifest": seed_manifest_path,
        "compiled": compiled_manifests,
        "family_contract": family_contract,
    }


def test_data_audit_recomputes_entity_and_family_clean_views(tmp_path):
    fixture = _build_data_fixture(tmp_path)
    audit = protocol.audit_data_contract(
        qzbc_root=fixture["qzbc_root"],
        requests_path=fixture["requests"],
        seed_manifest_path=fixture["seed_manifest"],
        compiled_manifests=fixture["compiled"],
        expected_request_rows=12,
        expected_family_clean_contract=fixture["family_contract"],
        expected_family_clean_task_counts=None,
        verify_test_recompile=False,
    )
    split = audit["recomputed_disjointness"]
    assert split["curated_asset_identities_pairwise_disjoint"] is True
    assert split["canonical_request_target_hashes_pairwise_disjoint"] is True
    assert split["source_record_families_pairwise_disjoint"] is False
    assert split["source_record_family_overlap_counts"]["train__test"] == 1
    assert split["synthetic_only_seed_rows_with_empty_curated_assets"]["test"] == 1
    assert audit["family_clean_view"]["request_rows"] == 8
    assert audit["family_clean_view"]["source_family_overlap_with_train_dev"] == 0
    assert audit["family_clean_view"]["provenance"].startswith("retrospective")


def test_claimed_manifest_pass_does_not_hide_exact_entity_overlap(tmp_path):
    fixture = _build_data_fixture(tmp_path, exact_entity_overlap=True)
    with _raises(ValueError, "curated asset overlap"):
        protocol.audit_data_contract(
            qzbc_root=fixture["qzbc_root"],
            requests_path=fixture["requests"],
            seed_manifest_path=fixture["seed_manifest"],
            compiled_manifests=fixture["compiled"],
            expected_request_rows=12,
            expected_family_clean_contract=fixture["family_contract"],
            expected_family_clean_task_counts=None,
            verify_test_recompile=False,
        )


def test_model_suite_binds_canonical_path_inventory_and_full_hash(tmp_path):
    entries = []
    for label in protocol.MODEL_ORDER:
        root = (tmp_path / label).resolve()
        root.mkdir()
        artifact = root / "model.safetensors"
        artifact.write_bytes(f"weights-{label}".encode())
        inventory = [
            {
                "name": artifact.name,
                "bytes": artifact.stat().st_size,
                "mtime_ns": artifact.stat().st_mtime_ns,
                "sha256": protocol.sha256_file(artifact),
            }
        ]
        content = [{key: row[key] for key in ("name", "bytes", "sha256")} for row in inventory]
        entries.append(
            {
                "label": label,
                "canonical_path": str(root),
                "inventory": inventory,
                "model_fingerprint": protocol.canonical_hash(content),
            }
        )
    suite = {
        "models": entries,
        "suite_fingerprint": protocol.canonical_hash(
            [(row["label"], row["model_fingerprint"]) for row in entries]
        ),
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    identity = protocol.validate_model_suite(
        suite_path, rehash_labels=("care_opd",)
    )
    assert identity["suite_fingerprint"] == suite["suite_fingerprint"]
    assert protocol.suite_entry(identity, "care_opd")["identity_check"] == "full_sha256"
    (tmp_path / "care_opd" / "model.safetensors").write_bytes(b"tampered-byte")
    with _raises(ValueError, "metadata changed|mtime changed|SHA-256 changed"):
        protocol.validate_model_suite(suite_path, rehash_labels=("care_opd",))


def test_complete_marker_detects_artifact_tampering(tmp_path):
    request = {"id": "row-1", "variant": "clean"}
    request_by_id = {"row-1": request}
    run_fp = "a" * 64
    protocol.atomic_json(
        tmp_path / "run_manifest.json",
        {
            "run_fingerprint": run_fp,
            "model_fingerprint": "b" * 64,
            "model_suite_fingerprint": "c" * 64,
            "data_contract_fingerprint": "d" * 64,
        },
    )
    _write_jsonl(
        tmp_path / "qzbc_rollouts.jsonl",
        [{"id": "row-1", "variant": "clean", "passed": True}],
    )
    _write_jsonl(tmp_path / "qzbc_opd.jsonl", [])
    _write_jsonl(tmp_path / "qzbc_dpo.jsonl", [])
    protocol.atomic_json(tmp_path / "summary.json", {"rollouts": 1, "passed": 1})
    protocol.atomic_json(
        tmp_path / "evaluation_summary.json", {"run_fingerprint": run_fp}
    )
    artifacts = protocol.artifact_inventory(
        [tmp_path / name for name in evaluator.OUTPUT_ARTIFACTS],
        relative_to=tmp_path,
    )
    protocol.atomic_json(
        tmp_path / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "run_fingerprint": run_fp,
            "rows": 1,
            "artifact_names": list(evaluator.OUTPUT_ARTIFACTS),
            "artifacts": artifacts,
            "artifacts_fingerprint": protocol.canonical_hash(artifacts),
            "model_fingerprint": "b" * 64,
            "suite_fingerprint": "c" * 64,
            "data_contract_fingerprint": "d" * 64,
        },
    )
    assert evaluator.completed_valid(
        tmp_path,
        expected_run_fingerprint=run_fp,
        request_by_id=request_by_id,
    )
    (tmp_path / "qzbc_dpo.jsonl").write_text("{}\n", encoding="utf-8")
    assert not evaluator.completed_valid(
        tmp_path,
        expected_run_fingerprint=run_fp,
        request_by_id=request_by_id,
    )


def test_quick_audit_validates_bound_requests_and_large_file_metadata(tmp_path):
    fixture = _build_data_fixture(tmp_path)
    audit = protocol.audit_data_contract(
        qzbc_root=fixture["qzbc_root"],
        requests_path=fixture["requests"],
        seed_manifest_path=fixture["seed_manifest"],
        compiled_manifests=fixture["compiled"],
        expected_request_rows=12,
        expected_family_clean_contract=fixture["family_contract"],
        expected_family_clean_task_counts=None,
        verify_test_recompile=False,
    )
    audit_path = tmp_path / "DATA_AUDIT.json"
    protocol.atomic_json(audit_path, audit)
    assert protocol.validate_frozen_data_audit_quick(
        audit_path, requests_path=fixture["requests"]
    )["data_contract_fingerprint"] == audit["data_contract_fingerprint"]
    with fixture["requests"].open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with _raises(ValueError, "prepared data input changed|inference requests"):
        protocol.validate_frozen_data_audit_quick(
            audit_path, requests_path=fixture["requests"]
        )


def test_family_clean_view_is_derived_from_fixed_request_ids(tmp_path):
    request_by_id = {}
    rollout_rows = []
    selected = []
    for seed_index, seed_id in enumerate(("seed-a", "seed-b")):
        for variant in protocol.EXPECTED_VARIANTS:
            item_id = f"{seed_id}-{variant}"
            request_by_id[item_id] = {
                "id": item_id,
                "seed_id": seed_id,
                "variant": variant,
            }
            rollout_rows.append(
                {
                    "id": item_id,
                    "variant": variant,
                    "passed": seed_index == 0 or variant == "clean",
                    "failure_types": [] if seed_index == 0 or variant == "clean" else [
                        "failed_evidence_boundary_recovery"
                    ],
                }
            )
            if seed_id == "seed-a":
                selected.append(item_id)
    model_dir = tmp_path / "model"
    _write_jsonl(model_dir / "qzbc_rollouts.jsonl", rollout_rows)
    pairs, payload, derived = evaluator._family_clean_payload(
        label="frozen",
        model_dir=model_dir,
        request_by_id=request_by_id,
        family_clean_request_ids=sorted(selected),
        derived_root=tmp_path / "views",
    )
    assert len(pairs) == 3
    assert payload["diagnostics"]["rollouts"] == 4
    assert payload["unified_metrics"]["perturbed_task_success_rate"] == 1.0
    assert protocol.sha256_file(derived) == payload["view_provenance"][
        "derived_rollouts_sha256"
    ]


def test_four_gpu_execution_topology_is_fingerprinted():
    args = argparse.Namespace(
        num_workers=evaluator.DEFAULT_NUM_WORKERS,
        devices=",".join(evaluator.DEFAULT_DEVICES),
    )
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
        evaluator._normalize_execution_topology(args)
    contract = evaluator._execution_contract(args)
    assert contract == {
        "num_workers": 4,
        "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "visible_physical_devices": ["4", "5", "6", "7"],
    }
    changed = dict(contract, visible_physical_devices=["0", "1", "2", "3"])
    assert protocol.canonical_hash(contract) != protocol.canonical_hash(changed)


class QZBCGroupedProtocolTests(unittest.TestCase):
    def _run_in_temp(self, function):
        with tempfile.TemporaryDirectory() as temporary:
            function(Path(temporary))

    def test_data_audit(self):
        self._run_in_temp(test_data_audit_recomputes_entity_and_family_clean_views)

    def test_false_manifest_pass(self):
        self._run_in_temp(test_claimed_manifest_pass_does_not_hide_exact_entity_overlap)

    def test_model_suite(self):
        self._run_in_temp(test_model_suite_binds_canonical_path_inventory_and_full_hash)

    def test_completion_tamper(self):
        self._run_in_temp(test_complete_marker_detects_artifact_tampering)

    def test_quick_audit(self):
        self._run_in_temp(test_quick_audit_validates_bound_requests_and_large_file_metadata)

    def test_family_clean_derivation(self):
        self._run_in_temp(test_family_clean_view_is_derived_from_fixed_request_ids)

    def test_four_gpu_topology(self):
        test_four_gpu_execution_topology_is_fingerprinted()


if __name__ == "__main__":
    unittest.main()
