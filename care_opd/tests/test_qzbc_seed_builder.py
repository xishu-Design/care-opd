from __future__ import annotations

import json
from pathlib import Path

from care_opd.qzbc_contract import OUTPUT_FIELDS, extract_source_ids
from care_opd.qzbc_seed_builder import (
    TASKS,
    build_seed_rows,
    generate_seed_corpus,
)


QZBC_ROOT = Path(__file__).resolve().parents[2] / "qzbc_source" / "qzbc-main"


def test_builder_covers_native_contract_and_verification() -> None:
    rows, manifest = build_seed_rows(
        QZBC_ROOT,
        seed=17,
        per_task=3,
        dev_ratio=0.2,
        test_ratio=0.2,
    )

    assert len(rows) == 24
    assert {row["request"]["task_type"] for row in rows} == set(TASKS)
    assert manifest["model_boundary"]["deploy_time_extra_agent"] is False
    assert manifest["data_policy"]["target_only_restates_evidence_packet"] is True
    assert (
        manifest["grounding_asset_split"]["strategy"]
        == "exact_asset_disjoint_deterministic_shuffle"
    )
    assert manifest["grounding_asset_split"]["overlap_check"] == "PASS"
    assert manifest["data_policy"]["reads_patient_database_or_runtime_memory"] is False
    assert all(manifest["split_counts"]["train"][task] == 1 for task in TASKS)
    assert all(manifest["split_counts"]["dev"][task] == 1 for task in TASKS)
    assert all(manifest["split_counts"]["test"][task] == 1 for task in TASKS)

    for row in rows:
        request = row["request"]
        target = row["target"]
        verification = row["verification"]
        target_text = json.dumps(target, ensure_ascii=False)
        packet_source_ids = set(extract_source_ids(request["evidence_packet"]))
        cited = {
            source_id
            for section in target["sections"]
            for source_id in section["source_ids"]
        }

        assert set(request) == {
            "role",
            "task_type",
            "input_text",
            "evidence_packet",
            "constraints",
        }
        assert tuple(target) == OUTPUT_FIELDS
        assert set(verification) == {
            "must_contain",
            "must_not_contain",
            "required_source_ids",
        }
        assert verification["required_source_ids"]
        assert set(verification["required_source_ids"]) == cited
        assert cited <= packet_source_ids
        assert all(token in target_text for token in verification["must_contain"])
        assert all(
            token not in target_text for token in verification["must_not_contain"]
        )
        assert "Evidence Packet" in target["summary"]
        assert "仅复述本轮 Evidence Packet" in target["disclaimer"]
        assert request["constraints"]["medical_fact_policy"] == (
            "only_restate_evidence_packet"
        )
        assert request["evidence_packet"]["provenance"]["contains_real_patient_data"] is False

    grounded_tasks = {
        row["request"]["task_type"]
        for row in rows
        if row["metadata"]["curated_assets"]
    }
    assert grounded_tasks == set(TASKS)
    asset_keys = {split: set() for split in ("train", "dev", "test")}
    for row in rows:
        for asset in row["metadata"]["curated_assets"]:
            assert set(asset) == {
                "source_id",
                "asset_type",
                "asset_id",
                "source_record_id",
                "license",
            }
            asset_keys[row["split"]].add(
                (
                    asset["asset_type"],
                    asset["asset_id"],
                    asset["source_record_id"],
                )
            )
    assert not (asset_keys["train"] & asset_keys["dev"])
    assert not (asset_keys["train"] & asset_keys["test"])
    assert not (asset_keys["dev"] & asset_keys["test"])


def test_total_is_deterministic_and_still_covers_all_tasks() -> None:
    first, first_manifest = build_seed_rows(
        QZBC_ROOT,
        seed=91,
        total=27,
        dev_ratio=0.2,
        test_ratio=0.2,
    )
    second, second_manifest = build_seed_rows(
        QZBC_ROOT,
        seed=91,
        total=27,
        dev_ratio=0.2,
        test_ratio=0.2,
    )

    assert first == second
    assert first_manifest == second_manifest
    assert len(first) == 27
    assert set(first_manifest["task_counts"]) == set(TASKS)
    assert min(first_manifest["task_counts"].values()) >= 3


def test_write_corpus_materializes_hashes_and_splits(tmp_path: Path) -> None:
    output = tmp_path / "formal_qzbc_seeds.jsonl"
    result = generate_seed_corpus(
        QZBC_ROOT,
        output,
        seed=42,
        per_task=3,
        dev_ratio=0.2,
        test_ratio=0.2,
    )

    assert result["counts"] == {"all": 24, "train": 8, "dev": 8, "test": 8}
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))
    all_rows = output.read_text(encoding="utf-8").splitlines()
    train_rows = result["train"].read_text(encoding="utf-8").splitlines()
    dev_rows = result["dev"].read_text(encoding="utf-8").splitlines()
    test_rows = result["test"].read_text(encoding="utf-8").splitlines()
    assert len(all_rows) == 24
    assert len(train_rows) == len(dev_rows) == len(test_rows) == 8
    assert all(
        manifest["outputs"][split]["sha256"]
        for split in ("all", "train", "dev", "test")
    )
