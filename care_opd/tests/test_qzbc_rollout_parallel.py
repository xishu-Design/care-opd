from pathlib import Path

import pytest

import care_opd.qzbc_rollout as rollout_module
from care_opd.qzbc_contract import canonical_response
from care_opd.qzbc_rollout import (
    _WorkerShard,
    _atomic_json,
    _atomic_jsonl,
    _exclusive_output_lock,
    _generate_candidates,
    _merge_worker_shards,
    _partition_indexed_rows,
    _publish_outputs,
    _resolve_worker_devices,
    _select_output_rows,
    _sha256_file,
    collect,
)
from care_opd.rollout import GenerationArguments


def _candidate(index: int, kind: str):
    identifier = f"row-{index:03d}"
    passed = kind == "clean"
    rollout = {
        "id": identifier,
        "passed": passed,
        "failure_types": (
            []
            if passed
            else ["invalid_generated_response"]
            if kind == "format"
            else ["missing_required_content"]
        ),
    }
    if passed:
        opd = {**rollout, "clean_control": True}
        dpo = None
    else:
        opd = {**rollout, "clean_control": False}
        dpo = {"id": identifier}
    return {
        "global_index": index,
        "rollout_row": rollout,
        "opd_row": opd,
        "dpo_row": dpo,
    }


def _write_shard(root: Path, worker_id: int, indices, candidates):
    input_path = root / f"worker_{worker_id}.input.jsonl"
    output_path = root / f"worker_{worker_id}.output.jsonl"
    complete_path = root / f"worker_{worker_id}.complete.json"
    error_path = root / f"worker_{worker_id}.error.json"
    _atomic_jsonl(input_path, [])
    _atomic_jsonl(output_path, candidates)
    _atomic_json(
        complete_path,
        {
            "worker_id": worker_id,
            "device": f"cuda:{worker_id}",
            "count": len(candidates),
            "sha256": _sha256_file(output_path),
        },
    )
    return _WorkerShard(
        worker_id=worker_id,
        device=f"cuda:{worker_id}",
        input_path=input_path,
        output_path=output_path,
        complete_path=complete_path,
        error_path=error_path,
        expected_indices=tuple(indices),
    )


def test_round_robin_partition_preserves_first_n_global_indices():
    rows = [{"id": index} for index in range(7)]
    partitions = _partition_indexed_rows(rows, 3)
    assert [[index for index, _ in part] for part in partitions] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]
    assert [row["id"] for part in partitions for _, row in part] == [
        0,
        3,
        6,
        1,
        4,
        2,
        5,
    ]


def test_three_shard_merge_is_byte_identical_to_single_worker(tmp_path):
    candidates = [
        _candidate(0, "clean"),
        _candidate(1, "semantic"),
        _candidate(2, "format"),
        _candidate(3, "semantic"),
        _candidate(4, "format"),
        _candidate(5, "semantic"),
        _candidate(6, "clean"),
    ]
    partitions = _partition_indexed_rows(candidates, 3)
    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    shards = []
    for worker_id, partition in enumerate(partitions):
        indices = [index for index, _ in partition]
        shard_candidates = [row for _, row in partition]
        shards.append(
            _write_shard(
                shard_root,
                worker_id,
                indices,
                shard_candidates,
            )
        )
    merged = _merge_worker_shards(shards, len(candidates))
    assert [row["global_index"] for row in merged] == list(range(len(candidates)))

    single_dir = tmp_path / "single"
    parallel_dir = tmp_path / "parallel"
    single_dir.mkdir()
    parallel_dir.mkdir()
    assert _publish_outputs(single_dir, candidates, 0.25) == _publish_outputs(
        parallel_dir, merged, 0.25
    )
    for filename in (
        "qzbc_rollouts.jsonl",
        "qzbc_opd.jsonl",
        "qzbc_dpo.jsonl",
        "summary.json",
    ):
        assert (single_dir / filename).read_bytes() == (
            parallel_dir / filename
        ).read_bytes()


def test_global_format_cap_is_not_applied_per_shard():
    candidates = [
        *(_candidate(index, "semantic") for index in range(3)),
        *(_candidate(index, "format") for index in range(3, 8)),
    ]
    _, opd_rows, dpo_rows, counts = _select_output_rows(candidates, 0.25)
    assert counts == {
        "semantic_failures": 3,
        "format_failures_total": 5,
        "format_failures_selected": 1,
    }
    assert [row["id"] for row in opd_rows] == [
        "row-000",
        "row-001",
        "row-002",
        "row-003",
    ]
    assert len(dpo_rows) == 4


def test_global_format_cap_does_not_depend_on_unique_ids():
    candidates = [
        _candidate(0, "semantic"),
        _candidate(1, "format"),
        _candidate(2, "format"),
        _candidate(3, "format"),
    ]
    for candidate in candidates:
        candidate["rollout_row"]["id"] = None
        candidate["opd_row"]["id"] = None
        if candidate["dpo_row"] is not None:
            candidate["dpo_row"]["id"] = None
    _, opd_rows, dpo_rows, counts = _select_output_rows(candidates, 0.25)
    assert counts["format_failures_selected"] == 0
    assert len(opd_rows) == len(dpo_rows) == 1
    assert opd_rows[0]["failure_types"] == ["missing_required_content"]


def test_no_semantic_failure_bootstrap_cap_is_global_256():
    candidates = [_candidate(index, "format") for index in range(300)]
    _, opd_rows, dpo_rows, counts = _select_output_rows(candidates, 0.25)
    assert counts["semantic_failures"] == 0
    assert counts["format_failures_total"] == 300
    assert counts["format_failures_selected"] == 256
    assert len(opd_rows) == len(dpo_rows) == 256
    assert opd_rows[-1]["id"] == "row-255"


def test_incomplete_worker_shard_is_rejected(tmp_path):
    candidates = [_candidate(0, "clean")]
    shard = _write_shard(tmp_path, 0, [0, 1], candidates)
    with pytest.raises(RuntimeError, match="incomplete or reordered shard"):
        _merge_worker_shards([shard], 2)


def test_summary_is_not_published_when_a_final_jsonl_write_fails(
    tmp_path,
    monkeypatch,
):
    original = rollout_module._atomic_jsonl

    def fail_on_opd(path, rows):
        if path.name == "qzbc_opd.jsonl":
            raise OSError("simulated disk failure")
        return original(path, rows)

    monkeypatch.setattr(rollout_module, "_atomic_jsonl", fail_on_opd)
    with pytest.raises(OSError, match="simulated disk failure"):
        _publish_outputs(tmp_path, [_candidate(0, "clean")], 0.25)
    assert not (tmp_path / "summary.json").exists()


def test_device_assignment_is_explicit_and_unambiguous():
    assert _resolve_worker_devices(3, "0,1,2", None) == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
    )
    assert _resolve_worker_devices(3, None, None) == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
    )
    assert _resolve_worker_devices(1, None, "cuda:4") == ("cuda:4",)
    with pytest.raises(ValueError, match="distinct"):
        _resolve_worker_devices(3, "0,0,1", None)
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_worker_devices(3, None, "cuda:0")
    with pytest.raises(ValueError, match="indexed explicitly"):
        _resolve_worker_devices(2, "cuda,cuda:1", None)
    assert _resolve_worker_devices(2, "00,cuda:01", None) == (
        "cuda:0",
        "cuda:1",
    )


def test_output_directory_lock_rejects_concurrent_collector(tmp_path):
    with _exclusive_output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already locked"):
            with _exclusive_output_lock(tmp_path):
                pass


def test_empty_selection_is_rejected_before_model_loading(tmp_path):
    requests = tmp_path / "empty.jsonl"
    requests.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no rollout requests selected"):
        collect(
            requests,
            tmp_path / "output",
            GenerationArguments(model="does-not-exist"),
            max_examples=0,
            num_workers=3,
            devices="0,1,2",
        )
    assert not (tmp_path / "output" / "summary.json").exists()


def test_generation_seed_uses_global_not_worker_local_index():
    target = canonical_response(
        {
            "title": "证据提示",
            "summary": "当前证据不足，需要复核。",
            "sections": [],
            "warnings": ["证据不足。"],
            "next_actions": ["请人工复核。"],
            "disclaimer": "不能替代人工复核。",
        }
    )

    class FakePolicy:
        tokenizer = object()

        def __init__(self):
            self.seeds = []

        def generate(self, prompt, *, seed, add_special_tokens):
            self.seeds.append(seed)
            return target

    row = {
        "id": "clean",
        "variant": "clean",
        "request": {
            "role": "patient",
            "task_type": "patient_education",
            "input_text": "解释。",
            "constraints": {},
            "evidence_packet": {},
        },
        "student_messages": [{"role": "user", "content": "解释。"}],
    }
    policy = FakePolicy()
    results = _generate_candidates(
        [(0, row), (3, row)],
        policy,
        seed=42,
        progress_stage="test_rollout_worker",
    )
    assert policy.seeds == [42, 45]
    assert [item["global_index"] for item in results] == [0, 3]
