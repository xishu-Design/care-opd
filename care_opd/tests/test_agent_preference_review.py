from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from competition_evidence.prepare_agent_preference_review import (
    CANDIDATE_CODES,
    _canonical_json_sha256,
    _sha256_text,
    prepare,
)
from competition_evidence.summarize_agent_preference_review import summarize


SYSTEMS = ("shennong_full", "agent_b", "agent_c", "agent_d")
REVIEWERS = ("expert_alpha", "expert_beta")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_inputs(root: Path) -> tuple[Path, dict[str, Path]]:
    tasks: list[dict] = []
    # Four items in each role/condition stratum make exact A-D balancing testable.
    for role, variant in (("doctor", "clean"), ("patient", "evidence_unavailable")):
        for index in range(4):
            tasks.append(
                {
                    "task_id": f"{role}-{variant}-{index}",
                    "family_token": f"family-{index}",
                    "role": role,
                    "variant": variant,
                    "modality": "thyroid_ultrasound",
                    "user_message_private": f"去标识化病例材料 {role} {variant} {index}",
                }
            )
    task_path = root / "tasks.private.jsonl"
    _write_jsonl(task_path, tasks)
    response_paths: dict[str, Path] = {}
    for system_index, system in enumerate(SYSTEMS):
        rows: list[dict] = []
        for task in tasks:
            text = f"匿名候选回答 {system_index + 1} / {task['task_id']}"
            rows.append(
                {
                    "task_id": task["task_id"],
                    "task_payload_sha256": _canonical_json_sha256(task),
                    "status": "success",
                    "display_response_private": text,
                    "response_sha256": _sha256_text(text),
                }
            )
        path = root / f"{system}.responses.private.jsonl"
        _write_jsonl(path, rows)
        response_paths[system] = path
    return task_path, response_paths


def _prepare_args(root: Path, tasks: Path, responses: dict[str, Path]) -> argparse.Namespace:
    return argparse.Namespace(
        tasks=tasks,
        response=[f"{system}={path}" for system, path in responses.items()],
        reviewer=list(REVIEWERS),
        blind_secret_env="TEST_EXPERT_BLIND_SECRET",
        identity_term=None,
        xlsx=False,
        output_private=root / "private",
        output_public=root / "public",
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _completed_ratings(
    prepared: dict, output_root: Path, *, invalid_tie: bool = False
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    mapping_rows = [
        json.loads(line)
        for line in prepared["mapping"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for row in mapping_rows:
        mapping.setdefault(
            (row["reviewer_id_private"], row["blind_item_id"]), {}
        )[row["candidate_code"]] = row["system_id_private"]

    completed: list[Path] = []
    for reviewer in REVIEWERS:
        template = prepared["reviewer_files"][reviewer]["csv"]
        fieldnames, rows = _read_csv(template)
        for index, row in enumerate(rows):
            code_to_system = mapping[(reviewer, row["blind_item_id"])]
            primary_code = next(
                code for code, system in code_to_system.items() if system == "shennong_full"
            )
            baseline_code = next(
                code for code, system in code_to_system.items() if system == "agent_b"
            )
            unsafe_code = next(
                code for code, system in code_to_system.items() if system == "agent_d"
            )
            if index == 0:
                row["best_choice"] = "TIE"
                row["tied_candidates"] = "" if invalid_tie else f"{primary_code}|{baseline_code}"
            elif index == 1:
                row["best_choice"] = "NONE"
            else:
                row["best_choice"] = primary_code
            row["unsafe_candidates"] = unsafe_code
            row["primary_reason"] = "FACT"
            row["confidence_1_3"] = "3"
            row["completed_at_utc"] = "2026-08-24T00:00:00Z"
        path = output_root / f"{reviewer}.completed.private.csv"
        _write_csv(path, fieldnames, rows)
        completed.append(path)
    return completed


def test_prepare_four_system_review_is_private_balanced_and_reviewer_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_EXPERT_BLIND_SECRET", "blind-secret-material-0123456789-abcdef")
    tasks, responses = _fixture_inputs(tmp_path)
    outputs = prepare(_prepare_args(tmp_path / "run", tasks, responses))

    mapping_rows = [
        json.loads(line)
        for line in outputs["mapping"].read_text(encoding="utf-8").splitlines()
    ]
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in mapping_rows:
        counts[
            (
                row["reviewer_id_private"],
                row["target_role"],
                row["evaluation_condition"],
                row["system_id_private"],
                row["candidate_code"],
            )
        ] += 1
    for reviewer in REVIEWERS:
        for role, variant in (("doctor", "clean"), ("patient", "evidence_unavailable")):
            for system in SYSTEMS:
                observed = [
                    counts[(reviewer, role, variant, system, code)]
                    for code in CANDIDATE_CODES
                ]
                assert observed == [1, 1, 1, 1]

    public_text = outputs["public_manifest"].read_text(encoding="utf-8")
    assert all(system not in public_text for system in SYSTEMS)
    assert all(reviewer not in public_text for reviewer in REVIEWERS)
    for reviewer in REVIEWERS:
        reviewer_text = outputs["reviewer_files"][reviewer]["csv"].read_text(
            encoding="utf-8-sig"
        )
        assert all(system not in reviewer_text for system in SYSTEMS)
        assert "family-" not in reviewer_text

    # A reviewer-specific mapping prevents experts from exchanging a global A-D key.
    first_task = "doctor-clean-0"
    by_reviewer: dict[str, dict[str, str]] = {}
    for row in mapping_rows:
        if row["task_id_private"] == first_task:
            by_reviewer.setdefault(row["reviewer_id_private"], {})[
                row["candidate_code"]
            ] = row["system_id_private"]
    assert by_reviewer[REVIEWERS[0]] != by_reviewer[REVIEWERS[1]]


def test_optional_xlsx_has_a_graceful_csv_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_EXPERT_BLIND_SECRET", "blind-secret-material-0123456789-abcdef")
    tasks, responses = _fixture_inputs(tmp_path)
    args = _prepare_args(tmp_path / "run", tasks, responses)
    args.xlsx = True
    outputs = prepare(args)
    assert outputs["xlsx_created_count"] in {0, len(REVIEWERS)}
    assert all(outputs["reviewer_files"][reviewer]["csv"].is_file() for reviewer in REVIEWERS)
    if outputs["xlsx_created_count"]:
        assert all("xlsx" in outputs["reviewer_files"][reviewer] for reviewer in REVIEWERS)


def test_prepare_rejects_identity_leakage_before_creating_review_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_EXPERT_BLIND_SECRET", "blind-secret-material-0123456789-abcdef")
    tasks, responses = _fixture_inputs(tmp_path)
    target = responses["agent_b"]
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    rows[0]["display_response_private"] = "我是 Groq 提供的智能体。"
    rows[0].pop("response_sha256", None)
    _write_jsonl(target, rows)
    with pytest.raises(ValueError, match="identity leakage detected"):
        prepare(_prepare_args(tmp_path / "run", tasks, responses))


@pytest.mark.parametrize("failure", ["binding", "status", "empty"])
def test_prepare_rejects_unbound_unsuccessful_or_empty_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setenv("TEST_EXPERT_BLIND_SECRET", "blind-secret-material-0123456789-abcdef")
    tasks, responses = _fixture_inputs(tmp_path)
    target = responses["agent_b"]
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    if failure == "binding":
        rows[0]["task_payload_sha256"] = "0" * 64
    elif failure == "status":
        rows[0]["status"] = "failed"
    else:
        rows[0]["display_response_private"] = ""
        rows[0].pop("response_sha256", None)
    _write_jsonl(target, rows)
    with pytest.raises(ValueError, match="binding mismatch|status|empty display response"):
        prepare(_prepare_args(tmp_path / "run", tasks, responses))


def test_summary_scores_unique_tie_none_breakdowns_and_private_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_EXPERT_BLIND_SECRET", "blind-secret-material-0123456789-abcdef")
    tasks, responses = _fixture_inputs(tmp_path)
    prepared = prepare(_prepare_args(tmp_path / "prepared", tasks, responses))
    ratings = _completed_ratings(prepared, tmp_path)
    args = argparse.Namespace(
        manifest=prepared["private_manifest"],
        mapping=prepared["mapping"],
        ratings=ratings,
        primary_agent="shennong_full",
        bootstrap_replicates=200,
        bootstrap_seed=17,
        output_private=tmp_path / "summary_private",
        output_public=tmp_path / "summary_public",
    )
    outputs = summarize(args)
    summary = json.loads(outputs["summary_public"].read_text(encoding="utf-8"))
    assert summary["status"] == "FINAL"
    assert summary["analysis"]["numeric_leader"] == "shennong_full"
    assert len(summary["pairwise_comparisons"]) == 3
    primary = next(
        row for row in summary["system_aggregates"] if row["system"] == "shennong_full"
    )
    assert primary["sole_best_count"] == 12
    assert primary["tied_best_inclusion_count"] == 2
    assert primary["none_acceptable_ballot_count"] == 2
    assert primary["fractional_points"] == 13.0
    assert primary["safety_qualified_first_choice_share"] == 0.8125
    assert primary["safety_qualified_fractional_score_per_100"] == 81.25
    assert primary["numeric_rank"] == 1

    _, breakdown = _read_csv(outputs["breakdown_public"])
    assert {row["dimension"] for row in breakdown} == {"role", "variant", "reviewer"}
    assert {row["dimension_value"] for row in breakdown if row["dimension"] == "reviewer"} == {
        "R1",
        "R2",
    }
    public_text = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for name, path in outputs.items()
        if name.endswith("public")
    )
    assert all(f"family-{index}" not in public_text for index in range(4))
    assert "去标识化病例材料" not in public_text
    assert all(reviewer not in public_text for reviewer in REVIEWERS)


def test_summary_enforces_tie_semantics_and_answer_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_EXPERT_BLIND_SECRET", "blind-secret-material-0123456789-abcdef")
    tasks, responses = _fixture_inputs(tmp_path)
    prepared = prepare(_prepare_args(tmp_path / "prepared", tasks, responses))
    invalid_ratings = _completed_ratings(prepared, tmp_path, invalid_tie=True)
    args = argparse.Namespace(
        manifest=prepared["private_manifest"],
        mapping=prepared["mapping"],
        ratings=invalid_ratings,
        primary_agent="shennong_full",
        bootstrap_replicates=10,
        bootstrap_seed=3,
        output_private=tmp_path / "summary_private",
        output_public=tmp_path / "summary_public",
    )
    with pytest.raises(ValueError, match="TIE requires 2-4"):
        summarize(args)

    # Restore valid ballots but alter one displayed answer: hashing must stop the summary.
    valid_ratings = _completed_ratings(prepared, tmp_path / "valid")
    fieldnames, rows = _read_csv(valid_ratings[0])
    rows[0]["answer_A"] += " tampered"
    _write_csv(valid_ratings[0], fieldnames, rows)
    args.ratings = valid_ratings
    with pytest.raises(ValueError, match="answer_A was changed"):
        summarize(args)


def test_safety_flagged_candidate_cannot_receive_best_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_EXPERT_BLIND_SECRET", "blind-secret-material-0123456789-abcdef")
    tasks, responses = _fixture_inputs(tmp_path)
    prepared = prepare(_prepare_args(tmp_path / "prepared", tasks, responses))
    ratings = _completed_ratings(prepared, tmp_path / "ratings")
    fieldnames, rows = _read_csv(ratings[0])
    rows[0]["unsafe_candidates"] = rows[0]["tied_candidates"].split("|")[0]
    _write_csv(ratings[0], fieldnames, rows)
    args = argparse.Namespace(
        manifest=prepared["private_manifest"],
        mapping=prepared["mapping"],
        ratings=ratings,
        primary_agent="shennong_full",
        bootstrap_replicates=10,
        bootstrap_seed=5,
        output_private=tmp_path / "summary_private",
        output_public=tmp_path / "summary_public",
    )
    with pytest.raises(ValueError, match="marked unsafe cannot be selected"):
        summarize(args)
