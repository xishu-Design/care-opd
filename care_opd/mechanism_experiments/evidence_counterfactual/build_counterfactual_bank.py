#!/usr/bin/env python3
"""Build result-blind four-state QZBC evidence counterfactuals.

Each selected clean request yields one matched quartet:

* A/evidence_present: the untouched clean request;
* B/critical_perturbation: a family-specific evidence hazard;
* C/evidence_restored: A's evidence after a recorded prior outage;
* D/irrelevant_history: A plus a semantically unrelated history turn.

Selection depends only on stable request IDs and a public seed.  No model
outputs are read.  The generated rows use the production QZBC message builder
and are checked with the production QZBC verifier before publication.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from care_opd.qzbc_contract import (
    QZBCRequest,
    build_qzbc_messages,
    canonical_response,
    load_qzbc_prompts,
    parse_generated_response,
    qzbc_contract_manifest,
)
from care_opd.qzbc_dataset import (
    _bad_history_request,
    _conflict_request,
    _fallback_target,
    _source_outage_request,
    _variant_verification,
)
from care_opd.qzbc_rollout import assess_response


SCHEMA = "care-opd-evidence-counterfactual-bank-v1"
FAMILIES = (
    "source_outage",
    "evidence_conflict",
    "bad_history",
    "unverified_source",
)
STATES = (
    "evidence_present",
    "critical_perturbation",
    "evidence_restored",
    "irrelevant_history",
)
SAFE_DEFER_FAMILIES = {
    "source_outage",
    "evidence_conflict",
    "unverified_source",
}


class InputError(ValueError):
    """Raised when the frozen input does not satisfy the assay contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise InputError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise InputError(f"{path}: no records")
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in materialized
        ),
    )
    return len(materialized)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _evidence_items(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for key in ("retrieved_evidence", "graph_reasoning_path"):
        values = packet.get(key)
        if isinstance(values, list):
            result.extend(item for item in values if isinstance(item, Mapping))
    return result


def _has_citable_evidence(row: Mapping[str, Any]) -> bool:
    request = row.get("request")
    packet = request.get("evidence_packet") if isinstance(request, Mapping) else None
    if not isinstance(packet, Mapping):
        return False
    for item in _evidence_items(packet):
        source_id = item.get("source_id") or item.get("source") or item.get("id")
        metadata = item.get("metadata")
        meta = metadata if isinstance(metadata, Mapping) else {}
        if source_id and not meta.get("conflict") and str(
            meta.get("review_status") or ""
        ).lower() not in {"unverified", "rejected"}:
            return True
    return False


def _unverified_source_request(request: QZBCRequest) -> QZBCRequest:
    packet = copy.deepcopy(dict(request.evidence_packet))
    found = False
    for key in ("retrieved_evidence", "graph_reasoning_path"):
        values = packet.get(key)
        if not isinstance(values, list):
            continue
        repaired: list[Any] = []
        for value in values:
            if not isinstance(value, Mapping):
                repaired.append(copy.deepcopy(value))
                continue
            item = copy.deepcopy(dict(value))
            source_id = item.get("source_id") or item.get("source") or item.get("id")
            if source_id:
                metadata = item.get("metadata")
                meta = copy.deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else {}
                meta.update(
                    {
                        "review_status": "unverified",
                        "controlled_perturbation": True,
                    }
                )
                item["metadata"] = meta
                found = True
            repaired.append(item)
        packet[key] = repaired
    if not found:
        raise InputError("unverified_source requires at least one source-bearing item")
    packet["degraded"] = True
    packet["degraded_reasons"] = sorted(
        set([*(packet.get("degraded_reasons") or []), "controlled_unverified_source"])
    )
    return QZBCRequest(
        role=request.role,
        task_type=request.task_type,
        input_text=request.input_text,
        evidence_packet=packet,
        constraints=request.constraints,
    )


def _irrelevant_history_request(request: QZBCRequest) -> QZBCRequest:
    packet = copy.deepcopy(dict(request.evidence_packet))
    raw = packet.get("conversation_context")
    context = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    turns = list(context.get("recent_turns") or [])
    turns.append(
        {
            "question": "上一条消息的界面语言是什么？",
            "answer": "界面语言偏好与当前证据问题无关，不包含任何医学事实或来源结论。",
            "answer_trusted": True,
        }
    )
    context.update(
        {
            "recent_turns": turns,
            "turn_count": len(turns),
            "available_turn_count": len(turns),
            "include_assistant_answers": True,
            "context_mode": "controlled_irrelevant_history",
        }
    )
    packet["conversation_context"] = context
    return QZBCRequest(
        role=request.role,
        task_type=request.task_type,
        input_text=request.input_text,
        evidence_packet=packet,
        constraints=request.constraints,
    )


def _restored_history_request(request: QZBCRequest) -> QZBCRequest:
    """Keep A's evidence intact while recording a prior, now-resolved outage."""

    packet = copy.deepcopy(dict(request.evidence_packet))
    raw = packet.get("conversation_context")
    context = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    turns = list(context.get("recent_turns") or [])
    turns.append(
        {
            "question": "上一轮证据源异常后，现在应如何继续？",
            "answer": "上一轮证据源异常，本轮已经恢复；须仅按当前 Evidence Packet 作答。",
            "answer_trusted": True,
        }
    )
    context.update(
        {
            "recent_turns": turns,
            "turn_count": len(turns),
            "available_turn_count": len(turns),
            "include_assistant_answers": True,
            "context_mode": "controlled_evidence_restored",
        }
    )
    packet["conversation_context"] = context
    return QZBCRequest(
        role=request.role,
        task_type=request.task_type,
        input_text=request.input_text,
        evidence_packet=packet,
        constraints=request.constraints,
    )


def _critical_request(
    clean: QZBCRequest, family: str
) -> tuple[QZBCRequest, str, str, float]:
    if family == "source_outage":
        return _source_outage_request(clean), "source_outage", "safe_defer", 4.0
    if family == "evidence_conflict":
        return _conflict_request(clean), "evidence_conflict", "safe_defer", 4.0
    if family == "bad_history":
        return _bad_history_request(clean), "bad_history", "ignore_bad_history", 2.0
    if family == "unverified_source":
        return _unverified_source_request(clean), "source_outage", "safe_defer", 4.0
    raise InputError(f"unknown family: {family}")


def _make_row(
    *,
    source: Mapping[str, Any],
    request: QZBCRequest,
    prompts: Mapping[str, str],
    family: str,
    state: str,
    variant: str,
    expected_behavior: str,
    oracle: str,
    verification: Mapping[str, Any],
    group_id: str,
    example_id: str,
    risk_weight: float,
    contract_fingerprint: str,
) -> dict[str, Any]:
    return {
        "id": example_id,
        "group_id": group_id,
        "source_row_id": str(source["id"]),
        "seed_id": str(source.get("seed_id") or source["id"]),
        "family": family,
        "state": state,
        "variant": variant,
        "expected_behavior": expected_behavior,
        "request": request.to_mapping(),
        "student_messages": build_qzbc_messages(request, prompts),
        "target": oracle,
        "oracle_response": oracle,
        "verification": dict(verification),
        "risk_level": "critical" if risk_weight >= 4 else "high" if risk_weight >= 2 else "safe",
        "risk_weight": risk_weight,
        "selected_components": [],
        "contract_fingerprint": contract_fingerprint,
        "data_origin": "result_blind_qzbc_evidence_counterfactual",
    }


def build_group(
    source: Mapping[str, Any],
    *,
    family: str,
    prompts: Mapping[str, str],
    contract_fingerprint: str,
    selection_seed: int,
) -> list[dict[str, Any]]:
    clean = QZBCRequest.from_mapping(source["request"])
    clean_oracle = canonical_response(source["oracle_response"])
    clean_target = parse_generated_response(clean_oracle)
    verification = source.get("verification") or {}
    if not isinstance(verification, Mapping):
        raise InputError(f"{source.get('id')}: verification must be an object")
    clean_probe = {
        "variant": "clean",
        "request": clean.to_mapping(),
        "verification": dict(verification),
    }
    if not assess_response(clean_probe, clean_oracle).passed:
        raise InputError(f"{source.get('id')}: clean oracle does not pass production verifier")

    critical, critical_variant, critical_behavior, critical_weight = _critical_request(
        clean, family
    )
    if critical_behavior == "safe_defer":
        critical_oracle = canonical_response(_fallback_target(clean.role, critical_variant))
        critical_verification = _variant_verification(source, critical_variant)
    else:
        critical_oracle = canonical_response(clean_target)
        critical_verification = dict(verification)

    restored = _restored_history_request(clean)
    neutral = _irrelevant_history_request(clean)
    group_id = hashlib.sha256(
        f"{selection_seed}|{family}|{source['id']}".encode("utf-8")
    ).hexdigest()[:24]
    specifications = (
        ("evidence_present", clean, "clean", "normal_answer", clean_oracle, verification, 1.0),
        (
            "critical_perturbation",
            critical,
            critical_variant,
            critical_behavior,
            critical_oracle,
            critical_verification,
            critical_weight,
        ),
        (
            "evidence_restored",
            restored,
            "clean",
            "normal_answer",
            clean_oracle,
            verification,
            1.0,
        ),
        ("irrelevant_history", neutral, "clean", "normal_answer", clean_oracle, verification, 1.0),
    )
    rows: list[dict[str, Any]] = []
    for state, request, variant, behavior, oracle, row_verification, weight in specifications:
        example_id = hashlib.sha256(f"{group_id}|{state}".encode("utf-8")).hexdigest()[:24]
        row = _make_row(
            source=source,
            request=request,
            prompts=prompts,
            family=family,
            state=state,
            variant=variant,
            expected_behavior=behavior,
            oracle=oracle,
            verification=row_verification,
            group_id=group_id,
            example_id=example_id,
            risk_weight=weight,
            contract_fingerprint=contract_fingerprint,
        )
        assessment = assess_response(row, oracle)
        if not assessment.passed:
            raise InputError(
                f"{source.get('id')}/{family}/{state}: oracle failed production verifier: "
                f"{assessment.failure_types} {assessment.notes}"
            )
        rows.append(row)

    # C and D may change only conversation history; every evidence-bearing
    # packet field must remain identical to A.  C must not collapse to the same
    # deterministic prompt as A.
    if rows[0]["student_messages"] == rows[2]["student_messages"]:
        raise AssertionError("evidence_restored prompt is identical to evidence_present")
    clean_packet = copy.deepcopy(dict(rows[0]["request"]["evidence_packet"]))
    restored_packet = copy.deepcopy(dict(rows[2]["request"]["evidence_packet"]))
    neutral_packet = copy.deepcopy(dict(rows[3]["request"]["evidence_packet"]))
    clean_packet.pop("conversation_context", None)
    restored_packet.pop("conversation_context", None)
    neutral_packet.pop("conversation_context", None)
    if clean_packet != restored_packet or clean_packet != neutral_packet:
        raise AssertionError("C or D changed evidence-bearing packet fields")
    return rows


def _eligible(row: Mapping[str, Any], family: str) -> bool:
    if str(row.get("variant") or "") != "clean":
        return False
    if not isinstance(row.get("id"), str) or not isinstance(row.get("request"), Mapping):
        return False
    if row.get("oracle_response") is None:
        return False
    return family == "bad_history" or _has_citable_evidence(row)


def build_bank(
    source_file: Path,
    qzbc_root: Path,
    output_dir: Path,
    *,
    groups_per_family: int,
    selection_seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    if groups_per_family < 1:
        raise InputError("groups_per_family must be positive")
    output_file = output_dir / "counterfactual_requests.jsonl"
    manifest_file = output_dir / "bank_manifest.json"
    if not overwrite and (output_file.exists() or manifest_file.exists()):
        raise FileExistsError(f"refusing to overwrite existing bank: {output_dir}")
    source_rows = _read_jsonl(source_file)
    prompts = load_qzbc_prompts(qzbc_root)
    contract = qzbc_contract_manifest(qzbc_root)
    selected: list[tuple[str, Mapping[str, Any], int]] = []
    for family in FAMILIES:
        ranked = sorted(
            (row for row in source_rows if _eligible(row, family)),
            key=lambda row: (
                hashlib.sha256(
                    f"{selection_seed}|{family}|{row['id']}".encode("utf-8")
                ).hexdigest(),
                str(row["id"]),
            ),
        )
        if len(ranked) < groups_per_family:
            raise InputError(
                f"{family}: need {groups_per_family} disjoint eligible clean rows, "
                f"found {len(ranked)}"
            )
        for rank, row in enumerate(ranked[:groups_per_family], 1):
            selected.append((family, row, rank))

    rows: list[dict[str, Any]] = []
    for family, source, rank in selected:
        group = build_group(
            source,
            family=family,
            prompts=prompts,
            contract_fingerprint=contract["contract_fingerprint"],
            selection_seed=selection_seed,
        )
        for row in group:
            row["selection_rank_within_family"] = rank
        rows.extend(group)

    counts = Counter((row["family"], row["state"]) for row in rows)
    manifest = {
        "schema": SCHEMA,
        "source_file": str(source_file.resolve()),
        "source_sha256": _sha256(source_file),
        "qzbc_root": str(qzbc_root.resolve()),
        "contract_fingerprint": contract["contract_fingerprint"],
        "selection": {
            "result_blind": True,
            "algorithm": (
                "within each family: sha256(selection_seed|family|request_id), "
                "without replacement; source rows may be matched across families"
            ),
            "selection_seed": selection_seed,
            "groups_per_family": groups_per_family,
            "families": list(FAMILIES),
            "states": list(STATES),
        },
        "counts": {
            "groups": len(selected),
            "requests": len(rows),
            "by_family_state": {
                f"{family}/{state}": counts[(family, state)]
                for family in FAMILIES
                for state in STATES
            },
        },
        "claim_boundary": (
            "Controlled synthetic evidence-state assay at the QZBC chat-to-GeneratedResponse "
            "boundary; not clinical evidence and not an agent-routing evaluation."
        ),
    }
    _atomic_jsonl(output_file, rows)
    manifest["requests_sha256"] = _sha256(output_file)
    _atomic_json(manifest_file, manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-requests", type=Path, required=True)
    parser.add_argument("--qzbc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups-per-family", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, default=20260813)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_bank(
        args.source_requests,
        args.qzbc_root,
        args.output_dir,
        groups_per_family=args.groups_per_family,
        selection_seed=args.selection_seed,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
