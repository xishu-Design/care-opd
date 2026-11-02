#!/usr/bin/env python3
"""Locked local-Hugging-Face evaluation for the private thyroid benchmark.

This runner deliberately separates private, report-level traces from
publication-safe aggregate artifacts.  It evaluates one or more local
checkpoints with the same deterministic generation contract (batch size one,
greedy decoding, thinking disabled, BF16 on CUDA), scores every response with
``thyroid_verifier.assess_thyroid_response``, and performs paired family-level
bootstrap comparisons that keep each clean/bad-history/outage/conflict quartet
intact.  Verifier task success is primary; literal field P/R/F1 is secondary
and restricted to clean plus bad-history states.

The program never copies prompts, responses, gold values, case identifiers, or
absolute private paths into the public output directory.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence


SCHEMA_VERSION = "care_opd.real_thyroid_external_eval.v4"
DEFAULT_BOOTSTRAP_SEED = 20260818
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
EVALUATION_TRACKS = (
    "structured_evidence_faithfulness",
    "raw_strict_extraction_challenge",
)
LEGACY_EVALUATION_TRACK = "raw_strict_extraction_challenge"
FOUR_STATE_VARIANTS = (
    "clean",
    "bad_history",
    "source_outage",
    "evidence_conflict",
)
PERTURBED_VARIANTS = tuple(
    variant for variant in FOUR_STATE_VARIANTS if variant != "clean"
)
SECONDARY_FIELD_VARIANTS = ("clean", "bad_history")
BOUNDARY_VARIANTS = ("source_outage", "evidence_conflict")
BOUNDARY_ASSESSMENT_FIELDS = (
    "boundary_recovery_valid",
    "boundary_semantics_valid",
    "boundary_structure_valid",
    "controlled_sentinel_free",
    "unsafe_clinical_conclusion",
)
PROTOCOL_ASSESSMENT_FIELDS = (
    "envelope_valid",
    "slot_schema_valid",
    "key_binding_valid",
    "packet_scaffolding_free",
    "auxiliary_fields_empty",
)
EXTRACTION_SCOPED_PROTOCOL_FIELDS = frozenset(
    {"slot_schema_valid", "key_binding_valid", "auxiliary_fields_empty"}
)
CONTENT_AXES = ("content", "grounded_content")
PUBLIC_MODALITIES = {
    "ultrasound",
    "pathology",
    "thyroid_ultrasound",
    "thyroid_pathology",
}
PUBLIC_FAILURE_TYPES = {
    "invalid_generated_response",
    "generation_error",
    "controlled_sentinel_leakage",
    "boundary_field_emission",
    "boundary_source_emission",
    "missing_boundary_structure",
    "missing_boundary_semantics",
    "unsafe_clinical_conclusion",
    "failed_evidence_boundary_recovery",
    "invalid_field_schema",
    "invalid_source_grounding",
    "unsupported_field",
    "literal_value_mismatch",
    "missing_gold_field",
    "copied_untrusted_history",
    "nonempty_auxiliary_fields",
}
MODEL_FILE_SUFFIXES = {
    ".bin",
    ".json",
    ".jinja",
    ".model",
    ".pt",
    ".pth",
    ".py",
    ".safetensors",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_GITIGNORE = "*\n!.gitignore\n"


class EvaluationContractError(ValueError):
    """Raised when a locked evaluation input or resume contract is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_text(path, text)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationContractError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise EvaluationContractError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise EvaluationContractError(f"{path}: no JSONL rows")
    return rows


def _case_id(row: Mapping[str, Any], line_number: int | None = None) -> str:
    for key in ("case_id", "id", "task_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    suffix = f" at row {line_number}" if line_number is not None else ""
    raise EvaluationContractError(f"evaluation row has no case_id/id/task_id{suffix}")


def _metadata_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    metadata = row.get("metadata")
    return metadata.get(key) if isinstance(metadata, Mapping) else None


def _public_modality(value: Any) -> str:
    modality = str(value or "").strip()
    return modality if modality in PUBLIC_MODALITIES else "unknown"


def _evaluation_track_contract(
    row: Mapping[str, Any], *, case_id: str
) -> tuple[str, bool]:
    raw_track = _metadata_value(row, "evaluation_track")
    track = str(raw_track or LEGACY_EVALUATION_TRACK).strip()
    if track not in EVALUATION_TRACKS:
        raise EvaluationContractError(
            f"{case_id}: unsupported evaluation_track {track!r}; expected one of {EVALUATION_TRACKS}"
        )
    expected_challenge = track == "raw_strict_extraction_challenge"
    raw_challenge = _metadata_value(row, "challenge_mode")
    if raw_challenge is None:
        challenge_mode = expected_challenge
    elif isinstance(raw_challenge, bool):
        challenge_mode = raw_challenge
    else:
        raise EvaluationContractError(f"{case_id}: challenge_mode must be boolean")
    if challenge_mode != expected_challenge:
        raise EvaluationContractError(
            f"{case_id}: challenge_mode is inconsistent with evaluation_track {track!r}"
        )
    return track, challenge_mode


def _family_variant(
    row: Mapping[str, Any],
    *,
    case_id: str,
) -> tuple[str, str, bool]:
    """Return a public-safe grouping contract without guessing perturbations.

    Legacy clean-only rows predate ``family_id``/``variant``.  They are mapped
    to one clean family per case solely for backward-compatible aggregation;
    the returned boolean records whether the grouping was explicit.
    """

    raw_family = _metadata_value(row, "family_id")
    raw_variant = _metadata_value(row, "variant")
    explicit = raw_family is not None or raw_variant is not None
    if explicit and (raw_family is None or raw_variant is None):
        raise EvaluationContractError(
            f"{case_id}: family_id and variant must either both be present or both be absent"
        )
    family_id = str(raw_family if raw_family is not None else case_id).strip()
    variant = str(raw_variant if raw_variant is not None else "clean").strip()
    if not family_id:
        raise EvaluationContractError(f"{case_id}: family_id is empty")
    if variant not in FOUR_STATE_VARIANTS:
        raise EvaluationContractError(
            f"{case_id}: unsupported variant {variant!r}; expected one of {FOUR_STATE_VARIANTS}"
        )
    return family_id, variant, explicit


def _validate_family_contract(
    rows: Sequence[MutableMapping[str, Any]],
) -> dict[str, Any]:
    by_family: dict[str, dict[str, MutableMapping[str, Any]]] = defaultdict(dict)
    explicit_flags: set[bool] = set()
    family_modalities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        case_id = str(row["_eval_case_id"])
        family_id, variant, explicit = _family_variant(row, case_id=case_id)
        explicit_flags.add(explicit)
        if variant in by_family[family_id]:
            raise EvaluationContractError(
                f"family {family_id!r} has duplicate variant {variant!r}"
            )
        row["_eval_family_id"] = family_id
        row["_eval_variant"] = variant
        by_family[family_id][variant] = row
        family_modalities[family_id].add(str(row["_eval_modality"]))
    if len(explicit_flags) > 1:
        raise EvaluationContractError(
            "locked inputs mix explicit family/variant rows with legacy clean-only rows"
        )
    mixed_modality = sorted(
        family_id for family_id, values in family_modalities.items() if len(values) != 1
    )
    if mixed_modality:
        raise EvaluationContractError(
            f"{len(mixed_modality)} families span multiple modalities"
        )

    variant_sets = {tuple(sorted(values)) for values in by_family.values()}
    clean_only = variant_sets == {("clean",)}
    expected = set(FOUR_STATE_VARIANTS)
    complete_four_state = bool(by_family) and all(
        set(values) == expected for values in by_family.values()
    )
    if not clean_only and not complete_four_state:
        observed = Counter(tuple(sorted(values)) for values in by_family.values())
        raise EvaluationContractError(
            "variant-aware evaluation must contain either clean-only families or closed "
            f"four-state families; observed={dict(observed)}"
        )
    return {
        "mode": "four_state" if complete_four_state else "clean_only",
        "explicit_grouping": explicit_flags == {True},
        "family_count": len(by_family),
        "variants": list(FOUR_STATE_VARIANTS) if complete_four_state else ["clean"],
        "rows_per_family": 4 if complete_four_state else 1,
        "by_family": by_family,
    }


def _validate_messages(value: Any, *, case_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise EvaluationContractError(f"{case_id}: messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise EvaluationContractError(f"{case_id}: message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise EvaluationContractError(f"{case_id}: malformed message {index}")
        messages.append({"role": str(role), "content": content})
    return messages


def _import_qzbc_contract() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    care_root = Path(__file__).resolve().parents[2]
    source_root = care_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from care_opd.qzbc_contract import (
            build_qzbc_messages,
            load_qzbc_prompts,
            qzbc_contract_manifest,
        )
    except ImportError as exc:
        raise RuntimeError("cannot import the CARE-OPD qzbc contract") from exc
    return build_qzbc_messages, load_qzbc_prompts, qzbc_contract_manifest


def _materialize_messages(
    seeds: Sequence[MutableMapping[str, Any]], qzbc_root: Path | None
) -> Mapping[str, Any] | None:
    """Validate prebuilt messages or build them from the locked qzbc request."""

    need_build = any(
        not isinstance(seed.get("student_messages") or seed.get("messages"), list)
        for seed in seeds
    )
    prompts: Mapping[str, str] | None = None
    contract_manifest: Mapping[str, Any] | None = None
    build_messages: Callable[..., Any] | None = None
    if need_build:
        if qzbc_root is None:
            raise EvaluationContractError(
                "at least one seed lacks prebuilt messages; pass --qzbc-root so the exact "
                "qzbc prompt contract can be materialized"
            )
        build_messages, load_prompts, manifest_fn = _import_qzbc_contract()
        prompts = load_prompts(qzbc_root)
        contract_manifest = manifest_fn(qzbc_root)
    elif qzbc_root is not None:
        _, _, manifest_fn = _import_qzbc_contract()
        contract_manifest = manifest_fn(qzbc_root)

    for seed in seeds:
        identifier = _case_id(seed)
        prebuilt = seed.get("student_messages") or seed.get("messages")
        if isinstance(prebuilt, list):
            seed["_eval_messages"] = _validate_messages(prebuilt, case_id=identifier)
            continue
        request = seed.get("request")
        if not isinstance(request, Mapping):
            raise EvaluationContractError(
                f"{identifier}: seed requires student_messages/messages or a request object"
            )
        assert build_messages is not None and prompts is not None
        seed["_eval_messages"] = _validate_messages(
            build_messages(request, prompts), case_id=identifier
        )
    return contract_manifest


def _field_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize common one-field gold records without exposing values publicly."""

    entity_id = row.get("entity_id")
    field_path = row.get("field_path")
    if not isinstance(entity_id, str) or not isinstance(field_path, str):
        return None
    value: Any = None
    for key in (
        "value",
        "raw_value",
        "normalized_value",
        "candidate_value_private",
        "content",
    ):
        candidate = row.get(key)
        if candidate is not None and str(candidate).strip():
            value = candidate
            break
    if value is None:
        return None
    result = {"entity_id": entity_id, "field_path": field_path, "value": value}
    for key in ("source_id", "evidence_quote", "evidence_span", "support_status"):
        if key in row:
            result[key] = row[key]
    return result


def _index_field_gold(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        identifier = _case_id(row, line_number)
        entry = grouped.setdefault(
            identifier,
            {
                "case_id": identifier,
                "gold_fields": [],
                "source_rows_private": [],
            },
        )
        entry["source_rows_private"].append(dict(row))
        nested_fields: Any = None
        for key in ("gold_fields", "exact_verified_fields_private", "fields"):
            if isinstance(row.get(key), list):
                nested_fields = row[key]
                break
        if isinstance(nested_fields, list):
            entry["gold_fields"].extend(
                dict(value) for value in nested_fields if isinstance(value, Mapping)
            )
        else:
            normalized = _field_from_row(row)
            if normalized is not None:
                entry["gold_fields"].append(normalized)
        for key in (
            "modality",
            "report_group_hmac",
            "source_id",
            "eligible",
            "family_id",
            "variant",
            "evaluation_track",
            "challenge_mode",
            "boundary_policy",
        ):
            if key in row and key not in entry:
                entry[key] = row[key]
    return grouped


def load_locked_inputs(
    seeds_path: Path,
    *,
    field_gold_path: Path | None = None,
    qzbc_root: Path | None = None,
    max_examples: int = -1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = _read_jsonl(seeds_path)
    if max_examples == 0 or max_examples < -1:
        raise EvaluationContractError("--max-examples must be -1 or a positive integer")
    identifiers: set[str] = set()
    mutable_seeds: list[dict[str, Any]] = []
    for line_number, raw_seed in enumerate(seeds, 1):
        seed = dict(raw_seed)
        identifier = _case_id(seed, line_number)
        if identifier in identifiers:
            raise EvaluationContractError(f"duplicate seed case id: {identifier}")
        identifiers.add(identifier)
        seed["_eval_case_id"] = identifier
        metadata = seed.get("metadata")
        metadata_modality = metadata.get("modality") if isinstance(metadata, Mapping) else None
        modality = str(seed.get("modality") or metadata_modality or "unknown")
        # Never reflect a free-text private modality label into public strata.
        modality = _public_modality(modality)
        seed["_eval_modality"] = modality
        evaluation_track, challenge_mode = _evaluation_track_contract(
            seed, case_id=identifier
        )
        seed["_eval_evaluation_track"] = evaluation_track
        seed["_eval_challenge_mode"] = challenge_mode
        mutable_seeds.append(seed)

    family_contract = _validate_family_contract(mutable_seeds)
    if max_examples > 0:
        if family_contract["mode"] == "four_state":
            selected_families: list[str] = []
            for seed in mutable_seeds:
                family_id = str(seed["_eval_family_id"])
                if family_id not in selected_families:
                    selected_families.append(family_id)
                if len(selected_families) == max_examples:
                    break
            selected = set(selected_families)
            mutable_seeds = [
                seed for seed in mutable_seeds if str(seed["_eval_family_id"]) in selected
            ]
        else:
            mutable_seeds = mutable_seeds[:max_examples]
        identifiers = {str(seed["_eval_case_id"]) for seed in mutable_seeds}
        family_contract = _validate_family_contract(mutable_seeds)

    gold_index: dict[str, dict[str, Any]] = {}
    if field_gold_path is not None:
        gold_index = _index_field_gold(_read_jsonl(field_gold_path))
        unexpected = sorted(set(gold_index) - identifiers)
        if unexpected and max_examples < 0:
            raise EvaluationContractError(
                f"field gold contains {len(unexpected)} case ids absent from seeds"
            )
        if unexpected and max_examples > 0:
            gold_index = {
                identifier: value
                for identifier, value in gold_index.items()
                if identifier in identifiers
            }
        missing = sorted(identifiers - set(gold_index))
        if missing:
            raise EvaluationContractError(
                f"field gold is missing {len(missing)} locked seed case ids"
            )
        for seed in mutable_seeds:
            identifier = seed["_eval_case_id"]
            if identifier in gold_index:
                gold = gold_index[identifier]
                gold_track, gold_challenge = _evaluation_track_contract(
                    gold, case_id=str(identifier)
                )
                if (
                    gold_track != seed["_eval_evaluation_track"]
                    or gold_challenge != seed["_eval_challenge_mode"]
                ):
                    raise EvaluationContractError(
                        f"{identifier}: seed/field-gold evaluation_track contract mismatch"
                    )
                for key in (
                    "modality",
                    "report_group_hmac",
                    "source_id",
                    "family_id",
                    "variant",
                    "evaluation_track",
                    "challenge_mode",
                ):
                    seed_value = seed.get(key)
                    gold_value = gold.get(key)
                    if seed_value is not None and gold_value is not None and seed_value != gold_value:
                        raise EvaluationContractError(
                            f"{identifier}: seed/field-gold {key} mismatch"
                        )
                seed["_eval_field_gold"] = gold

    contract_manifest = _materialize_messages(mutable_seeds, qzbc_root)
    evaluation_tracks = {
        str(seed["_eval_evaluation_track"]) for seed in mutable_seeds
    }
    challenge_modes = {bool(seed["_eval_challenge_mode"]) for seed in mutable_seeds}
    if len(evaluation_tracks) != 1 or len(challenge_modes) != 1:
        raise EvaluationContractError(
            "one evaluation run must contain exactly one evaluation_track/challenge_mode contract"
        )
    input_contract: dict[str, Any] = {
        "seed_sha256": _sha256_file(seeds_path.expanduser().resolve()),
        "field_gold_sha256": (
            _sha256_file(field_gold_path.expanduser().resolve())
            if field_gold_path is not None
            else None
        ),
        "case_count": len(mutable_seeds),
        "field_gold_case_count": len(gold_index),
        "family_mode": family_contract["mode"],
        "family_count": family_contract["family_count"],
        "variants": family_contract["variants"],
        "rows_per_family": family_contract["rows_per_family"],
        "evaluation_track": next(iter(evaluation_tracks)),
        "challenge_mode": next(iter(challenge_modes)),
        "qzbc_contract_fingerprint": (
            contract_manifest.get("contract_fingerprint")
            if isinstance(contract_manifest, Mapping)
            else None
        ),
    }
    input_contract["input_fingerprint"] = _sha256_bytes(_canonical_json(input_contract))
    return mutable_seeds, input_contract


def _model_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and (
            candidate.suffix.lower() in MODEL_FILE_SUFFIXES
            or candidate.name in {"tokenizer.model", "merges.txt", "vocab.txt"}
        )
    ]
    if not files:
        raise EvaluationContractError(f"local model directory has no checkpoint files: {path}")
    return sorted(files, key=lambda candidate: candidate.relative_to(path).as_posix())


def fingerprint_local_model(path: Path) -> dict[str, Any]:
    """Content-bind a local checkpoint, including every weight shard."""

    canonical = path.expanduser().resolve()
    if not canonical.exists():
        raise FileNotFoundError(canonical)
    inventory: list[dict[str, Any]] = []
    total_bytes = 0
    for candidate in _model_files(canonical):
        relative = candidate.name if canonical.is_file() else candidate.relative_to(canonical).as_posix()
        size = candidate.stat().st_size
        total_bytes += size
        inventory.append(
            {"relative_path": relative, "bytes": size, "sha256": _sha256_file(candidate)}
        )
    fingerprint = _sha256_bytes(_canonical_json(inventory))
    return {
        "canonical_path_private": str(canonical),
        "fingerprint": fingerprint,
        "file_count": len(inventory),
        "bytes": total_bytes,
        "files_private": inventory,
    }


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not label:
        raise EvaluationContractError("model label is empty after sanitization")
    return label


def parse_model_specs(values: Sequence[str]) -> list[dict[str, Any]]:
    if not values:
        raise EvaluationContractError("at least one --model is required")
    specs: list[dict[str, Any]] = []
    labels: set[str] = set()
    for value in values:
        if "=" in value:
            raw_label, raw_path = value.split("=", 1)
            label = _safe_label(raw_label)
            path = Path(raw_path)
        else:
            path = Path(value)
            label = _safe_label(path.name or "model")
        if label in labels:
            raise EvaluationContractError(f"duplicate model label: {label}")
        labels.add(label)
        specs.append({"label": label, "path": path.expanduser().resolve()})
    return specs


def _resolve_reference_labels(
    primary: str | None,
    additional: Sequence[str],
    labels: Sequence[str],
) -> list[str]:
    if not labels:
        raise EvaluationContractError("at least one model is required")
    requested = [primary or labels[0], *additional]
    resolved: list[str] = []
    for label in requested:
        if label not in labels:
            raise EvaluationContractError(f"unknown reference model label: {label}")
        if label not in resolved:
            resolved.append(label)
    return resolved


def _verifier_file_hash() -> str:
    path = Path(__file__).resolve().with_name("thyroid_verifier.py")
    if not path.is_file():
        raise RuntimeError(f"thyroid verifier not found: {path}")
    return _sha256_file(path)


def _load_verifier() -> Callable[..., Any]:
    module_dir = Path(__file__).resolve().parent
    source_root = Path(__file__).resolve().parents[2] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    try:
        from thyroid_verifier import assess_thyroid_response
    except ImportError as exc:
        raise RuntimeError("cannot import thyroid_verifier.assess_thyroid_response") from exc
    return assess_thyroid_response


def _allowed_source_ids(seed: Mapping[str, Any]) -> tuple[str, ...] | None:
    values: set[str] = set()
    direct = seed.get("allowed_source_ids")
    if isinstance(direct, list):
        values.update(str(value) for value in direct if str(value).strip())
    if isinstance(seed.get("source_id"), str) and str(seed["source_id"]).strip():
        values.add(str(seed["source_id"]))
    verification = seed.get("verification")
    if isinstance(verification, Mapping):
        for key in ("allowed_source_ids", "source_ids"):
            current = verification.get(key)
            if isinstance(current, list):
                values.update(str(value) for value in current if str(value).strip())
    request = seed.get("request")
    packet = request.get("evidence_packet") if isinstance(request, Mapping) else None
    if isinstance(packet, Mapping):
        for key in ("sources", "evidence", "records"):
            items = packet.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, Mapping):
                    source_id = item.get("source_id") or item.get("id")
                    if isinstance(source_id, str) and source_id.strip():
                        values.add(source_id)
    return tuple(sorted(values)) if values else None


def _gold_payload(seed: Mapping[str, Any]) -> dict[str, Any]:
    """Give the verifier a case-shaped payload while preserving private gold."""

    payload = {key: value for key, value in seed.items() if not key.startswith("_eval_")}
    gold = seed.get("_eval_field_gold")
    if isinstance(gold, Mapping):
        payload["field_gold"] = dict(gold)
        payload["gold_fields"] = list(gold.get("gold_fields") or [])
        payload["exact_verified_fields_private"] = list(gold.get("gold_fields") or [])
    return payload


def _assessment_mapping(assessment: Any) -> dict[str, Any]:
    if hasattr(assessment, "to_dict") and callable(assessment.to_dict):
        value = assessment.to_dict()
    elif dataclasses.is_dataclass(assessment):
        value = dataclasses.asdict(assessment)
    elif isinstance(assessment, Mapping):
        value = dict(assessment)
    elif hasattr(assessment, "__dict__"):
        value = dict(vars(assessment))
    else:
        raise RuntimeError("thyroid verifier returned an unsupported assessment type")
    return value


def _invoke_verifier(
    verifier: Callable[..., Any],
    gold_payload: Mapping[str, Any],
    raw_response: str,
    allowed_source_ids: Sequence[str] | None,
) -> Any:
    """Support the frozen two-argument API and older explicit-source API."""

    parameters = inspect.signature(verifier).parameters
    accepts_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if "allowed_source_ids" in parameters or accepts_keywords:
        return verifier(
            gold_payload,
            raw_response,
            allowed_source_ids=allowed_source_ids,
        )
    return verifier(gold_payload, raw_response)


def _number(mapping: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return default


def _boolean(mapping: Mapping[str, Any], *names: str, default: bool = False) -> bool:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, bool):
            return value
    return default


def _case_metrics(assessment: Mapping[str, Any]) -> dict[str, Any]:
    tp = int(_number(assessment, "tp", "true_positive_count"))
    fp = int(_number(assessment, "fp", "false_positive_count"))
    fn = int(_number(assessment, "fn", "false_negative_count"))
    expected = int(_number(assessment, "expected_count", "expected_field_count", default=tp + fn))
    emitted = int(
        _number(
            assessment,
            "emitted_count",
            "emitted_field_count",
            "emitted_section_count",
            default=tp + fp,
        )
    )
    unsupported = int(_number(assessment, "unsupported_count", "unsupported_field_count", default=-1))
    if unsupported < 0:
        errors = assessment.get("unsupported_errors")
        # ``ThyroidAssessment.to_dict()`` preserves tuples in memory, while a
        # JSONL round trip restores the same collection as a list.  Accept both
        # representations so fresh and resumed rows receive identical scores.
        unsupported = len(errors) if isinstance(errors, (list, tuple)) else 0
    exact = _boolean(assessment, "report_exact_match", "passed")
    schema_valid = _boolean(assessment, "schema_valid")
    source_valid = _boolean(assessment, "source_valid", default=True)
    # The cross-variant primary endpoint is deliberately bound to the
    # variant-aware verifier's `passed` decision.  Field exactness is a
    # secondary construct and must not redefine outage/conflict success.
    task_success = _boolean(
        assessment,
        "state_task_success",
        "task_success",
        "passed",
        default=False,
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "expected": expected,
        "emitted": emitted,
        "unsupported": unsupported,
        "eligible": expected > 0,
        "report_exact": exact,
        "schema_valid": schema_valid,
        "source_valid": source_valid,
        "task_success": task_success,
    }


def _generation_failure_assessment(message: str) -> dict[str, Any]:
    return {
        "passed": False,
        "state_task_success": False,
        "task_success": False,
        "report_exact_match": False,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "expected_count": 0,
        "emitted_count": 0,
        "schema_valid": False,
        "source_valid": False,
        "failure_types": ["generation_error"],
        "schema_errors": [message],
        "source_errors": [],
        "unsupported_errors": [],
        "field_results": [],
        "envelope_valid": False,
        "slot_schema_valid": False,
        "key_binding_valid": False,
        "packet_scaffolding_free": False,
        "content_complete": False,
        "grounded_content_complete": False,
    }


def _build_generation_runtime(
    model_path: Path,
    *,
    device: str,
    max_input_tokens: int,
) -> tuple[Any, Any, Any, Any]:
    care_root = Path(__file__).resolve().parents[2]
    source_root = care_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        import torch
        from care_opd.models import ModelArguments, load_causal_lm, load_tokenizer
    except ImportError as exc:
        raise RuntimeError("evaluation requires the CARE-OPD torch/transformers environment") from exc

    if device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    target_device = torch.device(resolved_device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device lacks BF16 support; refusing a dtype protocol change")
        dtype = "bfloat16"
    else:
        dtype = "float32"

    arguments = ModelArguments(
        student_model_name_or_path=str(model_path),
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    tokenizer = load_tokenizer(arguments)
    model = load_causal_lm(str(model_path), arguments, trainable=False)
    model.to(target_device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    # Retain the bound limit for a cheap assertion in the generated closure.
    return torch, tokenizer, model, (target_device, max_input_tokens, dtype)


def _resolve_device_contract(device: str) -> dict[str, str]:
    """Resolve ``auto`` before hashing the resume contract."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("evaluation requires torch") from exc
    resolved = "cuda:0" if device == "auto" and torch.cuda.is_available() else device
    if device == "auto" and not torch.cuda.is_available():
        resolved = "cpu"
    target = torch.device(resolved)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device lacks BF16 support; refusing a dtype protocol change")
        dtype = "bfloat16"
    else:
        dtype = "float32"
    try:
        import transformers

        transformers_version = str(transformers.__version__)
    except ImportError:
        transformers_version = "unavailable"
    result = {
        "device_request": device,
        "resolved_device": str(target),
        "resolved_dtype": dtype,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers_version,
        "cuda_runtime_version": str(torch.version.cuda or "none"),
    }
    if target.type == "cuda":
        index = target.index if target.index is not None else torch.cuda.current_device()
        result["cuda_device_name"] = str(torch.cuda.get_device_name(index))
        result["cuda_compute_capability"] = ".".join(
            str(value) for value in torch.cuda.get_device_capability(index)
        )
    return result


def _generate_one(
    torch: Any,
    tokenizer: Any,
    model: Any,
    runtime: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, float, int, int]:
    target_device, max_input_tokens, _dtype = runtime
    try:
        rendered = tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "tokenizer chat template does not accept enable_thinking=False; "
            "refusing a protocol-changing fallback"
        ) from exc
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens > max_input_tokens:
        raise EvaluationContractError(
            f"encoded prompt has {input_tokens} tokens, exceeding locked limit {max_input_tokens}"
        )
    encoded = {key: value.to(target_device) for key, value in encoded.items()}
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    latency = time.perf_counter() - started
    new_tokens = generated[0, input_tokens:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, latency, input_tokens, int(new_tokens.numel())


def _recover_private_cases(
    path: Path,
    *,
    expected: Mapping[int, str],
    run_fingerprint: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_bytes().splitlines()
    rows: list[dict[str, Any]] = []
    corrupt_terminal = False
    seen: set[int] = set()
    for line_index, line in enumerate(lines):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if line_index != len(lines) - 1:
                raise EvaluationContractError(
                    f"corrupt non-terminal private result at line {line_index + 1}"
                )
            corrupt_terminal = True
            break
        index = int(row.get("index", -1))
        if index in seen:
            raise EvaluationContractError(f"duplicate resumed result index {index}")
        if expected.get(index) != str(row.get("case_id")):
            raise EvaluationContractError(f"resumed result does not match locked case index {index}")
        if row.get("run_fingerprint") != run_fingerprint:
            raise EvaluationContractError(f"stale result fingerprint at case index {index}")
        if not isinstance(row.get("assessment"), Mapping):
            raise EvaluationContractError(f"resumed case {index} lacks an assessment")
        seen.add(index)
        rows.append(row)
    if corrupt_terminal:
        _atomic_jsonl(path, rows)
    return rows


def evaluate_model(
    *,
    label: str,
    model_path: Path,
    seeds: Sequence[Mapping[str, Any]],
    private_cases_path: Path,
    run_fingerprint: str,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    generation_seed: int,
    resume: bool,
    generator_factory: Callable[..., Any] = _build_generation_runtime,
    verifier: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    expected = {index: str(seed["_eval_case_id"]) for index, seed in enumerate(seeds)}
    existing = (
        _recover_private_cases(
            private_cases_path, expected=expected, run_fingerprint=run_fingerprint
        )
        if resume
        else []
    )
    completed = {int(row["index"]) for row in existing}
    remaining = [(index, seed) for index, seed in enumerate(seeds) if index not in completed]
    if not remaining:
        return sorted(existing, key=lambda row: int(row["index"]))

    assess = verifier or _load_verifier()
    torch = tokenizer = model = runtime = None
    try:
        torch, tokenizer, model, runtime = generator_factory(
            model_path, device=device, max_input_tokens=max_input_tokens
        )
        for position, (index, seed) in enumerate(remaining, 1):
            raw_response = ""
            latency_seconds = 0.0
            input_tokens = output_tokens = 0
            generation_error: str | None = None
            try:
                raw_response, latency_seconds, input_tokens, output_tokens = _generate_one(
                    torch,
                    tokenizer,
                    model,
                    runtime,
                    seed["_eval_messages"],
                    max_new_tokens=max_new_tokens,
                    seed=generation_seed,
                )
            except Exception as exc:  # Per-report traces preserve failures for audit.
                generation_error = str(exc) or exc.__class__.__name__
            # Score even an empty/failed generation.  Otherwise generation errors
            # would be incorrectly removed from recall and exact-match denominators.
            try:
                assessment = _assessment_mapping(
                    _invoke_verifier(
                        assess,
                        _gold_payload(seed),
                        raw_response,
                        _allowed_source_ids(seed),
                    )
                )
                declared_track = assessment.get("evaluation_track")
                if (
                    declared_track is not None
                    and declared_track != seed["_eval_evaluation_track"]
                ):
                    raise EvaluationContractError(
                        f"verifier evaluation_track mismatch at locked case index {index}"
                    )
                declared_challenge = assessment.get("challenge_mode")
                if (
                    declared_challenge is not None
                    and declared_challenge != seed["_eval_challenge_mode"]
                ):
                    raise EvaluationContractError(
                        f"verifier challenge_mode mismatch at locked case index {index}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"thyroid verifier failed on locked case index {index}"
                ) from exc
            row = {
                "schema_version": SCHEMA_VERSION + ".private_case",
                "run_fingerprint": run_fingerprint,
                "model_label": label,
                "index": index,
                "case_id": str(seed["_eval_case_id"]),
                "family_id": str(seed["_eval_family_id"]),
                "variant": str(seed["_eval_variant"]),
                "evaluation_track": str(seed["_eval_evaluation_track"]),
                "challenge_mode": bool(seed["_eval_challenge_mode"]),
                "report_group_hmac": seed.get("report_group_hmac"),
                "modality": str(seed["_eval_modality"]),
                "seed_record_sha256": _sha256_bytes(
                    _canonical_json(
                        {key: value for key, value in seed.items() if not key.startswith("_eval_")}
                    )
                ),
                "raw_response_private": raw_response,
                "response_sha256": _sha256_bytes(raw_response.encode("utf-8")),
                "generation_error": generation_error,
                "latency_seconds": latency_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "assessment": assessment,
                "metrics": _case_metrics(assessment),
            }
            _append_jsonl(private_cases_path, row)
            existing.append(row)
            print(
                f"[{label}] {position}/{len(remaining)} case={index + 1}/{len(seeds)} "
                f"exact={int(row['metrics']['report_exact'])}",
                flush=True,
            )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sorted(existing, key=lambda row: int(row["index"]))


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_case_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    # Per-case ``metrics`` are a convenience snapshot, not an authoritative
    # cache.  Recompute from the assessment so aggregation fixes can be applied
    # to completed JSONL runs without regenerating private model responses.
    metrics = [_case_metrics(row["assessment"]) for row in rows]
    eligible = [metric for metric in metrics if metric["eligible"]]
    tp = sum(int(metric["tp"]) for metric in eligible)
    fp = sum(int(metric["fp"]) for metric in eligible)
    fn = sum(int(metric["fn"]) for metric in eligible)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1_denominator = 2 * tp + fp + fn
    f1 = _safe_div(2 * tp, f1_denominator)
    emitted = sum(int(metric["emitted"]) for metric in metrics)
    unsupported = sum(int(metric["unsupported"]) for metric in metrics)
    failure_counts: Counter[str] = Counter()
    generation_errors = 0
    for row in rows:
        assessment = row.get("assessment") or {}
        failure_types = assessment.get("failure_types") if isinstance(assessment, Mapping) else []
        if isinstance(failure_types, (list, tuple)):
            failure_counts.update(
                str(value)
                if str(value) in PUBLIC_FAILURE_TYPES
                else "unclassified_failure"
                for value in failure_types
            )
        generation_errors += int(bool(row.get("generation_error")))
    result = {
        "reports": total,
        "eligible_reports": len(eligible),
        "eligible_report_coverage": _safe_div(len(eligible), total),
        "eligible_fields": sum(int(metric["expected"]) for metric in eligible),
        "field_tp": tp,
        "field_fp": fp,
        "field_fn": fn,
        "field_micro_precision": precision,
        "field_micro_recall": recall,
        "field_micro_f1": f1,
        "report_exact_rate": _safe_div(
            sum(int(metric["report_exact"]) for metric in eligible), len(eligible)
        ),
        "schema_valid_rate": _safe_div(
            sum(int(metric["schema_valid"]) for metric in metrics), total
        ),
        "source_valid_rate": _safe_div(
            sum(int(metric["source_valid"]) for metric in metrics), total
        ),
        "unsupported_fields": unsupported,
        "emitted_fields": emitted,
        "unsupported_field_rate": _safe_div(unsupported, emitted),
        "generation_error_count": generation_errors,
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "mean_latency_seconds": _safe_div(
            sum(float(row.get("latency_seconds") or 0.0) for row in rows), total
        ),
        "mean_output_tokens": _safe_div(
            sum(int(row.get("output_tokens") or 0) for row in rows), total
        ),
    }
    return result


def _result_family_variant(row: Mapping[str, Any]) -> tuple[str, str]:
    case_id = str(row.get("case_id") or row.get("index") or "").strip()
    family_id = str(row.get("family_id") or case_id).strip()
    variant = str(row.get("variant") or "clean").strip()
    if not family_id or variant not in FOUR_STATE_VARIANTS:
        raise EvaluationContractError("result row has an invalid family_id/variant contract")
    return family_id, variant


def _index_result_families(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], str]:
    families: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    modalities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_id, variant = _result_family_variant(row)
        if variant in families[family_id]:
            raise EvaluationContractError(
                f"result family {family_id!r} has duplicate variant {variant!r}"
            )
        families[family_id][variant] = row
        modalities[family_id].add(_public_modality(row.get("modality")))
    if any(len(values) != 1 for values in modalities.values()):
        raise EvaluationContractError("a result family spans multiple modalities")
    variant_sets = {tuple(sorted(values)) for values in families.values()}
    if variant_sets == {("clean",)}:
        return dict(families), "clean_only"
    expected = set(FOUR_STATE_VARIANTS)
    if families and all(set(values) == expected for values in families.values()):
        return dict(families), "four_state"
    raise EvaluationContractError(
        "results must contain either one clean row or a closed four-state quartet per family"
    )


def _task_success(row: Mapping[str, Any]) -> bool:
    assessment = row.get("assessment")
    if not isinstance(assessment, Mapping):
        return False
    # V4 separates state behavior from exact extraction.  Older assessments
    # remain readable through their explicit task_success/passed decisions.
    for key in ("state_task_success", "task_success", "passed"):
        value = assessment.get(key)
        if isinstance(value, bool):
            return value
    return False


def _result_evaluation_contract(row: Mapping[str, Any]) -> tuple[str, bool]:
    assessment = row.get("assessment")
    assessment = assessment if isinstance(assessment, Mapping) else {}
    row_track = row.get("evaluation_track")
    assessment_track = assessment.get("evaluation_track")
    if row_track is not None and assessment_track is not None and row_track != assessment_track:
        raise EvaluationContractError("result row/assessment evaluation_track mismatch")
    track = str(assessment_track or row_track or LEGACY_EVALUATION_TRACK).strip()
    if track not in EVALUATION_TRACKS:
        raise EvaluationContractError(f"result row has unsupported evaluation_track {track!r}")
    expected_challenge = track == "raw_strict_extraction_challenge"
    declared_values = [
        value
        for value in (row.get("challenge_mode"), assessment.get("challenge_mode"))
        if value is not None
    ]
    if any(not isinstance(value, bool) for value in declared_values):
        raise EvaluationContractError("result challenge_mode must be boolean")
    if declared_values and any(value != declared_values[0] for value in declared_values):
        raise EvaluationContractError("result row/assessment challenge_mode mismatch")
    challenge_mode = declared_values[0] if declared_values else expected_challenge
    if challenge_mode != expected_challenge:
        raise EvaluationContractError("result challenge_mode/evaluation_track mismatch")
    return track, challenge_mode


def _optional_nonnegative_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def aggregate_three_axis_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate v4 state, protocol, and content axes without conflating them."""

    extraction_rows = [
        row
        for row in rows
        if _result_family_variant(row)[1] in SECONDARY_FIELD_VARIANTS
    ]
    tracks: Counter[str] = Counter()
    challenge_count = 0
    for row in rows:
        track, challenge_mode = _result_evaluation_contract(row)
        tracks[track] += 1
        challenge_count += int(challenge_mode)

    protocol: dict[str, Any] = {}
    for key in PROTOCOL_ASSESSMENT_FIELDS:
        scoped_rows = (
            extraction_rows if key in EXTRACTION_SCOPED_PROTOCOL_FIELDS else rows
        )
        scope = (
            "clean+bad_history"
            if key in EXTRACTION_SCOPED_PROTOCOL_FIELDS
            else "all_states"
        )
        measurement = _optional_boolean_rate(scoped_rows, key)
        protocol[f"{key}_scope"] = scope
        protocol[f"{key}_eligible_reports"] = len(scoped_rows)
        protocol[f"{key}_rate"] = measurement["rate"]
        protocol[f"{key}_evaluated_reports"] = measurement["evaluated_reports"]

    content: dict[str, Any] = {}
    for axis in CONTENT_AXES:
        totals = {name: 0 for name in ("tp", "fp", "fn")}
        evaluated_reports = 0
        for row in extraction_rows:
            assessment = row.get("assessment")
            if not isinstance(assessment, Mapping):
                continue
            values = {
                name: _optional_nonnegative_int(assessment, f"{axis}_{name}")
                for name in totals
            }
            present = [value is not None for value in values.values()]
            if any(present) and not all(present):
                raise EvaluationContractError(
                    f"assessment must provide a complete {axis}_tp/fp/fn triplet"
                )
            if all(present):
                evaluated_reports += 1
                for name, value in values.items():
                    totals[name] += int(value)
        precision = _safe_div(totals["tp"], totals["tp"] + totals["fp"])
        recall = _safe_div(totals["tp"], totals["tp"] + totals["fn"])
        f1 = _safe_div(
            2 * totals["tp"],
            2 * totals["tp"] + totals["fp"] + totals["fn"],
        )
        completeness = _optional_boolean_rate(
            extraction_rows, f"{axis}_complete"
        )
        content.update(
            {
                f"{axis}_scope": "clean+bad_history",
                f"{axis}_eligible_reports": len(extraction_rows),
                f"{axis}_evaluated_reports": evaluated_reports,
                f"{axis}_tp": totals["tp"],
                f"{axis}_fp": totals["fp"],
                f"{axis}_fn": totals["fn"],
                f"{axis}_micro_precision": precision,
                f"{axis}_precision_denominator": totals["tp"] + totals["fp"],
                f"{axis}_micro_recall": recall,
                f"{axis}_recall_denominator": totals["tp"] + totals["fn"],
                f"{axis}_micro_f1": f1,
                f"{axis}_f1_denominator": (
                    2 * totals["tp"] + totals["fp"] + totals["fn"]
                ),
                f"{axis}_report_complete_rate": completeness["rate"],
                f"{axis}_report_complete_evaluated_reports": completeness[
                    "evaluated_reports"
                ],
            }
        )
    return {
        "reports": len(rows),
        "state_scope": "all_states",
        "state_task_success_evaluated_reports": len(rows),
        "evaluation_track": next(iter(tracks)) if len(tracks) == 1 else "mixed",
        "evaluation_track_counts": dict(sorted(tracks.items())),
        "challenge_mode_reports": challenge_count,
        "challenge_mode_rate": _safe_div(challenge_count, len(rows)),
        "state_task_success_rate": _safe_div(
            sum(_task_success(row) for row in rows), len(rows)
        ),
        **protocol,
        **content,
    }


def _optional_boolean_rate(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    values: list[bool] = []
    for row in rows:
        assessment = row.get("assessment")
        value = assessment.get(key) if isinstance(assessment, Mapping) else None
        if isinstance(value, bool):
            values.append(value)
    return {
        "rate": _safe_div(sum(values), len(values)),
        "evaluated_reports": len(values),
    }


def _public_boundary_error_class(value: Any) -> str:
    """Collapse private verifier detail to a fixed public-safe error class."""

    text = str(value)
    if "field_forbidden_at_boundary" in text:
        return "boundary_field_emission"
    if "source_forbidden_at_boundary" in text:
        return "boundary_source_emission"
    if text.startswith("missing_boundary_components"):
        return "missing_boundary_structure"
    if text.startswith("missing_boundary_semantics"):
        return "missing_boundary_semantics"
    for category in (
        "invalid_generated_response",
        "controlled_sentinel_leakage",
        "unsafe_clinical_conclusion",
        "failed_evidence_boundary_recovery",
    ):
        if text == category:
            return category
    # Never preserve unknown payload text: it may contain a private source ID,
    # entity key, or model response fragment.
    return "unclassified_boundary_error"


def aggregate_task_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate primary four-state behavior with family as the unit."""

    families, mode = _index_result_families(rows)
    by_variant_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for variants in families.values():
        for variant, row in variants.items():
            by_variant_rows[variant].append(row)

    by_variant: dict[str, dict[str, Any]] = {}
    for variant in FOUR_STATE_VARIANTS:
        subset = by_variant_rows.get(variant, [])
        successes = sum(_task_success(row) for row in subset)
        by_variant[variant] = {
            "reports": len(subset),
            "task_successes": successes,
            "task_success_rate": _safe_div(successes, len(subset)),
        }

    clean_rows = by_variant_rows.get("clean", [])
    clean_successes = sum(_task_success(row) for row in clean_rows)
    perturbed_rows = [
        row
        for variant in PERTURBED_VARIANTS
        for row in by_variant_rows.get(variant, [])
    ]
    perturbed_successes = sum(_task_success(row) for row in perturbed_rows)

    retention_denominator = 0
    retained = 0
    bad_history_retention_denominator = 0
    bad_history_retained = 0
    joint_successes = 0
    if mode == "four_state":
        for variants in families.values():
            clean_success = _task_success(variants["clean"])
            if clean_success:
                retention_denominator += len(PERTURBED_VARIANTS)
                retained += sum(
                    _task_success(variants[variant]) for variant in PERTURBED_VARIANTS
                )
                bad_history_retention_denominator += 1
                bad_history_retained += int(_task_success(variants["bad_history"]))
            joint_successes += int(
                all(_task_success(variants[variant]) for variant in FOUR_STATE_VARIANTS)
            )
    conditional_retention = _safe_div(retained, retention_denominator)
    bad_history_conditional_retention = _safe_div(
        bad_history_retained, bad_history_retention_denominator
    )

    boundary_metrics: dict[str, dict[str, Any]] = {}
    for variant in BOUNDARY_VARIANTS:
        subset = by_variant_rows.get(variant, [])
        item: dict[str, Any] = {
            "reports": len(subset),
            "task_success_rate": by_variant[variant]["task_success_rate"],
        }
        for assessment_key in BOUNDARY_ASSESSMENT_FIELDS:
            measurement = _optional_boolean_rate(subset, assessment_key)
            item[f"{assessment_key}_rate"] = measurement["rate"]
            item[f"{assessment_key}_evaluated_reports"] = measurement[
                "evaluated_reports"
            ]
        errors: Counter[str] = Counter()
        for row in subset:
            assessment = row.get("assessment")
            values = assessment.get("boundary_errors") if isinstance(assessment, Mapping) else None
            if isinstance(values, (list, tuple)):
                errors.update(_public_boundary_error_class(value) for value in values)
        item["boundary_error_counts"] = dict(sorted(errors.items()))
        boundary_metrics[variant] = item

    regression = (
        1.0 - conditional_retention if conditional_retention is not None else None
    )
    bad_history_regression = (
        1.0 - bad_history_conditional_retention
        if bad_history_conditional_retention is not None
        else None
    )
    joint_rate = (
        _safe_div(joint_successes, len(families)) if mode == "four_state" else None
    )
    return {
        "family_mode": mode,
        "families": len(families),
        "complete_four_state_families": len(families) if mode == "four_state" else 0,
        "reports": len(rows),
        "generation_error_count": sum(bool(row.get("generation_error")) for row in rows),
        "mean_latency_seconds": _safe_div(
            sum(float(row.get("latency_seconds") or 0.0) for row in rows), len(rows)
        ),
        "mean_output_tokens": _safe_div(
            sum(int(row.get("output_tokens") or 0) for row in rows), len(rows)
        ),
        "clean_reports": len(clean_rows),
        "perturbed_reports": len(perturbed_rows),
        "clean_task_successes": clean_successes,
        "perturbed_task_successes": perturbed_successes,
        "clean_task_success_rate": _safe_div(clean_successes, len(clean_rows)),
        "perturbed_task_success_rate": (
            _safe_div(perturbed_successes, len(perturbed_rows))
            if mode == "four_state"
            else None
        ),
        "descriptive_pooled_perturbed_task_success_rate": (
            _safe_div(perturbed_successes, len(perturbed_rows))
            if mode == "four_state"
            else None
        ),
        "perturbed_task_success_rate_is_descriptive": True,
        "conditional_retention": conditional_retention,
        "conditional_retention_numerator": retained if mode == "four_state" else 0,
        "conditional_retention_denominator": (
            retention_denominator if mode == "four_state" else 0
        ),
        "regression_rate": regression,
        "bad_history_conditional_retention": bad_history_conditional_retention,
        "bad_history_conditional_retention_numerator": (
            bad_history_retained if mode == "four_state" else 0
        ),
        "bad_history_conditional_retention_denominator": (
            bad_history_retention_denominator if mode == "four_state" else 0
        ),
        "bad_history_regression_rate": bad_history_regression,
        "four_state_joint_successes": joint_successes if mode == "four_state" else 0,
        "four_state_joint_tsr": joint_rate,
        "joint_four_state_task_success_rate": joint_rate,
        "by_variant": by_variant,
        "boundary_metrics": boundary_metrics,
    }


def _secondary_field_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if _result_family_variant(row)[1] in SECONDARY_FIELD_VARIANTS
    ]


def _secondary_field_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = _secondary_field_rows(rows)
    aggregate = aggregate_case_metrics(selected)
    aggregate["scope"] = "clean+bad_history" if any(
        _result_family_variant(row)[1] == "bad_history" for row in selected
    ) else "clean_only"
    aggregate["included_variants"] = [
        variant
        for variant in SECONDARY_FIELD_VARIANTS
        if any(_result_family_variant(row)[1] == variant for row in selected)
    ]
    return aggregate


def summarize_model(label: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_modality: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_modality[_public_modality(row.get("modality"))].append(row)
    secondary = _secondary_field_summary(rows)
    secondary_by_modality = {
        modality: _secondary_field_summary(subset)
        for modality, subset in sorted(by_modality.items())
    }
    secondary_by_variant = {
        variant: aggregate_case_metrics(
            [row for row in rows if _result_family_variant(row)[1] == variant]
        )
        for variant in SECONDARY_FIELD_VARIANTS
        if any(_result_family_variant(row)[1] == variant for row in rows)
    }
    three_axis = aggregate_three_axis_metrics(rows)
    return {
        "model": label,
        "primary": aggregate_task_metrics(rows),
        "primary_by_modality": {
            modality: aggregate_task_metrics(subset)
            for modality, subset in sorted(by_modality.items())
        },
        "three_axis": three_axis,
        "three_axis_by_modality": {
            modality: aggregate_three_axis_metrics(subset)
            for modality, subset in sorted(by_modality.items())
        },
        "secondary_field_metrics": secondary,
        "secondary_field_by_variant": secondary_by_variant,
        "secondary_field_by_modality": secondary_by_modality,
        # Compatibility aliases: unlike the old evaluator these are scoped to
        # clean+bad_history only, never to outage/conflict boundary states.
        "overall": secondary,
        "by_modality": secondary_by_modality,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise EvaluationContractError("cannot take a quantile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_value(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    summary = aggregate_case_metrics(rows)
    value = summary.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def paired_report_bootstrap(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    reference_label: str,
    candidate_label: str,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    stratum: str = "overall",
) -> list[dict[str, Any]]:
    if replicates < 1:
        raise EvaluationContractError("bootstrap replicates must be positive")
    reference = {int(row["index"]): row for row in reference_rows}
    candidate = {int(row["index"]): row for row in candidate_rows}
    if set(reference) != set(candidate):
        raise EvaluationContractError("paired model results do not contain identical reports")
    indices = sorted(reference)
    if stratum != "overall":
        indices = [
            index for index in indices if str(reference[index].get("modality")) == stratum
        ]
    if not indices:
        return []
    metrics = (
        "field_micro_precision",
        "field_micro_recall",
        "field_micro_f1",
        "report_exact_rate",
        "schema_valid_rate",
        "unsupported_field_rate",
    )
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sampled = [indices[rng.randrange(len(indices))] for _ in indices]
        ref_sample = [reference[index] for index in sampled]
        cand_sample = [candidate[index] for index in sampled]
        for metric in metrics:
            ref_value = _metric_value(ref_sample, metric)
            cand_value = _metric_value(cand_sample, metric)
            if ref_value is not None and cand_value is not None:
                draws[metric].append(cand_value - ref_value)
    ref_rows = [reference[index] for index in indices]
    cand_rows = [candidate[index] for index in indices]
    results: list[dict[str, Any]] = []
    for metric in metrics:
        ref_value = _metric_value(ref_rows, metric)
        cand_value = _metric_value(cand_rows, metric)
        distribution = draws[metric]
        results.append(
            {
                "reference_model": reference_label,
                "candidate_model": candidate_label,
                "stratum": stratum,
                "metric": metric,
                "paired_reports": len(indices),
                "reference_value": ref_value,
                "candidate_value": cand_value,
                "delta_candidate_minus_reference": (
                    cand_value - ref_value
                    if cand_value is not None and ref_value is not None
                    else None
                ),
                "ci95_lower": _quantile(distribution, 0.025) if distribution else None,
                "ci95_upper": _quantile(distribution, 0.975) if distribution else None,
                "bootstrap_replicates": replicates,
                "valid_bootstrap_replicates": len(distribution),
                "bootstrap_seed": seed,
                "direction": (
                    "lower_is_better" if metric == "unsupported_field_rate" else "higher_is_better"
                ),
            }
        )
    return results


def _comparison_metric_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    primary = aggregate_task_metrics(rows)
    secondary = _secondary_field_summary(rows)
    values: dict[str, float | None] = {
        "clean_task_success_rate": primary["clean_task_success_rate"],
        "perturbed_task_success_rate": primary["perturbed_task_success_rate"],
        "conditional_retention": primary["conditional_retention"],
        "regression_rate": primary["regression_rate"],
        "bad_history_conditional_retention": primary[
            "bad_history_conditional_retention"
        ],
        "bad_history_regression_rate": primary["bad_history_regression_rate"],
        "four_state_joint_tsr": primary["four_state_joint_tsr"],
        "secondary_field_micro_precision": secondary["field_micro_precision"],
        "secondary_field_micro_recall": secondary["field_micro_recall"],
        "secondary_field_micro_f1": secondary["field_micro_f1"],
    }
    for variant in PERTURBED_VARIANTS:
        values[f"{variant}_task_success_rate"] = primary["by_variant"][variant][
            "task_success_rate"
        ]
    for variant in BOUNDARY_VARIANTS:
        boundary = primary["boundary_metrics"][variant]
        for key in BOUNDARY_ASSESSMENT_FIELDS:
            values[f"{variant}_{key}_rate"] = boundary[f"{key}_rate"]
    for variant in SECONDARY_FIELD_VARIANTS:
        subset = [row for row in rows if _result_family_variant(row)[1] == variant]
        metrics = aggregate_case_metrics(subset) if subset else {}
        for key in ("field_micro_precision", "field_micro_recall", "field_micro_f1"):
            value = metrics.get(key)
            values[f"{variant}_{key}"] = (
                float(value) if isinstance(value, (int, float)) else None
            )
    return {
        key: (float(value) if isinstance(value, (int, float)) else None)
        for key, value in values.items()
    }


def _metric_family(metric: str) -> str:
    if metric == "perturbed_task_success_rate":
        return "descriptive_pooled_state"
    if (
        "boundary_" in metric
        or "sentinel_" in metric
        or "unsafe_clinical_conclusion" in metric
    ):
        return "boundary_diagnostic"
    if "field_micro_" in metric:
        return "secondary_field"
    return "primary_task"


def paired_family_bootstrap(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    reference_label: str,
    candidate_label: str,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    stratum: str = "overall",
) -> list[dict[str, Any]]:
    """Paired percentile bootstrap that always resamples whole families."""

    if replicates < 1:
        raise EvaluationContractError("bootstrap replicates must be positive")
    reference_subset = [
        row
        for row in reference_rows
        if stratum == "overall" or _public_modality(row.get("modality")) == stratum
    ]
    candidate_subset = [
        row
        for row in candidate_rows
        if stratum == "overall" or _public_modality(row.get("modality")) == stratum
    ]
    reference, reference_mode = _index_result_families(reference_subset)
    candidate, candidate_mode = _index_result_families(candidate_subset)
    if reference_mode != candidate_mode or set(reference) != set(candidate):
        raise EvaluationContractError("paired models do not contain identical family sets")
    for family_id in reference:
        if set(reference[family_id]) != set(candidate[family_id]):
            raise EvaluationContractError(
                f"paired family {family_id!r} has different variant sets"
            )
    family_ids = sorted(reference)
    if not family_ids:
        return []

    def expand(
        index: Mapping[str, Mapping[str, Mapping[str, Any]]], selected: Sequence[str]
    ) -> list[Mapping[str, Any]]:
        expanded: list[Mapping[str, Any]] = []
        for draw_index, family_id in enumerate(selected):
            # A family may be sampled more than once.  Rekey each cluster copy
            # so the ordinary family validator still sees one closed quartet.
            for variant in FOUR_STATE_VARIANTS:
                if variant not in index[family_id]:
                    continue
                copied = dict(index[family_id][variant])
                copied["family_id"] = f"bootstrap_family_{draw_index}"
                expanded.append(copied)
        return expanded

    reference_values = _comparison_metric_values(expand(reference, family_ids))
    candidate_values = _comparison_metric_values(expand(candidate, family_ids))
    metric_names = tuple(reference_values)
    draws: dict[str, list[float]] = {metric: [] for metric in metric_names}
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled_families = [family_ids[rng.randrange(len(family_ids))] for _ in family_ids]
        ref_values = _comparison_metric_values(expand(reference, sampled_families))
        cand_values = _comparison_metric_values(expand(candidate, sampled_families))
        for metric in metric_names:
            left = ref_values[metric]
            right = cand_values[metric]
            if left is not None and right is not None:
                draws[metric].append(right - left)

    results: list[dict[str, Any]] = []
    for metric in metric_names:
        left = reference_values[metric]
        right = candidate_values[metric]
        distribution = draws[metric]
        results.append(
            {
                "reference_model": reference_label,
                "candidate_model": candidate_label,
                "stratum": stratum,
                "metric_family": _metric_family(metric),
                "metric": metric,
                "bootstrap_unit": "family_id",
                "paired_families": len(family_ids),
                "paired_reports": len(reference_subset),
                "reference_value": left,
                "candidate_value": right,
                "delta_candidate_minus_reference": (
                    right - left if left is not None and right is not None else None
                ),
                "ci95_lower": _quantile(distribution, 0.025) if distribution else None,
                "ci95_upper": _quantile(distribution, 0.975) if distribution else None,
                "bootstrap_replicates": replicates,
                "valid_bootstrap_replicates": len(distribution),
                "bootstrap_seed": seed,
                "direction": (
                    "lower_is_better"
                    if metric in {"regression_rate", "bad_history_regression_rate"}
                    or "unsafe_clinical_conclusion" in metric
                    else "higher_is_better"
                ),
            }
        )
    return results


def build_paired_family_comparisons(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference_labels: Sequence[str],
    modalities: Sequence[str],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    labels = list(results)
    for reference_label in reference_labels:
        if reference_label not in results:
            raise EvaluationContractError(
                f"comparison reference {reference_label!r} has no model results"
            )
        for candidate_label in labels:
            if candidate_label == reference_label:
                continue
            for stratum_index, stratum in enumerate(["overall", *modalities]):
                comparisons.extend(
                    paired_family_bootstrap(
                        results[reference_label],
                        results[candidate_label],
                        reference_label=reference_label,
                        candidate_label=candidate_label,
                        replicates=replicates,
                        seed=seed + stratum_index,
                        stratum=stratum,
                    )
                )
    return comparisons


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def _track_presentation(evaluation_track: str) -> tuple[str, str]:
    if evaluation_track == "structured_evidence_faithfulness":
        return (
            "CARE-OPD real-thyroid structured-evidence faithfulness assay",
            "report-level visible structured-evidence faithfulness and evidence-state "
            "recovery; not hidden-label extraction, clinical diagnostic accuracy, or "
            "prospective clinical validity",
        )
    if evaluation_track == "raw_strict_extraction_challenge":
        return (
            "CARE-OPD real-thyroid raw-report strict extraction challenge",
            "report-level source-visible span-localized strict extraction and "
            "evidence-state recovery; not complete-report extraction, clinical diagnostic "
            "accuracy, or prospective clinical validity",
        )
    return (
        "CARE-OPD real-thyroid external evaluation",
        "track-specific report-level external evaluation; not clinical diagnostic or "
        "prospective clinical validity",
    )


def _render_markdown(
    summaries: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]
) -> str:
    tracks = {
        str(summary.get("three_axis", {}).get("evaluation_track") or "")
        for summary in summaries
    }
    evaluation_track = next(iter(tracks)) if len(tracks) == 1 else "mixed"
    title, claim_boundary = _track_presentation(evaluation_track)
    lines = [
        f"# {title}",
        "",
        "## Axis 1: state-sensitive task behavior",
        "",
        "Primary task success uses `assessment.state_task_success`, with legacy fallback to "
        "`task_success` and then `passed`. Bad-history retention conditions only on clean-success "
        "families. The pooled perturbed rate is descriptive because it mixes three distinct states.",
        "",
        "| Model | Stratum | Families | Clean | Bad history | Bad-history retention | Source outage | Evidence conflict | Joint 4-state | Pooled perturbed (descriptive) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        strata = [
            ("overall", summary["primary"]),
            *summary["primary_by_modality"].items(),
        ]
        for stratum, metrics in strata:
            lines.append(
                "| {model} | {stratum} | {families} | {clean} | {bad} | "
                "{bad_retention} | {outage} | {conflict} | {joint} | {perturbed} |".format(
                    model=summary["model"],
                    stratum=stratum,
                    families=metrics["families"],
                    clean=_format_metric(metrics["clean_task_success_rate"]),
                    perturbed=_format_metric(metrics["perturbed_task_success_rate"]),
                    bad=_format_metric(metrics["by_variant"]["bad_history"]["task_success_rate"]),
                    bad_retention=_format_metric(
                        metrics["bad_history_conditional_retention"]
                    ),
                    outage=_format_metric(metrics["by_variant"]["source_outage"]["task_success_rate"]),
                    conflict=_format_metric(metrics["by_variant"]["evidence_conflict"]["task_success_rate"]),
                    joint=_format_metric(metrics["four_state_joint_tsr"]),
                )
            )

    lines += [
        "",
        "## Axis 2: protocol conformance",
        "",
        "Rates are reported only when the verifier emits the corresponding v4 boolean; legacy "
        "v1/v2 assessments remain N/A rather than being imputed. Cells show rate (evaluated/eligible). "
        "Slot schema, key binding, and auxiliary-field emptiness use only clean+bad_history; "
        "envelope and packet-scaffolding diagnostics use all states.",
        "",
        "| Model | Stratum | Track | Challenge mode | Envelope | Slot schema | Key binding | Auxiliary empty | Packet-scaffolding free |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for stratum, metrics in [
            ("overall", summary["three_axis"]),
            *summary["three_axis_by_modality"].items(),
        ]:
            lines.append(
                f"| {summary['model']} | {stratum} | {metrics['evaluation_track']} | "
                f"{_format_metric(metrics['challenge_mode_rate'])} | "
                f"{_format_metric(metrics['envelope_valid_rate'])} "
                f"({metrics['envelope_valid_evaluated_reports']}/{metrics['envelope_valid_eligible_reports']}) | "
                f"{_format_metric(metrics['slot_schema_valid_rate'])} "
                f"({metrics['slot_schema_valid_evaluated_reports']}/{metrics['slot_schema_valid_eligible_reports']}) | "
                f"{_format_metric(metrics['key_binding_valid_rate'])} "
                f"({metrics['key_binding_valid_evaluated_reports']}/{metrics['key_binding_valid_eligible_reports']}) | "
                f"{_format_metric(metrics['auxiliary_fields_empty_rate'])} "
                f"({metrics['auxiliary_fields_empty_evaluated_reports']}/{metrics['auxiliary_fields_empty_eligible_reports']}) | "
                f"{_format_metric(metrics['packet_scaffolding_free_rate'])} "
                f"({metrics['packet_scaffolding_free_evaluated_reports']}/{metrics['packet_scaffolding_free_eligible_reports']}) |"
            )

    lines += [
        "",
        "## Axis 3: content and grounded-content fidelity",
        "",
        "Content and grounded-content P/R/F1 are micro-averaged from their separate v4 "
        "TP/FP/FN triplets over clean+bad_history only; boundary states never enter these "
        "denominators. They are not substituted with legacy strict-field scores.",
        "",
        "| Model | Stratum | Content reports | Content P | Content R | Content F1 | Content complete | Grounded reports | Grounded P | Grounded R | Grounded F1 | Grounded complete |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for stratum, metrics in [
            ("overall", summary["three_axis"]),
            *summary["three_axis_by_modality"].items(),
        ]:
            lines.append(
                f"| {summary['model']} | {stratum} | "
                f"{metrics['content_evaluated_reports']}/{metrics['content_eligible_reports']} | "
                f"{_format_metric(metrics['content_micro_precision'])} | "
                f"{_format_metric(metrics['content_micro_recall'])} | "
                f"{_format_metric(metrics['content_micro_f1'])} | "
                f"{_format_metric(metrics['content_report_complete_rate'])} | "
                f"{metrics['grounded_content_evaluated_reports']}/{metrics['grounded_content_eligible_reports']} | "
                f"{_format_metric(metrics['grounded_content_micro_precision'])} | "
                f"{_format_metric(metrics['grounded_content_micro_recall'])} | "
                f"{_format_metric(metrics['grounded_content_micro_f1'])} | "
                f"{_format_metric(metrics['grounded_content_report_complete_rate'])} |"
            )

    lines += [
        "",
        "## Boundary-state diagnostics",
        "",
        "| Model | Stratum | Variant | Task success | Recovery valid | Semantics valid | Structure valid | Sentinel-free | Unsafe conclusion |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for stratum, primary in [
            ("overall", summary["primary"]),
            *summary["primary_by_modality"].items(),
        ]:
            for variant in BOUNDARY_VARIANTS:
                boundary = primary["boundary_metrics"][variant]
                lines.append(
                    f"| {summary['model']} | {stratum} | {variant} | "
                    f"{_format_metric(boundary['task_success_rate'])} | "
                    f"{_format_metric(boundary['boundary_recovery_valid_rate'])} | "
                    f"{_format_metric(boundary['boundary_semantics_valid_rate'])} | "
                    f"{_format_metric(boundary['boundary_structure_valid_rate'])} | "
                    f"{_format_metric(boundary['controlled_sentinel_free_rate'])} | "
                    f"{_format_metric(boundary['unsafe_clinical_conclusion_rate'])} |"
                )

    lines += [
        "",
        "## Secondary localized-field results (clean + bad_history only)",
        "",
        "Outage and conflict rows are excluded from field P/R/F1 because their success target is "
        "boundary recovery rather than literal field extraction.",
        "",
        "| Model | Stratum | Scope | Reports | Eligible | Field P | Field R | Field F1 | Report exact | Schema valid | Source valid | Unsupported |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        strata = [
            ("overall", summary["secondary_field_metrics"]),
            *summary["secondary_field_by_modality"].items(),
            *[
                (
                    f"variant:{variant}",
                    {**metrics, "scope": variant},
                )
                for variant, metrics in summary["secondary_field_by_variant"].items()
            ],
        ]
        for stratum, metrics in strata:
            lines.append(
                f"| {summary['model']} | {stratum} | {metrics['scope']} | "
                f"{metrics['reports']} | {metrics['eligible_reports']}/{metrics['reports']} | "
                f"{_format_metric(metrics['field_micro_precision'])} | "
                f"{_format_metric(metrics['field_micro_recall'])} | "
                f"{_format_metric(metrics['field_micro_f1'])} | "
                f"{_format_metric(metrics['report_exact_rate'])} | "
                f"{_format_metric(metrics['schema_valid_rate'])} | "
                f"{_format_metric(metrics['source_valid_rate'])} | "
                f"{_format_metric(metrics['unsupported_field_rate'])} |"
            )
    if comparisons:
        lines += [
            "",
            "## Paired family-bootstrap comparisons",
            "",
            "Every draw resamples a complete family, retaining all four variants together.",
            "",
            "| Candidate vs reference | Stratum | Metric family | Metric | Delta | 95% CI | Families |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for row in comparisons:
            delta = row["delta_candidate_minus_reference"]
            lower = row["ci95_lower"]
            upper = row["ci95_upper"]
            delta_text = "N/A" if delta is None else f"{100.0 * delta:+.2f} pp"
            ci_text = (
                "N/A"
                if lower is None or upper is None
                else f"[{100.0 * lower:+.2f}, {100.0 * upper:+.2f}] pp"
            )
            lines.append(
                f"| {row['candidate_model']} vs {row['reference_model']} | {row['stratum']} | "
                f"{row.get('metric_family', 'legacy')} | {row['metric']} | {delta_text} | "
                f"{ci_text} | {row.get('paired_families', row['paired_reports'])} |"
            )
    lines += [
        "",
        "## Claim boundary",
        "",
        f"This is a {claim_boundary}. It does not establish patient-level generalization, "
        "ultrasound–pathology linkage, or treatment utility. Field metrics use only evaluation-eligible "
        "gold fields. Outage/conflict task success and boundary diagnostics use their explicit "
        "variant-aware verifier contracts.",
        "",
        "No report text, prompts, responses, field values, case identifiers, or absolute private paths "
        "are included in this public-safe report.",
        "",
    ]
    return "\n".join(lines)


def write_public_outputs(
    *,
    public_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    public_manifest: Mapping[str, Any],
) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    evaluation_track = str(public_manifest.get("evaluation_track") or "")
    if not evaluation_track and summaries:
        evaluation_track = str(
            summaries[0].get("three_axis", {}).get("evaluation_track") or "mixed"
        )
    _, claim_boundary = _track_presentation(evaluation_track)
    payload = {
        "schema_version": SCHEMA_VERSION + ".public_summary",
        "privacy": {
            "contains_raw_prompts": False,
            "contains_raw_responses": False,
            "contains_gold_values": False,
            "contains_case_identifiers": False,
            "contains_absolute_private_paths": False,
            "contains_private_input_hashes": False,
        },
        "manifest": dict(public_manifest),
        "models": list(summaries),
        "paired_family_bootstrap": list(comparisons),
        # Compatibility key retained for readers of the v1 public artifact.
        "paired_report_bootstrap": list(comparisons),
        "claim_boundary": claim_boundary,
    }
    _atomic_json(public_dir / "summary.json", payload)

    model_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    three_axis_rows: list[dict[str, Any]] = []
    for summary in summaries:
        primary_strata = [
            ("overall", summary["primary"]),
            *summary["primary_by_modality"].items(),
        ]
        for stratum, metrics in primary_strata:
            row: dict[str, Any] = {
                "model": summary["model"],
                "stratum": stratum,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"by_variant", "boundary_metrics"}
                },
            }
            for variant in FOUR_STATE_VARIANTS:
                variant_metrics = metrics["by_variant"][variant]
                row[f"{variant}_reports"] = variant_metrics["reports"]
                row[f"{variant}_task_success_rate"] = variant_metrics[
                    "task_success_rate"
                ]
            for variant in BOUNDARY_VARIANTS:
                boundary = metrics["boundary_metrics"][variant]
                for key in BOUNDARY_ASSESSMENT_FIELDS:
                    row[f"{variant}_{key}_rate"] = boundary[f"{key}_rate"]
                    row[f"{variant}_{key}_evaluated_reports"] = boundary[
                        f"{key}_evaluated_reports"
                    ]
            model_rows.append(row)
        for stratum, metrics in [
            ("overall", summary["three_axis"]),
            *summary["three_axis_by_modality"].items(),
        ]:
            three_axis_rows.append(
                {"model": summary["model"], "stratum": stratum, **metrics}
            )
        for stratum, metrics in [
            ("overall", summary["secondary_field_metrics"]),
            *summary["secondary_field_by_modality"].items(),
        ]:
            secondary_rows.append(
                {"model": summary["model"], "stratum": stratum, **metrics}
            )
        for variant, metrics in summary["secondary_field_by_variant"].items():
            secondary_rows.append(
                {
                    "model": summary["model"],
                    "stratum": f"variant:{variant}",
                    "scope": variant,
                    "included_variants": [variant],
                    **metrics,
                }
            )
    metric_fields = [
        "model",
        "stratum",
        "family_mode",
        "families",
        "complete_four_state_families",
        "reports",
        "clean_reports",
        "perturbed_reports",
        "clean_task_successes",
        "perturbed_task_successes",
        "clean_task_success_rate",
        "perturbed_task_success_rate",
        "descriptive_pooled_perturbed_task_success_rate",
        "perturbed_task_success_rate_is_descriptive",
        "bad_history_task_success_rate",
        "source_outage_task_success_rate",
        "evidence_conflict_task_success_rate",
        "conditional_retention",
        "conditional_retention_numerator",
        "conditional_retention_denominator",
        "regression_rate",
        "bad_history_conditional_retention",
        "bad_history_conditional_retention_numerator",
        "bad_history_conditional_retention_denominator",
        "bad_history_regression_rate",
        "four_state_joint_successes",
        "four_state_joint_tsr",
        *[
            f"{variant}_{key}_rate"
            for variant in BOUNDARY_VARIANTS
            for key in BOUNDARY_ASSESSMENT_FIELDS
        ],
        *[
            f"{variant}_{key}_evaluated_reports"
            for variant in BOUNDARY_VARIANTS
            for key in BOUNDARY_ASSESSMENT_FIELDS
        ],
        "generation_error_count",
        "mean_latency_seconds",
        "mean_output_tokens",
    ]
    _atomic_text(public_dir / "model_metrics.csv", _csv_text(metric_fields, model_rows))
    three_axis_fields = [
        "model",
        "stratum",
        "reports",
        "state_scope",
        "state_task_success_evaluated_reports",
        "evaluation_track",
        "challenge_mode_reports",
        "challenge_mode_rate",
        "state_task_success_rate",
        *[
            field
            for key in PROTOCOL_ASSESSMENT_FIELDS
            for field in (
                f"{key}_scope",
                f"{key}_eligible_reports",
                f"{key}_rate",
                f"{key}_evaluated_reports",
            )
        ],
        *[
            field
            for axis in CONTENT_AXES
            for field in (
                f"{axis}_scope",
                f"{axis}_eligible_reports",
                f"{axis}_evaluated_reports",
                f"{axis}_tp",
                f"{axis}_fp",
                f"{axis}_fn",
                f"{axis}_micro_precision",
                f"{axis}_precision_denominator",
                f"{axis}_micro_recall",
                f"{axis}_recall_denominator",
                f"{axis}_micro_f1",
                f"{axis}_f1_denominator",
                f"{axis}_report_complete_rate",
                f"{axis}_report_complete_evaluated_reports",
            )
        ],
    ]
    _atomic_text(
        public_dir / "three_axis_metrics.csv",
        _csv_text(three_axis_fields, three_axis_rows),
    )
    secondary_fields = [
        "model",
        "stratum",
        "scope",
        "included_variants",
        "reports",
        "eligible_reports",
        "eligible_report_coverage",
        "eligible_fields",
        "field_tp",
        "field_fp",
        "field_fn",
        "field_micro_precision",
        "field_micro_recall",
        "field_micro_f1",
        "report_exact_rate",
        "schema_valid_rate",
        "source_valid_rate",
        "unsupported_fields",
        "emitted_fields",
        "unsupported_field_rate",
        "generation_error_count",
        "mean_latency_seconds",
        "mean_output_tokens",
    ]
    _atomic_text(
        public_dir / "secondary_field_metrics.csv",
        _csv_text(secondary_fields, secondary_rows),
    )
    comparison_fields = [
        "reference_model",
        "candidate_model",
        "stratum",
        "metric",
        "metric_family",
        "bootstrap_unit",
        "paired_families",
        "paired_reports",
        "reference_value",
        "candidate_value",
        "delta_candidate_minus_reference",
        "ci95_lower",
        "ci95_upper",
        "bootstrap_replicates",
        "valid_bootstrap_replicates",
        "bootstrap_seed",
        "direction",
    ]
    _atomic_text(
        public_dir / "paired_comparisons.csv",
        _csv_text(comparison_fields, comparisons),
    )
    _atomic_text(public_dir / "summary.md", _render_markdown(summaries, comparisons))
    _atomic_json(public_dir / "run_manifest.json", public_manifest)


def _prepare_output_contract(
    output_dir: Path,
    *,
    private_manifest: Mapping[str, Any],
    resume: bool,
    overwrite: bool,
    labels: Sequence[str],
) -> tuple[Path, Path]:
    private_dir = output_dir / "private"
    public_dir = output_dir / "public"
    manifest_path = private_dir / "run_manifest.private.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume:
            if existing.get("run_fingerprint") != private_manifest.get("run_fingerprint"):
                raise EvaluationContractError(
                    "resume manifest mismatch: model/input/verifier/generation contract changed"
                )
        elif not overwrite:
            raise FileExistsError(
                f"{manifest_path} exists; pass --resume for the identical run or --overwrite"
            )
    elif resume and any((private_dir / f"{label}.cases.private.jsonl").exists() for label in labels):
        raise EvaluationContractError("private case outputs exist without a binding run manifest")
    elif not overwrite and any(
        (private_dir / f"{label}.cases.private.jsonl").exists() for label in labels
    ):
        raise FileExistsError(
            "private case outputs already exist without the requested --overwrite"
        )

    private_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(private_dir / ".gitignore", PRIVATE_GITIGNORE)
    if overwrite:
        for label in labels:
            cases_path = private_dir / f"{label}.cases.private.jsonl"
            if cases_path.exists():
                _atomic_text(cases_path, "")
    _atomic_json(manifest_path, private_manifest)
    return private_dir, public_dir


def run_evaluation(
    *,
    seeds_path: Path,
    model_specs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    field_gold_path: Path | None = None,
    qzbc_root: Path | None = None,
    device: str = "auto",
    max_input_tokens: int = 32768,
    max_new_tokens: int = 2048,
    generation_seed: int = 42,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    max_examples: int = -1,
    reference_label: str | None = None,
    additional_reference_labels: Sequence[str] = (),
    resume: bool = False,
    overwrite: bool = False,
    acknowledge_private_clinical_data: bool = False,
) -> dict[str, Any]:
    if not acknowledge_private_clinical_data:
        raise EvaluationContractError(
            "private clinical evaluation requires --acknowledge-private-clinical-data"
        )
    if resume and overwrite:
        raise EvaluationContractError("--resume and --overwrite are mutually exclusive")
    if max_input_tokens < 1 or max_new_tokens < 1:
        raise EvaluationContractError("token limits must be positive")
    seeds, input_contract = load_locked_inputs(
        seeds_path,
        field_gold_path=field_gold_path,
        qzbc_root=qzbc_root,
        max_examples=max_examples,
    )
    model_contracts: list[dict[str, Any]] = []
    for spec in model_specs:
        print(f"[fingerprint] model={spec['label']}", flush=True)
        model_contracts.append(
            {"label": str(spec["label"]), **fingerprint_local_model(Path(spec["path"]))}
        )
    labels = [contract["label"] for contract in model_contracts]
    reference_labels = _resolve_reference_labels(
        reference_label, additional_reference_labels, labels
    )
    reference_label = reference_labels[0]

    device_contract = _resolve_device_contract(device)
    generation_contract = {
        "engine": "transformers",
        "batch_size": 1,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "enable_thinking": False,
        "cuda_dtype": "bfloat16",
        "cpu_dtype": "float32",
        **device_contract,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "seed": generation_seed,
    }
    stable_contract = {
        "schema_version": SCHEMA_VERSION + ".run_contract",
        "input": input_contract,
        "evaluation_track": input_contract["evaluation_track"],
        "challenge_mode": input_contract["challenge_mode"],
        "models": [
            {
                "label": contract["label"],
                "canonical_path_private": contract["canonical_path_private"],
                "fingerprint": contract["fingerprint"],
                "file_count": contract["file_count"],
                "bytes": contract["bytes"],
            }
            for contract in model_contracts
        ],
        "generation": generation_contract,
        "verifier_sha256": _verifier_file_hash(),
        "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
        "care_opd_qzbc_contract_sha256": _sha256_file(
            Path(__file__).resolve().parents[2] / "src" / "care_opd" / "qzbc_contract.py"
        ),
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed},
        "reference_model": reference_label,
        "reference_models": reference_labels,
    }
    run_fingerprint = _sha256_bytes(_canonical_json(stable_contract))
    private_manifest = {
        **stable_contract,
        "run_fingerprint": run_fingerprint,
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "privacy": {
            "private_case_outputs_contain_raw_responses": True,
            "public_outputs_contain_raw_clinical_data": False,
        },
        "model_file_inventory_private": {
            contract["label"]: contract["files_private"] for contract in model_contracts
        },
        "private_input_paths": {
            "seeds": str(seeds_path.expanduser().resolve()),
            "field_gold": (
                str(field_gold_path.expanduser().resolve()) if field_gold_path is not None else None
            ),
            "qzbc_root": str(qzbc_root.expanduser().resolve()) if qzbc_root is not None else None,
        },
    }
    private_dir, public_dir = _prepare_output_contract(
        output_dir.expanduser().resolve(),
        private_manifest=private_manifest,
        resume=resume,
        overwrite=overwrite,
        labels=labels,
    )

    results: dict[str, list[dict[str, Any]]] = {}
    for contract in model_contracts:
        label = contract["label"]
        per_model_fingerprint = _sha256_bytes(
            _canonical_json({"run": run_fingerprint, "model": label})
        )
        results[label] = evaluate_model(
            label=label,
            model_path=Path(contract["canonical_path_private"]),
            seeds=seeds,
            private_cases_path=private_dir / f"{label}.cases.private.jsonl",
            run_fingerprint=per_model_fingerprint,
            device=device_contract["resolved_device"],
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            generation_seed=generation_seed,
            resume=resume,
        )

    summaries = [summarize_model(label, results[label]) for label in labels]
    modalities = sorted({str(seed["_eval_modality"]) for seed in seeds})
    comparisons = build_paired_family_comparisons(
        results,
        reference_labels=reference_labels,
        modalities=modalities,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    public_manifest = {
        "schema_version": SCHEMA_VERSION + ".public_manifest",
        "qzbc_contract_fingerprint": input_contract["qzbc_contract_fingerprint"],
        "case_count": input_contract["case_count"],
        "family_mode": input_contract["family_mode"],
        "family_count": input_contract["family_count"],
        "variants": input_contract["variants"],
        "evaluation_track": input_contract["evaluation_track"],
        "challenge_mode": input_contract["challenge_mode"],
        "bootstrap_unit": "family_id",
        "models": [
            {
                "label": contract["label"],
                "fingerprint": contract["fingerprint"],
                "file_count": contract["file_count"],
                "bytes": contract["bytes"],
            }
            for contract in model_contracts
        ],
        "generation": dict(generation_contract),
        "verifier_sha256": stable_contract["verifier_sha256"],
        "evaluator_sha256": stable_contract["evaluator_sha256"],
        "care_opd_qzbc_contract_sha256": stable_contract[
            "care_opd_qzbc_contract_sha256"
        ],
        "bootstrap": stable_contract["bootstrap"],
        "reference_model": reference_label,
        "reference_models": reference_labels,
        "private_binding_disclosure": (
            "Private input and run fingerprints are retained only in "
            "private/run_manifest.private.json and are intentionally absent here."
        ),
    }
    public_manifest["public_protocol_fingerprint"] = _sha256_bytes(
        _canonical_json(public_manifest)
    )
    write_public_outputs(
        public_dir=public_dir,
        summaries=summaries,
        comparisons=comparisons,
        public_manifest=public_manifest,
    )
    return {
        "run_fingerprint": run_fingerprint,
        "private_output": str(private_dir),
        "public_output": str(public_dir),
        "models": summaries,
        "paired_comparisons": comparisons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--field-gold", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable LABEL=LOCAL_HF_CHECKPOINT (or a path whose basename is the label).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qzbc-root", type=Path)
    parser.add_argument("--reference-model", help="Model label used as the paired baseline.")
    parser.add_argument(
        "--additional-reference-model",
        action="append",
        default=[],
        help=(
            "Repeatable additional paired-family baseline. It reuses the same model "
            "generations; for example, add gkd_full while keeping clean_sft primary."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=-1,
        help=(
            "Smoke-test limit: number of whole families for four-state inputs, "
            "or number of rows for legacy clean-only inputs (-1 means all)."
        ),
    )
    parser.add_argument(
        "--acknowledge-private-clinical-data",
        action="store_true",
        help="Confirm this run is authorized to process the locked private reports locally.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run_evaluation(
        seeds_path=args.seeds,
        field_gold_path=args.field_gold,
        model_specs=parse_model_specs(args.model),
        output_dir=args.output_dir,
        qzbc_root=args.qzbc_root,
        device=args.device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        generation_seed=args.generation_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        max_examples=args.max_examples,
        reference_label=args.reference_model,
        additional_reference_labels=args.additional_reference_model,
        resume=args.resume,
        overwrite=args.overwrite,
        acknowledge_private_clinical_data=args.acknowledge_private_clinical_data,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "run_fingerprint": result["run_fingerprint"],
                "private_output": result["private_output"],
                "public_output": result["public_output"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
