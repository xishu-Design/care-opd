#!/usr/bin/env python3
"""Fail-closed contract for the exact token-matched localization control."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

MATRIX_SCHEMA = "care_opd.causal_ablation_matrix.v1"
CONTRACT_SCHEMA = "care_opd.token_matched_control_contract.v1"
AUDIT_SCHEMA = "care_opd.token_matched_mask_audit.v1"
DATA_VIEW = "full_selected.jsonl"
MAX_LENGTH = 4096
MAX_RESPONSE_TOKENS = 1200
ADD_EOS = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_PATHS = {
    "collator": PROJECT_ROOT / "src/care_opd/collator.py",
    "token_audit": Path(__file__).resolve().parent / "audit_token_matching.py",
}

COMMON = {
    "baseline_mode": "care_opd",
    "random_mask_seed": 20260815,
    "use_risk_weighting": True,
    "oracle_risk_weighting_mode": "disabled",
    "use_teacher_trust": False,
    "retain_outside_repair": False,
    "distillation_coefficient": 1.0,
    "retention_coefficient": 0.0,
    "hard_oracle_ce_weight": 0.0,
    "max_length": MAX_LENGTH,
    "max_response_tokens": MAX_RESPONSE_TOKENS,
    "add_eos": ADD_EOS,
}

EXPECTED: dict[str, dict[str, Any]] = {
    "localized_kd_only": {**COMMON, "mask_mode": "span"},
    "random_token_kd_only": {
        **COMMON,
        "mask_mode": "random_token_matched",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_matrix(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError(f"{path}: unexpected matrix schema")
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError(f"{path}: variants must be a list")
    names = [item.get("name") for item in variants if isinstance(item, Mapping)]
    if names != list(EXPECTED):
        raise ValueError(f"{path}: expected arm order {list(EXPECTED)!r}")
    for variant in variants:
        name = str(variant["name"])
        if variant.get("data_view") != DATA_VIEW:
            raise ValueError(f"{name}: data_view must be {DATA_VIEW}")
        overrides = variant.get("overrides")
        if not isinstance(overrides, Mapping) or dict(overrides) != EXPECTED[name]:
            raise ValueError(
                f"{name}: overrides differ from the frozen token-matched contract"
            )
    left = dict(EXPECTED["localized_kd_only"])
    right = dict(EXPECTED["random_token_kd_only"])
    if left.pop("mask_mode") != "span" or right.pop("mask_mode") != "random_token_matched":
        raise AssertionError("invalid frozen mask modes")
    if left != right:
        raise AssertionError("formal localization arms differ by more than mask location")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": "PASS",
        "matrix": {"path": str(path), "sha256": _sha256(path)},
        "arm_order": names,
        "data_view": DATA_VIEW,
        "isolated_change": "verifier-selected token positions vs deterministic random positions",
        "trust_retention_and_oracle_disabled": True,
        "claim_boundary": (
            "The matrix isolates semantic token location for one matched training round. "
            "The tokenizer audit is additionally required to establish exact effective-token matching."
        ),
    }


def _same_model_reference(actual: Any, expected: str) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    actual_path = Path(actual)
    expected_path = Path(expected)
    if actual_path.exists() or expected_path.exists():
        return actual_path.resolve() == expected_path.resolve()
    return actual == expected


def _local_tokenizer_fingerprint(reference: str) -> str | None:
    """Recompute a local checkpoint fingerprint; never fetch remote IDs."""

    if not Path(reference).exists():
        return None
    try:
        from transformers import AutoTokenizer
        from care_opd.models import tokenizer_fingerprint
    except ImportError as exc:
        raise RuntimeError(
            "local tokenizer binding requires the training transformers environment"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=True)
    return tokenizer_fingerprint(tokenizer)


def audit_token_report(
    path: Path,
    *,
    data_path: Path,
    tokenizer_reference: str,
) -> dict[str, Any]:
    path = path.resolve()
    data_path = data_path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != AUDIT_SCHEMA or payload.get("status") != "COMPLETE":
        raise ValueError(f"{path}: token audit is not COMPLETE {AUDIT_SCHEMA}")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{path}: source binding is missing")
    if Path(str(source.get("path"))).resolve() != data_path:
        raise ValueError(f"{path}: token audit was run on a different data path")
    if source.get("sha256") != _sha256(data_path):
        raise ValueError(f"{path}: token audit data SHA-256 is stale")
    if not _same_model_reference(payload.get("tokenizer"), tokenizer_reference):
        raise ValueError(f"{path}: token audit used a different tokenizer reference")
    tokenizer_sha = payload.get("tokenizer_sha256")
    if not isinstance(tokenizer_sha, str) or len(tokenizer_sha) != 64:
        raise ValueError(f"{path}: tokenizer fingerprint is missing or invalid")
    local_tokenizer_sha = _local_tokenizer_fingerprint(tokenizer_reference)
    if local_tokenizer_sha is not None and local_tokenizer_sha != tokenizer_sha:
        raise ValueError(f"{path}: local tokenizer fingerprint is stale")
    if payload.get("max_length") != MAX_LENGTH:
        raise ValueError(f"{path}: token audit max_length does not match training")
    if payload.get("max_response_tokens") != MAX_RESPONSE_TOKENS:
        raise ValueError(
            f"{path}: token audit max_response_tokens does not match training"
        )
    if payload.get("add_eos") is not ADD_EOS:
        raise ValueError(f"{path}: token audit add_eos does not match training")
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError(f"{path}: token audit implementation binding is missing")
    for label, implementation_path in IMPLEMENTATION_PATHS.items():
        item = implementation.get(label)
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: token audit lacks implementation.{label}")
        if item.get("sha256") != _sha256(implementation_path):
            raise ValueError(f"{path}: token audit implementation.{label} is stale")
    if payload.get("random_mask_seed") != COMMON["random_mask_seed"]:
        raise ValueError(f"{path}: unexpected random mask seed")
    overall = payload.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError(f"{path}: overall metrics are missing")
    if source.get("rows") != overall.get("rows") or int(overall.get("rows", 0)) < 1:
        raise ValueError(f"{path}: source/overall row count is missing or inconsistent")
    required_equal = (
        overall.get("exact_count_match_rate") == 1.0
        and overall.get("exact_weight_multiset_match_rate") == 1.0
        and overall.get("active_semantic_tokens") == overall.get("active_random_tokens")
        and overall.get("zero_width_selected_tokens") == 0
        and int(overall.get("selected_tokens", 0)) > 0
        and int(overall.get("active_semantic_tokens", 0)) > 0
        and math.isclose(
            float(overall.get("semantic_weight_sum", float("nan"))),
            float(overall.get("random_weight_sum", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    if not required_equal:
        raise ValueError(f"{path}: exact effective-token contract did not pass")
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": _sha256(path),
        "data_sha256": source["sha256"],
        "tokenizer_reference": payload["tokenizer"],
        "tokenizer_sha256": tokenizer_sha,
        "local_tokenizer_reverified": local_tokenizer_sha is not None,
        "rows": overall.get("rows"),
        "exact_count_match_rate": 1.0,
        "exact_weight_multiset_match_rate": 1.0,
        "active_tokens_per_arm": overall.get("active_semantic_tokens"),
        "response_truncation_rows": overall.get("response_truncation_rows"),
        "implementation": dict(implementation),
    }


__all__ = ["CONTRACT_SCHEMA", "EXPECTED", "audit_matrix", "audit_token_report"]
