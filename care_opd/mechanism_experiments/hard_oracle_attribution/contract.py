#!/usr/bin/env python3
"""Fail-closed scientific contract for the hard-oracle attribution matrix."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

MATRIX_SCHEMA = "care_opd.causal_ablation_matrix.v1"
CONTRACT_SCHEMA = "care_opd.hard_oracle_attribution_contract.v1"
DATA_VIEW = "full_selected.jsonl"
ORACLE_WEIGHT = 0.25

EXPECTED: dict[str, dict[str, Any]] = {
    "care_with_oracle_ce": {
        "baseline_mode": "care_opd",
        "mask_mode": "span",
        "random_mask_seed": 20260815,
        "use_risk_weighting": True,
        "oracle_risk_weighting_mode": "enabled",
        "use_teacher_trust": True,
        "retain_outside_repair": True,
        "distillation_coefficient": 1.0,
        "retention_coefficient": 0.2,
        "hard_oracle_ce_weight": ORACLE_WEIGHT,
    },
    "care_without_oracle_ce": {
        "baseline_mode": "care_opd",
        "mask_mode": "span",
        "random_mask_seed": 20260815,
        "use_risk_weighting": True,
        "oracle_risk_weighting_mode": "enabled",
        "use_teacher_trust": True,
        "retain_outside_repair": True,
        "distillation_coefficient": 1.0,
        "retention_coefficient": 0.2,
        "hard_oracle_ce_weight": 0.0,
    },
    "gkd_without_oracle_ce": {
        "baseline_mode": "care_opd",
        "mask_mode": "full",
        "random_mask_seed": 20260815,
        "use_risk_weighting": False,
        "oracle_risk_weighting_mode": "enabled",
        "use_teacher_trust": False,
        "retain_outside_repair": False,
        "distillation_coefficient": 1.0,
        "retention_coefficient": 0.0,
        "hard_oracle_ce_weight": 0.0,
    },
    "gkd_plus_same_oracle_ce": {
        "baseline_mode": "care_opd",
        "mask_mode": "full",
        "random_mask_seed": 20260815,
        "use_risk_weighting": False,
        "oracle_risk_weighting_mode": "enabled",
        "use_teacher_trust": False,
        "retain_outside_repair": False,
        "distillation_coefficient": 1.0,
        "retention_coefficient": 0.0,
        "hard_oracle_ce_weight": ORACLE_WEIGHT,
    },
    "oracle_ce_only_on_policy": {
        "baseline_mode": "care_opd",
        "mask_mode": "span",
        "random_mask_seed": 20260815,
        "use_risk_weighting": False,
        "oracle_risk_weighting_mode": "enabled",
        "use_teacher_trust": False,
        "retain_outside_repair": False,
        "distillation_coefficient": 0.0,
        "retention_coefficient": 0.0,
        "hard_oracle_ce_weight": ORACLE_WEIGHT,
    },
}

CONTRASTS = [
    {
        "name": "oracle_increment_within_care",
        "left": "care_with_oracle_ce",
        "right": "care_without_oracle_ce",
        "isolated_change": "hard_oracle_ce_weight: 0.0 -> 0.25",
    },
    {
        "name": "oracle_increment_within_gkd",
        "left": "gkd_plus_same_oracle_ce",
        "right": "gkd_without_oracle_ce",
        "isolated_change": "hard_oracle_ce_weight: 0.0 -> 0.25",
    },
    {
        "name": "localized_package_without_oracle",
        "left": "care_without_oracle_ce",
        "right": "gkd_without_oracle_ce",
        "isolated_change": "CARE localization/trust/retention package at oracle CE = 0",
    },
    {
        "name": "localized_package_with_matched_oracle",
        "left": "care_with_oracle_ce",
        "right": "gkd_plus_same_oracle_ce",
        "isolated_change": "CARE localization/trust/retention package at matched oracle CE = 0.25",
    },
    {
        "name": "oracle_only_sufficiency",
        "left": "oracle_ce_only_on_policy",
        "right": "gkd_without_oracle_ce",
        "isolated_change": "replace full-response KD with localized hard-oracle CE on the same on-policy bank",
    },
]


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return actual == expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oracle_value(row: Mapping[str, Any]) -> Any:
    value = row.get("oracle_response", row.get("oracle_action"))
    if value is not None:
        return value
    privileged = row.get("privileged_teacher_context")
    if isinstance(privileged, Mapping):
        return privileged.get(
            "verified_oracle_response", privileged.get("verified_oracle_action")
        )
    return None


def audit_training_view(path: Path) -> dict[str, Any]:
    """Check that every failed on-policy row can activate the oracle objective."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            row_id = row.get("id", row.get("request_id"))
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"{path}:{line_no}: non-empty id/request_id is required")
            if row_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate row id {row_id!r}")
            seen.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")

    failures = [row for row in rows if not bool(row.get("clean_control", False))]
    if not failures:
        raise ValueError(f"{path}: no non-clean on-policy failures")
    missing_oracle = [
        str(row.get("id", row.get("request_id")))
        for row in failures
        if _oracle_value(row) is None
    ]
    not_host_verified = [
        str(row.get("id", row.get("request_id")))
        for row in failures
        if not bool(row.get("host_repair_verified", row.get("host_verified", False)))
    ]
    if missing_oracle:
        raise ValueError(
            f"{path}: {len(missing_oracle)} non-clean rows lack a verified oracle; "
            f"examples={missing_oracle[:5]}"
        )
    if not_host_verified:
        raise ValueError(
            f"{path}: {len(not_host_verified)} non-clean rows are not host verified; "
            f"examples={not_host_verified[:5]}"
        )
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": _sha256(path),
        "rows": len(rows),
        "clean_controls": len(rows) - len(failures),
        "on_policy_failure_rows": len(failures),
        "failure_rows_with_oracle": len(failures),
        "failure_rows_host_verified": len(failures),
        "scope": "pre-tokenization audit; post-training metrics must still prove nonzero hard_oracle_tokens",
    }


def audit_matrix(path: Path) -> dict[str, Any]:
    """Validate the exact five-arm contract and return a hash-bound report."""

    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError(f"{path}: expected schema_version={MATRIX_SCHEMA!r}")
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError(f"{path}: variants must be a list")
    names = [item.get("name") for item in variants if isinstance(item, Mapping)]
    if names != list(EXPECTED):
        raise ValueError(
            f"{path}: arm order/names must be exactly {list(EXPECTED)!r}, got {names!r}"
        )
    for variant in variants:
        name = str(variant["name"])
        if variant.get("data_view") != DATA_VIEW:
            raise ValueError(f"{name}: data_view must be {DATA_VIEW!r}")
        overrides = variant.get("overrides")
        if not isinstance(overrides, Mapping):
            raise ValueError(f"{name}: overrides must be an object")
        expected = EXPECTED[name]
        if set(overrides) != set(expected):
            raise ValueError(
                f"{name}: override keys differ; expected={sorted(expected)}, "
                f"actual={sorted(overrides)}"
            )
        for key, expected_value in expected.items():
            if not _equal(overrides.get(key), expected_value):
                raise ValueError(
                    f"{name}.{key}: expected {expected_value!r}, "
                    f"got {overrides.get(key)!r}"
                )

    # These pair checks prevent future edits from invalidating the causal label.
    by_name = {item["name"]: item["overrides"] for item in variants}
    for left, right in (
        ("care_with_oracle_ce", "care_without_oracle_ce"),
        ("gkd_plus_same_oracle_ce", "gkd_without_oracle_ce"),
    ):
        left_values = dict(by_name[left])
        right_values = dict(by_name[right])
        left_weight = left_values.pop("hard_oracle_ce_weight")
        right_weight = right_values.pop("hard_oracle_ce_weight")
        if left_values != right_values or not _equal(left_weight, ORACLE_WEIGHT) or not _equal(right_weight, 0.0):
            raise ValueError(f"{left} vs {right} must differ only in hard_oracle_ce_weight")

    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": "PASS",
        "matrix": {"path": str(path), "sha256": _sha256(path)},
        "arm_order": names,
        "data_view": DATA_VIEW,
        "oracle_ce_weight": ORACLE_WEIGHT,
        "oracle_risk_weighting": "explicitly enabled for every arm; inactive when oracle weight is zero",
        "matched_pairs": [
            ["care_with_oracle_ce", "care_without_oracle_ce"],
            ["gkd_plus_same_oracle_ce", "gkd_without_oracle_ce"],
        ],
        "contrasts": CONTRASTS,
        "claim_boundary": (
            "A one-round, one-training-seed matched intervention isolates the configured "
            "loss contribution for this on-policy bank. It does not quantify training-seed "
            "variance or establish cross-domain generality."
        ),
    }


__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRASTS",
    "EXPECTED",
    "audit_matrix",
    "audit_training_view",
]
