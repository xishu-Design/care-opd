#!/usr/bin/env python3
"""Audit completed training artifacts against the frozen attribution contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .contract import CONTRACT_SCHEMA, EXPECTED
except ImportError:
    from contract import CONTRACT_SCHEMA, EXPECTED

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_TRAINING_CODE = (
    "src/care_opd/collator.py",
    "src/care_opd/train_opd.py",
    "src/care_opd/losses.py",
    "src/care_opd/trainer_utils.py",
    "mechanism_experiments/causal_ablation/run_single_round.py",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    )
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no metrics")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return actual == expected


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    plan_path = root / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    contract = plan.get("scientific_contract")
    if not isinstance(contract, Mapping) or contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("experiment plan lacks the frozen hard-oracle scientific contract")
    if contract.get("status") != "PASS":
        raise ValueError("scientific contract did not pass")
    names = [arm.get("name") for arm in plan.get("arms", [])]
    if names != list(EXPECTED):
        raise ValueError(f"unexpected plan arms/order: {names!r}")
    expected_steps = int(plan["budget_contract"]["max_steps_per_arm"])
    training_code = plan.get("training_code")
    if not isinstance(training_code, Mapping):
        raise ValueError("experiment plan lacks training_code fingerprints")
    for relative in REQUIRED_TRAINING_CODE:
        recorded = training_code.get(relative)
        if not isinstance(recorded, Mapping):
            raise ValueError(f"training_code lacks {relative}")
        recorded_sha = recorded.get("sha256")
        if not isinstance(recorded_sha, str) or len(recorded_sha) != 64:
            raise ValueError(f"training_code has invalid SHA-256 for {relative}")
        current = PROJECT_ROOT / relative
        if _sha256(current) != recorded_sha:
            raise ValueError(
                f"training code drift after plan freeze: {relative}; "
                "audit with the exact archived code revision"
            )

    arm_reports: dict[str, Any] = {}
    for arm in plan["arms"]:
        name = str(arm["name"])
        arm_root = Path(arm["output_dir"])
        final = arm_root / "final"
        metadata_path = final / "run_metadata.json"
        metrics_path = arm_root / "metrics.jsonl"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("trainer") != "opd":
            raise ValueError(f"{name}: final metadata trainer is not opd")
        if int(metadata.get("global_step", -1)) != expected_steps:
            raise ValueError(
                f"{name}: expected global_step={expected_steps}, "
                f"got {metadata.get('global_step')!r}"
            )
        opd = metadata.get("opd")
        if not isinstance(opd, Mapping):
            raise ValueError(f"{name}: final metadata lacks opd switches")
        for key, expected_value in EXPECTED[name].items():
            if not _same(opd.get(key), expected_value):
                raise ValueError(
                    f"{name}: effective {key} expected {expected_value!r}, got {opd.get(key)!r}"
                )

        weights = [
            candidate
            for candidate in (
                final / "adapter_model.safetensors",
                final / "adapter_model.bin",
                final / "model.safetensors",
            )
            if candidate.is_file() and candidate.stat().st_size > 0
        ]
        if not weights:
            raise FileNotFoundError(f"{name}: no non-empty final model/adapter weight file")

        metrics = _read_jsonl(metrics_path)
        starts = [row for row in metrics if row.get("event") == "start"]
        if len(starts) != 1:
            raise ValueError(f"{name}: expected exactly one start metric event")
        if starts[0].get("method") != "care_opd":
            raise ValueError(f"{name}: unexpected core method label")
        switches = starts[0].get("effective_switches")
        if not isinstance(switches, Mapping):
            raise ValueError(f"{name}: start metric lacks effective_switches")
        logged_switches = {
            "mask_mode": "mask_mode",
            "random_mask_seed": "random_mask_seed",
            "risk_weighting": "use_risk_weighting",
            "oracle_risk_weighting_mode": "oracle_risk_weighting_mode",
            "teacher_trust": "use_teacher_trust",
            "retain_outside_repair": "retain_outside_repair",
            "distillation_coefficient": "distillation_coefficient",
            "retention_coefficient": "retention_coefficient",
            "hard_oracle_ce_weight": "hard_oracle_ce_weight",
        }
        for logged_key, contract_key in logged_switches.items():
            expected_value = EXPECTED[name][contract_key]
            if not _same(switches.get(logged_key), expected_value):
                raise ValueError(
                    f"{name}: logged effective switch {logged_key} expected "
                    f"{expected_value!r}, got {switches.get(logged_key)!r}"
                )
        complete = [row for row in metrics if row.get("event") == "complete"]
        if not complete or int(complete[-1].get("global_step", -1)) != expected_steps:
            raise ValueError(f"{name}: metrics do not contain a matching complete sentinel")
        train_rows = [row for row in metrics if row.get("event") == "train"]
        if not train_rows:
            raise ValueError(f"{name}: no train metric rows")
        oracle_weight = float(EXPECTED[name]["hard_oracle_ce_weight"])
        positive_oracle_token_events = sum(
            int(row.get("hard_oracle_tokens_local", 0) or 0) > 0 for row in train_rows
        )
        positive_oracle_loss_events = sum(
            float(row.get("hard_oracle_ce", 0.0) or 0.0) > 0.0 for row in train_rows
        )
        if oracle_weight > 0 and (
            positive_oracle_token_events == 0 or positive_oracle_loss_events == 0
        ):
            raise ValueError(f"{name}: configured oracle CE never became active")
        if oracle_weight == 0 and any(
            abs(float(row.get("hard_oracle_ce", 0.0) or 0.0)) > 1e-12
            for row in train_rows
        ):
            raise ValueError(f"{name}: nonzero hard-oracle loss with zero configured weight")

        arm_reports[name] = {
            "status": "PASS",
            "output_dir": str(arm_root),
            "global_step": expected_steps,
            "weight_file": str(weights[0]),
            "weight_bytes": weights[0].stat().st_size,
            "train_metric_events": len(train_rows),
            "positive_oracle_token_events": positive_oracle_token_events,
            "positive_oracle_loss_events": positive_oracle_loss_events,
            "effective_switches": {key: opd[key] for key in EXPECTED[name]},
        }

    return {
        "schema_version": "care_opd.hard_oracle_attribution_completion_audit.v1",
        "status": "PASS",
        "experiment_root": str(root),
        "plan": str(plan_path),
        "max_steps_per_arm": expected_steps,
        "training_code": dict(training_code),
        "arms": arm_reports,
        "claim_boundary": "This proves matched training completion, not evaluation completion or causal effect size.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit(args.experiment_root)
    output = args.output or args.experiment_root / "completion_audit.json"
    _atomic_json(output.resolve(), result)
    print("HARD_ORACLE_TRAINING_AUDIT=PASS")
    print(f"audit={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
