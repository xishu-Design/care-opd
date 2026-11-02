#!/usr/bin/env python3
"""Read-only, fail-closed verification of a completed multi-seed experiment.

This script never launches training/evaluation and never writes an artifact.  It
checks the frozen matrix, every nested training plan and final adapter, the
4200-request rollout contract, every family-clean summary, and the cross-seed
aggregate before printing a PASS sentinel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MATRIX_SCHEMA = "care_opd.causal_ablation_matrix.v1"
TRAIN_SUITE_SCHEMA = "care_opd.multiseed_ablation_plan.v1"
TRAIN_PLAN_SCHEMA = "care_opd.causal_ablation_plan.v1"
EVAL_SUITE_SCHEMA = "care_opd.multiseed_ablation_eval_plan.v1"
EVAL_PLAN_SCHEMA = "care_opd.causal_ablation_eval_plan.v1"
SUMMARY_SCHEMA = "care_opd.causal_ablation_summary.v1"
AGGREGATE_SCHEMA = "care_opd.multiseed_ablation_aggregate.v1"
TOKEN_AUDIT_SCHEMA = "care_opd.token_matched_mask_audit.v1"
TOKEN_CONTROL_ARMS = {"localized_kd_only", "random_token_kd_only"}


class VerificationError(ValueError):
    """Raised when any frozen-result contract is incomplete or inconsistent."""


def _fail(message: str) -> None:
    raise VerificationError(message)


def _sha256(path: Path) -> str:
    if not path.is_file():
        _fail(f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing JSON file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{path}: top-level JSON value must be an object")
    if schema is not None and value.get("schema_version") != schema:
        _fail(
            f"{path}: expected schema_version={schema!r}, "
            f"got {value.get('schema_version')!r}"
        )
    return value


def _require(mapping: Mapping[str, Any], keys: Iterable[str], *, where: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        _fail(f"{where}: missing keys {missing}")


def _binding(binding: Any, *, where: str, schema: str | None = None) -> tuple[Path, dict[str, Any] | None]:
    if not isinstance(binding, dict):
        _fail(f"{where}: binding must be an object")
    _require(binding, ("path", "sha256"), where=where)
    if not isinstance(binding["path"], str) or not isinstance(binding["sha256"], str):
        _fail(f"{where}: path and sha256 must be strings")
    path = Path(binding["path"])
    actual = _sha256(path)
    if actual != binding["sha256"]:
        _fail(f"{where}: SHA-256 mismatch for {path}; expected {binding['sha256']}, got {actual}")
    return path, _load_json(path, schema=schema) if schema else None


def _parse_seeds(raw: str | Sequence[int]) -> list[int]:
    if isinstance(raw, str):
        try:
            values = [int(item.strip()) for item in raw.split(",") if item.strip()]
        except ValueError as exc:
            raise VerificationError("expected comma-separated integer seeds") from exc
    else:
        values = [int(item) for item in raw]
    if len(values) < 2 or len(values) != len(set(values)) or any(item < 0 for item in values):
        _fail("expected at least two distinct, non-negative training seeds")
    return values


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where}: expected numeric value, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{where}: expected finite value, got {value!r}")
    return result


def _read_jsonl_ids(path: Path, *, expected_rows: int, where: str) -> list[str]:
    if not path.is_file():
        _fail(f"{where}: missing JSONL file {path}")
    ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                _fail(f"{path}:{line_number}: row must be an object")
            row_id = row.get("id", row.get("request_id"))
            if not isinstance(row_id, str) or not row_id:
                _fail(f"{path}:{line_number}: non-empty id/request_id is required")
            if row_id in seen:
                _fail(f"{path}:{line_number}: duplicate ID {row_id!r}")
            seen.add(row_id)
            ids.append(row_id)
    if len(ids) != expected_rows:
        _fail(f"{where}: expected {expected_rows} rows, got {len(ids)} in {path}")
    return ids


def _verify_metrics(path: Path, *, expected_steps: int, require_oracle: bool) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing training metrics: {path}")
    complete: list[dict[str, Any]] = []
    train_rows = 0
    oracle_tokens = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                _fail(f"{path}:{line_number}: metric row must be an object")
            if row.get("event") == "train":
                train_rows += 1
                value = row.get("hard_oracle_tokens_local", 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    oracle_tokens += int(value)
            elif row.get("event") == "complete":
                complete.append(row)
    if train_rows < 1:
        _fail(f"{path}: no train metric events")
    if len(complete) != 1 or complete[0].get("global_step") != expected_steps:
        _fail(f"{path}: expected exactly one complete event at global_step={expected_steps}")
    if require_oracle and oracle_tokens <= 0:
        _fail(f"{path}: positive oracle coefficient but no observed oracle tokens")
    return {"train_metric_events": train_rows, "oracle_tokens_logged": oracle_tokens}


def _matrix_contract(matrix_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix = _load_json(matrix_path, schema=MATRIX_SCHEMA)
    variants = matrix.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        _fail(f"{matrix_path}: at least two matrix variants are required")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            _fail(f"{matrix_path}: variants[{index}] must be an object")
        _require(variant, ("name", "data_view", "overrides"), where=f"variants[{index}]")
        name = variant["name"]
        if not isinstance(name, str) or not name or name in seen:
            _fail(f"{matrix_path}: invalid/duplicate arm name at variants[{index}]")
        if not isinstance(variant["overrides"], dict):
            _fail(f"{matrix_path}: {name}.overrides must be an object")
        seen.add(name)
        normalized.append(variant)
    return matrix, normalized


def _verify_training(
    *,
    matrix_path: Path,
    training_plan_path: Path,
    expected_seeds: Sequence[int],
) -> tuple[dict[str, Any], list[str], int, dict[str, Any]]:
    _, variants = _matrix_contract(matrix_path)
    expected_names = [str(item["name"]) for item in variants]
    by_name = {str(item["name"]): item for item in variants}
    matrix_hash = _sha256(matrix_path)
    suite = _load_json(training_plan_path, schema=TRAIN_SUITE_SCHEMA)
    design = suite.get("design")
    common = suite.get("common_inputs")
    entries = suite.get("seeds")
    if not isinstance(design, dict) or not isinstance(common, dict) or not isinstance(entries, list):
        _fail(f"{training_plan_path}: design/common_inputs/seeds are required")
    if design.get("training_seeds") != list(expected_seeds):
        _fail(f"{training_plan_path}: training seeds differ from {list(expected_seeds)}")
    if design.get("arm_names") != expected_names:
        _fail(f"{training_plan_path}: declared arms differ from matrix order {expected_names}")
    if len(entries) != len(expected_seeds):
        _fail(f"{training_plan_path}: expected {len(expected_seeds)} seed entries")
    matrix_binding = common.get("matrix")
    bound_matrix, _ = _binding(matrix_binding, where="training suite matrix")
    if _sha256(bound_matrix) != matrix_hash:
        _fail("training suite is bound to a different matrix")
    max_steps = common.get("max_steps")
    if not isinstance(max_steps, int) or max_steps < 1:
        _fail("training suite max_steps must be a positive integer")

    canonical_rows: list[tuple[str, int, str]] | None = None
    totals = {"arms": 0, "train_metric_events": 0, "oracle_tokens_logged": 0}
    for position, (entry, seed) in enumerate(zip(entries, expected_seeds, strict=True)):
        if not isinstance(entry, dict) or entry.get("training_seed") != seed:
            _fail(f"training seed entry {position} must be seed {seed}")
        _, nested = _binding(entry.get("plan"), where=f"seed {seed} training plan", schema=TRAIN_PLAN_SCHEMA)
        assert nested is not None
        budget = nested.get("budget_contract")
        arms = nested.get("arms")
        if not isinstance(budget, dict) or not isinstance(arms, list):
            _fail(f"seed {seed}: nested plan lacks budget_contract/arms")
        if budget.get("seed") != seed or budget.get("max_steps_per_arm") != max_steps:
            _fail(f"seed {seed}: seed or optimizer-step budget mismatch")
        if [arm.get("name") for arm in arms if isinstance(arm, dict)] != expected_names:
            _fail(f"seed {seed}: actual arms/count/order differ from supplied matrix")
        nested_matrix = nested.get("matrix")
        if not isinstance(nested_matrix, dict) or nested_matrix.get("sha256") != matrix_hash:
            _fail(f"seed {seed}: nested plan matrix hash mismatch")
        signatures: list[tuple[str, int, str]] = []
        for arm in arms:
            if not isinstance(arm, dict):
                _fail(f"seed {seed}: arm entry must be an object")
            name = str(arm["name"])
            spec = by_name[name]
            if arm.get("overrides") != spec["overrides"]:
                _fail(f"seed {seed} arm {name}: effective plan overrides differ from matrix")
            if Path(str(arm.get("data_path", ""))).name != spec["data_view"]:
                _fail(f"seed {seed} arm {name}: data view differs from matrix")
            rows = arm.get("rows")
            order_hash = arm.get("id_order_sha256")
            if not isinstance(rows, int) or rows < 1 or not isinstance(order_hash, str):
                _fail(f"seed {seed} arm {name}: invalid row/order contract")
            signatures.append((name, rows, order_hash))

            output = Path(str(arm.get("output_dir", "")))
            final = output / "final"
            metadata = _load_json(final / "run_metadata.json")
            if metadata.get("global_step") != max_steps:
                _fail(f"seed {seed} arm {name}: final global_step != {max_steps}")
            training = metadata.get("training")
            opd = metadata.get("opd")
            if not isinstance(training, dict) or not isinstance(opd, dict):
                _fail(f"seed {seed} arm {name}: final metadata lacks training/opd")
            if training.get("seed") != seed or training.get("max_steps") != max_steps:
                _fail(f"seed {seed} arm {name}: final metadata seed/max_steps mismatch")
            for key, expected in spec["overrides"].items():
                if key not in opd or not _same_number(opd[key], expected):
                    _fail(
                        f"seed {seed} arm {name}: final opd.{key}={opd.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            if not (final / "adapter_config.json").is_file():
                _fail(f"seed {seed} arm {name}: missing adapter_config.json")
            if not any((final / filename).is_file() for filename in ("adapter_model.safetensors", "adapter_model.bin")):
                _fail(f"seed {seed} arm {name}: missing adapter weights")
            metric_report = _verify_metrics(
                output / "metrics.jsonl",
                expected_steps=max_steps,
                require_oracle=float(spec["overrides"].get("hard_oracle_ce_weight", 0.0)) > 0.0,
            )
            totals["arms"] += 1
            totals["train_metric_events"] += metric_report["train_metric_events"]
            totals["oracle_tokens_logged"] += metric_report["oracle_tokens_logged"]
        if len({item[2] for item in signatures}) != 1:
            _fail(f"seed {seed}: arm row orders are not matched")
        if canonical_rows is None:
            canonical_rows = signatures
        elif signatures != canonical_rows:
            _fail(f"seed {seed}: data rows/order differ from the first training seed")
    return suite, expected_names, max_steps, totals


def _verify_summary(
    path: Path,
    *,
    arm_names: Sequence[str],
    requests_binding: Mapping[str, Any],
    family_binding: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _load_json(path, schema=SUMMARY_SCHEMA)
    if summary.get("requests") != requests_binding or summary.get("family_ids") != family_binding:
        _fail(f"{path}: request/family-clean binding mismatch")
    models = summary.get("models")
    relative = summary.get("reference_conditioned_request_metrics")
    # Summary JSON is intentionally serialized with ``sort_keys=True``.  Its
    # object-key order therefore cannot preserve the arm order from the frozen
    # evaluation plan; arm identity is the relevant contract here.
    if not isinstance(models, dict) or set(models) != set(arm_names):
        _fail(f"{path}: model arms differ from evaluation plan")
    if not isinstance(relative, dict) or set(relative) != set(arm_names):
        _fail(f"{path}: reference-conditioned arms differ from evaluation plan")
    for arm in arm_names:
        unified = models[arm].get("unified_metrics") if isinstance(models[arm], dict) else None
        if not isinstance(unified, dict):
            _fail(f"{path}: {arm} lacks unified_metrics")
        for metric in (
            "clean_task_success_rate",
            "perturbed_task_success_rate",
            "conditional_retention",
            "regression_rate",
        ):
            value = _finite(unified.get(metric), where=f"{path}:{arm}:{metric}")
            if not 0.0 <= value <= 1.0:
                _fail(f"{path}:{arm}:{metric} outside [0,1]")
    return summary


def _verify_evaluation(
    *,
    training_plan_path: Path,
    evaluation_plan_path: Path,
    arm_names: Sequence[str],
    expected_seeds: Sequence[int],
    expected_requests: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suite = _load_json(evaluation_plan_path, schema=EVAL_SUITE_SCHEMA)
    bound_training, _ = _binding(suite.get("training_suite"), where="evaluation training suite")
    if bound_training.resolve() != training_plan_path.resolve():
        _fail("evaluation suite is bound to a different training suite")
    training_suite = _load_json(training_plan_path, schema=TRAIN_SUITE_SCHEMA)
    training_entries = training_suite.get("seeds")
    if not isinstance(training_entries, list) or len(training_entries) != len(expected_seeds):
        _fail("training suite seed count changed before evaluation verification")
    training_by_seed = {
        item.get("training_seed"): item
        for item in training_entries
        if isinstance(item, dict)
    }
    if set(training_by_seed) != set(expected_seeds):
        _fail("training suite seed identities changed before evaluation verification")
    requests_path, _ = _binding(suite.get("requests"), where="evaluation requests")
    _binding(suite.get("family_clean_ids"), where="family-clean IDs")
    request_ids = _read_jsonl_ids(
        requests_path, expected_rows=expected_requests, where="frozen request file"
    )
    entries = suite.get("seeds")
    if not isinstance(entries, list) or len(entries) != len(expected_seeds):
        _fail("evaluation suite seed count mismatch")
    totals = {"rollout_files": 0, "rollout_rows": 0, "family_summaries": 0}
    for position, (entry, seed) in enumerate(zip(entries, expected_seeds, strict=True)):
        if not isinstance(entry, dict) or entry.get("training_seed") != seed:
            _fail(f"evaluation seed entry {position} must be seed {seed}")
        training_entry = training_by_seed[seed]
        if entry.get("training_plan") != training_entry.get("plan"):
            _fail(f"seed {seed}: evaluation suite is bound to the wrong training plan")
        _, nested = _binding(entry.get("evaluation_plan"), where=f"seed {seed} evaluation plan", schema=EVAL_PLAN_SCHEMA)
        assert nested is not None
        if nested.get("training_plan") != training_entry.get("plan"):
            _fail(f"seed {seed}: nested evaluation plan is bound to the wrong training plan")
        if nested.get("requests") != suite.get("requests"):
            _fail(f"seed {seed}: evaluation request binding differs from suite")
        arms = nested.get("arms")
        if not isinstance(arms, list) or [arm.get("name") for arm in arms if isinstance(arm, dict)] != list(arm_names):
            _fail(f"seed {seed}: evaluation arms/count/order mismatch")
        if entry.get("arm_names") != list(arm_names):
            _fail(f"seed {seed}: declared evaluation arm names mismatch")
        _, nested_training = _binding(
            training_entry.get("plan"),
            where=f"seed {seed} declared training plan",
            schema=TRAIN_PLAN_SCHEMA,
        )
        assert nested_training is not None
        training_arms = {
            str(item.get("name")): item
            for item in nested_training.get("arms", [])
            if isinstance(item, dict)
        }
        if set(training_arms) != set(arm_names):
            _fail(f"seed {seed}: evaluated arms differ from trained arms")
        summary_path_value = entry.get("summary")
        if not isinstance(summary_path_value, dict) or not isinstance(summary_path_value.get("path"), str):
            _fail(f"seed {seed}: missing family-clean summary path")
        summary = _verify_summary(
            Path(summary_path_value["path"]),
            arm_names=arm_names,
            requests_binding=suite["requests"],
            family_binding=suite["family_clean_ids"],
        )
        totals["family_summaries"] += 1
        for arm in arms:
            name = str(arm["name"])
            expected_final = Path(str(training_arms[name].get("output_dir", ""))) / "final"
            if Path(str(arm.get("training_output", ""))).resolve() != expected_final.parent.resolve():
                _fail(f"seed {seed} arm {name}: evaluation training_output is not the planned arm")
            if Path(str(arm.get("adapter", ""))).resolve() != expected_final.resolve():
                _fail(f"seed {seed} arm {name}: evaluation adapter is not the planned final adapter")
            manifest = arm.get("adapter_manifest")
            if not isinstance(manifest, dict):
                _fail(f"seed {seed} arm {name}: missing frozen adapter manifest")
            metadata_hashes = manifest.get("metadata_sha256")
            weight_files = manifest.get("weight_files")
            if not isinstance(metadata_hashes, dict) or not isinstance(weight_files, list):
                _fail(f"seed {seed} arm {name}: malformed adapter manifest")
            for filename in ("adapter_config.json", "run_metadata.json"):
                if metadata_hashes.get(filename) != _sha256(expected_final / filename):
                    _fail(f"seed {seed} arm {name}: stale {filename} adapter-manifest hash")
            actual_weights = [
                {"name": path.name, "bytes": path.stat().st_size}
                for path in sorted(
                    list(expected_final.glob("*.safetensors"))
                    + list(expected_final.glob("*.bin"))
                )
            ]
            if weight_files != actual_weights:
                _fail(f"seed {seed} arm {name}: adapter weight manifest changed")
            rollout_dir = Path(str(arm.get("rollout_output", "")))
            rollout_path = rollout_dir / "qzbc_rollouts.jsonl"
            ids = _read_jsonl_ids(
                rollout_path,
                expected_rows=expected_requests,
                where=f"seed {seed} arm {name} rollout",
            )
            if ids != request_ids:
                _fail(f"seed {seed} arm {name}: rollout IDs/order differ from frozen requests")
            rollout_summary = _load_json(rollout_dir / "summary.json")
            if rollout_summary.get("rollouts") != expected_requests:
                _fail(f"seed {seed} arm {name}: rollout summary is not complete")
            passed = rollout_summary.get("passed")
            if not isinstance(passed, int) or isinstance(passed, bool) or not 0 <= passed <= expected_requests:
                _fail(f"seed {seed} arm {name}: invalid passed count")
            source = summary["models"][name].get("source")
            source_path, _ = _binding(source, where=f"seed {seed} arm {name} summary source")
            if source_path.resolve() != rollout_path.resolve():
                _fail(f"seed {seed} arm {name}: family summary source is not planned rollout")
            totals["rollout_files"] += 1
            totals["rollout_rows"] += len(ids)
    return suite, totals


def _verify_aggregate(
    *,
    aggregate_path: Path,
    evaluation_plan_path: Path,
    arm_names: Sequence[str],
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    payload = _load_json(aggregate_path, schema=AGGREGATE_SCHEMA)
    bound_eval, _ = _binding(payload.get("evaluation_plan"), where="aggregate evaluation plan")
    if bound_eval.resolve() != evaluation_plan_path.resolve():
        _fail("aggregate is bound to a different evaluation plan")
    if payload.get("training_seeds") != list(expected_seeds) or payload.get("n_training_seeds") != len(expected_seeds):
        _fail("aggregate training seeds/count mismatch")
    if payload.get("arms") != list(arm_names):
        _fail("aggregate arms/count/order mismatch")
    per_seed = payload.get("per_seed")
    aggregate = payload.get("aggregate")
    metrics = payload.get("metrics")
    if not isinstance(per_seed, dict) or set(per_seed) != {str(seed) for seed in expected_seeds}:
        _fail("aggregate per_seed keys mismatch")
    if not isinstance(aggregate, dict) or set(aggregate) != set(arm_names):
        _fail("aggregate arm payload mismatch")
    if not isinstance(metrics, dict) or not metrics:
        _fail("aggregate metric definitions are missing")
    for arm in arm_names:
        for metric, definition in metrics.items():
            if not isinstance(definition, dict) or definition.get("direction") not in {"higher", "lower"}:
                _fail(f"aggregate metric {metric}: invalid direction")
            values = [
                _finite(per_seed[str(seed)][arm][metric], where=f"aggregate:{seed}:{arm}:{metric}")
                for seed in expected_seeds
            ]
            item = aggregate[arm].get(metric)
            if not isinstance(item, dict) or item.get("n_training_seeds") != len(values):
                _fail(f"aggregate:{arm}:{metric}: missing/incomplete aggregate")
            expected = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }
            for key, value in expected.items():
                if not _same_number(item.get(key), value):
                    _fail(f"aggregate:{arm}:{metric}:{key} is not recomputed from per_seed values")
            worst = min(values) if definition["direction"] == "higher" else max(values)
            worst_seeds = [seed for seed, value in zip(expected_seeds, values, strict=True) if value == worst]
            if not _same_number(item.get("worst_value"), worst) or item.get("worst_training_seeds") != worst_seeds:
                _fail(f"aggregate:{arm}:{metric}: worst-seed calculation mismatch")
    return {"metrics": len(metrics), "arms": len(arm_names), "seeds": len(expected_seeds)}


def _verify_token_audit(path: Path) -> dict[str, Any]:
    audit = _load_json(path, schema=TOKEN_AUDIT_SCHEMA)
    if audit.get("status") != "COMPLETE":
        _fail(f"{path}: token-matching audit status must be COMPLETE")
    overall = audit.get("overall")
    if not isinstance(overall, dict):
        _fail(f"{path}: token-matching overall section is required")
    for key in ("exact_count_match_rate", "exact_weight_multiset_match_rate"):
        if not _same_number(overall.get(key), 1.0):
            _fail(f"{path}: {key} must equal 1.0")
    rows = overall.get("rows")
    if not isinstance(rows, int) or rows < 1:
        _fail(f"{path}: audited row count must be positive")
    tokenizer_sha256 = audit.get("tokenizer_sha256")
    if not isinstance(tokenizer_sha256, str) or not tokenizer_sha256:
        _fail(f"{path}: tokenizer_sha256 must be a non-empty string")
    return {"rows": rows, "tokenizer_sha256": tokenizer_sha256}


def _verify_token_training_bindings(
    suite: Mapping[str, Any],
    *,
    token_audit_path: Path,
    audited_rows: int,
    audited_tokenizer_sha256: str,
) -> None:
    expected_path = token_audit_path.resolve()
    expected_hash = _sha256(expected_path)
    if suite.get("design", {}).get("contract_mode") != "token_matched":
        _fail("token-control training suite did not use contract_mode=token_matched")
    prerequisites = suite.get("prerequisite_audits")
    if not isinstance(prerequisites, list) or len(prerequisites) != 1:
        _fail("token-control suite must contain exactly one prerequisite audit")
    prerequisite = prerequisites[0]
    if (
        not isinstance(prerequisite, dict)
        or Path(str(prerequisite.get("path", ""))).resolve() != expected_path
        or prerequisite.get("sha256") != expected_hash
        or prerequisite.get("schema_version") != TOKEN_AUDIT_SCHEMA
        or prerequisite.get("status") != "COMPLETE"
    ):
        _fail("token-control suite prerequisite does not match --token-audit")
    entries = suite.get("seeds")
    if not isinstance(entries, list):
        _fail("token-control suite seed plans are missing")
    for entry in entries:
        seed = entry.get("training_seed") if isinstance(entry, dict) else None
        _, nested = _binding(
            entry.get("plan") if isinstance(entry, dict) else None,
            where=f"seed {seed} token-control training plan",
            schema=TRAIN_PLAN_SCHEMA,
        )
        assert nested is not None
        binding = nested.get("token_matching_audit")
        if (
            not isinstance(binding, dict)
            or binding.get("status") != "PASS"
            or Path(str(binding.get("path", ""))).resolve() != expected_path
            or binding.get("sha256") != expected_hash
            or binding.get("rows") != audited_rows
            or binding.get("tokenizer_sha256") != audited_tokenizer_sha256
        ):
            _fail(f"seed {seed}: token-matching audit binding is absent or stale")
        arms = nested.get("arms")
        if not isinstance(arms, list) or not arms:
            _fail(f"seed {seed}: token-control training arms are missing")
        for arm in arms:
            if not isinstance(arm, dict):
                _fail(f"seed {seed}: malformed token-control arm")
            name = str(arm.get("name", ""))
            metadata_path = Path(str(arm.get("output_dir", ""))) / "final" / "run_metadata.json"
            metadata = _load_json(metadata_path)
            if metadata.get("tokenizer_sha256") != audited_tokenizer_sha256:
                _fail(
                    f"seed {seed} arm {name}: completed tokenizer_sha256 differs "
                    "from the bound token-matching audit"
                )


def verify(
    *,
    matrix_path: Path,
    training_plan_path: Path,
    evaluation_plan_path: Path,
    aggregate_path: Path,
    expected_seeds: Sequence[int],
    expected_requests: int,
    token_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a completed suite and return a compact in-memory report."""

    if expected_requests < 1:
        _fail("expected_requests must be positive")
    seeds = _parse_seeds(expected_seeds)
    training_suite, arm_names, max_steps, training = _verify_training(
        matrix_path=matrix_path.resolve(),
        training_plan_path=training_plan_path.resolve(),
        expected_seeds=seeds,
    )
    _, evaluation = _verify_evaluation(
        training_plan_path=training_plan_path.resolve(),
        evaluation_plan_path=evaluation_plan_path.resolve(),
        arm_names=arm_names,
        expected_seeds=seeds,
        expected_requests=expected_requests,
    )
    aggregate = _verify_aggregate(
        aggregate_path=aggregate_path.resolve(),
        evaluation_plan_path=evaluation_plan_path.resolve(),
        arm_names=arm_names,
        expected_seeds=seeds,
    )
    is_token_control = set(arm_names) == TOKEN_CONTROL_ARMS
    if is_token_control and token_audit_path is None:
        _fail("the exact token-control suite requires --token-audit")
    token_audit = _verify_token_audit(token_audit_path.resolve()) if token_audit_path else None
    if is_token_control:
        assert token_audit_path is not None and token_audit is not None
        _verify_token_training_bindings(
            training_suite,
            token_audit_path=token_audit_path,
            audited_rows=int(token_audit["rows"]),
            audited_tokenizer_sha256=str(token_audit["tokenizer_sha256"]),
        )
    return {
        "status": "PASS",
        "matrix": str(matrix_path.resolve()),
        "training_seeds": seeds,
        "arms": arm_names,
        "max_steps_per_arm": max_steps,
        "expected_requests_per_rollout": expected_requests,
        "training": training,
        "evaluation": evaluation,
        "aggregate": aggregate,
        "token_matching_audit": token_audit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="42,43,44")
    parser.add_argument("--expected-requests", type=int, default=4200)
    parser.add_argument("--token-audit", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify(
            matrix_path=args.matrix,
            training_plan_path=args.training_plan,
            evaluation_plan_path=args.evaluation_plan,
            aggregate_path=args.aggregate,
            expected_seeds=_parse_seeds(args.expected_seeds),
            expected_requests=args.expected_requests,
            token_audit_path=args.token_audit,
        )
    except (VerificationError, FileNotFoundError, OSError) as exc:
        print(f"ICLR_REPAIR_COMPLETION_AUDIT=FAIL error={exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    print("ICLR_REPAIR_COMPLETION_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
