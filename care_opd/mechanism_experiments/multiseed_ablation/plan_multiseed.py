#!/usr/bin/env python3
"""Freeze one matched causal-ablation training plan per training seed."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanism_experiments.causal_ablation import run_single_round  # noqa: E402
from mechanism_experiments.hard_oracle_attribution import plan_experiment as oracle_planner  # noqa: E402
from mechanism_experiments.token_matched_control import plan_experiment as token_planner  # noqa: E402

try:
    from .common import (
        CONTRASTS_SCHEMA,
        TRAIN_SUITE_SCHEMA,
        atomic_json,
        atomic_text,
        load_json_object,
        parse_seeds,
        sha256,
    )
except ImportError:
    from common import (
        CONTRASTS_SCHEMA,
        TRAIN_SUITE_SCHEMA,
        atomic_json,
        atomic_text,
        load_json_object,
        parse_seeds,
        sha256,
    )


ORACLE_ARMS = {
    "care_with_oracle_ce",
    "care_without_oracle_ce",
    "gkd_without_oracle_ce",
    "gkd_plus_same_oracle_ce",
    "oracle_ce_only_on_policy",
}
TOKEN_CONTROL_ARMS = {"localized_kd_only", "random_token_kd_only"}


def _inline_pair(raw: str, known: set[str]) -> tuple[str, str, str]:
    if "=" not in raw:
        raise ValueError("--contrast must use NAME=LEFT-RIGHT")
    name, pair = raw.split("=", 1)
    candidates = [
        (pair[:index], pair[index + 1 :])
        for index, character in enumerate(pair)
        if character == "-" and pair[:index] in known and pair[index + 1 :] in known
    ]
    if not name or len(candidates) != 1:
        raise ValueError(
            f"invalid or ambiguous --contrast {raw!r}; expected NAME=LEFT-RIGHT "
            "with arms present in the matrix"
        )
    left, right = candidates[0]
    return name, left, right


def _load_contrasts(
    path: Path | None,
    arm_names: Sequence[str],
    inline_specs: Sequence[str],
) -> dict[str, Any]:
    source: dict[str, str] | None = None
    items: list[Any] = []
    if path is not None:
        resolved = path.resolve()
        payload = load_json_object(resolved, schema=CONTRASTS_SCHEMA)
        file_items = payload.get("contrasts")
        if not isinstance(file_items, list) or not file_items:
            raise ValueError(f"{resolved}: contrasts must be a non-empty list")
        source = {"path": str(resolved), "sha256": sha256(resolved)}
        items.extend(file_items)
    known = set(arm_names)
    for raw in inline_specs:
        name, left, right = _inline_pair(raw, known)
        items.append(
            {
                "name": name,
                "left": left,
                "right": right,
                "definition": "Pre-specified on the plan_multiseed command line.",
            }
        )
    names: set[str] = set()
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"contrasts[{index}] must be an object")
        name, left, right = item.get("name"), item.get("left"), item.get("right")
        definition = item.get("definition", "")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"contrasts[{index}] has invalid/duplicate name")
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            raise ValueError(f"contrast {name!r} requires distinct string arms")
        if left not in known or right not in known:
            raise ValueError(
                f"contrast {name!r} references arms absent from the matrix: "
                f"left={left!r}, right={right!r}"
            )
        if not isinstance(definition, str):
            raise ValueError(f"contrast {name!r} definition must be a string")
        names.add(name)
        normalized.append(
            {"name": name, "left": left, "right": right, "definition": definition}
        )
    return {
        "source": source,
        "inline_specs": list(inline_specs),
        "delta_definition": "left arm metric minus right arm metric within the same training seed",
        "items": normalized,
    }


def _matrix_arm_set(path: Path) -> set[str]:
    payload = load_json_object(path.resolve(), schema=run_single_round.MATRIX_SCHEMA)
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError(f"{path}: variants must be a list")
    names = {item.get("name") for item in variants if isinstance(item, dict)}
    if any(not isinstance(name, str) for name in names):
        raise ValueError(f"{path}: every matrix arm requires a string name")
    return {str(name) for name in names}


def _load_prerequisite_audits(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        payload = load_json_object(resolved)
        status = payload.get("status")
        if status not in {"PASS", "COMPLETE"}:
            raise ValueError(f"{resolved}: prerequisite audit status must be PASS or COMPLETE")
        records.append(
            {
                "path": str(resolved),
                "sha256": sha256(resolved),
                "schema_version": payload.get("schema_version"),
                "status": status,
            }
        )
    return records


def build_multiseed_plan(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_seeds(args.seeds)
    output_root = args.output_root.resolve()
    matrix_arms = _matrix_arm_set(args.matrix)
    requested_contract = getattr(args, "contract", "auto")
    if requested_contract not in {"auto", "generic", "hard_oracle", "token_matched"}:
        raise ValueError("contract must be auto, generic, hard_oracle, or token_matched")
    contract_mode = requested_contract
    if contract_mode == "auto":
        if matrix_arms == ORACLE_ARMS:
            contract_mode = "hard_oracle"
        elif matrix_arms == TOKEN_CONTROL_ARMS:
            contract_mode = "token_matched"
        else:
            contract_mode = "generic"
    prerequisites = _load_prerequisite_audits(
        list(getattr(args, "prerequisite_audit", None) or [])
    )
    if contract_mode == "hard_oracle" and matrix_arms != ORACLE_ARMS:
        raise ValueError("hard_oracle contract requires the frozen hard-oracle arm set")
    if contract_mode == "token_matched" and matrix_arms != TOKEN_CONTROL_ARMS:
        raise ValueError("token_matched contract requires the frozen token-control arm set")
    if matrix_arms == TOKEN_CONTROL_ARMS and len(prerequisites) != 1:
        raise ValueError(
            "token-matched matrix requires exactly one --prerequisite-audit from "
            "audit_token_matching.py"
        )
    seed_entries: list[dict[str, Any]] = []
    canonical_arm_names: list[str] | None = None
    canonical_contract: dict[str, Any] | None = None
    canonical_matched_signature: dict[str, Any] | None = None

    for seed in seeds:
        train_root = output_root / f"seed_{seed}" / "train"
        single_args = argparse.Namespace(
            matrix=args.matrix,
            views_dir=args.views_dir,
            train_config=args.train_config,
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            reference_model=args.reference_model,
            output_root=train_root,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            seed=seed,
            nproc_per_node=1,
            python=args.python,
            resume=args.resume,
            execute=False,
        )
        if contract_mode == "hard_oracle":
            plan, commands = oracle_planner.build_plan(single_args)
            if "scientific_contract" not in plan or "oracle_training_view_audit" not in plan:
                raise ValueError("hard-oracle planner omitted its scientific contract or view audit")
        elif contract_mode == "token_matched":
            single_args.token_audit = Path(prerequisites[0]["path"])
            plan, commands = token_planner.build_plan(single_args)
            if "scientific_contract" not in plan or "token_matching_audit" not in plan:
                raise ValueError("token-matched planner omitted its scientific contract or audit")
        else:
            plan, commands = run_single_round.build_plan(single_args)
        arm_names = [str(arm["name"]) for arm in plan["arms"]]
        matched_signature = {
            "matrix_sha256": plan["matrix"]["sha256"],
            "train_config_sha256": plan["train_config"]["sha256"],
            "student_initialization": plan["student_initialization"],
            "teacher": plan["teacher"],
            "reference_model": plan["reference_model"],
            "training_code": plan["training_code"],
            "scientific_contract": plan.get("scientific_contract"),
            "oracle_training_view_audit": plan.get("oracle_training_view_audit"),
            "token_matching_audit": plan.get("token_matching_audit"),
            "arms": [
                {
                    "name": arm["name"],
                    "data_sha256": arm["data_sha256"],
                    "rows": arm["rows"],
                    "id_order_sha256": arm["id_order_sha256"],
                    "overrides": arm["overrides"],
                }
                for arm in plan["arms"]
            ],
        }
        if canonical_arm_names is None:
            canonical_arm_names = arm_names
            canonical_contract = {
                key: value for key, value in plan["budget_contract"].items() if key != "seed"
            }
            canonical_matched_signature = matched_signature
        else:
            if arm_names != canonical_arm_names:
                raise ValueError(f"seed {seed}: arm names/order differ from the first seed")
            contract = {key: value for key, value in plan["budget_contract"].items() if key != "seed"}
            if contract != canonical_contract:
                raise ValueError(f"seed {seed}: budget contract differs from the first seed")
            if matched_signature != canonical_matched_signature:
                raise ValueError(f"seed {seed}: data/model/config signature differs from the first seed")

        plan_path = train_root / "experiment_plan.json"
        plan_text = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        atomic_text(plan_path, plan_text, freeze_existing=True)
        shell_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / "src"))}:{shlex.quote(str(PROJECT_ROOT))}${{PYTHONPATH:+:${{PYTHONPATH}}}}',
            "",
        ]
        shell_lines.extend(shlex.join(command) for command in commands)
        atomic_text(
            train_root / "commands.sh",
            "\n\n".join(shell_lines) + "\n",
            freeze_existing=True,
        )
        seed_entries.append(
            {
                "training_seed": seed,
                "training_root": str(train_root),
                "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
                "arms": [
                    {
                        "name": arm["name"],
                        "output_dir": arm["output_dir"],
                        "command": arm["command"],
                    }
                    for arm in plan["arms"]
                ],
            }
        )

    assert canonical_arm_names is not None
    contrasts_path = getattr(args, "contrasts_file", None)
    inline_specs = list(getattr(args, "contrast", None) or [])
    if contrasts_path is None and not inline_specs:
        if matrix_arms == ORACLE_ARMS:
            contrasts_path = HERE / "hard_oracle_contrasts.json"
        elif matrix_arms == TOKEN_CONTROL_ARMS:
            contrasts_path = HERE / "token_control_contrasts.json"
    contrasts = _load_contrasts(contrasts_path, canonical_arm_names, inline_specs)
    suite = {
        "schema_version": TRAIN_SUITE_SCHEMA,
        "design": {
            "estimand": "between-training-seed variability of matched causal-ablation arms",
            "training_seeds": seeds,
            "n_training_seeds": len(seeds),
            "arm_names": canonical_arm_names,
            "same_inputs_budget_and_arm_order_across_seeds": True,
            "only_intended_stochastic_treatment_variable": "training seed",
            "matched_signature": canonical_matched_signature,
            "contract_mode": contract_mode,
            "statistical_boundary": (
                "Per-seed request/cluster bootstrap quantifies held-out sampling uncertainty; "
                "it is not a substitute for variation across independently trained seeds."
            ),
        },
        "common_inputs": {
            "matrix": {"path": str(args.matrix.resolve()), "sha256": sha256(args.matrix.resolve())},
            "views_dir": str(args.views_dir.resolve()),
            "train_config": {
                "path": str(args.train_config.resolve()),
                "sha256": sha256(args.train_config.resolve()),
            },
            "student_model": args.student_model,
            "teacher_model": args.teacher_model,
            "reference_model": args.reference_model,
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "python": args.python,
            "resume": bool(args.resume),
            "nproc_per_node": 1,
        },
        "seeds": seed_entries,
        "contrasts": contrasts,
        "prerequisite_audits": prerequisites,
        "execution_requested": False,
    }
    atomic_json(output_root / "multiseed_plan.json", suite, freeze_existing=True)
    return suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=PROJECT_ROOT / "mechanism_experiments" / "causal_ablation" / "ablation_matrix.json",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="repeatable pre-specified contrast NAME=LEFT-RIGHT",
    )
    parser.add_argument(
        "--contract",
        choices=("auto", "generic", "hard_oracle", "token_matched"),
        default="auto",
        help="auto detects the formal hard-oracle and exact token-matched matrices",
    )
    parser.add_argument(
        "--prerequisite-audit",
        action="append",
        type=Path,
        default=[],
        help="hash-bind a PASS/COMPLETE prerequisite audit; required for token control",
    )
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--contrasts-file",
        type=Path,
        help="optional hash-bound, pre-specified paired arm contrasts",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite = build_multiseed_plan(args)
    print("PLAN_ONLY=True")
    print(f"training_seeds={','.join(str(seed) for seed in suite['design']['training_seeds'])}")
    print(f"arms_per_seed={len(suite['design']['arm_names'])}")
    print(f"plan={args.output_root.resolve() / 'multiseed_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
