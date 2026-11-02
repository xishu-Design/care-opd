#!/usr/bin/env python3
"""Freeze per-training-seed rollout and family-clean summary commands."""

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

from mechanism_experiments.causal_ablation import evaluate_arms  # noqa: E402

try:
    from .common import (
        EVAL_SUITE_SCHEMA,
        TRAIN_SUITE_SCHEMA,
        atomic_json,
        atomic_text,
        load_json_object,
        require_keys,
        sha256,
        verify_file_binding,
    )
except ImportError:
    from common import (
        EVAL_SUITE_SCHEMA,
        TRAIN_SUITE_SCHEMA,
        atomic_json,
        atomic_text,
        load_json_object,
        require_keys,
        sha256,
        verify_file_binding,
    )


def build_evaluation_suite(args: argparse.Namespace) -> dict[str, Any]:
    training_plan_path = args.training_plan.resolve()
    training_suite = load_json_object(training_plan_path, schema=TRAIN_SUITE_SCHEMA)
    requests = args.requests.resolve()
    family_ids = args.family_clean_ids.resolve()
    if not requests.is_file() or not family_ids.is_file():
        raise FileNotFoundError(requests if not requests.is_file() else family_ids)
    output_root = args.output_root.resolve()
    entries: list[dict[str, Any]] = []
    all_commands: list[list[str]] = []

    for index, seed_entry in enumerate(training_suite.get("seeds", [])):
        if not isinstance(seed_entry, dict):
            raise ValueError(f"training suite seeds[{index}] must be an object")
        require_keys(seed_entry, ("training_seed", "plan"), where=f"training suite seeds[{index}]")
        seed = seed_entry["training_seed"]
        train_plan_path, nested_train = verify_file_binding(
            seed_entry["plan"],
            where=f"training suite seeds[{index}].plan",
            schema="care_opd.causal_ablation_plan.v1",
        )
        if nested_train.get("budget_contract", {}).get("seed") != seed:
            raise ValueError(f"seed {seed}: nested plan seed mismatch")
        # This entry point is intentionally post-training: do not freeze rollout
        # commands for missing, stale, or only partially completed adapters.
        evaluate_arms._validate_training_outputs(nested_train)
        seed_root = output_root / f"seed_{seed}"
        rollout_root = seed_root / "rollouts"
        eval_args = argparse.Namespace(
            experiment_plan=train_plan_path,
            requests=requests,
            output_root=rollout_root,
            python=args.python,
            num_workers=args.num_workers,
            devices=args.devices,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            seed=args.generation_seed,
            execute=False,
        )
        eval_plan, eval_commands = evaluate_arms.build_plan(eval_args)
        eval_plan_path = seed_root / "evaluation_plan.json"
        atomic_json(eval_plan_path, eval_plan, freeze_existing=True)
        arm_names = [str(arm["name"]) for arm in eval_plan["arms"]]
        summary_dir = seed_root / "summary"
        summary_command = [
            args.python,
            str(PROJECT_ROOT / "mechanism_experiments" / "causal_ablation" / "summarize_arms.py"),
            "--requests",
            str(requests),
            "--family-clean-ids",
            str(family_ids),
        ]
        for arm in eval_plan["arms"]:
            summary_command += [
                "--rollout",
                f"{arm['name']}={Path(arm['rollout_output']) / 'qzbc_rollouts.jsonl'}",
            ]
        summary_command += ["--reference", args.reference, "--output-dir", str(summary_dir)]
        if args.reference not in arm_names:
            raise ValueError(f"seed {seed}: reference {args.reference!r} is not an arm")
        commands = list(eval_commands) + [summary_command]
        command_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / "src"))}:{shlex.quote(str(PROJECT_ROOT))}${{PYTHONPATH:+:${{PYTHONPATH}}}}',
            "",
        ] + [shlex.join(command) for command in commands]
        atomic_text(seed_root / "commands.sh", "\n\n".join(command_lines) + "\n", freeze_existing=True)
        all_commands.extend(commands)
        entries.append(
            {
                "training_seed": seed,
                "training_plan": seed_entry["plan"],
                "evaluation_plan": {"path": str(eval_plan_path), "sha256": sha256(eval_plan_path)},
                "rollout_root": str(rollout_root),
                "summary": {"path": str(summary_dir / "summary.json")},
                "arm_names": arm_names,
                "commands": commands,
            }
        )

    if len(entries) < 2:
        raise ValueError("training suite must contain at least two seeds")
    global_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / "src"))}:{shlex.quote(str(PROJECT_ROOT))}${{PYTHONPATH:+:${{PYTHONPATH}}}}',
        "",
    ] + [shlex.join(command) for command in all_commands]
    atomic_text(output_root / "commands.sh", "\n\n".join(global_lines) + "\n", freeze_existing=True)
    suite = {
        "schema_version": EVAL_SUITE_SCHEMA,
        "training_suite": {"path": str(training_plan_path), "sha256": sha256(training_plan_path)},
        "requests": {"path": str(requests), "sha256": sha256(requests)},
        "family_clean_ids": {"path": str(family_ids), "sha256": sha256(family_ids)},
        "protocol": {
            "generation_seed": args.generation_seed,
            "greedy": True,
            "num_workers": args.num_workers,
            "devices": args.devices,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "reference_arm": args.reference,
            "same_requests_verifier_and_generation_protocol_across_training_seeds": True,
        },
        "contrasts": training_suite.get("contrasts", {"source": None, "items": []}),
        "seeds": entries,
        "execution_requested": False,
    }
    atomic_json(output_root / "multiseed_evaluation_plan.json", suite, freeze_existing=True)
    return suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--family-clean-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--reference", default="full_selected")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite = build_evaluation_suite(args)
    print("PLAN_ONLY=True")
    print(f"training_seeds={','.join(str(entry['training_seed']) for entry in suite['seeds'])}")
    print(f"plan={args.output_root.resolve() / 'multiseed_evaluation_plan.json'}")
    print(f"commands={args.output_root.resolve() / 'commands.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
