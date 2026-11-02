#!/usr/bin/env python3
"""Audit and generate a hard-oracle attribution training plan; never train."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .contract import audit_matrix, audit_training_view
except ImportError:  # Direct script execution.
    from contract import audit_matrix, audit_training_view

from mechanism_experiments.causal_ablation import run_single_round


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=HERE / "oracle_attribution_matrix.json")
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser


def build_plan(args: argparse.Namespace):
    contract = audit_matrix(args.matrix)
    view_audit = audit_training_view(args.views_dir / contract["data_view"])
    delegated = argparse.Namespace(
        matrix=args.matrix,
        views_dir=args.views_dir,
        train_config=args.train_config,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        reference_model=args.reference_model,
        output_root=args.output_root,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        nproc_per_node=args.nproc_per_node,
        python=args.python,
        resume=args.resume,
        execute=False,
    )
    plan, commands = run_single_round.build_plan(delegated)
    plan["scientific_contract"] = contract
    plan["oracle_training_view_audit"] = view_audit
    plan["execution_requested"] = False
    return plan, commands


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan, commands = build_plan(args)
    root = args.output_root.resolve()
    run_single_round._atomic_text(
        root / "experiment_plan.json",
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    shell = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / "src"))}:{shlex.quote(str(PROJECT_ROOT))}${{PYTHONPATH:+:${{PYTHONPATH}}}}',
        "",
    ]
    shell.extend(shlex.join(command) for command in commands)
    run_single_round._atomic_text(root / "commands.sh", "\n\n".join(shell) + "\n")
    print("PLAN_ONLY=True")
    print("CONTRACT=PASS")
    print(f"plan={root / 'experiment_plan.json'}")
    print(f"commands={root / 'commands.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
