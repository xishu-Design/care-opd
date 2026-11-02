#!/usr/bin/env python3
"""Generate or explicitly execute QZBC rollout commands for trained arms."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

try:
    from .build_views import _sha256
except ImportError:
    from build_views import _sha256


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        raise


def _validate_training_outputs(training_plan: dict[str, Any]) -> None:
    """Fail closed before spending GPU time on incomplete/mixed adapters."""

    expected_step = int(training_plan["budget_contract"]["max_steps_per_arm"])
    for arm in training_plan["arms"]:
        name = str(arm["name"])
        arm_root = Path(arm["output_dir"])
        final_dir = arm_root / "final"
        adapter_config = final_dir / "adapter_config.json"
        weights = list(final_dir.glob("*.safetensors")) + list(final_dir.glob("*.bin"))
        metadata_path = final_dir / "run_metadata.json"
        metrics_path = arm_root / "metrics.jsonl"
        if not adapter_config.is_file() or not weights or not metadata_path.is_file():
            raise FileNotFoundError(f"{name}: incomplete final adapter at {final_dir}")
        if any(path.stat().st_size <= 0 for path in weights):
            raise ValueError(f"{name}: empty adapter weight file")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("global_step", -1)) != expected_step:
            raise ValueError(
                f"{name}: final global_step={metadata.get('global_step')} != {expected_step}"
            )
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        complete = False
        with metrics_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{metrics_path}:{line_number}: invalid JSON"
                    ) from exc
                if (
                    row.get("event") == "complete"
                    and int(row.get("global_step", -1)) == expected_step
                ):
                    complete = True
        if not complete:
            raise ValueError(f"{name}: no complete@{expected_step} metrics event")


def _adapter_manifest(final_dir: Path) -> dict[str, Any] | None:
    if not final_dir.is_dir():
        return None
    small_files = ("adapter_config.json", "run_metadata.json")
    metadata = {
        name: _sha256(final_dir / name)
        for name in small_files
        if (final_dir / name).is_file()
    }
    weights = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(
            list(final_dir.glob("*.safetensors")) + list(final_dir.glob("*.bin"))
        )
    ]
    return {"metadata_sha256": metadata, "weight_files": weights}


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[list[str]]]:
    plan_path = args.experiment_plan.resolve()
    requests = args.requests.resolve()
    if not plan_path.is_file() or not requests.is_file():
        raise FileNotFoundError(plan_path if not plan_path.is_file() else requests)
    training_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if training_plan.get("schema_version") != "care_opd.causal_ablation_plan.v1":
        raise ValueError("unexpected training plan schema")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    if args.num_workers > 1:
        devices = [item.strip() for item in args.devices.split(",") if item.strip()]
        if len(devices) != args.num_workers:
            raise ValueError("--devices count must equal --num-workers")
    commands: list[list[str]] = []
    arms: list[dict[str, Any]] = []
    for arm in training_plan["arms"]:
        final_dir = Path(arm["output_dir"]) / "final"
        # OPD LoRA outputs retain an adapter whose base is the common student
        # initialization; qzbc_rollout already supports base+adapter loading.
        model_value = training_plan["student_initialization"]["value"]
        command = [
            args.python,
            "-m",
            "care_opd.qzbc_rollout",
            "--requests",
            str(requests),
            "--output-dir",
            str((args.output_root.resolve() / arm["name"])),
            "--model",
            model_value,
            "--adapter",
            str(final_dir),
            "--dtype",
            args.dtype,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--greedy",
            "--seed",
            str(args.seed),
            "--num-workers",
            str(args.num_workers),
            "--devices",
            args.devices,
        ]
        commands.append(command)
        arms.append(
            {
                "name": arm["name"],
                "training_output": arm["output_dir"],
                "adapter": str(final_dir),
                "adapter_manifest": _adapter_manifest(final_dir),
                "rollout_output": str(args.output_root.resolve() / arm["name"]),
                "command": command,
            }
        )
    payload = {
        "schema_version": "care_opd.causal_ablation_eval_plan.v1",
        "training_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "requests": {"path": str(requests), "sha256": _sha256(requests)},
        "protocol": {
            "generator": "care_opd.qzbc_rollout",
            "greedy": True,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "num_workers": args.num_workers,
            "devices": args.devices,
            "note": "The existing qzbc parser and verifier are reused; no alternate evaluator is introduced.",
        },
        "arms": arms,
        "execution_requested": bool(args.execute),
    }
    return payload, commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-plan", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="run after writing plan; default is plan only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan, commands = build_plan(args)
    root = args.output_root.resolve()
    _atomic_text(root / "evaluation_plan.json", json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    repo_root = Path(__file__).resolve().parents[2]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'export PYTHONPATH={shlex.quote(str(repo_root / "src"))}:{shlex.quote(str(repo_root))}${{PYTHONPATH:+:${{PYTHONPATH}}}}',
        "",
    ] + [shlex.join(command) for command in commands]
    _atomic_text(root / "commands.sh", "\n\n".join(lines) + "\n")
    print(f"PLAN_ONLY={not args.execute}")
    print(f"plan={root / 'evaluation_plan.json'}")
    if args.execute:
        _validate_training_outputs(
            json.loads(args.experiment_plan.resolve().read_text(encoding="utf-8"))
        )
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        project_paths = os.pathsep.join((str(repo_root / "src"), str(repo_root)))
        env["PYTHONPATH"] = (
            project_paths + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        )
        for command in commands:
            subprocess.run(command, cwd=repo_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
