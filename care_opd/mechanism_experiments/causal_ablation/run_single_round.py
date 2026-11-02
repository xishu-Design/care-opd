#!/usr/bin/env python3
"""Validate and plan (or explicitly execute) equal-budget OPD ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .build_views import _read_jsonl, _row_id, _sha256
except ImportError:  # Direct ``python path/to/run_single_round.py`` execution.
    from build_views import _read_jsonl, _row_id, _sha256

MATRIX_SCHEMA = "care_opd.causal_ablation_matrix.v1"
PLAN_SCHEMA = "care_opd.causal_ablation_plan.v1"
ALLOWED_OVERRIDES = {
    "baseline_mode",
    "mask_mode",
    "random_mask_seed",
    "use_risk_weighting",
    "oracle_risk_weighting_mode",
    "use_teacher_trust",
    "retain_outside_repair",
    "distillation_coefficient",
    "retention_coefficient",
    "hard_oracle_ce_weight",
    "kl_mode",
    "temperature",
    "js_beta",
    "min_host_verification_confidence",
    "min_teacher_top_probability",
    "min_teacher_margin",
    "max_teacher_normalized_entropy",
    "max_teacher_student_js",
    "max_teacher_top_log_ratio",
    "response_chunk_size",
    "max_length",
    "max_response_tokens",
    "add_eos",
}


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


def _load_matrix(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError(f"{path}: expected schema_version={MATRIX_SCHEMA!r}")
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        raise ValueError(f"{path}: variants must contain at least two arms")
    seen: set[str] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"{path}: variants[{index}] must be an object")
        name, view, overrides = variant.get("name"), variant.get("data_view"), variant.get("overrides")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"{path}: variants[{index}] has invalid/duplicate name")
        if not isinstance(view, str) or Path(view).name != view:
            raise ValueError(f"{path}: variants[{index}].data_view must be a file name")
        if not isinstance(overrides, dict):
            raise ValueError(f"{path}: variants[{index}].overrides must be an object")
        unknown = sorted(set(overrides) - ALLOWED_OVERRIDES)
        if unknown:
            raise ValueError(f"{path}: variants[{index}] has unsupported overrides {unknown}")
        if overrides.get("baseline_mode", "care_opd") != "care_opd":
            raise ValueError(
                f"{path}: custom arms must use baseline_mode=care_opd; preset bundles would overwrite switches"
            )
        seen.add(name)
    return payload


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TypeError(f"unsupported CLI override value: {value!r}")


def _fingerprint_model(value: str) -> dict[str, Any]:
    path = Path(value)
    result: dict[str, Any] = {"value": value, "kind": "local_path" if path.exists() else "model_id"}
    if path.is_file():
        result["sha256"] = _sha256(path)
    elif path.is_dir():
        # Hash the immutable-looking model metadata, not multi-GiB weights.
        candidates = [path / name for name in ("config.json", "adapter_config.json", "tokenizer_config.json")]
        metadata = [(candidate.name, _sha256(candidate)) for candidate in candidates if candidate.is_file()]
        result["metadata_sha256"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()
        result["metadata_files"] = metadata
    return result


def _training_code_fingerprints(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Bind a plan to the exact code that constructs and optimizes its loss."""

    relative_paths = (
        "src/care_opd/collator.py",
        "src/care_opd/train_opd.py",
        "src/care_opd/losses.py",
        "src/care_opd/trainer_utils.py",
        "mechanism_experiments/causal_ablation/run_single_round.py",
    )
    result: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    return result


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[list[str]]]:
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive and explicit for equal-budget comparison")
    if args.save_steps < 1:
        raise ValueError("--save-steps must be positive")
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be positive")
    matrix = _load_matrix(args.matrix)
    config = args.train_config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    views_dir = args.views_dir.resolve()
    output_root = args.output_root.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    commands: list[list[str]] = []
    arms: list[dict[str, Any]] = []
    expected_ids: list[str] | None = None
    for variant in matrix["variants"]:
        data_path = views_dir / variant["data_view"]
        rows = _read_jsonl(data_path)
        ids = [_row_id(row, where=str(data_path)) for row in rows]
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(
                f"{data_path}: ID order differs from first arm; optimizer/data budget is not matched"
            )
        output_dir = output_root / variant["name"]
        command = [
            args.python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            "-m",
            "care_opd.train_opd",
            "--config",
            str(config),
            "--model",
            args.student_model,
            "--teacher",
            args.teacher_model,
            "--data",
            str(data_path.resolve()),
            "--output",
            str(output_dir),
            "--seed",
            str(args.seed),
            "--num_train_epochs",
            "1",
            "--max_steps",
            str(args.max_steps),
            "--save_steps",
            str(args.save_steps),
        ]
        if args.reference_model:
            command += ["--reference-model", args.reference_model]
        if args.resume:
            command += ["--resume_from_checkpoint", "auto"]
        for key in sorted(variant["overrides"]):
            command += [f"--{key}", _cli_value(variant["overrides"][key])]
        commands.append(command)
        arms.append(
            {
                "name": variant["name"],
                "data_path": str(data_path.resolve()),
                "data_sha256": _sha256(data_path),
                "rows": len(rows),
                "id_order_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
                "output_dir": str(output_dir),
                "overrides": variant["overrides"],
                "command": command,
            }
        )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "matrix": {"path": str(args.matrix.resolve()), "sha256": _sha256(args.matrix)},
        "train_config": {"path": str(config), "sha256": _sha256(config)},
        "training_code": _training_code_fingerprints(repo_root),
        "budget_contract": {
            "unit": "optimizer_updates",
            "max_steps_per_arm": args.max_steps,
            "checkpoint_save_steps": args.save_steps,
            "epochs_cap": 1,
            "seed": args.seed,
            "nproc_per_node": args.nproc_per_node,
            "same_row_ids_and_order": True,
            "note": (
                "All arms match rows, row order, and optimizer updates. Selected-token "
                "mass is only claimed matched when a matrix-specific tokenizer audit "
                "proves it; full-response/outside arms deliberately alter token mass."
            ),
        },
        "student_initialization": _fingerprint_model(args.student_model),
        "teacher": _fingerprint_model(args.teacher_model),
        "reference_model": _fingerprint_model(args.reference_model) if args.reference_model else None,
        "arms": arms,
        "execution_requested": bool(args.execute),
    }
    return plan, commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--matrix", type=Path, default=here / "ablation_matrix.json")
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--reference-model")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="uniform periodic-checkpoint interval appended to every arm (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true", help="pass resume_from_checkpoint=auto to every arm")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute arms sequentially after writing the plan (default: plan only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan, commands = build_plan(args)
    output_root = args.output_root.resolve()
    _atomic_text(
        output_root / "experiment_plan.json",
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    repo_root = Path(__file__).resolve().parents[2]
    shell = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'export PYTHONPATH={shlex.quote(str(repo_root / "src"))}:{shlex.quote(str(repo_root))}${{PYTHONPATH:+:${{PYTHONPATH}}}}',
        "",
    ]
    shell.extend(shlex.join(command) for command in commands)
    _atomic_text(output_root / "commands.sh", "\n\n".join(shell) + "\n")
    print(f"PLAN_ONLY={not args.execute}")
    print(f"plan={output_root / 'experiment_plan.json'}")
    print(f"commands={output_root / 'commands.sh'}")
    if args.execute:
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        project_paths = os.pathsep.join((str(repo_root / "src"), str(repo_root)))
        env["PYTHONPATH"] = (
            project_paths + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        )
        for arm, command in zip(plan["arms"], commands):
            arm_dir = Path(arm["output_dir"])
            arm_dir.mkdir(parents=True, exist_ok=True)
            _atomic_text(
                arm_dir / "ablation_arm.json",
                json.dumps(arm, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            subprocess.run(command, cwd=repo_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
