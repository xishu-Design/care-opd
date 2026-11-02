"""Resumable orchestration for the formal CARE-OPD server experiment.

This module deliberately contains integration logic only.  It builds commands
for the public data, training, rollout, and evaluation entrypoints, records
stage markers, merges PEFT adapters between online rounds, and routes paired
benchmark archives into :mod:`care_opd.evaluate`.

The formal comparison uses method-specific on-policy rollouts with an identical
case order, episode budget, decoding configuration, and seed schedule.  Thus
GKD-full, TrOPD-full, and CARE-OPD see data from their own current policies
without silently giving one method more environment interactions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


FORMAL_STUDENT = "Qwen/Qwen3.5-4B"
FORMAL_TEACHER = "Qwen/Qwen3.5-9B"
ONLINE_METHODS = ("gkd_full", "tropd_full", "care_opd")
EVALUATION_SPLITS = ("dev", "test")
PHASES = ("data", "baselines", "online", "evaluation", "all")


@dataclass(frozen=True)
class ExperimentSettings:
    project_root: Path
    config: Path
    data_dir: Path
    output_root: Path
    student: str = FORMAL_STUDENT
    teacher: str = FORMAL_TEACHER
    python: str = sys.executable
    nproc_per_node: int = 1
    rounds: int = 3
    rollouts_per_round: int = 12_000
    train_seed: int = 42
    eval_seeds: tuple[int, ...] = (42, 43, 44)
    max_steps: int = 6
    max_new_tokens: int = 384
    temperature: float = 0.9
    top_p: float = 0.95
    epochs_per_round: int = 1
    methods: tuple[str, ...] = ONLINE_METHODS
    eval_baseline: str = "auto"
    eval_max_episodes: int = -1
    torch_dtype: str = "bfloat16"
    use_lora: bool = True
    use_qlora: bool = False
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.nproc_per_node < 1:
            raise ValueError("nproc_per_node must be positive")
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if self.rollouts_per_round == 0 or self.rollouts_per_round < -1:
            raise ValueError("rollouts_per_round must be -1 or positive")
        if self.max_steps < 1 or self.max_new_tokens < 1:
            raise ValueError("generation limits must be positive")
        if not self.eval_seeds:
            raise ValueError("at least one evaluation seed is required")
        if self.use_qlora and not self.use_lora:
            raise ValueError("use_qlora requires use_lora")
        unknown = sorted(set(self.methods) - set(ONLINE_METHODS))
        if unknown:
            raise ValueError(f"unsupported online methods: {unknown}")


@dataclass(frozen=True)
class Stage:
    name: str
    phase: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    marker: Path
    resumable_output_dir: Optional[Path] = None
    adopt_existing: bool = True

    @property
    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "command": list(self.command),
            "outputs": [str(path) for path in self.outputs],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "command": list(self.command),
            "command_shell": shlex.join(self.command),
            "outputs": [str(path) for path in self.outputs],
            "marker": str(self.marker),
            "resumable_output_dir": (
                None
                if self.resumable_output_dir is None
                else str(self.resumable_output_dir)
            ),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ModelArtifact:
    model: str
    adapter: Optional[str] = None


def _stage_marker(settings: ExperimentSettings, name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return settings.output_root / ".stages" / f"{slug}.json"


def _module_command(
    settings: ExperimentSettings,
    module: str,
    arguments: Iterable[str],
    *,
    distributed: bool = False,
) -> tuple[str, ...]:
    command = [settings.python]
    if distributed and settings.nproc_per_node > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={settings.nproc_per_node}",
                "-m",
                module,
            ]
        )
    else:
        command.extend(["-m", module])
    command.extend(str(value) for value in arguments)
    return tuple(command)


def _stage(
    settings: ExperimentSettings,
    *,
    name: str,
    phase: str,
    module: str,
    arguments: Iterable[str],
    outputs: Iterable[Path],
    distributed: bool = False,
    resumable_output_dir: Optional[Path] = None,
    adopt_existing: bool = True,
) -> Stage:
    return Stage(
        name=name,
        phase=phase,
        command=_module_command(
            settings,
            module,
            arguments,
            distributed=distributed,
        ),
        outputs=tuple(path.resolve() for path in outputs),
        marker=_stage_marker(settings, name).resolve(),
        resumable_output_dir=(
            None
            if resumable_output_dir is None
            else resumable_output_dir.resolve()
        ),
        adopt_existing=adopt_existing,
    )


def _training_arguments(
    settings: ExperimentSettings,
    *,
    model: str,
    data: Path,
    output: Path,
    epochs: int,
) -> list[str]:
    return [
        "--config",
        str(settings.config),
        "--model",
        model,
        "--data",
        str(data),
        "--output",
        str(output),
        "--seed",
        str(settings.train_seed),
        "--num_train_epochs",
        str(epochs),
        "--use_lora",
        str(settings.use_lora).lower(),
        "--use_qlora",
        str(settings.use_qlora).lower(),
        "--trust_remote_code",
        str(settings.trust_remote_code).lower(),
    ]


def _collection_arguments(
    settings: ExperimentSettings,
    *,
    cases: Path,
    output_dir: Path,
    policy: ModelArtifact,
    seed: int,
    max_episodes: Optional[int] = None,
    greedy: bool = False,
) -> list[str]:
    arguments = [
        "--cases",
        str(cases),
        "--output-dir",
        str(output_dir),
        "--model",
        policy.model,
        "--dtype",
        settings.torch_dtype,
        "--max-episodes",
        str(
            settings.rollouts_per_round
            if max_episodes is None
            else max_episodes
        ),
        "--max-steps",
        str(settings.max_steps),
        "--max-new-tokens",
        str(settings.max_new_tokens),
        "--temperature",
        str(settings.temperature),
        "--top-p",
        str(settings.top_p),
        "--seed",
        str(seed),
    ]
    if policy.adapter:
        arguments.extend(["--adapter", policy.adapter])
    if greedy:
        arguments.append("--greedy")
    if settings.trust_remote_code:
        arguments.append("--trust-remote-code")
    return arguments


def _collection_outputs(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "raw_rollouts.jsonl",
        output_dir / "opd_examples.jsonl",
        output_dir / "revision_examples.jsonl",
        output_dir / "verified_sft_examples.jsonl",
        output_dir / "verified_kd_examples.jsonl",
        output_dir / "collection_audit.json",
    )


def _synthesis_outputs(data_dir: Path) -> tuple[Path, ...]:
    names = ["manifest.json", "knowledge_seed.snapshot.json"]
    for split in ("train", "dev", "test"):
        names.extend(
            [
                f"cases_{split}.jsonl",
                f"oracle_{split}.jsonl",
                f"clean_oracle_{split}.jsonl",
                f"supervised_kd_{split}.jsonl",
                f"recovery_{split}.jsonl",
                f"full_recovery_{split}.jsonl",
                f"first_failure_revision_{split}.jsonl",
                f"transition_splice_{split}.jsonl",
                f"verified_pairs_{split}.jsonl",
                f"paired_eval_{split}.jsonl",
                f"paired_prefix_plan_{split}.jsonl",
            ]
        )
    return tuple(data_dir / name for name in names)


def _merge_stage(
    settings: ExperimentSettings,
    *,
    name: str,
    phase: str,
    base_model: str,
    adapter: Path,
    output: Path,
) -> Stage:
    arguments = [
        "merge-adapter",
        "--base-model",
        base_model,
        "--adapter",
        str(adapter),
        "--output",
        str(output),
        "--torch-dtype",
        settings.torch_dtype,
    ]
    if settings.trust_remote_code:
        arguments.append("--trust-remote-code")
    return _stage(
        settings,
        name=name,
        phase=phase,
        module="care_opd.experiment",
        arguments=arguments,
        outputs=(output / "config.json", output / "merge_metadata.json"),
    )


def _after_training(
    settings: ExperimentSettings,
    *,
    stages: list[Stage],
    phase: str,
    name: str,
    base_model: str,
    training_output: Path,
) -> ModelArtifact:
    final = training_output / "final"
    if not settings.use_lora:
        return ModelArtifact(model=str(final))
    merged = training_output / "merged"
    stages.append(
        _merge_stage(
            settings,
            name=f"{name}.merge",
            phase=phase,
            base_model=base_model,
            adapter=final,
            output=merged,
        )
    )
    return ModelArtifact(model=str(merged))


def build_plan(
    settings: ExperimentSettings,
    phase: str = "all",
) -> list[Stage]:
    """Build the deterministic formal pipeline without executing it."""

    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    stages: list[Stage] = []

    # Data synthesis and executable leakage/oracle validation.
    stages.append(
        _stage(
            settings,
            name="data.synthesize",
            phase="data",
            module="care_opd.synthesis",
            arguments=(
                "--config",
                str(settings.config),
                "--output",
                str(settings.data_dir),
                "--overwrite",
            ),
            outputs=_synthesis_outputs(settings.data_dir),
        )
    )
    validation_report = settings.output_root / "audits" / "data_validation.json"
    stages.append(
        _stage(
            settings,
            name="data.validate",
            phase="data",
            module="care_opd.verifier",
            arguments=(
                "--data-dir",
                str(settings.data_dir),
                "--report",
                str(validation_report),
            ),
            outputs=(validation_report,),
            adopt_existing=False,
        )
    )

    # One common warm start is merged and becomes the exact initialization for
    # every trainable baseline and every online objective.
    warm_output = settings.output_root / "baselines" / "clean_sft"
    warm_arguments = _training_arguments(
        settings,
        model=settings.student,
        data=settings.data_dir / "clean_oracle_train.jsonl",
        output=warm_output,
        epochs=1,
    )
    warm_arguments.extend(["--baseline", "clean_sft"])
    stages.append(
        _stage(
            settings,
            name="baseline.clean_sft.train",
            phase="baselines",
            module="care_opd.train_sft",
            arguments=warm_arguments,
            outputs=(warm_output / "final" / "run_metadata.json",),
            distributed=True,
            resumable_output_dir=warm_output,
        )
    )
    warm_artifact = _after_training(
        settings,
        stages=stages,
        phase="baselines",
        name="baseline.clean_sft",
        base_model=settings.student,
        training_output=warm_output,
    )

    baseline_artifacts: dict[str, ModelArtifact] = {
        "frozen": ModelArtifact(settings.student),
        "clean_sft": warm_artifact,
    }
    sft_specs = (
        ("recovery_sft", "full_recovery_train.jsonl", "target"),
        (
            "direct_revision_sft",
            "first_failure_revision_train.jsonl",
            "revised_response",
        ),
    )
    for baseline, filename, target_field in sft_specs:
        output = settings.output_root / "baselines" / baseline
        arguments = _training_arguments(
            settings,
            model=warm_artifact.model,
            data=settings.data_dir / filename,
            output=output,
            epochs=1,
        )
        # Both generated views expose an explicit `target`; specifying it also
        # permits a descriptive baseline label in SFTArguments.
        arguments.extend(
            [
                "--baseline",
                baseline,
                "--target_field",
                target_field,
            ]
        )
        stages.append(
            _stage(
                settings,
                name=f"baseline.{baseline}.train",
                phase="baselines",
                module="care_opd.train_sft",
                arguments=arguments,
                outputs=(output / "final" / "run_metadata.json",),
                distributed=True,
                resumable_output_dir=output,
            )
        )
        if settings.use_lora:
            baseline_artifacts[baseline] = ModelArtifact(
                model=warm_artifact.model,
                adapter=str(output / "final"),
            )
        else:
            baseline_artifacts[baseline] = ModelArtifact(
                model=str(output / "final")
            )

    dpo_output = settings.output_root / "baselines" / "recovery_dpo"
    dpo_arguments = _training_arguments(
        settings,
        model=warm_artifact.model,
        data=settings.data_dir / "verified_pairs_train.jsonl",
        output=dpo_output,
        epochs=1,
    )
    if not settings.use_lora:
        dpo_arguments.extend(["--reference-model", warm_artifact.model])
    stages.append(
        _stage(
            settings,
            name="baseline.recovery_dpo.train",
            phase="baselines",
            module="care_opd.train_dpo",
            arguments=dpo_arguments,
            outputs=(dpo_output / "final" / "run_metadata.json",),
            distributed=True,
            resumable_output_dir=dpo_output,
        )
    )
    if settings.use_lora:
        baseline_artifacts["recovery_dpo"] = ModelArtifact(
            model=warm_artifact.model,
            adapter=str(dpo_output / "final"),
        )
    else:
        baseline_artifacts["recovery_dpo"] = ModelArtifact(
            model=str(dpo_output / "final")
        )

    # SeqKD uses a fixed greedy teacher-policy archive.  Supervised KD instead
    # uses fixed host-verified oracle sequences and obtains dense teacher
    # distributions on those prefixes.  The rollout exporter also retains a
    # matched hard/soft diagnostic view, but that is not the main baseline.
    teacher_collection = (
        settings.output_root / "baselines" / "teacher_policy_collection"
    )
    stages.append(
        _stage(
            settings,
            name="baseline.teacher_policy.collect",
            phase="baselines",
            module="care_opd.rollout",
            arguments=_collection_arguments(
                settings,
                cases=settings.data_dir / "cases_train.jsonl",
                output_dir=teacher_collection,
                policy=ModelArtifact(settings.teacher),
                seed=settings.train_seed,
                greedy=True,
            ),
            outputs=_collection_outputs(teacher_collection),
        )
    )

    seqkd_output = settings.output_root / "baselines" / "seqkd"
    seqkd_arguments = _training_arguments(
        settings,
        model=warm_artifact.model,
        data=teacher_collection / "verified_sft_examples.jsonl",
        output=seqkd_output,
        epochs=1,
    )
    seqkd_arguments.extend(
        ["--baseline", "seqkd", "--target_field", "target"]
    )
    stages.append(
        _stage(
            settings,
            name="baseline.seqkd.train",
            phase="baselines",
            module="care_opd.train_sft",
            arguments=seqkd_arguments,
            outputs=(seqkd_output / "final" / "run_metadata.json",),
            distributed=True,
            resumable_output_dir=seqkd_output,
        )
    )
    baseline_artifacts["seqkd"] = (
        ModelArtifact(
            model=warm_artifact.model,
            adapter=str(seqkd_output / "final"),
        )
        if settings.use_lora
        else ModelArtifact(model=str(seqkd_output / "final"))
    )

    supervised_kd_output = settings.output_root / "baselines" / "supervised_kd"
    supervised_kd_arguments = _training_arguments(
        settings,
        model=warm_artifact.model,
        data=settings.data_dir / "supervised_kd_train.jsonl",
        output=supervised_kd_output,
        epochs=1,
    )
    supervised_kd_arguments.extend(
        [
            "--teacher",
            settings.teacher,
            "--baseline_mode",
            "gkd_full",
            "--max_response_tokens",
            str(settings.max_new_tokens),
        ]
    )
    stages.append(
        _stage(
            settings,
            name="baseline.supervised_kd.train",
            phase="baselines",
            module="care_opd.train_opd",
            arguments=supervised_kd_arguments,
            outputs=(supervised_kd_output / "final" / "run_metadata.json",),
            distributed=True,
            resumable_output_dir=supervised_kd_output,
        )
    )
    baseline_artifacts["supervised_kd"] = (
        ModelArtifact(
            model=warm_artifact.model,
            adapter=str(supervised_kd_output / "final"),
        )
        if settings.use_lora
        else ModelArtifact(model=str(supervised_kd_output / "final"))
    )

    # Agent-R-style is genuinely iterative: each round recollects from its
    # current policy, retains the first verifier-localized failure per episode,
    # trains on the verified revision, and merges before recollection.
    agent_current = warm_artifact
    for round_index in range(settings.rounds):
        round_name = f"round_{round_index:02d}"
        round_root = (
            settings.output_root
            / "baselines"
            / "agent_r_style"
            / round_name
        )
        collection_dir = round_root / "collection"
        round_seed = settings.train_seed + round_index * 100_003
        stages.append(
            _stage(
                settings,
                name=f"baseline.agent_r_style.{round_name}.collect",
                phase="baselines",
                module="care_opd.rollout",
                arguments=_collection_arguments(
                    settings,
                    cases=settings.data_dir / "cases_train.jsonl",
                    output_dir=collection_dir,
                    policy=agent_current,
                    seed=round_seed,
                ),
                outputs=_collection_outputs(collection_dir),
            )
        )

        # ToolReflection is the one-round first-failure revision control over
        # the exact same round-0 archive used to initialize iterative Agent-R.
        if round_index == 0:
            toolreflection_output = (
                settings.output_root / "baselines" / "toolreflection_sft"
            )
            toolreflection_arguments = _training_arguments(
                settings,
                model=warm_artifact.model,
                data=collection_dir / "revision_examples.jsonl",
                output=toolreflection_output,
                epochs=1,
            )
            toolreflection_arguments.extend(
                [
                    "--baseline",
                    "toolreflection_sft",
                    "--target_field",
                    "target",
                ]
            )
            stages.append(
                _stage(
                    settings,
                    name="baseline.toolreflection_sft.train",
                    phase="baselines",
                    module="care_opd.train_sft",
                    arguments=toolreflection_arguments,
                    outputs=(
                        toolreflection_output / "final" / "run_metadata.json",
                    ),
                    distributed=True,
                    resumable_output_dir=toolreflection_output,
                )
            )
            baseline_artifacts["toolreflection_sft"] = (
                ModelArtifact(
                    model=warm_artifact.model,
                    adapter=str(toolreflection_output / "final"),
                )
                if settings.use_lora
                else ModelArtifact(
                    model=str(toolreflection_output / "final")
                )
            )

        agent_training_output = round_root / "train"
        agent_arguments = _training_arguments(
            settings,
            model=agent_current.model,
            data=collection_dir / "revision_examples.jsonl",
            output=agent_training_output,
            epochs=settings.epochs_per_round,
        )
        agent_arguments.extend(
            [
                "--baseline",
                "agent_r_style",
                "--target_field",
                "target",
            ]
        )
        stages.append(
            _stage(
                settings,
                name=f"baseline.agent_r_style.{round_name}.train",
                phase="baselines",
                module="care_opd.train_sft",
                arguments=agent_arguments,
                outputs=(
                    agent_training_output / "final" / "run_metadata.json",
                ),
                distributed=True,
                resumable_output_dir=agent_training_output,
            )
        )
        agent_current = _after_training(
            settings,
            stages=stages,
            phase="baselines",
            name=f"baseline.agent_r_style.{round_name}",
            base_model=agent_current.model,
            training_output=agent_training_output,
        )
    baseline_artifacts["agent_r_style"] = agent_current

    # Each objective follows its own current policy.  The case file, prefix
    # order, episode count, sampling parameters, and round seed are identical,
    # which is the fair comparison possible without making a method off-policy.
    online_artifacts: dict[str, ModelArtifact] = {}
    for method in settings.methods:
        current = warm_artifact
        for round_index in range(settings.rounds):
            round_name = f"round_{round_index:02d}"
            round_root = settings.output_root / "online" / method / round_name
            collection_dir = round_root / "collection"
            round_seed = settings.train_seed + round_index * 100_003
            collect_arguments = _collection_arguments(
                settings,
                cases=settings.data_dir / "cases_train.jsonl",
                output_dir=collection_dir,
                policy=current,
                seed=round_seed,
            )
            stages.append(
                _stage(
                    settings,
                    name=f"online.{method}.{round_name}.collect",
                    phase="online",
                    module="care_opd.rollout",
                    arguments=collect_arguments,
                    outputs=_collection_outputs(collection_dir),
                )
            )

            training_output = round_root / "train"
            opd_arguments = _training_arguments(
                settings,
                model=current.model,
                data=collection_dir / "opd_examples.jsonl",
                output=training_output,
                epochs=settings.epochs_per_round,
            )
            opd_arguments.extend(
                [
                    "--teacher",
                    settings.teacher,
                    "--baseline_mode",
                    method,
                    "--max_response_tokens",
                    str(settings.max_new_tokens),
                ]
            )
            if not settings.use_lora and method == "care_opd":
                opd_arguments.extend(["--reference-model", current.model])
            stages.append(
                _stage(
                    settings,
                    name=f"online.{method}.{round_name}.train",
                    phase="online",
                    module="care_opd.train_opd",
                    arguments=opd_arguments,
                    outputs=(training_output / "final" / "run_metadata.json",),
                    distributed=True,
                    resumable_output_dir=training_output,
                )
            )
            current = _after_training(
                settings,
                stages=stages,
                phase="online",
                name=f"online.{method}.{round_name}",
                base_model=current.model,
                training_output=training_output,
            )
        online_artifacts[method] = current

    # Formal evaluation must come from the paired benchmark runner: it applies
    # the preregistered equal-cost critical/nuisance prefixes to the exact same
    # hidden world and also emits the clean branch.  Reflexion is inference-
    # time prompting over the frozen student.  PALADIN retrieval is applied to
    # the recovery-SFT checkpoint, matching its train-plus-retrieval recipe.
    evaluation_models = {
        **baseline_artifacts,
        **online_artifacts,
        "reflexion": ModelArtifact(settings.student),
        "paladin_retrieval": baseline_artifacts["recovery_sft"],
    }
    policy_modes = {
        "reflexion": "reflexion",
        "paladin_retrieval": "paladin_retrieval",
    }
    evaluation_rows: list[Path] = []
    for method, artifact in evaluation_models.items():
        for split in EVALUATION_SPLITS:
            for seed in settings.eval_seeds:
                evaluation_root = (
                    settings.output_root
                    / "evaluation"
                    / method
                    / split
                    / f"seed_{seed}"
                )
                benchmark_rows = evaluation_root / "benchmark_rows.jsonl"
                benchmark_arguments = [
                    "--cases",
                    str(settings.data_dir / f"cases_{split}.jsonl"),
                    "--recovery-file",
                    str(settings.data_dir / f"recovery_{split}.jsonl"),
                    "--prefix-plan",
                    str(
                        settings.data_dir
                        / f"paired_prefix_plan_{split}.jsonl"
                    ),
                    "--output",
                    str(benchmark_rows),
                    "--model",
                    artifact.model,
                    "--method",
                    method,
                    "--split",
                    split,
                    "--conditions",
                    "clean,critical,nuisance",
                    "--policy-mode",
                    policy_modes.get(method, "direct"),
                    "--dtype",
                    settings.torch_dtype,
                    "--max-cases",
                    str(settings.eval_max_episodes),
                    "--max-total-steps",
                    str(settings.max_steps),
                    "--max-new-tokens",
                    str(settings.max_new_tokens),
                    "--temperature",
                    str(settings.temperature),
                    "--top-p",
                    str(settings.top_p),
                    "--seed",
                    str(seed),
                ]
                if artifact.adapter:
                    benchmark_arguments.extend(["--adapter", artifact.adapter])
                stages.append(
                    _stage(
                        settings,
                        name=(
                            f"evaluation.{method}.{split}."
                            f"seed_{seed}.benchmark"
                        ),
                        phase="evaluation",
                        module="care_opd.benchmark",
                        arguments=benchmark_arguments,
                        outputs=(benchmark_rows,),
                    )
                )
                evaluation_rows.append(benchmark_rows)

    report_dir = settings.output_root / "reports"
    aggregate_arguments = [
        "--input",
        *(str(path) for path in evaluation_rows),
        "--output-dir",
        str(report_dir),
        "--baseline",
        settings.eval_baseline,
        "--seed",
        str(settings.train_seed),
    ]
    stages.append(
        _stage(
            settings,
            name="evaluation.aggregate",
            phase="evaluation",
            module="care_opd.evaluate",
            arguments=aggregate_arguments,
            outputs=(
                report_dir / "main_table.csv",
                report_dir / "statistical_tests.json",
            ),
        )
    )

    _validate_plan(stages)
    if phase == "all":
        return stages
    return [stage for stage in stages if stage.phase == phase]


def _validate_plan(stages: Sequence[Stage]) -> None:
    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("stage names must be unique")
    markers = [stage.marker for stage in stages]
    if len(markers) != len(set(markers)):
        raise ValueError("stage markers must be unique")
    for stage in stages:
        if not stage.outputs:
            raise ValueError(f"{stage.name} has no completion artifact")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


class StageRunner:
    """Execute a plan with command-fingerprinted, output-checked markers."""

    def __init__(
        self,
        *,
        project_root: Path,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        self.project_root = project_root
        self.dry_run = dry_run
        self.force = force

    @staticmethod
    def _outputs_exist(stage: Stage) -> bool:
        return all(path.exists() for path in stage.outputs)

    @staticmethod
    def _checkpoint_exists(output_dir: Path) -> bool:
        return any(
            child.is_dir() and re.fullmatch(r"checkpoint-\d+", child.name)
            for child in output_dir.glob("checkpoint-*")
        )

    def _write_marker(
        self,
        stage: Stage,
        *,
        disposition: str,
        executed_command: Sequence[str],
    ) -> None:
        _atomic_json(
            stage.marker,
            {
                "schema_version": "care-opd-stage-v1",
                "stage": stage.name,
                "phase": stage.phase,
                "fingerprint": stage.fingerprint,
                "disposition": disposition,
                "command": list(executed_command),
                "outputs": [str(path) for path in stage.outputs],
            },
        )

    def run(self, stage: Stage) -> str:
        print(f"[care-run] {stage.name}")
        print(f"[care-run] $ {shlex.join(stage.command)}")
        if self.dry_run:
            return "dry-run"

        if stage.marker.is_file() and not self.force:
            payload = json.loads(stage.marker.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != stage.fingerprint:
                raise RuntimeError(
                    f"{stage.marker} belongs to a different command; "
                    "choose a new output root or pass --force"
                )
            if not self._outputs_exist(stage):
                missing = [str(path) for path in stage.outputs if not path.exists()]
                raise FileNotFoundError(
                    f"stage marker exists but outputs are missing: {missing}"
                )
            print("[care-run] complete marker and outputs found; skipping")
            return "skipped"

        if (
            not self.force
            and not stage.marker.exists()
            and stage.adopt_existing
            and self._outputs_exist(stage)
        ):
            self._write_marker(
                stage,
                disposition="adopted-existing",
                executed_command=stage.command,
            )
            print("[care-run] complete outputs found; adopted and skipped")
            return "adopted"

        command = list(stage.command)
        if (
            stage.resumable_output_dir is not None
            and self._checkpoint_exists(stage.resumable_output_dir)
            and "--resume_from_checkpoint" not in command
        ):
            command.extend(["--resume_from_checkpoint", "latest"])
            print("[care-run] incomplete training output found; resuming latest checkpoint")

        subprocess.run(command, cwd=self.project_root, check=True)
        if not self._outputs_exist(stage):
            missing = [str(path) for path in stage.outputs if not path.exists()]
            raise RuntimeError(
                f"{stage.name} returned successfully but did not create {missing}"
            )
        self._write_marker(
            stage,
            disposition="executed",
            executed_command=command,
        )
        return "executed"


def merge_adapter(
    *,
    base_model: str,
    adapter: Path,
    output: Path,
    torch_dtype: str,
    trust_remote_code: bool,
) -> None:
    """Merge a completed LoRA/QLoRA adapter into a reusable full checkpoint."""

    if output.exists():
        raise FileExistsError(
            f"{output} already exists; stage adoption should have skipped this merge"
        )
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    from peft import PeftModel

    from .models import (
        ModelArguments,
        load_causal_lm,
        load_tokenizer,
        tokenizer_fingerprint,
    )

    model_args = ModelArguments(
        student_model_name_or_path=base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = load_tokenizer(model_args)
    base = load_causal_lm(base_model, model_args, trainable=False)
    peft_model = PeftModel.from_pretrained(base, str(adapter))
    merged = peft_model.merge_and_unload()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.partial-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir()
    merged.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    _atomic_json(
        temporary / "merge_metadata.json",
        {
            "schema_version": "care-opd-merged-adapter-v1",
            "base_model": base_model,
            "adapter": str(adapter.resolve()),
            "torch_dtype": torch_dtype,
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        },
    )
    os.replace(temporary, output)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("experiment orchestration requires pyyaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be an object")
    return payload


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _settings_from_namespace(namespace: argparse.Namespace) -> ExperimentSettings:
    inferred_root = Path(__file__).resolve().parents[2]
    project_root = (
        inferred_root
        if namespace.project_root is None
        else Path(namespace.project_root).resolve()
    )
    config_path = _resolve_project_path(
        project_root,
        namespace.config or "configs/formal.yaml",
    )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = _load_yaml(config_path)
    project = _section(config, "project")
    data = _section(config, "data")
    models = _section(config, "models")
    online = _section(config, "online")
    optimization = _section(config, "optimization")
    lora = _section(config, "lora")
    evaluation = _section(config, "evaluation")

    data_dir = _resolve_project_path(
        project_root,
        namespace.data_dir or data.get("output_dir", "data/generated"),
    )
    output_root = _resolve_project_path(
        project_root,
        namespace.output_root
        or project.get("output_root", "runs/formal"),
    )
    configured_eval_seeds = evaluation.get("seeds", (42, 43, 44))
    if not isinstance(configured_eval_seeds, (list, tuple)):
        configured_eval_seeds = (42, 43, 44)
    use_lora = (
        bool(lora.get("enabled", True))
        if namespace.use_lora is None
        else namespace.use_lora
    )
    return ExperimentSettings(
        project_root=project_root,
        config=config_path,
        data_dir=data_dir,
        output_root=output_root,
        student=str(namespace.student or models.get("student", FORMAL_STUDENT)),
        teacher=str(namespace.teacher or models.get("teacher", FORMAL_TEACHER)),
        python=str(namespace.python or sys.executable),
        nproc_per_node=int(
            namespace.nproc_per_node
            or os.environ.get("NPROC_PER_NODE", "1")
        ),
        rounds=int(namespace.rounds or online.get("rounds", 3)),
        rollouts_per_round=int(
            namespace.rollouts_per_round
            or online.get("rollouts_per_round", 12_000)
        ),
        train_seed=int(namespace.seed or project.get("seed", 42)),
        eval_seeds=(
            _parse_int_csv(namespace.eval_seeds)
            if namespace.eval_seeds
            else tuple(int(seed) for seed in configured_eval_seeds)
        ),
        max_steps=int(namespace.max_steps or data.get("max_steps", 6)),
        max_new_tokens=int(
            namespace.max_new_tokens or models.get("max_new_tokens", 384)
        ),
        temperature=float(
            namespace.temperature or online.get("temperature", 0.9)
        ),
        top_p=float(namespace.top_p or online.get("top_p", 0.95)),
        epochs_per_round=int(
            namespace.epochs_per_round
            or optimization.get("epochs_per_round", 1)
        ),
        methods=(
            _parse_csv(namespace.methods)
            if namespace.methods
            else ONLINE_METHODS
        ),
        eval_baseline=str(namespace.eval_baseline or "auto"),
        eval_max_episodes=int(
            namespace.eval_max_episodes
            if namespace.eval_max_episodes is not None
            else -1
        ),
        torch_dtype=str(models.get("dtype", "bfloat16")),
        use_lora=use_lora,
        use_qlora=bool(models.get("load_in_4bit", False)),
        trust_remote_code=bool(models.get("trust_remote_code", False)),
    )


def _add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--student")
    parser.add_argument("--teacher")
    parser.add_argument("--python")
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--rollouts-per-round", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-seeds")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--epochs-per-round", type=int)
    parser.add_argument("--methods")
    parser.add_argument("--eval-baseline")
    parser.add_argument("--eval-max-episodes", type=int)
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def _plan_payload(
    settings: ExperimentSettings,
    phase: str,
    stages: Sequence[Stage],
) -> dict[str, Any]:
    settings_payload = asdict(settings)
    for key in ("project_root", "config", "data_dir", "output_root"):
        settings_payload[key] = str(settings_payload[key])
    return {
        "schema_version": "care-opd-experiment-plan-v1",
        "phase": phase,
        "settings": settings_payload,
        "fairness_contract": {
            "collection_mode": "method_specific_on_policy",
            "case_file": str(settings.data_dir / "cases_train.jsonl"),
            "case_order": "file_order",
            "episodes_per_method_round": settings.rollouts_per_round,
            "round_seed_formula": "train_seed + round_index * 100003",
            "decoding": {
                "max_steps": settings.max_steps,
                "max_new_tokens": settings.max_new_tokens,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
            },
            "initial_checkpoint": "common merged clean-SFT warm start",
            "data_isolation": "each objective collects from its own current policy",
        },
        "evaluation_contract": {
            "splits": list(EVALUATION_SPLITS),
            "conditions": ["clean", "critical", "nuisance"],
            "baseline_selection": settings.eval_baseline,
            "auto_selection_rule": (
                "maximum dev PRAR among non-care_opd methods; "
                "paired inference on test"
            ),
        },
        "stages": [stage.as_dict() for stage in stages],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute a pipeline phase")
    _add_settings_arguments(run_parser)
    run_parser.add_argument("--phase", choices=PHASES, default="all")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="ignore existing stage markers; existing artifacts are not deleted",
    )

    plan_parser = subparsers.add_parser("plan", help="print the resolved plan")
    _add_settings_arguments(plan_parser)
    plan_parser.add_argument("--phase", choices=PHASES, default="all")

    merge_parser = subparsers.add_parser(
        "merge-adapter",
        help="merge a PEFT adapter into a full checkpoint",
    )
    merge_parser.add_argument("--base-model", required=True)
    merge_parser.add_argument("--adapter", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--torch-dtype", default="bfloat16")
    merge_parser.add_argument("--trust-remote-code", action="store_true")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Keep the console entrypoint convenient: `care-run --config ...` means
    # `care-run run --config ...`.
    if arguments and arguments[0].startswith("-"):
        arguments.insert(0, "run")
    namespace = _build_parser().parse_args(arguments)

    if namespace.command == "merge-adapter":
        merge_adapter(
            base_model=namespace.base_model,
            adapter=namespace.adapter.resolve(),
            output=namespace.output.resolve(),
            torch_dtype=namespace.torch_dtype,
            trust_remote_code=namespace.trust_remote_code,
        )
        return 0
    settings = _settings_from_namespace(namespace)
    stages = build_plan(settings, namespace.phase)
    payload = _plan_payload(settings, namespace.phase, stages)
    if namespace.command == "plan":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not namespace.dry_run:
        _atomic_json(settings.output_root / "experiment_plan.json", payload)
    runner = StageRunner(
        project_root=settings.project_root,
        dry_run=namespace.dry_run,
        force=namespace.force,
    )
    for stage in stages:
        runner.run(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
