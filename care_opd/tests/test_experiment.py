from __future__ import annotations

import json
import sys
from pathlib import Path

from care_opd.experiment import (
    ExperimentSettings,
    Stage,
    StageRunner,
    build_plan,
)
from care_opd.models import AdapterArguments, ModelArguments
from care_opd.train_dpo import DPOArguments
from care_opd.train_opd import OPDArguments
from care_opd.train_sft import SFTArguments
from care_opd.trainer_utils import TrainingArguments, parse_dataclass_args


def _settings(tmp_path: Path, *, rounds: int = 2) -> ExperimentSettings:
    root = Path(__file__).resolve().parents[1]
    return ExperimentSettings(
        project_root=root,
        config=root / "configs" / "smoke.yaml",
        data_dir=tmp_path / "data",
        output_root=tmp_path / "runs",
        python=sys.executable,
        rounds=rounds,
        rollouts_per_round=16,
        eval_seeds=(42,),
        max_new_tokens=32,
    )


def _module_argv(stage: Stage, module: str) -> list[str]:
    command = list(stage.command)
    module_index = command.index(module)
    return command[module_index + 1 :]


def _by_name(stages: list[Stage]) -> dict[str, Stage]:
    return {stage.name: stage for stage in stages}


def test_plan_orders_formal_flow_and_uses_real_cli_flags(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stages = build_plan(settings)
    indexed = _by_name(stages)
    names = [stage.name for stage in stages]

    assert names.index("data.synthesize") < names.index("data.validate")
    assert names.index("data.validate") < names.index("baseline.clean_sft.train")
    synthesis_outputs = {
        output.name for output in indexed["data.synthesize"].outputs
    }
    assert "supervised_kd_train.jsonl" in synthesis_outputs
    assert names.index("baseline.recovery_dpo.train") < names.index(
        "online.gkd_full.round_00.collect"
    )
    assert all(
        "care_opd.benchmark" in stage.command
        for stage in stages
        if stage.phase == "evaluation" and stage.name != "evaluation.aggregate"
    )
    assert names[-1] == "evaluation.aggregate"

    sft_stages = [
        indexed["baseline.clean_sft.train"],
        indexed["baseline.recovery_sft.train"],
        indexed["baseline.direct_revision_sft.train"],
        indexed["baseline.toolreflection_sft.train"],
        indexed["baseline.seqkd.train"],
        indexed["baseline.agent_r_style.round_00.train"],
    ]
    parsed_sft = []
    for stage in sft_stages:
        parsed_sft.append(
            parse_dataclass_args(
                (ModelArguments, AdapterArguments, TrainingArguments, SFTArguments),
                _module_argv(stage, "care_opd.train_sft"),
            )
        )
    assert parsed_sft[0][-1].baseline == "clean_sft"
    assert parsed_sft[1][-1].baseline == "recovery_sft"
    assert parsed_sft[1][-1].target_field == "target"
    assert parsed_sft[2][-1].baseline == "direct_revision_sft"
    assert parsed_sft[2][-1].target_field == "revised_response"
    assert parsed_sft[3][-1].baseline == "toolreflection_sft"
    assert parsed_sft[4][-1].baseline == "seqkd"
    assert parsed_sft[5][-1].baseline == "agent_r_style"
    assert parsed_sft[5][-1].target_field == "target"

    dpo = parse_dataclass_args(
        (ModelArguments, AdapterArguments, TrainingArguments, DPOArguments),
        _module_argv(
            indexed["baseline.recovery_dpo.train"],
            "care_opd.train_dpo",
        ),
    )
    assert dpo[2].train_file.endswith("verified_pairs_train.jsonl")

    supervised_kd = parse_dataclass_args(
        (ModelArguments, AdapterArguments, TrainingArguments, OPDArguments),
        _module_argv(
            indexed["baseline.supervised_kd.train"],
            "care_opd.train_opd",
        ),
    )
    assert supervised_kd[-1].baseline_mode == "gkd_full"
    assert supervised_kd[2].train_file.endswith("supervised_kd_train.jsonl")

    for method in settings.methods:
        for round_index in range(settings.rounds):
            name = f"online.{method}.round_{round_index:02d}.train"
            parsed = parse_dataclass_args(
                (ModelArguments, AdapterArguments, TrainingArguments, OPDArguments),
                _module_argv(indexed[name], "care_opd.train_opd"),
            )
            assert parsed[-1].baseline_mode == method
            train_file = Path(parsed[2].train_file)
            assert train_file.name == "opd_examples.jsonl"
            assert train_file.parent.name == "collection"


def test_online_methods_have_matched_collection_budget_and_own_policy(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, rounds=2)
    indexed = _by_name(build_plan(settings))
    invariant_flags = (
        "--cases",
        "--max-episodes",
        "--max-steps",
        "--max-new-tokens",
        "--temperature",
        "--top-p",
        "--seed",
    )

    for round_index in range(settings.rounds):
        arguments_by_method: dict[str, list[str]] = {}
        for method in settings.methods:
            stage = indexed[
                f"online.{method}.round_{round_index:02d}.collect"
            ]
            arguments_by_method[method] = _module_argv(
                stage, "care_opd.rollout"
            )
        reference = arguments_by_method[settings.methods[0]]
        for arguments in arguments_by_method.values():
            for flag in invariant_flags:
                assert arguments[arguments.index(flag) + 1] == reference[
                    reference.index(flag) + 1
                ]

        # Data stays method-specific because each file was generated by that
        # method's current policy, while the collection schedule is matched.
        train_files = {
            _module_argv(
                indexed[
                    f"online.{method}.round_{round_index:02d}.train"
                ],
                "care_opd.train_opd",
            )[
                _module_argv(
                    indexed[
                        f"online.{method}.round_{round_index:02d}.train"
                    ],
                    "care_opd.train_opd",
                ).index("--data")
                + 1
            ]
            for method in settings.methods
        }
        assert len(train_files) == len(settings.methods)

    for stage in indexed.values():
        if stage.name.endswith(".collect"):
            output_names = {path.name for path in stage.outputs}
            assert "revision_examples.jsonl" in output_names
            assert "verified_sft_examples.jsonl" in output_names
            assert "verified_kd_examples.jsonl" in output_names


def test_agent_r_recollects_from_merged_current_policy(tmp_path: Path) -> None:
    indexed = _by_name(build_plan(_settings(tmp_path, rounds=2)))
    round_zero_train = indexed["baseline.agent_r_style.round_00.train"]
    round_one_collect = indexed["baseline.agent_r_style.round_01.collect"]
    train_arguments = _module_argv(round_zero_train, "care_opd.train_sft")
    collect_arguments = _module_argv(round_one_collect, "care_opd.rollout")

    train_data = Path(train_arguments[train_arguments.index("--data") + 1])
    round_one_model = Path(
        collect_arguments[collect_arguments.index("--model") + 1]
    )
    assert train_data.parts[-3:] == (
        "round_00",
        "collection",
        "revision_examples.jsonl",
    )
    assert round_one_model.parts[-3:] == ("round_00", "train", "merged")


def test_training_stage_auto_resumes_and_marker_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "training"
    (output_dir / "checkpoint-00000010").mkdir(parents=True)
    final_metadata = output_dir / "final" / "run_metadata.json"
    marker = tmp_path / "markers" / "train.json"
    stage = Stage(
        name="train",
        phase="online",
        command=("trainer", "--output", str(output_dir)),
        outputs=(final_metadata,),
        marker=marker,
        resumable_output_dir=output_dir,
    )
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        assert cwd == tmp_path
        assert check is True
        calls.append(list(command))
        final_metadata.parent.mkdir(parents=True)
        final_metadata.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("care_opd.experiment.subprocess.run", fake_run)
    runner = StageRunner(project_root=tmp_path)
    assert runner.run(stage) == "executed"
    assert calls == [
        [
            "trainer",
            "--output",
            str(output_dir),
            "--resume_from_checkpoint",
            "latest",
        ]
    ]
    assert runner.run(stage) == "skipped"
    assert len(calls) == 1
    assert json.loads(marker.read_text(encoding="utf-8"))["fingerprint"] == (
        stage.fingerprint
    )


def test_evaluation_uses_paired_benchmark_and_inference_baselines(
    tmp_path: Path,
) -> None:
    indexed = _by_name(build_plan(_settings(tmp_path, rounds=1)))
    direct = indexed["evaluation.care_opd.test.seed_42.benchmark"]
    reflexion = indexed["evaluation.reflexion.test.seed_42.benchmark"]
    paladin = indexed["evaluation.paladin_retrieval.test.seed_42.benchmark"]

    for stage in (direct, reflexion, paladin):
        arguments = _module_argv(stage, "care_opd.benchmark")
        assert arguments[arguments.index("--cases") + 1].endswith(
            "cases_test.jsonl"
        )
        assert arguments[arguments.index("--recovery-file") + 1].endswith(
            "recovery_test.jsonl"
        )
        assert arguments[arguments.index("--prefix-plan") + 1].endswith(
            "paired_prefix_plan_test.jsonl"
        )
        assert arguments[arguments.index("--conditions") + 1] == (
            "clean,critical,nuisance"
        )

    for method in ("care_opd", "reflexion", "paladin_retrieval"):
        dev = indexed[f"evaluation.{method}.dev.seed_42.benchmark"]
        arguments = _module_argv(dev, "care_opd.benchmark")
        assert arguments[arguments.index("--cases") + 1].endswith(
            "cases_dev.jsonl"
        )
        assert arguments[arguments.index("--split") + 1] == "dev"

    reflexion_args = _module_argv(reflexion, "care_opd.benchmark")
    paladin_args = _module_argv(paladin, "care_opd.benchmark")
    assert reflexion_args[reflexion_args.index("--policy-mode") + 1] == (
        "reflexion"
    )
    assert paladin_args[paladin_args.index("--policy-mode") + 1] == (
        "paladin_retrieval"
    )
    recovery = indexed["baseline.recovery_sft.train"]
    recovery_output = _module_argv(
        recovery, "care_opd.train_sft"
    )
    recovery_output = recovery_output[recovery_output.index("--output") + 1]
    assert paladin_args[paladin_args.index("--adapter") + 1] == (
        str(Path(recovery_output) / "final")
    )
    aggregate = _module_argv(
        indexed["evaluation.aggregate"],
        "care_opd.evaluate",
    )
    assert aggregate[aggregate.index("--baseline") + 1] == "auto"
