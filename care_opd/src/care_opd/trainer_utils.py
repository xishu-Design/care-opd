"""Shared configuration, JSONL I/O, logging, and resumable training helpers."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import argparse
import time
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


@dataclass
class TrainingArguments:
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    output_dir: str = "outputs/care_opd"
    seed: int = 42
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_length: int = 4096
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    save_steps: int = 250
    dataloader_num_workers: int = 0
    resume_from_checkpoint: Optional[str] = None
    report_to: str = "none"


class JsonlDataset:
    """Strict in-memory JSONL dataset with line-numbered diagnostics."""

    def __init__(self, path: str | os.PathLike[str], *, required_fields: Sequence[str] = ()):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.records: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{self.path}:{line_number}: row must be an object")
                missing = [field for field in required_fields if row.get(field) is None]
                if missing:
                    raise ValueError(
                        f"{self.path}:{line_number}: missing required fields {missing}"
                    )
                self.records.append(row)
        if not self.records:
            raise ValueError(f"{self.path} contains no usable records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_dataclass_args(
    dataclass_types: Sequence[Type[Any]],
    argv: Optional[Sequence[str]] = None,
) -> Tuple[Any, ...]:
    """Parse dataclasses from a formal nested config plus CLI overrides.

    Accepted forms include ``trainer config.yaml``, ``trainer --config
    config.yaml``, and either form followed by overrides.  Convenience aliases
    required by the server scripts are ``--model``, ``--teacher``, ``--data``,
    ``--view``, and ``--output``.
    """

    if argv is None:
        import sys

        arguments = list(sys.argv[1:])
    else:
        arguments = list(argv)

    config_path: Optional[Path] = None
    if arguments and not arguments[0].startswith("-") and Path(arguments[0]).suffix.lower() in {
        ".json",
        ".yaml",
        ".yml",
    }:
        config_path = Path(arguments.pop(0)).resolve()
    elif "--config" in arguments:
        index = arguments.index("--config")
        if index + 1 >= len(arguments):
            raise ValueError("--config requires a path")
        config_path = Path(arguments[index + 1]).resolve()
        del arguments[index : index + 2]

    nested: Dict[str, Any] = {}
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if config_path.suffix.lower() == ".json":
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML config parsing requires pyyaml") from exc
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be an object")
        nested = payload

    known_fields: Dict[str, Tuple[Type[Any], Any, Any]] = {}
    for dataclass_type in dataclass_types:
        hints = get_type_hints(dataclass_type)
        for item in fields(dataclass_type):
            if item.name in known_fields:
                raise ValueError(f"duplicate CLI field across dataclasses: {item.name}")
            known_fields[item.name] = (
                dataclass_type,
                item,
                hints.get(item.name, str),
            )

    defaults = _formal_config_defaults(nested, known_fields.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="student_model_name_or_path")
    parser.add_argument("--teacher", dest="teacher_model_name_or_path")
    parser.add_argument("--reference-model", dest="reference_model_name_or_path")
    parser.add_argument("--data", dest="train_file")
    parser.add_argument("--view")
    parser.add_argument("--output", dest="output_dir")
    for name, (_, item, annotation) in known_fields.items():
        default = defaults.get(name, _field_default(item))
        converter = _argument_converter(annotation, default)
        if isinstance(default, bool):
            parser.add_argument(
                f"--{name}",
                nargs="?",
                const=True,
                default=default,
                type=_parse_bool,
            )
            parser.add_argument(
                f"--no-{name.replace('_', '-')}",
                dest=name,
                action="store_false",
            )
        else:
            parser.add_argument(f"--{name}", default=default, type=converter)
    namespace = vars(parser.parse_args(arguments))

    view = namespace.pop("view", None)
    if view and namespace.get("train_file") is None:
        generated_root = (
            nested.get("data", {}).get("output_dir", "data/generated")
            if isinstance(nested.get("data", {}), Mapping)
            else "data/generated"
        )
        base = Path(generated_root)
        if config_path is not None and not base.is_absolute():
            base = config_path.parent.parent / base
        namespace["train_file"] = str(base / "views" / f"{view}.jsonl")

    instances = []
    for dataclass_type in dataclass_types:
        names = {item.name for item in fields(dataclass_type)}
        values = {name: namespace[name] for name in names}
        instances.append(dataclass_type(**values))
    return tuple(instances)


def _field_default(item: Any) -> Any:
    if item.default is not MISSING:
        return item.default
    if item.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return item.default_factory()
    return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def _argument_converter(annotation: Any, default: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {Union, getattr(__import__("types"), "UnionType", object())}:
        candidates = [candidate for candidate in get_args(annotation) if candidate is not type(None)]
        if candidates:
            annotation = candidates[0]
    if annotation in {str, int, float}:
        return annotation
    if isinstance(default, (str, int, float)):
        return type(default)
    return str


def _formal_config_defaults(
    config: Mapping[str, Any],
    allowed_fields: Iterable[str],
) -> Dict[str, Any]:
    """Map the repository's formal nested schema to trainer dataclass fields."""

    allowed = set(allowed_fields)
    defaults: Dict[str, Any] = {}

    def assign(field_name: str, value: Any) -> None:
        if field_name in allowed and value is not None:
            defaults[field_name] = value

    project = config.get("project", {}) if isinstance(config.get("project", {}), Mapping) else {}
    data = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    models = config.get("models", {}) if isinstance(config.get("models", {}), Mapping) else {}
    optimization = (
        config.get("optimization", {})
        if isinstance(config.get("optimization", {}), Mapping)
        else {}
    )
    lora = config.get("lora", {}) if isinstance(config.get("lora", {}), Mapping) else {}

    assign("seed", project.get("seed"))
    assign("output_dir", project.get("output_root"))
    assign("student_model_name_or_path", models.get("student"))
    assign("teacher_model_name_or_path", models.get("teacher"))
    assign("torch_dtype", models.get("dtype"))
    assign("attn_implementation", models.get("attn_implementation"))
    assign("use_qlora", models.get("load_in_4bit"))
    assign("gradient_checkpointing", models.get("gradient_checkpointing"))
    assign("max_length", models.get("max_length"))

    assign("learning_rate", optimization.get("learning_rate"))
    assign("warmup_ratio", optimization.get("warmup_ratio"))
    assign("weight_decay", optimization.get("weight_decay"))
    assign("max_grad_norm", optimization.get("max_grad_norm"))
    assign("gradient_accumulation_steps", optimization.get("gradient_accumulation_steps"))
    assign("per_device_train_batch_size", optimization.get("per_device_batch_size"))
    assign("save_steps", optimization.get("save_steps"))
    assign("num_train_epochs", optimization.get("epochs_per_round"))
    assign("temperature", optimization.get("temperature"))
    assign("js_beta", optimization.get("jsd_beta"))
    assign("hard_oracle_ce_weight", optimization.get("hard_oracle_ce_weight"))
    assign(
        "max_teacher_top_log_ratio",
        optimization.get("trust_log_ratio"),
    )
    assign("min_teacher_margin", optimization.get("min_teacher_margin"))
    assign("response_chunk_size", optimization.get("response_chunk_size"))
    divergence = optimization.get("divergence")
    if divergence is not None:
        assign("kl_mode", {"jsd": "js", "rkl": "reverse_kl", "fkl": "forward_kl"}.get(divergence, divergence))
    if optimization.get("causal_mask") is not None:
        assign("mask_mode", "span" if optimization["causal_mask"] else "full")
    assign("use_teacher_trust", optimization.get("trust_mask"))
    assign("retention_coefficient", optimization.get("clean_retention_weight"))
    assign("max_response_tokens", models.get("max_new_tokens"))

    assign("use_lora", lora.get("enabled"))
    assign("lora_rank", lora.get("rank"))
    assign("lora_alpha", lora.get("alpha"))
    assign("lora_dropout", lora.get("dropout"))
    targets = lora.get("target_modules")
    if isinstance(targets, list):
        assign("lora_target_modules", ",".join(str(item) for item in targets))
    return defaults


def create_accelerator(training_args: TrainingArguments) -> Any:
    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise RuntimeError("training requires accelerate") from exc
    if training_args.bf16 and training_args.fp16:
        raise ValueError("bf16 and fp16 are mutually exclusive")
    precision = "bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "no")
    return Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=precision,
        log_with=None if training_args.report_to == "none" else training_args.report_to,
        project_dir=training_args.output_dir,
    )


def build_optimizer(model: Any, args: TrainingArguments) -> Any:
    import torch

    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    groups = [
        {"params": decay_parameters, "weight_decay": args.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def total_update_steps(dataloader_length: int, args: TrainingArguments) -> int:
    updates_per_epoch = math.ceil(
        dataloader_length / max(args.gradient_accumulation_steps, 1)
    )
    if args.max_steps > 0:
        return args.max_steps
    return max(updates_per_epoch * args.num_train_epochs, 1)


def build_scheduler(optimizer: Any, total_steps: int, args: TrainingArguments) -> Any:
    try:
        from transformers import get_scheduler
    except ImportError as exc:
        raise RuntimeError("scheduler construction requires transformers") from exc
    warmup_steps = int(total_steps * args.warmup_ratio)
    return get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


class MetricLogger:
    def __init__(self, path: str | os.PathLike[str], *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _format_progress_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(seconds + 0.5)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_progress_metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


class ProgressReporter:
    """Flush rank-safe, nohup-friendly progress lines with elapsed time and ETA."""

    def __init__(
        self,
        *,
        stage: str,
        total_steps: int,
        initial_step: int = 0,
        enabled: bool = True,
    ) -> None:
        self.stage = stage.replace(" ", "_")
        self.total_steps = max(int(total_steps), 0)
        self.initial_step = max(int(initial_step), 0)
        self.current_step = self.initial_step
        self.enabled = enabled
        self.started_at = time.monotonic()
        if self.enabled:
            self._emit("start")

    def _timing(self, step: int) -> Tuple[float, Optional[float]]:
        elapsed = max(time.monotonic() - self.started_at, 0.0)
        completed = max(step - self.initial_step, 0)
        remaining = max(self.total_steps - step, 0)
        eta = None
        if completed > 0:
            eta = elapsed * remaining / completed
        return elapsed, eta

    def _emit(
        self,
        status: str,
        *,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        elapsed, eta = self._timing(self.current_step)
        percentage = (
            min(max(self.current_step / self.total_steps, 0.0), 1.0) * 100.0
            if self.total_steps
            else 100.0
        )
        fields = [
            "[progress]",
            f"stage={self.stage}",
            f"status={status}",
            f"step={self.current_step}/{self.total_steps}",
            f"pct={percentage:.2f}%",
            f"elapsed={_format_progress_duration(elapsed)}",
            f"eta={_format_progress_duration(eta)}",
        ]
        for key, value in (metrics or {}).items():
            fields.append(f"{key}={_format_progress_metric(value)}")
        print(" ".join(fields), flush=True)

    def update(self, step: int, **metrics: Any) -> None:
        self.current_step = max(int(step), self.current_step)
        self._emit("running", metrics=metrics)

    def status(self, status: str, **metrics: Any) -> None:
        self._emit(status.replace(" ", "_"), metrics=metrics)

    def complete(self, **metrics: Any) -> None:
        self.current_step = max(self.current_step, self.total_steps)
        self._emit("complete", metrics=metrics)


@dataclass
class ResumeState:
    global_step: int = 0
    epoch: int = 0
    step_in_epoch: int = 0


def find_latest_checkpoint(output_dir: str | os.PathLike[str]) -> Optional[Path]:
    root = Path(output_dir)
    candidates = []
    if root.exists():
        for path in root.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume_checkpoint(args: TrainingArguments) -> Optional[Path]:
    value = args.resume_from_checkpoint
    if not value:
        return None
    if value.lower() == "auto":
        return find_latest_checkpoint(args.output_dir)
    if value.lower() in {"latest", "true"}:
        checkpoint = find_latest_checkpoint(args.output_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"no checkpoint-* directory found under {args.output_dir}"
            )
        return checkpoint
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def save_checkpoint(
    accelerator: Any,
    *,
    output_dir: str,
    state: ResumeState,
    extra_state: Optional[Mapping[str, Any]] = None,
) -> Path:
    checkpoint = Path(output_dir) / f"checkpoint-{state.global_step:08d}"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint))
    if accelerator.is_main_process:
        payload: Dict[str, Any] = asdict(state)
        if extra_state:
            payload["extra_state"] = dict(extra_state)
        atomic_write_json(checkpoint / "trainer_state.json", payload)
    accelerator.wait_for_everyone()
    return checkpoint


def load_checkpoint(accelerator: Any, checkpoint: Path) -> ResumeState:
    accelerator.load_state(str(checkpoint))
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return ResumeState(
        global_step=int(payload["global_step"]),
        epoch=int(payload["epoch"]),
        step_in_epoch=int(payload["step_in_epoch"]),
    )


def save_final_model(
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    output_dir: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    destination = Path(output_dir) / "final"
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(
            destination,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(destination)
        if metadata:
            atomic_write_json(destination / "run_metadata.json", metadata)
    accelerator.wait_for_everyone()


def move_batch_to_device(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device) if hasattr(value, "to") else value
    return result


def should_stop(global_step: int, args: TrainingArguments, total_steps: int) -> bool:
    return global_step >= total_steps or (
        args.max_steps > 0 and global_step >= args.max_steps
    )
