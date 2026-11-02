"""Lazy Hugging Face/PEFT model loading for full, LoRA, and QLoRA training."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple


@dataclass
class ModelArguments:
    student_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Student causal-LM path or Hugging Face identifier."},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model with the same tokenizer/vocabulary."},
    )
    reference_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional frozen reference for retention/DPO full tuning."},
    )
    tokenizer_name_or_path: Optional[str] = None
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    low_cpu_mem_usage: bool = True


@dataclass
class AdapterArguments:
    use_lora: bool = True
    use_qlora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = (
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    lora_modules_to_save: Optional[str] = None
    qlora_compute_dtype: str = "bfloat16"
    qlora_quant_type: str = "nf4"
    qlora_double_quant: bool = True


def _require_transformers() -> Tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model loading requires torch and transformers; install the project "
            "training dependencies first."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str) -> Any:
    torch, _, _ = _require_transformers()
    normalized = name.lower().replace("torch.", "")
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": "auto",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported dtype {name!r}")
    return aliases[normalized]


def load_tokenizer(
    model_args: ModelArguments,
    *,
    model_name_or_path: Optional[str] = None,
) -> Any:
    _, _, AutoTokenizer = _require_transformers()
    path = (
        model_args.tokenizer_name_or_path
        or model_name_or_path
        or model_args.student_model_name_or_path
    )
    if not path:
        raise ValueError("a tokenizer or model path is required")
    loader_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.revision,
        "trust_remote_code": model_args.trust_remote_code,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            **loader_kwargs,
        )
    except (ValueError, TypeError, AttributeError) as tokenizer_error:
        # Official Qwen3.5 checkpoints may expose their text tokenizer through
        # AutoProcessor because the architecture is multimodal.  Keep the
        # training path text-only while honoring the official checkpoint API.
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(path, **loader_kwargs)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("AutoProcessor has no tokenizer attribute")
        except Exception as processor_error:
            raise RuntimeError(
                f"neither AutoTokenizer nor AutoProcessor could load {path!r}"
            ) from processor_error
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """SHA-256 over the complete token-id mapping and special-token contract."""

    payload = {
        "vocab": sorted(
            ((str(token), int(index)) for token, index in tokenizer.get_vocab().items()),
            key=lambda item: (item[1], item[0]),
        ),
        "special": {
            name: getattr(tokenizer, name, None)
            for name in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
            )
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def assert_same_tokenizer(student_tokenizer: Any, teacher_tokenizer: Any) -> None:
    """Fail closed when dense token alignment would be invalid."""

    student_hash = tokenizer_fingerprint(student_tokenizer)
    teacher_hash = tokenizer_fingerprint(teacher_tokenizer)
    if student_hash != teacher_hash:
        raise ValueError(
            "CARE-OPD dense KL requires identical token-to-id vocabularies; "
            f"student hash {student_hash} != teacher hash {teacher_hash}"
        )
    special_names = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    )
    mismatched = [
        name
        for name in special_names
        if getattr(student_tokenizer, name, None)
        != getattr(teacher_tokenizer, name, None)
    ]
    if mismatched:
        raise ValueError(f"student/teacher special token mismatch: {mismatched}")


def _quantization_config(adapter_args: AdapterArguments) -> Any:
    if not adapter_args.use_qlora:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("QLoRA requires a transformers build with BitsAndBytesConfig") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=adapter_args.qlora_quant_type,
        bnb_4bit_use_double_quant=adapter_args.qlora_double_quant,
        bnb_4bit_compute_dtype=resolve_dtype(adapter_args.qlora_compute_dtype),
    )


def _local_device_map() -> Optional[Dict[str, int]]:
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        local_rank = 0
    try:
        import torch

        if torch.cuda.is_available():
            return {"": local_rank}
    except ImportError:
        pass
    return None


def load_causal_lm(
    model_name_or_path: str,
    model_args: ModelArguments,
    *,
    quantization_config: Any = None,
    trainable: bool,
) -> Any:
    _, AutoModelForCausalLM, _ = _require_transformers()
    kwargs: Dict[str, Any] = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.revision,
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": resolve_dtype(model_args.torch_dtype),
        "low_cpu_mem_usage": model_args.low_cpu_mem_usage,
    }
    if model_args.attn_implementation:
        kwargs["attn_implementation"] = model_args.attn_implementation
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = _local_device_map()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except ValueError as causal_error:
        # transformers>=5 exposes Qwen3.5 multimodal checkpoints through an
        # automatic multimodal class.  Class names changed across release
        # candidates, so try the official generic classes in a stable order.
        import transformers

        candidates = (
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        )
        model = None
        errors = [causal_error]
        for class_name in candidates:
            auto_class = getattr(transformers, class_name, None)
            if auto_class is None:
                continue
            try:
                model = auto_class.from_pretrained(model_name_or_path, **kwargs)
                break
            except (ValueError, TypeError, AttributeError) as fallback_error:
                errors.append(fallback_error)
        if model is None:
            raise RuntimeError(
                f"no causal or multimodal auto-model class supports "
                f"{model_name_or_path!r}; last error: {errors[-1]}"
            ) from errors[-1]
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return model


def attach_lora(
    model: Any,
    adapter_args: AdapterArguments,
    *,
    gradient_checkpointing: bool,
) -> Any:
    if not adapter_args.use_lora:
        if adapter_args.use_qlora:
            raise ValueError("use_qlora requires use_lora")
        return model
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise RuntimeError("LoRA/QLoRA training requires peft") from exc

    if adapter_args.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    targets = [item.strip() for item in adapter_args.lora_target_modules.split(",") if item.strip()]
    modules_to_save = None
    if adapter_args.lora_modules_to_save:
        modules_to_save = [
            item.strip()
            for item in adapter_args.lora_modules_to_save.split(",")
            if item.strip()
        ]
    config = LoraConfig(
        r=adapter_args.lora_rank,
        lora_alpha=adapter_args.lora_alpha,
        lora_dropout=adapter_args.lora_dropout,
        target_modules=targets,
        modules_to_save=modules_to_save,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_student(
    model_args: ModelArguments,
    adapter_args: AdapterArguments,
    *,
    gradient_checkpointing: bool,
) -> Tuple[Any, Any]:
    if not model_args.student_model_name_or_path:
        raise ValueError("--student_model_name_or_path is required")
    tokenizer = load_tokenizer(model_args)
    student = load_causal_lm(
        model_args.student_model_name_or_path,
        model_args,
        quantization_config=_quantization_config(adapter_args),
        trainable=True,
    )
    if gradient_checkpointing:
        student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        student.config.use_cache = False
    student = attach_lora(
        student,
        adapter_args,
        gradient_checkpointing=gradient_checkpointing,
    )
    return student, tokenizer


def load_teacher(
    model_args: ModelArguments,
    student_tokenizer: Any,
) -> Any:
    if not model_args.teacher_model_name_or_path:
        raise ValueError("--teacher_model_name_or_path is required for OPD")
    teacher_tokenizer = load_tokenizer(
        model_args,
        model_name_or_path=model_args.teacher_model_name_or_path,
    )
    assert_same_tokenizer(student_tokenizer, teacher_tokenizer)
    teacher = load_causal_lm(
        model_args.teacher_model_name_or_path,
        model_args,
        trainable=False,
    )
    return teacher


def load_reference(
    model_args: ModelArguments,
    student_tokenizer: Any,
) -> Optional[Any]:
    if not model_args.reference_model_name_or_path:
        return None
    reference_tokenizer = load_tokenizer(
        model_args,
        model_name_or_path=model_args.reference_model_name_or_path,
    )
    assert_same_tokenizer(student_tokenizer, reference_tokenizer)
    return load_causal_lm(
        model_args.reference_model_name_or_path,
        model_args,
        trainable=False,
    )


def has_disable_adapter(model: Any) -> bool:
    return callable(getattr(model, "disable_adapter", None))


@contextlib.contextmanager
def adapter_disabled(model: Any) -> Iterator[None]:
    """Temporarily expose the frozen base policy beneath a PEFT adapter."""

    method = getattr(model, "disable_adapter", None)
    if method is None:
        raise RuntimeError(
            "No frozen reference was supplied and the student is not a PEFT model; "
            "set --reference_model_name_or_path or enable LoRA"
        )
    with method():
        yield


def parameter_report(model: Any) -> Dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable": trainable, "total": total}
