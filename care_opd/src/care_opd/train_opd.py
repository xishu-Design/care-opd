"""Dense on-policy distillation for CARE-OPD and full-response baselines."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .collator import OPDCollator
from .losses import (
    dense_distillation_loss,
    masked_hard_target_cross_entropy,
    teacher_trust_mask,
)
from .models import (
    AdapterArguments,
    ModelArguments,
    adapter_disabled,
    load_reference,
    load_student,
    load_teacher,
    parameter_report,
    tokenizer_fingerprint,
)
from .trainer_utils import (
    JsonlDataset,
    MetricLogger,
    ProgressReporter,
    ResumeState,
    TrainingArguments,
    build_optimizer,
    build_scheduler,
    create_accelerator,
    load_checkpoint,
    move_batch_to_device,
    parse_dataclass_args,
    resolve_resume_checkpoint,
    save_checkpoint,
    save_final_model,
    seed_everything,
    should_stop,
    total_update_steps,
)


@dataclass
class OPDArguments:
    """Method knobs; ``baseline_mode`` applies a preregistered switch bundle."""

    baseline_mode: str = "care_opd"
    mask_mode: str = "span"
    random_mask_seed: int = 20260815
    use_risk_weighting: bool = True
    oracle_risk_weighting_mode: str = "inherit"
    use_teacher_trust: bool = True
    retain_outside_repair: bool = True
    distillation_coefficient: float = 1.0
    retention_coefficient: float = 0.2
    hard_oracle_ce_weight: float = 0.0
    kl_mode: str = "js"
    temperature: float = 1.0
    js_beta: float = 0.9
    min_host_verification_confidence: float = 0.0
    # Deprecated compatibility alias for early configs.  It gates host
    # certificate confidence, never sampled teacher-output verification.
    min_sample_teacher_confidence: Optional[float] = None
    min_teacher_top_probability: float = 0.0
    min_teacher_margin: float = 0.0
    max_teacher_normalized_entropy: float = 1.0
    max_teacher_student_js: float = float("inf")
    max_teacher_top_log_ratio: float = 3.0
    # Qwen3.5 has a 248k-token vocabulary.  Sixteen response positions keep
    # the dense float32 distribution arithmetic bounded on batch-size-one runs.
    response_chunk_size: int = 16
    max_response_tokens: int = 384
    add_eos: bool = True

    def apply_baseline_bundle(self) -> "OPDArguments":
        mode = self.baseline_mode.lower()
        if mode in {"care_opd", "proposed"}:
            return self
        if mode in {"gkd_full", "full_response_opd"}:
            self.mask_mode = "full"
            self.use_risk_weighting = False
            self.use_teacher_trust = False
            self.retain_outside_repair = False
            self.retention_coefficient = 0.0
            self.hard_oracle_ce_weight = 0.0
            return self
        if mode in {"tropd_full", "trust_region_full"}:
            self.mask_mode = "full"
            self.use_risk_weighting = False
            self.use_teacher_trust = True
            self.retain_outside_repair = False
            self.retention_coefficient = 0.0
            self.hard_oracle_ce_weight = 0.0
            return self
        if mode in {"span_only", "causal_span"}:
            self.mask_mode = "span"
            self.use_risk_weighting = False
            self.use_teacher_trust = False
            self.retention_coefficient = 0.0
            self.hard_oracle_ce_weight = 0.0
            return self
        raise ValueError(
            "baseline_mode must be care_opd, gkd_full, tropd_full, or span_only"
        )

    def validate(self) -> None:
        if self.mask_mode not in {"span", "full", "random_token_matched"}:
            raise ValueError(
                "mask_mode must be span, full, or random_token_matched"
            )
        if self.kl_mode not in {"forward_kl", "reverse_kl", "js"}:
            raise ValueError("kl_mode must be forward_kl, reverse_kl, or js")
        if self.oracle_risk_weighting_mode not in {
            "inherit",
            "enabled",
            "disabled",
        }:
            raise ValueError(
                "oracle_risk_weighting_mode must be inherit, enabled, or disabled"
            )
        if (
            self.distillation_coefficient < 0
            or self.retention_coefficient < 0
            or self.hard_oracle_ce_weight < 0
        ):
            raise ValueError("loss coefficients must be non-negative")
        if self.response_chunk_size < 1 or self.max_response_tokens < 1:
            raise ValueError("response chunk/token limits must be positive")
        if self.min_host_verification_confidence < 0:
            raise ValueError("host verification confidence threshold must be non-negative")

    def host_confidence_threshold(self) -> float:
        if self.min_sample_teacher_confidence is not None:
            return float(self.min_sample_teacher_confidence)
        return float(self.min_host_verification_confidence)


def _supports_selected_logits(model: Any) -> bool:
    """Whether a model advertises Qwen-style selected-position logits."""

    target = getattr(model, "module", model)
    if hasattr(target, "get_base_model"):
        target = target.get_base_model()
    try:
        parameters = inspect.signature(target.forward).parameters
    except (TypeError, ValueError):
        return False
    return "logits_to_keep" in parameters


def _forward_response_chunk(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    prompt_lengths: Any,
    response_lengths: Any,
    chunk_start: int,
    chunk_end: int,
) -> Any:
    """Forward only the requested response prediction positions when supported.

    Qwen3.5 accepts a tensor-valued ``logits_to_keep`` argument.  For a batch
    with different prompt lengths we pass the union of required absolute
    positions and gather each sample's positions afterward.  Other model
    families fall back to full logits with an explicit efficiency flag.
    """

    import torch

    offsets = torch.arange(chunk_start, chunk_end, device=input_ids.device)
    positions = prompt_lengths.to(input_ids.device).unsqueeze(1) - 1 + offsets
    valid = offsets.unsqueeze(0) < response_lengths.to(input_ids.device).unsqueeze(1)
    valid &= positions >= 0
    valid &= positions < input_ids.shape[1]
    union = torch.unique(positions[valid], sorted=True)
    if union.numel() == 0:
        raise ValueError("response chunk contains no valid causal positions")

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    selected_supported = _supports_selected_logits(model)
    if selected_supported:
        kwargs["logits_to_keep"] = union
    output = model(**kwargs)
    logits = output.logits
    if logits.shape[1] == input_ids.shape[1]:
        safe_positions = positions.clamp(0, input_ids.shape[1] - 1)
        aligned = logits.gather(
            1,
            safe_positions.unsqueeze(-1).expand(
                positions.shape[0], positions.shape[1], logits.shape[-1]
            ),
        )
        memory_efficient = False
    elif logits.shape[1] == union.numel():
        lookup = torch.searchsorted(union, positions.clamp(union[0], union[-1]))
        lookup = lookup.clamp(0, union.numel() - 1)
        aligned = logits.gather(
            1,
            lookup.unsqueeze(-1).expand(
                positions.shape[0], positions.shape[1], logits.shape[-1]
            ),
        )
        memory_efficient = True
    else:
        raise RuntimeError(
            "model returned an unexpected sequence dimension after logits_to_keep: "
            f"{logits.shape[1]} (input={input_ids.shape[1]}, selected={union.numel()})"
        )
    return aligned, valid, memory_efficient


def _response_chunk_targets(
    *,
    input_ids: Any,
    prompt_lengths: Any,
    response_lengths: Any,
    chunk_start: int,
    chunk_end: int,
) -> Any:
    """Gather target token ids for a response-position chunk."""

    import torch

    offsets = torch.arange(chunk_start, chunk_end, device=input_ids.device)
    positions = prompt_lengths.to(input_ids.device).unsqueeze(1) + offsets
    valid = offsets.unsqueeze(0) < response_lengths.to(input_ids.device).unsqueeze(1)
    valid &= positions >= 0
    valid &= positions < input_ids.shape[1]
    safe_positions = positions.clamp(0, input_ids.shape[1] - 1)
    return input_ids.gather(1, safe_positions), valid


def _synchronized_response_widths(
    accelerator: Any,
    *,
    response_width: int,
    oracle_width: int,
    has_active_oracle: bool,
) -> tuple[int, int, bool]:
    """Return rank-global response widths/eligibility in one collective.

    DDP broadcasts module buffers at the start of every wrapped student
    forward.  Consequently, every rank must execute the same number of
    response and hard-oracle forwards even when its local sample is shorter or
    has no verified oracle target.  Gathering both local widths and the oracle
    eligibility bit establishes one shared plan before any data-dependent
    forward.
    """

    import torch

    local_widths = torch.tensor(
        [int(response_width), int(oracle_width), int(has_active_oracle)],
        dtype=torch.long,
        device=accelerator.device,
    )
    gathered = accelerator.gather(local_widths)
    if gathered.numel() % 3:
        raise RuntimeError(
            "distributed response-width gather returned an invalid shape: "
            f"{tuple(gathered.shape)}"
        )
    maxima = gathered.reshape(-1, 3).amax(dim=0)
    return (
        max(int(maxima[0].item()), 1),
        int(maxima[1].item()),
        bool(maxima[2].item()),
    )


def _pad_response_axis(tensor: Any, width: int) -> Any:
    """Right-pad a [batch, response] tensor to a shared rank-global width."""

    import torch.nn.functional as F

    if tensor.ndim != 2:
        raise ValueError("response-aligned tensors must have rank two")
    missing = int(width) - int(tensor.shape[1])
    if missing < 0:
        raise ValueError(
            f"cannot pad response width {tensor.shape[1]} down to {width}"
        )
    if missing == 0:
        return tensor
    return F.pad(tensor, (0, missing), value=0)


def _risk_scaled_chunk_loss(
    weighted_mean: Any,
    *,
    active_weight: Any,
    normalization_tokens: Any,
) -> Any:
    """Preserve absolute consequence weights under batch-size-one training.

    ``weighted_mean`` is normalized by the sum of active weights inside the
    generic loss helper.  Multiplying that denominator back reconstructs the
    weighted numerator.  Dividing by the *unweighted* eligible-token count
    makes a risk-4 repair contribute four times a risk-1 repair instead of
    cancelling the risk multiplier in numerator and denominator.
    """

    return weighted_mean * active_weight / normalization_tokens


def _forward_dummy_response_token(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    prompt_lengths: Any,
) -> tuple[Any, bool]:
    """Run one differentiable, fully masked student token for DDP alignment."""

    import torch

    dummy_lengths = torch.ones_like(prompt_lengths)
    logits, _, efficient = _forward_response_chunk(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        response_lengths=dummy_lengths,
        chunk_start=0,
        chunk_end=1,
    )
    return logits, efficient


def _reference_response_chunk(
    *,
    accelerator: Any,
    student: Any,
    reference: Optional[Any],
    batch: Any,
    chunk_start: int,
    chunk_end: int,
) -> Any:
    import torch

    with torch.no_grad():
        if reference is not None:
            logits, _, efficient = _forward_response_chunk(
                reference,
                input_ids=batch["student_input_ids"],
                attention_mask=batch["student_attention_mask"],
                prompt_lengths=batch["student_prompt_lengths"],
                response_lengths=batch["response_lengths"],
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
        else:
            base_student = accelerator.unwrap_model(student)
            with adapter_disabled(base_student):
                logits, _, efficient = _forward_response_chunk(
                    base_student,
                    input_ids=batch["student_input_ids"],
                    attention_mask=batch["student_attention_mask"],
                    prompt_lengths=batch["student_prompt_lengths"],
                    response_lengths=batch["response_lengths"],
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
    return logits.detach(), efficient


def train(
    model_args: ModelArguments,
    adapter_args: AdapterArguments,
    training_args: TrainingArguments,
    opd_args: OPDArguments,
) -> None:
    if not training_args.train_file:
        raise ValueError("--train_file is required")
    opd_args.apply_baseline_bundle()
    opd_args.validate()
    host_confidence_threshold = opd_args.host_confidence_threshold()
    seed_everything(training_args.seed)
    accelerator = create_accelerator(training_args)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(
            "[progress] "
            f"stage=opd:{opd_args.baseline_mode} status=loading_student "
            f"model={model_args.student_model_name_or_path}",
            flush=True,
        )
    student, tokenizer = load_student(
        model_args,
        adapter_args,
        gradient_checkpointing=training_args.gradient_checkpointing,
    )
    if accelerator.is_main_process:
        print(
            "[progress] "
            f"stage=opd:{opd_args.baseline_mode} status=loading_teacher "
            f"model={model_args.teacher_model_name_or_path}",
            flush=True,
        )
    teacher = load_teacher(model_args, tokenizer)
    teacher.to(accelerator.device)
    teacher.eval()
    reference = None
    if opd_args.retention_coefficient > 0:
        if accelerator.is_main_process:
            print(
                "[progress] "
                f"stage=opd:{opd_args.baseline_mode} status=loading_reference",
                flush=True,
            )
        reference = load_reference(model_args, tokenizer)
        if reference is not None:
            reference.to(accelerator.device)
            reference.eval()
        elif not adapter_args.use_lora:
            raise ValueError(
                "clean retention under full-model tuning requires "
                "--reference_model_name_or_path"
            )

    # qzbc-native rows use structured ``teacher_messages`` and deliberately
    # store ``teacher_prompt=None`` so the tokenizer's current chat template is
    # applied at training time. Requiring the legacy scalar teacher_prompt
    # would reject every valid qzbc OPD row before collation.
    if accelerator.is_main_process:
        print(
            "[progress] "
            f"stage=opd:{opd_args.baseline_mode} status=loading_data "
            f"path={training_args.train_file}",
            flush=True,
        )
    dataset = JsonlDataset(training_args.train_file)
    for row_index, row in enumerate(dataset.records, start=1):
        if row.get("teacher_messages") is None and row.get("teacher_prompt") is None:
            raise ValueError(
                f"{training_args.train_file}:{row_index}: "
                "teacher_messages or teacher_prompt is required"
            )
    collator = OPDCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_response_length=opd_args.max_response_tokens,
        mask_mode=opd_args.mask_mode,
        random_mask_seed=opd_args.random_mask_seed,
        add_eos=opd_args.add_eos,
        retain_outside_repair=opd_args.retain_outside_repair,
        use_risk_weighting=opd_args.use_risk_weighting,
        oracle_risk_weighting_mode=opd_args.oracle_risk_weighting_mode,
    )

    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(training_args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=training_args.dataloader_num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = build_optimizer(student, training_args)
    scheduler_steps = total_update_steps(len(dataloader), training_args)
    scheduler = build_scheduler(optimizer, scheduler_steps, training_args)
    student, optimizer, dataloader, scheduler = accelerator.prepare(
        student, optimizer, dataloader, scheduler
    )
    total_steps = total_update_steps(len(dataloader), training_args)

    state = ResumeState()
    checkpoint = resolve_resume_checkpoint(training_args)
    if checkpoint is not None:
        state = load_checkpoint(accelerator, checkpoint)
    logger = MetricLogger(
        Path(training_args.output_dir) / "metrics.jsonl",
        enabled=accelerator.is_main_process,
    )
    logger.log(
        {
            "event": "start",
            "method": opd_args.baseline_mode,
            "dataset_size": len(dataset),
            "total_steps": total_steps,
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            "parameters": parameter_report(accelerator.unwrap_model(student)),
            "effective_switches": {
                "mask_mode": opd_args.mask_mode,
                "random_mask_seed": opd_args.random_mask_seed,
                "risk_weighting": opd_args.use_risk_weighting,
                "oracle_risk_weighting_mode": (
                    opd_args.oracle_risk_weighting_mode
                ),
                "teacher_trust": opd_args.use_teacher_trust,
                "retain_outside_repair": opd_args.retain_outside_repair,
                "distillation_coefficient": (
                    opd_args.distillation_coefficient
                ),
                "retention_coefficient": opd_args.retention_coefficient,
                "hard_oracle_ce_weight": opd_args.hard_oracle_ce_weight,
                "ddp_chunk_plan": "rank_global_max_with_masked_dummy",
                "risk_normalization": "unweighted_eligible_token_count",
                "host_verification_confidence_threshold": (
                    host_confidence_threshold
                ),
            },
        }
    )
    progress = ProgressReporter(
        stage=f"opd:{opd_args.baseline_mode}",
        total_steps=total_steps,
        initial_step=state.global_step,
        enabled=accelerator.is_main_process,
    )

    student.train()
    stop = False
    start_epoch = state.epoch
    start_step_in_epoch = state.step_in_epoch
    if start_step_in_epoch >= len(dataloader):
        start_epoch += 1
        start_step_in_epoch = 0

    for epoch in range(start_epoch, training_args.num_train_epochs):
        epoch_loader = dataloader
        offset = 0
        if epoch == start_epoch and start_step_in_epoch:
            offset = start_step_in_epoch
            epoch_loader = accelerator.skip_first_batches(dataloader, offset)
        for local_index, batch in enumerate(epoch_loader):
            step_in_epoch = local_index + offset
            batch = move_batch_to_device(batch, accelerator.device)
            with accelerator.accumulate(student):
                local_active_oracle_mask = (
                    batch["oracle_mask"]
                    & batch["host_verified"].unsqueeze(1)
                    & (
                        batch["host_verification_confidence"].unsqueeze(1)
                        >= host_confidence_threshold
                    )
                )
                max_response, max_oracle_response, any_active_oracle = (
                    _synchronized_response_widths(
                        accelerator,
                        response_width=batch["distill_mask"].shape[1],
                        oracle_width=batch["oracle_mask"].shape[1],
                        has_active_oracle=bool(
                            local_active_oracle_mask.any().item()
                        ),
                    )
                )

                # Local collation pads only within a rank.  Pad masks and
                # weights once more to the rank-global chunk plan so partial
                # final chunks retain matching shapes.
                distill_mask = _pad_response_axis(
                    batch["distill_mask"], max_response
                )
                retention_mask = _pad_response_axis(
                    batch["retention_mask"], max_response
                )
                token_weights = _pad_response_axis(
                    batch["token_weights"], max_response
                )
                active_oracle_mask = _pad_response_axis(
                    local_active_oracle_mask, max_oracle_response
                )
                oracle_token_weights = _pad_response_axis(
                    batch["oracle_token_weights"], max_oracle_response
                )

                # Keep normalization over the complete local response while
                # backpropagating one position chunk at a time.  Trust-rejected
                # tokens contribute zero rather than changing the denominator.
                distill_normalization_tokens = (
                    distill_mask.sum().to(torch.float32).clamp_min(1.0)
                )
                retention_normalization_tokens = (
                    retention_mask.sum().to(torch.float32).clamp_min(1.0)
                )
                oracle_normalization_tokens = (
                    active_oracle_mask.sum().to(torch.float32).clamp_min(1.0)
                )
                use_hard_oracle = bool(
                    opd_args.hard_oracle_ce_weight > 0
                    and any_active_oracle
                    and max_oracle_response > 0
                )
                batch_total = torch.zeros((), device=accelerator.device)
                distillation_total = torch.zeros((), device=accelerator.device)
                retention_total = torch.zeros((), device=accelerator.device)
                hard_oracle_total = torch.zeros((), device=accelerator.device)
                candidate_tokens = 0
                trusted_tokens = 0
                retention_tokens = 0
                hard_oracle_tokens = int(active_oracle_mask.sum().item())
                diagnostic_sums = {
                    "top_probability": 0.0,
                    "entropy": 0.0,
                    "js": 0.0,
                    "margin": 0.0,
                    "top_log_ratio": 0.0,
                }
                selected_logits_efficient = True

                for chunk_start in range(
                    0, max_response, opd_args.response_chunk_size
                ):
                    chunk_end = min(
                        chunk_start + opd_args.response_chunk_size, max_response
                    )
                    local_has_response = bool(
                        (batch["response_lengths"] > chunk_start).any().item()
                    )
                    if not local_has_response:
                        # Another rank has a longer response.  This rank still
                        # performs one wrapped student forward/backward so DDP
                        # collective ordering remains identical.
                        dummy_logits, dummy_efficient = (
                            _forward_dummy_response_token(
                                student,
                                input_ids=batch["student_input_ids"],
                                attention_mask=batch[
                                    "student_attention_mask"
                                ],
                                prompt_lengths=batch[
                                    "student_prompt_lengths"
                                ],
                            )
                        )
                        dummy_loss = dummy_logits.sum() * 0.0
                        accelerator.backward(dummy_loss)
                        batch_total += dummy_loss.detach()
                        selected_logits_efficient &= dummy_efficient
                        del dummy_logits
                        continue

                    student_logits, valid, student_efficient = _forward_response_chunk(
                        student,
                        input_ids=batch["student_input_ids"],
                        attention_mask=batch["student_attention_mask"],
                        prompt_lengths=batch["student_prompt_lengths"],
                        response_lengths=batch["response_lengths"],
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                    )
                    with torch.no_grad():
                        (
                            teacher_logits,
                            teacher_valid,
                            teacher_efficient,
                        ) = _forward_response_chunk(
                            teacher,
                            input_ids=batch["teacher_input_ids"],
                            attention_mask=batch["teacher_attention_mask"],
                            prompt_lengths=batch["teacher_prompt_lengths"],
                            response_lengths=batch["response_lengths"],
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                        )
                        teacher_logits = teacher_logits.detach()
                    selected_logits_efficient &= student_efficient and teacher_efficient

                    local_slice = slice(chunk_start, chunk_end)
                    candidate_mask = (
                        distill_mask[:, local_slice]
                        & valid
                        & teacher_valid
                    )
                    if opd_args.use_teacher_trust:
                        active_mask, trust = teacher_trust_mask(
                            student_logits,
                            teacher_logits,
                            candidate_mask,
                            sample_verified=batch["host_verified"],
                            sample_confidence=batch[
                                "host_verification_confidence"
                            ],
                            min_sample_confidence=(
                                host_confidence_threshold
                            ),
                            min_teacher_top_probability=(
                                opd_args.min_teacher_top_probability
                            ),
                            min_teacher_margin=opd_args.min_teacher_margin,
                            max_teacher_normalized_entropy=(
                                opd_args.max_teacher_normalized_entropy
                            ),
                            max_teacher_student_js=(
                                opd_args.max_teacher_student_js
                            ),
                            max_teacher_top_log_ratio=(
                                opd_args.max_teacher_top_log_ratio
                            ),
                        )
                    else:
                        active_mask = (
                            candidate_mask
                            & batch["host_verified"].unsqueeze(1)
                            & (
                                batch["host_verification_confidence"].unsqueeze(1)
                                >= host_confidence_threshold
                            )
                        )
                        from .losses import TrustDiagnostics

                        candidates = int(candidate_mask.sum().item())
                        retained = int(active_mask.sum().item())
                        trust = TrustDiagnostics(
                            candidate_tokens=candidates,
                            retained_tokens=retained,
                            retained_fraction=(
                                retained / candidates if candidates else 0.0
                            ),
                            mean_teacher_top_probability=0.0,
                            mean_teacher_normalized_entropy=0.0,
                            mean_teacher_student_js=0.0,
                            mean_teacher_margin=0.0,
                            mean_teacher_top_log_ratio=0.0,
                        )

                    local_weights = token_weights[:, local_slice]
                    local_distillation_mean, _ = dense_distillation_loss(
                        student_logits,
                        teacher_logits,
                        active_mask,
                        token_weights=local_weights,
                        mode=opd_args.kl_mode,
                        temperature=opd_args.temperature,
                        js_beta=opd_args.js_beta,
                    )
                    active_weight = (
                        local_weights * active_mask.to(local_weights.dtype)
                    ).sum()
                    scaled_distillation = _risk_scaled_chunk_loss(
                        local_distillation_mean,
                        active_weight=active_weight,
                        normalization_tokens=distill_normalization_tokens,
                    )

                    local_retention_mask = (
                        retention_mask[:, local_slice] & valid
                    )
                    local_retention_count = int(local_retention_mask.sum().item())
                    scaled_retention = student_logits.sum() * 0.0
                    if (
                        opd_args.retention_coefficient > 0
                        and local_retention_count
                    ):
                        reference_logits, reference_efficient = (
                            _reference_response_chunk(
                                accelerator=accelerator,
                                student=student,
                                reference=reference,
                                batch=batch,
                                chunk_start=chunk_start,
                                chunk_end=chunk_end,
                            )
                        )
                        selected_logits_efficient &= reference_efficient
                        local_retention_mean, _ = dense_distillation_loss(
                            student_logits,
                            reference_logits,
                            local_retention_mask,
                            mode="forward_kl",
                            temperature=opd_args.temperature,
                        )
                        scaled_retention = (
                            local_retention_mean
                            * local_retention_mask.sum()
                            / retention_normalization_tokens
                        )
                        del reference_logits

                    chunk_loss = (
                        opd_args.distillation_coefficient * scaled_distillation
                        + opd_args.retention_coefficient * scaled_retention
                    )
                    accelerator.backward(chunk_loss)
                    batch_total += chunk_loss.detach()
                    distillation_total += scaled_distillation.detach()
                    retention_total += scaled_retention.detach()
                    candidate_tokens += trust.candidate_tokens
                    trusted_tokens += trust.retained_tokens
                    retention_tokens += local_retention_count
                    diagnostic_sums["top_probability"] += (
                        trust.mean_teacher_top_probability
                        * trust.candidate_tokens
                    )
                    diagnostic_sums["entropy"] += (
                        trust.mean_teacher_normalized_entropy
                        * trust.candidate_tokens
                    )
                    diagnostic_sums["js"] += (
                        trust.mean_teacher_student_js * trust.candidate_tokens
                    )
                    diagnostic_sums["margin"] += (
                        trust.mean_teacher_margin * trust.candidate_tokens
                    )
                    diagnostic_sums["top_log_ratio"] += (
                        trust.mean_teacher_top_log_ratio
                        * trust.candidate_tokens
                    )
                    del student_logits, teacher_logits

                if use_hard_oracle:
                    for chunk_start in range(
                        0,
                        max_oracle_response,
                        opd_args.response_chunk_size,
                    ):
                        chunk_end = min(
                            chunk_start + opd_args.response_chunk_size,
                            max_oracle_response,
                        )
                        local_slice = slice(chunk_start, chunk_end)
                        local_oracle_mask = active_oracle_mask[:, local_slice]
                        local_has_oracle_response = bool(
                            (
                                batch["oracle_response_lengths"]
                                > chunk_start
                            )
                            .any()
                            .item()
                        )
                        if not local_has_oracle_response:
                            # Clean controls intentionally have no oracle
                            # response.  Use a masked zero-loss token instead
                            # of skipping the DDP forward on this rank.
                            dummy_logits, dummy_efficient = (
                                _forward_dummy_response_token(
                                    student,
                                    input_ids=batch["oracle_input_ids"],
                                    attention_mask=batch[
                                        "oracle_attention_mask"
                                    ],
                                    prompt_lengths=batch[
                                        "oracle_prompt_lengths"
                                    ],
                                )
                            )
                            hard_chunk_loss = dummy_logits.sum() * 0.0
                            accelerator.backward(hard_chunk_loss)
                            batch_total += hard_chunk_loss.detach()
                            selected_logits_efficient &= dummy_efficient
                            del dummy_logits
                            continue
                        oracle_logits, oracle_valid, oracle_efficient = (
                            _forward_response_chunk(
                                student,
                                input_ids=batch["oracle_input_ids"],
                                attention_mask=batch["oracle_attention_mask"],
                                prompt_lengths=batch["oracle_prompt_lengths"],
                                response_lengths=batch[
                                    "oracle_response_lengths"
                                ],
                                chunk_start=chunk_start,
                                chunk_end=chunk_end,
                            )
                        )
                        oracle_targets, target_valid = _response_chunk_targets(
                            input_ids=batch["oracle_input_ids"],
                            prompt_lengths=batch["oracle_prompt_lengths"],
                            response_lengths=batch["oracle_response_lengths"],
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                        )
                        local_oracle_mask = (
                            local_oracle_mask & oracle_valid & target_valid
                        )
                        local_oracle_weights = oracle_token_weights[
                            :, local_slice
                        ]
                        local_oracle_ce, _ = (
                            masked_hard_target_cross_entropy(
                                oracle_logits,
                                oracle_targets,
                                local_oracle_mask,
                                token_weights=local_oracle_weights,
                            )
                        )
                        active_oracle_weight = (
                            local_oracle_weights
                            * local_oracle_mask.to(
                                local_oracle_weights.dtype
                            )
                        ).sum()
                        scaled_oracle_ce = _risk_scaled_chunk_loss(
                            local_oracle_ce,
                            active_weight=active_oracle_weight,
                            normalization_tokens=oracle_normalization_tokens,
                        )
                        hard_chunk_loss = (
                            opd_args.hard_oracle_ce_weight
                            * scaled_oracle_ce
                        )
                        accelerator.backward(hard_chunk_loss)
                        batch_total += hard_chunk_loss.detach()
                        hard_oracle_total += scaled_oracle_ce.detach()
                        selected_logits_efficient &= oracle_efficient
                        del oracle_logits

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        (p for p in student.parameters() if p.requires_grad),
                        training_args.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss = batch_total
                distillation_loss = distillation_total
                retention_loss = retention_total
                hard_oracle_ce = hard_oracle_total

            if accelerator.sync_gradients:
                state.global_step += 1
                state.epoch = epoch
                state.step_in_epoch = step_in_epoch + 1
                progress.update(
                    state.global_step,
                    epoch=epoch + 1,
                    batch=f"{step_in_epoch + 1}/{len(dataloader)}",
                    loss=float(loss.detach().float().item()),
                    kd=float(distillation_loss.detach().float().item()),
                    retention=float(retention_loss.detach().float().item()),
                    oracle=float(hard_oracle_ce.detach().float().item()),
                    trusted_fraction=(
                        trusted_tokens / candidate_tokens
                        if candidate_tokens
                        else 0.0
                    ),
                    lr=scheduler.get_last_lr()[0],
                )
                if state.global_step % training_args.logging_steps == 0:
                    logger.log(
                        {
                            "event": "train",
                            "step": state.global_step,
                            "epoch": epoch,
                            "loss": accelerator.gather(loss.detach()).mean().item(),
                            "distillation_loss": accelerator.gather(
                                distillation_loss.detach()
                            )
                            .mean()
                            .item(),
                            "retention_loss": accelerator.gather(
                                retention_loss.detach()
                            )
                            .mean()
                            .item(),
                            "hard_oracle_ce": accelerator.gather(
                                hard_oracle_ce.detach()
                            )
                            .mean()
                            .item(),
                            "candidate_tokens_local": candidate_tokens,
                            "trusted_tokens_local": trusted_tokens,
                            "trusted_fraction_local": (
                                trusted_tokens / candidate_tokens
                                if candidate_tokens
                                else 0.0
                            ),
                            "teacher_top_probability_local": (
                                diagnostic_sums["top_probability"]
                                / max(candidate_tokens, 1)
                            ),
                            "teacher_entropy_local": (
                                diagnostic_sums["entropy"]
                                / max(candidate_tokens, 1)
                            ),
                            "teacher_student_js_local": (
                                diagnostic_sums["js"] / max(candidate_tokens, 1)
                            ),
                            "teacher_margin_local": (
                                diagnostic_sums["margin"]
                                / max(candidate_tokens, 1)
                            ),
                            "teacher_top_log_ratio_local": (
                                diagnostic_sums["top_log_ratio"]
                                / max(candidate_tokens, 1)
                            ),
                            "retention_tokens_local": retention_tokens,
                            "hard_oracle_tokens_local": hard_oracle_tokens,
                            "selected_logits_memory_efficient": (
                                selected_logits_efficient
                            ),
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                if (
                    training_args.save_steps > 0
                    and state.global_step % training_args.save_steps == 0
                ):
                    progress.status(
                        "saving_checkpoint",
                        checkpoint_step=state.global_step,
                    )
                    save_checkpoint(
                        accelerator,
                        output_dir=training_args.output_dir,
                        state=state,
                        extra_state={
                            "method": opd_args.baseline_mode,
                            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                        },
                    )
                    progress.status("checkpoint_saved")
                if should_stop(state.global_step, training_args, total_steps):
                    stop = True
                    break
            else:
                progress.status(
                    "accumulating",
                    epoch=epoch + 1,
                    batch=f"{step_in_epoch + 1}/{len(dataloader)}",
                    accum=(
                        f"{(step_in_epoch % training_args.gradient_accumulation_steps) + 1}"
                        f"/{training_args.gradient_accumulation_steps}"
                    ),
                    loss=float(loss.detach().float().item()),
                    kd=float(distillation_loss.detach().float().item()),
                    retention=float(retention_loss.detach().float().item()),
                    oracle=float(hard_oracle_ce.detach().float().item()),
                )
        state.epoch = epoch + 1
        state.step_in_epoch = 0
        if stop:
            break

    progress.status("saving_final")
    save_final_model(
        accelerator,
        student,
        tokenizer,
        training_args.output_dir,
        metadata={
            "trainer": "opd",
            "method": opd_args.baseline_mode,
            "global_step": state.global_step,
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            "model": asdict(model_args),
            "adapter": asdict(adapter_args),
            "training": asdict(training_args),
            "opd": asdict(opd_args),
        },
    )
    logger.log({"event": "complete", "global_step": state.global_step})
    progress.complete()


def main(argv: Optional[Sequence[str]] = None) -> None:
    model_args, adapter_args, training_args, opd_args = parse_dataclass_args(
        (ModelArguments, AdapterArguments, TrainingArguments, OPDArguments),
        argv=argv,
    )
    train(model_args, adapter_args, training_args, opd_args)


if __name__ == "__main__":
    main()
