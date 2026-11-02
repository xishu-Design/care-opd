"""Reference-anchored DPO baseline on verified recovery preferences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from .collator import DPOCollator
from .losses import dpo_loss, selected_causal_log_probs
from .models import (
    AdapterArguments,
    ModelArguments,
    adapter_disabled,
    load_reference,
    load_student,
    parameter_report,
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
class DPOArguments:
    prompt_field: str = "prompt"
    messages_field: str = "messages"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    beta: float = 0.1
    label_smoothing: float = 0.0
    length_normalize_logps: bool = False


def _sequence_logps(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    label_mask: Any,
    *,
    length_normalize: bool,
) -> Tuple[Any, bool]:
    sequence_logps, _, selected_mask, efficient = selected_causal_log_probs(
        model,
        input_ids,
        attention_mask,
        label_mask,
    )
    if length_normalize:
        lengths = selected_mask.sum(dim=-1).clamp_min(1)
        sequence_logps = sequence_logps / lengths
    return sequence_logps, efficient


def _reference_pair_logps(
    *,
    accelerator: Any,
    student: Any,
    reference: Optional[Any],
    batch: Any,
    length_normalize: bool,
) -> Tuple[Any, Any, bool]:
    import torch

    with torch.no_grad():
        if reference is not None:
            chosen, chosen_efficient = _sequence_logps(
                reference,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_label_mask"],
                length_normalize=length_normalize,
            )
            rejected, rejected_efficient = _sequence_logps(
                reference,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_label_mask"],
                length_normalize=length_normalize,
            )
            return chosen, rejected, chosen_efficient and rejected_efficient

        base_student = accelerator.unwrap_model(student)
        with adapter_disabled(base_student):
            chosen, chosen_efficient = _sequence_logps(
                base_student,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_label_mask"],
                length_normalize=length_normalize,
            )
            rejected, rejected_efficient = _sequence_logps(
                base_student,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_label_mask"],
                length_normalize=length_normalize,
            )
        return chosen, rejected, chosen_efficient and rejected_efficient


def train(
    model_args: ModelArguments,
    adapter_args: AdapterArguments,
    training_args: TrainingArguments,
    dpo_args: DPOArguments,
) -> None:
    if not training_args.train_file:
        raise ValueError("--train_file is required")
    seed_everything(training_args.seed)
    accelerator = create_accelerator(training_args)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(
            "[progress] stage=dpo status=loading_student "
            f"model={model_args.student_model_name_or_path}",
            flush=True,
        )
    student, tokenizer = load_student(
        model_args,
        adapter_args,
        gradient_checkpointing=training_args.gradient_checkpointing,
    )
    if accelerator.is_main_process:
        print("[progress] stage=dpo status=loading_reference", flush=True)
    reference = load_reference(model_args, tokenizer)
    if reference is None and not adapter_args.use_lora:
        raise ValueError(
            "full-model DPO requires --reference_model_name_or_path; "
            "LoRA DPO can use the adapter-disabled base"
        )
    if reference is not None:
        reference.to(accelerator.device)
        reference.eval()

    if accelerator.is_main_process:
        print(
            "[progress] stage=dpo status=loading_data "
            f"path={training_args.train_file}",
            flush=True,
        )
    dataset = JsonlDataset(
        training_args.train_file,
        required_fields=(dpo_args.chosen_field, dpo_args.rejected_field),
    )
    collator = DPOCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        prompt_field=dpo_args.prompt_field,
        messages_field=dpo_args.messages_field,
        chosen_field=dpo_args.chosen_field,
        rejected_field=dpo_args.rejected_field,
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
            "baseline": "dpo",
            "dataset_size": len(dataset),
            "total_steps": total_steps,
            "parameters": parameter_report(accelerator.unwrap_model(student)),
        }
    )
    progress = ProgressReporter(
        stage="dpo",
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
                chosen_logps, chosen_efficient = _sequence_logps(
                    student,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_label_mask"],
                    length_normalize=dpo_args.length_normalize_logps,
                )
                rejected_logps, rejected_efficient = _sequence_logps(
                    student,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_label_mask"],
                    length_normalize=dpo_args.length_normalize_logps,
                )
                (
                    reference_chosen,
                    reference_rejected,
                    reference_efficient,
                ) = _reference_pair_logps(
                    accelerator=accelerator,
                    student=student,
                    reference=reference,
                    batch=batch,
                    length_normalize=dpo_args.length_normalize_logps,
                )
                loss, diagnostics = dpo_loss(
                    chosen_logps,
                    rejected_logps,
                    reference_chosen,
                    reference_rejected,
                    beta=dpo_args.beta,
                    label_smoothing=dpo_args.label_smoothing,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        (p for p in student.parameters() if p.requires_grad),
                        training_args.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                state.global_step += 1
                state.epoch = epoch
                state.step_in_epoch = step_in_epoch + 1
                progress.update(
                    state.global_step,
                    epoch=epoch + 1,
                    batch=f"{step_in_epoch + 1}/{len(dataloader)}",
                    loss=float(loss.detach().float().item()),
                    lr=scheduler.get_last_lr()[0],
                )
                if state.global_step % training_args.logging_steps == 0:
                    payload = {
                        "event": "train",
                        "step": state.global_step,
                        "epoch": epoch,
                        "loss": accelerator.gather(loss.detach()).mean().item(),
                        "selected_logits_memory_efficient": (
                            chosen_efficient
                            and rejected_efficient
                            and reference_efficient
                        ),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    payload.update(
                        {
                            key: accelerator.gather(value).mean().item()
                            for key, value in diagnostics.items()
                        }
                    )
                    logger.log(payload)
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
                        extra_state={"baseline": "dpo"},
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
            "trainer": "dpo",
            "global_step": state.global_step,
            "model": asdict(model_args),
            "adapter": asdict(adapter_args),
            "training": asdict(training_args),
            "dpo": asdict(dpo_args),
        },
    )
    logger.log({"event": "complete", "global_step": state.global_step})
    progress.complete()


def main(argv: Optional[Sequence[str]] = None) -> None:
    model_args, adapter_args, training_args, dpo_args = parse_dataclass_args(
        (ModelArguments, AdapterArguments, TrainingArguments, DPOArguments),
        argv=argv,
    )
    train(model_args, adapter_args, training_args, dpo_args)


if __name__ == "__main__":
    main()
