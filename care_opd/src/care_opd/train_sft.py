"""Clean expert-SFT and direct-revision-SFT baseline trainer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from .collator import SFTCollator
from .losses import selected_causal_log_probs
from .models import AdapterArguments, ModelArguments, load_student, parameter_report
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
class SFTArguments:
    baseline: str = "clean_sft"
    prompt_field: str = "prompt"
    messages_field: str = "messages"
    target_field: Optional[str] = None
    add_eos: bool = True

    def resolved_target_field(self) -> str:
        if self.target_field:
            return self.target_field
        if self.baseline == "clean_sft":
            return "target"
        if self.baseline == "direct_revision_sft":
            return "revised_response"
        raise ValueError(
            "baseline must be clean_sft or direct_revision_sft, "
            "or provide --target_field explicitly"
        )


def train(
    model_args: ModelArguments,
    adapter_args: AdapterArguments,
    training_args: TrainingArguments,
    sft_args: SFTArguments,
) -> None:
    if not training_args.train_file:
        raise ValueError("--train_file is required")
    seed_everything(training_args.seed)
    accelerator = create_accelerator(training_args)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(
            "[progress] "
            f"stage=sft:{sft_args.baseline} status=loading_student "
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
            f"stage=sft:{sft_args.baseline} status=loading_data "
            f"path={training_args.train_file}",
            flush=True,
        )
    dataset = JsonlDataset(training_args.train_file)
    collator = SFTCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        prompt_field=sft_args.prompt_field,
        messages_field=sft_args.messages_field,
        target_field=sft_args.resolved_target_field(),
        add_eos=sft_args.add_eos,
    )

    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(training_args.seed)
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
    if accelerator.is_main_process:
        logger.log(
            {
                "event": "start",
                "baseline": sft_args.baseline,
                "dataset_size": len(dataset),
                "total_steps": total_steps,
                "parameters": parameter_report(accelerator.unwrap_model(student)),
            }
        )
    progress = ProgressReporter(
        stage=f"sft:{sft_args.baseline}",
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
                _, token_logps, selected_mask, selected_logits_efficient = (
                    selected_causal_log_probs(
                        student,
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["label_mask"],
                    )
                )
                loss = -(
                    token_logps.sum()
                    / selected_mask.sum().to(token_logps.dtype).clamp_min(1.0)
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
                    mean_loss = accelerator.gather(loss.detach()).mean().item()
                    logger.log(
                        {
                            "event": "train",
                            "step": state.global_step,
                            "epoch": epoch,
                            "loss": mean_loss,
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
                        extra_state={"baseline": sft_args.baseline},
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
            "trainer": "sft",
            "baseline": sft_args.baseline,
            "global_step": state.global_step,
            "model": asdict(model_args),
            "adapter": asdict(adapter_args),
            "training": asdict(training_args),
            "sft": asdict(sft_args),
        },
    )
    logger.log({"event": "complete", "global_step": state.global_step})
    progress.complete()


def main(argv: Optional[Sequence[str]] = None) -> None:
    model_args, adapter_args, training_args, sft_args = parse_dataclass_args(
        (ModelArguments, AdapterArguments, TrainingArguments, SFTArguments),
        argv=argv,
    )
    train(model_args, adapter_args, training_args, sft_args)


if __name__ == "__main__":
    main()
