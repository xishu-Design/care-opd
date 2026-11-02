#!/usr/bin/env python3
"""Tiny multi-GPU regression for CARE-OPD's rank-synchronized chunk plan.

Run with exactly three torchrun ranks.  The ranks deliberately receive
different response/oracle widths, including clean zero-width oracle targets,
then cross metric gathers after gradient accumulation.  No language model or
dataset is loaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from accelerate import Accelerator

from care_opd.train_opd import (
    _forward_dummy_response_token,
    _forward_response_chunk,
    _synchronized_response_widths,
)


class TinyResponseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.head = torch.nn.Linear(8, 64, bias=False)
        # Exercise DDP's forward-time buffer broadcast, which exposed the
        # original collective mismatch in Qwen3.5.
        self.register_buffer("ddp_probe_buffer", torch.arange(64))

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache=False,
        logits_to_keep=None,
    ):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        logits = self.head(hidden)
        logits = logits + self.ddp_probe_buffer[0].to(logits.dtype) * 0.0
        if logits_to_keep is not None:
            logits = logits.index_select(1, logits_to_keep)
        return SimpleNamespace(logits=logits)


def _inputs(width: int, device: torch.device, offset: int):
    prompt_length = 2
    length = prompt_length + width
    input_ids = (
        torch.arange(length, device=device).unsqueeze(0) + offset
    ) % 63 + 1
    return {
        "input_ids": input_ids.long(),
        "attention_mask": torch.ones_like(input_ids),
        "prompt_lengths": torch.tensor(
            [prompt_length], dtype=torch.long, device=device
        ),
        "response_lengths": torch.tensor(
            [width], dtype=torch.long, device=device
        ),
    }


def _one_chunk(
    accelerator: Accelerator,
    model: torch.nn.Module,
    *,
    local_width: int,
    chunk_start: int,
    chunk_end: int,
    active: bool,
    offset: int,
) -> torch.Tensor:
    inputs = _inputs(local_width, accelerator.device, offset)
    if local_width > chunk_start:
        logits, _, _ = _forward_response_chunk(
            model,
            **inputs,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        coefficient = 1.0e-4 if active else 0.0
        return logits.float().square().mean() * coefficient

    logits, _ = _forward_dummy_response_token(
        model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        prompt_lengths=inputs["prompt_lengths"],
    )
    return logits.float().sum() * 0.0


def main() -> None:
    accelerator = Accelerator(gradient_accumulation_steps=2)
    if accelerator.num_processes != 3:
        raise SystemExit(
            f"expected exactly 3 processes, got {accelerator.num_processes}"
        )

    torch.manual_seed(1234)
    model = TinyResponseModel().to(accelerator.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    chunk_size = 4

    for update in range(3):
        for micro_step in range(2):
            if update == 2:
                local_response_width = 0
                local_oracle_width = 0
                local_oracle_active = False
            else:
                pattern = (
                    accelerator.process_index + micro_step + update
                ) % 3
                local_response_width = (1, 5, 9)[pattern]
                local_oracle_width = (0, 7, 2)[pattern]
                local_oracle_active = (False, True, False)[pattern]

            with accelerator.accumulate(model):
                response_width, oracle_width, any_active_oracle = (
                    _synchronized_response_widths(
                        accelerator,
                        response_width=local_response_width,
                        oracle_width=local_oracle_width,
                        has_active_oracle=local_oracle_active,
                    )
                )
                total = torch.zeros((), device=accelerator.device)
                for chunk_start in range(0, response_width, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, response_width)
                    loss = _one_chunk(
                        accelerator,
                        model,
                        local_width=local_response_width,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        active=True,
                        offset=1,
                    )
                    accelerator.backward(loss)
                    total += loss.detach()

                if any_active_oracle and oracle_width > 0:
                    for chunk_start in range(0, oracle_width, chunk_size):
                        chunk_end = min(
                            chunk_start + chunk_size, oracle_width
                        )
                        loss = _one_chunk(
                            accelerator,
                            model,
                            local_width=local_oracle_width,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            active=local_oracle_active,
                            offset=17,
                        )
                        accelerator.backward(loss)
                        total += loss.detach()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        gathered = accelerator.gather(total.reshape(1))
        if accelerator.is_main_process:
            print(
                f"[ddp-smoke] update={update + 1}/3 "
                f"gathered_ranks={gathered.numel()} status=ok",
                flush=True,
            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("DDP_CHUNK_SYNC_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
