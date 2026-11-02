"""Numerically stable objectives used by CARE-OPD and its baselines."""

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

KLMode = Literal["forward_kl", "reverse_kl", "js"]


def supports_selected_logits(model: Any) -> bool:
    """Return whether the wrapped model exposes ``logits_to_keep``.

    Qwen3.5 has a large vocabulary.  Asking it to materialize logits for every
    prompt token is unnecessary for response-only SFT/DPO and can dominate
    memory.  The helper unwraps DDP/PEFT before inspecting the base signature.
    """

    target = getattr(model, "module", model)
    if hasattr(target, "get_base_model"):
        target = target.get_base_model()
    try:
        return "logits_to_keep" in inspect.signature(target.forward).parameters
    except (TypeError, ValueError):
        return False


def selected_causal_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Score only causal prediction positions selected by ``label_mask``.

    Returns ``(sequence_sums, token_log_probs, selected_mask, efficient)``.
    The token tensors use the union of supervised absolute positions across the
    batch, so variable prompt lengths remain aligned without full-sequence
    logits.  Models without ``logits_to_keep`` fall back to their full output
    and report ``efficient=False`` for the run audit.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [B, T]")
    if attention_mask.shape != input_ids.shape or label_mask.shape != input_ids.shape:
        raise ValueError("attention_mask and label_mask must match input_ids")
    if input_ids.shape[1] < 2:
        raise ValueError("causal scoring requires at least two tokens")

    prediction_mask = label_mask[:, 1:].bool() & attention_mask[:, 1:].bool()
    absolute_positions = torch.arange(
        input_ids.shape[1] - 1, device=input_ids.device
    )
    union = absolute_positions[prediction_mask.any(dim=0)]
    if union.numel() == 0:
        raise ValueError("label_mask selects no causal target tokens")

    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    advertised = supports_selected_logits(model)
    if advertised:
        kwargs["logits_to_keep"] = union
    output = model(**kwargs)
    logits = output.logits
    if logits.shape[1] == input_ids.shape[1]:
        selected_logits = logits.index_select(1, union)
        efficient = False
    elif logits.shape[1] == union.numel():
        selected_logits = logits
        efficient = True
    else:
        raise RuntimeError(
            "unexpected logits sequence dimension: "
            f"{logits.shape[1]} (input={input_ids.shape[1]}, selected={union.numel()})"
        )

    selected_targets = input_ids.index_select(1, union + 1)
    selected_mask = prediction_mask.index_select(1, union)
    token_log_probs = torch.nn.functional.log_softmax(
        selected_logits.float(), dim=-1
    ).gather(-1, selected_targets.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * selected_mask.to(token_log_probs.dtype)
    return (
        token_log_probs.sum(dim=-1),
        token_log_probs,
        selected_mask,
        efficient,
    )


def sequence_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Return a boolean ``[batch, time]`` mask from integer lengths."""

    if lengths.ndim != 1:
        raise ValueError(f"lengths must be rank 1, got {tuple(lengths.shape)}")
    if max_length is None:
        max_length = int(lengths.max().item()) if lengths.numel() else 0
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def gather_response_logits(
    logits: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
    max_response_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align causal-LM logits to response tokens.

    Input examples are right padded and have the form ``prompt + response``.
    The logit that predicts response token zero is located at
    ``prompt_length - 1``.  Student and teacher prompts may have different
    lengths; applying this function to each side makes the response-token axes
    exactly align.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}")
    batch, time, vocab = logits.shape
    if prompt_lengths.shape != (batch,) or response_lengths.shape != (batch,):
        raise ValueError("prompt_lengths/response_lengths must match logits batch")
    if torch.any(prompt_lengths < 1):
        raise ValueError("every prompt must contain at least one token")
    if max_response_length is None:
        max_response_length = int(response_lengths.max().item())
    offsets = torch.arange(max_response_length, device=logits.device)
    indices = prompt_lengths.to(logits.device).unsqueeze(1) - 1 + offsets.unsqueeze(0)
    valid = sequence_mask(response_lengths.to(logits.device), max_response_length)
    valid &= indices >= 0
    valid &= indices < time
    safe_indices = indices.clamp(0, max(time - 1, 0))
    aligned = logits.gather(1, safe_indices.unsqueeze(-1).expand(batch, max_response_length, vocab))
    return aligned, valid


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"value/mask shape mismatch: {values.shape} vs {mask.shape}")
    effective = mask.to(values.dtype)
    if weights is not None:
        if weights.shape != values.shape:
            raise ValueError(f"weight shape mismatch: {weights.shape} vs {values.shape}")
        effective = effective * weights.to(values.dtype)
    denominator = effective.sum().clamp_min(eps)
    return (values * effective).sum() / denominator


def token_distribution_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    mode: KLMode = "reverse_kl",
    temperature: float = 1.0,
    js_beta: float = 0.5,
) -> torch.Tensor:
    """Return a dense per-token divergence with shape ``[B, R]``.

    ``forward_kl`` is :math:`KL(p_T || p_S)`, ``reverse_kl`` is
    :math:`KL(p_S || p_T)`, and ``js`` is a generalized Jensen-Shannon
    divergence.  The standard distillation ``T^2`` scaling is included.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits must align exactly, got "
            f"{student_logits.shape} and {teacher_logits.shape}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 < js_beta < 1.0:
        raise ValueError("js_beta must lie strictly between 0 and 1")

    # Distribution arithmetic in float32 is important under bf16/4-bit training.
    s_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    s_prob = s_log.exp()
    t_prob = t_log.exp()

    if mode == "forward_kl":
        divergence = (t_prob * (t_log - s_log)).sum(dim=-1)
    elif mode == "reverse_kl":
        divergence = (s_prob * (s_log - t_log)).sum(dim=-1)
    elif mode == "js":
        # beta weights the teacher component.
        mixture = js_beta * t_prob + (1.0 - js_beta) * s_prob
        mixture_log = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
        teacher_term = (t_prob * (t_log - mixture_log)).sum(dim=-1)
        student_term = (s_prob * (s_log - mixture_log)).sum(dim=-1)
        divergence = js_beta * teacher_term + (1.0 - js_beta) * student_term
    else:
        raise ValueError(f"unknown KL mode: {mode}")
    return divergence * (temperature**2)


def dense_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    token_weights: Optional[torch.Tensor] = None,
    mode: KLMode = "reverse_kl",
    temperature: float = 1.0,
    js_beta: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute masked dense distillation and return ``(loss, token_losses)``."""

    token_losses = token_distribution_divergence(
        student_logits,
        teacher_logits.detach(),
        mode=mode,
        temperature=temperature,
        js_beta=js_beta,
    )
    loss = _masked_weighted_mean(token_losses, token_mask, token_weights)
    return loss, token_losses.detach()


def masked_hard_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    token_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross entropy for verified hard targets on selected causal components."""

    if logits.ndim != 3:
        raise ValueError("logits must be [B, R, V]")
    if targets.shape != logits.shape[:2] or token_mask.shape != targets.shape:
        raise ValueError("targets/token_mask must match logits [B, R]")
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    loss = _masked_weighted_mean(
        token_losses,
        token_mask,
        token_weights,
    )
    return loss, token_losses.detach()


@dataclass(frozen=True)
class TrustDiagnostics:
    candidate_tokens: int
    retained_tokens: int
    retained_fraction: float
    mean_teacher_top_probability: float
    mean_teacher_normalized_entropy: float
    mean_teacher_student_js: float
    mean_teacher_margin: float
    mean_teacher_top_log_ratio: float


def teacher_trust_mask(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    sample_verified: Optional[torch.Tensor] = None,
    sample_confidence: Optional[torch.Tensor] = None,
    min_sample_confidence: float = 0.0,
    min_teacher_top_probability: float = 0.0,
    min_teacher_margin: float = float("-inf"),
    max_teacher_normalized_entropy: float = 1.0,
    max_teacher_student_js: float = float("inf"),
    max_teacher_top_log_ratio: float = float("inf"),
) -> Tuple[torch.Tensor, TrustDiagnostics]:
    """Build a conservative token-level trust region for teacher supervision.

    The mask combines host verification, sample confidence, teacher sharpness,
    and an optional teacher/student mismatch cap.  Mismatch gating is disabled
    by default because disagreement is also the source of useful supervision.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if candidate_mask.shape != student_logits.shape[:-1]:
        raise ValueError("candidate_mask must be [B, R]")

    with torch.no_grad():
        t_log = F.log_softmax(teacher_logits.float(), dim=-1)
        t_prob = t_log.exp()
        top_probability = t_prob.max(dim=-1).values
        top_values, top_indices = t_prob.topk(k=min(2, t_prob.shape[-1]), dim=-1)
        if top_values.shape[-1] == 1:
            teacher_margin = top_values[..., 0]
        else:
            teacher_margin = top_values[..., 0] - top_values[..., 1]
        vocab_size = teacher_logits.shape[-1]
        entropy = -(t_prob * t_log).sum(dim=-1)
        normalized_entropy = entropy / max(math.log(max(vocab_size, 2)), 1e-8)

        s_log = F.log_softmax(student_logits.detach().float(), dim=-1)
        s_prob = s_log.exp()
        mixture = 0.5 * (t_prob + s_prob)
        mixture_log = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
        js = 0.5 * (
            (t_prob * (t_log - mixture_log)).sum(dim=-1)
            + (s_prob * (s_log - mixture_log)).sum(dim=-1)
        )
        teacher_top_logp = t_log.gather(-1, top_indices[..., :1]).squeeze(-1)
        student_on_teacher_top_logp = s_log.gather(
            -1, top_indices[..., :1]
        ).squeeze(-1)
        teacher_top_log_ratio = (
            teacher_top_logp - student_on_teacher_top_logp
        ).abs()

        trusted = candidate_mask.bool()
        trusted &= top_probability >= min_teacher_top_probability
        trusted &= teacher_margin >= min_teacher_margin
        trusted &= normalized_entropy <= max_teacher_normalized_entropy
        if math.isfinite(max_teacher_student_js):
            trusted &= js <= max_teacher_student_js
        if math.isfinite(max_teacher_top_log_ratio):
            trusted &= teacher_top_log_ratio <= max_teacher_top_log_ratio
        if sample_verified is not None:
            trusted &= sample_verified.to(trusted.device).bool().unsqueeze(1)
        if sample_confidence is not None:
            trusted &= (
                sample_confidence.to(top_probability.device).unsqueeze(1)
                >= min_sample_confidence
            )

        candidates = int(candidate_mask.sum().item())
        retained = int(trusted.sum().item())
        diagnostic_mask = candidate_mask.bool()

        def masked_average(value: torch.Tensor) -> float:
            if not diagnostic_mask.any():
                return 0.0
            return float(value[diagnostic_mask].mean().item())

        diagnostics = TrustDiagnostics(
            candidate_tokens=candidates,
            retained_tokens=retained,
            retained_fraction=(retained / candidates if candidates else 0.0),
            mean_teacher_top_probability=masked_average(top_probability),
            mean_teacher_normalized_entropy=masked_average(normalized_entropy),
            mean_teacher_student_js=masked_average(js),
            mean_teacher_margin=masked_average(teacher_margin),
            mean_teacher_top_log_ratio=masked_average(teacher_top_log_ratio),
        )
    return trusted, diagnostics


def causal_token_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    label_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sequence sums and token log-probabilities for causal LM outputs."""

    if logits.shape[:2] != input_ids.shape or input_ids.shape != label_mask.shape:
        raise ValueError("logits, input_ids, and label_mask sequence shapes must match")
    shifted_logits = logits[:, :-1].float()
    shifted_targets = input_ids[:, 1:]
    shifted_mask = label_mask[:, 1:].bool()
    token_log_probs = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, shifted_targets.unsqueeze(-1)
    ).squeeze(-1)
    token_log_probs = token_log_probs * shifted_mask.to(token_log_probs.dtype)
    return token_log_probs.sum(dim=-1), token_log_probs


def masked_causal_cross_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    """Causal cross entropy normalized by the number of supervised tokens."""

    sequence_sums, token_log_probs = causal_token_log_probs(logits, input_ids, label_mask)
    del sequence_sums
    shifted_mask = label_mask[:, 1:].to(token_log_probs.dtype)
    return -(token_log_probs.sum() / shifted_mask.sum().clamp_min(1.0))


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Standard reference-anchored DPO objective."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    reference_ratio = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_ratio - reference_ratio)
    positive = -F.logsigmoid(logits)
    negative = -F.logsigmoid(-logits)
    loss = ((1.0 - label_smoothing) * positive + label_smoothing * negative).mean()
    diagnostics = {
        "preference_accuracy": (logits > 0).float().mean().detach(),
        "preference_margin": logits.mean().detach(),
        "chosen_reward": (beta * (policy_chosen_logps - reference_chosen_logps))
        .mean()
        .detach(),
        "rejected_reward": (beta * (policy_rejected_logps - reference_rejected_logps))
        .mean()
        .detach(),
    }
    return loss, diagnostics
