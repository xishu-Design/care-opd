"""Data collators for token-aligned OPD, SFT, and DPO training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


DEFAULT_RISK_MULTIPLIERS: Dict[str, float] = {
    "safe": 1.0,
    "low": 1.0,
    "medium": 1.5,
    "high": 2.0,
    "critical": 3.0,
}

DEFAULT_ERROR_MULTIPLIERS: Dict[str, float] = {
    "unsafe_green": 2.0,
    "premature_clearance": 2.0,
    "missed_contraindication": 2.0,
    "ignored_conflict": 1.75,
    "unsupported_claim": 1.5,
    "wrong_source": 1.25,
    "underspecified_query": 1.1,
    "false_red": 0.75,
    "over_escalation": 0.75,
}


def _as_list(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected a single tokenized sequence")
        value = value[0]
    return [int(token) for token in value]


def _render_messages(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    # Dataset exporters may include the supervised assistant turn for standard
    # chat-SFT compatibility.  The collator must never leak it into the prompt.
    prompt_messages = list(messages)
    if prompt_messages and prompt_messages[-1].get("role") == "assistant":
        prompt_messages = prompt_messages[:-1]
    if not hasattr(tokenizer, "apply_chat_template"):
        return (
            "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)
            + "\nassistant:"
        )
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_prompt(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    text_field: str,
    messages_field: str,
) -> str:
    if row.get(messages_field) is not None:
        return _render_messages(tokenizer, row[messages_field])
    value = row.get(text_field)
    if value is None:
        raise KeyError(f"row must contain {text_field!r} or {messages_field!r}")
    return str(value)


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return _as_list(encoded["input_ids"])


def _encode_response_with_offsets(
    tokenizer: Any,
    response: str,
    *,
    add_eos: bool,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = _as_list(encoded["input_ids"])
    raw_offsets = encoded.get("offset_mapping")
    if raw_offsets is not None:
        if isinstance(raw_offsets, torch.Tensor):
            raw_offsets = raw_offsets.tolist()
        if raw_offsets and isinstance(raw_offsets[0], list) and len(raw_offsets) == 1:
            raw_offsets = raw_offsets[0]
        offsets = [(int(start), int(end)) for start, end in raw_offsets]
    else:
        # Slow-tokenizer fallback.  It is conservative at boundaries but keeps
        # component masks usable without requiring a fast tokenizer.
        offsets = []
        cursor = 0
        for index in range(len(token_ids)):
            prefix_len = len(
                _encode(tokenizer, response[: cursor + 1], add_special_tokens=False)
            )
            while cursor < len(response) and prefix_len <= index:
                cursor += 1
                prefix_len = len(
                    _encode(tokenizer, response[: cursor + 1], add_special_tokens=False)
                )
            offsets.append((max(cursor - 1, 0), min(cursor + 1, len(response))))

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if add_eos and eos_id is not None and (not token_ids or token_ids[-1] != eos_id):
        token_ids.append(int(eos_id))
        offsets.append((len(response), len(response)))
    return token_ids, offsets


def _span_bounds(span: Mapping[str, Any]) -> Tuple[int, int]:
    start = span.get("start", span.get("char_start"))
    end = span.get("end", span.get("char_end"))
    if start is None or end is None:
        raise ValueError(f"repair span lacks start/end: {span}")
    return int(start), int(end)


def _token_span_mask(
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Mapping[str, Any]],
    *,
    base_weight: float,
    use_explicit_risk_weight: bool = True,
) -> Tuple[List[bool], List[float]]:
    selected = [False] * len(offsets)
    weights = [0.0] * len(offsets)
    for span in spans:
        start, end = _span_bounds(span)
        if start < 0 or end <= start:
            raise ValueError(f"invalid repair span [{start}, {end})")
        # ``risk_weight`` exported by the verifier is already an absolute
        # consequence weight.  ``weight`` is the optional relative per-span
        # multiplier.  Treating both as multipliers would silently square the
        # configured 1/2/4 risk scale.
        if use_explicit_risk_weight and span.get("risk_weight") is not None:
            span_weight = float(span["risk_weight"])
        else:
            span_weight = base_weight * float(span.get("weight", 1.0))
        for token_index, (token_start, token_end) in enumerate(offsets):
            # Zero-width special tokens are included only by a full-response mask.
            overlaps = token_end > token_start and token_start < end and token_end > start
            if overlaps:
                selected[token_index] = True
                weights[token_index] = max(weights[token_index], span_weight)
    return selected, weights


def _stable_row_identity(row: Mapping[str, Any]) -> str:
    """Return the immutable row key required by deterministic mask controls."""

    for key in ("id", "request_id", "trajectory_id", "seed_id"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    raise ValueError(
        "random_token_matched mask requires id/request_id/trajectory_id/seed_id"
    )


def _random_token_matched_mask(
    offsets: Sequence[Tuple[int, int]],
    selected: Sequence[bool],
    weights: Sequence[float],
    *,
    row_identity: str,
    seed: int,
) -> Tuple[List[bool], List[float]]:
    """Move a span mask to deterministic random response-token positions.

    The control preserves, per example, both the exact number of selected
    non-special response tokens and the multiset of their loss weights.  It is
    constructed after tokenization, so it is genuinely token matched rather
    than merely character-length matched.  Zero-width special tokens (for
    example EOS) are never sampled because verifier spans cannot select them.
    """

    if len(offsets) != len(selected) or len(selected) != len(weights):
        raise ValueError("offsets/selected/weights length mismatch")
    candidates = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start
    ]
    source_indices = [index for index, value in enumerate(selected) if value]
    if any(index not in candidates for index in source_indices):
        raise ValueError("span mask selected a zero-width special token")
    if len(source_indices) > len(candidates):
        raise ValueError("selected-token count exceeds sampleable response tokens")

    def rank(index: int) -> bytes:
        payload = f"{int(seed)}\0{row_identity}\0{index}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    chosen = sorted(candidates, key=lambda index: (rank(index), index))[
        : len(source_indices)
    ]
    # Preserve source-span weight order while mapping it onto hash-ranked
    # random positions.  Sorting weights and positions together would create a
    # spurious low-weight-early/high-weight-late positional pattern.
    source_weights = [float(weights[index]) for index in source_indices]
    random_selected = [False] * len(offsets)
    random_weights = [0.0] * len(offsets)
    for index, weight in zip(chosen, source_weights):
        random_selected[index] = True
        random_weights[index] = weight

    if len(random_selected) != len(selected):
        raise AssertionError("random mask changed response-axis length")
    if sum(random_selected) != len(source_indices):
        raise AssertionError("random mask changed selected-token count")
    if sorted(
        random_weights[index]
        for index, value in enumerate(random_selected)
        if value
    ) != sorted(float(weights[index]) for index in source_indices):
        raise AssertionError("random mask changed positive loss-weight multiset")
    if any(
        value and offsets[index][1] <= offsets[index][0]
        for index, value in enumerate(random_selected)
    ):
        raise AssertionError("random mask selected a zero-width special token")
    return random_selected, random_weights


def _canonical_json(value: Any) -> str:
    """Serialize structured training targets exactly like the data exporter."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _oracle_component_spans(
    oracle_text: str,
    components: Sequence[str],
) -> List[Dict[str, Any]]:
    """Project verified component names onto a canonical oracle action.

    A missing or synthetic component marker falls back to the full oracle
    action.  This is conservative: silently dropping an unprojectable repair
    would turn the auxiliary hard target into a no-op.
    """

    normalized = [str(component) for component in components if component]
    if not normalized or any(component.startswith("__") for component in normalized):
        return [{"start": 0, "end": max(len(oracle_text), 1)}]

    spans: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for component in normalized:
        marker = json.dumps(component, ensure_ascii=False) + ":"
        marker_start = oracle_text.find(marker)
        if marker_start < 0:
            return [{"start": 0, "end": max(len(oracle_text), 1)}]
        value_start = marker_start + len(marker)
        try:
            _, consumed = decoder.raw_decode(oracle_text[value_start:])
        except json.JSONDecodeError:
            return [{"start": 0, "end": max(len(oracle_text), 1)}]
        spans.append(
            {
                "start": value_start,
                "end": value_start + consumed,
                "component": component,
            }
        )
    return spans


def _repair_components(row: Mapping[str, Any]) -> List[str]:
    explicit = row.get("component_mask", row.get("selected_components"))
    if explicit:
        return [str(component) for component in explicit]
    return [
        str(span["component"])
        for span in (row.get("repair_spans") or [])
        if span.get("component")
    ]


def _pad_2d(
    sequences: Sequence[Sequence[int]],
    *,
    pad_value: int,
    dtype: torch.dtype = torch.long,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_length = max((len(sequence) for sequence in sequences), default=0)
    result = torch.full((len(sequences), max_length), pad_value, dtype=dtype)
    attention = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        if sequence:
            result[row_index, : len(sequence)] = torch.tensor(sequence, dtype=dtype)
            attention[row_index, : len(sequence)] = 1
    return result, attention


def _pad_float(
    sequences: Sequence[Sequence[float]],
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    max_length = max((len(sequence) for sequence in sequences), default=0)
    result = torch.full((len(sequences), max_length), pad_value, dtype=torch.float32)
    for row_index, sequence in enumerate(sequences):
        if sequence:
            result[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.float32)
    return result


@dataclass
class OPDCollator:
    """Collate online student rollouts for token-aligned dense distillation."""

    tokenizer: Any
    max_length: int = 4096
    max_response_length: Optional[int] = 384
    mask_mode: str = "span"
    random_mask_seed: int = 20260815
    add_eos: bool = True
    retain_outside_repair: bool = True
    use_risk_weighting: bool = True
    oracle_risk_weighting_mode: str = "inherit"
    risk_multipliers: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RISK_MULTIPLIERS)
    )
    error_multipliers: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ERROR_MULTIPLIERS)
    )

    def __post_init__(self) -> None:
        if self.mask_mode not in {"span", "full", "random_token_matched"}:
            raise ValueError(
                "mask_mode must be 'span', 'full', or 'random_token_matched'"
            )
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if self.oracle_risk_weighting_mode not in {
            "inherit",
            "enabled",
            "disabled",
        }:
            raise ValueError(
                "oracle_risk_weighting_mode must be inherit, enabled, or disabled"
            )

    def _risk_weight_for(
        self, row: Mapping[str, Any], *, enabled: bool
    ) -> float:
        if not enabled:
            return 1.0
        explicit = row.get("risk_weight")
        if explicit is not None:
            base = float(explicit)
        else:
            base = float(self.risk_multipliers.get(str(row.get("risk_level", "safe")), 1.0))
        error = str(row.get("error_type", ""))
        return base * float(self.error_multipliers.get(error, 1.0))

    def _risk_weight(self, row: Mapping[str, Any]) -> float:
        return self._risk_weight_for(row, enabled=self.use_risk_weighting)

    def _oracle_risk_enabled(self) -> bool:
        if self.oracle_risk_weighting_mode == "inherit":
            return self.use_risk_weighting
        return self.oracle_risk_weighting_mode == "enabled"

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("cannot collate an empty batch")

        student_sequences: List[List[int]] = []
        teacher_sequences: List[List[int]] = []
        oracle_sequences: List[List[int]] = []
        prompt_lengths_student: List[int] = []
        prompt_lengths_teacher: List[int] = []
        prompt_lengths_oracle: List[int] = []
        response_lengths: List[int] = []
        oracle_response_lengths: List[int] = []
        distill_masks: List[List[float]] = []
        retention_masks: List[List[float]] = []
        token_weights: List[List[float]] = []
        oracle_masks: List[List[float]] = []
        oracle_token_weights: List[List[float]] = []
        verified: List[float] = []
        confidence: List[float] = []

        for row in rows:
            student_prompt = render_prompt(
                self.tokenizer,
                row,
                text_field="student_prompt",
                messages_field="student_messages",
            )
            teacher_prompt = render_prompt(
                self.tokenizer,
                row,
                text_field="teacher_prompt",
                messages_field="teacher_messages",
            )
            response = row.get("rollout", row.get("student_response", row.get("response")))
            if response is None:
                raise KeyError("OPD row must contain rollout/student_response/response")
            response = str(response)

            full_student_prompt_ids = _encode(
                self.tokenizer,
                student_prompt,
                add_special_tokens=row.get("student_messages") is None,
            )
            student_prompt_ids = list(full_student_prompt_ids)
            teacher_prompt_ids = _encode(
                self.tokenizer,
                teacher_prompt,
                add_special_tokens=row.get("teacher_messages") is None,
            )
            response_ids, offsets = _encode_response_with_offsets(
                self.tokenizer, response, add_eos=self.add_eos
            )

            # Always preserve at least one prompt token.  Long student rollouts
            # are truncated from the right before prompt context is shortened.
            response_cap = self.max_length - 1
            if self.max_response_length is not None:
                response_cap = min(response_cap, self.max_response_length)
            if len(response_ids) > response_cap:
                response_ids = response_ids[:response_cap]
                offsets = offsets[:response_cap]

            def fit_prompt(prompt_ids: List[int]) -> List[int]:
                available = self.max_length - len(response_ids)
                if available < 1:
                    raise ValueError("no room for a causal prompt token")
                return prompt_ids[-available:] if len(prompt_ids) > available else prompt_ids

            student_prompt_ids = fit_prompt(student_prompt_ids)
            teacher_prompt_ids = fit_prompt(teacher_prompt_ids)
            if not student_prompt_ids or not teacher_prompt_ids:
                raise ValueError("tokenizer produced an empty prompt")

            risk_weight = self._risk_weight(row)
            oracle_risk_enabled = self._oracle_risk_enabled()
            oracle_risk_weight = self._risk_weight_for(
                row, enabled=oracle_risk_enabled
            )
            spans = row.get("repair_spans", row.get("component_spans", [])) or []
            if self.mask_mode == "full":
                selected = [True] * len(response_ids)
                weights = [risk_weight] * len(response_ids)
            else:
                selected, weights = _token_span_mask(
                    offsets,
                    spans,
                    base_weight=risk_weight,
                    use_explicit_risk_weight=self.use_risk_weighting,
                )
                if self.mask_mode == "random_token_matched":
                    selected, weights = _random_token_matched_mask(
                        offsets,
                        selected,
                        weights,
                        row_identity=_stable_row_identity(row),
                        seed=self.random_mask_seed,
                    )

            clean_control = bool(row.get("clean_control", False))
            if clean_control:
                retention = [True] * len(response_ids)
            elif self.retain_outside_repair:
                retention = [not value for value in selected]
            else:
                retention = [False] * len(response_ids)

            # qzbc-native post-training supervises a GeneratedResponse rather
            # than CARE-OPD's simulator Action.  Keep the old field as a
            # backwards-compatible fallback for benchmark experiments.
            oracle_value = row.get("oracle_response")
            if oracle_value is None:
                oracle_value = row.get("oracle_action")
            if oracle_value is None:
                privileged = row.get("privileged_teacher_context")
                if isinstance(privileged, Mapping):
                    oracle_value = privileged.get(
                        "verified_oracle_response",
                        privileged.get("verified_oracle_action"),
                    )
            if oracle_value is None:
                oracle_prompt_ids = list(full_student_prompt_ids[-self.max_length :])
                oracle_response_ids: List[int] = []
                oracle_selected: List[bool] = []
                oracle_weights: List[float] = []
            else:
                if isinstance(oracle_value, str):
                    try:
                        parsed_oracle = json.loads(oracle_value)
                    except json.JSONDecodeError as exc:
                        raise ValueError("oracle_response must be valid JSON") from exc
                    if not isinstance(parsed_oracle, Mapping):
                        raise TypeError("oracle_response JSON must be an object")
                    oracle_text = _canonical_json(parsed_oracle)
                elif isinstance(oracle_value, Mapping):
                    oracle_text = _canonical_json(oracle_value)
                else:
                    raise TypeError(
                        "oracle_response/oracle_action must be a JSON string or mapping"
                    )
                oracle_response_ids, oracle_offsets = _encode_response_with_offsets(
                    self.tokenizer,
                    oracle_text,
                    add_eos=self.add_eos,
                )
                response_cap = self.max_length - 1
                if self.max_response_length is not None:
                    response_cap = min(response_cap, self.max_response_length)
                oracle_response_ids = oracle_response_ids[:response_cap]
                oracle_offsets = oracle_offsets[:response_cap]
                oracle_available = self.max_length - len(oracle_response_ids)
                oracle_prompt_ids = full_student_prompt_ids[-oracle_available:]
                oracle_spans = _oracle_component_spans(
                    oracle_text,
                    _repair_components(row),
                )
                oracle_selected, oracle_weights = _token_span_mask(
                    oracle_offsets,
                    oracle_spans,
                    base_weight=oracle_risk_weight,
                    use_explicit_risk_weight=oracle_risk_enabled,
                )

            student_sequences.append(student_prompt_ids + response_ids)
            teacher_sequences.append(teacher_prompt_ids + response_ids)
            oracle_sequences.append(oracle_prompt_ids + oracle_response_ids)
            prompt_lengths_student.append(len(student_prompt_ids))
            prompt_lengths_teacher.append(len(teacher_prompt_ids))
            prompt_lengths_oracle.append(len(oracle_prompt_ids))
            response_lengths.append(len(response_ids))
            oracle_response_lengths.append(len(oracle_response_ids))
            distill_masks.append([float(value) for value in selected])
            retention_masks.append([float(value) for value in retention])
            token_weights.append(weights)
            oracle_masks.append([float(value) for value in oracle_selected])
            oracle_token_weights.append(oracle_weights)
            # Eligibility is certified by the executable host intervention.
            # It is deliberately distinct from verifying a sampled/greedy
            # teacher output, which dense logit distillation does not require.
            host_verified = row.get("host_repair_verified")
            if host_verified is None:
                host_verified = row.get("host_verified", False)
            verified.append(float(bool(host_verified)))
            confidence.append(
                float(
                    row.get(
                        "host_verification_confidence",
                        0.0,
                    )
                )
            )

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        student_ids, student_attention = _pad_2d(
            student_sequences, pad_value=int(pad_id)
        )
        teacher_ids, teacher_attention = _pad_2d(
            teacher_sequences, pad_value=int(pad_id)
        )
        oracle_ids, oracle_attention = _pad_2d(
            oracle_sequences, pad_value=int(pad_id)
        )
        return {
            "student_input_ids": student_ids,
            "student_attention_mask": student_attention,
            "teacher_input_ids": teacher_ids,
            "teacher_attention_mask": teacher_attention,
            "oracle_input_ids": oracle_ids,
            "oracle_attention_mask": oracle_attention,
            "student_prompt_lengths": torch.tensor(
                prompt_lengths_student, dtype=torch.long
            ),
            "teacher_prompt_lengths": torch.tensor(
                prompt_lengths_teacher, dtype=torch.long
            ),
            "oracle_prompt_lengths": torch.tensor(
                prompt_lengths_oracle, dtype=torch.long
            ),
            "response_lengths": torch.tensor(response_lengths, dtype=torch.long),
            "oracle_response_lengths": torch.tensor(
                oracle_response_lengths, dtype=torch.long
            ),
            "distill_mask": _pad_float(distill_masks).bool(),
            "retention_mask": _pad_float(retention_masks).bool(),
            "token_weights": _pad_float(token_weights),
            "oracle_mask": _pad_float(oracle_masks).bool(),
            "oracle_token_weights": _pad_float(oracle_token_weights),
            "host_verified": torch.tensor(verified, dtype=torch.bool),
            "host_verification_confidence": torch.tensor(
                confidence, dtype=torch.float32
            ),
        }


@dataclass
class SFTCollator:
    tokenizer: Any
    max_length: int = 4096
    prompt_field: str = "prompt"
    messages_field: str = "messages"
    target_field: str = "target"
    add_eos: bool = True

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        sequences: List[List[int]] = []
        label_masks: List[List[int]] = []
        for row in rows:
            prompt = render_prompt(
                self.tokenizer,
                row,
                text_field=self.prompt_field,
                messages_field=self.messages_field,
            )
            target = row.get(self.target_field)
            if target is None:
                raise KeyError(f"SFT row lacks target field {self.target_field!r}")
            prompt_ids = _encode(
                self.tokenizer,
                prompt,
                add_special_tokens=row.get(self.messages_field) is None,
            )
            target_ids, _ = _encode_response_with_offsets(
                self.tokenizer, str(target), add_eos=self.add_eos
            )
            if len(target_ids) > self.max_length - 1:
                target_ids = target_ids[: self.max_length - 1]
            available = self.max_length - len(target_ids)
            prompt_ids = prompt_ids[-available:]
            sequence = prompt_ids + target_ids
            sequences.append(sequence)
            label_masks.append([0] * len(prompt_ids) + [1] * len(target_ids))

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        input_ids, attention = _pad_2d(sequences, pad_value=int(pad_id))
        labels, _ = _pad_2d(label_masks, pad_value=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "label_mask": labels.bool(),
        }


@dataclass
class DPOCollator:
    tokenizer: Any
    max_length: int = 4096
    prompt_field: str = "prompt"
    messages_field: str = "messages"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    add_eos: bool = True

    def _build(
        self, prompt_ids: List[int], response: str
    ) -> Tuple[List[int], List[int]]:
        response_ids, _ = _encode_response_with_offsets(
            self.tokenizer, response, add_eos=self.add_eos
        )
        if len(response_ids) > self.max_length - 1:
            response_ids = response_ids[: self.max_length - 1]
        fitted_prompt = prompt_ids[-(self.max_length - len(response_ids)) :]
        return (
            fitted_prompt + response_ids,
            [0] * len(fitted_prompt) + [1] * len(response_ids),
        )

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_sequences: List[List[int]] = []
        rejected_sequences: List[List[int]] = []
        chosen_masks: List[List[int]] = []
        rejected_masks: List[List[int]] = []
        for row in rows:
            prompt = render_prompt(
                self.tokenizer,
                row,
                text_field=self.prompt_field,
                messages_field=self.messages_field,
            )
            if row.get(self.chosen_field) is None or row.get(self.rejected_field) is None:
                raise KeyError("DPO row must contain chosen and rejected responses")
            prompt_ids = _encode(
                self.tokenizer,
                prompt,
                add_special_tokens=row.get(self.messages_field) is None,
            )
            chosen, chosen_mask = self._build(
                prompt_ids, str(row[self.chosen_field])
            )
            rejected, rejected_mask = self._build(
                prompt_ids, str(row[self.rejected_field])
            )
            chosen_sequences.append(chosen)
            rejected_sequences.append(rejected)
            chosen_masks.append(chosen_mask)
            rejected_masks.append(rejected_mask)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        chosen_ids, chosen_attention = _pad_2d(
            chosen_sequences, pad_value=int(pad_id)
        )
        rejected_ids, rejected_attention = _pad_2d(
            rejected_sequences, pad_value=int(pad_id)
        )
        chosen_label_mask, _ = _pad_2d(chosen_masks, pad_value=0)
        rejected_label_mask, _ = _pad_2d(rejected_masks, pad_value=0)
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_attention,
            "chosen_label_mask": chosen_label_mask.bool(),
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_attention,
            "rejected_label_mask": rejected_label_mask.bool(),
        }
