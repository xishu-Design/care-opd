from types import SimpleNamespace

import pytest
import torch

from care_opd.collator import OPDCollator, _random_token_matched_mask
from care_opd.train_opd import (
    OPDArguments,
    _forward_response_chunk,
    _pad_response_axis,
    _response_chunk_targets,
    _risk_scaled_chunk_loss,
    _synchronized_response_widths,
)


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3
    padding_side = "right"

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        ids = [ord(character) + 10 for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode_ids(self, ids):
        return "".join(
            chr(token - 10)
            for token in ids
            if token not in {self.pad_token_id, self.eos_token_id, self.bos_token_id}
        )


def _opd_row(**overrides):
    rollout = '{"decision":"clear","kind":"decide"}'
    value_start = rollout.index('"clear"')
    row = {
        "student_prompt": "PUBLIC_STATE",
        "teacher_prompt": "PUBLIC_STATE<PRIVILEGED>SECRET</PRIVILEGED>",
        "rollout": rollout,
        "repair_spans": [
            {
                "start": value_start,
                "end": value_start + len('"clear"'),
                "component": "decision",
                "risk_weight": 4.0,
            }
        ],
        "risk_weight": 4.0,
        "host_repair_verified": True,
        "host_verification_confidence": 1.0,
        "teacher_verified": False,
        "oracle_action": {"decision": "block", "kind": "decide"},
    }
    row.update(overrides)
    return row


def test_opd_collator_builds_causal_oracle_target_without_prompt_leakage():
    tokenizer = _CharacterTokenizer()
    batch = OPDCollator(
        tokenizer,
        max_length=256,
        max_response_length=128,
    )([_opd_row()])

    student_text = tokenizer.decode_ids(batch["student_input_ids"][0].tolist())
    teacher_text = tokenizer.decode_ids(batch["teacher_input_ids"][0].tolist())
    oracle_text = tokenizer.decode_ids(batch["oracle_input_ids"][0].tolist())

    assert "SECRET" not in student_text
    assert '"clear"' in student_text
    assert "SECRET" in teacher_text
    assert '"block"' in oracle_text
    assert '"clear"' not in oracle_text
    assert batch["host_verified"].tolist() == [True]

    oracle_response = oracle_text[len("PUBLIC_STATE") :]
    selected_characters = [
        character
        for character, selected in zip(
            oracle_response,
            batch["oracle_mask"][0].tolist(),
        )
        if selected
    ]
    assert "".join(selected_characters) == '"block"'
    assert set(
        batch["oracle_token_weights"][0][batch["oracle_mask"][0]].tolist()
    ) == {4.0}


def test_host_certificate_gate_is_fail_closed_and_supports_both_final_scopes():
    collator = OPDCollator(
        _CharacterTokenizer(),
        max_length=256,
        max_response_length=128,
    )
    batch = collator(
        [
            _opd_row(
                host_repair_verified=True,
                host_verification_confidence=0.8,
            ),
            _opd_row(
                host_repair_verified=None,
                host_verification_confidence=1.0,
                teacher_verified=True,
            ),
            _opd_row(
                host_repair_verified=None,
                host_verified=True,
                host_verification_confidence=1.0,
                teacher_verified=True,
            ),
        ]
    )
    assert batch["host_verified"].tolist() == [True, False, True]
    assert batch["host_verification_confidence"].tolist() == [
        pytest.approx(0.8),
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]


def test_risk_ablation_ignores_row_and_span_consequence_weights():
    batch = OPDCollator(
        _CharacterTokenizer(),
        max_length=256,
        max_response_length=128,
        use_risk_weighting=False,
    )([_opd_row()])

    assert set(batch["token_weights"][0][batch["distill_mask"][0]].tolist()) == {
        1.0
    }
    assert set(
        batch["oracle_token_weights"][0][batch["oracle_mask"][0]].tolist()
    ) == {1.0}


def test_oracle_risk_weighting_can_be_enabled_independently_of_distillation():
    batch = OPDCollator(
        _CharacterTokenizer(),
        max_length=256,
        max_response_length=128,
        use_risk_weighting=False,
        oracle_risk_weighting_mode="enabled",
    )([_opd_row()])

    assert set(batch["token_weights"][0][batch["distill_mask"][0]].tolist()) == {
        1.0
    }
    assert set(
        batch["oracle_token_weights"][0][batch["oracle_mask"][0]].tolist()
    ) == {4.0}


def test_oracle_risk_weighting_inherit_preserves_legacy_behavior():
    enabled = OPDCollator(
        _CharacterTokenizer(),
        max_length=256,
        max_response_length=128,
        use_risk_weighting=True,
        oracle_risk_weighting_mode="inherit",
    )([_opd_row()])
    disabled = OPDCollator(
        _CharacterTokenizer(),
        max_length=256,
        max_response_length=128,
        use_risk_weighting=False,
        oracle_risk_weighting_mode="inherit",
    )([_opd_row()])

    assert set(
        enabled["oracle_token_weights"][0][enabled["oracle_mask"][0]].tolist()
    ) == {4.0}
    assert set(
        disabled["oracle_token_weights"][0][disabled["oracle_mask"][0]].tolist()
    ) == {1.0}


def test_random_token_matched_mask_preserves_count_and_weight_multiset():
    offsets = [(index, index + 1) for index in range(12)] + [(12, 12)]
    selected = [False] * len(offsets)
    weights = [0.0] * len(offsets)
    for index, weight in zip((1, 2, 7), (4.0, 2.0, 4.0)):
        selected[index] = True
        weights[index] = weight

    first_selected, first_weights = _random_token_matched_mask(
        offsets,
        selected,
        weights,
        row_identity="REQUEST_001",
        seed=20260815,
    )
    second_selected, second_weights = _random_token_matched_mask(
        offsets,
        selected,
        weights,
        row_identity="REQUEST_001",
        seed=20260815,
    )

    assert first_selected == second_selected
    assert first_weights == second_weights
    assert sum(first_selected) == sum(selected) == 3
    assert sorted(weight for weight in first_weights if weight) == [2.0, 4.0, 4.0]
    assert not first_selected[-1]  # zero-width EOS/special position


def test_random_token_matched_collator_is_exact_and_fail_closed_without_id():
    tokenizer = _CharacterTokenizer()
    row = _opd_row(id="REQUEST_002")
    span_batch = OPDCollator(
        tokenizer,
        max_length=256,
        max_response_length=128,
        mask_mode="span",
    )([row])
    random_batch = OPDCollator(
        tokenizer,
        max_length=256,
        max_response_length=128,
        mask_mode="random_token_matched",
        random_mask_seed=7,
    )([row])

    span_mask = span_batch["distill_mask"][0]
    random_mask = random_batch["distill_mask"][0]
    assert int(span_mask.sum()) == int(random_mask.sum())
    assert sorted(
        span_batch["token_weights"][0][span_mask].tolist()
    ) == sorted(random_batch["token_weights"][0][random_mask].tolist())
    assert torch.equal(
        random_batch["retention_mask"][0],
        ~random_mask,
    )

    row_without_id = _opd_row()
    with pytest.raises(ValueError, match="requires id"):
        OPDCollator(
            tokenizer,
            max_length=256,
            max_response_length=128,
            mask_mode="random_token_matched",
        )([row_without_id])


class _SelectedResponseModel(torch.nn.Module):
    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache=False,
        logits_to_keep=None,
    ):
        del attention_mask, use_cache
        time = input_ids.shape[1]
        vocab = 5
        logits = (
            torch.arange(time, dtype=torch.float32)
            .view(1, time, 1)
            .expand(input_ids.shape[0], time, vocab)
        )
        if logits_to_keep is not None:
            logits = logits.index_select(1, logits_to_keep)
        return SimpleNamespace(logits=logits)


def test_response_chunk_helpers_align_variable_prompt_lengths():
    input_ids = torch.tensor(
        [
            [10, 11, 20, 21, 22, 0],
            [12, 13, 14, 30, 31, 0],
        ]
    )
    attention = input_ids.ne(0).long()
    prompt_lengths = torch.tensor([2, 3])
    response_lengths = torch.tensor([3, 2])

    logits, valid, efficient = _forward_response_chunk(
        _SelectedResponseModel(),
        input_ids=input_ids,
        attention_mask=attention,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        chunk_start=0,
        chunk_end=3,
    )
    targets, target_valid = _response_chunk_targets(
        input_ids=input_ids,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        chunk_start=0,
        chunk_end=3,
    )

    assert efficient
    assert valid.tolist() == [[True, True, True], [True, True, False]]
    assert target_valid.tolist() == valid.tolist()
    assert targets.tolist() == [[20, 21, 22], [30, 31, 0]]
    assert logits[:, :, 0].tolist() == [[1.0, 2.0, 3.0], [2.0, 3.0, 3.0]]


def test_non_care_baselines_disable_hard_oracle_auxiliary():
    args = OPDArguments(
        baseline_mode="gkd_full",
        hard_oracle_ce_weight=0.1,
    ).apply_baseline_bundle()
    assert args.hard_oracle_ce_weight == 0.0
    assert OPDArguments().response_chunk_size == 16


class _ThreeRankWidthAccelerator:
    device = torch.device("cpu")

    def gather(self, local_widths):
        assert local_widths.tolist() == [5, 0, 0]
        return torch.tensor(
            [5, 0, 0, 3, 7, 1, 9, 2, 0], dtype=torch.long
        )


def test_ddp_chunk_plan_uses_rank_global_widths_and_pads_clean_oracle():
    response_width, oracle_width, has_active_oracle = (
        _synchronized_response_widths(
            _ThreeRankWidthAccelerator(),
            response_width=5,
            oracle_width=0,
            has_active_oracle=False,
        )
    )

    assert (response_width, oracle_width) == (9, 7)
    assert has_active_oracle
    clean_oracle_mask = torch.zeros((1, 0), dtype=torch.bool)
    padded = _pad_response_axis(clean_oracle_mask, oracle_width)
    assert padded.shape == (1, 7)
    assert not padded.any()


class _AllZeroWidthAccelerator:
    device = torch.device("cpu")

    def gather(self, local_widths):
        return torch.zeros(9, dtype=local_widths.dtype)


def test_ddp_chunk_plan_keeps_one_dummy_response_for_all_zero_batch():
    response_width, oracle_width, has_active_oracle = (
        _synchronized_response_widths(
            _AllZeroWidthAccelerator(),
            response_width=0,
            oracle_width=0,
            has_active_oracle=False,
        )
    )

    assert response_width == 1
    assert oracle_width == 0
    assert not has_active_oracle


def test_risk_normalization_preserves_absolute_consequence_multiplier():
    token_mean = torch.tensor(2.0)
    eligible_tokens = torch.tensor(1.0)
    risk_one = _risk_scaled_chunk_loss(
        token_mean,
        active_weight=torch.tensor(1.0),
        normalization_tokens=eligible_tokens,
    )
    risk_four = _risk_scaled_chunk_loss(
        token_mean,
        active_weight=torch.tensor(4.0),
        normalization_tokens=eligible_tokens,
    )

    assert risk_four.item() == pytest.approx(4.0 * risk_one.item())
