import torch

from care_opd.collator import _token_span_mask
from care_opd.losses import (
    causal_token_log_probs,
    masked_hard_target_cross_entropy,
    selected_causal_log_probs,
)


class _SelectedLogitsModel(torch.nn.Module):
    def __init__(self, time: int, vocab: int) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(
            torch.arange(time * vocab, dtype=torch.float32).reshape(time, vocab)
            / 17.0
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache=False,
        logits_to_keep=None,
    ):
        del attention_mask, use_cache
        logits = self.table.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        if logits_to_keep is not None:
            logits = logits.index_select(1, logits_to_keep)
        return type("Output", (), {"logits": logits})()


def test_selected_response_scoring_matches_full_causal_scoring() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 0], [4, 3, 2, 1, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    label_mask = torch.tensor(
        [[0, 0, 1, 1, 0], [0, 0, 0, 1, 0]], dtype=torch.bool
    )
    model = _SelectedLogitsModel(time=5, vocab=6)

    selected_sums, _, selected_mask, efficient = selected_causal_log_probs(
        model, input_ids, attention, label_mask
    )
    full_logits = model(
        input_ids=input_ids,
        attention_mask=attention,
        logits_to_keep=None,
    ).logits
    full_sums, _ = causal_token_log_probs(full_logits, input_ids, label_mask)

    assert efficient
    assert selected_mask.sum().item() == 3
    torch.testing.assert_close(selected_sums, full_sums)


def test_absolute_span_risk_weight_is_not_squared() -> None:
    _, absolute = _token_span_mask(
        [(0, 2), (2, 4)],
        [{"start": 0, "end": 2, "risk_weight": 4.0}],
        base_weight=4.0,
    )
    _, relative = _token_span_mask(
        [(0, 2), (2, 4)],
        [{"start": 0, "end": 2, "weight": 0.5}],
        base_weight=4.0,
    )
    assert absolute == [4.0, 0.0]
    assert relative == [2.0, 0.0]


def test_hard_oracle_ce_uses_only_weighted_causal_components() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [4.0, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[1, 1, 2]])
    mask = torch.tensor([[True, False, True]])
    weights = torch.tensor([[4.0, 100.0, 1.0]])

    loss, token_losses = masked_hard_target_cross_entropy(
        logits,
        targets,
        mask,
        token_weights=weights,
    )
    expected = (4.0 * token_losses[0, 0] + token_losses[0, 2]) / 5.0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 1]).item() == 0
