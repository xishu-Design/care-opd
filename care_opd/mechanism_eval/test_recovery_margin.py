"""Offline unit tests for :mod:`mechanism_eval.recovery_margin`.

The tests use a character tokenizer and a tiny deterministic torch module.  No
checkpoint, network access, generation, or training dependency is involved.
"""

from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json

import pytest
import torch

from mechanism_eval import recovery_margin as margin


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3

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
            ids = [self.bos_token_id, *ids]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "\n".join(f"{row['role']}:{row['content']}" for row in messages) + "\nassistant:"


class SelectedOnlyToyModel(torch.nn.Module):
    """Assign a high logit to every actual next token at selected positions."""

    def __init__(self, vocab_size=512) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.calls: list[list[int]] = []

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache=False,
        logits_to_keep=None,
    ):
        assert not use_cache
        assert attention_mask.shape == input_ids.shape
        positions = logits_to_keep.tolist()
        self.calls.append(positions)
        logits = torch.zeros(
            input_ids.shape[0], len(positions), self.vocab_size, device=input_ids.device
        )
        for output_index, position in enumerate(positions):
            target = input_ids[:, position + 1]
            logits[:, output_index].scatter_(1, target[:, None], 4.0)
        return SimpleNamespace(logits=logits)


def generated(title: str) -> str:
    return (
        '{"title":"'
        + title
        + '","summary":"s","sections":[],"warnings":[],"next_actions":[],"disclaimer":"d"}'
    )


def row(rejected: str, components=None):
    return {
        "id": "example-1",
        "student_messages": [{"role": "user", "content": "PUBLIC"}],
        "rollout": rejected,
        "oracle_response": generated("oracle"),
        "selected_components": components or ["title"],
    }


def test_pair_uses_identical_fitted_prompt_and_complete_local_projection():
    pair = margin._prepare_pair(
        row(generated("wrong")),
        CharacterTokenizer(),
        max_length=256,
        max_response_tokens=128,
        add_eos=True,
    )
    assert pair["prompt_ids"]
    assert pair["oracle"]["projection"]["coverage"] == 1.0
    assert pair["rejected"]["projection"]["coverage"] == 1.0
    assert not pair["oracle"]["projection"]["fallback"]
    assert sum(pair["oracle"]["projection"]["mask"]) > 0


def test_invalid_rejected_projection_is_explicit_full_response_fallback():
    pair = margin._prepare_pair(
        row("not-json", ["__full_response_fallback__"]),
        CharacterTokenizer(),
        max_length=256,
        max_response_tokens=128,
        add_eos=True,
    )
    assert pair["oracle"]["projection"]["fallback"]
    assert pair["rejected"]["projection"]["fallback"]
    # EOS is deliberately outside a character-span fallback.
    assert sum(pair["rejected"]["projection"]["mask"]) == len("not-json")


def test_missing_rejected_component_is_partial_not_localized_fallback():
    pair = margin._prepare_pair(
        row(generated("wrong"), ["title", "absent_field"]),
        CharacterTokenizer(),
        max_length=256,
        max_response_tokens=128,
        add_eos=True,
    )
    projection = pair["rejected"]["projection"]
    assert projection["coverage"] == pytest.approx(0.5)
    assert projection["unprojected_components"] == ["absent_field"]
    assert not projection["fallback"]


def test_response_scoring_chunks_only_selected_causal_positions():
    model = SelectedOnlyToyModel(vocab_size=512)
    values = margin._score_response_tokens(
        model,
        prompt_ids=[20, 21, 22],
        response_ids=[30, 31, 32, 33, 34],
        device=torch.device("cpu"),
        chunk_size=2,
    )
    assert len(values) == 5
    assert [len(call) for call in model.calls] == [2, 2, 1]
    assert model.calls == [[2, 3], [4, 5], [6]]
    assert all(value < 0 for value in values)


def test_hash_selection_is_deterministic_and_order_independent():
    rows = [{"id": str(index)} for index in range(10)]
    first = margin._select_rows(rows, max_examples=4, selection_seed=17)
    second = margin._select_rows(list(reversed(rows)), max_examples=4, selection_seed=17)
    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_formal_suite_binds_label_path_and_recorded_fingerprint(tmp_path):
    roots = {}
    entries = []
    for index, label in enumerate(("clean_sft", "gkd_full", "care_opd"), 1):
        root = (tmp_path / label).resolve()
        root.mkdir()
        roots[label] = root
        entries.append(
            {
                "label": label,
                "canonical_path": str(root),
                "model_fingerprint": f"{index:064x}",
            }
        )
    suite = {
        "models": entries,
        "suite_fingerprint": "a" * 64,
    }
    suite_path = tmp_path / "model_suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    models = [(label, str(roots[label])) for label in roots]
    bound = margin._suite_binding(suite_path, models)
    assert bound["suite_fingerprint"] == "a" * 64
    assert bound["sha256"] == hashlib.sha256(suite_path.read_bytes()).hexdigest()
    assert [row["label"] for row in bound["models"]] == list(roots)

    wrong = tmp_path / "wrong"
    wrong.mkdir()
    with pytest.raises(margin.InputError, match="model path mismatch"):
        margin._suite_binding(
            suite_path,
            [("clean_sft", str(wrong)), ("gkd_full", str(roots["gkd_full"])), ("care_opd", str(roots["care_opd"]))],
        )
