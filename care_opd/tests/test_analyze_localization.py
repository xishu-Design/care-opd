import csv
import json
import sys
from types import SimpleNamespace

import pytest

from mechanism_eval.analyze_localization import (
    _load_tokenizer,
    analyze_rows,
    run,
)


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    name_or_path = "character-tokenizer"
    is_fast = True

    def __len__(self):
        return 65536

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del add_special_tokens
        result = {"input_ids": [ord(character) + 10 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


def _rows():
    return [
        {
            "id": "localized",
            "variant": "bad_history",
            "rollout": '{"title":"x","summary":"repair","sections":[]}',
            "failure_types": ["invented_source_id", "missing_required_content"],
            "selected_components": ["summary"],
            "risk_weight": 2.0,
        },
        {
            "id": "fallback",
            "variant": "invalid_json",
            "rollout": "not json",
            "failure_types": ["invalid_generated_response"],
            "selected_components": ["title"],
            "risk_weight": 4.0,
        },
        {
            "id": "partial",
            "variant": "bad_history",
            "rollout": '{"title":"x"}',
            "failure_types": ["missing_required_content"],
            "selected_components": ["title", "summary"],
            "risk_weight": 1.0,
        },
    ]


def test_localization_reuses_component_projection_and_token_mask():
    result = analyze_rows(_rows(), _CharacterTokenizer())
    examples = {item["id"]: item for item in result["examples"]}

    localized = examples["localized"]
    assert localized["selected_tokens"] == len('"repair"')
    assert localized["full_response_tokens"] == len(_rows()[0]["rollout"]) + 1
    assert localized["full_response_fallback"] is False
    assert localized["supervision_pollution_multiplier"] == pytest.approx(
        localized["full_response_tokens"] / localized["selected_tokens"]
    )
    assert localized["risk_weighted_selected_tokens"] == pytest.approx(
        2.0 * localized["selected_tokens"]
    )

    fallback = examples["fallback"]
    assert fallback["full_response_fallback"] is True
    assert fallback["selected_tokens"] == len("not json")
    assert fallback["full_response_tokens"] == len("not json") + 1

    partial = examples["partial"]
    assert partial["full_response_fallback"] is False
    assert partial["partial_projection"] is True
    assert partial["projected_components"] == ["title"]
    assert partial["unprojected_components"] == ["summary"]

    assert result["overall"]["rows"] == 3
    assert result["overall"]["full_fallback_rate"] == pytest.approx(1 / 3)
    assert result["overall"]["partial_projection_rate"] == pytest.approx(1 / 3)
    assert result["strata"]["by_variant"]["bad_history"]["rows"] == 2
    assert result["strata"]["by_failure_type"]["missing_required_content"][
        "rows"
    ] == 2
    assert result["strata"]["by_component"]["summary"]["rows"] == 2
    assert result["field_distribution"]["title"]["rows"] == 2


def test_reports_are_complete_atomic_and_machine_readable(tmp_path):
    bank = tmp_path / "failure_bank.jsonl"
    bank.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in _rows()),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"
    payload = run(
        bank,
        "unused-model-path",
        output_dir,
        tokenizer=_CharacterTokenizer(),
    )

    expected_names = {
        "localization_analysis.json",
        "localization_analysis.csv",
        "localization_analysis.md",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_names

    saved = json.loads((output_dir / "localization_analysis.json").read_text("utf-8"))
    assert saved["schema"] == "care-opd-localization-analysis-v1"
    assert saved["settings"]["model_weights_loaded"] is False
    assert saved["input"]["rows"] == 3
    assert saved["outputs"] == payload["outputs"]

    with (output_dir / "localization_analysis.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))
    assert records[0]["scope"] == "overall"
    assert {record["scope"] for record in records} == {
        "overall",
        "by_variant",
        "by_failure_type",
        "by_component",
    }
    markdown = (output_dir / "localization_analysis.md").read_text("utf-8")
    assert "tokenizer-only mechanism diagnostic" in markdown
    assert "By selected component" in markdown


def test_tokenizer_loader_does_not_import_or_instantiate_a_model(monkeypatch):
    calls = []
    sentinel = object()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return sentinel

    fake_transformers = SimpleNamespace(AutoTokenizer=_AutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded = _load_tokenizer(
        "checkpoint/tokenizer",
        trust_remote_code=False,
        local_files_only=True,
    )
    assert loaded is sentinel
    assert calls == [
        (
            "checkpoint/tokenizer",
            {
                "trust_remote_code": False,
                "local_files_only": True,
                "use_fast": True,
            },
        )
    ]
