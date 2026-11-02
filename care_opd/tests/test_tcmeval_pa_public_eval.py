from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "public_eval"
    / "tcmeval_pa"
    / "evaluate_tcmeval_pa.py"
)
SPEC = importlib.util.spec_from_file_location("tcmeval_pa_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


def test_prompt_modes_have_distinct_fingerprints_and_blind_prompt_is_constant():
    single = evaluator.Question(0, "1", "题一", "A.甲\nB.乙", ("A", "B"), ("A",), "single", "c", "r", "s")
    multiple = evaluator.Question(1, "2", "题二", "A.甲\nB.乙", ("A", "B"), ("A", "B"), "multiple", "c", "r", "s")
    assert evaluator.prompt_messages(single, "official")[0] != evaluator.prompt_messages(multiple, "official")[0]
    assert evaluator.prompt_messages(single, "label_blind")[0] == evaluator.prompt_messages(multiple, "label_blind")[0]
    assert evaluator.canonical_hash(evaluator.prompt_spec("official")) != evaluator.canonical_hash(evaluator.prompt_spec("label_blind"))


def test_option_set_and_strict_format_are_separate():
    clean = evaluator.parse_prediction("A、C", ("A", "B", "C"))
    verbose = evaluator.parse_prediction("答案：A、C", ("A", "B", "C"))
    invalid = evaluator.parse_prediction("A、Z", ("A", "B", "C"))
    assert clean["parsed_options"] == ["A", "C"] and clean["strict_format"]
    assert verbose["parsed_options"] == ["A", "C"] and not verbose["strict_format"]
    assert not invalid["parse_ok"] and invalid["invalid_letters"] == ["Z"]


def test_duplicate_stems_remain_distinct_by_id_and_options():
    first = evaluator.Question(0, "10", "相同题干", "A.甲\nB.乙", ("A", "B"), ("A",), "single", "c", "r", "s")
    second = evaluator.Question(1, "11", "相同题干", "A.丙\nB.丁", ("A", "B"), ("B",), "single", "c", "r", "s")
    assert first.id != second.id
    assert evaluator.prompt_messages(first, "label_blind") != evaluator.prompt_messages(second, "label_blind")
