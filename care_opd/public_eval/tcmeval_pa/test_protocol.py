#!/usr/bin/env python3
"""CPU-only regression checks for the TCMEval-PA evaluation contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_tcmeval_pa as evaluation  # noqa: E402


class DatasetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = HERE / "data" / "TCMEval-PA.xlsx"
        cls.questions = evaluation.load_questions(cls.data)

    def test_official_v3_cardinality(self) -> None:
        self.assertEqual(len(self.questions), 328)
        self.assertEqual(len({item.id for item in self.questions}), 328)
        self.assertEqual(
            sum(item.question_type == "single" for item in self.questions), 297
        )
        self.assertEqual(
            sum(item.question_type == "multiple" for item in self.questions), 31
        )

    def test_embedded_b1_is_not_an_option_label(self) -> None:
        for official_id in ("294", "295"):
            question = next(item for item in self.questions if item.id == official_id)
            self.assertIn("B1", question.candidate_answers.replace("\n", ""))
            self.assertEqual(question.allowed_options, ("A", "B", "C", "D", "E"))

    def test_prompt_protocols_are_separate(self) -> None:
        official = evaluation.prompt_spec("official")
        blind = evaluation.prompt_spec("label_blind")
        self.assertTrue(official["type_reconstructed_from_gold_cardinality"])
        self.assertFalse(blind["type_reconstructed_from_gold_cardinality"])
        self.assertNotEqual(
            evaluation.canonical_hash(official), evaluation.canonical_hash(blind)
        )


class AnswerParserTest(unittest.TestCase):
    def test_option_set_normalization(self) -> None:
        parsed = evaluation.parse_prediction("C,A", ("A", "B", "C", "D", "E"))
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["parsed_options"], ["A", "C"])
        self.assertFalse(parsed["strict_format"])

    def test_fullwidth_letters(self) -> None:
        parsed = evaluation.parse_prediction("ＡＣ", ("A", "B", "C", "D", "E"))
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["parsed_options"], ["A", "C"])

    def test_english_prefix_fails_closed(self) -> None:
        parsed = evaluation.parse_prediction("Answer: A", ("A", "B", "C", "D", "E"))
        self.assertFalse(parsed["parse_ok"])

    def test_out_of_inventory_letter_fails(self) -> None:
        parsed = evaluation.parse_prediction("E", ("A", "B", "C", "D"))
        self.assertFalse(parsed["parse_ok"])
        self.assertEqual(parsed["invalid_letters"], ["E"])


if __name__ == "__main__":
    unittest.main()
