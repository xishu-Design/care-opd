from __future__ import annotations

import unittest

from mechanism_experiments.real_thyroid_external.run_local_agent_annotation import (
    _candidate_map,
    _merge_case,
)


def _task() -> dict:
    source = "Synthetic report states size 8 mm."
    return {
        "case_id": "THY-US-000000000000000000000000",
        "report_group_hmac": "rg_" + "0" * 64,
        "modality": "thyroid_ultrasound",
        "source_text_private": source,
        "source_record_sha256": "unused-by-pure-merge-test",
        "candidate_fields": [
            {"entity_id": "N01", "field_path": "size.raw", "raw_value": "8 mm"}
        ],
    }


def _field(value: str, quote: str, *, support: str = "exact_supported") -> dict:
    return {
        "entity_id": "N01",
        "field_path": "size.raw",
        "value": value,
        "unit": "mm",
        "support_status": support,
        "evidence_quote": quote,
        "reason_code": "SYNTHETIC_TEST",
    }


class LocalAgentQueueContractTest(unittest.TestCase):
    def test_candidate_map_uses_stable_runner_keys(self) -> None:
        self.assertEqual(_candidate_map(_task()), {("N01", "size.raw"): "8 mm"})

    def test_strict_acceptance_requires_three_way_value_agreement(self) -> None:
        task = _task()
        locator = {"fields": [_field("8 mm", "8 mm")]}
        auditor = {"fields": [_field("8 mm", "8 mm")]}
        adjudicator = {
            "fields": [
                {
                    "entity_id": "N01",
                    "field_path": "size.raw",
                    "decision": "accept_sheet_candidate",
                    "selected_quote": "8 mm",
                    "normalized_value": "8 mm",
                    "unit": "mm",
                    "reason_code": "SYNTHETIC_AGREEMENT",
                }
            ]
        }
        accepted, review = _merge_case(task, locator, auditor, adjudicator)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(review, [])

        mismatched_locator = {"fields": [_field("9 mm", "8 mm")]}
        accepted, review = _merge_case(
            task, mismatched_locator, auditor, adjudicator
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(review), 1)

    def test_blind_new_field_is_preserved_for_human_review_only(self) -> None:
        task = _task()
        task["candidate_fields"] = []
        locator = {"fields": [_field("8 mm", "8 mm")]}
        accepted, review = _merge_case(task, locator, {"fields": []}, {"fields": []})
        self.assertEqual(accepted, [])
        self.assertEqual(len(review), 1)
        self.assertIsNone(review[0]["candidate_value_private"])
        self.assertEqual(review[0]["reference_status"], "human_review_required")


if __name__ == "__main__":
    unittest.main()
