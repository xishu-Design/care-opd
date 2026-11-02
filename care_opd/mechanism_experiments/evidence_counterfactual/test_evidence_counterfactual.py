from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from care_opd.qzbc_contract import canonical_response

from .build_counterfactual_bank import FAMILIES, STATES, build_group
from .score_counterfactuals import score


def _source() -> dict:
    target = {
        "title": "Evidence summary",
        "summary": "Only the supplied record is confirmed.",
        "sections": [
            {
                "heading": "Confirmed",
                "content": "This statement is grounded in the packet.",
                "source_ids": ["SRC_TEST"],
            }
        ],
        "warnings": [],
        "next_actions": [],
        "disclaimer": "For controlled testing only.",
    }
    return {
        "id": "clean-source-1",
        "seed_id": "seed-1",
        "variant": "clean",
        "request": {
            "role": "student",
            "task_type": "student_ai_chat",
            "input_text": "Summarize the supplied evidence.",
            "constraints": {},
            "evidence_packet": {
                "retrieved_evidence": [
                    {
                        "source_id": "SRC_TEST",
                        "summary": "A controlled source record.",
                        "metadata": {"review_status": "verified"},
                    }
                ],
                "conversation_context": {
                    "recent_turns": [],
                    "turn_count": 0,
                    "available_turn_count": 0,
                    "include_assistant_answers": True,
                },
            },
        },
        "oracle_response": canonical_response(target),
        "verification": {"required_source_ids": ["SRC_TEST"]},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class EvidenceCounterfactualTest(unittest.TestCase):
    def _group(self, family: str) -> list[dict]:
        return build_group(
            _source(),
            family=family,
            prompts={"student_ai_chat": "Return a GeneratedResponse JSON object."},
            contract_fingerprint="a" * 64,
            selection_seed=7,
        )

    def test_group_has_restored_marker_and_evidence_preserving_controls(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                rows = self._group(family)
                self.assertEqual([row["state"] for row in rows], list(STATES))
                self.assertNotEqual(rows[0]["student_messages"], rows[2]["student_messages"])
                clean_packet = dict(rows[0]["request"]["evidence_packet"])
                restored_packet = dict(rows[2]["request"]["evidence_packet"])
                neutral_packet = dict(rows[3]["request"]["evidence_packet"])
                clean_packet.pop("conversation_context", None)
                restored_packet.pop("conversation_context", None)
                neutral_packet.pop("conversation_context", None)
                self.assertEqual(clean_packet, restored_packet)
                self.assertEqual(clean_packet, neutral_packet)

    def test_score_recomputes_production_verifier_and_joint_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            rows = self._group("bad_history")
            requests = tmp_path / "requests.jsonl"
            _write_jsonl(requests, rows)
            paths = []
            for label in ("clean_sft", "care_opd"):
                path = tmp_path / f"{label}.jsonl"
                _write_jsonl(
                    path,
                    [
                        {"id": row["id"], "rollout": row["oracle_response"], "passed": True}
                        for row in rows
                    ],
                )
                paths.append((label, path))
            result = score(
                requests,
                paths,
                tmp_path / "results",
                bootstrap_repetitions=50,
                bootstrap_seed=7,
            )
            all_rows = [row for row in result["metrics"] if row["family"] == "all"]
            self.assertEqual(len(all_rows), 2)
            self.assertTrue(all(row["four_state_joint_tsr"] == 1.0 for row in all_rows))
            self.assertTrue((tmp_path / "results" / "counterfactual_scored.jsonl").is_file())

    def test_score_rejects_recorded_verifier_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            rows = self._group("bad_history")
            requests = tmp_path / "requests.jsonl"
            _write_jsonl(requests, rows)
            paths = []
            for label in ("one", "two"):
                path = tmp_path / f"{label}.jsonl"
                rollout_rows = [
                    {"id": row["id"], "rollout": row["oracle_response"], "passed": True}
                    for row in rows
                ]
                if label == "one":
                    rollout_rows[0]["passed"] = False
                _write_jsonl(path, rollout_rows)
                paths.append((label, path))
            with self.assertRaisesRegex(Exception, "recorded pass differs"):
                score(
                    requests,
                    paths,
                    tmp_path / "results",
                    bootstrap_repetitions=20,
                    bootstrap_seed=7,
                )


if __name__ == "__main__":
    unittest.main()
