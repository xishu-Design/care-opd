"""Deterministic offline tests for the qualitative-case selector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from . import select_qualitative_cases as selector
except ImportError:  # Direct invocation from this directory.
    import select_qualitative_cases as selector


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _rollout(item_id: str, variant: str, passed: bool, text: str) -> dict:
    return {
        "id": item_id,
        "variant": variant,
        "passed": passed,
        "failure_types": [] if passed else ["invented_source_id"],
        "selected_components": [] if passed else ["sections"],
        "student_messages": [{"role": "user", "content": f"case {item_id}"}],
        "rollout": text,
    }


def _bank(item_id: str, variant: str, failure_type: str) -> dict:
    return {
        "id": item_id,
        "variant": variant,
        "failure_types": [failure_type],
        "selected_components": ["sections"],
        "student_messages": [{"role": "user", "content": f"input {item_id}"}],
        "oracle_response": '{"title":"oracle"}',
    }


def _margin(item_id: str, label: str, value: float) -> dict:
    return {
        "id": item_id,
        "model_label": label,
        "full": {"margin": value},
        "selected": {
            "localized_complete": True,
            "localized_margin": value + 0.1,
            "classification": "localized_complete",
        },
        "outside": {"margin": value - 0.1},
    }


def _fixture(root: Path, *, reverse: bool = False) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    categories = {
        "bad": ("bad_history", "failed_evidence_boundary_recovery"),
        "outage": ("source_outage", "missing_required_content"),
        "conflict": ("evidence_conflict", "failed_evidence_boundary_recovery"),
        "invented": ("source_outage", "invented_source_id"),
        "remaining": ("clean", "missing_required_content"),
    }
    bank_rows = [_bank(item_id, *values) for item_id, values in categories.items()]
    all_ids = [*categories, "regression", "control"]
    clean_rows: list[dict] = []
    gkd_rows: list[dict] = []
    care_rows: list[dict] = []
    for item_id in all_ids:
        variant = categories.get(item_id, ("clean", ""))[0]
        clean_passed = item_id in {"regression", "control"}
        care_passed = item_id not in {"remaining", "regression"}
        clean_rows.append(_rollout(item_id, variant, clean_passed, f"clean {item_id}"))
        gkd_rows.append(_rollout(item_id, variant, item_id != "remaining", f"gkd {item_id}"))
        care_text = (
            "email test@example.org phone 13800138000 id 11010519491231002X "
            + ("x" * 300)
            if item_id == "bad"
            else f"care {item_id}"
        )
        care_rows.append(_rollout(item_id, variant, care_passed, care_text))
    margin_rows: list[dict] = []
    for index, item_id in enumerate(categories):
        for offset, label in enumerate(selector.MODEL_LABELS):
            margin_rows.append(_margin(item_id, label, index + offset * 0.2))
    if reverse:
        for rows in (bank_rows, clean_rows, gkd_rows, care_rows, margin_rows):
            rows.reverse()
    paths = {
        "clean": root / "clean.jsonl",
        "gkd": root / "gkd.jsonl",
        "care": root / "care.jsonl",
        "bank": root / "bank.jsonl",
        "margin": root / "margin.jsonl",
    }
    for name, rows in (
        ("clean", clean_rows),
        ("gkd", gkd_rows),
        ("care", care_rows),
        ("bank", bank_rows),
        ("margin", margin_rows),
    ):
        _write_jsonl(paths[name], rows)
    return paths


def _run(paths: dict[str, Path], output: Path) -> dict:
    return selector.select_cases(
        clean_rollouts_path=paths["clean"],
        gkd_rollouts_path=paths["gkd"],
        care_rollouts_path=paths["care"],
        failure_bank_path=paths["bank"],
        margin_path=paths["margin"],
        output_dir=output,
        selection_seed=11,
        max_preview_characters=120,
    )


class QualitativeCaseTests(unittest.TestCase):
    def test_six_distinct_slots_and_redacted_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _run(_fixture(root), root / "out")
            self.assertEqual(payload["status"], "COMPLETE")
            self.assertEqual(payload["selected_slots"], 6)
            ids = [case["id"] for case in payload["cases"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(payload["cases"][4]["id"], "remaining")
            self.assertEqual(payload["cases"][5]["id"], "regression")
            bad = next(case for case in payload["cases"] if case["id"] == "bad")
            care_preview = bad["models"]["care_opd"]["response"]["preview"]
            self.assertIn("[EMAIL]", care_preview)
            self.assertIn("[PHONE]", care_preview)
            self.assertNotIn("test@example.org", care_preview)
            self.assertTrue((root / "out" / "qualitative_cases.json").is_file())
            self.assertTrue((root / "out" / "qualitative_cases.md").is_file())

    def test_selection_is_input_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _run(_fixture(root / "a"), root / "out-a")
            second = _run(_fixture(root / "b", reverse=True), root / "out-b")
            first_ids = [case.get("id") for case in first["cases"]]
            second_ids = [case.get("id") for case in second["cases"]]
            self.assertEqual(first_ids, second_ids)

    def test_missing_slots_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _fixture(root)
            # Remove regression eligibility while leaving all input identities paired.
            clean_rows = [json.loads(line) for line in paths["clean"].read_text().splitlines()]
            for row in clean_rows:
                if row["id"] == "regression":
                    row["passed"] = False
                    row["failure_types"] = ["invented_source_id"]
                    row["selected_components"] = ["sections"]
            _write_jsonl(paths["clean"], clean_rows)
            payload = _run(paths, root / "out")
            regression = payload["cases"][-1]
            self.assertEqual(payload["status"], "COMPLETE_WITH_MISSING_SLOTS")
            self.assertEqual(regression["status"], "MISSING")
            self.assertIn("no eligible", regression["missing_reason"])


if __name__ == "__main__":
    unittest.main()
