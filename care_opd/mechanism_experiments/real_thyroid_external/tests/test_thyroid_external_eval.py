from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mechanism_experiments.real_thyroid_external import thyroid_external_eval as evaluation
from mechanism_experiments.real_thyroid_external.thyroid_verifier import (
    assess_thyroid_response,
)


SOURCE_ID = "THY-US-0123456789abcdef01234567-SRC"


def _gold(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "source_id": SOURCE_ID,
        "gold_fields": [
            {
                "entity_id": "N01",
                "field_path": "location",
                "target_value_private": "左叶",
                "source_id": SOURCE_ID,
            },
            {
                "entity_id": "N01",
                "field_path": "size.raw",
                "target_value_private": "12 mm",
                "source_id": SOURCE_ID,
            },
        ],
    }


def _response(*, exact: bool) -> str:
    sections = []
    if exact:
        sections = [
            {
                "heading": "N01|location",
                "content": "左叶",
                "source_ids": [SOURCE_ID],
            },
            {
                "heading": "N01|size.raw",
                "content": "12 mm",
                "source_ids": [SOURCE_ID],
            },
        ]
    return json.dumps(
        {
            "title": "",
            "summary": "",
            "sections": sections,
            "warnings": [],
            "next_actions": [],
            "disclaimer": "",
        },
        ensure_ascii=False,
    )


def _case(index: int, case_id: str, *, exact: bool, modality: str) -> dict:
    assessment = assess_thyroid_response(_gold(case_id), _response(exact=exact)).to_dict()
    return {
        "index": index,
        "case_id": case_id,
        "modality": modality,
        "assessment": assessment,
        "metrics": evaluation._case_metrics(assessment),
        "latency_seconds": 0.1,
        "output_tokens": 20,
        "generation_error": None,
    }


def _state_case(
    index: int,
    family_id: str,
    variant: str,
    *,
    success: bool,
    exact: bool = True,
    modality: str = "ultrasound",
) -> dict:
    row = _case(
        index,
        f"{family_id}::{variant}",
        exact=exact,
        modality=modality,
    )
    row["family_id"] = family_id
    row["variant"] = variant
    row["assessment"]["passed"] = success
    row["assessment"]["task_success"] = success
    row["assessment"]["state_task_success"] = success
    if variant in evaluation.BOUNDARY_VARIANTS:
        for key in evaluation.BOUNDARY_ASSESSMENT_FIELDS:
            row["assessment"][key] = (
                not success if key == "unsafe_clinical_conclusion" else success
            )
        row["assessment"]["boundary_errors"] = [] if success else ["synthetic_failure"]
    row["metrics"] = evaluation._case_metrics(row["assessment"])
    return row


class ThyroidExternalEvaluationTest(unittest.TestCase):
    def test_primary_and_additional_references_are_validated_and_deduplicated(self) -> None:
        labels = ["clean_sft", "gkd_full", "care_opd"]
        self.assertEqual(
            evaluation._resolve_reference_labels(
                "clean_sft", ["gkd_full", "clean_sft"], labels
            ),
            ["clean_sft", "gkd_full"],
        )
        with self.assertRaisesRegex(
            evaluation.EvaluationContractError, "unknown reference model"
        ):
            evaluation._resolve_reference_labels("clean_sft", ["missing"], labels)
        parsed = evaluation._parser().parse_args(
            [
                "--seeds",
                "seeds.jsonl",
                "--model",
                "clean_sft=/models/clean",
                "--model",
                "gkd_full=/models/gkd",
                "--model",
                "care_opd=/models/care",
                "--output-dir",
                "out",
                "--reference-model",
                "clean_sft",
                "--additional-reference-model",
                "gkd_full",
            ]
        )
        self.assertEqual(parsed.additional_reference_model, ["gkd_full"])

    def test_run_requires_private_data_acknowledgement_before_io(self) -> None:
        with self.assertRaisesRegex(
            evaluation.EvaluationContractError,
            "acknowledge-private-clinical-data",
        ):
            evaluation.run_evaluation(
                seeds_path=Path("missing.private.jsonl"),
                model_specs=[],
                output_dir=Path("unused"),
            )

    def test_aggregate_metrics_and_modality_strata(self) -> None:
        rows = [
            _case(0, "case-a", exact=True, modality="ultrasound"),
            _case(1, "case-b", exact=False, modality="pathology"),
        ]
        summary = evaluation.summarize_model("care", rows)
        overall = summary["overall"]
        self.assertEqual(overall["eligible_reports"], 2)
        self.assertEqual(overall["eligible_fields"], 4)
        self.assertEqual((overall["field_tp"], overall["field_fp"], overall["field_fn"]), (2, 0, 2))
        self.assertEqual(overall["field_micro_precision"], 1.0)
        self.assertEqual(overall["field_micro_recall"], 0.5)
        self.assertAlmostEqual(overall["field_micro_f1"], 2 / 3)
        self.assertEqual(overall["report_exact_rate"], 0.5)
        self.assertEqual(set(summary["by_modality"]), {"pathology", "ultrasound"})

    def test_four_state_primary_metrics_and_secondary_field_scope(self) -> None:
        outcomes = {
            "family-a": {
                "clean": True,
                "bad_history": True,
                "source_outage": False,
                "evidence_conflict": True,
            },
            "family-b": {
                "clean": False,
                "bad_history": True,
                "source_outage": True,
                "evidence_conflict": True,
            },
        }
        rows = []
        for family_id, states in outcomes.items():
            for variant, success in states.items():
                rows.append(
                    _state_case(
                        len(rows),
                        family_id,
                        variant,
                        success=success,
                        # Boundary rows deliberately have no exact fields;
                        # they must not contaminate secondary field metrics.
                        exact=variant in evaluation.SECONDARY_FIELD_VARIANTS,
                    )
                )
        summary = evaluation.summarize_model("care", rows)
        primary = summary["primary"]
        self.assertEqual(primary["family_mode"], "four_state")
        self.assertEqual(primary["families"], 2)
        self.assertEqual(primary["clean_task_success_rate"], 0.5)
        self.assertAlmostEqual(primary["perturbed_task_success_rate"], 5 / 6)
        self.assertAlmostEqual(primary["conditional_retention"], 2 / 3)
        self.assertAlmostEqual(primary["regression_rate"], 1 / 3)
        self.assertEqual(primary["bad_history_conditional_retention"], 1.0)
        self.assertEqual(primary["bad_history_conditional_retention_denominator"], 1)
        self.assertTrue(primary["perturbed_task_success_rate_is_descriptive"])
        self.assertEqual(primary["four_state_joint_tsr"], 0.0)
        self.assertEqual(
            primary["by_variant"]["source_outage"]["task_success_rate"], 0.5
        )
        self.assertEqual(
            primary["boundary_metrics"]["source_outage"][
                "boundary_recovery_valid_rate"
            ],
            0.5,
        )
        secondary = summary["secondary_field_metrics"]
        self.assertEqual(secondary["scope"], "clean+bad_history")
        self.assertEqual(secondary["reports"], 4)
        self.assertEqual(secondary["field_micro_f1"], 1.0)

    def test_clean_only_primary_metrics_are_explicitly_unavailable(self) -> None:
        rows = [_case(0, "case-a", exact=True, modality="ultrasound")]
        primary = evaluation.summarize_model("care", rows)["primary"]
        self.assertEqual(primary["family_mode"], "clean_only")
        self.assertEqual(primary["clean_task_success_rate"], 1.0)
        for key in (
            "perturbed_task_success_rate",
            "conditional_retention",
            "regression_rate",
            "four_state_joint_tsr",
        ):
            self.assertIsNone(primary[key])

    def test_v4_state_success_precedence_and_three_axis_aggregation(self) -> None:
        legacy = _state_case(0, "family-a", "clean", success=True)
        legacy["assessment"].pop("state_task_success", None)
        self.assertTrue(evaluation._task_success(legacy))
        legacy["assessment"]["state_task_success"] = False
        self.assertFalse(evaluation._task_success(legacy))

        rows = []
        for index, values in enumerate(((1, 1, 1), (2, 0, 1))):
            row = _state_case(index, f"family-{index}", "clean", success=True)
            row["evaluation_track"] = "structured_evidence_faithfulness"
            row["challenge_mode"] = False
            row["assessment"].update(
                {
                    "evaluation_track": "structured_evidence_faithfulness",
                    "challenge_mode": False,
                    "envelope_valid": True,
                    "slot_schema_valid": index == 0,
                    "key_binding_valid": True,
                    "packet_scaffolding_free": True,
                    "content_tp": values[0],
                    "content_fp": values[1],
                    "content_fn": values[2],
                    "content_complete": index == 1,
                    "grounded_content_tp": values[0],
                    "grounded_content_fp": values[1],
                    "grounded_content_fn": values[2],
                    "grounded_content_complete": False,
                }
            )
            rows.append(row)
        axes = evaluation.aggregate_three_axis_metrics(rows)
        self.assertEqual(axes["evaluation_track"], "structured_evidence_faithfulness")
        self.assertEqual(axes["challenge_mode_rate"], 0.0)
        self.assertEqual(axes["slot_schema_valid_rate"], 0.5)
        self.assertEqual((axes["content_tp"], axes["content_fp"], axes["content_fn"]), (3, 1, 2))
        self.assertEqual(axes["content_micro_precision"], 0.75)
        self.assertEqual(axes["content_micro_recall"], 0.6)
        self.assertAlmostEqual(axes["content_micro_f1"], 2 / 3)
        self.assertEqual(axes["content_report_complete_rate"], 0.5)

    def test_extraction_axes_exclude_boundary_empty_to_empty_scores(self) -> None:
        rows = []
        for variant in evaluation.FOUR_STATE_VARIANTS:
            row = _state_case(len(rows), "family-a", variant, success=True)
            extraction = variant in evaluation.SECONDARY_FIELD_VARIANTS
            row["assessment"].update(
                {
                    "envelope_valid": not extraction,
                    "slot_schema_valid": variant == "clean" or not extraction,
                    "key_binding_valid": variant == "clean" or not extraction,
                    "packet_scaffolding_free": not extraction,
                    "content_tp": 1 if variant == "clean" else 0,
                    "content_fp": 1 if variant == "bad_history" else 0,
                    "content_fn": 1 if variant == "bad_history" else 0,
                    "content_complete": variant != "bad_history",
                    "grounded_content_tp": 1 if variant == "clean" else 0,
                    "grounded_content_fp": 1 if variant == "bad_history" else 0,
                    "grounded_content_fn": 1 if variant == "bad_history" else 0,
                    "grounded_content_complete": variant != "bad_history",
                }
            )
            rows.append(row)
        axes = evaluation.aggregate_three_axis_metrics(rows)
        self.assertEqual(axes["slot_schema_valid_scope"], "clean+bad_history")
        self.assertEqual(axes["slot_schema_valid_eligible_reports"], 2)
        self.assertEqual(axes["slot_schema_valid_evaluated_reports"], 2)
        self.assertEqual(axes["slot_schema_valid_rate"], 0.5)
        self.assertEqual(axes["key_binding_valid_rate"], 0.5)
        self.assertEqual(axes["envelope_valid_scope"], "all_states")
        self.assertEqual(axes["envelope_valid_eligible_reports"], 4)
        self.assertEqual(axes["envelope_valid_rate"], 0.5)
        self.assertEqual(axes["content_scope"], "clean+bad_history")
        self.assertEqual(axes["content_eligible_reports"], 2)
        self.assertEqual(axes["content_evaluated_reports"], 2)
        self.assertEqual(
            (axes["content_tp"], axes["content_fp"], axes["content_fn"]),
            (1, 1, 1),
        )
        self.assertEqual(axes["content_micro_f1"], 0.5)
        self.assertEqual(axes["content_report_complete_rate"], 0.5)
        self.assertEqual(axes["grounded_content_report_complete_rate"], 0.5)

    def test_empty_generation_remains_in_eligible_denominator(self) -> None:
        assessment = assess_thyroid_response(_gold("case-a"), "").to_dict()
        metrics = evaluation._case_metrics(assessment)
        self.assertTrue(metrics["eligible"])
        self.assertEqual(metrics["expected"], 2)
        self.assertEqual(metrics["fn"], 2)
        self.assertFalse(metrics["report_exact"])

    def test_unsupported_errors_accept_runtime_tuples_and_json_lists(self) -> None:
        base = {
            "tp": 0,
            "fp": 2,
            "fn": 1,
            "expected_field_count": 1,
            "emitted_section_count": 2,
        }
        for errors in (("first", "second"), ["first", "second"]):
            with self.subTest(collection_type=type(errors).__name__):
                metrics = evaluation._case_metrics({**base, "unsupported_errors": errors})
                self.assertEqual(metrics["unsupported"], 2)
        self.assertEqual(
            evaluation._case_metrics(
                {**base, "unsupported_errors": "not-a-collection-of-errors"}
            )["unsupported"],
            0,
        )
        self.assertEqual(
            evaluation._case_metrics(
                {
                    **base,
                    "unsupported_count": 1,
                    "unsupported_errors": ["first", "second"],
                }
            )["unsupported"],
            1,
        )

    def test_aggregate_recomputes_stale_cached_metrics_from_assessment(self) -> None:
        assessment = {
            "tp": 0,
            "fp": 1,
            "fn": 1,
            "expected_field_count": 1,
            "emitted_section_count": 1,
            "unsupported_errors": ("unsupported",),
            "passed": False,
            "schema_valid": True,
            "source_valid": True,
            "failure_types": ["unsupported_field"],
        }
        row = {
            "assessment": assessment,
            "metrics": {**evaluation._case_metrics(assessment), "unsupported": 0},
            "generation_error": None,
        }
        aggregate = evaluation.aggregate_case_metrics([row])
        self.assertEqual(aggregate["unsupported_fields"], 1)
        self.assertEqual(aggregate["unsupported_field_rate"], 1.0)
        self.assertEqual(aggregate["field_micro_precision"], 0.0)
        self.assertEqual(aggregate["field_micro_recall"], 0.0)
        self.assertEqual(aggregate["field_micro_f1"], 0.0)

    def test_field_f1_is_count_based_with_only_empty_population_unavailable(self) -> None:
        def aggregate(tp: int, fp: int, fn: int) -> dict:
            expected = tp + fn
            assessment = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "expected_field_count": expected,
                "emitted_section_count": tp + fp,
                "unsupported_errors": [],
            }
            return evaluation.aggregate_case_metrics(
                [
                    {
                        "assessment": assessment,
                        "generation_error": None,
                    }
                ]
            )

        self.assertEqual(aggregate(0, 0, 1)["field_micro_f1"], 0.0)
        self.assertEqual(aggregate(0, 1, 1)["field_micro_f1"], 0.0)
        self.assertAlmostEqual(aggregate(2, 0, 2)["field_micro_f1"], 2 / 3)
        self.assertIsNone(aggregate(0, 0, 0)["field_micro_f1"])

    def test_paired_report_bootstrap_is_fixed_seed_and_report_level(self) -> None:
        reference = [
            _case(0, "case-a", exact=True, modality="ultrasound"),
            _case(1, "case-b", exact=False, modality="pathology"),
        ]
        candidate = [
            _case(0, "case-a", exact=True, modality="ultrasound"),
            _case(1, "case-b", exact=True, modality="pathology"),
        ]
        first = evaluation.paired_report_bootstrap(
            reference,
            candidate,
            reference_label="base",
            candidate_label="care",
            replicates=200,
            seed=17,
        )
        second = evaluation.paired_report_bootstrap(
            reference,
            candidate,
            reference_label="base",
            candidate_label="care",
            replicates=200,
            seed=17,
        )
        self.assertEqual(first, second)
        exact = next(row for row in first if row["metric"] == "report_exact_rate")
        self.assertEqual(exact["paired_reports"], 2)
        self.assertEqual(exact["delta_candidate_minus_reference"], 0.5)
        self.assertGreaterEqual(exact["valid_bootstrap_replicates"], 1)

    def test_paired_family_bootstrap_resamples_closed_quartets(self) -> None:
        reference = []
        candidate = []
        reference_outcomes = {
            "family-a": [True, True, False, True],
            "family-b": [False, True, True, True],
        }
        candidate_outcomes = {
            "family-a": [True, True, True, True],
            "family-b": [False, True, True, True],
        }
        for family_id in reference_outcomes:
            for variant, ref_success, cand_success in zip(
                evaluation.FOUR_STATE_VARIANTS,
                reference_outcomes[family_id],
                candidate_outcomes[family_id],
                strict=True,
            ):
                reference.append(
                    _state_case(
                        len(reference), family_id, variant, success=ref_success
                    )
                )
                candidate.append(
                    _state_case(
                        len(candidate), family_id, variant, success=cand_success
                    )
                )
        first = evaluation.paired_family_bootstrap(
            reference,
            candidate,
            reference_label="base",
            candidate_label="care",
            replicates=200,
            seed=31,
        )
        second = evaluation.paired_family_bootstrap(
            reference,
            candidate,
            reference_label="base",
            candidate_label="care",
            replicates=200,
            seed=31,
        )
        self.assertEqual(first, second)
        retention = next(row for row in first if row["metric"] == "conditional_retention")
        self.assertEqual(retention["bootstrap_unit"], "family_id")
        self.assertEqual(retention["paired_families"], 2)
        self.assertAlmostEqual(retention["delta_candidate_minus_reference"], 1 / 3)
        self.assertGreater(retention["valid_bootstrap_replicates"], 0)

    def test_one_generation_set_supports_clean_and_gkd_references(self) -> None:
        rows = []
        for variant in evaluation.FOUR_STATE_VARIANTS:
            rows.append(
                _state_case(len(rows), "family-a", variant, success=True)
            )
        comparisons = evaluation.build_paired_family_comparisons(
            {
                "clean_sft": rows,
                "gkd_full": [dict(row) for row in rows],
                "care_opd": [dict(row) for row in rows],
            },
            reference_labels=["clean_sft", "gkd_full"],
            modalities=[],
            replicates=10,
            seed=7,
        )
        care_references = {
            row["reference_model"]
            for row in comparisons
            if row["candidate_model"] == "care_opd"
            and row["metric"] == "clean_task_success_rate"
        }
        self.assertEqual(care_references, {"clean_sft", "gkd_full"})

    def test_max_examples_selects_whole_families_and_allows_extra_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds_path = root / "seeds.private.jsonl"
            gold_path = root / "gold.private.jsonl"
            seeds = []
            gold = []
            for family_id in ("family-a", "family-b"):
                for variant in evaluation.FOUR_STATE_VARIANTS:
                    case_id = f"{family_id}::{variant}"
                    seeds.append(
                        {
                            "case_id": case_id,
                            "family_id": family_id,
                            "variant": variant,
                            "modality": "ultrasound",
                            "source_id": SOURCE_ID,
                            "student_messages": [
                                {"role": "system", "content": "system"},
                                {"role": "user", "content": "private"},
                            ],
                        }
                    )
                    gold.append(
                        {
                            "case_id": case_id,
                            "family_id": family_id,
                            "variant": variant,
                            "source_id": SOURCE_ID,
                            "gold_fields": (
                                _gold(case_id)["gold_fields"]
                                if variant in evaluation.SECONDARY_FIELD_VARIANTS
                                else []
                            ),
                        }
                    )
            seeds_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeds),
                encoding="utf-8",
            )
            gold_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold),
                encoding="utf-8",
            )
            loaded, contract = evaluation.load_locked_inputs(
                seeds_path, field_gold_path=gold_path, max_examples=1
            )
            self.assertEqual(len(loaded), 4)
            self.assertEqual({row["_eval_family_id"] for row in loaded}, {"family-a"})
            self.assertEqual(contract["family_mode"], "four_state")
            self.assertEqual(contract["family_count"], 1)
            self.assertEqual(
                contract["evaluation_track"], "raw_strict_extraction_challenge"
            )
            self.assertTrue(contract["challenge_mode"])

    def test_locked_input_join_and_public_output_do_not_leak_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds = root / "seeds.private.jsonl"
            gold = root / "field_gold.private.jsonl"
            raw_sentinel = "PRIVATE_REPORT_SENTINEL"
            seed_row = {
                "id": "private-case-id",
                "modality": "ultrasound",
                "source_id": SOURCE_ID,
                "student_messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": raw_sentinel},
                ],
            }
            seeds.write_text(json.dumps(seed_row, ensure_ascii=False) + "\n", encoding="utf-8")
            gold.write_text(json.dumps(_gold("private-case-id"), ensure_ascii=False) + "\n", encoding="utf-8")
            loaded, contract = evaluation.load_locked_inputs(seeds, field_gold_path=gold)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(loaded[0]["_eval_field_gold"]["gold_fields"]), 2)
            self.assertEqual(contract["case_count"], 1)

            rows = [_case(0, "private-case-id", exact=True, modality="ultrasound")]
            public = root / "public"
            evaluation.write_public_outputs(
                public_dir=public,
                summaries=[evaluation.summarize_model("care", rows)],
                comparisons=[],
                public_manifest={"run_fingerprint": "f" * 64},
            )
            combined = "\n".join(
                path.read_text(encoding="utf-8") for path in public.iterdir() if path.is_file()
            )
            self.assertTrue((public / "model_metrics.csv").is_file())
            self.assertTrue((public / "secondary_field_metrics.csv").is_file())
            self.assertTrue((public / "three_axis_metrics.csv").is_file())
            markdown = (public / "summary.md").read_text(encoding="utf-8")
            self.assertLess(
                markdown.index("Axis 1: state-sensitive task behavior"),
                markdown.index("Secondary localized-field results"),
            )
            self.assertNotIn(raw_sentinel, combined)
            self.assertNotIn("private-case-id", combined)
            self.assertNotIn("左叶", combined)

    def test_public_boundary_error_counts_strip_private_payloads(self) -> None:
        private_source = "PRIVATE_SOURCE_ID_SENTINEL"
        private_entity = "PRIVATE_ENTITY_SENTINEL|secret.field"
        private_modality = "PRIVATE_MODALITY_SENTINEL"
        private_failure = "PRIVATE_FAILURE_SENTINEL:raw-model-fragment"
        rows = []
        for variant in evaluation.FOUR_STATE_VARIANTS:
            row = _state_case(
                len(rows), "family-private", variant, success=variant == "clean"
            )
            row["modality"] = private_modality
            if variant == "clean":
                row["assessment"]["failure_types"] = [private_failure]
            if variant == "source_outage":
                row["assessment"]["boundary_errors"] = [
                    f"sections[0]:source_forbidden_at_boundary:{private_source}",
                    f"sections[1]:field_forbidden_at_boundary:{private_entity}",
                    "unknown_detail:PRIVATE_RESPONSE_FRAGMENT",
                ]
            rows.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary) / "public"
            evaluation.write_public_outputs(
                public_dir=public,
                summaries=[evaluation.summarize_model("care", rows)],
                comparisons=[],
                public_manifest={"public_protocol_fingerprint": "f" * 64},
            )
            combined = "\n".join(
                path.read_text(encoding="utf-8")
                for path in public.iterdir()
                if path.is_file()
            )
            self.assertNotIn(private_source, combined)
            self.assertNotIn(private_entity, combined)
            self.assertNotIn("PRIVATE_RESPONSE_FRAGMENT", combined)
            self.assertNotIn(private_modality, combined)
            self.assertNotIn(private_failure, combined)
            self.assertIn("boundary_source_emission", combined)
            self.assertIn("boundary_field_emission", combined)
            self.assertIn("unclassified_boundary_error", combined)
            self.assertIn("unclassified_failure", combined)
            self.assertIn("unknown", combined)

    def test_resume_recovers_only_a_corrupt_terminal_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.private.jsonl"
            valid = {
                "index": 0,
                "case_id": "case-a",
                "run_fingerprint": "run",
                "assessment": {"tp": 1},
            }
            path.write_bytes(
                (json.dumps(valid) + "\n" + '{"index":').encode("utf-8")
            )
            recovered = evaluation._recover_private_cases(
                path, expected={0: "case-a"}, run_fingerprint="run"
            )
            self.assertEqual(recovered, [valid])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
