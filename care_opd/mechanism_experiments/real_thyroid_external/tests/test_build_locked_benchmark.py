import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_DIR = HERE.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
CARE_ROOT = PACKAGE_DIR.parents[1]
if str(CARE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CARE_ROOT / "src"))

from build_locked_benchmark import (  # noqa: E402
    BAD_HISTORY_SENTINEL,
    BenchmarkContractError,
    CONFLICT_SENTINEL,
    DEFAULT_TASK_MODE,
    SCHEMA_VERSION,
    TRANSFORM_PROTOCOL,
    VARIANTS,
    build_benchmark,
)
from care_opd.qzbc_contract import build_qzbc_messages, load_qzbc_prompts  # noqa: E402
from thyroid_verifier import assess_thyroid_response  # noqa: E402


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _materialized_payload(request):
    prompts = load_qzbc_prompts(CARE_ROOT.parent / "qzbc_source" / "qzbc-main")
    messages = build_qzbc_messages(request, prompts)
    return json.loads(messages[-1]["content"]), json.dumps(
        messages, ensure_ascii=False
    )


def _canonical(*, reviewed=True, allowed=True):
    source_text = "右叶见结节，大小10mm，边缘清晰。"
    return {
        "schema_version": "care_opd.real_thyroid_external.v1",
        "case_id": "THY-US-0123456789abcdef01234567",
        "report_group_hmac": "rg_" + "1" * 64,
        "source_id": "SRC_THY_PRIVATE_001",
        "source_record_hmac": "src_" + "2" * 64,
        "modality": "thyroid_ultrasound",
        "split": "locked_external_test",
        "linkage": {},
        "source": {
            "source_text_private": source_text,
            "deidentification_status": (
                "reviewed_private_use" if reviewed else "unreviewed"
            ),
            "model_input_allowed": allowed,
        },
        "entities": [
            {
                "entity_id": "N01",
                "entity_status": "candidate_unreviewed",
                "fields": [
                    {
                        "field_path": "ultrasound.size",
                        "candidate_value_private": "10mm",
                        "candidate_status": "reported_candidate",
                    },
                    {
                        "field_path": "ultrasound.margin",
                        "candidate_value_private": "清晰",
                        "candidate_status": "reported_candidate",
                    },
                    {
                        "field_path": "ultrasound.calcification",
                        "candidate_value_private": None,
                        "candidate_status": "unresolved_missingness",
                    },
                ],
            }
        ],
        "quality_private": {},
        "annotation": {},
        "governance": {},
        "private_provenance": {},
    }


def _exact(record, *, status="exact_verified", corrupt_hash=False):
    source = record["source"]["source_text_private"]
    quote = "10mm"
    start = source.index(quote)
    text_hash = hashlib.sha256(quote.encode("utf-8")).hexdigest()
    if corrupt_hash:
        text_hash = "0" * 64
    return {
        "case_id": record["case_id"],
        "report_group_hmac": record["report_group_hmac"],
        "modality": record["modality"],
        "entity_id": "N01",
        "field_path": "ultrasound.size",
        "candidate_value_private": quote,
        "normalized_value": quote,
        "unit": "mm",
        "reference_status": status,
        "support_status": "exact_supported",
        "evidence_spans": [
            {
                "start": start,
                "end": start + len(quote),
                "text_sha256": text_hash,
            }
        ],
    }


def _provided_clinical_gold(record):
    localized = _exact(record, status="clinical_gold_provided")
    localized["alignment_status"] = "exact_unique_verbatim"
    unlocalized = {
        "case_id": record["case_id"],
        "report_group_hmac": record["report_group_hmac"],
        "modality": record["modality"],
        "entity_id": "N01",
        "field_path": "ultrasound.margin",
        "candidate_value_private": "清晰",
        "normalized_value": "清晰",
        "unit": "",
        "reference_status": "clinical_gold_provided",
        "support_status": "clinical_gold_provided",
        "alignment_status": "not_verbatim",
        "evidence_spans": [],
    }
    return [localized, unlocalized]


class LockedBenchmarkBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.canonical_path = self.root / "canonical.private.jsonl"
        self.exact_path = self.root / "exact.private.jsonl"
        self.output = self.root / "output"

    def tearDown(self):
        self.temporary.cleanup()

    def _build(
        self,
        canonical_rows,
        exact_rows,
        *,
        task_mode=DEFAULT_TASK_MODE,
        output=None,
    ):
        _write_jsonl(self.canonical_path, canonical_rows)
        _write_jsonl(self.exact_path, exact_rows)
        return build_benchmark(
            self.canonical_path,
            self.exact_path,
            output or self.output,
            task_mode=task_mode,
            acknowledge_private_clinical_data=True,
        )

    def test_builds_four_state_family_gold_and_safe_manifest(self):
        record = _canonical()
        manifest = self._build([record], [_exact(record)])

        self.assertEqual(
            SCHEMA_VERSION, "care_opd.real_thyroid_external.locked_benchmark.v5"
        )
        self.assertEqual(TRANSFORM_PROTOCOL["version"], 5)
        self.assertEqual(
            manifest["schema_version"], f"{SCHEMA_VERSION}.aggregate_manifest"
        )

        seeds = _read_jsonl(self.output / "locked_external_qzbc.private.jsonl")
        gold_rows = _read_jsonl(self.output / "field_gold.private.jsonl")
        self.assertEqual([row["variant"] for row in seeds], list(VARIANTS))
        self.assertEqual([row["variant"] for row in gold_rows], list(VARIANTS))
        self.assertEqual(len(seeds), 4)
        self.assertEqual(len(gold_rows), 4)

        by_variant = {row["variant"]: row for row in seeds}
        gold_by_variant = {row["variant"]: row for row in gold_rows}
        for variant, seed in by_variant.items():
            with self.subTest(materialized_request=variant):
                _, rendered = _materialized_payload(seed["request"])
                for hidden_token in (
                    "bad_history",
                    "source_outage",
                    "evidence_conflict",
                    "CONTROLLED_",
                    "controlled_perturbation",
                ):
                    self.assertNotIn(hidden_token, rendered)
        clean = by_variant["clean"]
        self.assertEqual(clean["report_group_hmac"], record["report_group_hmac"])
        self.assertEqual(clean["source_id"], record["source_id"])
        self.assertEqual(clean["modality"], "thyroid_ultrasound")
        self.assertEqual(clean["request"]["role"], "doctor")
        self.assertEqual(clean["request"]["task_type"], "doctor_ai_chat")
        self.assertEqual(
            set(clean["target"]),
            {"title", "summary", "sections", "warnings", "next_actions", "disclaimer"},
        )
        self.assertEqual(clean["target"]["title"], "")
        self.assertEqual(clean["target"]["warnings"], [])
        self.assertEqual(
            clean["target"]["sections"],
            [
                {
                    "heading": "N01|ultrasound.size",
                    "content": "10mm",
                    "source_ids": [record["source_id"]],
                }
            ],
        )
        self.assertEqual(
            clean["request"]["evidence_packet"]["structured_context"][
                "allowed_extraction_schema"
            ],
            [{"entity_id": "N01", "field_paths": ["ultrasound.size"]}],
        )
        self.assertEqual(clean["evaluation_track"], DEFAULT_TASK_MODE)
        self.assertEqual(
            clean["request"]["evidence_packet"]["structured_context"][
                "required_output_slots"
            ],
            [
                {
                    "heading": "N01|ultrasound.size",
                    "source_ids": [record["source_id"]],
                }
            ],
        )
        self.assertNotIn(
            "ENTITY|field_path", clean["request"]["input_text"]
        )
        self.assertIn("条件式输出契约", clean["request"]["input_text"])
        self.assertIn(
            'title、summary、disclaimer 必须为空字符串 ""',
            clean["request"]["input_text"],
        )
        clean_contract = clean["request"]["constraints"][
            "conditional_output_contract"
        ]
        self.assertEqual(clean_contract["contract_version"], 1)
        self.assertEqual(clean_contract["with_citable_slots"]["title"], "")
        self.assertEqual(clean_contract["with_citable_slots"]["warnings"], [])
        self.assertFalse(
            clean_contract["without_citable_slots"][
                "medical_field_sections_allowed"
            ]
        )
        clean_payload, _ = _materialized_payload(clean["request"])
        self.assertEqual(
            clean_payload["constraints"]["conditional_output_contract"],
            clean_contract,
        )
        contract_text = json.dumps(clean_contract, ensure_ascii=False)
        for hidden_label in VARIANTS:
            self.assertNotIn(hidden_label, contract_text)
        self.assertEqual(
            len(clean["request"]["evidence_packet"]["retrieved_evidence"]), 1
        )

        for variant in VARIANTS:
            seed = by_variant[variant]
            gold = gold_by_variant[variant]
            expected_id = f"{record['case_id']}::{variant}"
            self.assertEqual(seed["family_id"], record["case_id"])
            self.assertEqual(seed["example_id"], expected_id)
            self.assertEqual(seed["case_id"], expected_id)
            self.assertEqual(seed["id"], expected_id + "_locked_external")
            self.assertEqual(seed["case_id"], gold["case_id"])
            self.assertEqual(gold["family_id"], record["case_id"])
            self.assertEqual(gold["variant"], variant)
            self.assertEqual(gold["reference_scope"], "verified_subset")

        for variant in ("clean", "bad_history"):
            gold = gold_by_variant[variant]
            self.assertIsNone(gold["boundary_policy"])
            self.assertEqual(len(gold["gold_fields"]), 1)
            self.assertEqual(gold["gold_fields"][0]["value"], "10mm")
            self.assertEqual(
                gold["gold_fields"][0]["target_value_private"], "10mm"
            )
            self.assertEqual(
                gold["gold_fields"][0]["source_id"], record["source_id"]
            )

        bad_history = by_variant["bad_history"]
        turn = bad_history["request"]["evidence_packet"]["conversation_context"][
            "recent_turns"
        ][0]
        self.assertTrue(turn["answer_trusted"])
        self.assertIn("按模型记忆", turn["answer"])
        request_json = json.dumps(bad_history["request"], ensure_ascii=False)
        self.assertNotIn(BAD_HISTORY_SENTINEL, request_json)
        self.assertNotIn('"controlled_variant"', request_json)
        self.assertNotIn('"bad_history"', request_json)
        self.assertEqual(
            bad_history["controlled_transform"]["sentinel"], BAD_HISTORY_SENTINEL
        )
        _, materialized_text = _materialized_payload(bad_history["request"])
        self.assertNotIn("bad_history", materialized_text)
        self.assertNotIn(BAD_HISTORY_SENTINEL, materialized_text)
        self.assertNotIn("CONTROLLED_", materialized_text)
        self.assertNotIn("controlled_perturbation", materialized_text)
        self.assertIn("按模型记忆", materialized_text)

        outage = by_variant["source_outage"]
        outage_packet = outage["request"]["evidence_packet"]
        self.assertEqual(outage_packet["retrieved_evidence"], [])
        self.assertTrue(outage_packet["degraded"])
        self.assertEqual(
            outage_packet["degraded_reasons"], ["no_citable_evidence_available"]
        )
        outage_payload, outage_text = _materialized_payload(outage["request"])
        self.assertEqual(outage_payload["evidence_packet"]["retrieved_evidence"], [])
        self.assertEqual(outage_payload["evidence_packet"]["structured_context"], {})
        self.assertIn("without_citable_slots", outage_text)
        self.assertNotIn("source_outage", outage_text)
        self.assertNotIn("CONTROLLED_", outage_text)
        self.assertNotIn("controlled_perturbation", outage_text)

        conflict = by_variant["evidence_conflict"]
        self.assertFalse(conflict["controlled_transform"]["sentinel_exposed_to_model"])
        conflict_packet = conflict["request"]["evidence_packet"]
        conflict_evidence = conflict_packet["retrieved_evidence"]
        self.assertEqual(
            [item["source_id"] for item in conflict_evidence], [record["source_id"]]
        )
        self.assertTrue(
            all(item["metadata"]["conflict"] for item in conflict_evidence)
        )
        self.assertTrue(
            all(item["metadata"]["citable"] is False for item in conflict_evidence)
        )
        self.assertNotIn(CONFLICT_SENTINEL, json.dumps(conflict_packet, ensure_ascii=False))
        self.assertEqual(
            conflict_packet["structured_context"]["required_output_slots"], []
        )
        self.assertEqual(
            conflict_packet["structured_context"]["allowed_extraction_schema"], []
        )
        conflict_payload, conflict_text = _materialized_payload(conflict["request"])
        conflict_context = conflict_payload["evidence_packet"]["structured_context"]
        self.assertEqual(conflict_context["required_output_slots"], [])
        self.assertEqual(conflict_context["allowed_extraction_schema"], [])
        self.assertFalse(
            conflict_payload["constraints"]["conditional_output_contract"][
                "without_citable_slots"
            ]["medical_field_sections_allowed"]
        )
        self.assertNotIn("evidence_conflict", conflict_text)
        self.assertNotIn("CONTROLLED_", conflict_text)
        self.assertNotIn("controlled_perturbation", conflict_text)
        self.assertNotIn("N01|ultrasound.size", conflict_text)

        for variant in ("source_outage", "evidence_conflict"):
            seed = by_variant[variant]
            gold = gold_by_variant[variant]
            self.assertEqual(gold["gold_fields"], [])
            self.assertTrue(gold["boundary_policy"]["required"])
            self.assertFalse(
                gold["boundary_policy"]["medical_field_sections_allowed"]
            )
            self.assertEqual(gold["boundary_policy"]["citable_source_ids"], [])
            self.assertTrue(seed["target"]["summary"])
            self.assertTrue(seed["target"]["warnings"])
            self.assertTrue(seed["target"]["next_actions"])
            self.assertTrue(seed["target"]["disclaimer"])
            self.assertTrue(seed["target"]["sections"])
            self.assertTrue(
                all("|" not in section["heading"] for section in seed["target"]["sections"])
            )
            self.assertTrue(
                all(not section["source_ids"] for section in seed["target"]["sections"])
            )

        boundary = json.loads(
            (self.output / "boundary_only.private.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        self.assertEqual(boundary["unresolved_field_count"], 2)
        self.assertEqual(
            boundary["empty_or_unresolved_semantics"], "unknown_not_negative"
        )
        self.assertTrue(boundary["clinical_interpretation_forbidden"])

        public_text = (self.output / "benchmark_manifest.json").read_text(
            encoding="utf-8"
        )
        for private_value in (
            record["case_id"],
            record["report_group_hmac"],
            record["source_id"],
            record["source"]["source_text_private"],
            "10mm",
            str(self.canonical_path),
        ):
            self.assertNotIn(private_value, public_text)
        self.assertEqual(manifest["counts"]["primary_report_families"], 1)
        self.assertEqual(manifest["counts"]["primary_qzbc_seed_rows"], 4)
        self.assertEqual(manifest["counts"]["rows_per_primary_family"], 4)
        self.assertEqual(manifest["counts"]["exact_verified_fields"], 1)
        self.assertEqual(
            manifest["counts"]["gold_field_instances_across_variants"], 2
        )
        self.assertEqual(
            manifest["gold_completeness"]["scope"], "verified_subset"
        )
        self.assertEqual(
            manifest["gold_completeness"]["exact_verified_coverage_of_populated_candidates"],
            0.5,
        )
        self.assertTrue(manifest["construction_protocol"]["result_blind"])
        self.assertEqual(
            set(manifest["by_variant"]),
            set(VARIANTS),
        )
        self.assertTrue((self.output / "benchmark_lock.private.json").is_file())
        private_lock = json.loads(
            (self.output / "benchmark_lock.private.json").read_text(encoding="utf-8")
        )
        self.assertEqual(private_lock["outputs"]["locked_external_qzbc.private.jsonl"]["rows"], 4)
        self.assertEqual(len(private_lock["inputs"]["canonical"]["sha256"]), 64)
        self.assertEqual(
            (self.output / ".gitignore").read_text(encoding="utf-8"),
            "*\n!.gitignore\n",
        )

    def test_task_tracks_filter_fields_and_bind_dual_source_four_state(self):
        record = _canonical()
        reference_rows = _provided_clinical_gold(record)
        raw_output = self.root / "raw"
        structured_output = self.root / "structured"

        raw_manifest = self._build(
            [record],
            reference_rows,
            task_mode="raw_report_hidden_gold",
            output=raw_output,
        )
        structured_manifest = build_benchmark(
            self.canonical_path,
            self.exact_path,
            structured_output,
            task_mode="structured",
            acknowledge_private_clinical_data=True,
        )

        self.assertEqual(
            raw_manifest["evaluation_track"], "raw_strict_extraction_challenge"
        )
        self.assertEqual(raw_manifest["counts"]["primary_report_families"], 1)
        self.assertEqual(raw_manifest["counts"]["primary_gold_fields"], 1)
        self.assertEqual(
            raw_manifest["counts"]["available_reference_fields_before_track_filter"],
            2,
        )
        self.assertEqual(raw_manifest["counts"]["available_span_localized_fields"], 1)
        self.assertEqual(
            raw_manifest["gold_completeness"]["scope"],
            "source_visible_span_localized_subset",
        )

        self.assertEqual(
            structured_manifest["evaluation_track"],
            "structured_evidence_faithfulness",
        )
        self.assertEqual(structured_manifest["counts"]["primary_report_families"], 1)
        self.assertEqual(structured_manifest["counts"]["primary_gold_fields"], 2)
        self.assertEqual(
            structured_manifest["gold_completeness"]["scope"],
            "all_nonempty_structured_clinical_gold",
        )
        self.assertNotEqual(
            raw_manifest["construction_protocol"]["transform_protocol_sha256"],
            structured_manifest["construction_protocol"]["transform_protocol_sha256"],
        )

        seeds = {
            row["variant"]: row
            for row in _read_jsonl(
                structured_output / "locked_external_qzbc.private.jsonl"
            )
        }
        gold = {
            row["variant"]: row
            for row in _read_jsonl(structured_output / "field_gold.private.jsonl")
        }
        raw_source = record["source_id"]
        structured_source = raw_source + "::STRUCTURED_CLINICAL_RECORD"
        expected_sources = [raw_source, structured_source]

        clean = seeds["clean"]
        self.assertEqual(clean["evaluation_track"], "structured_evidence_faithfulness")
        self.assertEqual(clean["allowed_source_ids"], expected_sources)
        evidence = clean["request"]["evidence_packet"]["retrieved_evidence"]
        self.assertEqual([item["source_id"] for item in evidence], expected_sources)
        self.assertEqual(evidence[1]["metadata"]["evidence_role"], "structured_clinical_record")
        self.assertTrue(evidence[1]["metadata"]["citable"])
        structured_content = json.loads(evidence[1]["content"])
        self.assertEqual(
            structured_content["fields"],
            [
                {
                    "entity_id": "N01",
                    "field_path": "ultrasound.margin",
                    "value": "清晰",
                },
                {
                    "entity_id": "N01",
                    "field_path": "ultrasound.size",
                    "value": "10mm",
                },
            ],
        )
        slots = clean["request"]["evidence_packet"]["structured_context"][
            "required_output_slots"
        ]
        self.assertEqual(
            slots,
            [
                {
                    "heading": "N01|ultrasound.margin",
                    "source_ids": [structured_source],
                },
                {
                    "heading": "N01|ultrasound.size",
                    "source_ids": [structured_source],
                },
            ],
        )
        self.assertNotIn("ENTITY|field_path", clean["request"]["input_text"])
        self.assertTrue(
            all(section["source_ids"] == [structured_source] for section in clean["target"]["sections"])
        )
        self.assertTrue(
            all(field["source_id"] == structured_source for field in gold["clean"]["gold_fields"])
        )

        outage = seeds["source_outage"]
        self.assertEqual(
            outage["request"]["evidence_packet"]["retrieved_evidence"], []
        )
        self.assertEqual(
            gold["source_outage"]["boundary_policy"]["forbidden_source_ids"],
            expected_sources,
        )
        conflict_evidence = seeds["evidence_conflict"]["request"][
            "evidence_packet"
        ]["retrieved_evidence"]
        self.assertEqual(
            [item["source_id"] for item in conflict_evidence],
            expected_sources,
        )
        self.assertTrue(
            all(item["metadata"]["conflict"] for item in conflict_evidence)
        )
        self.assertTrue(
            all(item["metadata"]["citable"] is False for item in conflict_evidence)
        )
        conflict_payload, conflict_text = _materialized_payload(
            seeds["evidence_conflict"]["request"]
        )
        self.assertEqual(
            conflict_payload["evidence_packet"]["structured_context"][
                "required_output_slots"
            ],
            [],
        )
        for forbidden in (
            "bad_history",
            "source_outage",
            "evidence_conflict",
            "CONTROLLED_",
            "controlled_perturbation",
        ):
            self.assertNotIn(forbidden, conflict_text)

        lock = json.loads(
            (structured_output / "benchmark_lock.private.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lock["evaluation_track"], "structured_evidence_faithfulness")
        self.assertEqual(lock["eligibility"]["eligible_primary_fields"], 2)
        self.assertEqual(lock["eligibility"]["available_span_localized_fields"], 1)

    def test_v5_oracle_passes_strict_verifier_for_both_tracks(self):
        record = _canonical()
        reference_rows = _provided_clinical_gold(record)

        for alias in ("raw", "structured"):
            with self.subTest(track=alias):
                output = self.root / f"oracle_{alias}"
                self._build(
                    [record],
                    reference_rows,
                    task_mode=alias,
                    output=output,
                )
                seeds = {
                    row["variant"]: row
                    for row in _read_jsonl(
                        output / "locked_external_qzbc.private.jsonl"
                    )
                }
                gold = {
                    row["variant"]: row
                    for row in _read_jsonl(output / "field_gold.private.jsonl")
                }
                for variant in VARIANTS:
                    self.assertEqual(
                        gold[variant]["schema_version"],
                        f"{SCHEMA_VERSION}.field_gold",
                    )
                    assessment = assess_thyroid_response(
                        gold[variant],
                        seeds[variant]["target"],
                        allowed_source_ids=seeds[variant]["allowed_source_ids"],
                    )
                    self.assertTrue(
                        assessment.passed,
                        msg=(
                            f"{alias}/{variant} oracle failed: "
                            f"{assessment.failure_types} {assessment.boundary_errors}"
                        ),
                    )

    def test_rejects_unknown_task_mode(self):
        record = _canonical()
        with self.assertRaises(BenchmarkContractError):
            self._build([record], [_exact(record)], task_mode="unknown_track")

    def test_four_state_rows_are_deterministic_across_output_directories(self):
        record = _canonical()
        self._build([record], [_exact(record)])
        first = {
            name: (self.output / name).read_bytes()
            for name in (
                "locked_external_qzbc.private.jsonl",
                "field_gold.private.jsonl",
                "boundary_only.private.jsonl",
            )
        }
        second_output = self.root / "second_output"
        build_benchmark(
            self.canonical_path,
            self.exact_path,
            second_output,
            acknowledge_private_clinical_data=True,
        )
        for name, content in first.items():
            self.assertEqual(content, (second_output / name).read_bytes())

    def test_empty_exact_layer_is_boundary_only_not_negative(self):
        manifest = self._build([_canonical()], [])
        self.assertEqual(manifest["status"], "BOUNDARY_ONLY")
        self.assertEqual(manifest["counts"]["primary_qzbc_seed_rows"], 0)
        self.assertEqual(
            manifest["counts"]["boundary_only_reports_without_exact_gold"], 1
        )
        self.assertEqual(
            (self.output / "locked_external_qzbc.private.jsonl").read_text(
                encoding="utf-8"
            ),
            "",
        )
        boundary = json.loads(
            (self.output / "boundary_only.private.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        self.assertEqual(boundary["benchmark_disposition"], "boundary_only_no_exact_gold")
        self.assertEqual(
            boundary["empty_or_unresolved_semantics"], "unknown_not_negative"
        )

    def test_rejects_agent_silver_in_primary_input(self):
        record = _canonical()
        with self.assertRaises(BenchmarkContractError):
            self._build([record], [_exact(record, status="consensus_silver")])

    def test_rejects_unreviewed_or_disallowed_source(self):
        for record in (_canonical(reviewed=False), _canonical(allowed=False)):
            with self.subTest(source=record["source"]):
                with self.assertRaises(PermissionError):
                    self._build([record], [])

    def test_rejects_span_hash_mismatch(self):
        record = _canonical()
        with self.assertRaises(BenchmarkContractError):
            self._build([record], [_exact(record, corrupt_hash=True)])

    def test_requires_explicit_private_data_acknowledgement(self):
        record = _canonical()
        _write_jsonl(self.canonical_path, [record])
        _write_jsonl(self.exact_path, [_exact(record)])
        with self.assertRaises(PermissionError):
            build_benchmark(self.canonical_path, self.exact_path, self.output)


if __name__ == "__main__":
    unittest.main()
