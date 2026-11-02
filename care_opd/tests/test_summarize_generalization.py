from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "public_eval" / "summarize_generalization.py"
SPEC = importlib.util.spec_from_file_location("summarize_generalization", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _tcmeval_comparison(baseline: str, *, passed: bool = True) -> dict:
    low = -0.01 if passed else -0.03
    intervals = {
        endpoint: {"delta": 0.0, "low": low, "high": 0.03}
        for endpoint in MODULE.PRIMARY_GATE_ENDPOINTS
    }
    return {
        "baseline": baseline,
        "paired_units": 328,
        "delta_and_95_ci": intervals,
        "noninferiority": {
            "margin_absolute": -0.02,
            "endpoints": {
                endpoint: {
                    "margin": -0.02,
                    "ci95_low": low,
                    "pass": passed,
                }
                for endpoint in MODULE.PRIMARY_GATE_ENDPOINTS
            },
            "joint_pass": passed,
        },
    }


def _tcmbench_comparison(*, passed: bool = True) -> dict:
    low = -0.01 if passed else -0.03
    metrics = {
        f"{endpoint}_delta_care_minus_baseline": {
            "estimate": 0.0,
            "ci95_lower": low,
            "ci95_upper": 0.03,
            "noninferiority_margin": -0.02,
            "noninferiority_pass": passed,
        }
        for endpoint in MODULE.PRIMARY_GATE_ENDPOINTS
    }
    return {
        "overlap_clean": {
            "primary_case_cluster_bootstrap": {
                "paired_rows": 200,
                "metrics": metrics,
                "joint_primary_noninferiority_pass": passed,
            }
        }
    }


def _mtcmb_comparison(baseline: str, *, passed: bool = True) -> dict:
    low = -0.04 if passed else -0.06
    return {
        "baseline": baseline,
        "paired_units": 50,
        "margin_absolute": -0.05,
        "endpoints": {
            endpoint: {
                "delta": 0.0,
                "ci95_low": low,
                "ci95_high": 0.06,
                "margin": -0.05,
                "noninferiority_pass": passed,
            }
            for endpoint in MODULE.PRIMARY_GATE_ENDPOINTS
        },
        "joint_primary_noninferiority_pass": passed,
    }


def _internal_payload() -> dict:
    return {
        "models": {
            label: {
                "unified_metrics": {
                    "clean_unique_units": 2,
                    "paired_units": 6,
                    "clean_task_success_rate": 1.0,
                    "perturbed_task_success_rate": 0.5,
                    "conditional_retention": 0.5,
                },
                "diagnostics": {
                    "rollouts": 8,
                    "overall_mixed_task_success_rate": 0.625,
                },
            }
            for label in MODULE.MODEL_ORDER
        }
    }


def _formal_internal_payload() -> dict:
    records = []
    for label in MODULE.MODEL_ORDER:
        records.append(
            {
                "model": label,
                "benchmark": "QZBC grouped family-clean test",
                "track": "identity-bound family-clean paired variants",
                "track_provenance": "Internal formal / retrospective result-blind filter",
                "official_or_derived": "internal_formal_derived",
                "evidence_role": "in-domain mechanism contamination-sensitivity",
                "clean_n": 975,
                "perturbed_n": 2925,
                "unified_metrics": {
                    "clean_task_success_rate": 0.8,
                    "perturbed_task_success_rate": 0.7,
                    "conditional_retention": 0.75,
                    "paired_units": 2925,
                },
                "full_view_n": 4200,
                "full_view_tsr": 0.65,
                "effectiveness_display_key": "QZBC perturbed TSR",
                "robustness_display_key": "QZBC retention",
            }
        )
    return {
        "schema_version": "care-opd-qzbc-grouped-suite-v1",
        "checkpoint_identity_binding": {
            "status": "PASS_CURRENT_CHECKPOINT_BYTES"
        },
        "split_integrity_binding": {
            "status": "PASS_CURRENT_SPLIT_AND_TEST_RECOMPILE",
            "training_provenance": {
                "status": "CURRENT_DATA_AUDITED_TRAINING_BYTES_UNVERIFIED"
            },
        },
        "main_table_records": records,
    }


def _controlled_payload() -> dict:
    return {
        "controlled_derived_results": {
            "models": [
                {
                    "model": label,
                    "paired_units": 328,
                    "clean_task_success_rate": 0.6,
                    "perturbed_task_success_rate": 0.58,
                    "conditional_retention": 0.9,
                }
                for label in MODULE.MODEL_ORDER
            ],
            "care_paired_comparisons": [
                _tcmeval_comparison("clean_sft", passed=True),
                _tcmeval_comparison("gkd_full", passed=True),
            ],
        }
    }


def _tcmbench_payload() -> dict:
    metrics = {
        "paired_units": 200,
        "clean_task_success_rate": 0.6,
        "perturbed_task_success_rate": 0.57,
        "conditional_retention": 0.9,
    }
    return {
        "models": {
            label: {
                "overlap_clean": {
                    "public_paired_robustness_track": metrics,
                    "micro": {"n": 300, "accuracy": 0.55},
                }
            }
            for label in MODULE.MODEL_ORDER
        },
        "main_pairwise_comparisons": {
            "care_opd_vs_clean_sft": _tcmbench_comparison(passed=False),
            "care_opd_vs_gkd_full": _tcmbench_comparison(passed=True),
        },
    }


def _mtcmb_payload() -> dict:
    return {
        "main_table_records": [
            {
                "model": label,
                "track": "provenance_clean9_option_permutation",
                "clean_n": 50,
                "perturbed_n": 50,
                "unified_metrics": {
                    "paired_units": 50,
                    "clean_task_success_rate": 0.6,
                    "perturbed_task_success_rate": 0.58,
                    "conditional_retention": 0.9,
                },
            }
            for label in MODULE.MODEL_ORDER
        ],
        "primary_paired_noninferiority": [
            _mtcmb_comparison("clean_sft", passed=True),
            _mtcmb_comparison("gkd_full", passed=True),
        ],
    }


class GeneralizationSummaryTest(unittest.TestCase):
    def test_tcmeval_official_extraction(self) -> None:
        payload = {
            "dataset_sha256": "905cb28f145ba50f3b1c912265ff6649e0bb38f56adf6cd588d77ca768db92c1",
            "models": [
                {"model": label, "n": 328, "option_set_accuracy": index / 10}
                for index, label in enumerate(MODULE.MODEL_ORDER)
            ]
        }
        rows = MODULE.extract_tcmeval_official(payload)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1]["model"], "care_opd")
        self.assertEqual(rows[-1]["clean_task_success_rate"], 0.5)
        self.assertIsNone(rows[-1]["perturbed_task_success_rate"])

    def test_compact_matrix_refuses_duplicate_cells(self) -> None:
        rows = [
            {"model": "frozen", "key": "same", "value": 0.1},
            {"model": "frozen", "key": "same", "value": 0.2},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE._compact_matrix(rows, "key", "value")

    def test_internal_extraction_uses_paired_not_mixed_as_primary(self) -> None:
        models = {}
        for label in MODULE.MODEL_ORDER:
            models[label] = {
                "unified_metrics": {
                    "clean_unique_units": 2,
                    "paired_units": 6,
                    "clean_task_success_rate": 1.0,
                    "perturbed_task_success_rate": 0.5,
                    "conditional_retention": 0.5,
                },
                "diagnostics": {
                    "rollouts": 8,
                    "overall_mixed_task_success_rate": 0.625,
                },
            }
        row = MODULE.extract_internal({"models": models})[0]
        self.assertEqual(row["perturbed_task_success_rate"], 0.5)
        self.assertEqual(row["full_view_tsr"], 0.625)
        self.assertEqual(row["official_or_derived"], "internal_legacy")

    def test_formal_internal_extraction_requires_identity_and_split_disclosure(self) -> None:
        payload = _formal_internal_payload()
        rows = MODULE.extract_internal(payload)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1]["model"], "care_opd")
        self.assertEqual(rows[-1]["perturbed_task_success_rate"], 0.7)
        self.assertEqual(rows[-1]["full_view_n"], 4200)
        self.assertEqual(rows[-1]["official_or_derived"], "internal_formal_derived")
        payload["split_integrity_binding"]["status"] = "RETROSPECTIVE_UNVERIFIED"
        with self.assertRaisesRegex(ValueError, "split binding did not pass"):
            MODULE.extract_internal(payload)

    def test_three_public_gate_extractors_emit_closed_common_schema(self) -> None:
        rows = [
            *MODULE.extract_tcmeval_gates(_controlled_payload()),
            *MODULE.extract_tcmbench_gates(_tcmbench_payload()),
            *MODULE.extract_mtcmb_gates(_mtcmb_payload()),
        ]
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["benchmark"] for row in rows}, set(MODULE.PRIMARY_GATE_BENCHMARKS)
        )
        for benchmark in MODULE.PRIMARY_GATE_BENCHMARKS:
            self.assertEqual(
                {
                    row["baseline"]
                    for row in rows
                    if row["benchmark"] == benchmark
                },
                set(MODULE.PRIMARY_GATE_BASELINES),
            )
        self.assertIn(
            "conditional_retention_care_minus_baseline_ci95_low", rows[0]
        )
        self.assertIsInstance(rows[0]["joint_pass"], bool)

    def test_gate_extractor_rejects_missing_baseline_and_inconsistent_pass(self) -> None:
        missing = _controlled_payload()
        missing["controlled_derived_results"]["care_paired_comparisons"].pop()
        with self.assertRaisesRegex(ValueError, "expected exactly"):
            MODULE.extract_tcmeval_gates(missing)

        inconsistent = _mtcmb_payload()
        inconsistent["primary_paired_noninferiority"][0]["endpoints"][
            "conditional_retention"
        ]["noninferiority_pass"] = False
        with self.assertRaisesRegex(ValueError, "pass is inconsistent"):
            MODULE.extract_mtcmb_gates(inconsistent)

    def test_complete_summary_writes_public_gate_outputs_without_average(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {
                "internal_qzbc": _internal_payload(),
                "tcmeval_official": {
                    "dataset_sha256": "905cb28f145ba50f3b1c912265ff6649e0bb38f56adf6cd588d77ca768db92c1",
                    "models": [
                        {"model": label, "n": 328, "option_set_accuracy": 0.6}
                        for label in MODULE.MODEL_ORDER
                    ],
                },
                "tcmeval_controlled": _controlled_payload(),
                "tcmbench": _tcmbench_payload(),
                "mtcmb": _mtcmb_payload(),
            }
            paths = {}
            for name, payload in payloads.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path
            output = root / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                result = MODULE.summarize(
                    argparse.Namespace(
                        internal_qzbc=paths["internal_qzbc"],
                        tcmeval_official=paths["tcmeval_official"],
                        tcmeval_controlled=paths["tcmeval_controlled"],
                        cmb=None,
                        tcmbench=paths["tcmbench"],
                        mtcmb=paths["mtcmb"],
                        preregistration_lock=None,
                        output_dir=output,
                        require_complete=True,
                    )
                )
            gates = json.loads(
                (output / "noninferiority_gates.json").read_text(encoding="utf-8")
            )
            self.assertTrue((output / "noninferiority_gates.csv").is_file())
            self.assertTrue(gates["no_cross_benchmark_average"])
            clean = gates["clean_sft_primary_baseline_gate_summary"]
            self.assertEqual(clean["benchmarks_evaluated"], 3)
            self.assertEqual(clean["benchmarks_joint_passed"], 2)
            self.assertIn("not a cross-benchmark average", clean["interpretation"])
            self.assertEqual(
                result["public_noninferiority_gates"]["rows"], gates["rows"]
            )
            markdown = (output / "main_table.md").read_text(encoding="utf-8")
            self.assertIn("Pre-registered public non-inferiority gates", markdown)
            self.assertIn("2 / 3", markdown)


if __name__ == "__main__":
    unittest.main()
