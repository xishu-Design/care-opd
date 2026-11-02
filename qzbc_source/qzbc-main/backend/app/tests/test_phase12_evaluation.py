from pathlib import Path

from app.core.config import get_settings
from app.evaluation.ablation import AblationRunner
from app.evaluation.dataset_loader import DATASET_FILES, dataset_manifest, load_all_datasets
from app.evaluation.runner import EvaluationRunner
from app.harness.phase4_runner import Phase4HarnessRunner
from app.harness.demo import reset_phase4_demo_data


def test_phase12_fixed_datasets_exceed_acceptance_threshold() -> None:
    manifest = dataset_manifest()
    datasets = load_all_datasets("full")

    assert manifest["version"] == "phase12-v1"
    assert manifest["total_cases"] == 185
    assert manifest["total_cases"] >= 100
    assert len(datasets) == 9
    assert len(DATASET_FILES) == 9
    assert all(cases for cases in datasets.values())
    assert len({case.case_id for cases in datasets.values() for case in cases}) == 185


def test_phase12_full_evaluation_generates_ppt_ready_report(db_session) -> None:
    run = EvaluationRunner(db_session, suite="full", reset_demo_data=True).run()
    report = run.report_json

    assert run.total_cases == 185
    assert run.passed == 185
    assert run.failed == 0
    assert run.pass_rate == 1.0
    assert report["overall_score"] == run.overall_score
    assert report["safety_score"] == 1.0
    assert report["agent_trace_score"] == 1.0
    assert report["llm_safety_score"] == 1.0
    assert report["metrics"]["safety"]["patient_high_risk_refusal_rate"] == 1.0
    assert report["metrics"]["safety"]["mcp_unauthorized_block_rate"] == 1.0
    assert report["metrics"]["evidence"]["evidence_source_id_coverage_rate"] == 1.0
    assert report["metrics"]["evidence"]["lightrag_retrieval_hit_rate"] == 1.0
    assert report["metrics"]["evidence"]["local_knowledge_retrieval_hit_rate"] == 0.0
    assert report["metrics"]["agent"]["human_confirm_interrupt_rate"] == 1.0
    assert report["metrics"]["agent"]["resume_success_rate"] == 1.0
    assert report["metrics"]["llm"]["schema_valid_rate"] == 1.0
    assert len(report["ppt_ready_tables"]) >= 3
    assert len(report["ppt_ready_charts"]) >= 2
    assert "# GuardTCM Evaluation Report" in run.markdown_report
    assert "核心结论" in run.markdown_report


def test_phase12_ablation_has_at_least_five_directional_comparisons(db_session) -> None:
    original_provider = get_settings().llm_provider
    report = AblationRunner(db_session).run(suite="quick")
    directional = [item for item in report["experiments"] if item["name"] != "template_vs_ollama"]

    assert report["experiment_count"] == 8
    assert len(directional) >= 5
    assert all(item["delta"] < 0 for item in directional)
    assert {item["name"] for item in directional} >= {
        "baseline_rag",
        "without_mcp_gateway",
        "without_evidence_packet",
        "without_safety_guard",
        "without_mem0",
        "without_graphiti_path",
        "without_human_confirm",
    }
    llm = next(item for item in report["experiments"] if item["name"] == "template_vs_ollama")
    assert "fallback_visible" in llm
    assert get_settings().llm_provider == original_provider == "template"
    assert report["ppt_ready_tables"]
    assert report["ppt_ready_charts"]


def test_phase12_ablation_config_is_versioned() -> None:
    path = Path(__file__).parents[1] / "evaluation" / "ablation" / "ablation_config.yaml"
    text = path.read_text(encoding="utf-8")
    assert "version: phase12-v1" in text
    assert text.count("  - name:") >= 8


def test_phase4_reset_handles_later_phase_foreign_key_references(db_session) -> None:
    runner = Phase4HarnessRunner(db_session, suite="quick", reset_demo_data=True)
    runner.review_risk_prescription()
    reset_phase4_demo_data(db_session)
    reset_phase4_demo_data(db_session)
