from sqlalchemy import select

from app.db.models import AgentTrace, AgentTraceEvent, EvidencePacketLog, HarnessResult, MCPToolCall
from app.harness.case_loader import load_phase4_cases
from app.harness.case_schema import HarnessCase, PHASE4
from app.harness.phase4_runner import Phase4HarnessRunner
from app.harness.runners.phase4_classified_runners import phase4_runner_catalog
from app.tests.conftest import auth_header, login, register


def test_phase4_case_loader_schema_and_suites():
    cases = load_phase4_cases("full")
    assert len(cases) >= 16
    assert all(case.phase == PHASE4 for case in cases)
    assert all(case.case_id and case.category and case.severity and case.setup and case.action and case.expected is not None and case.assertions for case in cases)

    quick = load_phase4_cases("quick")
    demo = load_phase4_cases("demo")
    regression = load_phase4_cases("regression")
    assert quick
    assert [case.case_id for case in demo] == ["P4_DEMO_001"]
    assert regression

    catalog = phase4_runner_catalog()
    assert {runner.name for runner in catalog} >= {
        "RolePermissionRunner",
        "MCPPermissionRunner",
        "SafetyRunner",
        "EvidenceRunner",
        "DeidentificationRunner",
        "MemoryIsolationRunner",
        "GraphConsistencyRunner",
        "FallbackRunner",
        "DemoScenarioRunner",
    }


def test_phase4_runner_persists_results_metrics_and_failure_reason(db_session):
    run = Phase4HarnessRunner(db_session, suite="quick").run()
    assert run.phase == PHASE4
    assert run.suite == "quick"
    assert run.passed == run.total_cases
    assert run.metrics_json["mcp_block_rate"] == 1.0
    assert run.markdown_summary.startswith("# GuardTCM Phase 4 Harness Report")

    rows = db_session.scalars(select(HarnessResult).where(HarnessResult.run_id == run.id)).all()
    assert rows
    assert all(row.case_id and row.severity and row.actual_json is not None and row.expected_json is not None for row in rows)

    bad_case = HarnessCase(
        case_id="P4_UNKNOWN",
        name="unsupported case",
        category="demo_scenario",
        severity="low",
        role="system",
        action={"runner": "service_runner", "service": "missing"},
        assertions=["must_fail"],
    )
    failed = Phase4HarnessRunner(db_session, suite="demo").run_case(bad_case)
    assert failed.passed is False
    assert failed.failure_reason == "NO_HANDLER"


def test_phase4_trace_links_mcp_tool_calls_and_evidence_packets(db_session):
    run = Phase4HarnessRunner(db_session, suite="demo").run()
    result = db_session.scalar(select(HarnessResult).where(HarnessResult.run_id == run.id, HarnessResult.case_id == "P4_DEMO_001"))
    assert result is not None
    assert result.trace_id
    assert result.evidence_packet_id

    trace = db_session.scalar(select(AgentTrace).where(AgentTrace.trace_id == result.trace_id))
    assert trace is not None
    events = db_session.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == result.trace_id)).all()
    assert {event.node_name for event in events} >= {"AuthNode", "MCPToolNode", "EvidencePacketNode", "ResponseNode"}

    assert db_session.scalar(select(MCPToolCall).where(MCPToolCall.trace_id == result.trace_id)) is not None
    assert db_session.scalar(select(EvidencePacketLog).where(EvidencePacketLog.trace_id == result.trace_id)) is not None


def test_phase4_report_and_trace_api(client):
    register(client, "doctor_phase4_api", "doctor")
    token = login(client, "doctor_phase4_api", "doctor")

    run = client.post(
        "/api/harness/run",
        headers=auth_header(token),
        json={"phase": PHASE4, "suite": "demo", "reset_demo_data": True},
    )
    assert run.status_code == 200
    run_data = run.json()["data"]
    assert run_data["phase"] == PHASE4
    assert run_data["suite"] == "demo"
    assert run_data["passed"] == run_data["total_cases"]

    report = client.get(f"/api/harness/runs/{run_data['run_id']}/report", headers=auth_header(token))
    assert report.status_code == 200
    report_data = report.json()["data"]
    assert report_data["markdown_summary"].startswith("# GuardTCM Phase 4 Harness Report")
    assert report_data["competition_metrics"]["demo_scenario_passed"] is True
    assert report_data["key_traces"]
    assert report_data["overall_score"] >= 0
    assert report_data["safety_score"] >= 0
    assert report_data["evidence_score"] >= 0
    assert report_data["agent_trace_score"] >= 0
    assert report_data["llm_safety_score"] >= 0
    assert report_data["ppt_ready_tables"]
    assert report_data["ppt_ready_charts"]

    trace_id = report_data["key_traces"][0]
    trace = client.get(f"/api/harness/traces/{trace_id}", headers=auth_header(token))
    assert trace.status_code == 200
    trace_data = trace.json()["data"]
    assert trace_data["trace_id"] == trace_id
    assert trace_data["events"]
    assert trace_data["tool_calls"]
    assert trace_data["evidence_packets"]
