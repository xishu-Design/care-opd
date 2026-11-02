import hashlib

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.nodes.evidence_packet_node import EvidencePacketNode
from app.agents.nodes.mcp_tool_node import MCPToolNode
from app.agents.state import initial_agent_state
from app.core.config import get_settings
from app.db.base import Base
from app.db import models  # noqa: F401
from app.db.models import User
from app.evaluation.private_evidence_control import authorize_private_evidence_control
from app.schemas.agent import PrivateEvalEvidenceControlRequest


MATERIAL = "证据1（real_record）：右叶低回声结节7mm，边界清楚。"
DIGEST = hashlib.sha256(MATERIAL.encode("utf-8")).hexdigest()


@pytest.fixture()
def private_db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine, future=True)()
    session.add(
        User(
            id=1,
            username="private_eval_patient",
            hashed_password="not-used",
            role="patient",
            display_name="Private Eval Patient",
        )
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _request(condition: str = "clean") -> PrivateEvalEvidenceControlRequest:
    return PrivateEvalEvidenceControlRequest(
        protocol="three_role_paired_fact_v1",
        condition=condition,
        task_token="T-abcdefgh12345678",
        case_material_sha256=DIGEST,
        case_material=MATERIAL,
    )


def test_private_control_is_disabled_by_default() -> None:
    with pytest.raises(HTTPException) as exc:
        authorize_private_evidence_control(
            control=_request(),
            message=MATERIAL,
            supplied_key="x" * 32,
            client_host="127.0.0.1",
        )
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    ("condition", "expected_count", "expected_degraded"),
    [("clean", 1, False), ("evidence_unavailable_counterfactual", 0, True)],
)
def test_private_control_bypasses_live_adapters_and_enforces_count(
    private_db_session, condition: str, expected_count: int, expected_degraded: bool
) -> None:
    control = {
        "protocol": "three_role_paired_fact_v1",
        "condition": condition,
        "task_token": "T-abcdefgh12345678",
        "case_material_sha256": DIGEST,
        "case_material": MATERIAL,
        "expected_retrieved_evidence_count": expected_count,
        "authorization": "loopback_secret_and_runtime_gate",
    }
    state = initial_agent_state(
        role="patient",
        task_type="patient_education",
        user_id=1,
        session_id="private-eval-test",
        request_path="/api/patient/education-chat",
        graph_name="PatientEducationGraph",
        input_text=MATERIAL,
        private_eval_evidence_control=control,
    )
    state = MCPToolNode(private_db_session)(state)
    assert [row["tool_name"] for row in state["tool_requests"]] == [
        "private_eval_locked_evidence"
    ]
    packet_state = EvidencePacketNode(private_db_session)(state)
    packet = packet_state["evidence_packet"]
    assert len(packet["retrieved_evidence"]) == expected_count
    assert packet["degraded"] is expected_degraded
    audit = packet["backend_summary"]["private_eval_evidence_control"]
    assert audit["condition"] == condition
    assert audit["actual_retrieved_evidence_count"] == expected_count
    assert audit["evidence_count_verified"] is True
    assert audit["live_adapter_calls_bypassed"] is True
    assert audit["graphiti_bypassed"] is True
    assert audit["mem0_bypassed"] is True


def test_private_runtime_gate_fails_closed_on_wrong_secret(monkeypatch) -> None:
    settings = get_settings()
    names = (
        "private_eval_evidence_control_enabled",
        "private_eval_evidence_control_secret",
        "private_eval_isolated_database_ack",
        "app_env",
        "database_url",
        "graph_backend",
        "memory_backend",
        "graphiti_ingest_chat_history",
        "mem0_ingest_chat_history",
        "conversation_context_enabled",
        "lightrag_enabled",
        "qa_knowledge_enabled",
        "allow_adapter_fallback",
        "llm_allow_fallback",
        "llm_provider",
    )
    original = {name: getattr(settings, name) for name in names}
    try:
        settings.private_eval_evidence_control_enabled = True
        settings.private_eval_evidence_control_secret = "a" * 32
        settings.private_eval_isolated_database_ack = "I_CONFIRM_PRIVATE_ISOLATED_DATABASE"
        settings.app_env = "competition_eval"
        settings.database_url = "sqlite:///private-eval.private.db"
        settings.graph_backend = "persistent_graph"
        settings.memory_backend = "persistent_memory"
        settings.graphiti_ingest_chat_history = False
        settings.mem0_ingest_chat_history = False
        settings.conversation_context_enabled = False
        settings.lightrag_enabled = False
        settings.qa_knowledge_enabled = False
        settings.allow_adapter_fallback = False
        settings.llm_allow_fallback = False
        settings.llm_provider = "ollama"
        for variable in (
            "GRAPHITI_TELEMETRY_ENABLED",
            "MEM0_TELEMETRY",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_TRACING",
        ):
            monkeypatch.setenv(variable, "false")
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        authorized = authorize_private_evidence_control(
            control=_request(),
            message=MATERIAL,
            supplied_key="a" * 32,
            client_host="127.0.0.1",
        )
        assert authorized is not None
        assert authorized["expected_retrieved_evidence_count"] == 1
        with pytest.raises(HTTPException) as exc:
            authorize_private_evidence_control(
                control=_request(),
                message=MATERIAL,
                supplied_key="b" * 32,
                client_host="127.0.0.1",
            )
        assert exc.value.status_code == 403
    finally:
        for name, value in original.items():
            setattr(settings, name, value)
