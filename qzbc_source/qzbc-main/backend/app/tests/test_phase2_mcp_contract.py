from sqlalchemy import select

from app.db.models import MCPToolCall
from app.mcp.registry import all_tool_specs
from app.tests.conftest import auth_header, doctor_with_patient, login, register


def test_tool_registry_contracts_and_tools_endpoint(client):
    register(client, "doctor_tools", "doctor")
    token = login(client, "doctor_tools", "doctor")
    response = client.get("/api/mcp/tools", headers=auth_header(token))
    assert response.status_code == 200
    tools = response.json()["data"]
    assert len(tools) == len(all_tool_specs())
    assert all(tool["name"] and tool["allowed_roles"] and tool["allowed_task_types"] for tool in tools)
    assert all(tool["requires_confirmation"] for tool in tools if tool["risk_level"] == "high")


def test_gateway_rejects_unregistered_and_cross_role_memory(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    del patient
    patient_token = login(client, "patient_a", "patient")

    missing = client.post(
        "/api/mcp/mock-call",
        headers=auth_header(doctor_token),
        json={"tool_name": "not_a_registered_tool", "arguments": {}},
    )
    assert missing.status_code == 200
    assert missing.json()["data"]["allowed"] is False
    assert missing.json()["data"]["error_code"] == "TOOL_NOT_REGISTERED"

    cross_role = client.post(
        "/api/mcp/mock-call",
        headers=auth_header(patient_token),
        json={
            "tool_name": "mock_search_role_memory",
            "arguments": {"owner_role": "doctor"},
            "context": {"task_type": "patient_education"},
        },
    )
    assert cross_role.status_code == 200
    assert cross_role.json()["data"]["allowed"] is False
    assert cross_role.json()["data"]["error_code"] == "MEMORY_SCOPE_VIOLATION"

    calls = db_session.scalars(select(MCPToolCall)).all()
    assert any(call.tool_name == "not_a_registered_tool" and not call.allowed for call in calls)
    assert any(call.tool_name == "mock_search_role_memory" and not call.allowed for call in calls)
