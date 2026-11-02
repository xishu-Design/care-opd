from sqlalchemy import select

from app.db.models import RoleMemory, User
from app.services.memory_consolidation_service import MEMORY_SOURCE, MemoryAction, MemoryConsolidationAgent
from app.tests.conftest import auth_header, login, register


def _user(db_session, user_id: int, role: str) -> None:
    db_session.add(
        User(
            id=user_id,
            username=f"memory_{role}_{user_id}",
            hashed_password="test",
            role=role,
            display_name=f"Memory {role}",
        )
    )
    db_session.flush()


def test_agent_ignores_ordinary_medical_question(db_session):
    result = MemoryConsolidationAgent(db_session).consolidate_chat(
        owner_user_id=1,
        owner_role="student",
        user_message="脾气虚证和脾阳虚证有什么区别？",
        source_chat_id=10,
        trace_id="trace-10",
    )
    assert result.decision.action == MemoryAction.IGNORE
    assert db_session.scalars(select(RoleMemory)).all() == []


def test_agent_stores_only_user_authored_stable_preference(db_session):
    _user(db_session, 1, "patient")
    result = MemoryConsolidationAgent(db_session).consolidate_chat(
        owner_user_id=1,
        owner_role="patient",
        user_message="以后回答时请用通俗易懂的语言，少用术语。",
        source_chat_id=11,
        trace_id="trace-11",
    )
    assert result.decision.action == MemoryAction.ADD
    row = db_session.get(RoleMemory, result.memory_id)
    assert row.source == MEMORY_SOURCE
    assert row.memory_type == "response_preference"
    assert row.content_json["content"] == "以后回答时请用通俗易懂的语言，少用术语"
    assert "answer" not in row.content_json
    assert row.content_json["source_chat_ids"] == [11]


def test_same_memory_key_is_updated_not_duplicated(db_session):
    _user(db_session, 2, "doctor")
    agent = MemoryConsolidationAgent(db_session)
    first = agent.consolidate_chat(
        owner_user_id=2,
        owner_role="doctor",
        user_message="以后回答时请简短一些。",
        source_chat_id=20,
        trace_id="trace-20",
    )
    second = agent.consolidate_chat(
        owner_user_id=2,
        owner_role="doctor",
        user_message="以后回答时请详细展开。",
        source_chat_id=21,
        trace_id="trace-21",
    )
    assert first.memory_id == second.memory_id
    assert second.decision.action == MemoryAction.UPDATE
    rows = agent.list_active(owner_user_id=2, owner_role="doctor")
    assert len(rows) == 1
    assert "详细" in rows[0]["content"]
    assert rows[0]["revision"] == 2
    row = db_session.get(RoleMemory, first.memory_id)
    assert row.content_json["source_chat_ids"] == [21]
    assert row.content_json["previous_versions"][0]["source_chat_ids"] == [20]


def test_student_source_preference_is_canonicalized_and_acknowledged(db_session):
    agent = MemoryConsolidationAgent(db_session)
    message = "请记住我是一个学生，问复杂问题时希望你引经据典回答，推荐权威书籍上的知识点。"
    decision = agent.decide(role="student", user_message=message)

    assert decision.action == MemoryAction.ADD
    assert decision.category == "learning_preference"
    assert decision.memory_key == "teaching_source_style"
    assert decision.content == "学生偏好：复杂中医药知识问题优先依据可追溯的中医经典或权威教材回答；尽量注明书名和可定位篇章，并给出延伸阅读。"
    acknowledgement = agent.preference_acknowledgement(owner_role="student", user_message=message)
    assert acknowledgement is not None
    assert "问诊流程" not in acknowledgement
    assert "原话" not in acknowledgement


def test_one_off_request_for_detailed_explanation_is_not_a_long_term_preference(db_session):
    agent = MemoryConsolidationAgent(db_session)
    message = "请详细讲解脾气虚证与脾阳虚证的区别，并给出复习问题。"

    decision = agent.decide(role="student", user_message=message)

    assert decision.action == MemoryAction.IGNORE
    assert agent.preference_acknowledgement(owner_role="student", user_message=message) is None


def test_medical_case_with_long_term_history_is_not_a_preference_instruction():
    agent = MemoryConsolidationAgent(None)
    message = (
        "请对以下公开、已匿名化的中医教学病例进行辨证论治分析。"
        "病例：患者长期反复不适，既往检查结果仍需结合本次表现复核。"
    )

    decision = agent.decide(role="doctor", user_message=message)

    assert decision.action == MemoryAction.BLOCK
    assert decision.reason == "protected_medical_or_identity_fact_use_structured_record"
    assert agent.preference_acknowledgement(owner_role="doctor", user_message=message) is None


def test_identical_memory_deduplicates_and_keeps_both_sources(db_session):
    _user(db_session, 4, "student")
    agent = MemoryConsolidationAgent(db_session)
    first = agent.consolidate_chat(
        owner_user_id=4,
        owner_role="student",
        user_message="以后回答时请先说结论。",
        source_chat_id=40,
        trace_id="trace-40",
    )
    second = agent.consolidate_chat(
        owner_user_id=4,
        owner_role="student",
        user_message="以后回答时请先说结论。",
        source_chat_id=41,
        trace_id="trace-41",
    )
    assert second.decision.action == MemoryAction.UNCHANGED
    assert second.memory_id == first.memory_id
    assert db_session.get(RoleMemory, first.memory_id).content_json["source_chat_ids"] == [40, 41]


def test_agent_blocks_medical_facts_from_free_text_long_term_memory(db_session):
    decision = MemoryConsolidationAgent(db_session).decide(
        role="patient",
        user_message="请记住我对丹参过敏",
    )
    assert decision.action == MemoryAction.BLOCK
    assert "structured_record" in decision.reason


def test_temporary_fact_remains_short_term(db_session):
    decision = MemoryConsolidationAgent(db_session).decide(
        role="student",
        user_message="请记住我明天要复习方剂学",
    )
    assert decision.action == MemoryAction.IGNORE
    assert decision.reason == "temporary_context_stays_in_short_term_history"


def test_deleting_sole_source_chat_deactivates_derived_memory(db_session):
    _user(db_session, 3, "student")
    agent = MemoryConsolidationAgent(db_session)
    result = agent.consolidate_chat(
        owner_user_id=3,
        owner_role="student",
        user_message="我的学习目标是系统掌握方剂配伍。",
        source_chat_id=31,
        trace_id="trace-31",
    )
    synced = agent.detach_chat_source(owner_user_id=3, owner_role="student", source_chat_id=31)
    assert synced == [result.memory_id]
    assert agent.list_active(owner_user_id=3, owner_role="student") == []


def test_long_term_memory_api_is_role_scoped_and_supports_forget(client):
    register(client, "memory_student", "student")
    register(client, "memory_doctor", "doctor")
    student_token = login(client, "memory_student", "student")
    doctor_token = login(client, "memory_doctor", "doctor")

    saved = client.post(
        "/api/student/ai-memory",
        headers=auth_header(student_token),
        json={"content": "请记住我的学习目标是系统学习方剂学"},
    )
    assert saved.status_code == 200, saved.text
    memory_id = saved.json()["data"]["memory_id"]

    listed = client.get("/api/student/ai-memory", headers=auth_header(student_token))
    assert [item["id"] for item in listed.json()["data"]] == [memory_id]
    assert client.get("/api/student/ai-memory", headers=auth_header(doctor_token)).status_code == 403
    assert client.delete(f"/api/student/ai-memory/{memory_id}", headers=auth_header(doctor_token)).status_code == 403

    deleted = client.delete(f"/api/student/ai-memory/{memory_id}", headers=auth_header(student_token))
    assert deleted.status_code == 200
    assert client.get("/api/student/ai-memory", headers=auth_header(student_token)).json()["data"] == []


def test_explicit_api_rejects_allergy_as_memory(client):
    register(client, "memory_patient", "patient")
    token = login(client, "memory_patient", "patient")
    response = client.post(
        "/api/patient/ai-memory",
        headers=auth_header(token),
        json={"content": "请记住我对丹参过敏"},
    )
    assert response.status_code == 422
    assert "结构化资料" in response.text
