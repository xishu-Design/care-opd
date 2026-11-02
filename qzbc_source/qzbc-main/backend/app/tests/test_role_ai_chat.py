from sqlalchemy import select

from app.agents.graphs.role_ai_chat_graph import RoleAIChatGraph
from app.db.models import RoleAIChat, RoleMemory, User
from app.services.role_ai_chat_service import save_role_ai_chat
from app.tests.conftest import auth_header, doctor_with_patient, login, register


def _result(role: str, question: str) -> dict:
    return {
        "answer": f"{role}:{question}",
        "trace_id": f"trace-{role}",
        "generated_response": {
            "title": "测试问答",
            "summary": f"{role}回答",
            "sections": [{"heading": "说明", "content": question, "source_ids": []}],
            "warnings": [],
            "next_actions": [],
            "disclaimer": "",
        },
        "llm_result": {"provider": "template", "model": "test", "content": question},
        "evidence_packet": {"visibility": "doctor_internal" if role == "doctor" else "student_learning"},
    }


def test_doctor_and_student_ai_chat_history_is_persistent_and_isolated(client, db_session):
    doctor = register(client, "chat_doctor", "doctor")
    student = register(client, "chat_student", "student")
    doctor_token = login(client, "chat_doctor", "doctor")
    student_token = login(client, "chat_student", "student")

    doctor_chat = save_role_ai_chat(
        db_session,
        user_id=doctor["id"],
        role="doctor",
        question="四君子汤的治法是什么？",
        result=_result("doctor", "四君子汤的治法是什么？"),
    )
    student_chat = save_role_ai_chat(
        db_session,
        user_id=student["id"],
        role="student",
        question="请教学解释脾气虚证。",
        result=_result("student", "请教学解释脾气虚证。"),
    )
    assert doctor_chat.created_at
    assert student_chat.created_at

    doctor_history = client.get("/api/doctor/ai-chat/history", headers=auth_header(doctor_token))
    student_history = client.get("/api/student/ai-chat/history", headers=auth_header(student_token))

    assert [item["question"] for item in doctor_history.json()["data"]] == ["四君子汤的治法是什么？"]
    assert [item["question"] for item in student_history.json()["data"]] == ["请教学解释脾气虚证。"]
    assert doctor_history.json()["data"][0]["role"] == "doctor"
    assert student_history.json()["data"][0]["role"] == "student"
    assert doctor_history.json()["data"][0]["created_at"]
    assert student_history.json()["data"][0]["created_at"]

    assert client.get("/api/doctor/ai-chat/history", headers=auth_header(student_token)).status_code == 403
    assert client.get("/api/student/ai-chat/history", headers=auth_header(doctor_token)).status_code == 403

    chats = db_session.scalars(select(RoleAIChat).order_by(RoleAIChat.id)).all()
    memories = db_session.scalars(select(RoleMemory).where(RoleMemory.memory_type == "ai_chat_turn")).all()
    assert len(chats) == 2
    assert {chat.role for chat in chats} == {"doctor", "student"}
    # Complete Q/A turns stay in the authoritative chat tables. They are no
    # longer duplicated into long-term memory where an old model answer could
    # contaminate a later response.
    assert memories == []

    doctor_chat_id = doctor_chat.id
    denied = client.delete(f"/api/doctor/ai-chat/{doctor_chat_id}", headers=auth_header(student_token))
    assert denied.status_code == 403
    deleted = client.delete(f"/api/doctor/ai-chat/{doctor_chat_id}", headers=auth_header(doctor_token))
    assert deleted.status_code == 200
    assert client.get("/api/doctor/ai-chat/history", headers=auth_header(doctor_token)).json()["data"] == []
    db_session.expire_all()
    assert db_session.get(RoleAIChat, doctor_chat_id) is None
    remaining_doctor_memories = db_session.scalars(
        select(RoleMemory).where(RoleMemory.owner_user_id == doctor["id"], RoleMemory.source == "role_ai_chat")
    ).all()
    assert remaining_doctor_memories == []


def test_role_ai_chat_rejects_empty_message(client):
    register(client, "empty_chat_doctor", "doctor")
    token = login(client, "empty_chat_doctor", "doctor")
    response = client.post("/api/doctor/ai-chat", headers=auth_header(token), json={"message": ""})
    assert response.status_code == 422


def test_role_ai_chat_stream_emits_answer_before_complete_and_persists(client):
    register(client, "stream_chat_doctor", "doctor")
    token = login(client, "stream_chat_doctor", "doctor")
    response = client.post(
        "/api/doctor/ai-chat/stream",
        headers=auth_header(token),
        json={"message": "请解释四君子汤的组成。"},
    )
    assert response.status_code == 200
    body = response.text
    assert "event: status" in body
    assert "event: answer" in body
    assert "event: complete" in body
    assert body.index("event: answer") < body.index("event: complete")

    history = client.get("/api/doctor/ai-chat/history", headers=auth_header(token))
    assert history.status_code == 200
    assert history.json()["data"][-1]["question"] == "请解释四君子汤的组成。"


def test_role_ai_chat_graph_uses_role_specific_evidence_visibility(client, db_session):
    doctor_data = register(client, "graph_chat_doctor", "doctor")
    student_data = register(client, "graph_chat_student", "student")
    doctor = db_session.get(User, doctor_data["id"])
    student = db_session.get(User, student_data["id"])

    doctor_result = RoleAIChatGraph(db_session).invoke(doctor, "四君子汤的组成是什么？")
    student_result = RoleAIChatGraph(db_session).invoke(student, "请解释四君子汤的组成。")

    assert doctor_result["generated_response"]["title"] == "医生专业问答"
    assert student_result["generated_response"]["title"] == "中医药教学问答"
    assert doctor_result["evidence_packet"]["visibility"] == "doctor_internal"
    assert student_result["evidence_packet"]["visibility"] == "student_learning"
    assert doctor_result["evidence_packet"]["backend_summary"]["memory_backend"] == "persistent_memory"
    assert student_result["evidence_packet"]["backend_summary"]["memory_backend"] == "persistent_memory"


def test_student_memory_preference_does_not_trigger_medical_answer(client, db_session):
    student_data = register(client, "memory_pref_student", "student")
    student = db_session.get(User, student_data["id"])

    result = RoleAIChatGraph(db_session).invoke(
        student,
        "请记住我是一个学生，复杂问题请引经据典，并推荐权威书籍。",
    )

    assert result["llm_result"]["provider"] == "memory_consolidation_policy"
    assert "优先依据可追溯的中医经典或权威教材" in result["answer"]
    assert "问诊流程" not in result["answer"]


def test_doctor_ai_chat_reads_only_current_doctors_bound_patient_roster(client):
    doctor_token, patient = doctor_with_patient(client)
    response = client.post(
        "/api/doctor/ai-chat",
        headers=auth_header(doctor_token),
        json={"message": "我现在的患者有谁？"},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    roster = data["evidence_packet"]["structured_context"]["doctor_patient_roster"]
    assert [item["patient_id"] for item in roster] == [patient["id"]]
    assert roster[0]["display_name"] == "Patient A"
    assert "Patient A" in data["generated_response"]["sections"][0]["content"]
    assert data["evidence_packet"]["backend_summary"]["business_context_types"] == ["doctor_patient_roster"]


def test_doctor_ai_chat_reads_named_patient_prescription_from_structured_record(client):
    doctor_token, patient = doctor_with_patient(client)
    created = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "咳嗽随诊方",
            "usage_method": "每日一剂，温服",
            "doctor_notes": "避免自行加减药材",
            "items": [
                {"herb_name": "川贝母", "dose": "2", "unit": "g"},
                {"herb_name": "麦冬", "dose": "4", "unit": "g"},
            ],
        },
    )
    assert created.status_code == 200, created.text

    response = client.post(
        "/api/doctor/ai-chat",
        headers=auth_header(doctor_token),
        json={"message": "我给 Patient A 开的处方是什么？"},
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    response_text = "\n".join(
        [data["generated_response"]["summary"]]
        + [section["content"] for section in data["generated_response"]["sections"]]
    )
    latest = data["evidence_packet"]["structured_context"]["doctor_patient_roster"][0]["latest_prescription"]
    assert latest["title"] == "咳嗽随诊方"
    assert latest["items"] == [
        {"herb_name": "川贝母", "dose": "2", "unit": "g", "notes": ""},
        {"herb_name": "麦冬", "dose": "4", "unit": "g", "notes": ""},
    ]
    assert "川贝母2g、麦冬4g" in response_text
    assert "四君子汤" not in response_text
    assert data["llm_result"]["provider"] == "doctor_business_context_policy"

    follow_up = client.post(
        "/api/doctor/ai-chat",
        headers=auth_header(doctor_token),
        json={"message": "它的用法和备注是什么？"},
    )
    assert follow_up.status_code == 200, follow_up.text
    follow_up_text = "\n".join(
        [follow_up.json()["data"]["generated_response"]["summary"]]
        + [section["content"] for section in follow_up.json()["data"]["generated_response"]["sections"]]
    )
    assert "每日一剂，温服" in follow_up_text
    assert "避免自行加减药材" in follow_up_text
    assert "四君子汤" not in follow_up_text


def test_doctor_streaming_chat_uses_same_structured_prescription_policy(client):
    doctor_token, patient = doctor_with_patient(client)
    created = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "结构化流式核对方",
            "items": [{"herb_name": "百合", "dose": "4", "unit": "g"}],
        },
    )
    assert created.status_code == 200, created.text

    response = client.post(
        "/api/doctor/ai-chat/stream",
        headers=auth_header(doctor_token),
        json={"message": "我给 Patient A 开的处方是什么？"},
    )

    assert response.status_code == 200
    streamed_answer = response.text.split("event: complete", maxsplit=1)[0]
    assert "百合4g" in streamed_answer
    assert "四君子汤" not in streamed_answer
    assert "doctor_business_context_policy" in response.text
