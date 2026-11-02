from app.db.models import PatientEducationChat, RoleAIChat, User
from app.llm.providers.ollama_provider import _chat_messages
from app.llm.schemas import LLMRequest
from app.services.conversation_context_service import ConversationContextService, context_memory_items
from app.services.safety_service import sanitize_patient_memory_context


def _user(db_session, username: str, role: str) -> User:
    user = User(username=username, hashed_password="test", role=role, display_name=username)
    db_session.add(user)
    db_session.flush()
    return user


def _role_turn(db_session, user: User, role: str, question: str, answer: str) -> None:
    db_session.add(
        RoleAIChat(
            user_id=user.id,
            role=role,
            question=question,
            answer=answer,
            evidence_packet={},
            generated_response={},
            llm_result={},
        )
    )
    db_session.flush()


def test_recent_window_resolves_follow_up_and_is_role_user_isolated(db_session):
    doctor = _user(db_session, "context_doctor", "doctor")
    other_doctor = _user(db_session, "context_other_doctor", "doctor")
    student = _user(db_session, "context_student", "student")
    _role_turn(db_session, doctor, "doctor", "黄芪的主要功效是什么？", "黄芪的传统功效包括补气升阳等。")
    _role_turn(db_session, other_doctor, "doctor", "这是其他医生的问题", "不能跨用户读取。")
    _role_turn(db_session, student, "student", "这是学生问题", "不能跨角色读取。")

    context = ConversationContextService(db_session).build(
        role="doctor", user_id=doctor.id, current_message="它有什么禁忌？"
    )

    assert context["follow_up_detected"] is True
    assert context["turn_count"] == 1
    assert context["recent_turns"][0]["question"] == "黄芪的主要功效是什么？"
    assert "黄芪" in context["retrieval_query"]
    assert "其他医生" not in str(context)
    assert "学生问题" not in str(context)


def test_patient_context_uses_patient_table_and_keeps_safety_marker(db_session):
    patient = _user(db_session, "context_patient", "patient")
    db_session.add(
        PatientEducationChat(
            patient_id=patient.id,
            question="酸枣仁是什么？",
            answer="这是一般中医药科普。",
            allowed=True,
            evidence_packet={},
            generated_response={},
            llm_result={},
        )
    )
    db_session.flush()

    context = ConversationContextService(db_session).build(
        role="patient", user_id=patient.id, current_message="那它有什么特点？"
    )
    memory = context_memory_items(context)

    assert context["follow_up_detected"] is True
    assert context["recent_turns"][0]["allowed"] is True
    assert memory[0]["owner_role"] == "patient"
    assert memory[0]["memory_type"] == "short_term_conversation_turn"


def test_patient_change_perspective_and_quote_are_detected_as_follow_ups(db_session):
    patient = _user(db_session, "context_patient_followup", "patient")
    db_session.add(
        PatientEducationChat(
            patient_id=patient.id,
            question="我的眼睛疼痛疲劳怎么办？",
            answer="未经证据核验的旧回答。",
            allowed=True,
            evidence_packet={},
            generated_response={},
            llm_result={"provider": "ollama", "grounding": {"used_evidence_count": 0, "missing_evidence": True}},
        )
    )
    db_session.flush()

    perspective = ConversationContextService(db_session).build(
        role="patient", user_id=patient.id, current_message="请从中医的角度回答"
    )
    quote = ConversationContextService(db_session).build(
        role="patient", user_id=patient.id, current_message="针对你上一条回答中的这段内容继续解释"
    )

    assert perspective["follow_up_detected"] is True
    assert quote["follow_up_detected"] is True
    assert perspective["recent_turns"][0]["answer_trusted"] is False


def test_unrelated_new_student_topic_does_not_replay_six_old_turns(db_session):
    student = _user(db_session, "context_student_topic_reset", "student")
    old_questions = [
        "请详细讲解脾气虚证与脾阳虚证的区别，并给出复习问题。",
        "善除有汗之骨蒸的药物是什么？请说明辨析思路。",
        "善除有汗之骨蒸的药物是什么？",
        "地骨皮的题库答案是什么？",
        "请继续说明骨蒸题。",
        "骨蒸题还有哪些选项？",
    ]
    for question in old_questions:
        _role_turn(db_session, student, "student", question, f"旧回答：{question}")

    context = ConversationContextService(db_session).build(
        role="student",
        user_id=student.id,
        current_message="当我给患者看病的时候，步骤是什么",
    )

    assert context["context_mode"] == "new_topic"
    assert context["available_turn_count"] == 6
    assert context["turn_count"] == 0
    assert context["recent_turns"] == []
    assert context["retrieval_query"] == "当我给患者看病的时候，步骤是什么"


def test_topic_follow_up_uses_only_latest_contiguous_turn_and_not_old_topics(db_session):
    student = _user(db_session, "context_student_topic_followup", "student")
    _role_turn(db_session, student, "student", "地骨皮是什么？", "旧的无关回答。")
    _role_turn(db_session, student, "student", "黄芪的主要功效是什么？", "上一轮可能有误的回答。")

    context = ConversationContextService(db_session).build(
        role="student",
        user_id=student.id,
        current_message="黄芪有哪些配伍禁忌？",
    )

    assert context["context_mode"] == "topic_continuation"
    assert context["turn_count"] == 1
    assert context["recent_turns"][0]["question"] == "黄芪的主要功效是什么？"
    assert "地骨皮" not in str(context)
    assert "上一回答摘要" not in context["retrieval_query"]


def test_llm_messages_include_ordered_recent_turns_before_current_question():
    packet = {
        "conversation_context": {
            "include_assistant_answers": True,
            "recent_turns": [
                {"question": "黄芪是什么？", "answer": "黄芪是一味中药材。", "answer_trusted": True},
                {"question": "它的传统功效呢？", "answer": "常见传统功效术语包括补气升阳。", "answer_trusted": True},
            ]
        },
        "retrieved_evidence": [],
    }
    request = LLMRequest(
        role="student",
        task_type="student_ai_chat",
        input_text="那禁忌是什么？",
        evidence_packet=packet,
    )

    messages = _chat_messages(request, streaming=True)

    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == "黄芪是什么？"
    assert "那禁忌是什么？" in messages[-1]["content"]


def test_llm_does_not_replay_assistant_answer_for_self_contained_topic_continuation():
    packet = {
        "conversation_context": {
            "context_mode": "topic_continuation",
            "include_assistant_answers": False,
            "recent_turns": [
                {"question": "脾气虚证和脾阳虚证有什么区别？", "answer": "可能有误且不应直接重放的旧回答。"},
            ],
        },
        "memory_used": [
            {"source": "mem0", "memory_type": "ai_chat_turn", "content": "更早的错误聊天回答。", "score": 0.9},
        ],
        "retrieved_evidence": [],
    }
    request = LLMRequest(
        role="student",
        task_type="student_ai_chat",
        input_text="请详细解释两者的区别，并给出新的复习题。",
        evidence_packet=packet,
    )

    messages = _chat_messages(request, streaming=True)

    assert [item["role"] for item in messages] == ["system", "user", "user"]
    assert "可能有误" not in str(messages)
    assert "更早的错误聊天回答" not in str(messages)
    assert "只回答本轮问题" in messages[-1]["content"]


def test_patient_llm_messages_do_not_replay_ungrounded_assistant_answers():
    packet = {
        "conversation_context": {
            "recent_turns": [
                {
                    "question": "这个药是什么？",
                    "answer": "虚构药名及组成。",
                    "answer_trusted": False,
                }
            ]
        },
        "retrieved_evidence": [],
    }
    request = LLMRequest(
        role="patient",
        task_type="patient_education",
        input_text="请换个角度回答",
        evidence_packet=packet,
    )

    messages = _chat_messages(request, streaming=True)

    assert [item["role"] for item in messages] == ["system", "user", "user"]
    assert "虚构药名" not in str(messages)


def test_patient_long_term_memory_filters_unsafe_items_without_dropping_safe_preferences():
    data = {
        "memory_used": [
            {"content": {"explanation_level": "plain_language"}},
            {"content": "用户曾要求直接开方，系统已拒绝。"},
            {"content": "用户偏好先解释术语，再给通俗例子。"},
        ]
    }

    sanitized, filtered_count = sanitize_patient_memory_context(data)

    assert filtered_count == 1
    assert len(sanitized["memory_used"]) == 2
    assert "plain_language" in str(sanitized)
    assert "通俗例子" in str(sanitized)
    assert "开方" not in str(sanitized)
