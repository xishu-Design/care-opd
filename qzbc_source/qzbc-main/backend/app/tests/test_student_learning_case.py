from app.db.models import KnowledgeClaim, LearningCase
from app.llm.schemas import LLMResult
from app.llm.service import LLMService
from app.services.learning_service import contextualize_student_quiz, evaluate_student_formula, generate_mtcmb_teaching_explanation
from app.tests.conftest import auth_header, create_reviewed_prescription, doctor_with_patient, login, register


def test_student_can_view_deidentified_case_and_submit_answer(client):
    doctor_token, patient = doctor_with_patient(client)
    prescription_id = create_reviewed_prescription(client, doctor_token, patient["id"])
    client.post(f"/api/doctor/prescriptions/{prescription_id}/confirm", headers=auth_header(doctor_token))
    client.post(f"/api/doctor/prescriptions/{prescription_id}/publish", headers=auth_header(doctor_token))
    deid = client.post(f"/api/doctor/prescriptions/{prescription_id}/deidentify", headers=auth_header(doctor_token))
    assert deid.status_code == 200
    case_id = deid.json()["data"]["id"]
    assert "Patient A" not in str(deid.json()["data"]["anonymized_case"])

    register(client, "student_case", "student")
    student_token = login(client, "student_case", "student")

    cases = client.get("/api/student/learning-cases", headers=auth_header(student_token))
    assert cases.status_code == 200
    assert cases.json()["data"][0]["id"] == case_id
    assert "Patient A" not in str(cases.json()["data"])

    detail = client.get(f"/api/student/learning-cases/{case_id}", headers=auth_header(student_token))
    assert detail.status_code == 200
    assert detail.json()["data"]["standard_analysis"] is None

    standard = client.get(f"/api/student/learning-cases/{case_id}/standard-analysis", headers=auth_header(student_token))
    assert standard.status_code == 200
    assert standard.json()["data"]["standard_analysis"]

    answer = client.post(
        f"/api/student/learning-cases/{case_id}/submit-answer",
        headers=auth_header(student_token),
        json={"answer_text": "存在用法用量风险，需要医生确认。", "has_risk": True, "risk_level": "yellow"},
    )
    assert answer.status_code == 200
    assert answer.json()["data"]["feedback"]

    progress = client.get("/api/student/learning-progress", headers=auth_header(student_token))
    assert progress.status_code == 200
    assert progress.json()["data"]["completed_training_count"] == 1
    assert progress.json()["data"]["submission_count"] == 1


def test_mtcmb_question_is_evaluated_as_knowledge_quiz_not_patient_case(db_session):
    payload = {
        "case_id": "CASE_MTCMB_KNOWLEDGE_TEST",
        "case_type": "mtcmb_student_evaluation",
        "visibility": "student_learning",
        "deidentified": True,
        "source_id": "SRC_MTCMB",
        "patient_profile": {},
        "symptoms": [],
        "prescription": {"items": []},
        "knowledge_points": ["中医知识评测"],
        "training_payload": {
            "question": "测试题：哪项正确？",
            "options": ["A.甲", "B.乙"],
            "answer": "B",
        },
    }
    case = LearningCase(title="MTCMB 知识测验", anonymized_case=payload, standard_analysis="题库标准答案：B")
    db_session.add(case)
    db_session.flush()

    result = evaluate_student_formula(case, [], answer_text="B.乙")
    quiz = contextualize_student_quiz(payload, [{"question": "测试题：哪项正确？", "answer": "B"}])

    assert result["is_correct"] is True
    assert result["submitted_answer"] == "B.乙"
    assert result["expected_answer"] == "B"
    assert result["reference_herbs"] == []
    assert "临床" in result["feedback"]
    assert quiz[0]["question"] == "测试题：哪项正确？"
    assert "病例资料" not in quiz[0]["question"]


def test_formula_training_marks_non_reference_herb_incorrect_and_reports_semantic_evidence(db_session):
    case = LearningCase(
        title="脱敏训练",
        anonymized_case={
            "case_type": "prescription_review_training",
            "visibility": "student_learning",
            "deidentified": True,
            "symptoms": ["失眠，乏力"],
            "prescription": {"items": [{"herb_name": "酸枣仁"}]},
        },
        standard_analysis="教学参考",
    )
    db_session.add_all(
        [
            case,
            KnowledgeClaim(
                claim_id="claim_suanzaoren_function",
                subject_label="酸枣仁",
                subject_type="herb",
                predicate="HAS_TCM_FUNCTION",
                object_label="养心安神",
                source_id="SRC_TEST",
                source_record_id="suanzaoren",
            ),
            KnowledgeClaim(
                claim_id="claim_gancao_function",
                subject_label="甘草",
                subject_type="herb",
                predicate="HAS_TCM_FUNCTION",
                object_label="补脾益气",
                source_id="SRC_TEST",
                source_record_id="gancao",
            ),
        ]
    )
    db_session.flush()

    result = evaluate_student_formula(case, ["甘草"], answer_text="因为她失眠", db=db_session)

    assert result["judgment"] == "incorrect"
    assert result["matched_herbs"] == []
    assert result["missing_herbs"] == ["酸枣仁"]
    assert result["extra_herbs"] == ["甘草"]
    assert result["semantic_assessments"][0]["verdict"] == "not_supported_by_current_case"
    assert result["semantic_assessments"][0]["known_functions"] == ["补脾益气"]
    assert result["student_reasoning"] == "因为她失眠"
    assert "判定：不正确" in result["feedback"]


def test_syndrome_training_scores_source_labels_without_inventing_a_formula(db_session):
    case = LearningCase(
        title="TCMEval-SDT 测试案例",
        anonymized_case={
            "case_type": "syndrome_differentiation_training",
            "visibility": "student_learning",
            "deidentified": True,
            "source_id": "SRC_TCMEVAL_SDT",
            "patient_profile": {"age": 38, "gender": "男"},
            "symptoms": ["鼻流血", "口渴鼻干"],
            "prescription": {"items": []},
            "training_payload": {
                "expected_pathogenesis": ["热伤肺络", "血热不固"],
                "expected_syndromes": ["热伤阳络", "血热妄行"],
            },
        },
        standard_analysis="来源标注的中医证候：热伤阳络；血热妄行。",
    )
    db_session.add(case)
    db_session.flush()

    partial = evaluate_student_formula(case, [], answer_text="考虑热伤阳络", db=db_session)
    correct = evaluate_student_formula(case, [], answer_text="热伤阳络，血热妄行", db=db_session)

    assert partial["judgment"] == "partial"
    assert partial["missing_herbs"] == ["血热妄行"]
    assert partial["reference_herbs"] == []
    assert correct["judgment"] == "correct"
    assert correct["score"] == 100


def test_syndrome_standard_analysis_is_source_labelled_and_does_not_call_model(client, db_session):
    case = LearningCase(
        title="TCMEval-SDT 标准解析测试",
        anonymized_case={
            "case_type": "syndrome_differentiation_training",
            "visibility": "student_learning",
            "deidentified": True,
            "source_id": "SRC_TCMEVAL_SDT",
            "patient_profile": {"age": 38, "gender": "男"},
            "symptoms": ["鼻流血"],
            "prescription": {"items": []},
            "training_payload": {
                "clinical_data": "男性，38岁。主诉鼻流血。",
                "expected_pathogenesis": ["热伤肺络"],
                "expected_syndromes": ["热伤阳络"],
            },
        },
        standard_analysis="来源标注的中医证候：热伤阳络。",
    )
    db_session.add(case)
    db_session.commit()
    register(client, "student_tcmeval_analysis", "student")
    token = login(client, "student_tcmeval_analysis", "student")

    response = client.get(f"/api/student/learning-cases/{case.id}/standard-analysis", headers=auth_header(token))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["llm_result"]["provider"] == "deterministic_tcmeval_sdt_standard_analysis"
    assert data["generated_response"]["summary"] == "来源标注的中医证候：热伤阳络。"
    assert data["graph_node_path"] == []


def test_mtcmb_standard_analysis_returns_immediately_without_agent_graph(client, db_session):
    """A question-bank item must not wait on optional Graphiti/Ollama services."""
    payload = {
        "case_id": "CASE_MTCMB_STANDARD_ANALYSIS_TEST",
        "case_type": "mtcmb_student_evaluation",
        "visibility": "student_learning",
        "deidentified": True,
        "source_id": "SRC_MTCMB",
        "knowledge_points": ["中医知识评测"],
        "training_payload": {
            "question": "测试题：哪项正确？",
            "options": ["A.甲", "B.乙"],
            "answer": "B",
        },
    }
    case = LearningCase(title="MTCMB 标准解析测试", anonymized_case=payload, standard_analysis="题库标准答案：B")
    db_session.add(case)
    db_session.commit()

    register(client, "student_mtcmb_analysis", "student")
    student_token = login(client, "student_mtcmb_analysis", "student")
    response = client.get(f"/api/student/learning-cases/{case.id}/standard-analysis", headers=auth_header(student_token))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["title"] == "MTCMB 标准解析测试"
    assert data["quiz"][0]["answer"] == "B"
    assert data["llm_result"]["provider"] == "deterministic_mtcmb_standard_analysis"
    assert data["graph_node_path"] == []


def test_mtcmb_ollama_explanation_keeps_answer_key_as_authority(db_session, monkeypatch):
    payload = {
        "case_type": "mtcmb_student_evaluation",
        "visibility": "student_learning",
        "deidentified": True,
        "source_id": "SRC_MTCMB",
        "training_payload": {
            "question": "六淫邪气中，最易伤肺的是？",
            "options": ["A.火", "B.燥"],
            "answer": "B",
        },
    }
    case = LearningCase(title="MTCMB 讲解测试", anonymized_case=payload, standard_analysis="题库标准答案：B")

    def fake_generate(self, evidence_packet, context):
        del self, evidence_packet
        assert context["constraints"]["answer"] == "B"
        assert context["constraints"]["options"] == ["A.火", "B.燥"]
        return LLMResult(
            request_id="llm_test",
            provider="ollama",
            model="qwen2.5:7b",
            content="燥邪易伤肺。",
            structured_output={"title": "题库教学讲解", "summary": "燥邪易伤肺。", "sections": []},
        )

    monkeypatch.setattr(LLMService, "generate_student_quiz_explanation", fake_generate)
    result = generate_mtcmb_teaching_explanation(db_session, case, user_id=1)

    assert result["llm_result"]["provider"] == "ollama"
    assert result["generated_response"]["summary"] == "燥邪易伤肺。"
