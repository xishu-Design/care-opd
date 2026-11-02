from app.db.models import PatientEducationChat
from app.tests.conftest import auth_header, create_reviewed_prescription, doctor_with_patient, login


def test_patient_can_view_only_published_patient_version(client):
    doctor_token, patient = doctor_with_patient(client)
    patient_token = login(client, "patient_a", "patient")
    prescription_id = create_reviewed_prescription(client, doctor_token, patient["id"])

    detail_before_publish = client.get(
        f"/api/patient/prescriptions/{prescription_id}",
        headers=auth_header(patient_token),
    )
    assert detail_before_publish.status_code == 404

    client.post(f"/api/doctor/prescriptions/{prescription_id}/confirm", headers=auth_header(doctor_token))
    client.post(f"/api/doctor/prescriptions/{prescription_id}/publish", headers=auth_header(doctor_token))

    detail = client.get(f"/api/patient/prescriptions/{prescription_id}", headers=auth_header(patient_token))
    assert detail.status_code == 200
    data = detail.json()["data"]
    assert "doctor_notes" not in data
    assert data["patient_instruction"]["doctor_note"] == "内部备注"
    assert "每日一剂" in data["patient_instruction"]["doctor_usage_method"]
    assert data["diet_precaution"]["doctor_note"] == "内部备注"
    assert data["diet_precaution"]["ai_explanation"]
    assert data["diet_precaution"]["llm"]["provider"]
    assert data["patient_instruction"]["safety_statement"]
    assert data["diet_precaution"]["content"]


def test_patient_page_preserves_doctor_diet_note_and_food_interaction_evidence(client):
    doctor_token, patient = doctor_with_patient(client)
    patient_token = login(client, "patient_a", "patient")
    created = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "饮食备注测试",
            "usage_method": "每日三次",
            "doctor_notes": "需要清淡饮食，不要吃辣的和冰的",
            "items": [{"herb_name": "人参", "dose": "4", "unit": "g", "notes": "饭后温服"}],
        },
    )
    prescription_id = created.json()["data"]["id"]
    assert client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(doctor_token)).status_code == 200
    assert client.post(f"/api/doctor/prescriptions/{prescription_id}/confirm", headers=auth_header(doctor_token)).status_code == 200
    assert client.post(f"/api/doctor/prescriptions/{prescription_id}/publish", headers=auth_header(doctor_token)).status_code == 200

    data = client.get(f"/api/patient/prescriptions/{prescription_id}", headers=auth_header(patient_token)).json()["data"]
    assert data["patient_instruction"]["doctor_note"] == "需要清淡饮食，不要吃辣的和冰的"
    assert data["patient_instruction"]["doctor_usage_method"] == "每日三次"
    assert data["patient_instruction"]["item_notes"] == [{"herb_name": "人参", "note": "饭后温服"}]
    assert data["diet_precaution"]["doctor_note"] == "需要清淡饮食，不要吃辣的和冰的"
    assert data["diet_precaution"]["food_interactions"][0]["food_items"] == ["萝卜"]
    assert data["diet_precaution"]["food_interactions"][0]["relation_label"] == "传统服药食忌参考"
    assert data["diet_precaution"]["food_interactions"][0]["source_url"].startswith("https://")


def test_patient_ai_chat_uses_current_published_prescription_and_doctor_note(client):
    doctor_token, patient = doctor_with_patient(client)
    patient_token = login(client, "patient_a", "patient")
    created = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "患者当前处方",
            "usage_method": "每日三次，饭后温服",
            "doctor_notes": "需要清淡饮食，不要吃辣的和冰的",
            "items": [{"herb_name": "人参", "dose": "4", "unit": "g"}],
        },
    )
    prescription_id = created.json()["data"]["id"]
    assert client.post(f"/api/doctor/prescriptions/{prescription_id}/review", headers=auth_header(doctor_token)).status_code == 200
    assert client.post(f"/api/doctor/prescriptions/{prescription_id}/confirm", headers=auth_header(doctor_token)).status_code == 200
    assert client.post(f"/api/doctor/prescriptions/{prescription_id}/publish", headers=auth_header(doctor_token)).status_code == 200

    response = client.post(
        "/api/patient/education-chat",
        headers=auth_header(patient_token),
        json={"message": "根据医生现在给我开的处方，我在饮食上有什么注意？"},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    rows = data["evidence_packet"]["structured_context"]["published_prescriptions"]
    assert rows[0]["id"] == prescription_id
    assert rows[0]["is_current"] is True
    assert rows[0]["patient_instruction"]["doctor_note"] == "需要清淡饮食，不要吃辣的和冰的"
    assert "需要清淡饮食" in data["generated_response"]["sections"][2]["content"]
    assert data["llm_result"]["grounding"]["missing_evidence"] is False
    assert "published_prescriptions" in data["evidence_packet"]["backend_summary"]["business_context_types"]


def test_patient_safety_guard_blocks_medical_requests(client):
    doctor_token, patient = doctor_with_patient(client)
    del doctor_token, patient
    patient_token = login(client, "patient_a", "patient")

    blocked = client.post(
        "/api/patient/education-chat",
        headers=auth_header(patient_token),
        json={"message": "我最近失眠，帮我开个方。"},
    )
    assert blocked.status_code == 200
    assert blocked.json()["data"]["allowed"] is False
    assert blocked.json()["data"]["reason"] == "patient_medical_request_not_allowed"

    allowed = client.post(
        "/api/patient/education-chat",
        headers=auth_header(patient_token),
        json={"message": "酸枣仁是什么？"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["data"]["allowed"] is True


def test_patient_education_stream_returns_status_answer_and_completion(client):
    doctor_token, patient = doctor_with_patient(client)
    del doctor_token, patient
    patient_token = login(client, "patient_a", "patient")

    response = client.post(
        "/api/patient/education-chat/stream",
        headers=auth_header(patient_token),
        json={"message": "酸枣仁是什么？"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: status" in response.text
    assert "event: answer" in response.text
    assert "event: complete" in response.text


def test_patient_education_history_persists_after_chat(client):
    doctor_token, patient = doctor_with_patient(client)
    del doctor_token, patient
    patient_token = login(client, "patient_a", "patient")

    response = client.post(
        "/api/patient/education-chat",
        headers=auth_header(patient_token),
        json={"message": "酸枣仁是什么？"},
    )
    assert response.status_code == 200

    history = client.get("/api/patient/education-chat/history", headers=auth_header(patient_token))
    assert history.status_code == 200
    assert history.json()["data"][-1]["question"] == "酸枣仁是什么？"
    assert history.json()["data"][-1]["answer"]


def test_patient_personal_eye_symptoms_use_deterministic_safety_policy(client):
    doctor_token, patient = doctor_with_patient(client)
    del doctor_token, patient
    patient_token = login(client, "patient_a", "patient")

    response = client.post(
        "/api/patient/education-chat",
        headers=auth_header(patient_token),
        json={"message": "我的眼睛疼痛疲劳，头脑发昏怎么办，在饮食上需要注意什么？"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["allowed"] is True
    assert data["llm_result"]["provider"] == "patient_safety_policy"
    assert "不能直接等同于中医的“肝火”" in data["safe_reply"]
    assert "视力下降" in data["safe_reply"]
    assert "具有清肝明目作用的中药" not in data["safe_reply"]


def test_patient_can_delete_only_own_chat_record(client, db_session):
    doctor_token, patient = doctor_with_patient(client)
    patient_token = login(client, "patient_a", "patient")
    response = client.post(
        "/api/patient/education-chat",
        headers=auth_header(patient_token),
        json={"message": "酸枣仁是什么？"},
    )
    assert response.status_code == 200
    history = client.get("/api/patient/education-chat/history", headers=auth_header(patient_token)).json()["data"]
    chat_id = history[-1]["id"]

    assert client.delete(f"/api/patient/education-chat/{chat_id}", headers=auth_header(doctor_token)).status_code == 403
    deleted = client.delete(f"/api/patient/education-chat/{chat_id}", headers=auth_header(patient_token))

    assert deleted.status_code == 200
    assert client.get("/api/patient/education-chat/history", headers=auth_header(patient_token)).json()["data"] == []
    db_session.expire_all()
    assert db_session.get(PatientEducationChat, chat_id) is None


def test_doctor_can_update_existing_patient_allergy_history(client):
    doctor_token, patient = doctor_with_patient(client)

    updated = client.patch(
        f"/api/doctor/patients/{patient['id']}",
        headers=auth_header(doctor_token),
        json={"allergy_history": "对丹参过敏"},
    )

    assert updated.status_code == 200
    assert updated.json()["data"]["allergy_history"] == "对丹参过敏"
    detail = client.get(f"/api/doctor/patients/{patient['id']}", headers=auth_header(doctor_token))
    assert detail.json()["data"]["allergy_history"] == "对丹参过敏"
