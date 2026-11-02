from app.tests.conftest import auth_header, doctor_with_patient, login, register


def test_confirmed_symptoms_flow_to_history_and_student_quiz(client):
    doctor_token, patient = doctor_with_patient(client)
    symptom_context = "夜间咳嗽三天，咽干，少痰，睡眠受影响"
    created = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "人口学与症状贯通测试",
            "symptom_context": symptom_context,
            "usage_method": "每日一剂，水煎服，遵医嘱。",
            "doctor_notes": "需结合年龄和性别人工辨证。",
            "items": [{"herb_name": "麦冬", "dose": "9", "unit": "g"}],
        },
    )
    assert created.status_code == 200, created.text
    prescription = created.json()["data"]
    assert prescription["symptom_context"] == symptom_context
    assert prescription["patient_age_at_visit"] == 36
    assert prescription["patient_sex_at_visit"] == "female"

    reviewed = client.post(
        f"/api/doctor/prescriptions/{prescription['id']}/review",
        headers=auth_header(doctor_token),
    )
    assert reviewed.status_code == 200, reviewed.text
    snapshot = reviewed.json()["data"]["evidence_packet"]["structured_context"]["prescription"]
    assert snapshot["symptom_context"] == symptom_context
    assert snapshot["patient_age_at_visit"] == 36
    assert snapshot["patient_sex_at_visit"] == "female"

    confirmed = client.post(
        f"/api/doctor/prescriptions/{prescription['id']}/confirm",
        headers=auth_header(doctor_token),
    )
    assert confirmed.status_code == 200, confirmed.text

    patient_detail = client.get(
        f"/api/doctor/patients/{patient['id']}",
        headers=auth_header(doctor_token),
    ).json()["data"]
    assert symptom_context in patient_detail["medical_history"]
    assert "【诊疗记录1】" in patient_detail["medical_history"]
    assert "RX-" not in patient_detail["medical_history"]
    assert patient_detail["current_symptoms"] == symptom_context

    deidentified = client.post(
        f"/api/doctor/prescriptions/{prescription['id']}/deidentify",
        headers=auth_header(doctor_token),
    )
    assert deidentified.status_code == 200, deidentified.text
    case = deidentified.json()["data"]["anonymized_case"]
    assert case["patient_profile"]["age"] == 36
    assert case["patient_profile"]["gender"] == "female"
    assert case["symptoms"] == [symptom_context]
    assert patient["display_name"] not in str(case)

    register(client, "clinical_context_student", "student")
    student_token = login(client, "clinical_context_student", "student")
    analysis = client.get(
        f"/api/student/learning-cases/{deidentified.json()['data']['id']}/standard-analysis",
        headers=auth_header(student_token),
    )
    assert analysis.status_code == 200, analysis.text
    quizzes = analysis.json()["data"]["quiz"]
    packet_case = analysis.json()["data"]["evidence_packet"]["structured_context"]["learning_case"]["anonymized_case"]
    assert packet_case["patient_profile"]["age"] == 36
    assert packet_case["patient_profile"]["gender"] == "female"
    assert packet_case["symptoms"] == [symptom_context]
    assert quizzes
    for quiz in quizzes:
        assert "女" in quiz["question"]
        assert "36岁" in quiz["question"]
        assert symptom_context in quiz["question"]
        assert quiz["case_context"]["age"] == 36
        assert quiz["case_context"]["sex"] == "female"
        assert quiz["case_context"]["symptoms"] == [symptom_context]


def test_binding_existing_patient_updates_age_and_sex(client):
    register(client, "existing_profile_doctor", "doctor")
    register(client, "existing_profile_patient", "patient")
    doctor_token = login(client, "existing_profile_doctor", "doctor")
    bound = client.post(
        "/api/doctor/patients",
        headers=auth_header(doctor_token),
        json={
            "username": "existing_profile_patient",
            "password": "password123",
            "display_name": "已注册患者",
            "age": 58,
            "sex": "male",
            "current_symptoms": "腰膝酸软",
        },
    )
    assert bound.status_code == 200, bound.text
    assert bound.json()["data"]["age"] == 58
    assert bound.json()["data"]["sex"] == "male"
    assert bound.json()["data"]["current_symptoms"] == "腰膝酸软"


def test_doctor_can_create_unregistered_patient_then_review_with_default_gram_unit(client):
    register(client, "direct_create_doctor", "doctor")
    doctor_token = login(client, "direct_create_doctor", "doctor")
    created_patient = client.post(
        "/api/doctor/patients",
        headers=auth_header(doctor_token),
        json={
            "username": "direct_create_patient",
            "password": "patient_password123",
            "display_name": "医生端新建患者",
            "age": 45,
            "sex": "female",
            "current_symptoms": "胃脘隐痛、食欲欠佳",
        },
    )
    assert created_patient.status_code == 200, created_patient.text
    patient = created_patient.json()["data"]
    # The doctor-side create flow creates a real patient account, so no prior
    # patient registration step is required.
    assert login(client, "direct_create_patient", "patient", password="patient_password123")

    prescription = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient["id"],
            "title": "医生端直接建档审方测试",
            "symptom_context": "胃脘隐痛、食欲欠佳",
            "usage_method": "每日一剂，水煎服，遵医嘱。",
            "doctor_notes": "测试默认克单位。",
            "items": [{"herb_name": "白术", "dose": "9"}],
        },
    )
    assert prescription.status_code == 200, prescription.text
    data = prescription.json()["data"]
    assert data["items"] == [{"id": data["items"][0]["id"], "herb_name": "白术", "dose": "9", "unit": "g", "notes": None}]

    reviewed = client.post(
        f"/api/doctor/prescriptions/{data['id']}/review",
        headers=auth_header(doctor_token),
    )
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["data"]["evidence_packet"]["structured_context"]["prescription"]["items"][0]["unit"] == "g"
