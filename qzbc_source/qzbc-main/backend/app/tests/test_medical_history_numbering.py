from app.services.medical_history_service import append_confirmed_encounter, normalize_medical_history


def test_legacy_rx_marker_is_displayed_as_patient_local_record_number():
    legacy = "无\n【诊疗记录 RX-51】2026-07-19；处方：辅助拟方；医生确认症状/诊疗背景：咳嗽"

    normalized = normalize_medical_history(legacy)

    assert normalized.startswith("【诊疗记录1】")
    assert "RX-51" not in normalized
    assert not normalized.startswith("无")


def test_new_encounter_continues_patient_local_sequence():
    history = "【诊疗记录 RX-51】2026-07-19；处方：处方一；医生确认症状/诊疗背景：咳嗽"

    updated = append_confirmed_encounter(
        history,
        visit_date="2026-07-20",
        prescription_title="处方二",
        symptoms="咽痛",
    )

    assert "【诊疗记录1】" in updated
    assert "【诊疗记录2】" in updated
    assert "RX-" not in updated
