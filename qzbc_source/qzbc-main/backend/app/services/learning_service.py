import re

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import PrescriptionStatus
from app.db.models import (
    AllergyRecord,
    DeidentificationLog,
    LearningCase,
    MedicalHistory,
    PatientHealthRecord,
    PatientProfile,
    Prescription,
    PrescriptionReview,
    KnowledgeClaim,
    KnowledgeEntry,
    StudentAnswer,
    User,
)
from app.services.prescription_service import PrescriptionStateService, get_prescription_for_doctor
from app.services.medical_history_service import normalize_medical_history


def _age_range(age: int | None) -> str:
    if age is None:
        return "未知年龄段"
    start = max((age // 10) * 10, 0)
    return f"{start}-{start + 9}岁"


SENSITIVE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?:patient|HOSP|OPD|internal-record)[-_ ]?[A-Za-z0-9-]+", re.IGNORECASE),
    re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2})?"),
)
ADDRESS_MARKERS = ("省", "市", "区", "县", "街道", "路", "号")


def redact_learning_text(text: str | None, identity_values: list[str] | None = None) -> str:
    value = str(text or "")
    for identity in identity_values or []:
        if identity:
            value = value.replace(identity, "[已脱敏]")
    for pattern in SENSITIVE_PATTERNS:
        value = pattern.sub("[已脱敏]", value)
    if sum(marker in value for marker in ADDRESS_MARKERS) >= 2:
        return "[地址已脱敏]"
    return value


def create_deidentified_learning_case(db: Session, doctor: User, prescription_id: int) -> LearningCase:
    prescription = get_prescription_for_doctor(db, doctor.id, prescription_id)
    if prescription.status not in [
        PrescriptionStatus.DOCTOR_CONFIRMED.value,
        PrescriptionStatus.PUBLISHED_TO_PATIENT.value,
        PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value,
    ]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Prescription must be confirmed before deidentification")

    existing = db.scalar(select(LearningCase).where(LearningCase.source_prescription_id == prescription.id))
    if existing:
        return existing

    patient = db.get(User, prescription.patient_id)
    profile = db.scalar(select(PatientProfile).where(PatientProfile.user_id == prescription.patient_id))
    medical = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == prescription.patient_id))
    allergy = db.scalar(select(AllergyRecord).where(AllergyRecord.patient_id == prescription.patient_id))
    review = db.scalar(
        select(PrescriptionReview)
        .where(PrescriptionReview.prescription_id == prescription.id)
        .order_by(PrescriptionReview.created_at.desc())
    )

    identity_values = [patient.display_name, patient.username] if patient else []
    health = db.scalar(select(PatientHealthRecord).where(PatientHealthRecord.patient_id == prescription.patient_id))
    symptoms = redact_learning_text(
        prescription.symptom_context or (health.current_symptoms if health else ""),
        identity_values,
    )
    age_at_visit = prescription.patient_age_at_visit
    if age_at_visit is None and profile:
        age_at_visit = profile.age
    sex_at_visit = prescription.patient_sex_at_visit or (profile.sex if profile else None)
    anonymized_case = {
        "case_id": f"RX_DEID_{prescription.id}",
        "case_type": "prescription_review_training",
        "visibility": "student_learning",
        "deidentified": True,
        "patient_profile": {
            "age": age_at_visit,
            "age_range": _age_range(age_at_visit),
            "gender": sex_at_visit,
        },
        "symptoms": [symptoms] if symptoms else [],
        "allergy_tags": [redact_learning_text(item, identity_values) for item in allergy.summary.split(",")] if allergy and allergy.summary else [],
        "medical_history_summary": redact_learning_text(normalize_medical_history(medical.summary if medical else ""), identity_values),
        "prescription": {"items": [{"herb_name": item.herb_name, "dose": item.dose, "unit": item.unit} for item in prescription.items]},
        "source_id": "SRC_GUARDTCM_DEIDENTIFIED_CASE",
        "source_record_id": f"prescription:{prescription.id}",
        "case_note": "该案例已脱敏，仅用于审方训练。",
    }
    standard_analysis = review.review_summary if review else "请关注用法用量、过敏史和特殊人群风险。"
    risk_points = review.evidence_packet.get("rule_findings", []) if review else []
    evidence_chain = review.evidence_packet.get("retrieved_evidence", []) if review else []

    case = LearningCase(
        source_prescription_id=prescription.id,
        title=f"脱敏审方训练案例 #{prescription.id}",
        anonymized_case=anonymized_case,
        standard_analysis=standard_analysis,
        risk_points=risk_points,
        evidence_chain=evidence_chain,
    )
    db.add(case)
    db.flush()
    db.add(
        DeidentificationLog(
            prescription_id=prescription.id,
            learning_case_id=case.id,
            removed_fields=["姓名", "手机号", "地址", "身份证", "精确就诊时间", "医院内部编号"],
            checker_status="auto_checked",
        )
    )
    if prescription.status != PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value:
        PrescriptionStateService(db).transition(
            prescription,
            PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value,
            doctor,
            "create_learning_case",
        )
    db.commit()
    db.refresh(case)
    # A tiny defensive check: never leak the real display name into the case JSON.
    if patient and any(identity and identity in str(case.anonymized_case) for identity in identity_values):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Deidentification check failed")
    return case


def list_learning_cases(db: Session, *, kind: str | None = None) -> list[LearningCase]:
    cases = db.scalars(select(LearningCase).order_by(LearningCase.created_at.desc())).all()
    visible = [case for case in cases if is_student_learning_case(case)]
    if kind == "clinical":
        visible = [case for case in visible if _case_type(case) == "prescription_review_training"]
    elif kind == "syndrome":
        visible = [case for case in visible if is_syndrome_differentiation_training(case.anonymized_case)]
    elif kind == "quiz":
        visible = [case for case in visible if is_mtcmb_knowledge_evaluation(case.anonymized_case)]
    elif kind == "composition":
        visible = [case for case in visible if _case_type(case) == "prescription_composition_learning_candidate"]
    return sorted(visible, key=lambda case: (not _has_symptom_and_reference_formula(case), -case.id))


def _case_type(case: LearningCase) -> str:
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    return str(payload.get("case_type") or "")


def get_learning_case(db: Session, case_id: int) -> LearningCase:
    case = db.get(LearningCase, case_id)
    if not case or not is_student_learning_case(case):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning case not found")
    return case


def is_student_learning_case(case: LearningCase | None) -> bool:
    if not case or not isinstance(case.anonymized_case, dict):
        return False
    return case.anonymized_case.get("deidentified") is True and case.anonymized_case.get("visibility") == "student_learning"


def _has_symptom_and_reference_formula(case: LearningCase) -> bool:
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    symptoms = payload.get("symptoms") or payload.get("current_symptoms") or []
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    prescription = payload.get("prescription")
    items = prescription.get("items", []) if isinstance(prescription, dict) else prescription if isinstance(prescription, list) else []
    return bool([item for item in symptoms if str(item).strip()]) and bool(items)


def student_case_context(payload: dict) -> dict:
    """Return the deidentified clinical facts that every case quiz must carry."""
    profile = payload.get("patient_profile") if isinstance(payload.get("patient_profile"), dict) else {}
    age = profile.get("age")
    age_text = f"{age}岁" if age is not None else str(profile.get("age_range") or "年龄未提供")
    sex = str(profile.get("gender") or profile.get("sex") or "").strip()
    sex_label = {"female": "女", "male": "男"}.get(sex, sex or "性别未提供")
    symptoms = payload.get("symptoms") or payload.get("current_symptoms") or payload.get("symptom_summary") or []
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    symptom_list = [str(item).strip() for item in symptoms if str(item).strip()]
    summary = f"{sex_label}，{age_text}；症状/诊疗背景：{'；'.join(symptom_list) or '原始记录未提供'}"
    return {
        "age": age,
        "age_text": age_text,
        "sex": sex,
        "sex_label": sex_label,
        "symptoms": symptom_list,
        "summary": summary,
    }


def contextualize_student_quiz(payload: dict, items: list[dict]) -> list[dict]:
    """Attach case context only to genuine deidentified clinical cases.

    MTCMB entries are source question-bank items, not patient cases.  Adding
    invented "病例资料" placeholders to them makes a knowledge quiz look like a
    broken prescription-review case, so keep their original question wording.
    """
    if is_mtcmb_knowledge_evaluation(payload):
        return [
            {
                **dict(item),
                "case_context": {"mode": "knowledge_evaluation", "source_note": "原始题库未提供患者病例资料"},
            }
            for item in items
        ]
    context = student_case_context(payload)
    result: list[dict] = []
    for item in items:
        row = dict(item)
        question = str(row.get("question") or "请结合病例资料完成辨析。").strip()
        if not question.startswith("病例资料："):
            question = f"病例资料：{context['summary']}\n问题：{question}"
        row["question"] = question
        row["case_context"] = context
        result.append(row)
    return result


def submit_student_answer(
    db: Session,
    student: User,
    case_id: int,
    answer_text: str,
    has_risk: bool,
    risk_level: str | None,
    proposed_herbs: list[str] | None = None,
) -> tuple[StudentAnswer, dict]:
    case = get_learning_case(db, case_id)
    evaluation = evaluate_student_formula(case, proposed_herbs or [], answer_text=answer_text, db=db)
    feedback = evaluation["feedback"]
    answer = StudentAnswer(
        student_id=student.id,
        learning_case_id=case_id,
        answer_text=answer_text,
        has_risk=has_risk,
        risk_level=risk_level,
        feedback=feedback,
    )
    db.add(answer)
    db.commit()
    db.refresh(answer)
    return answer, evaluation


def evaluate_student_formula(case: LearningCase, proposed_herbs: list[str], *, answer_text: str = "", db: Session | None = None) -> dict:
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    if is_mtcmb_knowledge_evaluation(payload):
        return _evaluate_mtcmb_knowledge_answer(payload, answer_text)
    if is_syndrome_differentiation_training(payload):
        return _evaluate_syndrome_differentiation_answer(payload, answer_text)
    prescription = payload.get("prescription")
    source_items = prescription.get("items", []) if isinstance(prescription, dict) else prescription if isinstance(prescription, list) else []
    reference = [str(item.get("herb_name") or item.get("name") or "").strip() for item in source_items if isinstance(item, dict)]
    reference = [item for item in reference if item]
    submitted = list(dict.fromkeys(str(item).strip() for item in proposed_herbs if str(item).strip()))
    matched = [item for item in submitted if item in reference]
    missing = [item for item in reference if item not in submitted]
    extra = [item for item in submitted if item not in reference]
    semantic_assessments = _assess_extra_herbs(db, extra, reference, payload) if db else []
    symptom_related = [item["herb_name"] for item in semantic_assessments if item["direct_symptom_matches"]]
    function_related = [item["herb_name"] for item in semantic_assessments if item["function_matches"] and item["herb_name"] not in symptom_related]
    unsupported = [
        item["herb_name"]
        for item in semantic_assessments
        if not item["direct_symptom_matches"] and not item["function_matches"]
    ]
    knowledge_points = [str(item) for item in payload.get("knowledge_points", []) if item] or ["处方组成对照", "来源与证据边界"]
    if not submitted:
        feedback = "尚未填写候选方剂。请依据脱敏案例中明确提供的症状与教学材料填写药材名称，再提交进行教学对照。"
    elif not reference:
        feedback = "该教学记录未提供参考处方，系统仅记录你的学习思路，不做处方正确性判断。"
    else:
        if not missing and not extra:
            feedback = f"判定：正确。提交 {len(submitted)} 味，与参考组成完全一致。"
        elif not matched:
            feedback = (
                f"判定：不正确。提交 {len(submitted)} 味，与参考组成一致 0 味；"
                f"缺少 {'、'.join(missing)}，额外填写 {'、'.join(extra)}。"
            )
        else:
            feedback = (
                f"判定：部分正确。提交 {len(submitted)} 味，与参考组成一致 {len(matched)} 味；"
                f"缺少 {'、'.join(missing) or '无'}，额外填写 {'、'.join(extra) or '无'}。"
            )
        if symptom_related:
            feedback += f" 其中 {'、'.join(symptom_related)} 与本案例症状存在可追溯关联，但不能因此替代缺失的参考药材。"
        if function_related:
            feedback += f" {'、'.join(function_related)} 与参考方存在部分功效关联，但不能据此判定为等价替代。"
        if unsupported:
            feedback += f" 未检索到 {'、'.join(unsupported)} 与本案例症状或参考功效的直接可追溯关联。"
        feedback += " 此结果仅用于教学复盘，不构成临床处方建议。"
    return {
        "feedback": feedback,
        "student_reasoning": str(answer_text or "").strip(),
        "submitted_herbs": submitted,
        "reference_herbs": reference,
        "matched_herbs": matched,
        "missing_herbs": missing,
        "extra_herbs": extra,
        "semantic_assessments": semantic_assessments,
        "judgment": (
            "correct" if reference and not missing and not extra
            else "incorrect" if reference and not matched
            else "partial" if reference
            else "not_scored"
        ),
        "score": _learning_score(reference, matched, missing, extra, semantic_assessments),
        "knowledge_points": knowledge_points,
        "model_feedback": "模型教学反馈将在提交后基于脱敏案例与可追溯来源生成。",
        "model_provider": "",
    }


def _learning_score(
    reference: list[str],
    matched: list[str],
    missing: list[str],
    extra: list[str],
    assessments: list[dict],
) -> int | None:
    if not reference:
        return None
    exact = 100 * len(matched) / len(reference)
    penalty = 12 * len(extra) + 8 * len(missing)
    partial_credit = 5 * sum(bool(item["direct_symptom_matches"] or item["function_matches"]) for item in assessments)
    return max(0, min(100, round(exact - penalty + partial_credit)))


def _assess_extra_herbs(db: Session, herbs: list[str], reference: list[str], payload: dict) -> list[dict]:
    case_terms = _learning_case_terms(payload)
    reference_functions = {
        value
        for herb in reference
        for value in _herb_learning_terms(db, herb)["functions"]
    }
    assessments: list[dict] = []
    for herb in herbs:
        terms = _herb_learning_terms(db, herb)
        direct = _overlapping_learning_terms(case_terms, terms["symptoms"] | terms["syndromes"])
        function_matches = _overlapping_learning_terms(reference_functions, terms["functions"])
        assessments.append(
            {
                "herb_name": herb,
                "direct_symptom_matches": sorted(direct),
                "function_matches": sorted(function_matches),
                "known_functions": _preferred_chinese_terms(terms["functions"]),
                "known_symptoms": _preferred_chinese_terms(terms["symptoms"]),
                "known_syndromes": _preferred_chinese_terms(terms["syndromes"]),
                "source_ids": sorted(terms["source_ids"]),
                "verdict": "related_but_not_equivalent" if direct or function_matches else "not_supported_by_current_case",
            }
        )
    return assessments


def _learning_case_terms(payload: dict) -> set[str]:
    raw = payload.get("symptoms") or payload.get("current_symptoms") or []
    values = raw if isinstance(raw, list) else [raw]
    terms: set[str] = set()
    for value in values:
        terms.update(item.strip() for item in re.split(r"[，,、；;\s]+", str(value)) if len(item.strip()) >= 2)
    return terms


def _herb_learning_terms(db: Session, herb_name: str) -> dict[str, set[str]]:
    result = {"functions": set(), "symptoms": set(), "syndromes": set(), "source_ids": set()}
    for claim in db.scalars(select(KnowledgeClaim).where(KnowledgeClaim.subject_label == herb_name)).all():
        label = str(claim.object_label or "").strip()
        if not label:
            continue
        if claim.predicate in {"HAS_TCM_FUNCTION", "HAS_EFFICACY", "HAS_FUNCTION"}:
            result["functions"].add(label)
        elif claim.predicate in {"ASSOCIATED_WITH_TCM_SYMPTOM", "ASSOCIATED_WITH_TCM_SYNDROME", "ASSOCIATED_WITH", "REFERENCES_SYMPTOM"}:
            result["symptoms"].add(label)
        result["source_ids"].add(str(claim.source_id))
    for entry in db.scalars(select(KnowledgeEntry).where(KnowledgeEntry.entry_type == "herb")).all():
        metadata = entry.metadata_json or {}
        display_name = str(metadata.get("display_name") or entry.title or "").strip()
        aliases = {str(item).strip() for item in metadata.get("aliases", []) if str(item).strip()} if isinstance(metadata.get("aliases"), list) else set()
        if herb_name != display_name and herb_name not in aliases:
            continue
        result["functions"].update(_string_set(metadata.get("display_functions")) | _string_set(metadata.get("functions")))
        result["symptoms"].update(_string_set(metadata.get("symptoms")) | _string_set(metadata.get("indirect_symptoms")))
        result["syndromes"].update(_string_set(metadata.get("syndromes")) | _string_set(metadata.get("indirect_syndromes")))
        result["source_ids"].add(str(entry.source_id))
    return result


def _string_set(value: object) -> set[str]:
    return {str(item).strip() for item in value if str(item).strip()} if isinstance(value, list) else set()


def _preferred_chinese_terms(values: set[str]) -> list[str]:
    """Keep student-facing teaching terms Chinese when a bilingual source exists."""
    chinese = sorted(value for value in values if re.search(r"[\u4e00-\u9fff]", value))
    return chinese or sorted(values)


def _overlapping_learning_terms(left: set[str], right: set[str]) -> set[str]:
    return {
        candidate
        for candidate in right
        if len(candidate) >= 2 and any(term in candidate or candidate in term for term in left if len(term) >= 2)
    }


def is_mtcmb_knowledge_evaluation(payload: dict) -> bool:
    return str(payload.get("case_type") or "") == "mtcmb_student_evaluation" and str(payload.get("source_id") or "") == "SRC_MTCMB"


def is_syndrome_differentiation_training(payload: dict) -> bool:
    return str(payload.get("case_type") or "") == "syndrome_differentiation_training"


def build_mtcmb_standard_analysis(case: LearningCase) -> dict:
    """Return a fast, deterministic analysis for a source question-bank item.

    A MTCMB item has no patient case to retrieve from Graphiti/Mem0 and needs no
    generative model.  Running the full student-agent graph for it both adds no
    educational value and can block the page on optional external services.
    """
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    training = payload.get("training_payload") if isinstance(payload.get("training_payload"), dict) else {}
    question = str(training.get("question") or "原始题库未提供题干").strip()
    answer = str(training.get("answer") or case.standard_analysis or "原始题库未提供标准答案").strip()
    options = training.get("options")
    source_id = str(payload.get("source_id") or "SRC_MTCMB")
    source_title = str(training.get("original_source") or "MTCMB")
    explanation = (
        f"题目：{question}\n\n"
        f"题库标准答案：{answer}\n\n"
        "说明：该结果仅用于题库学习与评测。原始条目不是患者病例，不包含患者年龄、性别、症状、处方或临床诊疗结论。"
    )
    generated = {
        "title": "MTCMB 题库标准解析",
        "summary": f"该题的题库标准答案为：{answer}",
        "sections": [
            {"heading": "题目", "content": question, "source_ids": [source_id]},
            {"heading": "使用边界", "content": "本题仅作教学评测，不可据此拟方、诊断或推断患者情况。", "source_ids": [source_id]},
        ],
        "warnings": ["MTCMB 是教学题库，不属于临床诊疗证据。"],
        "next_actions": ["核对题干与选项", "完成下一道知识测验"],
        "disclaimer": "题库标准答案仅用于学习参考。",
    }
    evidence = {
        "source_id": source_id,
        "title": source_title,
        "snippet": f"题目：{question}\n标准答案：{answer}",
        "metadata": {"visibility": ["student_learning"], "non_clinical_authority": True},
    }
    return {
        "id": case.id,
        "title": case.title,
        "anonymized_case": payload,
        "standard_analysis": explanation,
        "risk_points": [],
        "evidence_chain": [evidence],
        "knowledge_cards": [
            {
                "title": point,
                "content": "该知识点来自 MTCMB 题库，仅用于教学复盘。",
                "source_id": source_id,
                "visibility": "student_learning",
            }
            for point in payload.get("knowledge_points", ["中医知识评测"])
            if point
        ],
        "quiz": [{"question": question, "options": options, "answer": answer, "source_id": source_id}],
        "generated_response": generated,
        "llm_result": {
            "provider": "deterministic_mtcmb_standard_analysis",
            "model": "mtcmb-answer-key-v1",
            "content": generated["summary"],
            "grounding": {"used_evidence_count": 1, "missing_evidence": False, "source_ids": [source_id]},
            "metadata": {"fallback_used": False, "degraded": False, "latency_ms": 0},
            "safety_flags": ["non_clinical_teaching_only"],
        },
        "evidence_packet": {
            "packet_id": f"mtcmb-standard-{case.id}",
            "visibility": "student_learning",
            "retrieved_evidence": [evidence],
            "rule_findings": [],
            "tool_calls": [],
            "degraded": False,
        },
        "trace_id": None,
        "graph_node_path": [],
        "degraded": False,
        "fallback_used": False,
    }


def build_syndrome_standard_analysis(case: LearningCase) -> dict:
    """Render the licensed TCMEval benchmark labels without LLM embellishment."""
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    training = payload.get("training_payload") if isinstance(payload.get("training_payload"), dict) else {}
    pathogenesis = _answer_terms(training.get("expected_pathogenesis"))
    syndromes = _answer_terms(training.get("expected_syndromes"))
    clinical_data = str(training.get("clinical_data") or "来源未提供完整临床资料").strip()
    source_id = str(payload.get("source_id") or "SRC_TCMEVAL_SDT")
    explanation = (
        f"脱敏临床资料：{clinical_data}\n\n"
        f"来源标注的中医病机：{'；'.join(pathogenesis) or '未提供'}。\n"
        f"来源标注的中医证候：{'；'.join(syndromes) or '未提供'}。\n\n"
        "说明：这是 TCMEval-SDT 公开基准的来源标签，用于学生辨证学习与答案对照。"
        "该数据未提供处方组成，系统不会把它扩写成方剂、剂量或真实患者诊疗建议。"
    )
    generated = {
        "title": "TCMEval-SDT 标准辨证解析",
        "summary": f"来源标注的中医证候：{'；'.join(syndromes) or '未提供'}。",
        "sections": [
            {"heading": "脱敏临床资料", "content": clinical_data, "source_ids": [source_id]},
            {"heading": "中医病机", "content": "；".join(pathogenesis) or "未提供", "source_ids": [source_id]},
            {"heading": "中医证候", "content": "；".join(syndromes) or "未提供", "source_ids": [source_id]},
            {"heading": "使用边界", "content": "仅限公开教学基准的标签对照，不包含处方，不用于真实诊疗。", "source_ids": [source_id]},
        ],
        "warnings": ["来源病例未提供处方组成，不得据此补造方剂或剂量。"],
        "next_actions": ["根据症状复盘病机推理", "完成下一条辨证训练"],
        "disclaimer": "公开教学基准标签仅用于学习参考。",
    }
    evidence = {
        "source_id": source_id,
        "title": "TCMEval-SDT 中医证候辨证基准",
        "snippet": f"病机：{'；'.join(pathogenesis)}\n证候：{'；'.join(syndromes)}",
        "metadata": {"visibility": ["student_learning"], "non_clinical_authority": True, "license": "CC-BY-4.0"},
    }
    return {
        "id": case.id,
        "title": case.title,
        "anonymized_case": payload,
        "standard_analysis": explanation,
        "risk_points": [],
        "evidence_chain": [evidence],
        "knowledge_cards": [
            {"title": "中医病机", "content": "；".join(pathogenesis) or "未提供", "source_id": source_id, "visibility": "student_learning"},
            {"title": "中医证候", "content": "；".join(syndromes) or "未提供", "source_id": source_id, "visibility": "student_learning"},
        ],
        "quiz": [],
        "generated_response": generated,
        "llm_result": {
            "provider": "deterministic_tcmeval_sdt_standard_analysis",
            "model": "tcmeval-source-label-v1",
            "content": generated["summary"],
            "grounding": {"used_evidence_count": 1, "missing_evidence": False, "source_ids": [source_id]},
            "metadata": {"fallback_used": False, "degraded": False, "latency_ms": 0},
            "safety_flags": ["non_clinical_teaching_only", "no_prescription_inferred"],
        },
        "evidence_packet": {"packet_id": f"tcmeval-standard-{case.id}", "visibility": "student_learning", "retrieved_evidence": [evidence], "rule_findings": [], "tool_calls": [], "degraded": False},
        "trace_id": None,
        "graph_node_path": [],
        "degraded": False,
        "fallback_used": False,
    }


def generate_mtcmb_teaching_explanation(db: Session, case: LearningCase, *, user_id: int | None = None) -> dict:
    """Generate an optional, source-bounded Ollama explanation for one MTCMB item.

    This is deliberately separate from the fast standard-answer endpoint.  A
    question and its answer key should remain usable even when a local model is
    stopped or slow; the model enriches the explanation but never decides the
    answer.
    """
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    if not is_mtcmb_knowledge_evaluation(payload):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Only MTCMB question-bank items support this explanation")

    analysis = build_mtcmb_standard_analysis(case)
    training = payload.get("training_payload") if isinstance(payload.get("training_payload"), dict) else {}
    question = str(training.get("question") or "").strip()
    answer = str(training.get("answer") or case.standard_analysis or "").strip()
    options = [str(item) for item in training.get("options", []) if str(item).strip()] if isinstance(training.get("options"), list) else []

    from app.llm.service import LLMService

    result = LLMService(db).generate_student_quiz_explanation(
        analysis["evidence_packet"],
        {
            "input_text": f"题目：{question}\n选项：{'；'.join(options)}\n题库标准答案：{answer}",
            "user_id": user_id,
            "constraints": {
                "question": question,
                "options": options,
                "answer": answer,
                "source_id": payload.get("source_id", "SRC_MTCMB"),
                "non_clinical_teaching_only": True,
            },
        },
    )
    return {
        "generated_response": result.structured_output,
        "llm_result": result.model_dump(mode="json"),
    }


def get_student_learning_progress(db: Session, student: User) -> dict:
    """Derive progress from persisted submissions, never from browser state."""
    answers = db.scalars(
        select(StudentAnswer)
        .where(StudentAnswer.student_id == student.id)
        .order_by(StudentAnswer.created_at.desc(), StudentAnswer.id.desc())
    ).all()
    case_ids = {item.learning_case_id for item in answers}
    cases = {
        item.id: item
        for item in db.scalars(select(LearningCase).where(LearningCase.id.in_(case_ids))).all()
    } if case_ids else {}

    latest_answers_by_case: dict[int, StudentAnswer] = {}
    for row in answers:
        latest_answers_by_case.setdefault(row.learning_case_id, row)

    quiz_answered = 0
    wrong_quiz = 0
    wrong_quiz_case_ids: list[int] = []
    recent_records: list[dict] = []
    for row in latest_answers_by_case.values():
        case = cases.get(row.learning_case_id)
        payload = case.anonymized_case if case and isinstance(case.anonymized_case, dict) else {}
        is_quiz = is_mtcmb_knowledge_evaluation(payload)
        correct: bool | None = None
        if is_quiz and case:
            quiz_answered += 1
            correct = bool(_evaluate_mtcmb_knowledge_answer(payload, row.answer_text).get("is_correct"))
            if not correct:
                wrong_quiz += 1
                wrong_quiz_case_ids.append(case.id)
    for row in answers:
        case = cases.get(row.learning_case_id)
        payload = case.anonymized_case if case and isinstance(case.anonymized_case, dict) else {}
        is_quiz = is_mtcmb_knowledge_evaluation(payload)
        correct: bool | None = None
        if is_quiz and case:
            correct = bool(_evaluate_mtcmb_knowledge_answer(payload, row.answer_text).get("is_correct"))
        if len(recent_records) < 8:
            recent_records.append(
                {
                    "answer_id": row.id,
                    "case_id": row.learning_case_id,
                    "case_title": case.title if case else "已删除学习案例",
                    "kind": "知识测验" if is_quiz else "组成/审方训练",
                    "correct": correct,
                    "created_at": row.created_at.isoformat(),
                }
            )

    completed = len(case_ids)
    accuracy = round((quiz_answered - wrong_quiz) * 100 / quiz_answered) if quiz_answered else None
    if wrong_quiz:
        suggestion = f"有 {wrong_quiz} 道知识测验答案需要复习；可在“测验”页打开题目并查看 AI 教学讲解。"
    elif quiz_answered:
        suggestion = "知识测验目前全部答对；建议继续完成新的题目，并复盘处方组成训练。"
    elif completed:
        suggestion = "已完成组成/审方训练；继续完成知识测验后可形成正确率统计。"
    else:
        suggestion = "尚未提交训练答案；完成一次知识测验或组成训练后，这里会自动更新。"
    return {
        "completed_training_count": completed,
        "submission_count": len(answers),
        "quiz_answered_count": quiz_answered,
        "wrong_quiz_count": wrong_quiz,
        "quiz_accuracy": accuracy,
        "wrong_quiz_case_ids": wrong_quiz_case_ids,
        "review_suggestion": suggestion,
        "recent_records": recent_records,
    }


def _evaluate_mtcmb_knowledge_answer(payload: dict, answer_text: str) -> dict:
    training = payload.get("training_payload") if isinstance(payload.get("training_payload"), dict) else {}
    expected = str(training.get("answer") or "").strip()
    submitted = str(answer_text or "").strip()
    is_correct = bool(expected) and _normalize_quiz_answer(submitted) == _normalize_quiz_answer(expected)
    feedback = (
        "回答正确。该结果仅表示与 MTCMB 题库标准答案一致，不能据此延伸为临床诊疗结论。"
        if is_correct
        else "本次答案与 MTCMB 题库标准答案不一致。请查看标准解析，复盘题干和选项；该题不是患者病例，不能据此拟方。"
    )
    return {
        "feedback": feedback,
        "submitted_herbs": [],
        "reference_herbs": [],
        "matched_herbs": [],
        "missing_herbs": [],
        "extra_herbs": [],
        "knowledge_points": [str(item) for item in payload.get("knowledge_points", []) if item] or ["中医知识评测"],
        # Keep the submitted value with the deterministic result so clients can
        # render an unambiguous correct/incorrect marker next to the option.
        "submitted_answer": submitted,
        "expected_answer": expected,
        "is_correct": is_correct,
        "model_feedback": "MTCMB 题库答案由结构化对照生成，不调用模型补充临床解释。",
        "model_provider": "deterministic_mtcmb_answer_check",
    }


def _normalize_quiz_answer(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or "").upper())
    match = re.match(r"([A-Z])(?:[.、．:：]|$)", compact)
    return match.group(1) if match else compact


def _evaluate_syndrome_differentiation_answer(payload: dict, answer_text: str) -> dict:
    """Score only the source-labelled syndrome terms for a benchmark case.

    The source does not provide a formula, so asking for herbs here would both
    be misleading and encourage the model/UI to invent a prescription.  The
    result intentionally separates a source-label match from clinical truth.
    """
    training = payload.get("training_payload") if isinstance(payload.get("training_payload"), dict) else {}
    expected = _answer_terms(training.get("expected_syndromes"))
    submitted_text = str(answer_text or "").strip()
    matched = [item for item in expected if item in submitted_text]
    missing = [item for item in expected if item not in submitted_text]
    if not submitted_text:
        judgment, score, feedback = "incorrect", 0, "判定：不正确。尚未填写辨证结论，请根据脱敏临床资料完成中医证候辨析。"
    elif expected and not missing:
        judgment, score, feedback = "correct", 100, f"判定：正确。已覆盖来源标注的中医证候：{'、'.join(expected)}。"
    elif matched:
        judgment = "partial"
        score = round(100 * len(matched) / len(expected)) if expected else None
        feedback = f"判定：部分正确。已识别 {'、'.join(matched)}；仍需复盘 {'、'.join(missing)}。"
    else:
        judgment, score = "incorrect", 0
        feedback = f"判定：不正确。提交内容未命中本案例来源标注的中医证候：{'、'.join(expected) or '未提供'}。"
    feedback += " 此为公开基准的教学标签对照，不构成对真实患者的诊断或处方建议。"
    return {
        "feedback": feedback,
        "submitted_herbs": [],
        "reference_herbs": [],
        "matched_herbs": matched,
        "missing_herbs": missing,
        "extra_herbs": [],
        "semantic_assessments": [],
        "judgment": judgment,
        "score": score,
        "knowledge_points": [str(item) for item in payload.get("knowledge_points", []) if item] or ["中医证候辨析"],
        "expected_syndromes": expected,
        "model_feedback": "辨证正误由来源标注的结构化标签对照生成；可另行查看 AI 教学讲解，但模型不决定得分。",
        "model_provider": "deterministic_tcmeval_sdt_label_check",
    }


def _answer_terms(value: object) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [term.strip() for item in values for term in re.split(r"[；;、，,\n]+", str(item or "")) if term.strip()]


def generate_student_submission_feedback(db: Session, student: User, case_id: int, evaluation: dict) -> dict:
    """Generate explanatory feedback only after deterministic teaching comparison."""
    case = get_learning_case(db, case_id)
    payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
    if is_mtcmb_knowledge_evaluation(payload) or is_syndrome_differentiation_training(payload):
        return evaluation
    source_id = str(payload.get("source_id") or "SRC_GUARDTCM_DEIDENTIFIED_CASE")
    evidence_packet = {
        "packet_id": f"student-submission-{case.id}",
        "visibility": "student_learning",
        "case_summary": "脱敏审方训练提交后的教学对照。",
        "retrieved_evidence": [
            {
                "source_id": source_id,
                "title": case.title,
                "snippet": case.standard_analysis,
                "metadata": {"visibility": ["student_learning"]},
            }
        ],
        "rule_findings": [],
    }
    semantic_text = "；".join(
        (
            f"{item.get('herb_name')}：判定={item.get('verdict')}；"
            f"本地收录功效={'、'.join(item.get('known_functions', [])[:8]) or '未收录'}；"
            f"可追溯症状关联={'、'.join(item.get('direct_symptom_matches', [])) or '无'}；"
            f"与参考功效交集={'、'.join(item.get('function_matches', [])) or '无'}；"
            f"来源={'、'.join(item.get('source_ids', [])) or '未收录'}"
        )
        for item in evaluation.get("semantic_assessments", [])
    ) or "无"
    context = {
        "user_id": student.id,
        "input_text": (
            f"结构化判定：{evaluation.get('judgment') or 'not_scored'}；得分：{evaluation.get('score') if evaluation.get('score') is not None else '未评分'}。"
            f"学生候选药材：{'、'.join(evaluation['submitted_herbs']) or '未填写'}。"
            f"学生拟方理由：{str(evaluation.get('student_reasoning') or '未填写')}。"
            f"教学对照：一致 {'、'.join(evaluation['matched_herbs']) or '无'}；"
            f"待复盘 {'、'.join(evaluation['missing_herbs']) or '无'}；"
            f"额外 {'、'.join(evaluation['extra_herbs']) or '无'}。"
            f"语义关联：{semantic_text}。"
            "请只解释以上结构化教学对照：先明确判定与缺失/额外药材；然后逐味用“药材功效—与本病例的可追溯关联—为何不等价替代”说明。"
            "若本地收录了某味药的功效，必须具体列出这些功效；若没有收录则明确写未收录，不能自行补造。"
            "还要单独评价学生拟方理由：它是否把病例中给出的症状与学生实际提交的药材建立了可追溯联系；若只是重复症状、没有说明药材依据，明确写“拟方理由不足”。该评价不改变结构化得分。"
            "不得声称药材可治疗此病例，不得反向推断参考方的临床适应证，不要给出处方、剂量或患者指令。"
        ),
        "constraints": {
            "visibility": "student_learning",
            "submission_feedback": True,
            "structured_judgment": evaluation.get("judgment"),
            "must_not_override_structured_judgment": True,
            "no_patient_treatment_claim": True,
            "evaluate_student_reasoning": True,
        },
    }
    from app.llm.service import LLMService

    result = LLMService(db).generate_student_analysis(evidence_packet, context)
    generated = result.structured_output or {}
    sections = generated.get("sections") if isinstance(generated.get("sections"), list) else []
    section_text = "\n".join(
        f"{str(section.get('heading') or '教学说明')}：{str(section.get('content') or '').strip()}"
        for section in sections
        if isinstance(section, dict) and str(section.get("content") or "").strip()
    )
    model_text = "\n".join(part for part in [str(generated.get("summary") or result.content or "").strip(), section_text] if part).strip()
    structured_prefix = f"结构化判定为：{_judgment_label(str(evaluation.get('judgment') or 'not_scored'))}。"
    evaluation["model_feedback"] = f"{structured_prefix}{model_text}" if model_text else f"{structured_prefix}{evaluation['feedback']}"
    evaluation["model_provider"] = result.provider
    return evaluation


def _judgment_label(value: str) -> str:
    return {"correct": "正确", "partial": "部分正确", "incorrect": "不正确", "not_scored": "未评分"}.get(value, "未评分")
