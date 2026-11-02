from __future__ import annotations

import re
import json
import urllib.request
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeClaim, KnowledgeEntry, User
from app.core.config import get_settings
from app.knowledge.entity_quality import is_valid_tcm_herb_label
from app.services.allergy_service import allergy_conflicts_for_herbs
from app.services.patient_service import get_patient_for_doctor


def suggest_prescription_candidates(db: Session, doctor: User, *, patient_id: int, symptoms: str = "", query: str = "") -> dict[str, Any]:
    patient = get_patient_for_doctor(db, doctor.id, patient_id)
    search_text = (symptoms or query or patient.current_symptoms or patient.body_state or "").strip()
    claims = db.scalars(select(KnowledgeClaim).order_by(KnowledgeClaim.subject_label, KnowledgeClaim.object_label)).all()
    matched = [claim for claim in claims if _claim_is_eligible(claim) and _matches(claim, search_text)]
    herbs: dict[str, dict[str, Any]] = {}
    for claim in matched:
        if not _is_recommendable_herb(claim.subject_label):
            continue
        item = herbs.setdefault(claim.subject_label, _new_herb_candidate(claim.subject_label))
        if claim.predicate in {"HAS_TCM_FUNCTION", "HAS_EFFICACY", "HAS_FUNCTION"}:
            item["functions"].append(claim.object_label)
        if claim.predicate in {"ASSOCIATED_WITH_TCM_SYMPTOM", "ASSOCIATED_WITH_TCM_SYNDROME", "ASSOCIATED_WITH", "REFERENCES_SYMPTOM"}:
            item["symptoms"].append(claim.object_label)
        item["source_ids"].add(claim.source_id)
        item["risk_labels"].add(claim.content_risk_level)
        item["match_score"] += _claim_match_score(claim, search_text)
        _append_reason(item, _claim_reason(claim))

    # Some imported herb catalogues store reviewed Chinese efficacy terms in
    # entry metadata but do not have a corresponding graph claim.  Include
    # those entries in retrieval so the catalogue and candidate panel do not
    # disagree.  Raw DragonTCM rows remain excluded from recommendations.
    for entry in db.scalars(select(KnowledgeEntry).where(KnowledgeEntry.entry_type == "herb")).all():
        if entry.source_id == "SRC_DRAGON_TCM":
            continue
        metadata = entry.metadata_json or {}
        herb_name = str(metadata.get("display_name") or entry.title)
        if not _is_recommendable_herb(herb_name):
            continue
        functions = _strings(metadata.get("display_functions")) or _strings(metadata.get("functions"))
        symptoms = _strings(metadata.get("symptoms"))
        syndromes = _strings(metadata.get("syndromes"))
        indirect_functions = _strings(metadata.get("indirect_functions"))
        indirect_symptoms = _strings(metadata.get("indirect_symptoms"))
        indirect_syndromes = _strings(metadata.get("indirect_syndromes"))
        chpa_reviewed = str(metadata.get("chpa_translation_review_status") or "") in {
            "doctor_reviewed",
            "medical_reviewed",
            "approved",
        }
        chpa_efficacies = _strings(metadata.get("chpa_efficacies")) if chpa_reviewed else []
        chpa_symptoms = _strings(metadata.get("chpa_symptoms")) if chpa_reviewed else []
        aliases = _strings(metadata.get("aliases"))
        catalog_score, reasons = _catalog_match_score(
            search_text,
            herb_name,
            aliases,
            functions,
            symptoms,
            syndromes=syndromes,
            indirect_functions=indirect_functions,
            indirect_symptoms=indirect_symptoms,
            indirect_syndromes=indirect_syndromes,
            chpa_efficacies=chpa_efficacies,
            chpa_symptoms=chpa_symptoms,
        )
        if not catalog_score:
            continue
        item = herbs.setdefault(herb_name, _new_herb_candidate(herb_name))
        item["functions"].extend([*functions, *chpa_efficacies])
        item["symptoms"].extend([*symptoms, *syndromes, *indirect_symptoms, *indirect_syndromes, *chpa_symptoms])
        item["source_ids"].add(entry.source_id)
        item["source_ids"].update(_strings(metadata.get("relation_source_ids")))
        item["risk_labels"].add(metadata.get("evidence_level", "open_catalog"))
        if indirect_syndromes or indirect_symptoms:
            item["risk_labels"].add("indirect_formula_relation")
        if chpa_efficacies or chpa_symptoms:
            item["risk_labels"].add("chpa_statistical_association_pending_review")
        item["match_score"] += catalog_score
        for reason in reasons:
            _append_reason(item, reason)

    formula_suggestions = []
    for formula in db.scalars(select(KnowledgeEntry).where(KnowledgeEntry.entry_type == "formula")).all():
        if formula.source_id == "SRC_DRAGON_TCM":
            continue
        # Numbered disease labels in DragonTCM are source grouping labels, not
        # standard formula names; never expose them as prescription candidates.
        if re.search(r"#\s*\d+\s*$", formula.title):
            continue
        metadata = formula.metadata_json or {}
        components = metadata.get("components") if isinstance(metadata.get("components"), list) else []
        component_names = [str(component.get("normalized_name") or component.get("herb_name") or "") for component in components if isinstance(component, dict)]
        functions = _strings(metadata.get("display_functions")) or _strings(metadata.get("functions"))
        symptoms = _strings(metadata.get("symptoms"))
        syndromes = _strings(metadata.get("syndromes"))
        score, graph_match_score = _formula_match_scores(
            search_text,
            functions=functions,
            symptoms=symptoms,
            syndromes=syndromes,
        )
        # TCM-MKG formulas are only shown when the patient's text directly
        # matches its source-linked treatment method, syndrome or symptom.
        # Herb overlap alone is not a clinically meaningful match.
        # If a source provides syndrome/symptom relations, a broad function
        # word alone must not override conflicting structured indications.
        function_only_catalog = not syndromes and not symptoms
        if score and (graph_match_score > 0 or function_only_catalog):
            formula_suggestions.append({"title": str(metadata.get("display_name") or formula.title), "herbs": component_names, "functions": functions[:6], "symptoms": symptoms[:6], "syndromes": syndromes[:6], "source_ids": metadata.get("source_ids") or [formula.source_id], "status": "candidate_only", "match_score": score, "graph_match_score": graph_match_score, "review_status": metadata.get("review_status", "source_catalog")})

    _add_formula_component_candidates(herbs, formula_suggestions)
    allergy_summary = patient.allergy_history.strip()
    excluded_allergy_herbs = allergy_conflicts_for_herbs(allergy_summary, herbs.keys())
    for herb_name in excluded_allergy_herbs:
        herbs.pop(herb_name, None)
    safe_formulas = []
    excluded_formula_suggestions = []
    for formula in formula_suggestions:
        conflicts = allergy_conflicts_for_herbs(allergy_summary, formula.get("herbs") or [])
        if conflicts:
            excluded_formula_suggestions.append({"title": formula["title"], "matched_allergens": conflicts})
        else:
            safe_formulas.append(formula)
    formula_suggestions = safe_formulas
    ranked_herbs, ranked_formulas, rerank_meta = _rerank_candidates_with_ollama(
        search_text,
        list(herbs.values()),
        formula_suggestions,
    )
    return {
        "patient_id": patient_id,
        "symptoms": search_text,
        "query": search_text,
        "allergy_history": allergy_summary,
        "excluded_allergy_herbs": excluded_allergy_herbs,
        "excluded_formula_suggestions": excluded_formula_suggestions,
        "suggestions": [
            {
                "herb_name": item["herb_name"],
                "functions": sorted(set(item["functions"]))[:6],
                "symptoms": sorted(set(item["symptoms"]))[:6],
                "source_ids": sorted(item["source_ids"]),
                "risk_labels": sorted(item["risk_labels"]),
                "match_reasons": item["match_reasons"][:3],
                "no_dose_or_usage": True,
                "status": "candidate_only",
            }
            for item in ranked_herbs[:12]
        ],
        "formula_suggestions": ranked_formulas[:8],
        "llm_rerank": rerank_meta,
        "herb_llm_rerank": rerank_meta,
        "no_match_message": (
            "未找到与当前症状或证候直接匹配、且可追溯的候选。请补充经医生辨证确认的中医证候后再检索。"
            if not herbs and not ranked_formulas
            else None
        ),
        "requires_human_confirm": True,
        "disclaimer": (
            "仅为内部知识库候选，不包含剂量或用法；已按当前患者过敏史过滤直接冲突的药材和方剂，"
            "但仍必须进入结构化审方、医生人工确认和独立发布流程。"
        ),
    }


def _matches(claim: KnowledgeClaim, query: str) -> bool:
    if not query:
        return True
    tokens = _query_terms(query)
    # Source excerpts frequently contain a whole record and can mention many
    # unrelated symptoms.  Candidate retrieval must use only the drug name
    # and the specific relation value carried by this claim.
    haystack = f"{claim.subject_label} {claim.object_label}".lower()
    return any(token.lower() in haystack for token in tokens)


def _claim_is_eligible(claim: KnowledgeClaim) -> bool:
    """Only doctor-reviewed graph claims may influence prescription candidates."""
    return (
        claim.source_id not in {"SRC_DRAGON_TCM", "SRC_MTCMB", "SRC_SHENNONG_TCM_DATASET"}
        and claim.review_status in {"verified", "approved", "doctor_verified"}
    )


def _query_terms(query: str) -> list[str]:
    """Return literal terms plus a small, reviewable set of TCM retrieval synonyms."""
    terms = [query.strip(), *[token for token in query.replace("，", " ").replace("、", " ").split() if token]]
    expansions = {
        "气虚": ("补气", "益气"),
        "血虚": ("养血", "补血"),
        "失眠": ("安神",),
        "咳嗽": ("止咳",),
        "发热": ("清热",),
        "疼痛": ("止痛",),
        "眼睛疲劳": ("明目",),
        "眼疲劳": ("明目",),
        "视疲劳": ("明目",),
        "肠胃虚弱": ("脾胃虚弱证", "健脾"),
        "胃肠虚弱": ("脾胃虚弱证", "健脾"),
    }
    # A doctor's narrative often contains a direct symptom such as “患者咳嗽”.
    # Extract specific, reviewable TCM terms from it, but do not expand generic
    # “疼痛” inside a longer narrative into every analgesic candidate.
    standalone_terms = set(terms)
    for term, values in expansions.items():
        is_embedded_specific_term = term != "疼痛" and term in query
        if term in standalone_terms or is_embedded_specific_term:
            terms.extend((term, *values))
    return list(dict.fromkeys(term for term in terms if term))


def _values_match(query: str, values: list[str]) -> bool:
    if not query:
        return True
    lowered_values = [value.lower() for value in values if value]
    return any(term.lower() in value for term in _query_terms(query) for value in lowered_values)


def _new_herb_candidate(herb_name: str) -> dict[str, Any]:
    return {
        "herb_name": herb_name,
        "functions": [],
        "symptoms": [],
        "source_ids": set(),
        "risk_labels": set(),
        "match_score": 0,
        "formula_relation_score": 0,
        "match_reasons": [],
    }


def _is_recommendable_herb(label: str) -> bool:
    """Reject generic extraction placeholders even when they contain Han text."""
    return is_valid_tcm_herb_label(label) and label.strip() not in {"药味", "药材", "中药", "药物", "本品", "未知"}


def _append_reason(item: dict[str, Any], reason: str) -> None:
    if reason and reason not in item["match_reasons"]:
        item["match_reasons"].append(reason)


def _claim_match_score(claim: KnowledgeClaim, query: str) -> int:
    predicate_weight = {
        "ASSOCIATED_WITH_TCM_SYNDROME": 12,
        "ASSOCIATED_WITH_TCM_SYMPTOM": 10,
        "ASSOCIATED_WITH": 8,
        "REFERENCES_SYMPTOM": 8,
        "HAS_TCM_FUNCTION": 4,
        "HAS_EFFICACY": 4,
        "HAS_FUNCTION": 4,
    }.get(claim.predicate, 2)
    return predicate_weight + _term_match_score(query, [claim.object_label])


def _claim_reason(claim: KnowledgeClaim) -> str:
    labels = {
        "ASSOCIATED_WITH_TCM_SYNDROME": "证候关联",
        "ASSOCIATED_WITH_TCM_SYMPTOM": "症状关联",
        "ASSOCIATED_WITH": "症状关联",
        "REFERENCES_SYMPTOM": "症状关联",
        "HAS_TCM_FUNCTION": "功效关联",
        "HAS_EFFICACY": "功效关联",
        "HAS_FUNCTION": "功效关联",
    }
    return f"{labels.get(claim.predicate, '知识关系')}：{claim.object_label}"


def _catalog_match_score(
    query: str,
    herb_name: str,
    aliases: list[str],
    functions: list[str],
    symptoms: list[str],
    *,
    syndromes: list[str] | None = None,
    indirect_functions: list[str] | None = None,
    indirect_symptoms: list[str] | None = None,
    indirect_syndromes: list[str] | None = None,
    chpa_efficacies: list[str] | None = None,
    chpa_symptoms: list[str] | None = None,
) -> tuple[int, list[str]]:
    terms = _query_terms(query)
    syndromes = syndromes or []
    indirect_functions = indirect_functions or []
    indirect_symptoms = indirect_symptoms or []
    indirect_syndromes = indirect_syndromes or []
    chpa_efficacies = chpa_efficacies or []
    chpa_symptoms = chpa_symptoms or []
    score = 0
    reasons: list[str] = []
    for term in terms:
        if term.lower() in herb_name.lower() or any(term.lower() in alias.lower() for alias in aliases):
            score += 14
            reasons.append(f"名称/别名匹配：{term}")
        for value in symptoms:
            if term.lower() in value.lower():
                score += 10
                reasons.append(f"症状关联：{value}")
        for value in syndromes:
            if term.lower() in value.lower():
                score += 12
                reasons.append(f"证候关联：{value}")
        for value in functions:
            if term.lower() in value.lower():
                score += 6
                reasons.append(f"功效关联：{value}")
        for value in indirect_syndromes:
            if term.lower() in value.lower():
                score += 5
                reasons.append(f"经方剂关联证候：{value}")
        for value in indirect_symptoms:
            if term.lower() in value.lower():
                score += 4
                reasons.append(f"经方剂关联症状：{value}")
        for value in indirect_functions:
            if term.lower() in value.lower():
                score += 2
                reasons.append(f"关联方剂功效：{value}")
        for value in chpa_symptoms:
            if term.lower() in value.lower():
                score += 6
                reasons.append(f"CHPA统计症状关联（待复核）：{value}")
        for value in chpa_efficacies:
            if term.lower() in value.lower():
                score += 3
                reasons.append(f"CHPA统计功效关联（待复核）：{value}")
    return score, list(dict.fromkeys(reasons))


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def _term_match_score(query: str, terms: list[str]) -> int:
    query = query.strip()
    if not query:
        return 0
    score = 0
    query_terms = _query_terms(query)
    for term in terms:
        if len(term) >= 2 and any(value in term for value in query_terms):
            score += 3
        for token in query_terms:
            if token in term:
                score += 1
    return score


def _formula_match_scores(query: str, *, functions: list[str], symptoms: list[str], syndromes: list[str]) -> tuple[int, int]:
    """Prefer direct syndrome and symptom relations over broad efficacy words."""
    graph_score = (
        _term_match_score(query, syndromes) * 5
        + _term_match_score(query, symptoms) * 4
    )
    return graph_score + _term_match_score(query, functions), graph_score


def _add_formula_component_candidates(herbs: dict[str, dict[str, Any]], formulas: list[dict[str, Any]]) -> None:
    """Traverse matched formula→component relations to enrich herb candidates."""
    graph_formulas = [formula for formula in formulas if formula.get("graph_match_score", 0) > 0]
    for formula in sorted(graph_formulas, key=lambda item: (-item["graph_match_score"], item["title"]))[:12]:
        relation_score = max(2, min(5, int(formula["graph_match_score"]) // 4))
        syndrome_text = "、".join(formula["syndromes"]) or "未关联证候"
        for herb_name in formula["herbs"]:
            if not _is_recommendable_herb(herb_name):
                continue
            item = herbs.setdefault(herb_name, _new_herb_candidate(herb_name))
            if relation_score > item["formula_relation_score"]:
                item["match_score"] += relation_score - item["formula_relation_score"]
                item["formula_relation_score"] = relation_score
            item["source_ids"].update(str(value) for value in formula["source_ids"] if value)
            item["risk_labels"].add("formula_component_relation")
            _append_reason(item, f"关联方剂：{formula['title']}（{syndrome_text}）")


def _rerank_candidates_with_ollama(
    query: str,
    herb_candidates: list[dict[str, Any]],
    formula_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ordered_herbs = sorted(herb_candidates, key=lambda item: (-item["match_score"], item["herb_name"]))
    ordered_formulas = sorted(
        formula_candidates,
        key=lambda item: (-item["match_score"], -len(item["herbs"]), item["title"]),
    )
    settings = get_settings()
    if settings.llm_provider != "ollama" or not (ordered_herbs or ordered_formulas):
        return ordered_herbs, ordered_formulas, {"used": False, "provider": "rule_match"}

    herb_shortlist = ordered_herbs[:12]
    formula_shortlist = ordered_formulas[:12]
    herb_payload = [
        {
            "id": str(index),
            "药材": item["herb_name"],
            "功效": sorted(set(item["functions"]))[:6],
            "症状或证候": sorted(set(item["symptoms"]))[:6],
            "命中依据": item["match_reasons"][:3],
        }
        for index, item in enumerate(herb_shortlist)
    ]
    formula_payload = [
        {
            "id": str(index),
            "方剂": item["title"],
            "功效": item["functions"],
            "证候": item["syndromes"],
            "症状": item["symptoms"],
        }
        for index, item in enumerate(formula_shortlist)
    ]
    prompt = (
        "你是医生端候选排序助手。仅依据给出的结构化候选字段和患者文本排序，"
        "不得新增、删除或修改任何药材、方剂、功效、症状或证候。"
        "优先直接证候和症状关联，其次功效关联。"
        "只输出JSON：{\"herb_ids\":[药材候选id...],\"formula_ids\":[方剂候选id...]}。\n"
        f"患者文本：{query}\n"
        f"药材候选：{json.dumps(herb_payload, ensure_ascii=False)}\n"
        f"方剂候选：{json.dumps(formula_payload, ensure_ascii=False)}"
    )
    try:
        request = urllib.request.Request(
            settings.ollama_base_url.rstrip("/") + "/api/generate",
            data=json.dumps(
                {
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=settings.ollama_generate_timeout_seconds) as response:
            result = json.loads(json.loads(response.read().decode())["response"])
        ranked_herbs = _rank_from_ids(result.get("herb_ids"), herb_shortlist, ordered_herbs)
        ranked_formulas = _rank_from_ids(result.get("formula_ids"), formula_shortlist, ordered_formulas)
        return ranked_herbs, ranked_formulas, {"used": True, "provider": "ollama", "model": settings.ollama_model}
    except Exception:
        return ordered_herbs, ordered_formulas, {"used": False, "provider": "rule_match", "fallback_used": True}


def _rank_from_ids(raw_ids: Any, shortlist: list[dict[str, Any]], ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = raw_ids if isinstance(raw_ids, list) else []
    ids = [int(value) for value in values if str(value).isdigit() and int(value) < len(shortlist)]
    used = set(ids)
    return (
        [shortlist[index] for index in dict.fromkeys(ids)]
        + [item for index, item in enumerate(shortlist) if index not in used]
        + ordered[len(shortlist) :]
    )
