"""Deterministic seed knowledge for the synthetic CARE-OPD benchmark.

The assertions below deliberately define a *closed simulation*.  They are not
presented as an authoritative materia-medica database.  A competition or
clinical-facing deployment must replace this seed with licensed, versioned and
expert-reviewed evidence while retaining the same schema and data lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Dict, Mapping, Sequence, Tuple

from .schemas import Decision, Perturbation, RiskFamily, Split, stable_hash


RESEARCH_ONLY_NOTICE = (
    "研究用途合成数据：本项目只评估智能体在工具失败后的证据恢复能力，"
    "不提供诊断、处方或治疗建议；全部规则必须在真实使用前由持证专家复核。"
)


@dataclass(frozen=True)
class RuleSeed:
    rule_id: str
    family: RiskFamily
    trigger_items: Tuple[str, ...]
    aliases: Tuple[str, ...]
    rule_fact: str
    risk_decision: Decision
    patient_fact_kind: str
    guideline_query_terms: Tuple[str, ...]
    severity: float


# These are benchmark assertions, phrased explicitly as simulation rules.
RULES: Tuple[RuleSeed, ...] = (
    RuleSeed(
        "SIM-SPECIAL-TAOREN",
        RiskFamily.SPECIAL_STATE,
        ("桃仁",),
        ("桃核仁", "桃仁饮片"),
        "仿真规则：若特殊生理状态为阳性，含桃仁方案需阻断并人工复核。",
        Decision.BLOCK,
        "special_state",
        ("桃仁", "特殊生理状态", "仿真规则"),
        1.00,
    ),
    RuleSeed(
        "SIM-SPECIAL-HONGHUA",
        RiskFamily.SPECIAL_STATE,
        ("红花",),
        ("草红花", "红蓝花"),
        "仿真规则：若特殊生理状态为阳性，含红花方案需阻断并人工复核。",
        Decision.BLOCK,
        "special_state",
        ("红花", "特殊生理状态", "仿真规则"),
        1.00,
    ),
    RuleSeed(
        "SIM-PAIR-FUZI-BANXIA",
        RiskFamily.INCOMPATIBLE_PAIR,
        ("附子", "半夏"),
        ("制附子", "法半夏"),
        "仿真规则：附子与半夏组合被标记为不相容，需阻断并人工复核。",
        Decision.BLOCK,
        "prescription_pair",
        ("附子", "半夏", "不相容"),
        0.95,
    ),
    RuleSeed(
        "SIM-PAIR-GANCAO-HAIZAO",
        RiskFamily.INCOMPATIBLE_PAIR,
        ("甘草", "海藻"),
        ("炙甘草", "海藻饮片"),
        "仿真规则：甘草与海藻组合被标记为不相容，需阻断并人工复核。",
        Decision.BLOCK,
        "prescription_pair",
        ("甘草", "海藻", "不相容"),
        0.95,
    ),
    RuleSeed(
        "SIM-ALLERGY-DANGGUI",
        RiskFamily.ALLERGY,
        ("当归",),
        ("秦归", "当归饮片"),
        "仿真规则：若患者过敏记录命中当归，本次方案需阻断。",
        Decision.BLOCK,
        "allergy",
        ("当归", "过敏", "仿真规则"),
        1.00,
    ),
    RuleSeed(
        "SIM-ALLERGY-HUANGQI",
        RiskFamily.ALLERGY,
        ("黄芪",),
        ("绵黄芪", "黄耆"),
        "仿真规则：若患者过敏记录命中黄芪，本次方案需阻断。",
        Decision.BLOCK,
        "allergy",
        ("黄芪", "过敏", "仿真规则"),
        1.00,
    ),
    RuleSeed(
        "SIM-MED-DANSHEN",
        RiskFamily.CONCURRENT_MEDICATION,
        ("丹参",),
        ("紫丹参", "丹参饮片", "合并用药A"),
        "仿真规则：丹参与仿真药物类别 RX-A 同用时需进入人工复核。",
        Decision.REVIEW,
        "concurrent_medication",
        ("丹参", "RX-A", "同用", "仿真规则"),
        0.85,
    ),
    RuleSeed(
        "SIM-MED-TAOREN",
        RiskFamily.CONCURRENT_MEDICATION,
        ("桃仁",),
        ("桃核仁", "桃仁饮片", "合并用药B"),
        "仿真规则：桃仁与仿真药物类别 RX-B 同用时需进入人工复核。",
        Decision.REVIEW,
        "concurrent_medication",
        ("桃仁", "RX-B", "同用", "仿真规则"),
        0.85,
    ),
)


CLEAR_PRESCRIPTIONS: Tuple[Tuple[str, ...], ...] = (
    ("白术", "茯苓"),
    ("陈皮", "茯苓"),
    ("黄芪", "白术"),
    ("当归", "白芍"),
    ("麦冬", "五味子"),
    ("山药", "茯苓"),
)


FILLER_ITEMS: Tuple[str, ...] = (
    "白术",
    "茯苓",
    "陈皮",
    "白芍",
    "麦冬",
    "五味子",
    "山药",
    "炙甘草",
)


AGE_BANDS: Tuple[str, ...] = ("18-30", "31-45", "46-60", "61-75")
CHIEF_CONCERNS: Tuple[str, ...] = (
    "合成复诊：睡眠变化",
    "合成复诊：食欲变化",
    "合成复诊：疲劳变化",
    "合成复诊：腹部不适变化",
    "合成复诊：症状总体稳定",
)


# Exact risk-family × perturbation pairs are held apart.  The primitive family
# and primitive perturbation generally occur in train, while their composition
# is novel in dev/test.  This supports a real compositional generalization test
# without pretending that surface-random record splits are OOD.
COMPOSITION_MATRIX: Mapping[Split, Tuple[Tuple[RiskFamily, Perturbation], ...]] = {
    Split.TRAIN: (
        (RiskFamily.SPECIAL_STATE, Perturbation.CLEAN),
        (RiskFamily.SPECIAL_STATE, Perturbation.SOURCE_OUTAGE),
        (RiskFamily.SPECIAL_STATE, Perturbation.SYNONYM_SHIFT),
        (RiskFamily.INCOMPATIBLE_PAIR, Perturbation.CLEAN),
        (RiskFamily.INCOMPATIBLE_PAIR, Perturbation.STALE_RECORD),
        (RiskFamily.INCOMPATIBLE_PAIR, Perturbation.DISTRACTOR_DENSE),
        (RiskFamily.ALLERGY, Perturbation.CLEAN),
        (RiskFamily.ALLERGY, Perturbation.ROUTE_SWAP),
        (RiskFamily.ALLERGY, Perturbation.PREREQUISITE),
        (RiskFamily.CONCURRENT_MEDICATION, Perturbation.CLEAN),
        (RiskFamily.CONCURRENT_MEDICATION, Perturbation.SOURCE_OUTAGE),
        (RiskFamily.CONCURRENT_MEDICATION, Perturbation.SYNONYM_SHIFT),
        (RiskFamily.CLEAR_CONTROL, Perturbation.CLEAN),
        (RiskFamily.CLEAR_CONTROL, Perturbation.DISTRACTOR_DENSE),
    ),
    Split.DEV: (
        (RiskFamily.SPECIAL_STATE, Perturbation.CONFLICTING_RECORD),
        (RiskFamily.SPECIAL_STATE, Perturbation.ROUTE_SWAP),
        (RiskFamily.INCOMPATIBLE_PAIR, Perturbation.SOURCE_OUTAGE),
        (RiskFamily.INCOMPATIBLE_PAIR, Perturbation.SYNONYM_SHIFT),
        (RiskFamily.ALLERGY, Perturbation.STALE_RECORD),
        (RiskFamily.ALLERGY, Perturbation.DISTRACTOR_DENSE),
        (RiskFamily.CONCURRENT_MEDICATION, Perturbation.PREREQUISITE),
        (RiskFamily.CONCURRENT_MEDICATION, Perturbation.CONFLICTING_RECORD),
        (RiskFamily.CLEAR_CONTROL, Perturbation.ROUTE_SWAP),
        (RiskFamily.CLEAR_CONTROL, Perturbation.SYNONYM_SHIFT),
    ),
    Split.TEST: (
        (RiskFamily.SPECIAL_STATE, Perturbation.PREREQUISITE),
        (RiskFamily.SPECIAL_STATE, Perturbation.DISTRACTOR_DENSE),
        (RiskFamily.INCOMPATIBLE_PAIR, Perturbation.ROUTE_SWAP),
        (RiskFamily.INCOMPATIBLE_PAIR, Perturbation.CONFLICTING_RECORD),
        (RiskFamily.ALLERGY, Perturbation.SOURCE_OUTAGE),
        (RiskFamily.ALLERGY, Perturbation.SYNONYM_SHIFT),
        (RiskFamily.CONCURRENT_MEDICATION, Perturbation.STALE_RECORD),
        (RiskFamily.CONCURRENT_MEDICATION, Perturbation.ROUTE_SWAP),
        (RiskFamily.CLEAR_CONTROL, Perturbation.SOURCE_OUTAGE),
        (RiskFamily.CLEAR_CONTROL, Perturbation.PREREQUISITE),
        (RiskFamily.MULTI_RISK, Perturbation.CONFLICTING_RECORD),
        (RiskFamily.MULTI_RISK, Perturbation.DISTRACTOR_DENSE),
    ),
}


ROUTE_LAYOUTS: Mapping[Split, Tuple[str, ...]] = {
    Split.TRAIN: ("canonical", "patient_memory_first"),
    Split.DEV: ("timeline_first", "duplicated_evidence"),
    Split.TEST: ("swapped_patient_sources", "doctor_fallback"),
}

QUERY_STYLES: Mapping[Split, Tuple[str, ...]] = {
    Split.TRAIN: ("canonical", "short"),
    Split.DEV: ("alias", "temporal"),
    Split.TEST: ("paraphrase", "entity_dense"),
}


PATIENT_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "special_state": ("特殊状态", "特殊生理情况", "状态字段"),
    "allergy": ("过敏", "不耐受记录", "既往反应"),
    "concurrent_medication": ("合并用药", "当前药物", "并用药物"),
}


def rule_by_id(rule_id: str) -> RuleSeed:
    for rule in RULES:
        if rule.rule_id == rule_id:
            return rule
    raise KeyError(rule_id)


def rules_for_family(family: RiskFamily) -> Tuple[RuleSeed, ...]:
    return tuple(rule for rule in RULES if rule.family == family)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return text


def term_present(query: str, term: str, aliases: Sequence[str] = ()) -> bool:
    """Return whether ``term`` or one of its alternatives occurs in ``query``.

    This helper is intentionally literal.  The simulator is not a clinical
    entity linker; it only provides deterministic closed-world matching.
    """

    normalized = normalize_text(query)
    return any(
        normalize_text(candidate) in normalized
        for candidate in (term, *aliases)
        if candidate
    )


def semantic_query_match(
    query: str,
    required_terms: Sequence[str],
    aliases: Sequence[str] = (),
    minimum_fraction: float = 0.66,
) -> bool:
    """Closed-world query matcher used only by the benchmark simulator."""

    if not query.strip() or not required_terms:
        return False
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must be in (0, 1]")

    # Aliases describe alternatives for one semantic slot, not magic tokens
    # that satisfy every required term.  Counting the same alias once for each
    # missing required term made a query such as "特殊状态" appear to contain a
    # patient identifier, a time qualifier, and a field name simultaneously.
    exact_hits = sum(term_present(query, term) for term in required_terms)
    alias_hit = any(term_present(query, alias) for alias in aliases)
    hits = min(len(required_terms), exact_hits + int(alias_hit))
    needed = max(1, int(len(required_terms) * minimum_fraction + 0.999))
    return hits >= needed


def seed_snapshot() -> Dict[str, object]:
    value: Dict[str, object] = {
        "notice": RESEARCH_ONLY_NOTICE,
        "rules": [rule.__dict__ for rule in RULES],
        "clear_prescriptions": CLEAR_PRESCRIPTIONS,
        "composition_matrix": {
            split.value: [(a.value, b.value) for a, b in pairs]
            for split, pairs in COMPOSITION_MATRIX.items()
        },
        "route_layouts": {k.value: v for k, v in ROUTE_LAYOUTS.items()},
        "query_styles": {k.value: v for k, v in QUERY_STYLES.items()},
    }
    value["seed_hash"] = stable_hash(value, 32)
    return value
