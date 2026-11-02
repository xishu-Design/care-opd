"""Formal deterministic data synthesis for CARE-OPD.

The generator creates:

* grouped train/dev/test cases with held-out family×perturbation compositions;
* oracle evidence-recovery trajectories;
* rejected-action checkpoints with causal minimal-repair certificates;
* a manifest containing hashes, lineage and split-integrity statistics.

Example
-------
PYTHONPATH=src python -m care_opd.synthesis --output data/formal \
  --train 12000 --dev 1500 --test 3000 --recoveries-per-case 3
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .environment import CareRecoveryEnv
from .knowledge import (
    AGE_BANDS,
    CHIEF_CONCERNS,
    CLEAR_PRESCRIPTIONS,
    COMPOSITION_MATRIX,
    FILLER_ITEMS,
    PATIENT_ALIASES,
    QUERY_STYLES,
    RESEARCH_ONLY_NOTICE,
    RULES,
    ROUTE_LAYOUTS,
    RuleSeed,
    rule_by_id,
    rules_for_family,
    seed_snapshot,
)
from .schemas import (
    Action,
    ActionKind,
    CaseSpec,
    Decision,
    Evidence,
    EvidenceSource,
    HiddenWorld,
    ObservationStatus,
    Perturbation,
    PrescriptionItem,
    PublicCase,
    RiskFamily,
    Split,
    canonical_json,
    jsonl_dump,
    stable_hash,
    to_jsonable,
)


DATA_FORMAT_VERSION = "care-opd-data-v2"
GENERATOR_VERSION = "care-opd-synth-v2"


@dataclass(frozen=True)
class SynthesisConfig:
    seed: int = 20260728
    train: int = 12000
    dev: int = 1500
    test: int = 3000
    recoveries_per_case: int = 3
    max_steps: int = 6
    overwrite: bool = False
    knowledge: Optional[str] = None

    def counts(self) -> Mapping[Split, int]:
        return {
            Split.TRAIN: self.train,
            Split.DEV: self.dev,
            Split.TEST: self.test,
        }


def _validate_synthesis_config(config: SynthesisConfig) -> None:
    if min(config.train, config.dev, config.test) < 1:
        raise ValueError("all split counts must be positive")
    if not 1 <= config.recoveries_per_case <= 8:
        raise ValueError("recoveries_per_case must be in [1, 8]")
    # The longest clean oracle path in the fixed simulator is four actions.
    # One additional slot is required to make a single injected failure
    # recoverable rather than turning it into deterministic budget exhaustion.
    if config.max_steps < 5:
        raise ValueError(
            "max_steps must be at least 5 for one-failure recovery"
        )


def _rng(seed: int, *parts: object) -> random.Random:
    digest = hashlib.sha256(
        "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _eid(case_token: str, label: str) -> str:
    return "ev_" + stable_hash({"case": case_token, "label": label}, 14)


def _rule_for_index(family: RiskFamily, index: int) -> Optional[RuleSeed]:
    if family in {RiskFamily.CLEAR_CONTROL, RiskFamily.MULTI_RISK}:
        return None
    candidates = rules_for_family(family)
    if not candidates:
        raise ValueError(f"no seed rule for {family.value}")
    return candidates[index % len(candidates)]


def _dose(rng: random.Random) -> float:
    return float(rng.choice((3, 4.5, 6, 9, 10, 12)))


def _prescription_for(
    family: RiskFamily,
    rule: Optional[RuleSeed],
    unsafe_polarity: bool,
    perturbation: Perturbation,
    rng: random.Random,
) -> Tuple[PrescriptionItem, ...]:
    if family == RiskFamily.CLEAR_CONTROL:
        names = list(rng.choice(CLEAR_PRESCRIPTIONS))
    elif family == RiskFamily.MULTI_RISK:
        names = ["桃仁", "丹参", rng.choice(FILLER_ITEMS)]
    elif rule is None:
        raise AssertionError("non-clear family needs a rule")
    elif family == RiskFamily.INCOMPATIBLE_PAIR and not unsafe_polarity:
        names = [rule.trigger_items[0], rng.choice(FILLER_ITEMS)]
    else:
        names = list(rule.trigger_items)
        if len(names) == 1:
            filler = rng.choice([x for x in FILLER_ITEMS if x not in names])
            names.append(filler)
    if perturbation == Perturbation.SYNONYM_SHIFT:
        surface_aliases: Dict[str, str] = {}
        for seed_rule in RULES:
            for item, alias in zip(
                seed_rule.trigger_items,
                seed_rule.aliases,
            ):
                surface_aliases.setdefault(item, alias)
        names = [surface_aliases.get(name, name) for name in names]
    return tuple(PrescriptionItem(name=name, dose_g=_dose(rng)) for name in names)


def _patient_fact(
    family: RiskFamily, unsafe_polarity: bool, rule: Optional[RuleSeed]
) -> Tuple[str, str, Decision]:
    if family == RiskFamily.SPECIAL_STATE:
        value = "positive" if unsafe_polarity else "negative"
        return f"special_state={value}", value, (
            Decision.BLOCK if unsafe_polarity else Decision.CLEAR
        )
    if family == RiskFamily.ALLERGY:
        item = rule.trigger_items[0] if rule else "unknown"
        value = item if unsafe_polarity else "none"
        return f"allergy={value}", value, (
            Decision.BLOCK if unsafe_polarity else Decision.CLEAR
        )
    if family == RiskFamily.CONCURRENT_MEDICATION:
        rx = "RX-A" if rule and "DANSHEN" in rule.rule_id else "RX-B"
        value = rx if unsafe_polarity else "none"
        return f"concurrent_medication={value}", value, (
            Decision.REVIEW if unsafe_polarity else Decision.CLEAR
        )
    if family == RiskFamily.INCOMPATIBLE_PAIR:
        value = "present" if unsafe_polarity else "absent"
        return f"prescription_pair={value}", value, (
            Decision.BLOCK if unsafe_polarity else Decision.CLEAR
        )
    if family == RiskFamily.MULTI_RISK:
        return "special_state=positive;concurrent_medication=RX-A", "multi", Decision.BLOCK
    return "screening=negative", "negative", Decision.CLEAR


def _patient_fact_key(family: RiskFamily, case_token: str) -> Optional[str]:
    if family == RiskFamily.SPECIAL_STATE:
        return f"patient:{case_token}:special_state"
    if family == RiskFamily.ALLERGY:
        return f"patient:{case_token}:allergy"
    if family == RiskFamily.CONCURRENT_MEDICATION:
        return f"patient:{case_token}:concurrent_medication"
    if family == RiskFamily.MULTI_RISK:
        return f"patient:{case_token}:special_state"
    if family == RiskFamily.CLEAR_CONTROL:
        return f"patient:{case_token}:screening"
    return None


def _canonical_patient_source(family: RiskFamily) -> EvidenceSource:
    if family == RiskFamily.ALLERGY:
        return EvidenceSource.MEM0
    return EvidenceSource.GRAPHITI


def _other_patient_source(source: EvidenceSource) -> EvidenceSource:
    return (
        EvidenceSource.GRAPHITI
        if source == EvidenceSource.MEM0
        else EvidenceSource.MEM0
    )


def _append(
    store: Dict[str, List[Evidence]], evidence: Evidence
) -> None:
    store[evidence.source.value].append(evidence)


def _make_evidence_store(
    *,
    case_token: str,
    patient_id: str,
    visit_time: int,
    family: RiskFamily,
    perturbation: Perturbation,
    rule: Optional[RuleSeed],
    patient_value: str,
    route_layout: str,
    rng: random.Random,
) -> Tuple[
    Mapping[str, Tuple[Evidence, ...]],
    Tuple[str, ...],
    Mapping[str, str],
    Tuple[str, ...],
    str,
]:
    store: Dict[str, List[Evidence]] = {
        source.value: [] for source in EvidenceSource
    }
    unavailable: List[str] = []
    prerequisites: Dict[str, str] = {}
    required_keys: List[str] = []

    rule_id = rule.rule_id if rule else "SIM-CLEAR-SCREEN"
    rule_fact = (
        rule.rule_fact
        if rule
        else "仿真规则：完成当前状态筛查且未命中种子风险时，可标记 clear。"
    )
    rule_key = f"rule:{rule_id}"
    required_keys.append(rule_key)
    rule_terms = (
        rule.guideline_query_terms
        if rule
        else ("当前状态筛查", "未命中风险", "仿真规则")
    )
    rule_aliases = rule.aliases if rule else ("安全筛查", "阴性筛查")
    rule_source = EvidenceSource.LIGHTRAG
    if perturbation == Perturbation.ROUTE_SWAP and family in {
        RiskFamily.INCOMPATIBLE_PAIR,
        RiskFamily.CLEAR_CONTROL,
    }:
        rule_source = EvidenceSource.GRAPHITI
    _append(
        store,
        Evidence(
            evidence_id=_eid(case_token, "rule-active"),
            source=rule_source,
            content=rule_fact,
            fact_key=rule_key,
            fact_value="active",
            timestamp=visit_time - 100,
            valid_from=visit_time - 1000,
            valid_until=visit_time + 1000,
            reliability=0.95,
            required_terms=tuple(rule_terms),
            aliases=tuple(rule_aliases),
        ),
    )

    patient_key = _patient_fact_key(family, case_token)
    if patient_key is not None:
        required_keys.insert(0, patient_key)
        primary = _canonical_patient_source(family)
        if perturbation == Perturbation.ROUTE_SWAP:
            primary = _other_patient_source(primary)
        backup = _other_patient_source(primary)
        label = patient_key.rsplit(":", 1)[-1]
        base_terms = (patient_id, label, "当前")
        aliases = PATIENT_ALIASES.get(
            label,
            ("当前筛查", "阴性筛查"),
        )
        active = Evidence(
            evidence_id=_eid(case_token, "patient-active"),
            source=primary,
            content=f"合成患者 {patient_id} 当前 {label}={patient_value}。",
            fact_key=patient_key,
            fact_value=patient_value,
            timestamp=visit_time - 1,
            valid_from=visit_time - 30,
            valid_until=visit_time + 30,
            reliability=0.90,
            required_terms=base_terms,
            aliases=aliases,
        )
        backup_evidence = Evidence(
            evidence_id=_eid(case_token, "patient-backup"),
            source=backup,
            content=f"合成患者 {patient_id} 复核记录 {label}={patient_value}。",
            fact_key=patient_key,
            fact_value=patient_value,
            timestamp=visit_time - 2,
            valid_from=visit_time - 30,
            valid_until=visit_time + 30,
            reliability=0.86,
            required_terms=(patient_id, label, "复核"),
            aliases=aliases,
        )

        if perturbation == Perturbation.STALE_RECORD:
            stale = Evidence(
                **{
                    **active.__dict__,
                    "evidence_id": _eid(case_token, "patient-stale"),
                    "content": f"合成患者 {patient_id} 旧记录 {label}=opposite。",
                    "fact_value": "opposite",
                    "timestamp": visit_time - 500,
                    "valid_from": visit_time - 1000,
                    "valid_until": visit_time - 300,
                    "stale": True,
                }
            )
            _append(store, stale)
            _append(store, backup_evidence)
        elif perturbation == Perturbation.CONFLICTING_RECORD:
            conflict_a = Evidence(
                **{
                    **active.__dict__,
                    "evidence_id": _eid(case_token, "conflict-a"),
                    "fact_value": patient_value,
                    "reliability": 0.80,
                    "contradictory_to": _eid(case_token, "conflict-b"),
                }
            )
            opposite = "negative" if patient_value != "negative" else "positive"
            conflict_b = Evidence(
                **{
                    **active.__dict__,
                    "evidence_id": _eid(case_token, "conflict-b"),
                    "content": f"合成患者 {patient_id} 同期冲突记录 {label}={opposite}。",
                    "fact_value": opposite,
                    "reliability": 0.80,
                    "contradictory_to": _eid(case_token, "conflict-a"),
                }
            )
            doctor_current = Evidence(
                evidence_id=_eid(case_token, "doctor-resolution"),
                source=EvidenceSource.DOCTOR,
                content=f"人工确认：{patient_id} 当前 {label}={patient_value}。",
                fact_key=patient_key,
                fact_value=patient_value,
                timestamp=visit_time,
                valid_from=visit_time,
                valid_until=visit_time + 30,
                reliability=1.0,
                required_terms=(patient_id, label, "确认"),
                aliases=aliases,
            )
            _append(store, conflict_a)
            _append(store, conflict_b)
            _append(store, doctor_current)
        else:
            _append(store, active)
            _append(store, backup_evidence)

        if perturbation == Perturbation.SOURCE_OUTAGE:
            unavailable.append(primary.value)
        if perturbation == Perturbation.PREREQUISITE:
            prerequisite_key = f"auth:{case_token}"
            prerequisites[primary.value] = prerequisite_key
            _append(
                store,
                Evidence(
                    evidence_id=_eid(case_token, "doctor-auth"),
                    source=EvidenceSource.DOCTOR,
                    content=f"人工确认：允许查询合成患者 {patient_id} 当前记录。",
                    fact_key=prerequisite_key,
                    fact_value="complete",
                    timestamp=visit_time,
                    valid_from=visit_time,
                    valid_until=visit_time + 30,
                    reliability=1.0,
                    required_terms=(patient_id, "前置授权", "确认"),
                ),
            )

    if patient_key is None and perturbation == Perturbation.STALE_RECORD:
        base_rule = store[rule_source.value].pop(0)
        stale_rule = Evidence(
            **{
                **base_rule.__dict__,
                "evidence_id": _eid(case_token, "rule-stale"),
                "content": (
                    f"过期仿真规则快照 {rule_id}；不得用于当前判定。"
                ),
                "fact_value": "obsolete",
                "timestamp": visit_time - 500,
                "valid_from": visit_time - 1000,
                "valid_until": visit_time - 300,
                "stale": True,
            }
        )
        backup_source = (
            EvidenceSource.MEM0
            if rule_source != EvidenceSource.MEM0
            else EvidenceSource.LIGHTRAG
        )
        active_backup = Evidence(
            **{
                **base_rule.__dict__,
                "evidence_id": _eid(case_token, "rule-current-backup"),
                "source": backup_source,
                "reliability": 0.90,
            }
        )
        _append(store, stale_rule)
        _append(store, active_backup)

    if patient_key is None and perturbation == Perturbation.CONFLICTING_RECORD:
        base_rule = store[rule_source.value].pop(0)
        conflict_a_id = _eid(case_token, "rule-conflict-a")
        conflict_b_id = _eid(case_token, "rule-conflict-b")
        conflict_a = Evidence(
            **{
                **base_rule.__dict__,
                "evidence_id": conflict_a_id,
                "reliability": 0.80,
                "contradictory_to": conflict_b_id,
            }
        )
        conflict_b = Evidence(
            **{
                **base_rule.__dict__,
                "evidence_id": conflict_b_id,
                "content": (
                    f"同期冲突的合成规则版本：{rule_id} 被标记为 inactive。"
                ),
                "fact_value": "inactive",
                "reliability": 0.80,
                "contradictory_to": conflict_a_id,
            }
        )
        doctor_resolution = Evidence(
            evidence_id=_eid(case_token, "rule-doctor-resolution"),
            source=EvidenceSource.DOCTOR,
            content=(
                f"人工确认：合成规则 {rule_id} 的当前仿真状态为 active。"
            ),
            fact_key=rule_key,
            fact_value="active",
            timestamp=visit_time,
            valid_from=visit_time,
            valid_until=visit_time + 30,
            reliability=1.0,
            required_terms=(patient_id, "规则复核", "确认"),
        )
        _append(store, conflict_a)
        _append(store, conflict_b)
        _append(store, doctor_resolution)

    if perturbation == Perturbation.SOURCE_OUTAGE and patient_key is None:
        unavailable.append(rule_source.value)
        # An independently indexed backup keeps the task recoverable.
        base_rule = store[rule_source.value][0]
        _append(
            store,
            Evidence(
                **{
                    **base_rule.__dict__,
                    "evidence_id": _eid(case_token, "rule-backup"),
                    "source": EvidenceSource.MEM0,
                    "reliability": 0.88,
                }
            ),
        )

    if perturbation == Perturbation.PREREQUISITE and patient_key is None:
        prerequisite_key = f"auth:{case_token}"
        prerequisites[rule_source.value] = prerequisite_key
        _append(
            store,
            Evidence(
                evidence_id=_eid(case_token, "doctor-auth"),
                source=EvidenceSource.DOCTOR,
                content=f"人工确认：允许读取合成规则 {rule_id}。",
                fact_key=prerequisite_key,
                fact_value="complete",
                timestamp=visit_time,
                valid_from=visit_time,
                valid_until=visit_time + 30,
                reliability=1.0,
                required_terms=(patient_id, "规则读取", "确认"),
            ),
        )

    if perturbation == Perturbation.DISTRACTOR_DENSE:
        for idx in range(5):
            source = rng.choice(tuple(EvidenceSource))
            _append(
                store,
                Evidence(
                    evidence_id=_eid(case_token, f"distractor-{idx}"),
                    source=source,
                    content=f"无关合成患者 P-X{idx} 的历史噪声记录。",
                    fact_key=f"distractor:{idx}",
                    fact_value="noise",
                    timestamp=visit_time - rng.randint(50, 500),
                    valid_from=visit_time - 1000,
                    valid_until=visit_time + 1000,
                    reliability=0.5,
                    required_terms=("历史", "记录", str(idx)),
                    is_distractor=True,
                ),
            )

    frozen_store = {
        source: tuple(sorted(entries, key=lambda e: e.evidence_id))
        for source, entries in store.items()
    }
    return (
        frozen_store,
        tuple(sorted(set(unavailable))),
        prerequisites,
        tuple(required_keys),
        rule_fact,
    )


def make_case(
    split: Split,
    index: int,
    global_seed: int,
    max_steps: int = 6,
) -> CaseSpec:
    pairs = COMPOSITION_MATRIX[split]
    pair_index = index // 2
    family, perturbation = pairs[pair_index % len(pairs)]
    rng = _rng(global_seed, split.value, index)
    unsafe = index % 2 == 0
    if family in {RiskFamily.CLEAR_CONTROL, RiskFamily.MULTI_RISK}:
        unsafe = family == RiskFamily.MULTI_RISK
    rule = _rule_for_index(family, pair_index)
    if family == RiskFamily.MULTI_RISK:
        rule = rule_by_id("SIM-SPECIAL-TAOREN")

    group_id = f"grp_{stable_hash([split.value, pair_index, family.value, perturbation.value], 14)}"
    case_token = stable_hash([global_seed, split.value, index], 12)
    patient_id = f"P-{case_token[:8].upper()}"
    visit_id = f"V-{case_token[4:].upper()}"
    visit_time = 100000 + index * 10
    prescription = _prescription_for(
        family,
        rule,
        unsafe,
        perturbation,
        rng,
    )
    patient_fact, patient_value, true_decision = _patient_fact(
        family, unsafe, rule
    )
    route_layout = ROUTE_LAYOUTS[split][pair_index % len(ROUTE_LAYOUTS[split])]
    query_style = QUERY_STYLES[split][
        (pair_index // len(ROUTE_LAYOUTS[split])) % len(QUERY_STYLES[split])
    ]
    (
        evidence_store,
        unavailable,
        prerequisites,
        required_keys,
        rule_fact,
    ) = _make_evidence_store(
        case_token=case_token,
        patient_id=patient_id,
        visit_time=visit_time,
        family=family,
        perturbation=perturbation,
        rule=rule,
        patient_value=patient_value,
        route_layout=route_layout,
        rng=rng,
    )

    public = PublicCase(
        patient_id=patient_id,
        visit_id=visit_id,
        age_band=rng.choice(AGE_BANDS),
        chief_concern=rng.choice(CHIEF_CONCERNS),
        prescription=prescription,
        declared_allergies=(),
        special_state="unknown",
        concurrent_medication="unknown",
    )
    rule_id = rule.rule_id if rule else "SIM-CLEAR-SCREEN"
    composition_key = "|".join(
        (
            family.value,
            perturbation.value,
            route_layout,
            query_style,
        )
    )
    case_id = f"case_{stable_hash([case_token, composition_key, unsafe], 16)}"
    world_id = f"world_{stable_hash([case_id, patient_fact, rule_id], 16)}"
    hidden = HiddenWorld(
        world_id=world_id,
        visit_time=visit_time,
        risk_family=family,
        true_decision=true_decision,
        rule_id=rule_id,
        patient_fact=patient_fact,
        rule_fact=rule_fact,
        required_fact_keys=required_keys,
        evidence_by_source=evidence_store,
        unavailable_sources=unavailable,
        prerequisite_by_source=prerequisites,
        route_layout=route_layout,
        query_style=query_style,
        perturbation=perturbation,
        counterfactual_pair_id=group_id,
    )
    return CaseSpec(
        case_id=case_id,
        group_id=group_id,
        split=split,
        seed=global_seed,
        composition_key=composition_key,
        public=public,
        hidden=hidden,
        metadata={
            "max_steps": max_steps,
            "research_only": True,
            "synthetic": True,
            "unsafe_polarity": unsafe,
            "generation_index": index,
            "generator_version": GENERATOR_VERSION,
        },
    )


POLICY_SYSTEM_PROMPT = (
    "你是 CARE-OPD 合成证据恢复策略。只能依据公开病例、历史动作和工具观察，"
    "输出一个符合契约的 JSON 动作；不得把本任务当作真实诊疗。"
)
PRIVILEGED_TEACHER_START = "<TRAINING_ONLY_VERIFIED_REPAIR>"
PRIVILEGED_TEACHER_END = "</TRAINING_ONLY_VERIFIED_REPAIR>"
_FORBIDDEN_POLICY_KEYS = {
    "hidden",
    "world_id",
    "risk_family",
    "true_decision",
    "rule_id",
    "patient_fact",
    "rule_fact",
    "required_fact_keys",
    "evidence_by_source",
    "unavailable_sources",
    "prerequisite_by_source",
    "route_layout",
    "query_style",
    "perturbation",
    "counterfactual_pair_id",
    "oracle_action",
    "repair_certificate",
    "privileged_teacher_context",
    "verified_oracle_repair",
    "verified_full_oracle_action",
}


def _reject_privileged_policy_keys(
    value: Any,
    path: str = "policy_state",
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_POLICY_KEYS:
                raise ValueError(
                    f"privileged field {path}.{key_text} cannot enter a "
                    "policy prompt"
                )
            _reject_privileged_policy_keys(
                nested,
                f"{path}.{key_text}",
            )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_privileged_policy_keys(
                nested,
                f"{path}[{index}]",
            )


def serialize_policy_prompt(
    state: Mapping[str, Any],
    *,
    rejected_action: Optional[Mapping[str, Any]] = None,
    feedback: Optional[Mapping[str, Any]] = None,
    instruction: str = "选择下一步动作。",
) -> str:
    _reject_privileged_policy_keys(state)
    payload: Dict[str, Any] = {
        "instruction": instruction,
        "policy_state": state,
    }
    if rejected_action is not None:
        payload["rejected_action"] = rejected_action
    if feedback is not None:
        payload["environment_feedback"] = feedback
    return POLICY_SYSTEM_PROMPT + "\n" + canonical_json(payload)


def _chat_messages(prompt: str, target: str) -> List[Dict[str, str]]:
    prefix = POLICY_SYSTEM_PROMPT + "\n"
    user_content = prompt[len(prefix) :] if prompt.startswith(prefix) else prompt
    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target},
    ]


def _oracle_row(case: CaseSpec) -> Dict[str, Any]:
    env = CareRecoveryEnv(case)
    steps: List[Dict[str, Any]] = []
    while not env.state.done:
        state_before = env.policy_state()
        action = env.oracle_action()
        _, reward, done, _ = env.step(action)
        record = env.state.steps[-1]
        steps.append(
            {
                "state": state_before,
                "action": to_jsonable(action),
                "observation": record.observation.policy_view(),
                "reward": reward,
                "done": done,
            }
        )
    initial_prompt = serialize_policy_prompt(steps[0]["state"])
    first_target = canonical_json(steps[0]["action"])
    return {
        "trajectory_id": f"oracle_{case.case_id}",
        "case_id": case.case_id,
        "group_id": case.group_id,
        "split": case.split.value,
        "composition_key": case.composition_key,
        "steps": steps,
        "resolved": env.state.resolved,
        "escalated": env.state.escalated,
        "total_steps": len(steps),
        "prompt": initial_prompt,
        "target": first_target,
        "messages": _chat_messages(initial_prompt, first_target),
    }


def _clean_oracle_examples(
    trajectory: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for index, step in enumerate(trajectory["steps"]):
        prompt = serialize_policy_prompt(step["state"])
        target = canonical_json(step["action"])
        examples.append(
            {
                "sample_id": f"{trajectory['trajectory_id']}_s{index}",
                "trajectory_id": trajectory["trajectory_id"],
                "case_id": trajectory["case_id"],
                "group_id": trajectory["group_id"],
                "split": trajectory["split"],
                "composition_key": trajectory["composition_key"],
                "step_index": index,
                "state": step["state"],
                "action": step["action"],
                "observation": step["observation"],
                "prompt": prompt,
                "target": target,
                "messages": _chat_messages(prompt, target),
            }
        )
    return examples


def _supervised_kd_example(
    clean_example: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert one verified oracle step into fixed-sequence KD format."""

    rollout = str(clean_example["target"])
    return {
        "sample_id": f"skd_{clean_example['sample_id']}",
        "trajectory_id": clean_example["trajectory_id"],
        "case_id": clean_example["case_id"],
        "group_id": clean_example["group_id"],
        "split": clean_example["split"],
        "composition_key": clean_example["composition_key"],
        "step_index": clean_example["step_index"],
        "student_prompt": clean_example["prompt"],
        "teacher_prompt": clean_example["prompt"],
        "rollout": rollout,
        "oracle_action": clean_example["action"],
        "risk_weight": 1.0,
        "component_mask": ["full_response"],
        "repair_spans": [
            {
                "start": 0,
                "end": len(rollout),
                "component": "full_response",
                "risk_weight": 1.0,
            }
        ],
        "bootstrap_kind": "supervised_fixed_oracle",
        "verification_scope": "host_oracle_target_only",
        "host_verified": True,
        "teacher_output_verified": False,
        "teacher_output_confidence": None,
        # Compatibility alias for existing collators.  The adjacent semantic
        # field prevents this from being reported as generated-output review.
        "teacher_verified": True,
        "teacher_verified_semantics": (
            "host_oracle_target_compatibility_alias_not_generated_output"
        ),
        "teacher_confidence": 1.0,
    }


def _wrong_source_action(oracle: Action) -> Action:
    kinds = (
        ActionKind.SEARCH_MEM0,
        ActionKind.SEARCH_GRAPHITI,
        ActionKind.SEARCH_LIGHTRAG,
        ActionKind.ASK_DOCTOR,
    )
    chosen = next(kind for kind in kinds if kind != oracle.kind)
    return Action(
        kind=chosen,
        query=oracle.query,
        rationale="合成故障注入：错误证据源。",
    )


def _rejected_candidates(env: CareRecoveryEnv) -> Tuple[Tuple[str, Action], ...]:
    oracle = env.oracle_action()
    candidates: List[Tuple[str, Action]] = []
    if oracle.kind.is_tool:
        candidates.extend(
            (
                ("wrong_source", _wrong_source_action(oracle)),
                (
                    "query_underspecified",
                    Action(
                        kind=oracle.kind,
                        query="复诊",
                        rationale="合成故障注入：查询缺失实体与时间约束。",
                    ),
                ),
                (
                    "premature_decision",
                    Action(
                        kind=ActionKind.DECIDE,
                        decision=Decision.CLEAR,
                        rationale="合成故障注入：未取证即放行。",
                    ),
                ),
            )
        )
    elif oracle.kind == ActionKind.DECIDE:
        wrong = (
            Decision.CLEAR
            if oracle.decision != Decision.CLEAR
            else Decision.BLOCK
        )
        candidates.extend(
            (
                (
                    "wrong_decision",
                    Action(
                        kind=ActionKind.DECIDE,
                        evidence_ids=oracle.evidence_ids,
                        decision=wrong,
                        rationale="合成故障注入：证据充分但判定方向错误。",
                    ),
                ),
                (
                    "missing_evidence_ids",
                    Action(
                        kind=ActionKind.DECIDE,
                        evidence_ids=(),
                        decision=oracle.decision,
                        rationale="合成故障注入：遗漏证据引用。",
                    ),
                ),
                (
                    "unnecessary_escalation",
                    Action(
                        kind=ActionKind.HUMAN_REVIEW,
                        rationale="合成故障注入：未尝试自动完成。",
                    ),
                ),
            )
        )
    return tuple(candidates)


def _recovery_rows(
    case: CaseSpec,
    recoveries_per_case: int,
) -> List[Dict[str, Any]]:
    # Import lazily to keep the environment independently usable.
    from .verifier import CausalRepairVerifier

    env = CareRecoveryEnv(case)
    checkpoints: List[CareRecoveryEnv] = []
    while not env.state.done:
        checkpoints.append(env.clone())
        env.step(env.oracle_action())

    pool: List[Tuple[CareRecoveryEnv, str, Action]] = []
    for checkpoint in checkpoints:
        for injection, action in _rejected_candidates(checkpoint):
            if action.kind == ActionKind.HUMAN_REVIEW:
                continue
            if not CausalRepairVerifier(checkpoint).assess(
                action
            ).resolves_subgoal:
                pool.append((checkpoint, injection, action))
    rng = _rng(case.seed, case.case_id, "recoveries")
    rng.shuffle(pool)

    rows: List[Dict[str, Any]] = []
    for index, (checkpoint, injection, rejected) in enumerate(
        pool[:recoveries_per_case]
    ):
        verifier = CausalRepairVerifier(checkpoint)
        certificate = verifier.localize(rejected)
        rejected_observation, _, _, _ = checkpoint.peek(rejected)

        # Actual error state followed by an adaptive oracle recovery.  This is
        # the on-policy view used by the proposed method.
        recovery_env = checkpoint.clone()
        recovery_env.step(rejected)
        post_failure_state = recovery_env.policy_state()
        recovery_steps: List[Dict[str, Any]] = []
        while not recovery_env.state.done:
            recovery_state = recovery_env.policy_state()
            recovery_action = recovery_env.oracle_action()
            recovery_env.step(recovery_action)
            recovery_record = recovery_env.state.steps[-1]
            recovery_steps.append(
                {
                    "state": recovery_state,
                    "action": to_jsonable(recovery_action),
                    "observation": recovery_record.observation.policy_view(),
                    "done": recovery_record.done,
                }
            )

        # Counterfactual splice view: replace the failed transition at the
        # transition point, then retain the verified oracle suffix.
        splice_env = checkpoint.clone()
        splice_env.step(certificate.intervened_action)
        splice_steps: List[Dict[str, Any]] = [
            {
                "state": checkpoint.policy_state(),
                "action": to_jsonable(certificate.intervened_action),
                "observation": splice_env.state.steps[-1].observation.policy_view(),
                "done": splice_env.state.steps[-1].done,
            }
        ]
        while not splice_env.state.done:
            splice_state = splice_env.policy_state()
            splice_action = splice_env.oracle_action()
            splice_env.step(splice_action)
            splice_record = splice_env.state.steps[-1]
            splice_steps.append(
                {
                    "state": splice_state,
                    "action": to_jsonable(splice_action),
                    "observation": splice_record.observation.policy_view(),
                    "done": splice_record.done,
                }
            )

        rows.append(
            {
                "recovery_id": f"rec_{stable_hash([case.case_id, injection, index], 16)}",
                "case_id": case.case_id,
                "group_id": case.group_id,
                "split": case.split.value,
                "composition_key": case.composition_key,
                "injection": injection,
                "state": checkpoint.policy_state(),
                "rejected_action": to_jsonable(rejected),
                "rejected_observation": rejected_observation.policy_view(),
                "oracle_action": to_jsonable(checkpoint.oracle_action()),
                "repair_certificate": to_jsonable(certificate),
                "risk_weight": certificate.before.risk_severity,
                "component_mask": list(certificate.selected_components),
                "post_failure_state": post_failure_state,
                "adaptive_recovery_steps": recovery_steps,
                "adaptive_recovery_resolved": recovery_env.state.resolved,
                "transition_splice_steps": splice_steps,
                "transition_splice_resolved": splice_env.state.resolved,
            }
        )
        row = rows[-1]
        rollout = canonical_json(row["rejected_action"])
        student_prompt = serialize_policy_prompt(row["state"])
        teacher_prompt = serialize_policy_prompt(
            row["state"],
            rejected_action=row["rejected_action"],
            feedback=row["rejected_observation"],
            instruction=(
                "验证器已定位失败。以教师身份给出最小、可验证的修复动作；"
                f"错误类型={certificate.failure_type.value}；"
                f"最小组件={list(certificate.selected_components)}。"
            ),
        )
        privileged_teacher_context = {
            "scope": "teacher_training_only",
            "must_not_enter_student_context": True,
            "verified_oracle_repair": to_jsonable(
                certificate.intervened_action
            ),
            "verified_full_oracle_action": to_jsonable(
                certificate.oracle_action
            ),
            "verification": {
                "certificate_hash": certificate.certificate_hash,
                "failure_type": certificate.failure_type.value,
                "minimal_components": list(
                    certificate.selected_components
                ),
                "verified_minimal": certificate.verified_minimal,
            },
        }
        teacher_prompt += (
            "\n"
            + PRIVILEGED_TEACHER_START
            + canonical_json(privileged_teacher_context)
            + PRIVILEGED_TEACHER_END
        )
        row.update(
            {
                "student_prompt": student_prompt,
                "teacher_prompt": teacher_prompt,
                "rollout": rollout,
                "repair_spans": _component_char_spans(
                    rollout,
                    row["rejected_action"],
                    certificate.selected_components,
                    certificate.before.risk_severity,
                ),
                "verification_scope": "host_repair_certificate_only",
                "host_repair_verified": bool(
                    certificate.verified_minimal
                ),
                # No teacher text is generated during offline synthesis, so
                # these fields must not imply greedy-output verification.
                "teacher_output_verified": False,
                "teacher_output_confidence": None,
                # Deprecated compatibility fields are deliberately false.
                "teacher_verified": False,
                "teacher_confidence": 0.0,
                "error_type": certificate.failure_type.value,
                "bootstrap_kind": "synthetic_offline_bootstrap",
                "privileged_teacher_context": privileged_teacher_context,
            }
        )
    return rows


def _component_char_spans(
    action_text: str,
    action: Mapping[str, Any],
    components: Sequence[str],
    risk_weight: float,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for component in components:
        value_text = canonical_json(action.get(component))
        marker = json.dumps(component, ensure_ascii=False) + ":" + value_text
        marker_start = action_text.find(marker)
        if marker_start < 0:
            continue
        start = marker_start + len(marker) - len(value_text)
        spans.append(
            {
                "start": start,
                "end": start + len(value_text),
                "component": component,
                "risk_weight": risk_weight,
            }
        )
    return spans


def _generation_views(
    recovery_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, List[Dict[str, Any]]]:
    views: Dict[str, List[Dict[str, Any]]] = {
        "full_recovery": [],
        "first_failure_revision": [],
        "transition_splice": [],
        "verified_pairs": [],
    }
    for row in recovery_rows:
        common = {
            "recovery_id": row["recovery_id"],
            "case_id": row["case_id"],
            "group_id": row["group_id"],
            "split": row["split"],
            "composition_key": row["composition_key"],
            "injection": row["injection"],
            "risk_weight": row["risk_weight"],
            "component_mask": row["component_mask"],
        }
        recovery_steps = row["adaptive_recovery_steps"]
        for recovery_step_index, recovery_step in enumerate(recovery_steps):
            first_step = recovery_step_index == 0
            prompt = serialize_policy_prompt(
                recovery_step["state"],
                rejected_action=(
                    row["rejected_action"] if first_step else None
                ),
                feedback=(
                    row["rejected_observation"] if first_step else None
                ),
                instruction=(
                    "从已发生的失败状态继续恢复，输出当前一步动作。"
                    if first_step
                    else "根据当前公开历史继续恢复，输出当前一步动作。"
                ),
            )
            target = canonical_json(recovery_step["action"])
            views["full_recovery"].append(
                {
                    **common,
                    "sample_id": (
                        f"{row['recovery_id']}_s{recovery_step_index}"
                    ),
                    "recovery_step_index": recovery_step_index,
                    "recovery_trace_length": len(recovery_steps),
                    "prompt_includes_failure_feedback": first_step,
                    "failure_state": row["state"],
                    "origin_rejected_action": row["rejected_action"],
                    "origin_rejected_observation": row[
                        "rejected_observation"
                    ],
                    "state": recovery_step["state"],
                    "action": recovery_step["action"],
                    "observation": recovery_step["observation"],
                    "done": recovery_step["done"],
                    "resolved": row["adaptive_recovery_resolved"],
                    "prompt": prompt,
                    "target": target,
                    "messages": _chat_messages(prompt, target),
                }
            )
        views["first_failure_revision"].append(
            {
                **common,
                "state": row["state"],
                "first_failed_action": row["rejected_action"],
                "feedback": row["rejected_observation"],
                "revised_action": row["repair_certificate"][
                    "intervened_action"
                ],
                "minimal_components": row["component_mask"],
                "verified": row["repair_certificate"]["verified_minimal"],
                "prompt": serialize_policy_prompt(
                    row["state"],
                    rejected_action=row["rejected_action"],
                    feedback=row["rejected_observation"],
                    instruction="仅修正导致失败的最小动作组件。",
                ),
                "revised_response": canonical_json(
                    row["repair_certificate"]["intervened_action"]
                ),
            }
        )
        revision = views["first_failure_revision"][-1]
        revision["target"] = revision["revised_response"]
        revision["messages"] = _chat_messages(
            revision["prompt"], revision["revised_response"]
        )
        views["transition_splice"].append(
            {
                **common,
                "prefix_history": row["state"]["history"],
                "rejected_transition": {
                    "state": row["state"],
                    "action": row["rejected_action"],
                    "observation": row["rejected_observation"],
                },
                "corrected_suffix": row["transition_splice_steps"],
                "resolved": row["transition_splice_resolved"],
                "prompt": serialize_policy_prompt(
                    row["state"],
                    rejected_action=row["rejected_action"],
                    feedback=row["rejected_observation"],
                    instruction="在首个失败转移点输出修正动作。",
                ),
                "target": canonical_json(
                    row["transition_splice_steps"][0]["action"]
                ),
            }
        )
        splice = views["transition_splice"][-1]
        splice["messages"] = _chat_messages(splice["prompt"], splice["target"])
        views["verified_pairs"].append(
            {
                **common,
                "state": row["state"],
                "rejected_action": row["rejected_action"],
                "preferred": row["repair_certificate"][
                    "intervened_action"
                ],
                "preference": "preferred",
                "failure_type": row["repair_certificate"]["failure_type"],
                "certificate_hash": row["repair_certificate"][
                    "certificate_hash"
                ],
                "prompt": serialize_policy_prompt(row["state"]),
                "chosen": canonical_json(
                    row["repair_certificate"]["intervened_action"]
                ),
                "rejected_text": canonical_json(row["rejected_action"]),
            }
        )
        pair = views["verified_pairs"][-1]
        # Fixed names used by common preference-training collators.
        pair["rejected"] = pair.pop("rejected_text")
    return views


def _paired_eval_rows(cases: Sequence[CaseSpec]) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[CaseSpec]] = defaultdict(list)
    for case in cases:
        by_group[case.group_id].append(case)
    rows: List[Dict[str, Any]] = []
    for group_id, group in sorted(by_group.items()):
        safe = [
            case
            for case in group
            if case.hidden.true_decision == Decision.CLEAR
        ]
        risky = [
            case
            for case in group
            if case.hidden.true_decision != Decision.CLEAR
        ]
        if not safe or not risky:
            continue
        safe_case = sorted(safe, key=lambda x: x.case_id)[0]
        risky_case = sorted(risky, key=lambda x: x.case_id)[0]
        safe_input = CareRecoveryEnv(safe_case).policy_state()
        risky_input = CareRecoveryEnv(risky_case).policy_state()
        rows.append(
            {
                "pair_id": group_id,
                "pair_type": "risk_label_contrast",
                "is_nuisance_control": False,
                "same_hidden_world": False,
                "semantics": (
                    "同一生成组的 risky-vs-safe 标签对：safe 的证据支持 clear，"
                    "risky 的证据支持 block/review。两者不是同一隐藏世界，"
                    "只用于成对安全判定，不用于 critical-vs-nuisance 恢复归因。"
                ),
                "split": safe_case.split.value,
                "composition_key": safe_case.composition_key,
                "safe": {
                    "case_id": safe_case.case_id,
                    "policy_input": safe_input,
                    "prompt": serialize_policy_prompt(
                        safe_input
                    ),
                    "role": "safe_clear_control",
                    "label": safe_case.hidden.true_decision.value,
                },
                "risky": {
                    "case_id": risky_case.case_id,
                    "policy_input": risky_input,
                    "prompt": serialize_policy_prompt(
                        risky_input
                    ),
                    "role": "risky_block_or_review",
                    "label": risky_case.hidden.true_decision.value,
                },
                "matched_attributes": [
                    "risk_family",
                    "perturbation",
                    "route_layout",
                    "query_style",
                    "generation_pair",
                ],
                "intentionally_varied_attributes": [
                    "patient_identity",
                    "risk_polarity",
                    "prescription_or_patient_fact_surface",
                ],
            }
        )
    return rows


def _oracle_continuation_summary(env: CareRecoveryEnv) -> Dict[str, Any]:
    clone = env.clone()
    steps = 0
    while not clone.state.done:
        clone.step(clone.oracle_action())
        steps += 1
    return {
        "steps": steps,
        "resolved": clone.state.resolved,
        "escalated": clone.state.escalated,
        "terminal_decision": (
            None
            if clone.state.terminal_decision is None
            else clone.state.terminal_decision.value
        ),
    }


def _paired_prefix_plan(case: CaseSpec) -> Dict[str, Any]:
    """Create clean/critical/nuisance starts with equal one-action detours.

    Critical and nuisance branches share the exact same latent world and
    checkpoint.  Their single prefix actions consume equal budget; only the
    critical prefix attempts an unsupported safety decision.
    """

    clean_env = CareRecoveryEnv(case)
    critical_env = clean_env.clone()
    critical_action = Action(
        kind=ActionKind.DECIDE,
        evidence_ids=(),
        decision=Decision.CLEAR,
        rationale="受控关键故障：未取证即尝试放行。",
    )
    critical_env.step(critical_action)
    critical_record = critical_env.state.steps[-1]

    nuisance_env = clean_env.clone()
    nuisance_action: Optional[Action] = None
    for kind in (
        ActionKind.SEARCH_MEM0,
        ActionKind.SEARCH_GRAPHITI,
        ActionKind.SEARCH_LIGHTRAG,
        ActionKind.ASK_DOCTOR,
    ):
        candidate = Action(
            kind=kind,
            query=f"NUISANCE-{case.case_id}-无关占位查询",
            rationale="受控无关故障：只消耗一步预算，不触碰目标证据。",
        )
        observation, _, _, _ = nuisance_env.peek(candidate)
        if observation.status == ObservationStatus.NO_MATCH:
            nuisance_action = candidate
            break
    if nuisance_action is None:
        raise RuntimeError(f"{case.case_id}: cannot construct nuisance prefix")
    nuisance_env.step(nuisance_action)
    nuisance_record = nuisance_env.state.steps[-1]
    clean_summary = _oracle_continuation_summary(clean_env)
    critical_summary = _oracle_continuation_summary(critical_env)
    nuisance_summary = _oracle_continuation_summary(nuisance_env)

    return {
        "plan_id": f"prefix_{case.case_id}",
        "case_id": case.case_id,
        "group_id": case.group_id,
        "split": case.split.value,
        "composition_key": case.composition_key,
        "pair_type": "same_world_equal_cost_prefix_control",
        "case_label": case.hidden.true_decision.value,
        "semantics": {
            "clean": "无注入起点。",
            "critical": (
                "未取证即尝试 clear 的关键故障；该动作不得被解释为临床建议。"
            ),
            "nuisance": (
                "不命中任何目标证据的无关查询；与 critical 等量消耗一步预算。"
            ),
            "contrast": (
                "critical-vs-nuisance 控制相同的一步预算损失；"
                "clean-vs-critical 单独不能识别恢复能力。"
            ),
        },
        "action_cost": {
            "critical": 1,
            "nuisance": 1,
            "equal": True,
        },
        "clean": {
            "prefix": None,
            "start_state": clean_env.policy_state(),
            "prompt": serialize_policy_prompt(clean_env.policy_state()),
            "remaining_budget": clean_env.state.remaining_budget,
            "host_only_oracle_continuation": clean_summary,
        },
        "critical": {
            "prefix_action": to_jsonable(critical_action),
            "prefix_observation": critical_record.observation.policy_view(),
            "failure_type": critical_record.observation.failure_type.value,
            "start_state": critical_env.policy_state(),
            "prompt": serialize_policy_prompt(critical_env.policy_state()),
            "remaining_budget": critical_env.state.remaining_budget,
            "host_only_oracle_continuation": critical_summary,
        },
        "nuisance": {
            "prefix_action": to_jsonable(nuisance_action),
            "prefix_observation": nuisance_record.observation.policy_view(),
            "failure_type": nuisance_record.observation.failure_type.value,
            "start_state": nuisance_env.policy_state(),
            "prompt": serialize_policy_prompt(nuisance_env.policy_state()),
            "remaining_budget": nuisance_env.state.remaining_budget,
            "host_only_oracle_continuation": nuisance_summary,
        },
        "interpretation": (
            "critical-vs-nuisance controls for the one-action budget loss; "
            "clean-vs-critical alone is not causal recovery evidence."
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_row(handle: Any, row: Any) -> None:
    handle.write(canonical_json(row) + "\n")


def build_dataset(output: Path, config: SynthesisConfig) -> Dict[str, Any]:
    _validate_synthesis_config(config)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        if not config.overwrite:
            raise FileExistsError(
                f"{output} is non-empty; pass --overwrite to replace generated files"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    snapshot_path = output / "knowledge_seed.snapshot.json"
    snapshot_path.write_text(
        json.dumps(seed_snapshot(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    external_knowledge_path: Optional[Path] = None
    external_binding: Optional[Dict[str, Any]] = None
    if config.knowledge:
        external_knowledge_path = Path(config.knowledge).resolve()
        if not external_knowledge_path.exists():
            raise FileNotFoundError(external_knowledge_path)
        external_value = json.loads(
            external_knowledge_path.read_text(encoding="utf-8")
        )
        if not isinstance(external_value, Mapping):
            raise ValueError("knowledge seed must be a JSON object")
        for key in (
            "schema_version",
            "provenance",
            "review_status",
            "research_only",
            "clinical_use_permitted",
            "rules",
        ):
            if key not in external_value:
                raise ValueError(
                    f"knowledge seed must contain top-level {key!r}"
                )
        if not external_value["research_only"]:
            raise ValueError(
                "bundled generator accepts only research_only knowledge seeds"
            )
        if external_value["schema_version"] != "care-opd-knowledge-v1":
            raise ValueError(
                "unsupported knowledge schema_version: "
                f"{external_value['schema_version']!r}"
            )
        if external_value["clinical_use_permitted"] is not False:
            raise ValueError(
                "bundled generator rejects knowledge permitting clinical use"
            )
        if not isinstance(external_value["provenance"], Mapping) or not isinstance(
            external_value["review_status"],
            Mapping,
        ):
            raise ValueError(
                "knowledge provenance and review_status must be objects"
            )
        raw_rules = external_value["rules"]
        if not isinstance(raw_rules, list):
            raise ValueError("knowledge rules must be an array")
        external_rules: Dict[str, Mapping[str, Any]] = {}
        for index, external_rule in enumerate(raw_rules):
            if not isinstance(external_rule, Mapping):
                raise ValueError(f"knowledge rule {index} must be an object")
            required_rule_fields = (
                "rule_id",
                "family",
                "trigger_items",
                "aliases",
                "simulator_assertion",
                "decision",
                "severity",
            )
            missing_fields = [
                field
                for field in required_rule_fields
                if field not in external_rule
            ]
            if missing_fields:
                raise ValueError(
                    f"knowledge rule {index} missing fields: "
                    + ", ".join(missing_fields)
                )
            rule_id = str(external_rule["rule_id"])
            if rule_id in external_rules:
                raise ValueError(
                    f"duplicate knowledge rule_id: {rule_id}"
                )
            assertion = str(external_rule["simulator_assertion"])
            if "仿真" not in assertion:
                raise ValueError(
                    f"{rule_id}: simulator_assertion must be explicitly synthetic"
                )
            external_rules[rule_id] = external_rule
        builtin_rules = {
            rule.rule_id: rule
            for family in (
                RiskFamily.SPECIAL_STATE,
                RiskFamily.INCOMPATIBLE_PAIR,
                RiskFamily.ALLERGY,
                RiskFamily.CONCURRENT_MEDICATION,
            )
            for rule in rules_for_family(family)
        }
        missing_rule_ids = sorted(set(builtin_rules) - set(external_rules))
        if missing_rule_ids:
            raise ValueError(
                "knowledge seed is missing generator rule IDs: "
                + ", ".join(missing_rule_ids)
            )
        incompatible: List[str] = []
        for rule_id, builtin in builtin_rules.items():
            external = external_rules[rule_id]
            expected_severity = (
                "safety_critical"
                if builtin.severity >= 0.95
                else "consequential"
            )
            if (
                str(external.get("family")) != builtin.family.value
                or str(external.get("decision")) != builtin.risk_decision.value
                or tuple(external.get("trigger_items", ()))
                != builtin.trigger_items
                or tuple(external.get("aliases", ())) != builtin.aliases
                or str(external.get("severity")) != expected_severity
            ):
                incompatible.append(rule_id)
        if incompatible:
            raise ValueError(
                "knowledge rules conflict with the closed simulator contract: "
                + ", ".join(incompatible)
            )
        external_binding = {
            "status": "BOUND_AND_SCHEMA_CHECKED",
            "execution_source": "built_in_closed_world_rules",
            "snapshot_role": (
                "integrity_and_lineage_binding_only_not_clinical_authority"
            ),
            "clinical_use_permitted": False,
            "rule_ids": sorted(builtin_rules),
            "review_status": external_value["review_status"],
            "sha256": _file_sha256(external_knowledge_path),
        }
        shutil.copyfile(
            external_knowledge_path,
            output / "external_knowledge.snapshot.json",
        )

    manifest: Dict[str, Any] = {
        "format_version": DATA_FORMAT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "research_notice": RESEARCH_ONLY_NOTICE,
        "config": to_jsonable(config),
        "seed_snapshot": snapshot_path.name,
        "external_knowledge_snapshot": (
            "external_knowledge.snapshot.json"
            if external_knowledge_path is not None
            else None
        ),
        "knowledge_binding": external_binding,
        "training_view_contract": {
            "sft": {
                "views": [
                    "clean_oracle",
                    "full_recovery",
                    "first_failure_revision",
                    "transition_splice",
                ],
                "text_fields": ["prompt", "target"],
                "chat_field": "messages",
                "serialization": "canonical JSON per action",
                "full_recovery_granularity": "one row per recovery step",
            },
            "dpo": {
                "view": "verified_pairs",
                "fields": ["prompt", "chosen", "rejected"],
                "serialization": "canonical JSON action strings",
            },
            "opd": {
                "view": "recovery",
                "fields": [
                    "student_prompt",
                    "teacher_prompt",
                    "rollout",
                    "repair_spans",
                ],
                "teacher_context": "teacher_training_only",
                "verification_semantics": (
                    "host repair certificate only; no generated teacher "
                    "output is claimed verified"
                ),
            },
            "supervised_kd": {
                "view": "supervised_kd",
                "fields": [
                    "student_prompt",
                    "teacher_prompt",
                    "rollout",
                    "repair_spans",
                ],
                "sequence_source": "host_verified_clean_oracle",
                "mask": "full_response",
                "teacher_verified_compatibility": (
                    "host oracle target; not generated teacher output"
                ),
            },
        },
        "paired_evaluation_contract": {
            "risk_label_view": "paired_eval",
            "budget_control_view": "paired_prefix_plan",
            "primary_recovery_contrast": "critical_vs_nuisance",
        },
        "splits": {},
        "files": {},
    }

    all_group_sets: Dict[str, set] = {}
    all_composition_sets: Dict[str, set] = {}
    for split, count in config.counts().items():
        cases = [
            make_case(split, index, config.seed, config.max_steps)
            for index in range(count)
        ]
        case_path = output / f"cases_{split.value}.jsonl"
        oracle_path = output / f"oracle_{split.value}.jsonl"
        clean_oracle_path = output / f"clean_oracle_{split.value}.jsonl"
        supervised_kd_path = (
            output / f"supervised_kd_{split.value}.jsonl"
        )
        recovery_path = output / f"recovery_{split.value}.jsonl"
        jsonl_dump(cases, str(case_path))

        oracle_count = 0
        oracle_resolved = 0
        clean_oracle_count = 0
        supervised_kd_count = 0
        with oracle_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as oracle_handle, clean_oracle_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as clean_handle:
            with supervised_kd_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as supervised_kd_handle:
                for case in cases:
                    trajectory = _oracle_row(case)
                    _write_jsonl_row(oracle_handle, trajectory)
                    oracle_count += 1
                    oracle_resolved += int(bool(trajectory["resolved"]))
                    for example in _clean_oracle_examples(trajectory):
                        _write_jsonl_row(clean_handle, example)
                        _write_jsonl_row(
                            supervised_kd_handle,
                            _supervised_kd_example(example),
                        )
                        clean_oracle_count += 1
                        supervised_kd_count += 1
        if oracle_resolved != oracle_count:
            raise RuntimeError(
                f"{split.value}: generator produced an unsolved oracle case"
            )

        generation_view_names = (
            "full_recovery",
            "first_failure_revision",
            "transition_splice",
            "verified_pairs",
        )
        generation_view_paths = {
            name: output / f"{name}_{split.value}.jsonl"
            for name in generation_view_names
        }
        generation_view_counts = Counter()
        recovery_count = 0
        with ExitStack() as stack:
            recovery_handle = stack.enter_context(
                recovery_path.open("w", encoding="utf-8", newline="\n")
            )
            view_handles = {
                name: stack.enter_context(
                    path.open("w", encoding="utf-8", newline="\n")
                )
                for name, path in generation_view_paths.items()
            }
            for case in cases:
                for recovery_row in _recovery_rows(
                    case,
                    config.recoveries_per_case,
                ):
                    _write_jsonl_row(recovery_handle, recovery_row)
                    recovery_count += 1
                    row_views = _generation_views((recovery_row,))
                    for view_name, view_rows in row_views.items():
                        for view_row in view_rows:
                            _write_jsonl_row(
                                view_handles[view_name],
                                view_row,
                            )
                            generation_view_counts[view_name] += 1

        view_paths: List[Path] = [
            supervised_kd_path,
            *generation_view_paths.values(),
        ]
        paired_eval_path = output / f"paired_eval_{split.value}.jsonl"
        paired_rows = _paired_eval_rows(cases)
        jsonl_dump(paired_rows, str(paired_eval_path))
        view_paths.append(paired_eval_path)
        prefix_plan_path = output / f"paired_prefix_plan_{split.value}.jsonl"
        prefix_plan_count = jsonl_dump(
            (_paired_prefix_plan(case) for case in cases),
            str(prefix_plan_path),
        )
        view_paths.append(prefix_plan_path)

        decisions = Counter(case.hidden.true_decision.value for case in cases)
        families = Counter(case.hidden.risk_family.value for case in cases)
        perturbations = Counter(
            case.hidden.perturbation.value for case in cases
        )
        all_group_sets[split.value] = {case.group_id for case in cases}
        all_composition_sets[split.value] = {
            case.composition_key for case in cases
        }
        manifest["splits"][split.value] = {
            "cases": len(cases),
            "oracle_trajectories": oracle_count,
            "recovery_checkpoints": recovery_count,
            "paired_evaluation_groups": len(paired_rows),
            "paired_prefix_plans": prefix_plan_count,
            "generation_views": {
                "clean_oracle": clean_oracle_count,
                "supervised_kd": supervised_kd_count,
                **{
                    name: generation_view_counts[name]
                    for name in generation_view_names
                },
            },
            "decision_distribution": dict(sorted(decisions.items())),
            "family_distribution": dict(sorted(families.items())),
            "perturbation_distribution": dict(sorted(perturbations.items())),
            "groups": len(all_group_sets[split.value]),
            "composition_keys": sorted(all_composition_sets[split.value]),
            "oracle_resolution_rate": (
                oracle_resolved / max(1, oracle_count)
            ),
        }
        for path in (
            case_path,
            oracle_path,
            clean_oracle_path,
            recovery_path,
            *view_paths,
        ):
            manifest["files"][path.name] = {
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }

    splits = tuple(split.value for split in Split)
    manifest["leakage_audit"] = {
        "group_pairwise_overlap": {
            f"{a}:{b}": len(all_group_sets[a] & all_group_sets[b])
            for idx, a in enumerate(splits)
            for b in splits[idx + 1 :]
        },
        "composition_pairwise_overlap": {
            f"{a}:{b}": len(
                all_composition_sets[a] & all_composition_sets[b]
            )
            for idx, a in enumerate(splits)
            for b in splits[idx + 1 :]
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest["files"][manifest_path.name] = {
        "sha256": _file_sha256(manifest_path),
        "bytes": manifest_path.stat().st_size,
    }
    # Re-write once so manifest lists itself; validation ignores its recursive hash.
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize the formal CARE-OPD recovery dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional project YAML; explicit CLI values take precedence.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--knowledge", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train", type=int)
    parser.add_argument("--dev", type=int)
    parser.add_argument("--test", type=int)
    parser.add_argument("--recoveries-per-case", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [
            _yaml_scalar(item)
            for item in value[1:-1].split(",")
            if item.strip()
        ]
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_project_yaml(path: Path) -> Mapping[str, Any]:
    """Load YAML, with a dependency-free fallback for top-level project/data.

    PyYAML remains the preferred parser.  The fallback intentionally extracts
    only the scalar fields consumed by this generator and ignores deeper
    training configuration.
    """

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        result: Dict[str, Dict[str, Any]] = {}
        section: Optional[str] = None
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if indent == 0 and stripped.endswith(":"):
                section = stripped[:-1]
                result.setdefault(section, {})
                continue
            if indent == 2 and section and ":" in stripped:
                key, value = stripped.split(":", 1)
                result[section][key.strip()] = _yaml_scalar(value)
        return result
    return yaml.safe_load(text) or {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    yaml_config: Mapping[str, Any] = {}
    project_root = Path.cwd()
    if args.config:
        config_path = args.config.resolve()
        yaml_config = _load_project_yaml(config_path)
        project_root = (
            config_path.parent.parent
            if config_path.parent.name.lower() == "configs"
            else config_path.parent
        )
    if not isinstance(yaml_config, Mapping):
        raise SystemExit("--config must contain a YAML mapping")
    project_section = yaml_config.get("project", {})
    data_section = yaml_config.get("data", {})
    if not isinstance(project_section, Mapping) or not isinstance(
        data_section,
        Mapping,
    ):
        raise SystemExit("project and data config sections must be mappings")

    def choose(explicit: Any, configured: Any, fallback: Any) -> Any:
        return (
            explicit
            if explicit is not None
            else configured
            if configured is not None
            else fallback
        )

    output = args.output
    if output is None and data_section.get("output_dir"):
        output = project_root / str(data_section["output_dir"])
    if output is None:
        raise SystemExit("provide --output or data.output_dir in --config")
    knowledge = args.knowledge
    if knowledge is None and data_section.get("knowledge"):
        knowledge = project_root / str(data_section["knowledge"])
    train = int(choose(args.train, data_section.get("train_episodes"), 12000))
    dev = int(choose(args.dev, data_section.get("dev_episodes"), 1500))
    test = int(choose(args.test, data_section.get("test_episodes"), 3000))
    recoveries = int(
        choose(
            args.recoveries_per_case,
            data_section.get("counterfactuals_per_failure"),
            3,
        )
    )
    max_steps = int(
        choose(args.max_steps, data_section.get("max_steps"), 6)
    )
    seed = int(choose(args.seed, project_section.get("seed"), 20260728))

    if min(train, dev, test) < 1:
        raise SystemExit("all split counts must be positive")
    if not 1 <= recoveries <= 8:
        raise SystemExit("--recoveries-per-case must be in [1, 8]")
    if max_steps < 5:
        raise SystemExit("--max-steps must be at least 5")
    config = SynthesisConfig(
        seed=seed,
        train=train,
        dev=dev,
        test=test,
        recoveries_per_case=recoveries,
        max_steps=max_steps,
        overwrite=args.overwrite,
        knowledge=None if knowledge is None else str(knowledge.resolve()),
    )
    manifest = build_dataset(output, config)
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    print(f"Dataset written to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
