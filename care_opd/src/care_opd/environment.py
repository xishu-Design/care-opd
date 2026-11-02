"""Evidence-recovery POMDP used to collect on-policy failure trajectories."""

from __future__ import annotations

import copy
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .knowledge import RESEARCH_ONLY_NOTICE, semantic_query_match
from .schemas import (
    Action,
    ActionKind,
    CaseSpec,
    Decision,
    EpisodeState,
    Evidence,
    EvidenceSource,
    FailureType,
    Observation,
    ObservationStatus,
    StepRecord,
)


KIND_TO_SOURCE: Mapping[ActionKind, EvidenceSource] = {
    ActionKind.SEARCH_MEM0: EvidenceSource.MEM0,
    ActionKind.SEARCH_GRAPHITI: EvidenceSource.GRAPHITI,
    ActionKind.SEARCH_LIGHTRAG: EvidenceSource.LIGHTRAG,
    ActionKind.ASK_DOCTOR: EvidenceSource.DOCTOR,
}

SOURCE_TO_KIND: Mapping[EvidenceSource, ActionKind] = {
    value: key for key, value in KIND_TO_SOURCE.items()
}


class CareRecoveryEnv:
    """A deterministic, finite-horizon synthetic safety-review environment.

    The policy observes only :meth:`policy_state`.  HiddenWorld is held by the
    simulator/verifier and must never be inserted into a model prompt.
    """

    def __init__(self, case: Optional[CaseSpec] = None):
        self.state: Optional[EpisodeState] = None
        if case is not None:
            self.reset(case)

    def reset(self, case: CaseSpec) -> Dict[str, object]:
        self.state = EpisodeState(case=case)
        return self.policy_state()

    def policy_state(self) -> Dict[str, object]:
        state = self._require_state()
        view = state.policy_view()
        view["research_notice"] = RESEARCH_ONLY_NOTICE
        return view

    def clone(self) -> "CareRecoveryEnv":
        other = CareRecoveryEnv()
        other.state = copy.deepcopy(self._require_state())
        return other

    def step(
        self, action: Action
    ) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        state = self._require_state()
        if state.done:
            raise RuntimeError("episode is already terminal")

        oracle_before = self.oracle_action()
        shape_error = self._action_shape_error(action)
        if shape_error is not None:
            observation = Observation(
                ObservationStatus.REJECTED,
                KIND_TO_SOURCE.get(action.kind),
                shape_error,
                failure_type=FailureType.INVALID_ACTION,
            )
            reward, done = -0.5, False
        elif action.kind.is_tool:
            observation, reward, done = self._tool_step(action)
        elif action.kind == ActionKind.DECIDE:
            observation, reward, done = self._decision_step(action)
        elif action.kind == ActionKind.HUMAN_REVIEW:
            observation, reward, done = self._human_review_step(action)
        else:  # pragma: no cover - Enum prevents this through normal parsing.
            observation = Observation(
                ObservationStatus.REJECTED,
                None,
                "不支持的动作。",
                failure_type=FailureType.INVALID_ACTION,
            )
            reward, done = -1.0, False

        index = len(state.steps)
        state.steps.append(
            StepRecord(
                index=index,
                action=action,
                observation=observation,
                reward=reward,
                done=done,
                oracle_action=oracle_before,
            )
        )
        if not done and state.remaining_budget <= 0:
            done = True
            state.done = True
            observation = Observation(
                ObservationStatus.BUDGET_EXHAUSTED,
                observation.source,
                "固定动作预算耗尽，未形成充分证据链。",
                evidence=observation.evidence,
                failure_type=FailureType.BUDGET_EXHAUSTION,
                recoverable=False,
            )
            reward = -1.0
            state.steps[-1] = StepRecord(
                index=index,
                action=action,
                observation=observation,
                reward=reward,
                done=True,
                oracle_action=oracle_before,
            )

        info = {
            "case_id": state.case.case_id,
            "resolved": state.resolved,
            "escalated": state.escalated,
            "terminal_decision": (
                None
                if state.terminal_decision is None
                else state.terminal_decision.value
            ),
            "failure_type": observation.failure_type.value,
            "oracle_aligned": action == oracle_before,
            "oracle_action": oracle_before,
        }
        return self.policy_state(), reward, done, info

    def peek(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, object]]:
        clone = self.clone()
        _, reward, done, info = clone.step(action)
        observation = clone._require_state().steps[-1].observation
        return observation, reward, done, info

    def oracle_action(self) -> Action:
        """Privileged deterministic policy used only for labels and verification."""

        state = self._require_state()
        case = state.case
        hidden = case.hidden

        missing = self._missing_fact_keys()
        if not missing:
            evidence_ids = self._minimal_support_ids(hidden.required_fact_keys)
            return Action(
                kind=ActionKind.DECIDE,
                evidence_ids=evidence_ids,
                decision=hidden.true_decision,
                rationale="依据已获取的有效证据完成仿真风险判定。",
            )

        target_key = missing[0]
        candidates: List[Evidence] = []
        for source_name, entries in hidden.evidence_by_source.items():
            if source_name in hidden.unavailable_sources:
                continue
            for evidence in entries:
                if (
                    evidence.fact_key == target_key
                    and evidence.active_at(hidden.visit_time)
                    and not evidence.is_distractor
                    and evidence.evidence_id not in state.acquired_evidence
                ):
                    candidates.append(evidence)

        if not candidates:
            return Action(
                kind=ActionKind.HUMAN_REVIEW,
                rationale="仍缺关键证据且无可用获取路径，按仿真安全策略转人工。",
            )

        # Doctor evidence is a fallback/resolution channel.  It must not let a
        # privileged oracle skip an available machine-source conflict that the
        # benchmark is meant to expose.
        non_doctor = [
            evidence
            for evidence in candidates
            if evidence.source != EvidenceSource.DOCTOR
        ]
        if non_doctor:
            candidates = non_doctor

        source_rank = self._source_rank(hidden.route_layout)
        candidates.sort(
            key=lambda e: (
                source_rank.get(e.source.value, 99),
                -e.reliability,
                e.evidence_id,
            )
        )
        target = candidates[0]
        prerequisite = hidden.prerequisite_by_source.get(target.source.value)
        if prerequisite and prerequisite not in state.completed_prerequisites:
            doctor_candidates = [
                e
                for e in hidden.evidence_by_source.get(EvidenceSource.DOCTOR.value, ())
                if e.fact_key == prerequisite
            ]
            terms = (
                self._render_query(doctor_candidates[0])
                if doctor_candidates
                else f"{case.public.patient_id} 前置授权 确认"
            )
            return Action(
                kind=ActionKind.ASK_DOCTOR,
                query=terms,
                rationale="先完成该证据源要求的合成前置确认。",
            )

        return Action(
            kind=SOURCE_TO_KIND[target.source],
            query=self._render_query(target),
            rationale="获取尚缺失的当前证据。",
        )

    def oracle_rollout(self) -> Tuple[StepRecord, ...]:
        clone = self.clone()
        while not clone._require_state().done:
            clone.step(clone.oracle_action())
        return tuple(clone._require_state().steps)

    def _tool_step(self, action: Action) -> Tuple[Observation, float, bool]:
        state = self._require_state()
        hidden = state.case.hidden
        source = KIND_TO_SOURCE[action.kind]

        if not action.query.strip():
            return (
                Observation(
                    ObservationStatus.INVALID_QUERY,
                    source,
                    "工具查询不能为空。",
                    failure_type=FailureType.QUERY_UNDERSPECIFIED,
                ),
                -0.35,
                False,
            )

        if source.value in hidden.unavailable_sources:
            prior_same = any(
                step.observation.status == ObservationStatus.UNAVAILABLE
                and step.observation.source == source
                for step in state.steps
            )
            failure = (
                FailureType.IGNORED_UNAVAILABLE
                if prior_same
                else FailureType.NONE
            )
            return (
                Observation(
                    ObservationStatus.UNAVAILABLE,
                    source,
                    f"{source.value} 当前不可用，请改走其他证据路径。",
                    failure_type=failure,
                ),
                -0.25 if not prior_same else -0.55,
                False,
            )

        prerequisite = hidden.prerequisite_by_source.get(source.value)
        if prerequisite and prerequisite not in state.completed_prerequisites:
            return (
                Observation(
                    ObservationStatus.NEEDS_PREREQUISITE,
                    source,
                    "需先完成该证据源的合成前置确认。",
                    failure_type=FailureType.MISSING_PREREQUISITE,
                ),
                -0.35,
                False,
            )

        entries = hidden.evidence_by_source.get(source.value, ())
        matched = tuple(
            e
            for e in entries
            if semantic_query_match(
                action.query,
                e.required_terms,
                e.aliases,
                minimum_fraction=0.66,
            )
        )
        if not matched:
            prior_same = any(
                step.observation.status == ObservationStatus.NO_MATCH
                and step.observation.source == source
                for step in state.steps
            )
            return (
                Observation(
                    ObservationStatus.NO_MATCH,
                    source,
                    "未命中可验证证据；可改写实体/时间约束或切换证据源。",
                    failure_type=(
                        FailureType.REPEATED_NO_MATCH
                        if prior_same
                        else FailureType.NONE
                    ),
                ),
                -0.2 if not prior_same else -0.45,
                False,
            )

        for evidence in matched:
            state.acquired_evidence[evidence.evidence_id] = evidence
            if evidence.source == EvidenceSource.DOCTOR:
                state.completed_prerequisites.append(evidence.fact_key)

        active_non_distractors = [
            e
            for e in matched
            if e.active_at(hidden.visit_time) and not e.is_distractor
        ]
        contradictory = any(e.contradictory_to for e in matched)
        if contradictory or self._contains_fact_conflict(matched):
            return (
                Observation(
                    ObservationStatus.CONFLICT,
                    source,
                    "返回记录存在时序或事实冲突，需获取更高可靠度/当前时点证据。",
                    evidence=matched,
                    failure_type=FailureType.NONE,
                ),
                0.0,
                False,
            )
        if not active_non_distractors:
            return (
                Observation(
                    ObservationStatus.FOUND,
                    source,
                    "仅找到过期或干扰记录，不能作为当前判定的充分证据。",
                    evidence=matched,
                    failure_type=FailureType.STALE_EVIDENCE_USED,
                ),
                -0.1,
                False,
            )
        return (
            Observation(
                ObservationStatus.FOUND,
                source,
                f"返回 {len(matched)} 条合成证据。",
                evidence=matched,
            ),
            0.15,
            False,
        )

    def _decision_step(self, action: Action) -> Tuple[Observation, float, bool]:
        state = self._require_state()
        hidden = state.case.hidden
        if action.decision is None:
            return (
                Observation(
                    ObservationStatus.REJECTED,
                    None,
                    "decide 动作必须给出 decision。",
                    failure_type=FailureType.INVALID_ACTION,
                ),
                -0.5,
                False,
            )

        supplied = [
            state.acquired_evidence[evidence_id]
            for evidence_id in action.evidence_ids
            if evidence_id in state.acquired_evidence
        ]
        unknown_ids = sorted(set(action.evidence_ids) - set(state.acquired_evidence))
        valid_keys = {
            evidence.fact_key
            for evidence in supplied
            if evidence.active_at(hidden.visit_time)
            and not evidence.is_distractor
            and evidence.reliability >= 0.7
        }
        missing = sorted(set(hidden.required_fact_keys) - valid_keys)
        if unknown_ids or missing:
            failure = (
                FailureType.PREMATURE_DECISION
                if not action.evidence_ids
                else FailureType.UNSUPPORTED_DECISION
            )
            return (
                Observation(
                    ObservationStatus.REJECTED,
                    None,
                    "证据链不足：存在未获取、过期或未引用的必需证据。"
                    + (f" 未获取ID={unknown_ids}。" if unknown_ids else ""),
                    failure_type=failure,
                ),
                -0.8,
                False,
            )

        # An unresolved contradiction cannot be silently collapsed into clearance.
        if self._has_unresolved_conflict(supplied):
            return (
                Observation(
                    ObservationStatus.REJECTED,
                    None,
                    "所引证据仍含未解决冲突。",
                    failure_type=FailureType.CONFLICT_IGNORED,
                ),
                -0.9,
                False,
            )

        if action.decision != hidden.true_decision:
            if action.decision == Decision.CLEAR:
                failure = FailureType.UNSAFE_CLEAR
                penalty = -1.5
            elif (
                hidden.true_decision == Decision.CLEAR
                and action.decision == Decision.BLOCK
            ):
                failure = FailureType.SAFE_FALSE_BLOCK
                penalty = -0.7
            else:
                failure = FailureType.UNSUPPORTED_DECISION
                penalty = -0.8
            return (
                Observation(
                    ObservationStatus.REJECTED,
                    None,
                    "判定与隐藏仿真规则不一致。",
                    failure_type=failure,
                    recoverable=state.remaining_budget > 1,
                ),
                penalty,
                False,
            )

        state.done = True
        state.resolved = True
        state.terminal_decision = action.decision
        return (
            Observation(
                ObservationStatus.ACCEPTED,
                None,
                "证据充分且判定与仿真规则一致。",
            ),
            1.0,
            True,
        )

    def _human_review_step(
        self, action: Action
    ) -> Tuple[Observation, float, bool]:
        state = self._require_state()
        state.done = True
        state.escalated = True
        failure = (
            FailureType.UNNECESSARY_ESCALATION
            if state.case.hidden.true_decision == Decision.CLEAR
            else FailureType.NONE
        )
        return (
            Observation(
                ObservationStatus.ESCALATED,
                None,
                "已转人工复核；此安全回退不计为自动恢复成功。",
                failure_type=failure,
                recoverable=False,
            ),
            -0.25,
            True,
        )

    def _missing_fact_keys(self) -> Tuple[str, ...]:
        state = self._require_state()
        hidden = state.case.hidden
        missing: List[str] = []
        for key in hidden.required_fact_keys:
            candidates = [
                evidence
                for evidence in state.acquired_evidence.values()
                if evidence.fact_key == key
                and evidence.active_at(hidden.visit_time)
                and not evidence.is_distractor
                and evidence.reliability >= 0.7
            ]
            if not candidates:
                missing.append(key)
                continue
            best = max(e.reliability for e in candidates)
            winners = {e.fact_value for e in candidates if e.reliability == best}
            if len(winners) != 1:
                missing.append(key)
        return tuple(missing)

    def _minimal_support_ids(
        self, keys: Sequence[str]
    ) -> Tuple[str, ...]:
        state = self._require_state()
        hidden = state.case.hidden
        selected: List[str] = []
        for key in keys:
            candidates = [
                e
                for e in state.acquired_evidence.values()
                if e.fact_key == key
                and e.active_at(hidden.visit_time)
                and not e.is_distractor
                and e.reliability >= 0.7
            ]
            candidates.sort(key=lambda e: (-e.reliability, e.evidence_id))
            if candidates:
                selected.append(candidates[0].evidence_id)
        return tuple(selected)

    def _render_query(self, evidence: Evidence) -> str:
        """Render a deterministic, policy-plausible query surface.

        Query styles alter only public surface form.  The oracle never emits
        hidden fact keys or prerequisite tokens into a student-visible action.
        """

        state = self._require_state()
        hidden = state.case.hidden
        style = hidden.query_style
        terms = list(evidence.required_terms)
        use_alias = (
            hidden.perturbation.value == "synonym_shift"
            or style in {"alias", "paraphrase"}
        )
        if use_alias and evidence.aliases and terms:
            # Patient evidence keeps the public patient identifier and swaps
            # the field-name slot; rule evidence swaps the leading concept.
            index = (
                1
                if terms[0] == state.case.public.patient_id and len(terms) > 1
                else 0
            )
            terms[index] = evidence.aliases[0]

        if style == "short":
            needed = max(1, int(len(terms) * 0.66 + 0.999))
            terms = terms[:needed]

        query = " ".join(terms)
        if style == "temporal":
            query += " 当前时点"
        elif style == "paraphrase":
            query = "请核验 " + query
        elif style == "entity_dense":
            query += (
                f" 患者 {state.case.public.patient_id}"
                f" 访次 {state.case.public.visit_id}"
            )
        return query

    @staticmethod
    def _action_shape_error(action: Action) -> Optional[str]:
        """Enforce kind-specific fields before any simulator transition."""

        if action.kind.is_tool:
            if action.decision is not None:
                return "工具动作的 decision 必须为 null。"
            if action.evidence_ids:
                return "工具动作不得携带 evidence_ids。"
        elif action.kind == ActionKind.DECIDE:
            if action.query.strip():
                return "decide 动作的 query 必须为空。"
        elif action.kind == ActionKind.HUMAN_REVIEW:
            if (
                action.query.strip()
                or action.evidence_ids
                or action.decision is not None
            ):
                return "human_review 不得携带查询、证据ID或判定字段。"
        return None

    @staticmethod
    def _contains_fact_conflict(entries: Sequence[Evidence]) -> bool:
        by_key: Dict[str, set] = {}
        for evidence in entries:
            by_key.setdefault(evidence.fact_key, set()).add(evidence.fact_value)
        return any(len(values) > 1 for values in by_key.values())

    @staticmethod
    def _has_unresolved_conflict(entries: Sequence[Evidence]) -> bool:
        active_by_key: Dict[str, List[Evidence]] = {}
        for evidence in entries:
            active_by_key.setdefault(evidence.fact_key, []).append(evidence)
        for candidates in active_by_key.values():
            values = {e.fact_value for e in candidates}
            if len(values) <= 1:
                continue
            best = max(e.reliability for e in candidates)
            winners = {e.fact_value for e in candidates if e.reliability == best}
            if len(winners) != 1:
                return True
        return False

    @staticmethod
    def _source_rank(route_layout: str) -> Mapping[str, int]:
        layouts: Mapping[str, Tuple[EvidenceSource, ...]] = {
            "canonical": (
                EvidenceSource.MEM0,
                EvidenceSource.GRAPHITI,
                EvidenceSource.LIGHTRAG,
                EvidenceSource.DOCTOR,
            ),
            "patient_memory_first": (
                EvidenceSource.MEM0,
                EvidenceSource.LIGHTRAG,
                EvidenceSource.GRAPHITI,
                EvidenceSource.DOCTOR,
            ),
            "timeline_first": (
                EvidenceSource.GRAPHITI,
                EvidenceSource.MEM0,
                EvidenceSource.LIGHTRAG,
                EvidenceSource.DOCTOR,
            ),
            "duplicated_evidence": (
                EvidenceSource.GRAPHITI,
                EvidenceSource.LIGHTRAG,
                EvidenceSource.MEM0,
                EvidenceSource.DOCTOR,
            ),
            "swapped_patient_sources": (
                EvidenceSource.MEM0,
                EvidenceSource.GRAPHITI,
                EvidenceSource.DOCTOR,
                EvidenceSource.LIGHTRAG,
            ),
            "doctor_fallback": (
                EvidenceSource.GRAPHITI,
                EvidenceSource.MEM0,
                EvidenceSource.DOCTOR,
                EvidenceSource.LIGHTRAG,
            ),
        }
        order = layouts.get(route_layout, layouts["canonical"])
        return {source.value: index for index, source in enumerate(order)}

    def _require_state(self) -> EpisodeState:
        if self.state is None:
            raise RuntimeError("call reset(case) before using the environment")
        return self.state
