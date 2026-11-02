"""Failure classification, causal repair localization and dataset validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .environment import CareRecoveryEnv
from .schemas import (
    ACTION_COMPONENTS,
    Action,
    ActionAssessment,
    ActionKind,
    CaseSpec,
    Decision,
    FailureType,
    Observation,
    ObservationStatus,
    RepairCertificate,
    Split,
    canonical_json,
    case_from_mapping,
    jsonl_iter,
    jsonl_load,
    stable_hash,
    to_jsonable,
)


RISK_SEVERITY: Mapping[FailureType, float] = {
    FailureType.NONE: 0.0,
    FailureType.PARSE_ERROR: 1.5,
    FailureType.INVALID_ACTION: 1.5,
    FailureType.WRONG_SOURCE: 1.0,
    FailureType.QUERY_UNDERSPECIFIED: 1.0,
    FailureType.REPEATED_NO_MATCH: 1.5,
    FailureType.IGNORED_UNAVAILABLE: 1.5,
    FailureType.STALE_EVIDENCE_USED: 2.5,
    FailureType.CONFLICT_IGNORED: 3.5,
    FailureType.MISSING_PREREQUISITE: 1.5,
    FailureType.PREMATURE_DECISION: 3.0,
    FailureType.UNSUPPORTED_DECISION: 3.0,
    FailureType.UNSAFE_CLEAR: 4.0,
    FailureType.SAFE_FALSE_BLOCK: 2.0,
    FailureType.UNNECESSARY_ESCALATION: 1.0,
    FailureType.CYCLE: 1.5,
    FailureType.BUDGET_EXHAUSTION: 3.0,
    FailureType.OTHER: 1.0,
}


class CausalRepairVerifier:
    """Counterfactually replace action components and find a minimal repair.

    This is a host-side diagnostic intervention.  It does not expose hidden
    state to the deployable policy.
    """

    def __init__(self, env: CareRecoveryEnv):
        self.env = env.clone()

    def assess(self, action: Action) -> ActionAssessment:
        oracle = self.env.oracle_action()
        observation, _, _, _ = self.env.peek(action)
        oracle_observation, _, _, _ = self.env.peek(oracle)
        resolves = self._same_subgoal(
            action,
            observation,
            oracle,
            oracle_observation,
        )
        failure = self._classify(
            action,
            observation,
            oracle,
            oracle_observation,
            resolves,
        )
        accepted = (
            observation.status == ObservationStatus.ACCEPTED
            or resolves
        )
        return ActionAssessment(
            accepted=accepted,
            failure_type=failure,
            status=observation.status,
            oracle_aligned=resolves,
            resolves_subgoal=resolves,
            risk_severity=RISK_SEVERITY[failure],
            notes=(observation.message,),
        )

    def assess_raw(self, action_json: str) -> ActionAssessment:
        try:
            action = Action.from_json(action_json)
        except ValueError as exc:
            failure = FailureType.PARSE_ERROR
            return ActionAssessment(
                accepted=False,
                failure_type=failure,
                status=ObservationStatus.REJECTED,
                oracle_aligned=False,
                resolves_subgoal=False,
                risk_severity=RISK_SEVERITY[failure],
                notes=(str(exc),),
            )
        return self.assess(action)

    def localize(self, rejected: Action) -> RepairCertificate:
        oracle = self.env.oracle_action()
        before = self.assess(rejected)
        passing: List[Tuple[Tuple[str, ...], Action, ActionAssessment]] = []

        if before.resolves_subgoal:
            passing.append(((), rejected, before))
        else:
            for size in range(1, len(ACTION_COMPONENTS) + 1):
                for components in itertools.combinations(ACTION_COMPONENTS, size):
                    intervened = self._intervene(rejected, oracle, components)
                    assessment = self.assess(intervened)
                    if assessment.resolves_subgoal:
                        passing.append((components, intervened, assessment))
                if passing:
                    break

        if not passing:
            # Replacing every component is guaranteed to yield the oracle, but
            # retain a defensive branch for corrupted external environments.
            components = ACTION_COMPONENTS
            intervened = oracle
            assessment = self.assess(oracle)
            passing.append((components, intervened, assessment))

        minimal_sets = tuple(
            sorted((item[0] for item in passing), key=lambda x: (len(x), x))
        )
        selected_components = minimal_sets[0]
        selected_action = next(
            action for components, action, _ in passing
            if components == selected_components
        )
        after = next(
            assessment for components, _, assessment in passing
            if components == selected_components
        )
        verified_minimal = self._verify_minimal(
            rejected, oracle, selected_components
        )
        state_hash = stable_hash(self.env.policy_state(), 32)
        payload = {
            "case_id": self.env.state.case.case_id,
            "state_hash": state_hash,
            "rejected_action": rejected,
            "oracle_action": oracle,
            "failure_type": before.failure_type,
            "minimal_component_sets": minimal_sets,
            "selected_components": selected_components,
            "intervened_action": selected_action,
            "before": before,
            "after": after,
            "verified_minimal": verified_minimal,
        }
        certificate_hash = stable_hash(payload, 32)
        return RepairCertificate(
            case_id=self.env.state.case.case_id,
            state_hash=state_hash,
            rejected_action=rejected,
            oracle_action=oracle,
            failure_type=before.failure_type,
            minimal_component_sets=minimal_sets,
            selected_components=selected_components,
            intervened_action=selected_action,
            before=before,
            after=after,
            verified_minimal=verified_minimal,
            certificate_hash=certificate_hash,
        )

    @staticmethod
    def _intervene(
        rejected: Action,
        oracle: Action,
        components: Sequence[str],
    ) -> Action:
        changes = {
            component: getattr(oracle, component) for component in components
        }
        return replace(rejected, **changes)

    def _verify_minimal(
        self,
        rejected: Action,
        oracle: Action,
        components: Sequence[str],
    ) -> bool:
        if not components:
            return self.assess(rejected).resolves_subgoal
        for size in range(len(components)):
            for subset in itertools.combinations(components, size):
                candidate = self._intervene(rejected, oracle, subset)
                if self.assess(candidate).resolves_subgoal:
                    return False
        return True

    def _evidence_keys(self, observation: Observation) -> set:
        visit_time = self.env.state.case.hidden.visit_time
        return {
            evidence.fact_key
            for evidence in observation.evidence
            if not evidence.is_distractor
            and evidence.active_at(visit_time)
            and evidence.reliability >= 0.7
        }

    def _same_subgoal(
        self,
        action: Action,
        observation: Observation,
        oracle: Action,
        oracle_observation: Observation,
    ) -> bool:
        if oracle.kind == ActionKind.DECIDE:
            return (
                action.kind == ActionKind.DECIDE
                and observation.status == ObservationStatus.ACCEPTED
            )
        if oracle.kind == ActionKind.HUMAN_REVIEW:
            return action.kind == ActionKind.HUMAN_REVIEW
        if not action.kind.is_tool:
            return False
        if observation.status not in {
            ObservationStatus.FOUND,
            ObservationStatus.CONFLICT,
        }:
            return False
        target_keys = self._evidence_keys(oracle_observation)
        actual_keys = self._evidence_keys(observation)
        return bool(target_keys & actual_keys)

    def _classify(
        self,
        action: Action,
        observation: Observation,
        oracle: Action,
        oracle_observation: Observation,
        resolves: bool,
    ) -> FailureType:
        if resolves:
            return FailureType.NONE
        if observation.failure_type != FailureType.NONE:
            return observation.failure_type
        if action.kind.is_tool and oracle.kind.is_tool:
            if action.kind != oracle.kind:
                return FailureType.WRONG_SOURCE
            if observation.status in {
                ObservationStatus.NO_MATCH,
                ObservationStatus.INVALID_QUERY,
            } and oracle_observation.status in {
                ObservationStatus.FOUND,
                ObservationStatus.CONFLICT,
            }:
                return FailureType.QUERY_UNDERSPECIFIED
        if action.kind == ActionKind.DECIDE and oracle.kind.is_tool:
            return FailureType.PREMATURE_DECISION
        if (
            action.kind == ActionKind.HUMAN_REVIEW
            and oracle.kind != ActionKind.HUMAN_REVIEW
        ):
            return FailureType.UNNECESSARY_ESCALATION
        return FailureType.OTHER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cases(data_dir: Path) -> Dict[Split, List[CaseSpec]]:
    result: Dict[Split, List[CaseSpec]] = {}
    for split in Split:
        path = data_dir / f"cases_{split.value}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        result[split] = [
            case_from_mapping(row) for row in jsonl_load(str(path))
        ]
    return result


def _replay_history(case: CaseSpec, state_row: Mapping[str, Any]) -> CareRecoveryEnv:
    env = CareRecoveryEnv(case)
    for entry in state_row.get("history", ()):
        env.step(Action.from_mapping(entry["action"]))
    return env


_STUDENT_FORBIDDEN_FRAGMENTS: Tuple[str, ...] = (
    "<TRAINING_ONLY_VERIFIED_REPAIR>",
    '"scope":"teacher_training_only"',
    '"must_not_enter_student_context":',
    '"verified_oracle_repair":',
    '"verified_full_oracle_action":',
    '"certificate_hash":',
    '"hidden":',
    '"world_id":',
    '"risk_family":',
    '"true_decision":',
    '"rule_id":',
    '"patient_fact":',
    '"rule_fact":',
    '"required_fact_keys":',
    '"evidence_by_source":',
    '"unavailable_sources":',
    '"prerequisite_by_source":',
    '"route_layout":',
    '"query_style":',
    '"perturbation":',
    '"counterfactual_pair_id":',
    '"oracle_action":',
    '"repair_certificate":',
)


def _check_student_prompt(
    prompt: Any,
    identifier: str,
    errors: List[str],
) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append(f"{identifier}: missing string prompt")
        return
    for fragment in _STUDENT_FORBIDDEN_FRAGMENTS:
        if fragment in prompt:
            errors.append(
                f"{identifier}: privileged material in student prompt: "
                f"{fragment[:40]}"
            )


def _check_messages(
    row: Mapping[str, Any],
    identifier: str,
    errors: List[str],
) -> None:
    from .synthesis import POLICY_SYSTEM_PROMPT

    messages = row.get("messages")
    target = row.get("target")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"{identifier}: messages must contain three turns")
        return
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        errors.append(f"{identifier}: invalid message roles {roles}")
    if messages[0].get("content") != POLICY_SYSTEM_PROMPT:
        errors.append(f"{identifier}: unexpected system message")
    user_content = messages[1].get("content")
    _check_student_prompt(user_content, f"{identifier}.messages[user]", errors)
    if isinstance(user_content, str) and POLICY_SYSTEM_PROMPT in user_content:
        errors.append(f"{identifier}: system prompt duplicated in user turn")
    if messages[2].get("content") != target:
        errors.append(f"{identifier}: assistant message differs from target")


def _check_canonical_action(
    text: Any,
    structured: Any,
    identifier: str,
    errors: List[str],
) -> None:
    if not isinstance(text, str):
        errors.append(f"{identifier}: action serialization must be a string")
    elif text != canonical_json(structured):
        errors.append(f"{identifier}: non-canonical action serialization")


def _validate_training_views(
    data_dir: Path,
    split: Split,
    case_lookup: Mapping[str, CaseSpec],
    errors: List[str],
) -> Dict[str, int]:
    """Validate direct SFT/DPO/OPD contracts and prompt isolation."""

    from .synthesis import (
        PRIVILEGED_TEACHER_END,
        PRIVILEGED_TEACHER_START,
    )

    counts: Dict[str, int] = {}
    sft_specs = {
        "clean_oracle": lambda row: row["action"],
        "first_failure_revision": lambda row: row["revised_action"],
        "transition_splice": lambda row: row["corrected_suffix"][0][
            "action"
        ],
    }
    for view, target_of in sft_specs.items():
        path = data_dir / f"{view}_{split.value}.jsonl"
        if not path.exists():
            errors.append(f"missing training view: {path.name}")
            continue
        counts[view] = 0
        for index, row in enumerate(jsonl_iter(str(path))):
            counts[view] += 1
            identifier = (
                row.get("sample_id")
                or row.get("recovery_id")
                or f"{path.name}:{index + 1}"
            )
            _check_student_prompt(row.get("prompt"), identifier, errors)
            _check_canonical_action(
                row.get("target"),
                target_of(row),
                identifier,
                errors,
            )
            _check_messages(row, identifier, errors)

    supervised_kd_path = (
        data_dir / f"supervised_kd_{split.value}.jsonl"
    )
    if not supervised_kd_path.exists():
        errors.append(
            f"missing supervised KD view: {supervised_kd_path.name}"
        )
    else:
        counts["supervised_kd"] = 0
        for index, row in enumerate(
            jsonl_iter(str(supervised_kd_path))
        ):
            counts["supervised_kd"] += 1
            identifier = row.get(
                "sample_id",
                f"{supervised_kd_path.name}:{index + 1}",
            )
            student_prompt = row.get("student_prompt")
            teacher_prompt = row.get("teacher_prompt")
            _check_student_prompt(student_prompt, identifier, errors)
            _check_student_prompt(
                teacher_prompt,
                f"{identifier}.teacher_prompt",
                errors,
            )
            if student_prompt != teacher_prompt:
                errors.append(
                    f"{identifier}: supervised KD prompts differ"
                )
            _check_canonical_action(
                row.get("rollout"),
                row.get("oracle_action"),
                f"{identifier}.rollout",
                errors,
            )
            rollout = row.get("rollout", "")
            expected_span = [
                {
                    "start": 0,
                    "end": len(rollout) if isinstance(rollout, str) else 0,
                    "component": "full_response",
                    "risk_weight": 1.0,
                }
            ]
            if (
                row.get("repair_spans") != expected_span
                or float(row.get("risk_weight", 0.0)) != 1.0
            ):
                errors.append(
                    f"{identifier}: invalid full-response KD mask"
                )
            if (
                row.get("verification_scope")
                != "host_oracle_target_only"
                or row.get("host_verified") is not True
                or row.get("teacher_verified") is not True
                or row.get("teacher_output_verified") is not False
                or row.get("teacher_output_confidence") is not None
                or "not_generated_output"
                not in str(row.get("teacher_verified_semantics", ""))
            ):
                errors.append(
                    f"{identifier}: invalid supervised KD verification metadata"
                )

    full_path = data_dir / f"full_recovery_{split.value}.jsonl"
    if not full_path.exists():
        errors.append(f"missing training view: {full_path.name}")
    else:
        counts["full_recovery"] = 0
        for index, row in enumerate(jsonl_iter(str(full_path))):
            counts["full_recovery"] += 1
            identifier = row.get(
                "recovery_id",
                f"{full_path.name}:{index + 1}",
            )
            _check_student_prompt(row.get("prompt"), identifier, errors)
            _check_canonical_action(
                row.get("target"),
                row.get("action"),
                identifier,
                errors,
            )
            step_index = row.get("recovery_step_index")
            trace_length = row.get("recovery_trace_length")
            if (
                not isinstance(step_index, int)
                or not isinstance(trace_length, int)
                or not 0 <= step_index < trace_length
            ):
                errors.append(
                    f"{identifier}: invalid recovery step index"
                )
            prompt = row.get("prompt", "")
            includes_feedback = row.get(
                "prompt_includes_failure_feedback"
            )
            if step_index == 0:
                if (
                    includes_feedback is not True
                    or '"rejected_action":' not in prompt
                    or '"environment_feedback":' not in prompt
                ):
                    errors.append(
                        f"{identifier}: first recovery step lacks feedback"
                    )
            elif (
                includes_feedback is not False
                or '"rejected_action":' in prompt
                or '"environment_feedback":' in prompt
            ):
                errors.append(
                    f"{identifier}: later recovery step repeats feedback"
                )
            _check_messages(row, identifier, errors)

    pair_path = data_dir / f"verified_pairs_{split.value}.jsonl"
    if not pair_path.exists():
        errors.append(f"missing training view: {pair_path.name}")
    else:
        counts["verified_pairs"] = 0
        for index, row in enumerate(jsonl_iter(str(pair_path))):
            counts["verified_pairs"] += 1
            identifier = row.get(
                "recovery_id",
                f"{pair_path.name}:{index + 1}",
            )
            _check_student_prompt(row.get("prompt"), identifier, errors)
            _check_canonical_action(
                row.get("chosen"),
                row.get("preferred"),
                f"{identifier}.chosen",
                errors,
            )
            _check_canonical_action(
                row.get("rejected"),
                row.get("rejected_action"),
                f"{identifier}.rejected",
                errors,
            )
            if row.get("chosen") == row.get("rejected"):
                errors.append(f"{identifier}: degenerate preference pair")

    recovery_path = data_dir / f"recovery_{split.value}.jsonl"
    if not recovery_path.exists():
        errors.append(f"missing OPD view: {recovery_path.name}")
    else:
        counts["recovery"] = 0
        for index, row in enumerate(jsonl_iter(str(recovery_path))):
            counts["recovery"] += 1
            identifier = row.get(
                "recovery_id",
                f"{recovery_path.name}:{index + 1}",
            )
            student_prompt = row.get("student_prompt")
            teacher_prompt = row.get("teacher_prompt")
            _check_student_prompt(student_prompt, identifier, errors)
            if (
                not isinstance(teacher_prompt, str)
                or PRIVILEGED_TEACHER_START not in teacher_prompt
                or PRIVILEGED_TEACHER_END not in teacher_prompt
            ):
                errors.append(
                    f"{identifier}: teacher prompt lacks training-only envelope"
                )
            context = row.get("privileged_teacher_context")
            if not isinstance(context, Mapping):
                errors.append(
                    f"{identifier}: missing privileged teacher context"
                )
            else:
                if context.get("scope") != "teacher_training_only":
                    errors.append(
                        f"{identifier}: invalid teacher-context scope"
                    )
                certificate = row["repair_certificate"]
                if context.get("verified_oracle_repair") != certificate.get(
                    "intervened_action"
                ):
                    errors.append(
                        f"{identifier}: teacher repair differs from certificate"
                    )
                verification = context.get("verification", {})
                if not verification.get("verified_minimal"):
                    errors.append(
                        f"{identifier}: teacher repair is not verified minimal"
                    )
                if (
                    isinstance(teacher_prompt, str)
                    and canonical_json(context) not in teacher_prompt
                ):
                    errors.append(
                        f"{identifier}: teacher context not serialized in prompt"
                    )
            _check_canonical_action(
                row.get("rollout"),
                row.get("rejected_action"),
                f"{identifier}.rollout",
                errors,
            )
            if (
                row.get("verification_scope")
                != "host_repair_certificate_only"
                or row.get("host_repair_verified") is not True
            ):
                errors.append(
                    f"{identifier}: invalid host-repair verification metadata"
                )
            if (
                row.get("teacher_output_verified") is not False
                or row.get("teacher_output_confidence") is not None
                or row.get("teacher_verified") is not False
                or row.get("teacher_confidence") != 0.0
            ):
                errors.append(
                    f"{identifier}: unverified teacher output marked verified"
                )

    paired_path = data_dir / f"paired_eval_{split.value}.jsonl"
    if not paired_path.exists():
        errors.append(f"missing paired view: {paired_path.name}")
    else:
        counts["paired_eval"] = 0
        for index, row in enumerate(jsonl_iter(str(paired_path))):
            counts["paired_eval"] += 1
            identifier = row.get("pair_id", f"{paired_path.name}:{index + 1}")
            if row.get("same_hidden_world") is not False:
                errors.append(
                    f"{identifier}: risky/safe pair must declare distinct worlds"
                )
            if row.get("safe", {}).get("label") != Decision.CLEAR.value:
                errors.append(f"{identifier}: safe member is not clear")
            if row.get("risky", {}).get("label") == Decision.CLEAR.value:
                errors.append(f"{identifier}: risky member is clear")
            for role in ("safe", "risky"):
                member = row.get(role, {})
                _check_student_prompt(
                    member.get("prompt"),
                    f"{identifier}.{role}",
                    errors,
                )

    prefix_path = data_dir / f"paired_prefix_plan_{split.value}.jsonl"
    if not prefix_path.exists():
        errors.append(f"missing paired view: {prefix_path.name}")
    else:
        counts["paired_prefix_plan"] = 0
        for index, row in enumerate(jsonl_iter(str(prefix_path))):
            counts["paired_prefix_plan"] += 1
            identifier = row.get("plan_id", f"{prefix_path.name}:{index + 1}")
            action_cost = row.get("action_cost", {})
            if (
                action_cost.get("equal") is not True
                or action_cost.get("critical") != 1
                or action_cost.get("nuisance") != 1
            ):
                errors.append(f"{identifier}: unequal prefix action cost")
            clean = row.get("clean", {})
            critical = row.get("critical", {})
            nuisance = row.get("nuisance", {})
            if (
                critical.get("remaining_budget")
                != nuisance.get("remaining_budget")
                or clean.get("remaining_budget", 0) - 1
                != critical.get("remaining_budget")
            ):
                errors.append(f"{identifier}: prefix budgets are not controlled")
            if critical.get("failure_type") != FailureType.PREMATURE_DECISION.value:
                errors.append(
                    f"{identifier}: critical prefix is not premature_decision"
                )
            if nuisance.get("failure_type") != FailureType.NONE.value:
                errors.append(
                    f"{identifier}: nuisance prefix is not benign"
                )
            for role, member in (
                ("clean", clean),
                ("critical", critical),
                ("nuisance", nuisance),
            ):
                _check_student_prompt(
                    member.get("prompt"),
                    f"{identifier}.{role}",
                    errors,
                )
                start_state = member.get("start_state", {})
                if start_state.get("case_id") != row.get("case_id"):
                    errors.append(
                        f"{identifier}.{role}: start-state case mismatch"
                    )
            critical_oracle = critical.get(
                "host_only_oracle_continuation",
                {},
            )
            nuisance_oracle = nuisance.get(
                "host_only_oracle_continuation",
                {},
            )
            if (
                critical_oracle.get("steps")
                != nuisance_oracle.get("steps")
                or critical_oracle.get("resolved")
                != nuisance_oracle.get("resolved")
            ):
                errors.append(
                    f"{identifier}: controlled branches differ in oracle feasibility"
                )
            if row.get("case_id") not in case_lookup:
                errors.append(f"{identifier}: unknown case_id")

    if counts.get("supervised_kd") != counts.get("clean_oracle"):
        errors.append(
            f"{split.value}: supervised_kd/clean_oracle count mismatch"
        )
    return counts


def validate_dataset(
    data_dir: Path,
    replay_sample: int = 256,
) -> Dict[str, Any]:
    data_dir = data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = _load_cases(data_dir)
    errors: List[str] = []
    warnings: List[str] = []

    case_ids: set = set()
    group_sets: Dict[Split, set] = {}
    composition_sets: Dict[Split, set] = {}
    for split, split_cases in cases.items():
        group_sets[split] = {case.group_id for case in split_cases}
        composition_sets[split] = {
            case.composition_key for case in split_cases
        }
        for case in split_cases:
            if case.case_id in case_ids:
                errors.append(f"duplicate case_id: {case.case_id}")
            case_ids.add(case.case_id)
            if not case.metadata.get("research_only"):
                errors.append(f"{case.case_id}: missing research_only flag")
            public_blob = canonical_json(case.policy_view())
            hidden_sentinels = (
                case.hidden.world_id,
                case.hidden.patient_fact,
                case.hidden.rule_fact,
                *case.hidden.required_fact_keys,
            )
            for sentinel in hidden_sentinels:
                if sentinel and sentinel in public_blob:
                    errors.append(
                        f"{case.case_id}: hidden-state leakage: {sentinel[:32]}"
                    )
            env = CareRecoveryEnv(case)
            rollout = env.oracle_rollout()
            if not rollout or not rollout[-1].done:
                errors.append(f"{case.case_id}: oracle did not terminate")
            oracle_clone = CareRecoveryEnv(case)
            for step in rollout:
                oracle_clone.step(step.action)
            if not oracle_clone.state.resolved:
                errors.append(f"{case.case_id}: oracle did not resolve")

    split_list = tuple(Split)
    for index, left in enumerate(split_list):
        for right in split_list[index + 1 :]:
            group_overlap = group_sets[left] & group_sets[right]
            comp_overlap = composition_sets[left] & composition_sets[right]
            if group_overlap:
                errors.append(
                    f"group leakage {left.value}/{right.value}: "
                    f"{len(group_overlap)}"
                )
            if comp_overlap:
                errors.append(
                    f"composition leakage {left.value}/{right.value}: "
                    f"{len(comp_overlap)}"
                )

    # Determinism: regenerate sampled rows from the recorded seed and index.
    from .synthesis import make_case

    checked_determinism = 0
    for split, split_cases in cases.items():
        stride = max(1, len(split_cases) // max(1, replay_sample // 3))
        for case in split_cases[::stride][: max(1, replay_sample // 3)]:
            index = int(case.metadata["generation_index"])
            regenerated = make_case(
                split,
                index,
                case.seed,
                int(case.metadata["max_steps"]),
            )
            if canonical_json(regenerated) != canonical_json(case):
                errors.append(f"{case.case_id}: deterministic replay mismatch")
            checked_determinism += 1

    case_lookup = {
        case.case_id: case
        for split_cases in cases.values()
        for case in split_cases
    }
    training_view_counts: Dict[str, Dict[str, int]] = {}
    for split in Split:
        training_view_counts[split.value] = _validate_training_views(
            data_dir,
            split,
            case_lookup,
            errors,
        )
        split_manifest = manifest.get("splits", {}).get(split.value, {})
        recorded_views = split_manifest.get("generation_views", {})
        expected_counts = {
            **recorded_views,
            "recovery": split_manifest.get("recovery_checkpoints"),
            "paired_eval": split_manifest.get(
                "paired_evaluation_groups"
            ),
            "paired_prefix_plan": split_manifest.get(
                "paired_prefix_plans"
            ),
        }
        for view_name, actual_count in training_view_counts[
            split.value
        ].items():
            expected_count = expected_counts.get(view_name)
            if expected_count is None:
                errors.append(
                    f"{split.value}: manifest lacks count for {view_name}"
                )
            elif int(expected_count) != actual_count:
                errors.append(
                    f"{split.value}: manifest count mismatch for "
                    f"{view_name}: {expected_count} != {actual_count}"
                )

    # Verify generated-file checksums (manifest's own recursive checksum excluded).
    for filename, info in manifest.get("files", {}).items():
        if filename == "manifest.json":
            continue
        path = data_dir / filename
        if not path.exists():
            errors.append(f"manifest file missing: {filename}")
        elif _sha256(path) != info.get("sha256"):
            errors.append(f"checksum mismatch: {filename}")

    # Re-check a bounded sample of causal certificates from public checkpoints.
    certificate_checked = 0
    for split in Split:
        sample_limit = max(1, replay_sample // 3)
        total_rows = int(
            manifest.get("splits", {})
            .get(split.value, {})
            .get("recovery_checkpoints", sample_limit)
        )
        stride = max(1, total_rows // sample_limit)
        sampled = 0
        recovery_path = data_dir / f"recovery_{split.value}.jsonl"
        for row_index, row in enumerate(jsonl_iter(str(recovery_path))):
            if row_index % stride:
                continue
            case = case_lookup[row["case_id"]]
            env = _replay_history(case, row["state"])
            verifier = CausalRepairVerifier(env)
            rejected = Action.from_mapping(row["rejected_action"])
            certificate = verifier.localize(rejected)
            recorded = row["repair_certificate"]
            if list(certificate.selected_components) != list(
                recorded["selected_components"]
            ):
                errors.append(
                    f"{row['recovery_id']}: repair localization mismatch"
                )
            if not certificate.verified_minimal:
                errors.append(
                    f"{row['recovery_id']}: non-minimal repair certificate"
                )
            certificate_checked += 1
            sampled += 1
            if sampled >= sample_limit:
                break

    report = {
        "status": "PASS" if not errors else "FAIL",
        "data_dir": str(data_dir),
        "cases": {split.value: len(rows) for split, rows in cases.items()},
        "determinism_cases_checked": checked_determinism,
        "repair_certificates_checked": certificate_checked,
        "training_view_counts": training_view_counts,
        "group_overlap": {
            f"{left.value}:{right.value}": len(
                group_sets[left] & group_sets[right]
            )
            for idx, left in enumerate(split_list)
            for right in split_list[idx + 1 :]
        },
        "composition_overlap": {
            f"{left.value}:{right.value}": len(
                composition_sets[left] & composition_sets[right]
            )
            for idx, left in enumerate(split_list)
            for right in split_list[idx + 1 :]
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CARE-OPD data leakage, oracle solvability and determinism."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replay-sample", type=int, default=256)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_dataset(args.data_dir, args.replay_sample)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
