import unittest

from care_opd.environment import CareRecoveryEnv
from care_opd.knowledge import semantic_query_match
from care_opd.schemas import (
    Action,
    ActionKind,
    Decision,
    FailureType,
    ObservationStatus,
    Split,
    canonical_json,
)
from care_opd.synthesis import make_case


class EnvironmentTest(unittest.TestCase):
    def test_clean_oracle_resolves(self) -> None:
        case = make_case(Split.TRAIN, 0, 42)
        env = CareRecoveryEnv(case)
        rollout = env.oracle_rollout()
        self.assertTrue(rollout)
        replay = CareRecoveryEnv(case)
        for step in rollout:
            replay.step(step.action)
        self.assertTrue(replay.state.resolved)
        self.assertLessEqual(len(rollout), case.metadata["max_steps"])

    def test_bad_query_is_recoverable(self) -> None:
        case = make_case(Split.TRAIN, 0, 42)
        env = CareRecoveryEnv(case)
        oracle = env.oracle_action()
        _, _, done, info = env.step(
            Action(kind=oracle.kind, query="复诊")
        )
        self.assertFalse(done)
        self.assertEqual(
            env.state.steps[-1].observation.status,
            ObservationStatus.NO_MATCH,
        )
        self.assertTrue(env.state.steps[-1].observation.recoverable)

    def test_tool_action_rejects_nonnull_decision_field(self) -> None:
        case = make_case(Split.TRAIN, 0, 42)
        env = CareRecoveryEnv(case)
        oracle = env.oracle_action()
        env.step(
            Action(
                kind=oracle.kind,
                query=oracle.query,
                decision=Decision.CLEAR,
            )
        )
        observation = env.state.steps[-1].observation
        self.assertEqual(observation.status, ObservationStatus.REJECTED)
        self.assertEqual(
            observation.failure_type,
            FailureType.INVALID_ACTION,
        )
        self.assertFalse(env.state.done)

    def test_source_outage_has_alternative_oracle_route(self) -> None:
        # train indices 2/3 share special_state × source_outage.
        case = make_case(Split.TRAIN, 2, 42)
        self.assertTrue(case.hidden.unavailable_sources)
        env = CareRecoveryEnv(case)
        first = env.oracle_action()
        source = first.kind.value
        self.assertNotIn(source, case.hidden.unavailable_sources)
        for step in env.oracle_rollout():
            pass
        replay = CareRecoveryEnv(case)
        for step in env.oracle_rollout():
            replay.step(step.action)
        self.assertTrue(replay.state.resolved)

    def test_conflict_forces_resolution(self) -> None:
        case = make_case(Split.DEV, 0, 42)
        env = CareRecoveryEnv(case)
        env.step(env.oracle_action())
        self.assertEqual(
            env.state.steps[-1].observation.status,
            ObservationStatus.CONFLICT,
        )
        next_action = env.oracle_action()
        self.assertEqual(next_action.kind, ActionKind.ASK_DOCTOR)
        for step in env.oracle_rollout():
            pass

    def test_alias_cannot_satisfy_all_query_terms(self) -> None:
        self.assertFalse(
            semantic_query_match(
                "特殊状态",
                ("P-EXAMPLE", "special_state", "当前"),
                ("特殊状态", "特殊生理情况"),
            )
        )
        self.assertTrue(
            semantic_query_match(
                "P-EXAMPLE 特殊状态 当前",
                ("P-EXAMPLE", "special_state", "当前"),
                ("特殊状态", "特殊生理情况"),
            )
        )

    def test_rejected_decision_does_not_reveal_hidden_fact_keys(self) -> None:
        case = make_case(Split.TRAIN, 0, 42)
        env = CareRecoveryEnv(case)
        env.step(
            Action(
                kind=ActionKind.DECIDE,
                decision=Decision.CLEAR,
                rationale="受控测试故障。",
            )
        )
        observation = env.state.steps[-1].observation
        self.assertEqual(
            observation.failure_type,
            FailureType.PREMATURE_DECISION,
        )
        visible = canonical_json(env.policy_state())
        for fact_key in case.hidden.required_fact_keys:
            self.assertNotIn(fact_key, visible)

    def test_oracle_history_omits_hidden_identifiers(self) -> None:
        case = make_case(Split.TEST, 0, 42)
        env = CareRecoveryEnv(case)
        while not env.state.done:
            env.step(env.oracle_action())
            visible = canonical_json(env.policy_state())
            self.assertNotIn(case.hidden.world_id, visible)
            for fact_key in case.hidden.required_fact_keys:
                self.assertNotIn(fact_key, visible)

    def test_synonym_shift_changes_surface_but_remains_solvable(self) -> None:
        # train pair-index 2 is special_state × synonym_shift.
        case = make_case(Split.TRAIN, 4, 42)
        self.assertEqual(case.hidden.perturbation.value, "synonym_shift")
        prescription_names = {item.name for item in case.public.prescription}
        self.assertTrue(
            prescription_names & {"桃核仁", "桃仁饮片", "草红花", "红蓝花"}
        )
        env = CareRecoveryEnv(case)
        for step in env.oracle_rollout():
            env.step(step.action)
        self.assertTrue(env.state.resolved)

    def test_rule_only_conflict_has_doctor_resolution(self) -> None:
        # test pair-index 3 is incompatible_pair × conflicting_record.
        case = make_case(Split.TEST, 6, 42)
        env = CareRecoveryEnv(case)
        env.step(env.oracle_action())
        self.assertEqual(
            env.state.steps[-1].observation.status,
            ObservationStatus.CONFLICT,
        )
        self.assertEqual(env.oracle_action().kind, ActionKind.ASK_DOCTOR)
        while not env.state.done:
            env.step(env.oracle_action())
        self.assertTrue(env.state.resolved)


if __name__ == "__main__":
    unittest.main()
