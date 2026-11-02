import unittest

from care_opd.environment import CareRecoveryEnv
from care_opd.schemas import (
    Action,
    ActionKind,
    Decision,
    FailureType,
    Split,
)
from care_opd.synthesis import make_case
from care_opd.verifier import CausalRepairVerifier


class VerifierTest(unittest.TestCase):
    def test_query_is_minimal_causal_repair(self) -> None:
        env = CareRecoveryEnv(make_case(Split.TRAIN, 0, 42))
        oracle = env.oracle_action()
        rejected = Action(kind=oracle.kind, query="复诊")
        certificate = CausalRepairVerifier(env).localize(rejected)
        self.assertTrue(certificate.verified_minimal)
        self.assertIn(("query",), certificate.minimal_component_sets)
        self.assertTrue(certificate.after.resolves_subgoal)
        self.assertFalse(certificate.before.resolves_subgoal)

    def test_decision_repair_does_not_overwrite_evidence_ids(self) -> None:
        env = CareRecoveryEnv(make_case(Split.TRAIN, 0, 42))
        while env.oracle_action().kind != ActionKind.DECIDE:
            env.step(env.oracle_action())
        oracle = env.oracle_action()
        wrong = Action(
            kind=ActionKind.DECIDE,
            evidence_ids=oracle.evidence_ids,
            decision=(
                Decision.CLEAR
                if oracle.decision != Decision.CLEAR
                else Decision.BLOCK
            ),
            rationale="wrong",
        )
        certificate = CausalRepairVerifier(env).localize(wrong)
        self.assertEqual(certificate.selected_components, ("decision",))
        self.assertEqual(
            certificate.intervened_action.evidence_ids,
            wrong.evidence_ids,
        )

    def test_kind_change_repairs_incompatible_fields(self) -> None:
        env = CareRecoveryEnv(make_case(Split.TRAIN, 0, 42))
        rejected = Action(
            kind=ActionKind.DECIDE,
            decision=Decision.CLEAR,
            rationale="受控测试：过早判定。",
        )
        certificate = CausalRepairVerifier(env).localize(rejected)
        self.assertIn("kind", certificate.selected_components)
        self.assertIn("query", certificate.selected_components)
        self.assertIn("decision", certificate.selected_components)
        self.assertIsNone(certificate.intervened_action.decision)
        self.assertTrue(certificate.after.resolves_subgoal)

    def test_raw_parse_failure_is_caught(self) -> None:
        env = CareRecoveryEnv(make_case(Split.TRAIN, 0, 42))
        assessment = CausalRepairVerifier(env).assess_raw("{not-json")
        self.assertFalse(assessment.accepted)
        self.assertGreater(assessment.risk_severity, 0)

    def test_stale_evidence_does_not_resolve_oracle_subgoal(self) -> None:
        # train pair-index 4 is incompatible_pair × stale_record.
        env = CareRecoveryEnv(make_case(Split.TRAIN, 8, 42))
        oracle = env.oracle_action()
        self.assertEqual(oracle.kind, ActionKind.SEARCH_MEM0)
        stale_action = Action(
            kind=ActionKind.SEARCH_LIGHTRAG,
            query=oracle.query,
            rationale="受控测试：查询过期规则源。",
        )
        assessment = CausalRepairVerifier(env).assess(stale_action)
        self.assertFalse(assessment.resolves_subgoal)
        self.assertEqual(
            assessment.failure_type,
            FailureType.STALE_EVIDENCE_USED,
        )


if __name__ == "__main__":
    unittest.main()
