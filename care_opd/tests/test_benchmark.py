from care_opd.benchmark import build_prefix_plans
from care_opd.environment import CareRecoveryEnv
from care_opd.schemas import Split
from care_opd.synthesis import make_case


def test_nuisance_prefix_consumes_budget_without_evidence():
    case = make_case(Split.TRAIN, index=0, global_seed=17, max_steps=6)
    plans = build_prefix_plans(case, [])
    env = CareRecoveryEnv(case)
    before = env.state.remaining_budget
    env.step(plans["nuisance"].action)
    assert env.state.remaining_budget == before - 1
    assert not env.state.acquired_evidence
    assert plans["critical"].action is not None

