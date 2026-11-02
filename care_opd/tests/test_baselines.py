from care_opd.baselines import RecoveryExemplar, RecoveryExemplarIndex


def test_exemplar_retrieval_is_deterministic():
    index = RecoveryExemplarIndex(
        [
            RecoveryExemplar(
                failure_signature="memory timeout unavailable",
                diagnosis="memory unavailable",
                recovery_action={"action": "switch_source"},
            ),
            RecoveryExemplar(
                failure_signature="query returned empty no_match",
                diagnosis="query too narrow",
                recovery_action={"action": "rewrite_query"},
            ),
        ]
    )
    first = index.search("memory tool timeout", k=1)[0]
    assert first.diagnosis == "memory unavailable"

