import unittest
from unittest.mock import patch

from care_opd.qzbc_agent_ab import (
    Arm,
    COMMON_GRAPH_NODES,
    _generated_response_valid,
    _run_case,
    compare,
)


def generated_response():
    return {
        "title": "证据边界",
        "summary": "当前回答受可见证据约束。",
        "sections": [
            {
                "heading": "依据",
                "content": "只引用当前证据。",
                "source_ids": ["SRC-1"],
            }
        ],
        "warnings": ["证据不足时需复核。"],
        "next_actions": ["咨询专业人员。"],
        "disclaimer": "不替代诊疗。",
    }


class QZBCAgentABTest(unittest.TestCase):
    def test_run_case_requires_real_graph_and_exact_model(self):
        nodes = sorted(COMMON_GRAPH_NODES | {"SafetyPreNode"})
        response = {
            "success": True,
            "data": {
                "generated_response": generated_response(),
                "llm_result": {
                    "provider": "ollama",
                    "model": "trained:latest",
                    "metadata": {"fallback_used": False, "degraded": False},
                },
                "graph_node_path": nodes,
                "fallback_used": False,
                "degraded": False,
                "evidence_packet": {
                    "retrieved_evidence": [{"source_id": "SRC-1"}]
                },
            },
        }
        case = {
            "id": "patient",
            "role": "patient",
            "path": "/api/patient/education-chat",
            "message": "一般科普",
        }
        with patch(
            "care_opd.qzbc_agent_ab._request_json", return_value=response
        ):
            result = _run_case(
                Arm("trained", "http://trained", "trained:latest"),
                case,
                "token",
                expected_provider="ollama",
                timeout=10,
            )
        self.assertTrue(result["integrity_pass"])
        self.assertTrue(result["checks"]["citation_ids_grounded"])

    def test_schema_rejects_action_like_output(self):
        self.assertTrue(_generated_response_valid(generated_response()))
        self.assertFalse(_generated_response_valid({"action": "search_mem0"}))

    def test_compare_rejects_one_cached_backend(self):
        with self.assertRaisesRegex(ValueError, "two backend URLs"):
            compare(
                Arm("base", "http://same", "base"),
                Arm("trained", "http://same", "trained"),
                cases=[],
                password="password123",
                expected_provider="ollama",
                timeout=10,
                reset_demo=False,
            )


if __name__ == "__main__":
    unittest.main()
