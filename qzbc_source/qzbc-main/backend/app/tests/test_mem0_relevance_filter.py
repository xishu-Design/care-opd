from app.mcp.adapters.mem0_real import _relevant_search_rows


def test_mem0_relevance_filter_drops_low_similarity_chat_history():
    rows = [
        {"id": "low-1", "score": 0.35, "memory": "旧的脾虚问答"},
        {"id": "low-2", "score": 0.41, "memory": "旧的骨蒸问答"},
        {"id": "high", "score": 0.78, "memory": "与当前学习主题相关的偏好"},
    ]

    selected = _relevant_search_rows(rows, min_score=0.55, limit=8)

    assert [item["id"] for item in selected] == ["high"]
