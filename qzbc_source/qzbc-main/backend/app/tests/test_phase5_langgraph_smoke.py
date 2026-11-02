from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class SmokeState(TypedDict, total=False):
    value: int
    graph_node_path: list[str]


def test_langgraph_stategraph_smoke_with_memory_checkpointer():
    def increment(state: SmokeState) -> SmokeState:
        return {
            "value": state.get("value", 0) + 1,
            "graph_node_path": state.get("graph_node_path", []) + ["IncrementNode"],
        }

    builder = StateGraph(SmokeState)
    builder.add_node("IncrementNode", increment)
    builder.add_edge(START, "IncrementNode")
    builder.add_edge("IncrementNode", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    result = graph.invoke(
        {"value": 1, "graph_node_path": []},
        config={"configurable": {"thread_id": "phase5-smoke"}},
    )

    assert result["value"] == 2
    assert result["graph_node_path"] == ["IncrementNode"]
