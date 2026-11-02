from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class InterruptSmokeState(TypedDict, total=False):
    value: int
    path: list[str]


def test_langgraph_interrupt_resume_smoke_with_command():
    def human_gate(state: InterruptSmokeState) -> InterruptSmokeState:
        resume_value = interrupt({"value": state.get("value", 0), "message": "confirm"})
        return {
            "value": state.get("value", 0) + int(resume_value["delta"]),
            "path": state.get("path", []) + ["HumanGate"],
        }

    builder = StateGraph(InterruptSmokeState)
    builder.add_node("HumanGate", human_gate)
    builder.add_edge(START, "HumanGate")
    builder.add_edge("HumanGate", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "phase6-smoke"}}

    interrupted = graph.invoke({"value": 1, "path": []}, config=config)
    assert "__interrupt__" in interrupted
    assert interrupted["path"] == []

    resumed = graph.invoke(Command(resume={"delta": 9}), config=config)
    assert resumed["value"] == 10
    assert resumed["path"] == ["HumanGate"]
