from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver


@lru_cache
def get_langgraph_checkpointer() -> InMemorySaver:
    return InMemorySaver()

