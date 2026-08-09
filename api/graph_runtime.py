"""Single compiled graph per process, plus checkpointer selection.

Compiling per request would rebuild the whole StateGraph and re-open the
checkpointer connection on every message.
"""

from __future__ import annotations

import atexit
from functools import lru_cache
from typing import Any

from api.config import settings


def _checkpointer() -> Any:
    """Postgres when configured, in-memory otherwise.

    MemorySaver loses every paused HITL run on restart — acceptable for local
    dev only. Production must run on Postgres.
    """
    if not settings.is_postgres:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        return MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))

    from langgraph.checkpoint.postgres import PostgresSaver

    manager = PostgresSaver.from_conn_string(settings.database_url)
    saver = manager.__enter__()
    saver.setup()
    atexit.register(lambda: manager.__exit__(None, None, None))
    return saver


@lru_cache(maxsize=1)
def get_graph():
    from langchain_anthropic import ChatAnthropic

    from app.graph import build_graph

    llm = ChatAnthropic(model=settings.llm_model)
    # Guardrail specs are per-company config; wired in a later iteration.
    return build_graph(llm, guardrail_specs=[], checkpointer=_checkpointer())


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
