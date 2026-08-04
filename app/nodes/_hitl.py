"""Reusable human-in-the-loop helper for graph nodes.

Thin wrapper over LangGraph's `interrupt()` so every HITL point in the
graph pauses/resumes the same way — `outlier_review` is the first case,
future ones (e.g. guardrail-violation confirmation) reuse this instead of
copy-pasting the interrupt call.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt


def ask_human(payload: dict[str, Any]) -> Any:
    """Pause the graph, surface `payload` to the UI, and return whatever the
    UI passes back via `Command(resume=...)` on continuation."""
    return interrupt(payload)
