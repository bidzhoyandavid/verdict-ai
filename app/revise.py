"""Cascading invalidation for the "revise a previous step" flow.

Pure, no LLM involved: given a step name, clear that step's own output plus
everything downstream, so the graph can re-enter the pipeline from that node
without dragging along stale results from before the change.
"""

from __future__ import annotations

from app.state import DOWNSTREAM_FIELDS, STEP_ORDER, ABTestState, _reset_value


def invalidate_from(state: ABTestState, step: str) -> ABTestState:
    """Return a copy of `state` with `step`'s output and all downstream
    fields cleared, and `last_completed_step` rewound to just before `step`.

    Raises
    ------
    ValueError
        If `step` is not a known pipeline step.
    """
    if step not in STEP_ORDER:
        raise ValueError(f"unknown step {step!r}, expected one of {STEP_ORDER}")

    new_state = dict(state)
    for field in DOWNSTREAM_FIELDS[step]:
        new_state[field] = _reset_value(field)

    idx = STEP_ORDER.index(step)
    new_state["last_completed_step"] = STEP_ORDER[idx - 1] if idx > 0 else None
    new_state["needs_clarification"] = False
    return new_state  # type: ignore[return-value]
