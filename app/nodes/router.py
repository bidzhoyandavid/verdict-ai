"""Router — the only node that classifies free-text user intent, including
the revise-step branch that triggers cascading invalidation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.revise import invalidate_from
from app.state import STEP_ORDER, ABTestState

Intent = Literal["new_analysis", "question", "clarify_answer", "revise_step"]

_REVISABLE_FIELDS = (
    "group_col",
    "metric_col",
    "id_col",
    "outlier_decision",
)


class RouterDecision(BaseModel):
    intent: Intent
    revise_step: str | None = Field(
        default=None,
        description=f"If intent is revise_step, which pipeline step to redo. One of {STEP_ORDER}.",
    )
    revised_fields: dict[str, Any] | None = Field(
        default=None,
        description=f"If intent is revise_step, which state fields to overwrite. One of {_REVISABLE_FIELDS}.",
    )


def route(state: ABTestState, llm: Any) -> dict:
    last_message = state["messages"][-1].content
    structured_llm = llm.with_structured_output(RouterDecision)
    decision: RouterDecision = structured_llm.invoke(
        [
            (
                "system",
                "Classify the analyst's message into one intent: "
                "new_analysis (starting fresh), question (asking about the "
                "current result, no recompute needed), clarify_answer "
                "(answering a prior clarifying question), or revise_step "
                f"(wants to redo a step with a changed parameter — steps: {STEP_ORDER}, "
                f"revisable fields: {_REVISABLE_FIELDS}).",
            ),
            ("human", last_message),
        ]
    )

    if decision.intent != "revise_step" or not decision.revise_step:
        return {"router_intent": decision.intent, "needs_clarification": False}

    new_state = invalidate_from(state, decision.revise_step)
    if decision.revised_fields:
        new_state.update(decision.revised_fields)
    new_state["router_intent"] = "revise_step"
    return new_state
