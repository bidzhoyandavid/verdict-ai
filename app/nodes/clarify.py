"""Clarify node — asks the analyst a targeted question instead of guessing
(ambiguous columns, empty selector ranked list, unclean validation)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.company_context import context_system_prompt_fragment
from app.state import ABTestState


def clarify_node(state: ABTestState, llm: Any) -> dict:
    reasons = []
    validation_report = state.get("validation_report")
    if validation_report and validation_report.get("issues"):
        reasons.append(f"Data validation issues: {validation_report['issues']}")
    if state.get("recommendation") is None and state.get("metric_profile"):
        reasons.append(
            "No registered statistical test fits this metric profile "
            f"({state['metric_profile']}) — likely needs a method still in development."
        )

    context_fragment = context_system_prompt_fragment(state["company_id"])
    system = (
        "You are helping an analyst run an A/B test analysis. "
        "Ask one short, specific clarifying question about the issue below — "
        "do not guess or proceed on your own."
    )
    if context_fragment:
        system += "\n\n" + context_fragment

    response = llm.invoke(
        [
            ("system", system),
            ("human", "Issue(s) to clarify: " + "; ".join(reasons) if reasons else "Ambiguous request, ask what's needed."),
        ]
    )
    text = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [AIMessage(content=text)],
        "needs_clarification": True,
        "last_completed_step": "clarify",
    }
