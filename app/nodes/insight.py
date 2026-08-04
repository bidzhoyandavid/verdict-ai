"""Insight node — turns the abex report.build_report() JSON contract into a
business-facing explanation. Never invents numbers beyond what's in state;
only prose framing comes from the LLM."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.company_context import context_system_prompt_fragment
from app.state import ABTestState

_UNIMPLEMENTED_METHODS = ("bayesian", "sequential", "cuped", "novelty", "segments")


def insight_node(state: ABTestState, llm: Any) -> dict:
    if state.get("srm_result", {}).get("has_srm"):
        payload = {"srm_result": state["srm_result"]}
        instruction = (
            "Sample ratio mismatch was detected — the experiment is invalid. "
            "Explain this to the analyst and that no significance result can be trusted "
            "until the allocation bug is fixed."
        )
    else:
        payload = {
            "test_result": state.get("test_result"),
            "guardrail_results": state.get("guardrail_results"),
            "outlier_decision": state.get("outlier_decision"),
        }
        instruction = (
            "Summarize this A/B test result for a business stakeholder: winner/no "
            "significant difference/need more data, effect size, any guardrail "
            "violations, and any caveats from warnings. Be concise."
        )

    context_fragment = context_system_prompt_fragment(state["company_id"])
    system = (
        "You explain abex statistical results in plain business language. "
        "Only use the numbers given below — never invent figures. "
        f"If asked about {_UNIMPLEMENTED_METHODS}, say it's not implemented yet."
    )
    if context_fragment:
        system += "\n\n" + context_fragment

    response = llm.invoke(
        [
            ("system", system),
            ("human", f"{instruction}\n\nData: {payload}"),
        ]
    )
    text = response.content if hasattr(response, "content") else str(response)

    return {"messages": [AIMessage(content=text)], "last_completed_step": "insight"}
