"""Assembles the LangGraph pipeline.

Deterministic nodes (validate_profile, srm_gate, test_selector, stat_test,
guardrail) call abex directly — no LLM in the loop. LLM is used only in
router/data_loader(column fallback)/outlier_review(recommendation text)/
clarify/insight. `outlier_review` interrupts via `_hitl.ask_human`, which
requires a checkpointer to pause/resume.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

from app.nodes.assumption_checks import assumption_checks_node
from app.nodes.charts import charts_node
from app.nodes.checks_summary import checks_summary_node
from app.nodes.clarify import clarify_node
from app.nodes.data_loader import load_node
from app.nodes.guardrail import guardrail_node
from app.nodes.insight import insight_node
from app.nodes.multiple_testing import multiple_testing_node
from app.nodes.outlier_review import outlier_review_node
from app.nodes.power_check import power_check_node
from app.nodes.ratio_metrics import ratio_metrics_node
from app.nodes.report_table import report_table_node
from app.nodes.router import route
from app.nodes.srm_gate import has_srm, srm_gate_node
from app.nodes.timeline_check import timeline_check_node
from app.nodes.verdict import verdict_node
from app.nodes.stat_test import stat_test_node
from app.nodes.test_selector import has_recommendation, test_selector_node
from app.nodes.validate_profile import needs_outlier_review, validate_profile_node
from app.state import ABTestState, next_step


def build_graph(
    llm: Any,
    guardrail_specs: list[dict] | None = None,
    checkpointer: Any | None = None,
):
    """`checkpointer` defaults to an in-process MemorySaver — fine for the CLI,
    but the API must pass a durable one (Postgres), otherwise every paused
    HITL run dies with the process."""
    guardrail_specs = guardrail_specs or []
    graph = StateGraph(ABTestState)

    graph.add_node("router", partial(route, llm=llm))
    graph.add_node("load", partial(load_node, llm=llm))
    graph.add_node("validate_profile", validate_profile_node)
    graph.add_node("outlier_review", outlier_review_node)
    graph.add_node("assumption_checks", assumption_checks_node)
    graph.add_node("timeline_check", timeline_check_node)
    graph.add_node("srm_gate", srm_gate_node)
    graph.add_node("test_selector", test_selector_node)
    graph.add_node("stat_test", stat_test_node)
    graph.add_node("ratio_metrics", ratio_metrics_node)
    graph.add_node("multiple_testing", multiple_testing_node)
    graph.add_node("guardrail", partial(guardrail_node, guardrail_specs=guardrail_specs))
    graph.add_node("power_check", power_check_node)
    graph.add_node("charts", charts_node)
    graph.add_node("report_table", report_table_node)
    graph.add_node("checks_summary", checks_summary_node)
    graph.add_node("verdict", verdict_node)
    graph.add_node("clarify", partial(clarify_node, llm=llm))
    graph.add_node("insight", partial(insight_node, llm=llm))

    graph.set_entry_point("router")

    def route_after_router(s: ABTestState) -> str:
        intent = s.get("router_intent", "new_analysis")
        if intent == "question":
            return "insight"
        if intent in ("revise_step", "clarify_answer"):
            return next_step(s)
        return "load"

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "load": "load",
            "validate_profile": "validate_profile",
            "assumption_checks": "assumption_checks",
            "timeline_check": "timeline_check",
            "outlier_review": "outlier_review",
            "srm_gate": "srm_gate",
            "test_selector": "test_selector",
            "stat_test": "stat_test",
            "ratio_metrics": "ratio_metrics",
            "multiple_testing": "multiple_testing",
            "guardrail": "guardrail",
            "power_check": "power_check",
            "report_table": "report_table",
            "checks_summary": "checks_summary",
            "verdict": "verdict",
            "charts": "charts",
            "insight": "insight",
        },
    )

    graph.add_conditional_edges(
        "load",
        lambda s: "clarify" if not (s.get("group_col") and s.get("metric_col")) else "validate_profile",
        {"clarify": "clarify", "validate_profile": "validate_profile"},
    )

    graph.add_conditional_edges(
        "validate_profile",
        lambda s: "clarify" if s.get("needs_clarification") else "assumption_checks",
        {"clarify": "clarify", "assumption_checks": "assumption_checks"},
    )

    graph.add_edge("assumption_checks", "timeline_check")

    graph.add_conditional_edges(
        "timeline_check",
        lambda s: "outlier_review" if needs_outlier_review(s) else "srm_gate",
        {"outlier_review": "outlier_review", "srm_gate": "srm_gate"},
    )

    graph.add_edge("outlier_review", "srm_gate")

    # SRM invalidates the experiment: skip the statistics, but still build the
    # table and charts so the analyst sees what the allocation actually looked like.
    graph.add_conditional_edges(
        "srm_gate",
        lambda s: "report_table" if has_srm(s) else "test_selector",
        {"report_table": "report_table", "test_selector": "test_selector"},
    )

    graph.add_conditional_edges(
        "test_selector",
        lambda s: "stat_test" if has_recommendation(s) else "clarify",
        {"stat_test": "stat_test", "clarify": "clarify"},
    )

    graph.add_edge("stat_test", "ratio_metrics")
    graph.add_edge("ratio_metrics", "multiple_testing")
    graph.add_edge("multiple_testing", "guardrail")
    graph.add_edge("guardrail", "power_check")
    graph.add_edge("power_check", "report_table")
    # Checks and the verdict read the finished table; charts read both.
    graph.add_edge("report_table", "checks_summary")
    graph.add_edge("checks_summary", "verdict")
    graph.add_edge("verdict", "charts")
    graph.add_edge("charts", "insight")
    graph.add_edge("clarify", END)
    graph.add_edge("insight", END)

    # State holds only JSON-ish values plus numpy scalars coming out of abex
    # reports; pickle_fallback covers the latter.
    if checkpointer is None:
        checkpointer = MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
    return graph.compile(checkpointer=checkpointer)
