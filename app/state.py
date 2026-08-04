"""Shared LangGraph state schema for the A/B-test agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ABTestState(TypedDict):
    company_id: str
    raw_data: pd.DataFrame | None
    group_col: str | None
    metric_col: str | None
    id_col: str | None

    validation_report: dict | None
    metric_profile: dict | None
    srm_result: dict | None
    recommendation: dict | None
    test_result: dict | None
    guardrail_results: list[dict]

    messages: Annotated[list[BaseMessage], add_messages]
    needs_clarification: bool
    last_completed_step: str | None
    router_intent: str | None

    # outlier HITL
    outlier_mask: dict | None
    outlier_options: list[dict] | None
    outlier_recommendation: str | None
    outlier_decision: dict | None
    treated_metric_col: str | None

    # data source
    data_source: str | None
    sql_template_name: str | None
    sql_params: dict | None

    is_paired_design: bool | None


STEP_ORDER: tuple[str, ...] = (
    "load",
    "validate_profile",
    "outlier_review",
    "srm_gate",
    "test_selector",
    "stat_test",
    "guardrail",
    "insight",
)

# Fields to clear when the given step is revised — i.e. everything the step
# itself produces plus everything downstream of it.
DOWNSTREAM_FIELDS: dict[str, list[str]] = {
    "load": [
        "validation_report",
        "metric_profile",
        "outlier_mask",
        "outlier_options",
        "outlier_recommendation",
        "outlier_decision",
        "treated_metric_col",
        "is_paired_design",
        "srm_result",
        "recommendation",
        "test_result",
        "guardrail_results",
    ],
    "validate_profile": [
        "validation_report",
        "metric_profile",
        "outlier_mask",
        "outlier_options",
        "outlier_recommendation",
        "outlier_decision",
        "treated_metric_col",
        "is_paired_design",
        "srm_result",
        "recommendation",
        "test_result",
        "guardrail_results",
    ],
    "outlier_review": [
        "outlier_decision",
        "treated_metric_col",
        "srm_result",
        "recommendation",
        "test_result",
        "guardrail_results",
    ],
    "srm_gate": ["srm_result", "recommendation", "test_result", "guardrail_results"],
    "test_selector": ["recommendation", "test_result", "guardrail_results"],
    "stat_test": ["test_result", "guardrail_results"],
    "guardrail": ["guardrail_results"],
    "insight": [],
}


def empty_state(company_id: str) -> ABTestState:
    return ABTestState(
        company_id=company_id,
        raw_data=None,
        group_col=None,
        metric_col=None,
        id_col=None,
        validation_report=None,
        metric_profile=None,
        srm_result=None,
        recommendation=None,
        test_result=None,
        guardrail_results=[],
        messages=[],
        needs_clarification=False,
        last_completed_step=None,
        router_intent=None,
        outlier_mask=None,
        outlier_options=None,
        outlier_recommendation=None,
        outlier_decision=None,
        treated_metric_col=None,
        data_source=None,
        sql_template_name=None,
        sql_params=None,
        is_paired_design=None,
    )


def next_step(state: ABTestState) -> str:
    """First pipeline step after `last_completed_step` (or the first step
    if nothing has run yet). Used to resume the graph at the right node
    after a revise/clarify-answer round trip."""
    last = state.get("last_completed_step")
    if last is None or last not in STEP_ORDER:
        return STEP_ORDER[0]
    idx = STEP_ORDER.index(last)
    return STEP_ORDER[min(idx + 1, len(STEP_ORDER) - 1)]


def active_metric_col(state: ABTestState) -> str | None:
    """Column downstream nodes should read: treated column if outliers were
    handled, otherwise the raw metric column."""
    return state.get("treated_metric_col") or state.get("metric_col")


_LIST_FIELDS = {"guardrail_results"}


def _reset_value(field: str) -> Any:
    return [] if field in _LIST_FIELDS else None
