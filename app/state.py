"""Shared LangGraph state schema for the A/B-test agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ABTestState(TypedDict):
    company_id: str
    # The dataset itself lives in `app.datasets`; state carries only the id.
    dataset_id: str | None
    group_col: str | None
    # Primary metric — the one outlier treatment and the test selector work on.
    metric_col: str | None
    # Every metric to analyze, primary first. Multiple metrics trigger the
    # mandatory multiple-comparisons correction.
    metric_cols: list[str]
    id_col: str | None
    timestamp_col: str | None
    # Явное указание ролей групп; пусто — определяем по названиям.
    control_group: str | None
    treatment_group: str | None
    group_assignment: dict | None

    # Per-company guardrail definitions, supplied by the caller with the run.
    guardrail_specs: list[dict]
    # Ratio metrics must be declared: {"name", "numerator", "denominator"}.
    ratio_metric_specs: list[dict]

    validation_report: dict | None
    metric_profile: dict | None
    srm_result: dict | None
    recommendation: dict | None
    # One abex report per metric, in `metric_cols` order.
    test_results: list[dict]
    # Per group and metric: n, mean, std, conversion — the raw material for the
    # results table, which the report contract itself does not carry.
    group_stats: list[dict]
    multiple_testing_result: dict | None
    assumption_results: dict | None
    # Единый список проверок со статусами для UI.
    checks: list[dict]
    verdict: dict | None
    guardrail_results: list[dict]
    power_result: dict | None
    timeline_warnings: list[str]
    # Deterministic final table: one row per metric.
    results_table: list[dict]
    charts: list[dict]

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
    treated_dataset_id: str | None

    # data source
    data_source: str | None
    sql_template_name: str | None
    sql_params: dict | None

    is_paired_design: bool | None


STEP_ORDER: tuple[str, ...] = (
    "load",
    "validate_profile",
    "assumption_checks",
    "timeline_check",
    "outlier_review",
    "srm_gate",
    "test_selector",
    "stat_test",
    "ratio_metrics",
    "multiple_testing",
    "guardrail",
    "power_check",
    "report_table",
    "checks_summary",
    "verdict",
    "charts",
    "insight",
)

# Every field a step produces, so revising a step can clear its own output plus
# everything downstream of it.
_STEP_OUTPUTS: dict[str, list[str]] = {
    "load": ["group_assignment"],
    "validate_profile": ["validation_report", "metric_profile", "outlier_mask", "is_paired_design"],
    "assumption_checks": ["assumption_results"],
    "timeline_check": ["timeline_warnings"],
    "outlier_review": [
        "outlier_options",
        "outlier_recommendation",
        "outlier_decision",
        "treated_metric_col",
        "treated_dataset_id",
    ],
    "srm_gate": ["srm_result"],
    "test_selector": ["recommendation"],
    "stat_test": ["test_results", "group_stats"],
    "ratio_metrics": [],
    "multiple_testing": ["multiple_testing_result"],
    "guardrail": ["guardrail_results"],
    "power_check": ["power_result"],
    "report_table": ["results_table"],
    "checks_summary": ["checks"],
    "verdict": ["verdict"],
    "charts": ["charts"],
    "insight": [],
}


def _downstream_fields() -> dict[str, list[str]]:
    """`step -> fields to clear`, derived from STEP_ORDER so a new step cannot
    be added to the pipeline without its invalidation rule."""
    fields: dict[str, list[str]] = {}
    for index, step in enumerate(STEP_ORDER):
        affected: list[str] = []
        for later_step in STEP_ORDER[index:]:
            affected.extend(_STEP_OUTPUTS[later_step])
        fields[step] = affected
    return fields


DOWNSTREAM_FIELDS: dict[str, list[str]] = _downstream_fields()


def empty_state(company_id: str) -> ABTestState:
    return ABTestState(
        company_id=company_id,
        dataset_id=None,
        group_col=None,
        metric_col=None,
        metric_cols=[],
        id_col=None,
        timestamp_col=None,
        control_group=None,
        treatment_group=None,
        group_assignment=None,
        guardrail_specs=[],
        ratio_metric_specs=[],
        validation_report=None,
        metric_profile=None,
        srm_result=None,
        recommendation=None,
        test_results=[],
        group_stats=[],
        multiple_testing_result=None,
        assumption_results=None,
        checks=[],
        verdict=None,
        guardrail_results=[],
        power_result=None,
        timeline_warnings=[],
        results_table=[],
        charts=[],
        messages=[],
        needs_clarification=False,
        last_completed_step=None,
        router_intent=None,
        outlier_mask=None,
        outlier_options=None,
        outlier_recommendation=None,
        outlier_decision=None,
        treated_metric_col=None,
        treated_dataset_id=None,
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
    """Column downstream nodes should read for the primary metric: treated
    column if outliers were handled, otherwise the raw metric column."""
    return state.get("treated_metric_col") or state.get("metric_col")


def analysis_metric_cols(state: ABTestState) -> list[str]:
    """Every metric to analyze, with the primary one swapped for its treated
    column when outlier treatment produced one."""
    metrics = list(state.get("metric_cols") or [])
    primary = state.get("metric_col")
    if not metrics and primary:
        metrics = [primary]

    treated = state.get("treated_metric_col")
    if treated and primary and primary in metrics:
        metrics[metrics.index(primary)] = treated
    return metrics


def source_metric_col(state: ABTestState, metric_col: str) -> str:
    """Inverse of the treated-column swap — the name to show a human."""
    if metric_col == state.get("treated_metric_col") and state.get("metric_col"):
        return state["metric_col"]
    return metric_col


_LIST_FIELDS = {
    "guardrail_results",
    "guardrail_specs",
    "ratio_metric_specs",
    "checks",
    "metric_cols",
    "test_results",
    "group_stats",
    "timeline_warnings",
    "results_table",
    "charts",
}


def _reset_value(field: str) -> Any:
    return [] if field in _LIST_FIELDS else None
