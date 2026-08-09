"""Guardrail check — runs alongside the primary metric, independent of
whether the primary result was significant/positive."""

from __future__ import annotations

from dataclasses import asdict

from abex.analysis.guardrails import check_guardrail

from app.datasets import load_active
from app.state import ABTestState


def guardrail_node(state: ABTestState, guardrail_specs: list[dict]) -> dict:
    """`guardrail_specs`: list of
    `{"metric_col": str, "max_allowed_degradation": float, "higher_is_better": bool}`
    for metrics present in the dataset alongside the primary metric.
    """
    df = load_active(state)
    group_col = state["group_col"]
    results = []

    if not guardrail_specs:
        return {"guardrail_results": [], "last_completed_step": "guardrail"}

    groups = {str(g): sub for g, sub in df.groupby(group_col)}
    group_names = sorted(groups.keys())
    control_name, treatment_name = group_names[0], group_names[1]

    for spec in guardrail_specs:
        metric_col = spec["metric_col"]
        if metric_col not in df.columns:
            continue
        result = check_guardrail(
            control=groups[control_name][metric_col],
            treatment=groups[treatment_name][metric_col],
            metric_name=metric_col,
            max_allowed_degradation=spec["max_allowed_degradation"],
            higher_is_better=spec.get("higher_is_better", True),
        )
        results.append(asdict(result))

    return {"guardrail_results": results, "last_completed_step": "guardrail"}
