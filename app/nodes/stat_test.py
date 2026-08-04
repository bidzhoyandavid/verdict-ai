"""Runs the selected stat test by dynamically importing `fn_path` from the
selector recommendation, plus effect size. No stats logic lives here —
this only wires abex functions together and builds the report."""

from __future__ import annotations

import importlib
from dataclasses import asdict

from abex.analysis.effect_size import effect_size_summary
from abex.report import build_report
from abex.stats.bootstrap import BootstrapResult
from abex.stats.frequentist import TestResult

from app.state import ABTestState, active_metric_col

ALPHA = 0.05


def _import_fn(fn_path: str):
    module_path, fn_name = fn_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def _split_groups(df, group_col: str, metric_col: str) -> dict[str, "pd.Series"]:
    return {str(g): sub[metric_col] for g, sub in df.groupby(group_col)}


def _split_groups_paired(df, group_col: str, metric_col: str, id_col: str) -> dict[str, "pd.Series"]:
    """Pivot to one row per id, one column per group, so `control`/`treatment`
    are aligned by id (required by paired tests like ttest_rel)."""
    wide = df.pivot_table(index=id_col, columns=group_col, values=metric_col, aggfunc="first")
    return {str(g): wide[g] for g in wide.columns}


def stat_test_node(state: ABTestState) -> dict:
    df = state["raw_data"]
    group_col = state["group_col"]
    metric_col = active_metric_col(state)
    id_col = state.get("id_col")
    recommendation = state["recommendation"]
    fn_path = recommendation["fn_path"]

    if state.get("is_paired_design") and id_col:
        groups = _split_groups_paired(df, group_col, metric_col, id_col)
    else:
        groups = _split_groups(df, group_col, metric_col)
    group_names = sorted(groups.keys())
    control_name, treatment_name = group_names[0], group_names[1]
    control, treatment = groups[control_name], groups[treatment_name]

    fn = _import_fn(fn_path)
    raw_result = fn(control, treatment)

    if isinstance(raw_result, TestResult):
        p_value, ci = raw_result.p_value, None
    elif isinstance(raw_result, BootstrapResult):
        p_value, ci = None, (raw_result.ci_low, raw_result.ci_high)
    else:
        raise TypeError(f"unsupported stat-test result type: {type(raw_result).__name__}")

    effect = effect_size_summary(control, treatment)
    report = build_report(
        metric=metric_col,
        method=recommendation["method_name"],
        p_value=p_value,
        effect=effect,
        alpha=ALPHA,
        ci=ci,
        warnings=list(recommendation.get("warnings", [])),
    )
    report["control_group"] = control_name
    report["treatment_group"] = treatment_name

    return {"test_result": report, "last_completed_step": "stat_test"}
