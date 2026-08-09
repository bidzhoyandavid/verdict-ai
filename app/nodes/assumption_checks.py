"""Formal assumption checks and covariate balance.

The selector already ranks methods by *approximated* assumption violations
(skewness, outlier share). This node measures them for real, so the report can
say what was actually violated instead of what was suspected — and so a
parametric verdict on a badly non-normal metric does not pass unremarked.

Covariate balance is the A/A-style sanity check: if groups differ on columns
the treatment cannot have touched, randomisation itself is suspect.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
from abex.design.covariate_balance import check_covariate_balance
from abex.stats.assumptions import check_equal_variance, check_normality

from app.datasets import load_active
from app.state import ABTestState, active_metric_col

# Балансу нужен ровно двухгрупповой дизайн, как и в abex.
TWO_GROUPS = 2
MAX_COVARIATES = 10


def _numeric_covariates(df: pd.DataFrame, state: ABTestState) -> list[str]:
    """Числовые колонки, не участвующие в анализе как метрики: их различие
    между группами говорит о проблеме рандомизации, а не об эффекте."""
    metrics = set(state.get("metric_cols") or [])
    metrics.update({state.get("metric_col"), state.get("treated_metric_col")} - {None})
    structural = {state.get("group_col"), state.get("id_col"), state.get("timestamp_col")} - {None}

    return [
        str(column)
        for column in df.columns
        if str(column) not in metrics
        and str(column) not in structural
        and pd.api.types.is_numeric_dtype(df[column])
    ][:MAX_COVARIATES]


def assumption_checks_node(state: ABTestState) -> dict:
    df = load_active(state)
    group_col = state.get("group_col")
    metric_col = active_metric_col(state)
    if not group_col or not metric_col or metric_col not in df.columns:
        return {"assumption_results": {}, "last_completed_step": "assumption_checks"}

    groups = {str(name): sub[metric_col] for name, sub in df.groupby(group_col)}

    normality = []
    for name, values in groups.items():
        try:
            normality.append(asdict(check_normality(values, group=name)))
        except ValueError as exc:
            normality.append({"group": name, "error": str(exc), "is_normal": None})

    try:
        equal_variance = asdict(check_equal_variance(groups))
    except ValueError as exc:
        equal_variance = {"error": str(exc), "is_equal": None}

    balance = []
    if len(groups) == TWO_GROUPS:
        for covariate in _numeric_covariates(df, state):
            try:
                balance.append(asdict(check_covariate_balance(df, covariate, group_col)))
            except (ValueError, TypeError, KeyError) as exc:
                balance.append({"covariate": covariate, "error": str(exc), "is_balanced": None})

    return {
        "assumption_results": {
            "metric": metric_col,
            "normality": normality,
            "equal_variance": equal_variance,
            "covariate_balance": balance,
        },
        "last_completed_step": "assumption_checks",
    }
