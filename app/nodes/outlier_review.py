"""HITL node: propose outlier-treatment candidates, ask the analyst, apply
the chosen one.

The candidate list and the recommendation are both built from real abex
calls on the actual metric column (not abstract method names) — the
recommendation itself is a deterministic rule over `MetricProfile`
(kind/skewness), not an LLM guess, so it doesn't second-guess live data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abex.data import outliers as abex_outliers

from app.nodes._hitl import ask_human
from app.state import ABTestState

SKEW_HIGH = 1.0


def _build_options(values: pd.Series, mask: pd.Series) -> list[dict]:
    options: list[dict] = []

    winsor = abex_outliers.winsorize(values)
    options.append({"method": "winsorize", "params": {"lower_q": 0.01, "upper_q": 0.99}, **_summary(winsor)})

    trimmed = abex_outliers.trim(values, mask)
    options.append({"method": "trim", "params": {}, **_summary(trimmed)})

    non_null = values.dropna()
    if len(non_null) >= 4:
        q1, q3 = np.percentile(non_null, [25, 75])
        iqr = q3 - q1
        lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        capped = abex_outliers.cap(values, lower=lower, upper=upper)
        options.append({"method": "cap", "params": {"lower": lower, "upper": upper}, **_summary(capped)})

    if (values.dropna() + 1.0 > 0).all():
        logged = abex_outliers.log_transform(values)
        options.append({"method": "log_transform", "params": {"offset": 1.0}, **_summary(logged)})

    options.append({"method": "none", "params": {}, "n_affected": 0, "share_affected": 0.0})
    return options


def _summary(result: abex_outliers.OutlierTreatmentResult) -> dict:
    return {"n_affected": result.n_affected, "share_affected": result.share_affected}


def _recommend(profile: dict) -> str:
    skew = abs(profile.get("skewness", 0.0))
    outlier_share = profile.get("outlier_share", 0.0)
    kind = profile.get("kind")

    if kind == "continuous" and skew > SKEW_HIGH and profile.get("zero_share", 0.0) < 0.5:
        return "log_transform" if skew > 2 * SKEW_HIGH else "winsorize"
    if outlier_share < 0.03:
        return "trim"
    return "winsorize"


def _apply(df: pd.DataFrame, metric_col: str, mask: pd.Series, decision: dict) -> tuple[pd.DataFrame, str | None]:
    method = decision["method"]
    params = decision.get("params", {})
    values = df[metric_col]

    if method == "none":
        return df, None
    if method == "winsorize":
        result = abex_outliers.winsorize(values, **params)
        treated_col = f"{metric_col}__treated"
        return df.assign(**{treated_col: result.treated}), treated_col
    if method == "trim":
        return df.loc[~mask].copy(), metric_col
    if method == "cap":
        result = abex_outliers.cap(values, **params)
        treated_col = f"{metric_col}__treated"
        return df.assign(**{treated_col: result.treated}), treated_col
    if method == "log_transform":
        result = abex_outliers.log_transform(values, **params)
        treated_col = f"{metric_col}__treated"
        return df.assign(**{treated_col: result.treated}), treated_col
    raise ValueError(f"unknown outlier treatment method: {method!r}")


def outlier_review_node(state: ABTestState) -> dict:
    df = state["raw_data"]
    metric_col = state["metric_col"]
    profile = state["metric_profile"]
    mask = pd.Series(state["outlier_mask"], index=df.index, dtype=bool)
    values = df[metric_col]

    options = _build_options(values, mask)
    recommendation = _recommend(profile)

    decision = ask_human(
        {
            "kind": "outlier_review",
            "metric_col": metric_col,
            "outlier_share": profile.get("outlier_share", 0.0),
            "options": options,
            "recommendation": recommendation,
        }
    )

    new_df, treated_col = _apply(df, metric_col, mask, decision)
    return {
        "raw_data": new_df,
        "outlier_options": options,
        "outlier_recommendation": recommendation,
        "outlier_decision": decision,
        "treated_metric_col": treated_col,
        "last_completed_step": "outlier_review",
    }
