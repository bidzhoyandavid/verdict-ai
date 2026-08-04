"""LangChain tool wrappers — thin, no statistics of their own. 1:1 with the
"Tools" section of ab_test_agent.md. Most pipeline nodes call abex directly
(no LLM tool-calling needed for deterministic steps); these exist for the
LLM-driven paths (column detection fallback, ad-hoc chat requests for
plots) where an actual tool-calling loop is used.

abex.viz.* is currently a stub (NotImplementedError) — plotting here is
implemented directly with plotly instead of delegating to it.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from abex.analysis import effect_size, guardrails
from abex.data import outliers, profiling, validators
from abex.design import srm
from abex.report import build_report as _build_report
from abex.selector.recommend import recommend_test
from abex.stats import multiple_testing
from langchain_core.tools import tool


@tool
def validate_dataset_tool(records: list[dict], group_col: str, metric_col: str, id_col: str | None = None) -> dict:
    """Validate a dataset (rows as records) for the given group/metric/id columns."""
    df = pd.DataFrame.from_records(records)
    return asdict(validators.validate(df, group_col=group_col, metric_col=metric_col, id_col=id_col))


@tool
def profile_metric_tool(records: list[dict], metric_col: str, group_col: str) -> dict:
    """Profile a metric column (kind, skew, outlier share, design balance)."""
    df = pd.DataFrame.from_records(records)
    return asdict(profiling.profile_metric(df, metric_col=metric_col, group_col=group_col))


@tool
def check_srm_tool(group_counts: dict[str, int], expected_ratios: dict[str, float] | None = None) -> dict:
    """Check for sample ratio mismatch given observed group counts."""
    return asdict(srm.check_srm(group_counts, expected_ratios=expected_ratios))


@tool
def recommend_test_tool(profile: dict, paired: bool = False) -> list[dict]:
    """Rank candidate statistical tests for a given metric profile dict."""
    ranked = recommend_test(profiling.MetricProfile(**profile), paired=paired)
    return [asdict(r) for r in ranked]


@tool
def compute_effect_size_tool(control: list[float], treatment: list[float]) -> dict:
    """Compute absolute diff, relative lift, and Cohen's d between two samples."""
    return asdict(effect_size.effect_size_summary(pd.Series(control), pd.Series(treatment)))


@tool
def check_guardrail_tool(
    control: list[float],
    treatment: list[float],
    metric_name: str,
    max_allowed_degradation: float,
    higher_is_better: bool = True,
) -> dict:
    """Check whether a guardrail metric degraded beyond an allowed tolerance."""
    result = guardrails.check_guardrail(
        pd.Series(control), pd.Series(treatment), metric_name, max_allowed_degradation, higher_is_better
    )
    return asdict(result)


@tool
def correct_multiple_tool(p_values: list[float], alpha: float = 0.05) -> dict:
    """Bonferroni-correct a list of p-values for multiple comparisons."""
    return asdict(multiple_testing.bonferroni(p_values, alpha=alpha))


@tool
def build_report_tool(
    metric: str,
    method: str,
    p_value: float | None,
    effect: dict,
    alpha: float = 0.05,
    ci: tuple[float, float] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    """Assemble the JSON report contract from an effect-size dict."""
    return _build_report(
        metric=metric,
        method=method,
        p_value=p_value,
        effect=effect_size.EffectSizeResult(**effect),
        alpha=alpha,
        ci=ci,
        warnings=warnings,
    )


@tool
def plot_distribution(records: list[dict], metric_col: str, group_col: str) -> go.Figure:
    """Plot the metric's distribution split by group (histogram + box)."""
    df = pd.DataFrame.from_records(records)
    return px.histogram(df, x=metric_col, color=group_col, marginal="box", barmode="overlay", opacity=0.6)


@tool
def plot_confidence_interval(point_estimate: float, ci_low: float, ci_high: float, label: str = "effect") -> go.Figure:
    """Plot a point estimate with its confidence interval as an error bar."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[label],
            y=[point_estimate],
            error_y=dict(type="data", array=[ci_high - point_estimate], arrayminus=[point_estimate - ci_low]),
            mode="markers",
            marker=dict(size=12),
        )
    )
    return fig


@tool
def detect_outliers_tool(values: list[float], method: str = "iqr", threshold: float = 1.5) -> list[bool]:
    """Flag outliers in a list of values using the IQR/MAD/zscore rule."""
    mask = outliers.detect_outliers(pd.Series(values), method=method, threshold=threshold)
    return mask.tolist()
