"""Builds the figures the UI renders.

Thin orchestration over `abex.viz` — the agent does not draw anything itself.
Figures leave as plotly JSON specs so the frontend renders them client-side
and the state stays serializable.

A failed chart is never fatal: losing a picture must not lose an analysis — but
it is logged at WARNING, because silently returning no charts is exactly how a
rendering regression goes unnoticed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from abex.viz import plot_confidence_interval, plot_distribution, plot_timeseries

from app.datasets import load_active, load_dataset
from app.state import ABTestState, active_metric_col, source_metric_col

logger = logging.getLogger(__name__)

# Больше — уже не помогает читать, а вес состояния растёт линейно.
MAX_DISTRIBUTION_CHARTS = 3


def _spec(figure: Any) -> dict:
    """plotly Figure -> plain dict, via its own JSON encoder (it knows about
    numpy and datetimes; a bare dict() does not)."""
    return json.loads(figure.to_json())


def _chart(kind: str, title: str, figure: Any) -> dict:
    return {"kind": kind, "title": title, **_spec(figure)}


def charts_node(state: ABTestState) -> dict:
    df = load_active(state)
    group_col = state.get("group_col")
    if not group_col:
        return {"charts": [], "last_completed_step": "charts"}

    charts: list[dict] = []
    primary = active_metric_col(state)

    metrics = [row["metric"] for row in (state.get("results_table") or [])][:MAX_DISTRIBUTION_CHARTS]
    if not metrics and primary:
        metrics = [source_metric_col(state, primary)]

    for metric in metrics:
        column = primary if metric == source_metric_col(state, str(primary)) else metric
        if column not in df.columns:
            continue
        try:
            charts.append(
                _chart(
                    "distribution",
                    f"Распределение: {metric}",
                    plot_distribution(df, str(column), group_col),
                )
            )
        except (ValueError, TypeError) as exc:
            logger.warning("distribution chart skipped for %s: %s", metric, exc)

    # Относительный эффект, а не абсолютный: иначе метрика в тысячах единиц
    # растягивает ось и остальные строки схлопываются в ноль.
    estimates = [
        {
            "metric": row["metric"],
            "estimate": row.get("relative_diff"),
            "ci_low": row.get("relative_ci_low"),
            "ci_high": row.get("relative_ci_high"),
            "significant": row.get("significant"),
        }
        for row in (state.get("results_table") or [])
    ]
    if estimates:
        try:
            charts.append(
                _chart(
                    "confidence_interval",
                    "Относительный эффект и 95% CI",
                    plot_confidence_interval(estimates, title="Относительный эффект и 95% CI"),
                )
            )
        except (ValueError, TypeError) as exc:
            logger.warning("CI chart skipped: %s", exc)

    timestamp_col = state.get("timestamp_col")
    if timestamp_col and primary:
        try:
            charts.append(
                _chart(
                    "timeseries",
                    f"Динамика: {source_metric_col(state, primary)}",
                    plot_timeseries(df, str(primary), group_col, timestamp_col),
                )
            )
        except (ValueError, TypeError) as exc:
            logger.warning("timeseries chart skipped: %s", exc)

    return {"charts": charts, "last_completed_step": "charts"}


def outlier_chart(state: ABTestState) -> dict | None:
    """Before/after picture for the HITL outlier step, built on demand."""
    from abex.viz import plot_outlier_treatment

    base_id, treated_id = state.get("dataset_id"), state.get("treated_dataset_id")
    metric_col, treated_col = state.get("metric_col"), state.get("treated_metric_col")
    if not base_id or not treated_id or not metric_col or not treated_col:
        return None

    try:
        original = load_dataset(base_id)[metric_col]
        treated = load_dataset(treated_id)[treated_col]
        method = (state.get("outlier_decision") or {}).get("method", "")
        return _chart("outliers", "Обработка выбросов", plot_outlier_treatment(original, treated, method))
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        logger.warning("outlier chart skipped: %s", exc)
        return None
