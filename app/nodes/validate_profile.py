"""Validate + profile node — deterministic, no LLM.

Computes the ValidationReport, MetricProfile, and an outlier mask (default
IQR rule) so `outlier_review` can build treatment candidates without
re-deriving detection logic.
"""

from __future__ import annotations

from dataclasses import asdict

from abex.data import outliers, profiling, validators

from app.state import ABTestState

OUTLIER_SHARE_HITL_THRESHOLD = 0.01


def validate_profile_node(state: ABTestState) -> dict:
    df = state["raw_data"]
    group_col = state["group_col"]
    metric_col = state["metric_col"]
    id_col = state.get("id_col")

    validation_report = validators.validate(df, group_col=group_col, metric_col=metric_col, id_col=id_col)
    metric_profile = profiling.profile_metric(df, metric_col=metric_col, group_col=group_col)
    mask = outliers.detect_outliers(df[metric_col], method="iqr")

    return {
        "validation_report": asdict(validation_report),
        "metric_profile": asdict(metric_profile),
        "outlier_mask": mask.tolist(),
        "last_completed_step": "validate_profile",
        "needs_clarification": not validation_report.is_clean,
    }


def needs_outlier_review(state: ABTestState) -> bool:
    profile = state.get("metric_profile") or {}
    return bool(profile) and profile.get("outlier_share", 0.0) > OUTLIER_SHARE_HITL_THRESHOLD
