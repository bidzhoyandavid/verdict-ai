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


def _is_paired_design(df, group_col: str, id_col: str | None) -> bool:
    """True iff every id appears exactly once in every group — the "same
    unit measured under N conditions" shape validators.validate() otherwise
    flags as duplicated ids."""
    if id_col is None:
        return False
    counts = df.groupby(id_col)[group_col].nunique()
    n_groups = df[group_col].nunique()
    per_id_group_counts = df.groupby([id_col, group_col]).size()
    return bool(len(counts) > 0 and (counts == n_groups).all() and (per_id_group_counts == 1).all())


def validate_profile_node(state: ABTestState) -> dict:
    df = state["raw_data"]
    group_col = state["group_col"]
    metric_col = state["metric_col"]
    id_col = state.get("id_col")

    validation_report = validators.validate(df, group_col=group_col, metric_col=metric_col, id_col=id_col)
    metric_profile = profiling.profile_metric(df, metric_col=metric_col, group_col=group_col)
    mask = outliers.detect_outliers(df[metric_col], method="iqr")

    is_paired = _is_paired_design(df, group_col, id_col)
    issues = validation_report.issues
    if is_paired:
        issues = [i for i in issues if not i.startswith(f"{validation_report.n_duplicates} duplicated rows by id column")]
    report_dict = asdict(validation_report)
    report_dict["issues"] = issues

    return {
        "validation_report": report_dict,
        "metric_profile": asdict(metric_profile),
        "outlier_mask": mask.tolist(),
        "is_paired_design": is_paired,
        "last_completed_step": "validate_profile",
        "needs_clarification": len(issues) > 0,
    }


def needs_outlier_review(state: ABTestState) -> bool:
    profile = state.get("metric_profile") or {}
    return bool(profile) and profile.get("outlier_share", 0.0) > OUTLIER_SHARE_HITL_THRESHOLD
