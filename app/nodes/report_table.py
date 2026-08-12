"""Assembles the final results table — one row per metric.

Fully deterministic: every number here already exists in state, this node only
joins the per-metric reports with the per-group descriptive statistics. No LLM
touches the table, so the numbers a stakeholder reads cannot be paraphrased
into something else.

Columns: metric, control value, treatment value, absolute diff, relative diff,
p-value, 95% CI, significant (bool).
"""

from __future__ import annotations

from typing import Any

from app.state import ABTestState


def _group_stat(group_stats: list[dict], metric: str, group: str, key: str) -> float | None:
    for row in group_stats:
        if row.get("metric") == metric and row.get("group") == group:
            return row.get(key)
    return None


def _group_value(group_stats: list[dict], metric: str, group: str) -> float | None:
    """Conversion for binary metrics, mean otherwise — the number a human
    means when they ask "what was it in the control group"."""
    for row in group_stats:
        if row.get("metric") == metric and row.get("group") == group:
            conversion = row.get("conversion")
            return conversion if conversion is not None else row.get("mean")
    return None


def _group_n(group_stats: list[dict], metric: str, group: str) -> int | None:
    for row in group_stats:
        if row.get("metric") == metric and row.get("group") == group:
            return row.get("n")
    return None


def _is_significant(report: dict) -> bool | None:
    """After a correction was applied, only the corrected verdict counts."""
    if "significant_after_correction" in report:
        return bool(report["significant_after_correction"])
    decision = report.get("decision")
    if decision is None:
        return None
    return decision == "significant"


def report_table_node(state: ABTestState) -> dict:
    reports = state.get("test_results") or []
    group_stats = state.get("group_stats") or []
    srm = state.get("srm_result") or {}

    rows: list[dict[str, Any]] = []
    for report in reports:
        metric = report.get("metric")
        control_group = report.get("control_group")
        treatment_group = report.get("treatment_group")
        effect = report.get("effect") or {}
        ci = report.get("ci") or {}
        relative_lift = effect.get("relative_lift")

        # Доверительный интервал в долях от контроля: метрики разного масштаба
        # (выручка в тысячах и время в единицах) иначе несравнимы на одном графике.
        control_value = _group_value(group_stats, str(metric), str(control_group))
        # База нормировки должна быть той же величиной, что и в CI: делить
        # интервал для разницы медиан на среднее контроля — смешивать оценки.
        estimand = report.get("ci_estimand") or "mean_diff"
        ci_base = (
            _group_stat(group_stats, str(metric), str(control_group), "median")
            if estimand == "median_diff"
            else control_value
        )
        scale = abs(ci_base) if isinstance(ci_base, (int, float)) and ci_base else None
        relative_ci_low = ci.get("low") / scale if scale and ci.get("low") is not None else None
        relative_ci_high = ci.get("high") / scale if scale and ci.get("high") is not None else None

        rows.append(
            {
                "metric": metric,
                "is_primary": bool(report.get("is_primary")),
                "control_group": control_group,
                "treatment_group": treatment_group,
                "control_value": control_value,
                "treatment_value": _group_value(group_stats, str(metric), str(treatment_group)),
                "n_control": _group_n(group_stats, str(metric), str(control_group)),
                "n_treatment": _group_n(group_stats, str(metric), str(treatment_group)),
                "absolute_diff": effect.get("absolute_diff"),
                "relative_diff": relative_lift,
                "p_value": report.get("p_value"),
                "adjusted_p_value": report.get("adjusted_p_value"),
                "ci_low": ci.get("low"),
                "ci_high": ci.get("high"),
                "relative_ci_low": relative_ci_low,
                "relative_ci_high": relative_ci_high,
                "significant": _is_significant(report),
                "ci_estimand": estimand,
                "method": report.get("method"),
                "warnings": list(report.get("warnings") or []),
            }
        )

    # CI — часть решающего правила, а не иллюстрация. Интервал, накрывающий
    # ноль, означает «эффект не доказан»; оставить при этом significant=true
    # значит показать читателю два взаимоисключающих вывода рядом. Побеждает
    # более консервативный: значимость снимается.
    for row in rows:
        spans_zero = (
            row["ci_low"] is not None
            and row["ci_high"] is not None
            and row["ci_low"] <= 0 <= row["ci_high"]
        )
        if row["significant"] and spans_zero:
            row["significant"] = False
            row["warnings"].append(
                f"p-value = {row['p_value']:.4g} ниже порога, но 95% CI "
                f"[{row['ci_low']:.4g}; {row['ci_high']:.4g}] накрывает ноль — "
                "эффект на границе, значимость снята"
                if row.get("p_value") is not None
                else "95% CI накрывает ноль — эффект не доказан, значимость снята"
            )

    # SRM invalidates the experiment outright: leaving "significant: true" in
    # the table would be actively misleading.
    if srm.get("has_srm"):
        for row in rows:
            row["significant"] = False
            row["warnings"].append("SRM: распределение по группам нарушено, результат невалиден")

    rows.sort(key=lambda row: (not row["is_primary"], str(row["metric"])))
    return {"results_table": rows, "last_completed_step": "report_table"}
