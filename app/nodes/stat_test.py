"""Runs the selected stat test on every metric, plus effect size, a bootstrap
CI for the difference, and per-group descriptive statistics.

No stats logic lives here — this only wires abex functions together and builds
the reports. The CI comes from `abex.stats.bootstrap` because abex's
`TestResult` carries only a p-value, and the results table needs an interval.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from abex.analysis.effect_size import effect_size_summary
from abex.report import build_report
from abex.stats.bootstrap import BootstrapResult, bootstrap_ci
from abex.stats.frequentist import TestResult

from app.datasets import load_active
from app.groups import ordered_pair
from app.state import ABTestState, analysis_metric_cols, source_metric_col

ALPHA = 0.05
# Дефолт abex. Меньше — интервал заметно «плавает» между запусками и начинает
# расходиться с p-value на границе значимости.
CI_RESAMPLES = 10_000
CI_RANDOM_STATE = 0

# Ранговые тесты проверяют сдвиг распределения, а не разницу средних. Считать
# для них CI разницы средних — сравнивать разные величины: на скошенной метрике
# ранговый p-value уверенно значим, а интервал средних накрывает ноль.
RANK_BASED_METHODS = frozenset({"mann_whitney", "wilcoxon_signed_rank", "kruskal_wallis"})


def _import_fn(fn_path: str):
    module_path, fn_name = fn_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def _split_groups(df: pd.DataFrame, group_col: str, metric_col: str) -> dict[str, pd.Series]:
    return {str(g): sub[metric_col] for g, sub in df.groupby(group_col)}


def _split_groups_paired(
    df: pd.DataFrame, group_col: str, metric_col: str, id_col: str
) -> dict[str, pd.Series]:
    """Pivot to one row per id, one column per group, so `control`/`treatment`
    are aligned by id (required by paired tests like ttest_rel)."""
    wide = df.pivot_table(index=id_col, columns=group_col, values=metric_col, aggfunc="first")
    return {str(g): wide[g] for g in wide.columns}


def _describe(values: pd.Series) -> dict:
    clean = values.dropna()
    stats = {
        "n": int(len(clean)),
        "mean": float(clean.mean()) if len(clean) else None,
        "median": float(clean.median()) if len(clean) else None,
        "std": float(clean.std()) if len(clean) > 1 else None,
        "sum": float(clean.sum()) if len(clean) else None,
    }
    # A 0/1 metric reads as a conversion rate; anything else has no rate.
    unique = set(pd.unique(clean)) if len(clean) else set()
    stats["conversion"] = float(clean.mean()) if unique and unique <= {0, 1, 0.0, 1.0} else None
    return stats


def _mean_diff(control, treatment) -> float:
    return float(treatment.mean() - control.mean())


def _median_diff(control, treatment) -> float:
    return float(np.median(treatment) - np.median(control))


def ci_estimand(method_name: str) -> str:
    """Какую величину оценивает интервал — ту же, что проверяет тест."""
    return "median_diff" if method_name in RANK_BASED_METHODS else "mean_diff"


def _difference_ci(
    control: pd.Series, treatment: pd.Series, estimand: str = "mean_diff"
) -> tuple[float, float] | None:
    """95% bootstrap CI for the effect the chosen test actually tests.

    Bootstrap makes no distributional assumption, which matters because the
    selector may well have picked a test for a skewed metric.
    """
    if len(control.dropna()) < 2 or len(treatment.dropna()) < 2:
        return None
    result = bootstrap_ci(
        control,
        treatment,
        statistic=_median_diff if estimand == "median_diff" else _mean_diff,
        n_resamples=CI_RESAMPLES,
        alpha=ALPHA,
        random_state=CI_RANDOM_STATE,
    )
    return float(result.ci_low), float(result.ci_high)


def _analyze_metric(
    df: pd.DataFrame,
    state: ABTestState,
    metric_col: str,
    fn,
    recommendation: dict,
) -> tuple[dict, list[dict]]:
    group_col = state["group_col"]
    id_col = state.get("id_col")

    if state.get("is_paired_design") and id_col:
        groups = _split_groups_paired(df, group_col, metric_col, id_col)
    else:
        groups = _split_groups(df, group_col, metric_col)

    control_name, treatment_name = ordered_pair(state, list(groups.keys()))
    group_names = [control_name, treatment_name]
    control, treatment = groups[control_name], groups[treatment_name]

    display_name = source_metric_col(state, metric_col)
    group_stats = [
        {"metric": display_name, "group": name, **_describe(groups[name])} for name in group_names
    ]

    estimand = ci_estimand(recommendation["method_name"])
    raw_result = fn(control, treatment)
    if isinstance(raw_result, TestResult):
        p_value = raw_result.p_value
        ci = _difference_ci(control, treatment, estimand)
    elif isinstance(raw_result, BootstrapResult):
        p_value = None
        estimand = "mean_diff"
        ci = (float(raw_result.ci_low), float(raw_result.ci_high))
    else:
        raise TypeError(f"unsupported stat-test result type: {type(raw_result).__name__}")

    effect = effect_size_summary(control, treatment)
    report = build_report(
        metric=display_name,
        method=recommendation["method_name"],
        p_value=p_value,
        effect=effect,
        alpha=ALPHA,
        ci=ci,
        warnings=list(recommendation.get("warnings", [])),
    )
    report["ci_estimand"] = estimand
    report["control_group"] = control_name
    report["treatment_group"] = treatment_name
    report["is_primary"] = metric_col in (state.get("metric_col"), state.get("treated_metric_col"))
    return report, group_stats


def stat_test_node(state: ABTestState) -> dict:
    df = load_active(state)
    recommendation = state["recommendation"]
    fn = _import_fn(recommendation["fn_path"])

    reports: list[dict] = []
    group_stats: list[dict] = []
    for metric_col in analysis_metric_cols(state):
        if metric_col not in df.columns:
            continue
        try:
            report, stats = _analyze_metric(df, state, metric_col, fn, recommendation)
        except (ValueError, IndexError, KeyError) as exc:
            # One unusable secondary metric (constant, single group, all-null)
            # must not sink the whole run — record why and keep going.
            reports.append(
                {
                    "metric": source_metric_col(state, metric_col),
                    "method": recommendation["method_name"],
                    "p_value": None,
                    "effect": None,
                    "ci": None,
                    "ci_estimand": None,
                    "decision": None,
                    "warnings": [f"не удалось посчитать: {exc}"],
                    "is_primary": False,
                }
            )
            continue
        reports.append(report)
        group_stats.extend(stats)

    return {
        "test_results": reports,
        "group_stats": group_stats,
        "last_completed_step": "stat_test",
    }
