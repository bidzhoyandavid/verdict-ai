"""Ratio metrics (revenue per user, CTR as sum/sum) via linearization.

A ratio of sums is not a mean of per-row ratios, and a t-test on the naive
per-row ratio understates the variance. abex linearizes against a pooled R0
(Deng et al. 2018) so an ordinary two-sample test becomes valid; the reported
effect is still on the ratio itself.

Specs come from the test config — a ratio metric cannot be guessed from
column names, it has to be declared:
`{"name": str, "numerator": str, "denominator": str}`.
"""

from __future__ import annotations

from abex.analysis.effect_size import effect_size_summary
from abex.report import build_report
from abex.stats.frequentist import t_test
from abex.stats.ratio import linearize_groups, ratio_effect

from app.datasets import load_active
from app.groups import ordered_pair
from app.state import ABTestState

ALPHA = 0.05


def ratio_metrics_node(state: ABTestState) -> dict:
    specs = state.get("ratio_metric_specs") or []
    if not specs:
        return {"last_completed_step": "ratio_metrics"}

    df = load_active(state)
    group_col = state.get("group_col")
    if not group_col or group_col not in df.columns:
        return {"last_completed_step": "ratio_metrics"}

    group_names = [str(name) for name in df[group_col].dropna().unique()]
    if len(group_names) < 2:
        return {"last_completed_step": "ratio_metrics"}
    control_name, treatment_name = ordered_pair(state, group_names)

    frames = {str(name): sub for name, sub in df.groupby(group_col)}
    reports = list(state.get("test_results") or [])
    group_stats = list(state.get("group_stats") or [])

    for spec in specs:
        name = spec.get("name") or f"{spec.get('numerator')}/{spec.get('denominator')}"
        numerator, denominator = spec.get("numerator"), spec.get("denominator")
        if numerator not in df.columns or denominator not in df.columns:
            reports.append(
                {
                    "metric": name,
                    "method": "ratio_linearized_t_test",
                    "p_value": None,
                    "effect": None,
                    "ci": None,
                    "decision": None,
                    "is_ratio": True,
                    "is_primary": False,
                    "warnings": [f"нет колонок {numerator!r}/{denominator!r} в датасете"],
                }
            )
            continue

        control, treatment = frames[control_name], frames[treatment_name]
        try:
            linearized = linearize_groups(
                control[numerator], control[denominator], treatment[numerator], treatment[denominator]
            )
            effect = ratio_effect(
                control[numerator], control[denominator], treatment[numerator], treatment[denominator]
            )
            test = t_test(linearized.control_linearized, linearized.treatment_linearized)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            reports.append(
                {
                    "metric": name,
                    "method": "ratio_linearized_t_test",
                    "p_value": None,
                    "effect": None,
                    "ci": None,
                    "decision": None,
                    "is_ratio": True,
                    "is_primary": False,
                    "warnings": [f"не удалось посчитать ratio-метрику: {exc}"],
                }
            )
            continue

        # Значимость считается на линеаризованных значениях, а размер эффекта
        # сообщается на самом отношении — иначе цифра ничего не значит для бизнеса.
        report = build_report(
            metric=name,
            method="ratio_linearized_t_test",
            p_value=test.p_value,
            effect=effect_size_summary(
                linearized.control_linearized, linearized.treatment_linearized
            ),
            alpha=ALPHA,
            warnings=["значимость на линеаризованных значениях, эффект — на самом отношении"],
        )
        report["effect"]["absolute_diff"] = float(effect.absolute_diff)
        report["effect"]["relative_lift"] = float(effect.relative_lift)
        report["control_group"] = control_name
        report["treatment_group"] = treatment_name
        report["is_ratio"] = True
        report["is_primary"] = False
        reports.append(report)

        group_stats.extend(
            [
                {
                    "metric": name,
                    "group": control_name,
                    "n": int(len(control)),
                    "mean": float(effect.control_ratio),
                    "std": None,
                    "sum": None,
                    "conversion": None,
                },
                {
                    "metric": name,
                    "group": treatment_name,
                    "n": int(len(treatment)),
                    "mean": float(effect.treatment_ratio),
                    "std": None,
                    "sum": None,
                    "conversion": None,
                },
            ]
        )

    return {
        "test_results": reports,
        "group_stats": group_stats,
        "last_completed_step": "ratio_metrics",
    }
