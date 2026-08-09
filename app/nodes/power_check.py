"""Turns "not significant" into either "no effect" or "need more data".

Without this the agent guesses. Here the question is answered concretely: to
detect an effect the size of the one actually observed, how many observations
per group would have been needed, and how far is that from what we have.

Runs only for the primary metric — the number is a decision aid for the next
test, not a per-metric report.
"""

from __future__ import annotations

from dataclasses import asdict

from abex.design.power import sample_size_mean, sample_size_proportion

from app.state import ABTestState

ALPHA = 0.05
TARGET_POWER = 0.8


def _primary_report(state: ABTestState) -> dict | None:
    reports = state.get("test_results") or []
    for report in reports:
        if report.get("is_primary"):
            return report
    return reports[0] if reports else None


def _primary_stats(state: ABTestState, metric: str) -> list[dict]:
    return [row for row in (state.get("group_stats") or []) if row.get("metric") == metric]


def power_check_node(state: ABTestState) -> dict:
    report = _primary_report(state)
    if not report or report.get("decision") == "significant":
        # Significant result — the sample was evidently sufficient.
        return {"power_result": None, "last_completed_step": "power_check"}

    metric = report.get("metric")
    stats = _primary_stats(state, str(metric))
    if len(stats) < 2:
        return {"power_result": None, "last_completed_step": "power_check"}

    effect = report.get("effect") or {}
    observed_diff = abs(float(effect.get("absolute_diff") or 0.0))
    if observed_diff == 0.0:
        return {"power_result": None, "last_completed_step": "power_check"}

    control, treatment = stats[0], stats[1]
    n_observed = min(int(control.get("n") or 0), int(treatment.get("n") or 0))
    conversion = control.get("conversion")

    try:
        if conversion is not None and 0 < conversion < 1:
            result = sample_size_proportion(
                baseline_rate=float(conversion),
                mde_abs=observed_diff,
                alpha=ALPHA,
                power=TARGET_POWER,
            )
        else:
            std = control.get("std")
            if not std or std <= 0:
                return {"power_result": None, "last_completed_step": "power_check"}
            result = sample_size_mean(
                std=float(std), mde_abs=observed_diff, alpha=ALPHA, power=TARGET_POWER
            )
    except ValueError:
        # Degenerate inputs (zero variance, non-positive MDE) — no honest answer.
        return {"power_result": None, "last_completed_step": "power_check"}

    required = result.sample_size_per_group
    payload = asdict(result)
    payload.update(
        {
            "metric": metric,
            "n_observed_per_group": n_observed,
            "required_per_group": required,
            "shortfall_per_group": max(required - n_observed, 0),
            "enough_data": n_observed >= required,
            "verdict": "no_effect" if n_observed >= required else "need_more_data",
        }
    )
    return {"power_result": payload, "last_completed_step": "power_check"}
