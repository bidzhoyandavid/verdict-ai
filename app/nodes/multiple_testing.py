"""Multiple-comparisons correction — mandatory whenever more than one metric
was tested.

Without it, testing N metrics at alpha=0.05 inflates the family-wise error
rate, so a "significant" secondary metric is often noise. The correction is
Bonferroni per the agent spec; the raw p-values stay in the reports so the
table can show both.
"""

from __future__ import annotations

from dataclasses import asdict

from abex.stats.multiple_testing import bonferroni

from app.state import ABTestState

ALPHA = 0.05


def multiple_testing_node(state: ABTestState) -> dict:
    reports = list(state.get("test_results") or [])
    testable = [r for r in reports if isinstance(r.get("p_value"), (int, float))]

    if len(testable) < 2:
        # One metric — nothing to correct, and bonferroni() rejects an empty list.
        return {"multiple_testing_result": None, "last_completed_step": "multiple_testing"}

    correction = bonferroni([float(r["p_value"]) for r in testable], alpha=ALPHA)
    result = asdict(correction)
    result["metrics"] = [r["metric"] for r in testable]

    for report, adjusted, reject in zip(testable, correction.adjusted_p_values, correction.reject):
        report["adjusted_p_value"] = float(adjusted)
        report["significant_after_correction"] = bool(reject)
        if report.get("decision") == "significant" and not reject:
            report["decision"] = "not_significant_after_correction"
            report.setdefault("warnings", []).append(
                f"значимость пропала после поправки Бонферрони на {len(testable)} метрик"
            )

    return {
        "test_results": reports,
        "multiple_testing_result": result,
        "last_completed_step": "multiple_testing",
    }
