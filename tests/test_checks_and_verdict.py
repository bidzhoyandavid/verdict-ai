"""Панель проверок и итоговый вердикт — обе части детерминированные."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.datasets import put_dataframe
from app.nodes.assumption_checks import assumption_checks_node
from app.nodes.checks_summary import checks_summary_node
from app.nodes.verdict import verdict_node
from app.state import empty_state


def _status(checks: list[dict], name: str) -> str:
    return next(check["status"] for check in checks if check["name"] == name)


def _skewed_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "group": ["control"] * 200 + ["treatment"] * 200,
            "user_id": range(400),
            "revenue": np.concatenate([rng.lognormal(3, 1.2, 200), rng.lognormal(3, 1.2, 200)]),
            "age": np.concatenate([rng.normal(30, 5, 200), rng.normal(30, 5, 200)]),
        }
    )


def test_assumption_checks_detect_non_normal_metric():
    df = _skewed_dataset()
    state = empty_state("_test")
    state.update(
        dataset_id=put_dataframe(df, company_id="_test"),
        group_col="group",
        metric_col="revenue",
        metric_cols=["revenue"],
        id_col="user_id",
    )

    results = assumption_checks_node(state)["assumption_results"]

    assert len(results["normality"]) == 2
    assert all(row["is_normal"] is False for row in results["normality"])
    assert results["equal_variance"]["is_equal"] is not None
    # age — не метрика, значит кандидат в ковариаты
    assert [row["covariate"] for row in results["covariate_balance"]] == ["age"]


def test_checks_summary_reports_every_check():
    state = empty_state("_test")
    state.update(
        validation_report={"n_rows": 400, "n_missing_metric": 0, "n_duplicates": 0, "issues": []},
        metric_profile={"kind": "continuous", "skewness": 0.2, "outlier_share": 0.0, "zero_share": 0.0, "is_balanced_design": True},
        srm_result={"has_srm": False, "p_value": 0.5, "observed_counts": {"a": 200, "b": 200}},
        recommendation={"method_name": "t_test", "violated_assumptions": [], "warnings": []},
        assumption_results={
            "normality": [{"group": "a", "is_normal": True, "p_value": 0.4, "method": "shapiro"}],
            "equal_variance": {"is_equal": True, "p_value": 0.6, "method": "levene(median)"},
            "covariate_balance": [{"covariate": "age", "is_balanced": True}],
        },
        test_results=[{"metric": "revenue"}],
        guardrail_results=[],
        timeline_warnings=["Тест длился 20.0 дн., грубых проблем с временной шкалой не видно."],
    )

    checks = checks_summary_node(state)["checks"]
    names = [check["name"] for check in checks]

    assert "Валидация данных" in names
    assert "Профиль метрики" in names
    assert "SRM (баланс групп)" in names
    assert "Нормальность распределения" in names
    assert "Баланс ковариат" in names
    assert _status(checks, "SRM (баланс групп)") == "ok"
    assert _status(checks, "Множественные сравнения") == "skipped"
    assert _status(checks, "Guardrail-метрики") == "skipped"


def test_srm_makes_the_experiment_invalid():
    state = empty_state("_test")
    state.update(
        srm_result={"has_srm": True},
        results_table=[{"metric": "revenue", "is_primary": True, "significant": False}],
        checks=[{"name": "SRM (баланс групп)", "status": "failed", "detail": "перекос"}],
    )

    result = verdict_node(state)["verdict"]

    assert result["code"] == "invalid"
    assert "не принимать решение" in result["action"].lower()


def test_negative_significant_effect_is_a_loser_not_a_winner():
    state = empty_state("_test")
    state["results_table"] = [
        {"metric": "revenue", "is_primary": True, "significant": True, "relative_diff": -0.07}
    ]

    result = verdict_node(state)["verdict"]

    assert result["code"] == "loser"
    assert "не раскатывать" in result["action"].lower()


def test_need_more_data_verdict_quotes_the_required_sample():
    state = empty_state("_test")
    state["results_table"] = [
        {"metric": "revenue", "is_primary": True, "significant": False, "relative_diff": 0.01}
    ]
    state["power_result"] = {
        "verdict": "need_more_data",
        "required_per_group": 5000,
        "n_observed_per_group": 200,
    }

    result = verdict_node(state)["verdict"]

    assert result["code"] == "need_more_data"
    assert "5000" in result["action"]


def test_failed_guardrail_blocks_a_winner():
    state = empty_state("_test")
    state["results_table"] = [
        {"metric": "revenue", "is_primary": True, "significant": True, "relative_diff": 0.1}
    ]
    state["checks"] = [{"name": "Guardrail-метрики", "status": "failed", "detail": "нарушены: latency"}]

    result = verdict_node(state)["verdict"]

    assert result["code"] == "inconclusive"
    assert result["blocking_checks"] == ["Guardrail-метрики"]


def test_charts_are_actually_produced():
    """Нода графиков глушит ошибки рендера — без этого теста регрессия в
    abex.viz проходит незаметно и анализ молча остаётся без картинок."""
    from app.nodes.charts import charts_node

    df = _skewed_dataset()
    state = empty_state("_test")
    state.update(
        dataset_id=put_dataframe(df, company_id="_test"),
        group_col="group",
        metric_col="revenue",
        metric_cols=["revenue"],
        results_table=[
            {
                "metric": "revenue",
                "is_primary": True,
                "absolute_diff": 1.0,
                "relative_diff": 0.05,
                "relative_ci_low": 0.01,
                "relative_ci_high": 0.09,
                "significant": True,
            }
        ],
    )

    charts = charts_node(state)["charts"]
    kinds = [chart["kind"] for chart in charts]

    assert "distribution" in kinds
    assert "confidence_interval" in kinds
    assert all(chart["data"] for chart in charts)
