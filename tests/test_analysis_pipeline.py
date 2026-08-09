"""Тесты детерминированной части пайплайна: мультиметрики, поправка,
мощность и сборка итоговой таблицы. LLM здесь не участвует.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.datasets import put_dataframe
from app.nodes.multiple_testing import multiple_testing_node
from app.nodes.power_check import power_check_node
from app.nodes.report_table import report_table_node
from app.nodes.stat_test import stat_test_node
from app.nodes.timeline_check import timeline_check_node
from app.state import empty_state


def _dataset(n: int = 400, lift: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame(
        {
            "group": ["control"] * half + ["treatment"] * half,
            "user_id": range(n),
            "revenue": np.concatenate([rng.normal(100, 15, half), rng.normal(100 + lift, 15, half)]),
            "orders": np.concatenate([rng.poisson(2, half), rng.poisson(2, half)]),
            "ts": pd.date_range("2026-01-01", periods=n, freq="h"),
        }
    )


def _state_with(df: pd.DataFrame, metrics: list[str]) -> dict:
    state = empty_state("_test")
    state.update(
        dataset_id=put_dataframe(df, company_id="_test"),
        group_col="group",
        metric_col=metrics[0],
        metric_cols=metrics,
        id_col="user_id",
        timestamp_col="ts",
        recommendation={"method_name": "t_test", "fn_path": "abex.stats.frequentist.t_test", "warnings": []},
    )
    return state


def test_stat_test_runs_every_metric_and_collects_group_stats():
    state = _state_with(_dataset(), ["revenue", "orders"])

    result = stat_test_node(state)

    assert [r["metric"] for r in result["test_results"]] == ["revenue", "orders"]
    assert all(r["ci"] is not None for r in result["test_results"])
    # Две метрики × две группы
    assert len(result["group_stats"]) == 4
    control_revenue = next(
        row for row in result["group_stats"] if row["metric"] == "revenue" and row["group"] == "control"
    )
    assert control_revenue["n"] == 200
    assert control_revenue["mean"] == pytest.approx(100, abs=3)


def test_single_metric_needs_no_correction():
    state = empty_state("_test")
    state["test_results"] = [{"metric": "revenue", "p_value": 0.01, "decision": "significant"}]

    result = multiple_testing_node(state)

    assert result["multiple_testing_result"] is None


def test_bonferroni_can_revoke_significance():
    state = empty_state("_test")
    state["test_results"] = [
        {"metric": "a", "p_value": 0.03, "decision": "significant", "warnings": []},
        {"metric": "b", "p_value": 0.04, "decision": "significant", "warnings": []},
        {"metric": "c", "p_value": 0.5, "decision": "not_significant", "warnings": []},
    ]

    result = multiple_testing_node(state)

    corrected = {r["metric"]: r for r in result["test_results"]}
    assert corrected["a"]["adjusted_p_value"] == pytest.approx(0.09)
    # 0.03 проходило порог в одиночку, но не после поправки на три метрики
    assert corrected["a"]["significant_after_correction"] is False
    assert corrected["a"]["decision"] == "not_significant_after_correction"
    assert result["multiple_testing_result"]["method"] == "bonferroni"


def test_power_check_says_need_more_data_for_a_tiny_effect():
    state = empty_state("_test")
    state["test_results"] = [
        {
            "metric": "revenue",
            "is_primary": True,
            "decision": "not_significant",
            "effect": {"absolute_diff": 0.5},
        }
    ]
    state["group_stats"] = [
        {"metric": "revenue", "group": "control", "n": 200, "mean": 100.0, "std": 15.0, "conversion": None},
        {"metric": "revenue", "group": "treatment", "n": 200, "mean": 100.5, "std": 15.0, "conversion": None},
    ]

    result = power_check_node(state)["power_result"]

    assert result["verdict"] == "need_more_data"
    assert result["required_per_group"] > result["n_observed_per_group"]
    assert result["shortfall_per_group"] > 0


def test_power_check_skipped_for_significant_result():
    state = empty_state("_test")
    state["test_results"] = [{"metric": "revenue", "is_primary": True, "decision": "significant"}]

    assert power_check_node(state)["power_result"] is None


def test_results_table_has_the_agreed_columns():
    state = empty_state("_test")
    state["test_results"] = [
        {
            "metric": "revenue",
            "is_primary": True,
            "method": "t_test",
            "p_value": 0.02,
            "decision": "significant",
            "effect": {"absolute_diff": 5.0, "relative_lift": 0.05},
            "ci": {"low": 1.0, "high": 9.0},
            "control_group": "control",
            "treatment_group": "treatment",
            "warnings": [],
        }
    ]
    state["group_stats"] = [
        {"metric": "revenue", "group": "control", "n": 200, "mean": 100.0, "conversion": None},
        {"metric": "revenue", "group": "treatment", "n": 200, "mean": 105.0, "conversion": None},
    ]

    row = report_table_node(state)["results_table"][0]

    assert row["metric"] == "revenue"
    assert row["control_value"] == 100.0
    assert row["treatment_value"] == 105.0
    assert row["absolute_diff"] == 5.0
    assert row["relative_diff"] == 0.05
    assert row["p_value"] == 0.02
    assert (row["ci_low"], row["ci_high"]) == (1.0, 9.0)
    assert row["significant"] is True


def test_srm_forces_every_row_to_insignificant():
    state = empty_state("_test")
    state["srm_result"] = {"has_srm": True}
    state["test_results"] = [
        {
            "metric": "revenue",
            "p_value": 0.001,
            "decision": "significant",
            "effect": {"absolute_diff": 5.0, "relative_lift": 0.05},
            "ci": {"low": 1.0, "high": 9.0},
            "control_group": "control",
            "treatment_group": "treatment",
            "warnings": [],
        }
    ]

    row = report_table_node(state)["results_table"][0]

    assert row["significant"] is False
    assert any("SRM" in w for w in row["warnings"])


def test_timeline_check_flags_a_short_test():
    df = _dataset(n=48)  # 48 часов = 2 дня
    state = _state_with(df, ["revenue"])

    warnings = timeline_check_node(state)["timeline_warnings"]

    assert any("меньше недели" in w for w in warnings)
    assert any("новизны" in w for w in warnings)


def test_timeline_check_without_timestamp_says_so():
    state = _state_with(_dataset(), ["revenue"])
    state["timestamp_col"] = None

    warnings = timeline_check_node(state)["timeline_warnings"]

    assert "нет колонки со временем" in warnings[0]


def test_numeric_column_with_time_in_the_name_is_not_a_timestamp():
    """`..._time_hour` — метрика, а не временная шкала: to_datetime молча
    прочитал бы числа как наносекунды и дал бы «тест длился 0 дней»."""
    from app.nodes.data_loader import detect_timestamp_col

    df = pd.DataFrame({"exploitation_time_hour": [1.5, 2.0, 3.25], "group": ["a", "b", "a"]})

    assert detect_timestamp_col(df) is None


def test_real_timestamp_column_is_detected():
    from app.nodes.data_loader import detect_timestamp_col

    df = pd.DataFrame({"event_date": ["2026-01-01", "2026-01-02", "2026-01-03"], "value": [1, 2, 3]})

    assert detect_timestamp_col(df) == "event_date"


def test_metric_cols_exclude_structural_columns():
    from app.nodes.data_loader import detect_metric_cols

    df = _dataset(n=10)
    metrics = detect_metric_cols(df, primary="revenue", group_col="group", id_col="user_id", timestamp_col="ts")

    assert metrics[0] == "revenue"
    assert "user_id" not in metrics
    assert "orders" in metrics


def test_table_flags_significance_that_contradicts_the_interval():
    """p<alpha при CI, накрывающем ноль, — граничный случай, о котором
    читателя таблицы надо предупредить явно."""
    state = empty_state("_test")
    state["test_results"] = [
        {
            "metric": "revenue",
            "p_value": 0.02,
            "decision": "significant",
            "effect": {"absolute_diff": 0.09, "relative_lift": 0.0006},
            "ci": {"low": -0.76, "high": 0.88},
            "control_group": "control",
            "treatment_group": "treatment",
            "warnings": [],
        }
    ]

    row = report_table_node(state)["results_table"][0]

    assert row["significant"] is True
    assert any("CI накрывает ноль" in w for w in row["warnings"])


def test_new_vs_old_is_not_decided_alphabetically():
    """«new_model» < «old_model» по алфавиту, но контроль здесь — old_model.
    Иначе знак эффекта и вердикт переворачиваются."""
    from app.groups import resolve_groups

    result = resolve_groups(["new_model", "old_model"])

    assert result["control"] == "old_model"
    assert result["treatment"] == "new_model"
    assert result["ambiguous"] is False


def test_control_and_variant_labels_are_recognised():
    from app.groups import resolve_groups

    assert resolve_groups(["variant", "control"])["control"] == "control"
    assert resolve_groups(["B", "A"])["control"] == "A"
    assert resolve_groups(["тест", "контроль"])["control"] == "контроль"


def test_explicit_config_beats_the_heuristic():
    from app.groups import resolve_groups

    result = resolve_groups(["new_model", "old_model"], control_hint="new_model")

    assert result["control"] == "new_model"
    assert result["treatment"] == "old_model"
    assert "настройках" in result["reason"]


def test_unrecognisable_labels_are_flagged_as_ambiguous():
    from app.groups import resolve_groups

    result = resolve_groups(["segment_x", "segment_y"])

    assert result["ambiguous"] is True
    assert "перевёрнут" in result["reason"]


def test_effect_direction_follows_the_assignment():
    """Проверяем сквозняком: эффект считается как вариант минус контроль."""
    df = _dataset(n=400, lift=5.0)
    df["group"] = df["group"].map({"control": "old_model", "treatment": "new_model"})
    state = _state_with(df, ["revenue"])
    state["group_assignment"] = {"control": "old_model", "treatment": "new_model"}

    report = stat_test_node(state)["test_results"][0]

    assert report["control_group"] == "old_model"
    assert report["treatment_group"] == "new_model"
    # В данных new_model на +5 выше, значит эффект должен быть положительным
    assert report["effect"]["absolute_diff"] > 0
