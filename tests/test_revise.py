from app.revise import invalidate_from
from app.state import empty_state


def _mid_pipeline_state():
    state = empty_state("_template")
    state.update(
        dataset_id="ds-placeholder",
        group_col="group",
        metric_col="revenue",
        validation_report={"issues": []},
        metric_profile={"kind": "continuous"},
        outlier_mask=[False, True],
        outlier_options=[{"method": "winsorize"}],
        outlier_recommendation="winsorize",
        outlier_decision={"method": "winsorize"},
        treated_metric_col="revenue__treated",
        treated_dataset_id="ds-treated",
        srm_result={"has_srm": False},
        recommendation={"method_name": "t_test"},
        test_results=[{"decision": "significant"}],
        group_stats=[{"metric": "revenue", "group": "control", "n": 10}],
        multiple_testing_result={"method": "bonferroni"},
        guardrail_results=[{"metric": "latency"}],
        power_result={"verdict": "no_effect"},
        results_table=[{"metric": "revenue"}],
        charts=[{"kind": "distribution"}],
        timeline_warnings=["ok"],
        last_completed_step="guardrail",
    )
    return state


def test_revise_metric_col_clears_everything_downstream_of_validate():
    state = _mid_pipeline_state()
    revised = invalidate_from(state, "validate_profile")

    assert revised["dataset_id"] == "ds-placeholder"  # upstream untouched
    assert revised["group_col"] == "group"
    assert revised["validation_report"] is None
    assert revised["metric_profile"] is None
    assert revised["outlier_decision"] is None
    assert revised["treated_metric_col"] is None
    assert revised["treated_dataset_id"] is None
    assert revised["srm_result"] is None
    assert revised["recommendation"] is None
    assert revised["test_results"] == []
    assert revised["group_stats"] == []
    assert revised["multiple_testing_result"] is None
    assert revised["guardrail_results"] == []
    assert revised["results_table"] == []
    assert revised["charts"] == []
    assert revised["last_completed_step"] == "load"


def test_revise_outlier_review_keeps_validation_and_profile():
    state = _mid_pipeline_state()
    revised = invalidate_from(state, "outlier_review")

    assert revised["validation_report"] == {"issues": []}
    assert revised["metric_profile"] == {"kind": "continuous"}
    assert revised["outlier_decision"] is None
    assert revised["treated_metric_col"] is None
    assert revised["treated_dataset_id"] is None
    assert revised["srm_result"] is None
    assert revised["test_results"] == []
    assert revised["results_table"] == []
    # timeline_check стоит выше outlier_review — его вывод не сбрасывается
    assert revised["timeline_warnings"] == ["ok"]
    assert revised["last_completed_step"] == "timeline_check"


def test_revise_guardrail_clears_guardrail_and_everything_after():
    state = _mid_pipeline_state()
    revised = invalidate_from(state, "guardrail")

    assert revised["test_results"] == [{"decision": "significant"}]
    assert revised["multiple_testing_result"] == {"method": "bonferroni"}
    assert revised["guardrail_results"] == []
    # Таблица и графики строятся после guardrail — их надо пересобрать
    assert revised["power_result"] is None
    assert revised["results_table"] == []
    assert revised["charts"] == []
    assert revised["last_completed_step"] == "multiple_testing"


def test_revise_unknown_step_raises():
    state = _mid_pipeline_state()
    try:
        invalidate_from(state, "not_a_step")
        assert False, "expected ValueError"
    except ValueError:
        pass
