from app.revise import invalidate_from
from app.state import empty_state


def _mid_pipeline_state():
    state = empty_state("_template")
    state.update(
        raw_data="df-placeholder",
        group_col="group",
        metric_col="revenue",
        validation_report={"issues": []},
        metric_profile={"kind": "continuous"},
        outlier_mask=[False, True],
        outlier_options=[{"method": "winsorize"}],
        outlier_recommendation="winsorize",
        outlier_decision={"method": "winsorize"},
        treated_metric_col="revenue__treated",
        srm_result={"has_srm": False},
        recommendation={"method_name": "t_test"},
        test_result={"decision": "significant"},
        guardrail_results=[{"metric": "latency"}],
        last_completed_step="guardrail",
    )
    return state


def test_revise_metric_col_clears_everything_downstream_of_validate():
    state = _mid_pipeline_state()
    revised = invalidate_from(state, "validate_profile")

    assert revised["raw_data"] == "df-placeholder"  # upstream untouched
    assert revised["group_col"] == "group"
    assert revised["validation_report"] is None
    assert revised["metric_profile"] is None
    assert revised["outlier_decision"] is None
    assert revised["treated_metric_col"] is None
    assert revised["srm_result"] is None
    assert revised["recommendation"] is None
    assert revised["test_result"] is None
    assert revised["guardrail_results"] == []
    assert revised["last_completed_step"] == "load"


def test_revise_outlier_review_keeps_validation_and_profile():
    state = _mid_pipeline_state()
    revised = invalidate_from(state, "outlier_review")

    assert revised["validation_report"] == {"issues": []}
    assert revised["metric_profile"] == {"kind": "continuous"}
    assert revised["outlier_decision"] is None
    assert revised["treated_metric_col"] is None
    assert revised["srm_result"] is None
    assert revised["test_result"] is None
    assert revised["last_completed_step"] == "validate_profile"


def test_revise_guardrail_only_clears_guardrail_results():
    state = _mid_pipeline_state()
    revised = invalidate_from(state, "guardrail")

    assert revised["test_result"] == {"decision": "significant"}
    assert revised["guardrail_results"] == []
    assert revised["last_completed_step"] == "stat_test"


def test_revise_unknown_step_raises():
    state = _mid_pipeline_state()
    try:
        invalidate_from(state, "not_a_step")
        assert False, "expected ValueError"
    except ValueError:
        pass
