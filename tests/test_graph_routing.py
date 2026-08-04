from app.nodes.srm_gate import has_srm
from app.nodes.test_selector import has_recommendation
from app.nodes.validate_profile import needs_outlier_review
from app.state import empty_state, next_step


def test_srm_stop_condition():
    state = empty_state()
    state["srm_result"] = {"has_srm": True}
    assert has_srm(state) is True

    state["srm_result"] = {"has_srm": False}
    assert has_srm(state) is False


def test_empty_selector_routes_to_clarify():
    state = empty_state()
    state["recommendation"] = None
    assert has_recommendation(state) is False

    state["recommendation"] = {"method_name": "t_test"}
    assert has_recommendation(state) is True


def test_low_outlier_share_skips_outlier_review():
    state = empty_state()
    state["metric_profile"] = {"outlier_share": 0.001}
    assert needs_outlier_review(state) is False

    state["metric_profile"] = {"outlier_share": 0.05}
    assert needs_outlier_review(state) is True


def test_next_step_resumes_after_last_completed():
    state = empty_state()
    assert next_step(state) == "load"

    state["last_completed_step"] = "validate_profile"
    assert next_step(state) == "outlier_review"

    state["last_completed_step"] = "guardrail"
    assert next_step(state) == "insight"

    state["last_completed_step"] = "insight"
    assert next_step(state) == "insight"  # clamps at the last step
