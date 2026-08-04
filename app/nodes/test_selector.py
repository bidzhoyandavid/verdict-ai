"""Test selector — thin wrapper over abex.selector.recommend_test, top-1
recommendation. Empty ranked list routes to Clarify instead of guessing."""

from __future__ import annotations

from dataclasses import asdict

from abex.data.profiling import MetricProfile
from abex.selector.recommend import recommend_test

from app.state import ABTestState


def test_selector_node(state: ABTestState, paired: bool = False) -> dict:
    profile_dict = state["metric_profile"]
    profile = MetricProfile(**profile_dict)
    ranked = recommend_test(profile, paired=paired)

    if not ranked:
        return {"recommendation": None, "needs_clarification": True, "last_completed_step": "test_selector"}

    top = ranked[0]
    return {"recommendation": asdict(top), "needs_clarification": False, "last_completed_step": "test_selector"}


def has_recommendation(state: ABTestState) -> bool:
    return state.get("recommendation") is not None
