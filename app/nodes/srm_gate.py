"""SRM gate — deterministic, strict alpha. If it fires, the experiment is
invalid and the graph should route straight to Insight, skipping the rest
of the stat pipeline."""

from __future__ import annotations

from dataclasses import asdict

from abex.design import srm

from app.state import ABTestState

SRM_ALPHA = 0.001


def srm_gate_node(state: ABTestState) -> dict:
    df = state["raw_data"]
    group_col = state["group_col"]
    counts = df[group_col].value_counts().to_dict()

    result = srm.check_srm({str(k): int(v) for k, v in counts.items()}, alpha=SRM_ALPHA)
    return {"srm_result": asdict(result), "last_completed_step": "srm_gate"}


def has_srm(state: ABTestState) -> bool:
    result = state.get("srm_result") or {}
    return bool(result.get("has_srm", False))
