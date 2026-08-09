"""Detects column roles on the dataset referenced by the state.

Detection is heuristic-first (name matching), LLM only as a fallback when
heuristics are ambiguous — keeps the common case free of an LLM round trip.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.datasets import load_active
from app.state import ABTestState

_GROUP_NAME_HINTS = ("group", "variant", "arm", "bucket", "cohort", "treatment")
_METRIC_NAME_HINTS = ("metric", "value", "revenue", "conversion", "count", "duration", "amount")
_ID_NAME_HINTS = ("user_id", "userid", "id", "uid", "customer_id")


def _heuristic_match(columns: list[str], hints: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for hint in hints:
        for lower_name, original in lowered.items():
            if hint in lower_name:
                return original
    return None


class ColumnRoles(BaseModel):
    group_col: str = Field(description="Column holding the experiment group/variant label")
    metric_col: str = Field(description="Column holding the metric to analyze")
    id_col: str | None = Field(default=None, description="Unique entity id column, if any")


def detect_column_roles(df: pd.DataFrame, llm: Any | None = None) -> dict:
    """Heuristic column-role detection, with an optional LLM fallback for
    ambiguous cases (heuristics found nothing for group/metric)."""
    columns = list(df.columns)
    group_col = _heuristic_match(columns, _GROUP_NAME_HINTS)
    metric_col = _heuristic_match(columns, _METRIC_NAME_HINTS)
    id_col = _heuristic_match(columns, _ID_NAME_HINTS)

    if (group_col is None or metric_col is None) and llm is not None:
        structured_llm = llm.with_structured_output(ColumnRoles)
        prompt = (
            "Dataframe columns: "
            f"{columns}. First rows:\n{df.head(5).to_string()}\n\n"
            "Pick the column holding the experiment group/variant label, the "
            "column holding the metric to analyze, and (if present) a unique "
            "entity id column. Only use column names that appear in the list above."
        )
        result: ColumnRoles = structured_llm.invoke(prompt)
        group_col = group_col or result.group_col
        metric_col = metric_col or result.metric_col
        id_col = id_col or result.id_col

    return {"group_col": group_col, "metric_col": metric_col, "id_col": id_col}


def load_node(state: ABTestState, llm: Any | None = None) -> dict:
    """LangGraph node: assumes `state["dataset_id"]` was already populated by
    the API (file upload) or by `sql_query_node`; here we only detect columns
    if they aren't set yet."""
    if state.get("group_col") and state.get("metric_col"):
        return {"last_completed_step": "load"}

    roles = detect_column_roles(load_active(state), llm=llm)
    return {**roles, "last_completed_step": "load"}
