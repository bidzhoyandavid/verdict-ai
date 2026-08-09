"""Detects column roles on the dataset referenced by the state.

Detection is heuristic-first (name matching), LLM only as a fallback when
heuristics are ambiguous — keeps the common case free of an LLM round trip.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.datasets import load_active
from app.groups import resolve_groups
from app.state import ABTestState

_GROUP_NAME_HINTS = ("group", "variant", "arm", "bucket", "cohort", "treatment")
_METRIC_NAME_HINTS = ("metric", "value", "revenue", "conversion", "count", "duration", "amount")
_ID_NAME_HINTS = ("user_id", "userid", "id", "uid", "customer_id")
_TIMESTAMP_NAME_HINTS = ("timestamp", "datetime", "date", "time", "dt", "created_at", "event_time")


def _heuristic_match(columns: list[str], hints: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for hint in hints:
        for lower_name, original in lowered.items():
            if hint in lower_name:
                return original
    return None


def detect_timestamp_col(df: pd.DataFrame) -> str | None:
    """Datetime dtype wins over a name match — a column literally called
    `date` but stored as an unparseable string is useless for timeline checks.
    """
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return str(column)

    # Числовые колонки исключаются до сопоставления имён: `to_datetime` молча
    # прочитает числа как наносекунды от эпохи, и метрика вроде
    # `car_exploitation_time_hour` (в имени есть "time") станет «временем теста».
    textual = [
        str(column)
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column])
    ]
    candidate = _heuristic_match(textual, _TIMESTAMP_NAME_HINTS)
    if candidate is None:
        return None
    parsed = pd.to_datetime(df[candidate], errors="coerce")
    return candidate if parsed.notna().mean() > 0.9 else None


def detect_metric_cols(
    df: pd.DataFrame,
    primary: str | None,
    group_col: str | None,
    id_col: str | None,
    timestamp_col: str | None,
) -> list[str]:
    """Every numeric column that isn't structural, primary metric first.

    Analyzing all of them is what makes the multiple-comparisons correction
    mandatory rather than decorative.
    """
    structural = {group_col, id_col, timestamp_col} - {None}
    numeric = [
        str(column)
        for column in df.columns
        if str(column) not in structural and pd.api.types.is_numeric_dtype(df[column])
    ]

    if primary and primary in numeric:
        numeric.remove(primary)
        return [primary, *numeric]
    return [primary, *numeric] if primary else numeric


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
    the API (file upload) or by `sql_query_node`; here we detect whatever
    column roles are still missing."""
    df = load_active(state)

    if state.get("group_col") and state.get("metric_col"):
        roles = {
            "group_col": state["group_col"],
            "metric_col": state["metric_col"],
            "id_col": state.get("id_col"),
        }
    else:
        roles = detect_column_roles(df, llm=llm)

    timestamp_col = state.get("timestamp_col") or detect_timestamp_col(df)
    metric_cols = state.get("metric_cols") or detect_metric_cols(
        df,
        primary=roles.get("metric_col"),
        group_col=roles.get("group_col"),
        id_col=roles.get("id_col"),
        timestamp_col=timestamp_col,
    )

    # Роли групп решаются один раз здесь: ниже по графу все ноды должны
    # считать разницу в одну и ту же сторону.
    group_assignment = None
    group_col = roles.get("group_col")
    if group_col and group_col in df.columns:
        names = [str(name) for name in df[group_col].dropna().unique()]
        if len(names) >= 2:
            group_assignment = resolve_groups(
                names, state.get("control_group"), state.get("treatment_group")
            )

    return {
        **roles,
        "timestamp_col": timestamp_col,
        "metric_cols": metric_cols,
        "group_assignment": group_assignment,
        "last_completed_step": "load",
    }
