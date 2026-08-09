"""ORM rows -> wire shapes.

The results table arrives from the graph already final — this module only
reshapes it and derives the one-line summary the tests list shows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from api.models import Message, Test, User
from api.schemas import CheckResult, MessageOut, ResultRow, TestOut, TestResults, UserOut, Verdict


def jsonable(value: Any) -> Any:
    """Приводит вывод abex к JSON-совместимым типам.

    Отчёты собираются из pandas/numpy, поэтому в них попадают np.bool_,
    np.int64 и np.float64 — драйвер БД на них падает.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        # JSON не знает NaN/Infinity; хранить их как null честнее, чем ловить
        # ошибку парсинга на фронте.
        return None
    return value


def initials(name: str, email: str) -> str:
    parts = [p for p in name.split() if p]
    if parts:
        return "".join(p[0].upper() for p in parts[:2])
    return email[:2].upper()


def user_out(user: User, onboarded: bool) -> UserOut:
    return UserOut(
        id=user.id,
        name=user.name or user.email.split("@")[0],
        email=user.email,
        role=user.role,  # type: ignore[arg-type]
        initials=initials(user.name, user.email),
        company_id=user.company_id,
        onboarded=onboarded,
    )


def _short_summary(rows: list[dict], srm: bool, power: dict | None) -> str:
    """Одна строка для колонки «Результаты» — по главной метрике."""
    if srm:
        return "SRM: результат невалиден"
    if not rows:
        return ""

    primary = next((row for row in rows if row.get("is_primary")), rows[0])
    lift = primary.get("relative_diff")
    lift_str = f"{lift * 100:+.1f}%" if isinstance(lift, (int, float)) else "—"

    p_value = primary.get("adjusted_p_value") or primary.get("p_value")
    p_str = f"p={p_value:.3g}" if isinstance(p_value, (int, float)) else ""

    if primary.get("significant"):
        verdict = "значимо"
    elif (power or {}).get("verdict") == "need_more_data":
        verdict = "нужно больше данных"
    else:
        verdict = "не значимо"

    parts = [f"{primary.get('metric')} {lift_str}", p_str, verdict]
    return ", ".join(part for part in parts if part)


def summarize(results: dict[str, Any] | None) -> TestResults | None:
    if not results:
        return None

    rows = results.get("results_table") or []
    srm = bool((results.get("srm_result") or {}).get("has_srm"))
    power = results.get("power")
    correction = (results.get("multiple_testing") or {}).get("method")

    violations = [
        str(item.get("metric_name") or item.get("metric"))
        for item in (results.get("guardrail_results") or [])
        if item.get("violated") or item.get("is_violated")
    ]

    verdict = results.get("verdict")
    return TestResults(
        rows=[ResultRow(**{k: v for k, v in row.items() if k in ResultRow.model_fields}) for row in rows],
        checks=[
            CheckResult(**{k: v for k, v in check.items() if k in CheckResult.model_fields})
            for check in (results.get("checks") or [])
        ],
        verdict=Verdict(**{k: v for k, v in verdict.items() if k in Verdict.model_fields})
        if verdict
        else None,
        short=_short_summary(rows, srm, power),
        srm_detected=srm,
        correction_applied=correction,
        power_verdict=(power or {}).get("verdict"),
        timeline_warnings=list(results.get("timeline_warnings") or []),
        guardrail_violations=violations,
        raw=results,
    )


def test_out(test: Test, include_charts: bool = False) -> TestOut:
    """`include_charts` только для детального эндпоинта: спеки plotly весят
    десятки килобайт и в списке тестов не нужны."""
    return TestOut(
        id=test.id,
        name=test.name,
        hypothesis=test.hypothesis,
        status=test.status,  # type: ignore[arg-type]
        decision=test.decision or "—",
        date=test.created_at.strftime("%d.%m.%Y"),
        dataset_id=test.dataset_id,
        results=summarize(test.results),
        charts=(test.charts or []) if include_charts else None,
        pending_interrupt=test.pending_interrupt,
        error=test.error,
    )


def message_out(message: Message, author_initials: str | None = None) -> MessageOut:
    return MessageOut(
        id=message.id,
        role=message.role,  # type: ignore[arg-type]
        author=message.author,
        text=message.text,
        initials=author_initials if message.role == "user" else None,
        results=message.results,
    )
