"""The bottom line: a single verdict plus what to do about it.

Deterministic on purpose. The verdict follows from the numbers by fixed rules,
so it cannot drift with the LLM's mood; the Insight node explains it, it does
not decide it.
"""

from __future__ import annotations

from app.state import ABTestState

INVALID = "invalid"
WINNER = "winner"
LOSER = "loser"
NO_EFFECT = "no_effect"
NEED_MORE_DATA = "need_more_data"
INCONCLUSIVE = "inconclusive"

LABELS = {
    INVALID: "Эксперимент невалиден",
    WINNER: "Есть победитель",
    LOSER: "Вариант хуже контроля",
    NO_EFFECT: "Значимой разницы нет",
    NEED_MORE_DATA: "Нужно больше данных",
    INCONCLUSIVE: "Результат неоднозначный",
}

ACTIONS = {
    INVALID: "Не принимать решение по этому тесту — сначала починить разбиение и перезапустить.",
    WINNER: "Можно раскатывать вариант, предварительно проверив guardrail-метрики и риски по срокам.",
    LOSER: "Не раскатывать: вариант значимо хуже контроля по главной метрике.",
    NO_EFFECT: "Разницы нет. Оставить контроль и искать более сильную гипотезу.",
    NEED_MORE_DATA: "Решение откладывается: добрать выборку до расчётного размера и посмотреть снова.",
    INCONCLUSIVE: "Решение откладывается: сигналы противоречивы, нужен повторный тест.",
}


def _primary_row(state: ABTestState) -> dict | None:
    rows = state.get("results_table") or []
    for row in rows:
        if row.get("is_primary"):
            return row
    return rows[0] if rows else None


def _blocking_checks(state: ABTestState) -> list[dict]:
    return [check for check in (state.get("checks") or []) if check["status"] == "failed"]


def _caveats(state: ABTestState) -> list[str]:
    return [
        f"{check['name']}: {check['detail']}"
        for check in (state.get("checks") or [])
        if check["status"] == "warning"
    ]


def verdict_node(state: ABTestState) -> dict:
    row = _primary_row(state)
    blocking = _blocking_checks(state)
    caveats = _caveats(state)
    power = state.get("power_result") or {}

    if (state.get("srm_result") or {}).get("has_srm"):
        code = INVALID
    elif blocking:
        # Нарушенный guardrail перевешивает выигрыш по главной метрике.
        code = INVALID if any("SRM" in c["name"] for c in blocking) else INCONCLUSIVE
    elif row is None:
        code = INCONCLUSIVE
    elif row.get("significant"):
        lift = row.get("relative_diff")
        code = LOSER if isinstance(lift, (int, float)) and lift < 0 else WINNER
    elif power.get("verdict") == "need_more_data":
        code = NEED_MORE_DATA
    else:
        code = NO_EFFECT

    action = ACTIONS[code]
    if code == NEED_MORE_DATA and power.get("required_per_group"):
        action = (
            f"Решение откладывается: нужно ~{power['required_per_group']} наблюдений на группу "
            f"(сейчас {power.get('n_observed_per_group')}), чтобы поймать эффект такого размера."
        )
    if code in (WINNER, LOSER) and blocking:
        action += " Внимание: часть обязательных проверок не пройдена — см. список ниже."

    return {
        "verdict": {
            "code": code,
            "label": LABELS[code],
            "action": action,
            "metric": (row or {}).get("metric"),
            "relative_diff": (row or {}).get("relative_diff"),
            "p_value": (row or {}).get("adjusted_p_value") or (row or {}).get("p_value"),
            "blocking_checks": [c["name"] for c in blocking],
            "caveats": caveats,
        },
        "last_completed_step": "verdict",
    }
