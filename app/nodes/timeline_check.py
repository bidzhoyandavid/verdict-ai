"""Timeline sanity checks: peeking and novelty-effect warnings.

Deliberately heuristic and text-only. A strict treatment needs always-valid
p-values (`abex.stats.sequential`) and novelty detection
(`abex.analysis.novelty`), both of which are still stubs — so this node warns
rather than pretending to measure. If it ever gets a real calculation, it
belongs in abex, not here.
"""

from __future__ import annotations

import pandas as pd

from app.datasets import load_active
from app.state import ABTestState

# Under a week a weekly cycle isn't covered; under two, novelty effects from
# the first exposure still dominate for most consumer products.
SHORT_TEST_DAYS = 7
NOVELTY_RISK_DAYS = 14


def timeline_check_node(state: ABTestState) -> dict:
    timestamp_col = state.get("timestamp_col")
    if not timestamp_col:
        return {
            "timeline_warnings": [
                "В данных нет колонки со временем — проверить peeking и novelty effect невозможно."
            ],
            "last_completed_step": "timeline_check",
        }

    df = load_active(state)
    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
    if timestamps.empty:
        return {
            "timeline_warnings": [f"Колонка {timestamp_col} не распарсилась как дата."],
            "last_completed_step": "timeline_check",
        }

    span_days = (timestamps.max() - timestamps.min()).total_seconds() / 86_400
    warnings: list[str] = []

    if span_days < SHORT_TEST_DAYS:
        warnings.append(
            f"Тест длился {span_days:.1f} дн. — меньше недели, недельная сезонность не покрыта."
        )
    if span_days < NOVELTY_RISK_DAYS:
        warnings.append(
            f"Тест длился {span_days:.1f} дн. — эффект новизны мог не успеть затухнуть, "
            "результат может завышать долгосрочный эффект."
        )

    # Peeking is about *when the decision is made*, not about the data itself;
    # the data can only tell us that the window looks still open.
    last_day_share = float((timestamps > timestamps.max() - pd.Timedelta(days=1)).mean())
    if last_day_share > 0.3:
        warnings.append(
            f"{last_day_share:.0%} наблюдений пришлись на последние сутки — похоже, тест ещё идёт. "
            "Промежуточные выводы подвержены peeking: фиксируйте срок остановки заранее."
        )

    if not warnings:
        warnings.append(
            f"Тест длился {span_days:.1f} дн., грубых проблем с временной шкалой не видно."
        )

    return {"timeline_warnings": warnings, "last_completed_step": "timeline_check"}
