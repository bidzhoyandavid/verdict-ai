"""Insight node — explains the finished analysis in business language.

The numbers live in `results_table` and are already final; this node only
frames them. It never re-derives a verdict: if the table says a metric is not
significant, the prose cannot promote it.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.company_context import context_system_prompt_fragment
from app.state import ABTestState

_UNIMPLEMENTED_METHODS = ("bayesian", "sequential", "cuped", "novelty", "segments")

_SYSTEM = (
    "You explain abex statistical results in plain business language, in Russian. "
    "Only use the numbers given below — never invent figures, never recompute, "
    "never call a metric significant if the table says otherwise. "
    "The results table is shown to the user separately, so do not repeat it "
    "row by row: interpret it. "
    f"If asked about {_UNIMPLEMENTED_METHODS}, say it is not implemented yet."
)


def _payload(state: ABTestState) -> dict:
    return {
        "verdict": state.get("verdict"),
        "checks": state.get("checks"),
        "results_table": state.get("results_table"),
        "assumptions": state.get("assumption_results"),
        "primary_metric": state.get("metric_col"),
        "srm_result": state.get("srm_result"),
        "multiple_testing": state.get("multiple_testing_result"),
        "guardrail_results": state.get("guardrail_results"),
        "power": state.get("power_result"),
        "timeline_warnings": state.get("timeline_warnings"),
        "outlier_decision": state.get("outlier_decision"),
        "method": (state.get("recommendation") or {}).get("method_name"),
    }


def _instruction(state: ABTestState) -> str:
    if (state.get("srm_result") or {}).get("has_srm"):
        return (
            "Sample ratio mismatch — эксперимент невалиден. Объясни, что выводы о "
            "значимости доверять нельзя, пока не починен баг в разбиении, и что "
            "именно стоит проверить."
        )

    parts = [
        "Объясни результат A/B-теста для стейкхолдера. Вердикт уже посчитан "
        "детерминированно — не меняй его, а объясни, почему он такой: размер эффекта "
        "в бизнес-смысле, на что опирается вывод, какие проверки его ослабляют.",
        "Отдельно упомяни вторичные метрики, только если там есть что-то важное.",
    ]
    if state.get("multiple_testing_result"):
        parts.append(
            "Метрик несколько — обязательно скажи, что применена поправка Бонферрони, "
            "и что значимость оценивается по скорректированным p-value."
        )
    if (state.get("power_result") or {}).get("verdict") == "need_more_data":
        parts.append(
            "Расчёт мощности говорит, что данных не хватает — назови, сколько наблюдений "
            "на группу нужно, чтобы поймать эффект такого размера."
        )
    if state.get("guardrail_results"):
        parts.append("Если есть нарушения guardrail-метрик — вынеси их вперёд, даже при позитивном эффекте.")
    if state.get("timeline_warnings"):
        parts.append("Упомяни риски по временной шкале (peeking, эффект новизны), если они есть.")
    return " ".join(parts)


def insight_node(state: ABTestState, llm: Any) -> dict:
    system = _SYSTEM
    context_fragment = context_system_prompt_fragment(state["company_id"])
    if context_fragment:
        system += "\n\n" + context_fragment

    response = llm.invoke(
        [
            ("system", system),
            ("human", f"{_instruction(state)}\n\nДанные: {_payload(state)}"),
        ]
    )
    text = response.content if hasattr(response, "content") else str(response)

    return {"messages": [AIMessage(content=text)], "last_completed_step": "insight"}
