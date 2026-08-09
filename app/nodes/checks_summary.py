"""Single list of every check the pipeline ran, with an explicit status.

Checks that happen silently might as well not have happened: an analyst has to
see that SRM was tested and passed, not merely that nothing was said about it.
Statuses are `ok` / `warning` / `failed` / `skipped`, where `skipped` means the
check could not run — never "we didn't bother".
"""

from __future__ import annotations

from typing import Any

from app.state import ABTestState

OK = "ok"
WARNING = "warning"
FAILED = "failed"
SKIPPED = "skipped"


def _check(name: str, status: str, detail: str, extra: dict | None = None) -> dict:
    return {"name": name, "status": status, "detail": detail, **(extra or {})}


def _validation_check(state: ABTestState) -> dict:
    report = state.get("validation_report")
    if not report:
        return _check("Валидация данных", SKIPPED, "Отчёт валидации отсутствует")

    issues = list(report.get("issues") or [])
    detail_parts = [
        f"строк: {report.get('n_rows')}" if report.get("n_rows") is not None else "",
        f"пропусков в метрике: {report.get('n_missing_metric')}"
        if report.get("n_missing_metric") is not None
        else "",
        f"дублей по id: {report.get('n_duplicates')}" if report.get("n_duplicates") is not None else "",
    ]
    detail = ", ".join(part for part in detail_parts if part) or "проверено"

    if issues:
        return _check("Валидация данных", WARNING, f"{detail}. Замечания: {'; '.join(issues)}", {"issues": issues})
    return _check("Валидация данных", OK, f"{detail}. Замечаний нет")


def _profile_check(state: ABTestState) -> dict:
    profile = state.get("metric_profile")
    if not profile:
        return _check("Профиль метрики", SKIPPED, "Профиль не рассчитан")

    detail = (
        f"тип: {profile.get('kind')}, скошенность: {_num(profile.get('skewness'))}, "
        f"доля выбросов: {_pct(profile.get('outlier_share'))}, "
        f"нулей: {_pct(profile.get('zero_share'))}, "
        f"дизайн {'сбалансирован' if profile.get('is_balanced_design') else 'не сбалансирован'}"
    )
    status = OK if profile.get("is_balanced_design") else WARNING
    return _check("Профиль метрики", status, detail, {"profile": profile})


def _group_assignment_check(state: ABTestState) -> dict:
    """Направление сравнения — не техническая деталь: если контроль и вариант
    поменяются местами, знак эффекта и вердикт станут противоположными."""
    assignment = state.get("group_assignment")
    if not assignment:
        return _check("Назначение групп", SKIPPED, "Роли групп не определялись")

    detail = (
        f"контроль: {assignment['control']}, вариант: {assignment['treatment']}. "
        f"{assignment['reason']}"
    )
    return _check(
        "Назначение групп",
        WARNING if assignment.get("ambiguous") else OK,
        detail,
        {"assignment": assignment},
    )


def _srm_check(state: ABTestState) -> dict:
    srm = state.get("srm_result")
    if not srm:
        return _check("SRM (баланс групп)", SKIPPED, "Проверка не выполнялась")
    counts = srm.get("observed_counts") or {}
    detail = f"наблюдений по группам: {counts}, p={_num(srm.get('p_value'))}"
    if srm.get("has_srm"):
        return _check("SRM (баланс групп)", FAILED, f"Перекос разбиения. {detail}")
    return _check("SRM (баланс групп)", OK, f"Перекоса нет. {detail}")


def _selector_check(state: ABTestState) -> dict:
    recommendation = state.get("recommendation")
    if not recommendation:
        return _check("Выбор критерия", SKIPPED, "Критерий не выбран")

    violated = list(recommendation.get("violated_assumptions") or [])
    warnings = list(recommendation.get("warnings") or [])
    detail = f"выбран {recommendation.get('method_name')}"
    if violated:
        return _check(
            "Выбор критерия",
            WARNING,
            f"{detail}; нарушены предпосылки: {', '.join(violated)}",
            {"warnings": warnings},
        )
    return _check("Выбор критерия", OK, f"{detail}, предпосылки не нарушены")


def _assumption_checks(state: ABTestState) -> list[dict]:
    results = state.get("assumption_results") or {}
    if not results:
        return [_check("Предпосылки теста", SKIPPED, "Проверки не выполнялись")]

    out: list[dict] = []

    normality = results.get("normality") or []
    not_normal = [row.get("group") for row in normality if row.get("is_normal") is False]
    if not normality:
        out.append(_check("Нормальность распределения", SKIPPED, "Не проверялась"))
    elif not_normal:
        detail = "; ".join(
            f"{row.get('group')}: {row.get('method')} p={_num(row.get('p_value'))}" for row in normality
        )
        out.append(
            _check(
                "Нормальность распределения",
                WARNING,
                f"Отклонена для групп {', '.join(map(str, not_normal))}. {detail}",
            )
        )
    else:
        out.append(_check("Нормальность распределения", OK, "Значимых отклонений не найдено"))

    variance = results.get("equal_variance") or {}
    if variance.get("is_equal") is None:
        out.append(_check("Равенство дисперсий", SKIPPED, variance.get("error", "Не проверялось")))
    elif variance.get("is_equal"):
        out.append(
            _check("Равенство дисперсий", OK, f"{variance.get('method')}, p={_num(variance.get('p_value'))}")
        )
    else:
        out.append(
            _check(
                "Равенство дисперсий",
                WARNING,
                f"Дисперсии различаются ({variance.get('method')}, p={_num(variance.get('p_value'))}) — "
                "нужен тест Уэлча или непараметрический",
            )
        )

    balance = results.get("covariate_balance") or []
    unbalanced = [row.get("covariate") for row in balance if row.get("is_balanced") is False]
    if not balance:
        out.append(_check("Баланс ковариат", SKIPPED, "Подходящих ковариат не найдено"))
    elif unbalanced:
        out.append(
            _check(
                "Баланс ковариат",
                WARNING,
                f"Группы различаются по: {', '.join(map(str, unbalanced))} — рандомизация под вопросом",
                {"covariates": balance},
            )
        )
    else:
        out.append(
            _check("Баланс ковариат", OK, f"Проверено ковариат: {len(balance)}, расхождений нет", {"covariates": balance})
        )

    return out


def _outlier_check(state: ABTestState) -> dict:
    decision = state.get("outlier_decision")
    profile = state.get("metric_profile") or {}
    share = profile.get("outlier_share")

    if decision:
        method = decision.get("method")
        if method == "none":
            return _check("Обработка выбросов", WARNING, f"Выбросов {_pct(share)}, обработка не применялась")
        return _check("Обработка выбросов", OK, f"Выбросов {_pct(share)}, применено: {method}")
    if share is not None and share > 0:
        return _check("Обработка выбросов", SKIPPED, f"Выбросов {_pct(share)} — ниже порога вмешательства")
    return _check("Обработка выбросов", SKIPPED, "Не требовалась")


def _correction_check(state: ABTestState) -> dict:
    correction = state.get("multiple_testing_result")
    n_metrics = len(state.get("test_results") or [])
    if not correction:
        if n_metrics <= 1:
            return _check("Множественные сравнения", SKIPPED, "Метрика одна, поправка не нужна")
        return _check("Множественные сравнения", SKIPPED, "Поправка не применялась")

    revoked = [
        report.get("metric")
        for report in (state.get("test_results") or [])
        if report.get("decision") == "not_significant_after_correction"
    ]
    detail = f"{correction.get('method')} на {len(correction.get('p_values') or [])} метрик"
    if revoked:
        return _check(
            "Множественные сравнения",
            WARNING,
            f"{detail}. Значимость снята с: {', '.join(map(str, revoked))}",
        )
    return _check("Множественные сравнения", OK, detail)


def _guardrail_check(state: ABTestState) -> dict:
    results = state.get("guardrail_results") or []
    if not results:
        return _check("Guardrail-метрики", SKIPPED, "Guardrail-метрики не заданы")
    violated = [
        str(item.get("metric_name") or item.get("metric"))
        for item in results
        if item.get("violated") or item.get("is_violated")
    ]
    if violated:
        return _check("Guardrail-метрики", FAILED, f"Нарушены: {', '.join(violated)}")
    return _check("Guardrail-метрики", OK, f"Проверено: {len(results)}, нарушений нет")


def _power_check(state: ABTestState) -> dict:
    power = state.get("power_result")
    if not power:
        return _check("Мощность", SKIPPED, "Не рассчитывалась (результат значим)")
    if power.get("enough_data"):
        return _check(
            "Мощность",
            OK,
            f"Выборки хватает: {power.get('n_observed_per_group')} на группу при необходимых "
            f"{power.get('required_per_group')}",
        )
    return _check(
        "Мощность",
        WARNING,
        f"Данных не хватает: есть {power.get('n_observed_per_group')} на группу, "
        f"нужно {power.get('required_per_group')} (не хватает {power.get('shortfall_per_group')})",
    )


def _timeline_check(state: ABTestState) -> dict:
    warnings = state.get("timeline_warnings") or []
    if not warnings:
        return _check("Временная шкала", SKIPPED, "Не проверялась")
    if len(warnings) == 1 and "не видно" in warnings[0]:
        return _check("Временная шкала", OK, warnings[0])
    if any("нет колонки со временем" in w for w in warnings):
        return _check("Временная шкала", SKIPPED, warnings[0])
    return _check("Временная шкала", WARNING, " ".join(warnings))


def _num(value: Any) -> str:
    return "—" if value is None else f"{value:.4g}"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{value:.2%}"


def checks_summary_node(state: ABTestState) -> dict:
    checks = [
        _validation_check(state),
        _group_assignment_check(state),
        _profile_check(state),
        _srm_check(state),
        *_assumption_checks(state),
        _outlier_check(state),
        _selector_check(state),
        _correction_check(state),
        _guardrail_check(state),
        _power_check(state),
        _timeline_check(state),
    ]
    return {"checks": checks, "last_completed_step": "checks_summary"}
