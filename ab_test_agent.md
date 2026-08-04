# AI-агент для анализа A/B тестов

## Назначение
Агент принимает данные A/B теста (метрики, группы), проводит статистический анализ, даёт вывод: победитель / не значимо / нужно больше данных. Интерфейс — чат в Streamlit, оркестрация — LangGraph, LLM-обвязка — LangChain. Вся статистика — через собственный пакет **`abex`** (`D:\Startups\abex`, wheel `abex-0.1.0-py3-none-any.whl`), агент — только оркестратор поверх него, свою статистику не считает.

## Стек
- **LangGraph** — граф состояний агента (control flow, tool calls, memory)
- **LangChain** — обёртки LLM, tools, prompt templates
- **abex** — вся статистика: профилирование, selector, тесты, effect size, guardrails, отчёт
- **Streamlit** — UI (чат + загрузка файла + графики)
- **plotly** — визуализация распределений, доверительных интервалов (обёртка над `abex.viz`)

## Пакет abex — карта модулей
```
abex/
├── data/
│   ├── loaders.py       (не инспектирован)
│   ├── validators.py    validate(df, group_col, metric_col, id_col=None) -> ValidationReport
│   ├── profiling.py     profile_metric(...) -> MetricProfile (kind, skew, outlier_share, ...)
│   └── outliers.py      detect_outliers(values, method="iqr"|"mad"|"zscore") -> bool mask
├── design/
│   ├── srm.py           check_srm(group_counts, expected_ratios=None, alpha=0.001) -> SRMResult
│   ├── power.py         sample_size_proportion(baseline_rate, mde_abs, alpha=0.05, power=0.8) -> PowerResult
│   └── covariate_balance.py  (не инспектирован)
├── selector/
│   ├── recommend.py     recommend_test(profile: MetricProfile, paired=False) -> list[Recommendation]
│   ├── rules.py          детерминированные правила отбора кандидатов (без ML)
│   └── registry.py       реестр методов (MethodSpec)
├── stats/
│   ├── frequentist.py   t_test(control, treatment, equal_var=False) -> TestResult (Welch по умолчанию)
│   ├── ratio.py          compute_ratio, линеаризация ratio-метрик (Deng et al. 2018) под frequentist/bootstrap
│   ├── bootstrap.py     bootstrap_ci(control, treatment, statistic, ...) -> BootstrapResult
│   ├── multiple_testing.py  bonferroni(p_values, alpha=0.05) -> CorrectionResult
│   ├── bayesian.py      ⚠️ TODO, NotImplementedError (beta_binomial)
│   ├── sequential.py    ⚠️ TODO, NotImplementedError (always_valid_p_value)
│   └── cuped.py         ⚠️ TODO, NotImplementedError (cuped_adjust)
├── analysis/
│   ├── effect_size.py   cohens_d(control, treatment) -> float; EffectSizeResult (abs_diff, relative_lift, ci)
│   ├── guardrails.py    check_guardrail(control, treatment, metric_name, max_allowed_degradation, higher_is_better=True) -> GuardrailResult
│   ├── novelty.py       ⚠️ TODO, NotImplementedError (detect_novelty_effect)
│   └── segments.py      ⚠️ TODO, NotImplementedError (segment_effects)
├── viz/
│   ├── distributions.py / outliers.py / timeseries.py  (не инспектированы)
└── report.py            build_report(metric, method, p_value, effect, alpha=0.05, ci=None, warnings=None) -> dict
```

**Важно:** bayesian, sequential, cuped, novelty, segments — заглушки (raise NotImplementedError). Агент не должен предлагать их пользователю как рабочие пути; при запросе — отвечать, что метод в разработке.

`report.py` задаёт JSON-контракт результата (`metric, method, effect, ci, p_value, decision, agreement, warnings`) — это то, что Insight Node парсит и превращает в текст. Не изобретать свой формат поверх него.

## Архитектура (LangGraph nodes)

```
[User Input] → [Router] → [Data Loader] → [Validate+Profile Node] → [SRM Gate] → [Test Selector] → [Stat Test Node] → [Guardrail Node] → [Insight Node] → [Response]
                   ↑                                                                                                                                  |
                   └──────────────────────────────────── [Clarify Node] ←── (если данных/контекста не хватает) ────────────────────────────────────┘
```

### Узлы графа
1. **Router** — определяет намерение: загрузка данных / вопрос по тесту / уточнение параметров.
2. **Data Loader** — парсит CSV/Excel через `abex.data.loaders`, определяет колонки (group, metric, id, timestamp).
3. **Validate+Profile Node** — `abex.data.validators.validate()` (типы, пропуски, дубликаты, кардинальность групп) + `abex.data.profiling.profile_metric()` (kind, skew, outlier_share, баланс дизайна). Если `outlier_share` большой — предупреждение, опционально `abex.data.outliers.detect_outliers`.
4. **SRM Gate** — `abex.design.srm.check_srm()` до любого вывода о значимости, alpha=0.001. При `has_srm=True` — граф останавливается, эксперимент считается невалидным, идёт сразу в Insight Node с сообщением об SRM.
5. **Test Selector** — `abex.selector.recommend.recommend_test(profile, paired=...)`, берётся top-1 рекомендация (`fn_path`), warnings/violated_assumptions пробрасываются в отчёт. Если ranked-список пуст — Clarify Node (не хватает данных/дизайн не поддержан).
6. **Stat Test Node** — динамический вызов метода по `fn_path` (`abex.stats.frequentist.t_test` / `abex.stats.ratio` + линеаризация / `abex.stats.bootstrap.bootstrap_ci`), `abex.analysis.effect_size.cohens_d`/`effect_size_summary`. Если метрик > 1 — `abex.stats.multiple_testing.bonferroni` на список p-value.
7. **Guardrail Node** — `abex.analysis.guardrails.check_guardrail()` по заранее заданным guardrail-метрикам (latency, errors и т.п.), если они присутствуют в данных.
8. **Insight Node** — на входе `abex.report.build_report()` (JSON-контракт), LLM переводит в бизнес-текст: значимость, effect size, guardrail-нарушения, риски. Явно помечает недоступные методы (bayesian/sequential/cuped/novelty/segments) как "в разработке", если пользователь их просит.
9. **Clarify Node** — неоднозначность колонок, пустой selector, недостаточная выборка → уточняющий вопрос вместо угадывания.

## State (LangGraph state schema)
```python
class ABTestState(TypedDict):
    raw_data: pd.DataFrame | None
    group_col: str | None
    metric_col: str | None
    id_col: str | None
    validation_report: dict | None      # abex.data.validators.ValidationReport
    metric_profile: dict | None         # abex.data.profiling.MetricProfile
    srm_result: dict | None             # abex.design.srm.SRMResult
    recommendation: dict | None         # abex.selector.recommend.Recommendation (top-1)
    test_result: dict | None            # abex.report.build_report() output
    guardrail_results: list[dict]
    messages: list[BaseMessage]
    needs_clarification: bool
```

## Tools (LangChain tools — тонкие обёртки над abex, без своей статистики)
- `load_dataset(file) -> DataFrame`
- `detect_column_roles(df) -> dict` (эвристика + LLM fallback)
- `validate_dataset(df, group_col, metric_col, id_col=None) -> ValidationReport` → `abex.data.validators.validate`
- `profile_metric(df, metric_col, group_col) -> MetricProfile` → `abex.data.profiling.profile_metric`
- `check_srm_tool(group_counts, expected_ratios=None) -> SRMResult` → `abex.design.srm.check_srm`
- `recommend_test_tool(profile, paired=False) -> list[Recommendation]` → `abex.selector.recommend.recommend_test`
- `run_stat_test(fn_path, control, treatment, **kwargs) -> TestResult|BootstrapResult` — динамический вызов метода из рекомендации
- `compute_effect_size(control, treatment) -> EffectSizeResult` → `abex.analysis.effect_size`
- `check_guardrail_tool(control, treatment, metric_name, max_allowed_degradation, higher_is_better=True) -> GuardrailResult`
- `correct_multiple(p_values, alpha=0.05) -> CorrectionResult` → `abex.stats.multiple_testing.bonferroni`
- `build_report_tool(...) -> dict` → `abex.report.build_report`
- `plot_distribution(df, metric_col, group_col) -> plotly.Figure` → `abex.viz.distributions`
- `plot_confidence_interval(result) -> plotly.Figure`

## Streamlit UI
- Левая колонка: загрузка файла, выбор колонок (auto-detect + возможность переопределить вручную).
- Центр: чат-интерфейс (`st.chat_message`), где агент объясняет находки и отвечает на вопросы.
- Правая колонка / под чатом: графики (распределения по группам, CI, timeline метрики).
- Кнопка "Запустить анализ" триггерит граф LangGraph, стриминг ответа через `st.write_stream`.

## Обязательные проверки (не опционально)
- SRM (`design.srm.check_srm`, alpha=0.001) — гейт перед любым выводом о значимости; при срабатывании эксперимент невалиден, дальше по графу тест не считается.
- Multiple comparisons — если метрик > 1, обязательная поправка `stats.multiple_testing.bonferroni` перед выводом о значимости.
- Guardrail-метрики — `analysis.guardrails.check_guardrail` проверяется параллельно с основной метрикой, нарушение отражается в Insight Node даже при значимом позитивном эффекте.
- Effect size (`analysis.effect_size`) всегда рядом с p-value, не только значимость.
- Peeking / novelty effect — сейчас **не покрыты** пакетом (`sequential.py`, `novelty.py` — заглушки). Пока агент может только текстово предупредить по эвристике (timestamp есть + тест короче ~1-2 недель), без строгого расчёта.

## Ограничения агента
- Не делает причинных выводов за пределами эксперимента.
- Не рекомендует ML там, где хватает теста из `abex.selector`.
- Выбор теста не изобретает сам — доверяет `selector.recommend_test`; если ranked-список пуст, эскалирует в Clarify, а не гадает вручную.
- Явно сообщает пользователю, если запрошенный метод (bayesian/sequential/cuped/novelty/segments) ещё не реализован в `abex`, вместо того чтобы подменять его самодельным расчётом.

## Дальнейшее развитие (следует за roadmap `abex`, не в MVP)
- `stats.bayesian.beta_binomial` — байесовский A/B режим.
- `stats.sequential.always_valid_p_value` — live-мониторинг тестов без proper peeking penalty.
- `stats.cuped.cuped_adjust` — снижение дисперсии через pre-period ковариаты.
- `analysis.novelty.detect_novelty_effect`, `analysis.segments.segment_effects` — детект novelty/primacy, разбивка по сегментам.
- Подключение к внутреннему хранилищу экспериментов вместо ручной загрузки файла.
