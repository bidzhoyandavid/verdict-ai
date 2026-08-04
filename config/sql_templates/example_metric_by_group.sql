-- Пример шаблона. Плейсхолдеры вида {{name}} заполняются агентом
-- (app/nodes/sql_query.py) через строгую замену текста, не через
-- LLM-генерацию SQL. Разрешённые плейсхолдеры этого шаблона:
--   start_date, end_date, group_col, metric_col, table_name
SELECT
    {{group_col}} AS group_label,
    {{metric_col}} AS metric_value
FROM {{table_name}}
WHERE event_date >= '{{start_date}}'
  AND event_date < '{{end_date}}'
