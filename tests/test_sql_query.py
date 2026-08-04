import pytest

from app.nodes.sql_query import SqlTemplateError, fill_template, template_placeholders


TEMPLATE = """SELECT {{group_col}} AS g, {{metric_col}} AS m
FROM {{table_name}}
WHERE event_date >= '{{start_date}}' AND event_date < '{{end_date}}'
"""


def test_template_placeholders_extracted():
    assert template_placeholders(TEMPLATE) == {"group_col", "metric_col", "table_name", "start_date", "end_date"}


def test_fill_template_substitutes_all_placeholders():
    filled = fill_template(
        TEMPLATE,
        {
            "group_col": "variant",
            "metric_col": "revenue",
            "table_name": "experiments.events",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
        },
    )
    assert "{{" not in filled
    assert "variant" in filled and "experiments.events" in filled


def test_fill_template_missing_param_raises():
    with pytest.raises(SqlTemplateError):
        fill_template(TEMPLATE, {"group_col": "variant"})


def test_fill_template_extra_param_raises():
    params = {
        "group_col": "variant",
        "metric_col": "revenue",
        "table_name": "t",
        "start_date": "2026-01-01",
        "end_date": "2026-02-01",
        "unexpected": "x",
    }
    with pytest.raises(SqlTemplateError):
        fill_template(TEMPLATE, params)


def test_non_select_query_rejected():
    from app.nodes.sql_query import _assert_read_only, SqlTemplateError as Err

    with pytest.raises(Err):
        _assert_read_only("DROP TABLE experiments.events")


def test_select_query_accepted():
    from app.nodes.sql_query import _assert_read_only

    _assert_read_only("SELECT 1")
    _assert_read_only("WITH x AS (SELECT 1) SELECT * FROM x")
