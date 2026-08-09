"""SQL-template data source: the agent never writes SQL itself.

It only (a) picks one of the analyst-authored templates in
`config/sql_templates/`, (b) fills the template's own named placeholders
with values, (c) runs the filled text through a read-only-assumed
SQLAlchemy engine after a defensive SELECT/WITH check. No free-form SQL
generation — this is a deliberate injection-safety boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.datasets import put_dataframe

COMPANIES_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "companies"
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def templates_dir(company_id: str) -> Path:
    return COMPANIES_DIR / company_id / "sql_templates"


class SqlTemplateError(ValueError):
    pass


def list_templates(templates_dir: Path) -> list[str]:
    if not templates_dir.exists():
        return []
    return sorted(p.stem for p in templates_dir.glob("*.sql"))


def load_template(name: str, templates_dir: Path) -> str:
    path = templates_dir / f"{name}.sql"
    if not path.exists():
        raise SqlTemplateError(f"unknown SQL template: {name!r}")
    return path.read_text(encoding="utf-8")


def template_placeholders(template_text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(template_text))


def fill_template(template_text: str, params: dict[str, str]) -> str:
    """Substitute `{{name}}` placeholders with `params[name]`.

    Raises
    ------
    SqlTemplateError
        If `params` is missing a placeholder the template requires, or
        supplies a name the template doesn't declare.
    """
    required = template_placeholders(template_text)
    missing = required - params.keys()
    if missing:
        raise SqlTemplateError(f"missing required params: {sorted(missing)}")
    extra = params.keys() - required
    if extra:
        raise SqlTemplateError(f"params not declared by template: {sorted(extra)}")

    def _sub(match: re.Match) -> str:
        return str(params[match.group(1)])

    return _PLACEHOLDER_RE.sub(_sub, template_text)


def _assert_read_only(sql: str) -> None:
    stripped = sql.strip().lstrip("-- ").strip()
    # skip leading SQL comment lines
    lines = [line for line in sql.strip().splitlines() if not line.strip().startswith("--")]
    body = "\n".join(lines).strip()
    if not re.match(r"^(SELECT|WITH)\b", body, re.IGNORECASE):
        raise SqlTemplateError("filled query must start with SELECT or WITH")


def run_sql_template(
    engine: Engine, template_name: str, params: dict[str, str], company_id: str
) -> pd.DataFrame:
    """Fill and execute a named template from the company's own template
    directory, returning the result as a DataFrame."""
    raw = load_template(template_name, templates_dir(company_id))
    filled = fill_template(raw, params)
    _assert_read_only(filled)
    with engine.connect() as conn:
        return pd.read_sql(text(filled), conn)


def sql_query_node(state: dict, engine: Engine) -> dict:
    """LangGraph node: reads `sql_template_name`/`sql_params` from state,
    executes against the calling company's own templates, and returns the
    resulting dataset id."""
    template_name = state.get("sql_template_name")
    params = state.get("sql_params") or {}
    company_id = state["company_id"]
    if not template_name:
        raise ValueError("sql_template_name must be set before sql_query_node runs")
    df = run_sql_template(engine, template_name, params, company_id)
    dataset_id = put_dataframe(df, company_id=company_id, source="sql")
    return {"dataset_id": dataset_id, "data_source": "sql"}
