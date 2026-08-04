"""Loads per-company product context for the LLM-facing nodes.

Each client company gets its own config directory under
`config/companies/{company_id}/`. Only `insight.py` and `clarify.py` read
this — deterministic stat nodes (SRM/selector/stat_test/guardrail/
outlier_review) never need it.
"""

from __future__ import annotations

from pathlib import Path

COMPANIES_DIR = Path(__file__).resolve().parent.parent / "config" / "companies"

_context_cache: dict[str, str] = {}


def _context_path(company_id: str) -> Path:
    return COMPANIES_DIR / company_id / "company_context.md"


def load_company_context(company_id: str) -> str:
    """Read the given company's context markdown file, if present.

    Returns an empty string when the file is missing or empty — callers
    should treat that as "no context available" and fall back to generic
    phrasing, not raise. Cached per company_id; use
    `invalidate_company_context_cache` after the company edits their config.
    """
    if company_id in _context_cache:
        return _context_cache[company_id]
    path = _context_path(company_id)
    text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    _context_cache[company_id] = text
    return text


def context_system_prompt_fragment(company_id: str) -> str:
    """Ready-to-inject system-prompt fragment, or "" if no context is set."""
    context = load_company_context(company_id)
    if not context:
        return ""
    return (
        "Контекст компании и продукта (используй для формулировок, "
        "не выдумывай ничего сверх этого текста):\n\n" + context
    )


def invalidate_company_context_cache(company_id: str | None = None) -> None:
    """Call after a company edits their config via the setup UI, otherwise
    the cached text keeps serving until process restart."""
    if company_id is None:
        _context_cache.clear()
    else:
        _context_cache.pop(company_id, None)

