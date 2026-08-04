"""Loads optional company/product context for the LLM-facing nodes.

Only `insight.py` and `clarify.py` read this — deterministic stat nodes
(SRM/selector/stat_test/guardrail/outlier_review) never need it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "company_context.md"


@lru_cache(maxsize=1)
def load_company_context(path: Path = DEFAULT_PATH) -> str:
    """Read the company-context markdown file, if present.

    Returns an empty string when the file is missing or empty — callers
    should treat that as "no context available" and fall back to generic
    phrasing, not raise.
    """
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def context_system_prompt_fragment() -> str:
    """Ready-to-inject system-prompt fragment, or "" if no context is set."""
    context = load_company_context()
    if not context:
        return ""
    return (
        "Контекст компании и продукта (используй для формулировок, "
        "не выдумывай ничего сверх этого текста):\n\n" + context
    )
