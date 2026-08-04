"""Onboarding agent — interviews a new company and generates
`config/companies/{slug}/company_context.md`.

No LangGraph here: the flow is linear (form -> chat -> one generation call
-> review -> write), so a plain LLM call is enough. If branching/looping is
ever needed, revisit this decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMPANIES_DIR = Path(__file__).resolve().parent.parent / "config" / "companies"

_TEMPLATE = """## Продукт

{product}

## Ключевые метрики

{metrics}

## Специфика домена

{domain}
"""

_GENERATION_SYSTEM_PROMPT = (
    "You write a company/product context brief for an A/B-testing analysis agent. "
    "Source material is a short intake form plus a follow-up chat with the company's "
    "representative. Produce exactly three sections in Russian, plain text (no markdown "
    "headers, those are added separately): product description, key metrics, domain "
    "specifics (seasonality, decision cycle length, regulatory constraints, anything that "
    "affects how A/B results should be interpreted). Be concise, factual, only use what "
    "was actually said — never invent details the user didn't provide. If a section has "
    "no material, write a short honest placeholder like 'не указано'."
)


@dataclass
class OnboardingIntake:
    """Everything collected from the form + chat, before LLM structuring."""

    company_name: str
    product_description: str = ""
    business_model: str = ""
    key_metrics: str = ""
    chat_notes: list[str] = field(default_factory=list)


def slugify(company_name: str) -> str:
    slug = company_name.strip().lower()
    slug = re.sub(r"[^a-z0-9а-яё]+", "-", slug)
    slug = slug.strip("-")
    return slug or "company"


def unique_company_slug(company_name: str) -> str:
    """slugify() plus a numeric suffix on collision — never overwrites an
    existing company directory."""
    base = slugify(company_name)
    slug = base
    i = 2
    while (COMPANIES_DIR / slug).exists():
        slug = f"{base}-{i}"
        i += 1
    return slug


def _response_text(response: Any) -> str:
    """ChatAnthropic's .content is either a plain string or a list of content
    blocks (e.g. thinking + text blocks); pull the text out either way."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def _intake_prompt(intake: OnboardingIntake) -> str:
    parts = [
        f"Компания: {intake.company_name}",
        f"Описание продукта (из формы): {intake.product_description or 'не указано'}",
        f"Модель (B2B/B2C и т.п.): {intake.business_model or 'не указано'}",
        f"Ключевые метрики (из формы): {intake.key_metrics or 'не указано'}",
    ]
    if intake.chat_notes:
        parts.append("Уточнения из чата:\n" + "\n".join(f"- {n}" for n in intake.chat_notes))
    return "\n".join(parts)


def generate_company_context(intake: OnboardingIntake, llm: Any) -> str:
    """Ask the LLM to structure the intake into the three template sections
    and return the ready-to-review company_context.md body (no HTML comment
    header — that's template-only boilerplate, not needed for real companies)."""
    response = llm.invoke(
        [
            ("system", _GENERATION_SYSTEM_PROMPT),
            (
                "human",
                _intake_prompt(intake)
                + "\n\nВерни ровно три абзаца текста в таком порядке: продукт, "
                "ключевые метрики, специфика домена. Без заголовков markdown, "
                "они добавятся отдельно.",
            ),
        ]
    )
    text = _response_text(response)
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    while len(paragraphs) < 3:
        paragraphs.append("не указано")
    product, metrics, domain = paragraphs[0], paragraphs[1], "\n\n".join(paragraphs[2:])
    return _TEMPLATE.format(product=product, metrics=metrics, domain=domain)


def write_company_context(slug: str, content: str) -> Path:
    """Write the (possibly user-edited) reviewed content to
    config/companies/{slug}/company_context.md. Creates the directory if new."""
    company_dir = COMPANIES_DIR / slug
    company_dir.mkdir(parents=True, exist_ok=True)
    path = company_dir / "company_context.md"
    path.write_text(content, encoding="utf-8")
    return path
