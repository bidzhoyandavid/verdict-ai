"""Terminal harness for the onboarding agent.

Usage:
    ANTHROPIC_API_KEY=... .venv/Scripts/python.exe scripts/run_onboarding.py

Walks: intake form (prompts) -> clarifying chat (free text, blank line to
finish) -> LLM-generated company_context.md -> review/edit -> write to
config/companies/{slug}/.
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

from langchain_anthropic import ChatAnthropic

from app.onboarding import (
    OnboardingIntake,
    generate_company_context,
    unique_company_slug,
    write_company_context,
)


def _intake_form() -> OnboardingIntake:
    print("--- Онбординг компании ---")
    company_name = input("Название компании: ").strip()
    while not company_name:
        company_name = input("Название компании (обязательно): ").strip()
    product_description = input("Что делает продукт? ").strip()
    business_model = input("Модель (B2B/B2C/другое): ").strip()
    key_metrics = input("Ключевые метрики (через запятую): ").strip()
    return OnboardingIntake(
        company_name=company_name,
        product_description=product_description,
        business_model=business_model,
        key_metrics=key_metrics,
    )


def _clarifying_chat() -> list[str]:
    print("\n--- Уточнения (специфика домена, сезонность, регуляторка и т.д.) ---")
    print("Пустая строка — закончить.")
    notes: list[str] = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        notes.append(line)
    return notes


def _review(draft: str) -> str:
    print("\n--- Сгенерированный company_context.md ---")
    print(draft)
    print("--- конец ---")
    choice = input("\nПринять как есть? [Y/n] ").strip().lower()
    if choice in ("", "y", "yes"):
        return draft
    print("Вставьте отредактированный текст, завершите строкой из одного '.':")
    lines: list[str] = []
    while True:
        line = input()
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    intake = _intake_form()
    intake.chat_notes = _clarifying_chat()

    llm = ChatAnthropic(model="claude-sonnet-5")
    print("\nГенерирую company_context.md...")
    draft = generate_company_context(intake, llm)

    final_text = _review(draft)

    slug = unique_company_slug(intake.company_name)
    path = write_company_context(slug, final_text)
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()
