"""Onboarding: turn the intake form + chat notes into the company's
`company_context.md`, which every later analysis reads for context.

`app.onboarding` is a linear LLM flow (no graph), so this router calls it
directly instead of going through the runner.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from api.config import settings
from api.deps import SessionDep, UserDep
from api.models import Company
from api.routers.files import _company_doc_path
from app.onboarding import OnboardingIntake, generate_company_context, write_company_context


class IntakeRequest(BaseModel):
    product_description: str = ""
    business_model: str = ""
    key_metrics: str = ""
    chat_notes: list[str] = []


class ContextOut(BaseModel):
    content: str


class ConfirmRequest(BaseModel):
    """Content as reviewed (and possibly edited) by the user."""

    content: str


router = APIRouter(prefix="/onboarding", tags=["onboarding"])


def _company(session, user) -> Company:
    company = session.get(Company, user.company_id)
    if company is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "company not found")
    return company


MAX_DOC_CHARS = 20_000


def _company_doc_text(company_id: str) -> str:
    """Загруженный на онбординге .md — основной источник фактов о продукте.
    Без него агенту нечего структурировать, и он честно пишет "не указано"."""
    path = _company_doc_path(company_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:MAX_DOC_CHARS]


@router.post("/draft", response_model=ContextOut)
def draft_context(payload: IntakeRequest, user: UserDep, session: SessionDep) -> ContextOut:
    from langchain_anthropic import ChatAnthropic

    company = _company(session, user)
    doc_text = _company_doc_text(user.company_id)
    description = "\n\n".join(part for part in (doc_text, payload.product_description) if part.strip())
    intake = OnboardingIntake(
        company_name=company.name,
        product_description=description,
        business_model=payload.business_model,
        key_metrics=payload.key_metrics,
        chat_notes=payload.chat_notes,
    )
    llm = ChatAnthropic(model=settings.llm_model)
    return ContextOut(content=generate_company_context(intake, llm))


@router.post("/confirm", response_model=ContextOut)
def confirm_context(payload: ConfirmRequest, user: UserDep, session: SessionDep) -> ContextOut:
    company = _company(session, user)
    write_company_context(company.slug, payload.content)
    company.onboarded = True
    session.add(company)
    session.commit()
    return ContextOut(content=payload.content)
