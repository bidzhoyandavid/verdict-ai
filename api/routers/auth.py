"""Signup / login. Signup also creates the company — the first user of a
company is always its Admin (per the product spec)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from api.deps import SessionDep, UserDep
from api.models import Company, User
from api.schemas import LoginRequest, SignupRequest, TokenOut, UserOut
from api.security import hash_password, issue_token, verify_password
from api.serializers import user_out
from app.onboarding import unique_company_slug

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, session: SessionDep) -> TokenOut:
    existing = session.scalar(select(User).where(User.email == str(payload.email)))
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")

    company = Company(name=payload.company, slug=unique_company_slug(payload.company))
    session.add(company)
    session.flush()

    user = User(
        company_id=company.id,
        email=str(payload.email),
        password_hash=hash_password(payload.password),
        name=payload.name,
        role="Admin",
    )
    session.add(user)
    session.commit()

    return TokenOut(access_token=issue_token(user.id, company.id), user=user_out(user, company.onboarded))


@router.post("/login", response_model=TokenOut)
def login(payload: LoginRequest, session: SessionDep) -> TokenOut:
    user = session.scalar(select(User).where(User.email == str(payload.email)))
    if user is None or not verify_password(payload.password, user.password_hash):
        # Same message either way — no account enumeration.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")

    company = session.get(Company, user.company_id)
    return TokenOut(
        access_token=issue_token(user.id, user.company_id),
        user=user_out(user, bool(company and company.onboarded)),
    )


@router.get("/me", response_model=UserOut)
def me(user: UserDep, session: SessionDep) -> UserOut:
    company = session.get(Company, user.company_id)
    return user_out(user, bool(company and company.onboarded))
