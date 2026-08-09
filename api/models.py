"""Product-side tables.

Deliberately *not* a mirror of `ABTestState`: the graph owns analysis state,
these tables own what the UI lists, filters and reloads. `Test.thread_id` is
the join key into the checkpointer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    # Directory name under config/companies/ holding company_context.md.
    slug: Mapped[str] = mapped_column(String(200), unique=True)
    onboarded: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    users: Mapped[list["User"]] = relationship(back_populates="company")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    name: Mapped[str] = mapped_column(String(200), default="")
    # First user of a company is always Admin — enforced at signup.
    role: Mapped[str] = mapped_column(String(20), default="Analyst")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    company: Mapped[Company] = relationship(back_populates="users")


class Test(Base):
    __tablename__ = "tests"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), index=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))

    name: Mapped[str] = mapped_column(String(300))
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    # queued | analyzing | awaiting_input | clarifying | done | failed
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    decision: Mapped[str] = mapped_column(String(50), default="")

    # One graph thread per test — this is the checkpointer key.
    thread_id: Mapped[str] = mapped_column(String(64), unique=True, default=_uuid)
    dataset_id: Mapped[str | None] = mapped_column(String(32), default=None)

    # Denormalized snapshot for the list view, so "All tests" never resurrects graphs.
    results: Mapped[dict | None] = mapped_column(JSON, default=None)
    # Live interrupt payload, so a page reload re-renders the pending question.
    pending_interrupt: Mapped[dict | None] = mapped_column(JSON, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    config: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    test_id: Mapped[str] = mapped_column(ForeignKey("tests.id"), index=True)
    role: Mapped[str] = mapped_column(String(10))  # agent | user
    author: Mapped[str] = mapped_column(String(200), default="")
    text: Mapped[str] = mapped_column(Text)
    results: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
