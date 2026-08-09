"""SQLAlchemy session plumbing.

Product state (companies, users, tests, chat messages) lives here. The
LangGraph checkpointer keeps agent state separately — the two are joined
only by `Test.thread_id`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import settings


class Base(DeclarativeBase):
    pass


def _engine_kwargs() -> dict:
    if settings.is_postgres:
        return {"pool_pre_ping": True}
    # SQLite: the graph runner touches the DB from worker threads.
    Path("./.data").mkdir(parents=True, exist_ok=True)
    return {"connect_args": {"check_same_thread": False}}


engine = create_engine(settings.database_url, future=True, **_engine_kwargs())
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with SessionLocal() as session:
        yield session


def create_all() -> None:
    from api import models  # noqa: F401  (registers mappers)

    Base.metadata.create_all(engine)
