"""Shared FastAPI dependencies: authentication and tenant scoping."""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from api.db import get_session
from api.models import Test, User
from api.security import decode_token

SessionDep = Annotated[Session, Depends(get_session)]


def _bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def current_user(
    request: Request,
    session: SessionDep,
    # EventSource cannot send an Authorization header, so SSE endpoints pass
    # the token as a query param. Same token, same verification.
    token: Annotated[str | None, Query()] = None,
) -> User:
    raw = _bearer(request) or token
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        payload = decode_token(raw)
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc

    user = session.get(User, payload.get("sub"))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown user")
    return user


UserDep = Annotated[User, Depends(current_user)]


def owned_test(test_id: str, session: SessionDep, user: UserDep) -> Test:
    """Fetch a test, scoped to the caller's company. 404 (not 403) on a
    foreign id — no cross-tenant existence leak."""
    test = session.get(Test, test_id)
    if test is None or test.company_id != user.company_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "test not found")
    return test


TestDep = Annotated[Test, Depends(owned_test)]
