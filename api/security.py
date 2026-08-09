"""Password hashing and JWT issuing/verification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from api.config import settings

_ALGORITHM = "HS256"
# bcrypt hashes at most 72 bytes and raises on longer input; truncating is the
# standard workaround (passlib did the same silently).
_MAX_PASSWORD_BYTES = 72


def _encode(password: str) -> bytes:
    return password.encode("utf-8")[:_MAX_PASSWORD_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_encode(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(_encode(password), password_hash.encode("utf-8"))


def issue_token(user_id: str, company_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "company_id": company_id,
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_ttl_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_token(token: str) -> dict:
    """Raises jwt.PyJWTError on anything invalid or expired."""
    return jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
