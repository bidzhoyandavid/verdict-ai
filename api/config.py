"""Runtime configuration, all from env so deploys differ only by env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get("VERDICT_DATABASE_URL", "sqlite:///./.data/verdict.db"))
    jwt_secret: str = field(default_factory=lambda: os.environ.get("VERDICT_JWT_SECRET", "dev-only-insecure-secret-replace-in-any-real-deploy"))
    jwt_ttl_hours: int = field(default_factory=lambda: int(os.environ.get("VERDICT_JWT_TTL_HOURS", "168")))
    llm_model: str = field(default_factory=lambda: os.environ.get("VERDICT_LLM_MODEL", "claude-sonnet-5"))
    uploads_dir: str = field(default_factory=lambda: os.environ.get("VERDICT_UPLOADS_DIR", ".data/uploads"))
    cors_origins: list[str] = field(default_factory=lambda: _csv_env("VERDICT_CORS_ORIGINS", "http://localhost:5173"))

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgres://", "postgresql://", "postgresql+"))


settings = Settings()
