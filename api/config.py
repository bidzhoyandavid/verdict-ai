"""Runtime configuration, all from env so deploys differ only by env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Загружается один раз при импорте конфига — до того, как что-либо прочитает
# os.environ. uvicorn сам .env не подхватывает.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEV_JWT_SECRET = "dev-only-insecure-secret-replace-in-any-real-deploy"


def _env(name: str, default: str) -> str:
    """Пустая переменная в .env — то же самое, что незаданная: иначе
    оставленный пустым ключ молча перебивает дефолт."""
    return os.environ.get(name, "").strip() or default


def _csv_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in _env(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: _env("VERDICT_DATABASE_URL", "sqlite:///./.data/verdict.db"))
    jwt_secret: str = field(default_factory=lambda: _env("VERDICT_JWT_SECRET", DEV_JWT_SECRET))
    jwt_ttl_hours: int = field(default_factory=lambda: int(_env("VERDICT_JWT_TTL_HOURS", "168")))
    llm_model: str = field(default_factory=lambda: _env("VERDICT_LLM_MODEL", "claude-sonnet-5"))
    uploads_dir: str = field(default_factory=lambda: _env("VERDICT_UPLOADS_DIR", ".data/uploads"))
    cors_origins: list[str] = field(default_factory=lambda: _csv_env("VERDICT_CORS_ORIGINS", "http://localhost:5173"))

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgres://", "postgresql://", "postgresql+"))

    @property
    def uses_dev_jwt_secret(self) -> bool:
        return self.jwt_secret == DEV_JWT_SECRET


settings = Settings()
