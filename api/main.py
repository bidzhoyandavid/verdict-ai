"""FastAPI entrypoint.

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

# OpenBLAS multithreading crashes under some Windows dev setups; abex is
# single-request-bound anyway.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import events
from api.config import settings
from api.db import create_all
from api.routers import auth, files, onboarding, team, tests

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_all()
    events.bind_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="Verdict AI API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(tests.router)
app.include_router(files.router)
app.include_router(team.router)
app.include_router(onboarding.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "checkpointer": "postgres" if settings.is_postgres else "memory"}
