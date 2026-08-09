"""Test isolation: dataset store and API database both go to temp locations.

The database URL must be set before `api.db` is imported, hence module level.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="verdict-tests-"))
os.environ.setdefault("VERDICT_DATABASE_URL", f"sqlite:///{(_TMP_ROOT / 'api.db').as_posix()}")
os.environ.setdefault("VERDICT_UPLOADS_DIR", str(_TMP_ROOT / "uploads"))

from app import datasets  # noqa: E402  (must follow the env setup above)


@pytest.fixture(autouse=True)
def isolated_dataset_store(tmp_path, monkeypatch):
    monkeypatch.setenv("VERDICT_DATASETS_DIR", str(tmp_path / "datasets"))
    datasets.clear_cache()
    yield
    datasets.clear_cache()
