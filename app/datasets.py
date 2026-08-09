"""Content store for experiment datasets.

Datasets live outside the graph state: the state carries a `dataset_id`,
nodes materialize the DataFrame on demand. Keeping a multi-MB DataFrame in
the state would mean re-serializing the whole dataset into the checkpointer
on every single node transition.

Datasets are immutable — a derived dataset (e.g. after outlier treatment)
gets its own id. Callers must treat a loaded DataFrame as read-only: it is
cached and shared across nodes within a process.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from abex.data import loaders

_ID_ALPHABET = set("0123456789abcdef")
_CACHE: dict[str, pd.DataFrame] = {}
_CACHE_MAX = 8


def datasets_root() -> Path:
    """Root of the dataset store. Overridable so tests and the API can point
    at a tmp dir / a mounted volume without touching call sites."""
    root = Path(os.environ.get("VERDICT_DATASETS_DIR", ".data/datasets"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_id(dataset_id: str) -> str:
    if not dataset_id or set(dataset_id) - _ID_ALPHABET:
        raise ValueError(f"malformed dataset_id: {dataset_id!r}")
    return dataset_id


def dataset_path(dataset_id: str) -> Path:
    return datasets_root() / f"{_validate_id(dataset_id)}.parquet"


def meta_path(dataset_id: str) -> Path:
    return datasets_root() / f"{_validate_id(dataset_id)}.json"


def put_dataframe(
    df: pd.DataFrame,
    *,
    company_id: str,
    source: str = "file",
    origin: str | None = None,
) -> str:
    """Persist `df` as a new immutable dataset and return its id.

    `origin` records the dataset this one was derived from, if any.
    """
    dataset_id = uuid.uuid4().hex
    # Row order is the contract between validate_profile (which computes the
    # outlier mask positionally) and outlier_review (which applies it).
    frame = df.reset_index(drop=True)
    frame.to_parquet(dataset_path(dataset_id), index=False)

    meta = {
        "dataset_id": dataset_id,
        "company_id": company_id,
        "source": source,
        "origin": origin,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
    }
    meta_path(dataset_id).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _CACHE[dataset_id] = frame
    _trim_cache()
    return dataset_id


def put_file(path: str | Path, *, company_id: str) -> str:
    """Ingest an uploaded CSV/Parquet file into the store."""
    return put_dataframe(load_file(path), company_id=company_id, source="file")


def load_file(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet file via abex.data.loaders based on extension."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return loaders.load_parquet(path)
    return loaders.load_csv(path)


def load_dataset(dataset_id: str) -> pd.DataFrame:
    """Materialize a dataset. Cached per process — do not mutate the result."""
    cached = _CACHE.get(dataset_id)
    if cached is not None:
        return cached

    path = dataset_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"dataset {dataset_id} not found at {path}")
    df = pd.read_parquet(path)
    _CACHE[dataset_id] = df
    _trim_cache()
    return df


def load_meta(dataset_id: str) -> dict[str, Any]:
    return json.loads(meta_path(dataset_id).read_text(encoding="utf-8"))


def active_dataset_id(state: Mapping[str, Any]) -> str | None:
    """Dataset downstream nodes should read: the outlier-treated one if
    outlier review produced it, otherwise the originally loaded one."""
    return state.get("treated_dataset_id") or state.get("dataset_id")


def load_active(state: Mapping[str, Any]) -> pd.DataFrame:
    dataset_id = active_dataset_id(state)
    if dataset_id is None:
        raise ValueError("dataset_id must be set before the graph runs")
    return load_dataset(dataset_id)


def _trim_cache() -> None:
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))


def clear_cache() -> None:
    _CACHE.clear()
