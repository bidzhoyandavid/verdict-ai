"""Dataset and company-doc uploads.

The uploaded file is converted into an immutable dataset right away, so the
graph never touches user-supplied paths.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.config import settings
from api.deps import UserDep
from api.schemas import DatasetOut
from app.datasets import load_meta, put_file

router = APIRouter(prefix="/files", tags=["files"])

ALLOWED_SUFFIXES = {".csv", ".parquet"}
MAX_BYTES = 200 * 1024 * 1024


@router.post("/datasets", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def upload_dataset(user: UserDep, file: UploadFile = File(...)) -> DatasetOut:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"expected one of {sorted(ALLOWED_SUFFIXES)}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        written = 0
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file too large")
            tmp.write(chunk)

    try:
        dataset_id = put_file(tmp_path, company_id=user.company_id)
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"could not read file: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    meta = load_meta(dataset_id)
    return DatasetOut(dataset_id=dataset_id, n_rows=meta["n_rows"], columns=meta["columns"])


def _company_doc_path(company_id: str) -> Path:
    return Path(settings.uploads_dir) / "company_docs" / company_id / "company.md"


@router.get("/company-doc")
def list_company_docs(user: UserDep) -> list[dict]:
    """At most one doc per company for now — versioning comes with the
    onboarding rework, the list shape is already what the UI expects."""
    path = _company_doc_path(user.company_id)
    if not path.exists():
        return []
    updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return [{"version": "v1", "filename": path.name, "updatedAt": updated.strftime("%d.%m.%Y")}]


@router.post("/company-doc", status_code=status.HTTP_201_CREATED)
async def upload_company_doc(user: UserDep, file: UploadFile = File(...)) -> dict:
    """Stores the onboarding .md as-is; the onboarding flow reads it from here."""
    if Path(file.filename or "").suffix.lower() != ".md":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "expected a .md file")

    target = _company_doc_path(user.company_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    return {"filename": file.filename, "size": target.stat().st_size}
