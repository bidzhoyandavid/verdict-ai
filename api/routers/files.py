"""Dataset and company-doc uploads.

The uploaded file is converted into an immutable dataset right away, so the
graph never touches user-supplied paths.
"""

from __future__ import annotations

import shutil
import tempfile
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


@router.post("/company-doc", status_code=status.HTTP_201_CREATED)
async def upload_company_doc(user: UserDep, file: UploadFile = File(...)) -> dict:
    """Stores the onboarding .md as-is; the onboarding flow reads it from here."""
    if Path(file.filename or "").suffix.lower() != ".md":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "expected a .md file")

    target_dir = Path(settings.uploads_dir) / "company_docs" / user.company_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "company.md"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    return {"filename": file.filename, "size": target.stat().st_size}
