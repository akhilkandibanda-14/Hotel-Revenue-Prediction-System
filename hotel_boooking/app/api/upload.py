from datetime import datetime, timezone
from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.utils.config import UPLOADED_FOLDER
from app.services.dataset_manager import ensure_dataset_directories


router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):

    ensure_dataset_directories()

    original_name = Path(file.filename or "dataset.csv").name

    if not original_name.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV datasets are supported."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    file_path = UPLOADED_FOLDER / f"{timestamp}_{original_name}"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "accepted",
        "message": "Dataset uploaded. Background training will process it.",
        "filename": original_name,
        "saved_to": str(file_path)
    }