from datetime import datetime, timezone
from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.utils.config import UPLOADED_FOLDER
from app.services.dataset_manager import (
    ensure_dataset_directories,
    list_pending_datasets,
    load_training_state,
)


router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("/status")
def get_dataset_training_status():
    """
    Returns continuous training status: processed files, pending queue, and latest training output.
    """
    state = load_training_state()
    pending = [p.name for p in list_pending_datasets()]
    processed_count = len(state.get("processed", {}))
    failed_count = len(state.get("failed", {}))

    latest_processed = None
    if state.get("processed"):
        sorted_processed = sorted(
            state["processed"].items(),
            key=lambda item: item[1].get("processed_at", ""),
            reverse=True,
        )
        if sorted_processed:
            latest_processed = {
                "filename": sorted_processed[0][0],
                **sorted_processed[0][1],
            }

    return {
        "status": "Success",
        "pending_count": len(pending),
        "pending_files": pending,
        "processed_count": processed_count,
        "failed_count": failed_count,
        "latest_training": latest_processed,
        "state": state,
    }


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