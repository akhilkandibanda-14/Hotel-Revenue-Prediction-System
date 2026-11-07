from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.model_manager import (
    list_available_versions,
    load_model_comparison,
    load_model_metadata,
    rollback_to_version,
)
from app.services.predictor import RevenuePredictor

router = APIRouter(
    prefix="/model",
    tags=["Model Management"]
)


@router.get("/info")
def get_model_info():
    """
    Returns metadata of the currently deployed production model.
    """
    metadata = load_model_metadata()
    if not metadata:
        return {
            "status": "No production model metadata available",
            "metadata": {}
        }
    return {
        "status": "Success",
        "metadata": metadata
    }


@router.get("/comparison")
def get_model_comparison():
    """
    Returns latest multi-model benchmark evaluation results across all 5 algorithms.
    """
    comparison = load_model_comparison()
    if not comparison:
        return {
            "status": "No comparison results found",
            "comparison": {}
        }
    return {
        "status": "Success",
        "comparison": comparison
    }


@router.get("/versions")
def get_model_versions():
    """
    Returns all available model versions and marks the currently deployed active version.
    """
    versions = list_available_versions()
    current = load_model_metadata().get("version", "v001")
    return {
        "status": "Success",
        "current_version": current,
        "available_versions": versions,
    }


@router.post("/rollback/{version}")
def rollback_model(version: str):
    """
    Rolls back the active production model to an earlier version.
    """
    try:
        updated_meta = rollback_to_version(version)
        # Force predictor reload
        predictor = RevenuePredictor()
        predictor.load_assets()

        return {
            "status": "Success",
            "message": f"Successfully rolled back production model to {version}",
            "metadata": updated_meta,
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
