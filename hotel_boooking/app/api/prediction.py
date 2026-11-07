from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.predictor import RevenuePredictor

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

predictor = RevenuePredictor()


class PredictionRequest(BaseModel):
    Date: str = Field(..., description="Target date in YYYY-MM-DD format", example="2026-06-30")
    strict: Optional[bool] = Field(
        False,
        description="If True, rejects requests missing key operational fields. If False, applies empirical seasonal profile."
    )
    # Optional operational booking features
    Month: Optional[str] = Field(None, example="June")
    Booking_Channel: Optional[str] = Field(None, example="Online")
    Guest_Type: Optional[str] = Field(None, example="Transient")
    Market_Segment: Optional[str] = Field(None, example="Leisure")
    Guest_Country: Optional[str] = Field(None, example="US")
    Season: Optional[str] = Field(None, example="Summer")

    Occupancy_Rate: Optional[float] = Field(None, ge=0.0, le=100.0, example=0.85)
    ADR: Optional[float] = Field(None, ge=0.0, example=150.0)
    RevPAR: Optional[float] = Field(None, ge=0.0, example=127.5)

    Available_Rooms: Optional[int] = Field(None, ge=0, example=100)
    Reserved_Rooms: Optional[int] = Field(None, ge=0, example=85)

    Fixed_Costs: Optional[float] = Field(None, ge=0.0, example=2000.0)
    Variable_Costs: Optional[float] = Field(None, ge=0.0, example=500.0)
    Total_Costs: Optional[float] = Field(None, ge=0.0, example=2500.0)


@router.post("/")
def predict(request: PredictionRequest):
    """
    Predicts expected hotel revenue for a given date and booking parameters.
    Backward-compatible: Accepts date-only requests via empirical seasonal profiling.
    """
    try:
        payload = {k: v for k, v in request.model_dump().items() if v is not None and k != "strict"}
        strict_mode = bool(request.strict)

        result = predictor.predict_detailed(payload, strict=strict_mode)

        return {
            "status": "Success",
            "Date": request.Date,
            "Predicted_Revenue": result["Predicted_Revenue"],
            "Model_Version": result.get("model_version", "v001"),
            "Algorithm": result.get("algorithm", "Unknown"),
            "Prediction_Mode": result.get("prediction_mode", "full_features"),
        }

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        return {
            "status": "Failed",
            "message": str(exc)
        }
