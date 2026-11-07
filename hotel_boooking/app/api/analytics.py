from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.anomaly_detector import anomaly_detector_service
from app.services.drift_detector import drift_detector_service
from app.services.explainer import explainer_service
from app.services.forecaster import forecaster_service
from app.services.optimizer import optimizer_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["Advanced Analytics & Forecasting"],
)


# --------------------------------------------------------------------------
# Request Schemas
# --------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    horizon: int = Field(30, description="Forecast horizon in days (7, 30, or 90)", example=30)
    start_date: Optional[str] = Field(None, description="Optional start date in YYYY-MM-DD format", example="2027-01-01")


class PricingOptimizationRequest(BaseModel):
    base_adr: Optional[float] = Field(None, description="Baseline ADR in dollars", example=150.0)
    base_occupancy: Optional[float] = Field(None, description="Baseline Occupancy Rate (0.0 to 1.0)", example=0.70)
    total_rooms: Optional[int] = Field(None, description="Total hotel room capacity", example=120)
    fixed_costs: Optional[float] = Field(None, description="Daily fixed overhead cost", example=2500.0)
    var_cost_per_room: Optional[float] = Field(None, description="Variable cost per occupied room night", example=15.0)
    custom_elasticity: Optional[float] = Field(None, description="Optional custom demand elasticity (e.g. -0.35)", example=-0.35)


class ExplainRequest(BaseModel):
    Date: Optional[str] = Field("2026-07-01", description="Reservation date (YYYY-MM-DD)", example="2026-07-01")
    Occupancy_Rate: Optional[float] = Field(0.75, example=0.75)
    ADR: Optional[float] = Field(165.0, example=165.0)
    Available_Rooms: Optional[int] = Field(120, example=120)
    Reserved_Rooms: Optional[int] = Field(90, example=90)
    Total_Costs: Optional[float] = Field(2800.0, example=2800.0)
    Booking_Channel: Optional[str] = Field("Online", example="Online")
    Guest_Type: Optional[str] = Field("Transient", example="Transient")
    Market_Segment: Optional[str] = Field("Leisure", example="Leisure")
    Guest_Country: Optional[str] = Field("US", example="US")
    Season: Optional[str] = Field("Summer", example="Summer")


class DriftRequest(BaseModel):
    candidate_file_path: Optional[str] = Field(None, description="Optional file path to uploaded candidate dataset")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@router.post("/forecast", summary="Revenue Forecasting (7, 30, 90 Days)")
def forecast_revenue(request: ForecastRequest):
    """
    Chronological revenue forecasting with 95% confidence bounds
    and historical backtesting evaluation metrics (RMSE, MAE, MAPE).
    """
    try:
        result = forecaster_service.forecast_revenue(
            horizon=request.horizon,
            start_date=request.start_date,
        )
        return {"status": "Success", **result}
    except Exception as e:
        logger.error(f"Revenue forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/occupancy-forecast", summary="Occupancy Rate Forecasting")
def forecast_occupancy(request: ForecastRequest):
    """
    Forecasts daily occupancy rate bounded in [0, 1.0] with
    weekday vs weekend patterns and historical seasonality metrics.
    """
    try:
        result = forecaster_service.forecast_occupancy(
            horizon=request.horizon,
            start_date=request.start_date,
        )
        return {"status": "Success", **result}
    except Exception as e:
        logger.error(f"Occupancy forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pricing-optimizer", summary="What-If ADR & Revenue Optimization")
def optimize_pricing(request: PricingOptimizationRequest):
    """
    Simulates demand shifts across +/-30% ADR change based on empirical
    price elasticity of demand, calculating revenue and profit maximizing scenarios.
    """
    try:
        result = optimizer_service.simulate_pricing_matrix(
            base_adr=request.base_adr,
            base_occupancy=request.base_occupancy,
            total_rooms=request.total_rooms,
            fixed_costs=request.fixed_costs,
            var_cost_per_room=request.var_cost_per_room,
            custom_elasticity=request.custom_elasticity,
        )
        return {"status": "Success", **result}
    except Exception as e:
        logger.error(f"Pricing optimizer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explain/global", summary="Global SHAP Feature Importance")
def explain_global(
    max_samples: int = Query(60, ge=10, le=200, description="Background sample count for SHAP")
):
    """
    Returns global feature importance ranked by mean absolute SHAP value
    for the active tree-based regression model.
    """
    try:
        result = explainer_service.get_global_importance(max_background_samples=max_samples)
        return {"status": "Success", **result}
    except Exception as e:
        logger.error(f"Global SHAP explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", summary="Instance Prediction SHAP Explanation")
def explain_instance(request: ExplainRequest):
    """
    Computes local SHAP decomposition for a specific reservation/hotel operational record,
    breaking down positive and negative feature contributions relative to expected base value.
    """
    try:
        input_payload = request.model_dump(exclude_none=True)
        result = explainer_service.explain_instance(input_payload)
        return {"status": "Success", **result}
    except Exception as e:
        logger.error(f"Instance SHAP explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies", summary="Revenue, Occupancy & ADR Anomaly Detection")
def detect_anomalies(
    contamination: float = Query(0.05, ge=0.01, le=0.20, description="Expected anomaly contamination rate"),
    z_threshold: float = Query(2.5, ge=1.5, le=4.0, description="Z-score statistical threshold"),
):
    """
    Scans historical operational records to identify revenue, occupancy, and ADR anomalies
    using joint statistical (IQR/Z-Score) and machine learning (IsolationForest) methods.
    """
    try:
        result = anomaly_detector_service.detect_anomalies(
            contamination=contamination,
            z_threshold=z_threshold,
        )
        return {"status": "Success", **result}
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift", summary="Data Drift Assessment")
def assess_drift(request: Optional[DriftRequest] = None):
    """
    Performs Kolmogorov-Smirnov tests, Wasserstein distance, categorical PSI,
    and missing-rate comparisons against baseline training distributions.
    """
    try:
        file_path = request.candidate_file_path if request else None
        result = drift_detector_service.assess_dataset_drift(candidate_file_path=file_path)
        return {"status": "Success", **result}
    except Exception as e:
        logger.error(f"Data drift assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

