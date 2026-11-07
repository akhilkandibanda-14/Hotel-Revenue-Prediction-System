from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.business_recommender import business_recommender
from app.services.cancellation_analytics import cancellation_service
from app.services.cancellation_model import cancellation_model_service
from app.services.competitor_pricing import competitor_pricing_service
from app.services.demand_forecaster import demand_forecaster
from app.services.prebooking_predictor import prebooking_predictor
from app.utils import config
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/intelligence",
    tags=["Phase 4: Revenue Intelligence & Demand Prediction"],
)


# --------------------------------------------------------------------------
# Request Schemas
# --------------------------------------------------------------------------
class DemandForecastRequest(BaseModel):
    horizon: int = Field(30, description="Forecast horizon in days (7, 30, or 90)", example=30)
    start_date: Optional[str] = Field(None, description="Optional start date in YYYY-MM-DD format", example="2027-01-01")


class PrebookingEstimateRequest(BaseModel):
    check_in_date: str = Field(..., description="Arrival date (YYYY-MM-DD)", example="2026-07-15")
    check_out_date: str = Field(..., description="Departure date (YYYY-MM-DD)", example="2026-07-18")
    booking_date: Optional[str] = Field(None, description="Date reservation was created (YYYY-MM-DD)", example="2026-06-30")
    quoted_adr: Optional[float] = Field(None, ge=10.0, description="Offered room rate per night in dollars", example=175.0)
    rooms_count: int = Field(1, ge=1, le=50, description="Number of rooms requested", example=1)
    guests_count: int = Field(2, ge=1, le=20, description="Total guests in reservation", example=2)
    booking_channel: str = Field("Direct", description="Distribution channel (Direct, Online, OTA, Corporate)", example="Direct")
    market_segment: str = Field("Leisure", description="Market classification", example="Leisure")
    room_type: str = Field("Standard", description="Room tier (Standard, Deluxe, Executive Suite, Presidential Suite)", example="Standard")


class CompetitorEntryRequest(BaseModel):
    date_str: str = Field(..., description="Observation date (YYYY-MM-DD)", example="2026-07-01")
    competitor_name: str = Field(..., description="Name of benchmark property", example="Grand Bay Luxury Resort")
    competitor_adr: float = Field(..., ge=1.0, description="Competitor rate per night", example=190.0)
    our_adr: float = Field(..., ge=1.0, description="Our property rate per night", example=175.0)
    room_type: Optional[str] = Field("Standard Deluxe", description="Compared room category", example="Standard Deluxe")
    notes: Optional[str] = Field("", description="Context or observation notes", example="Promotional summer weekend rate")


class CompetitorEvalRequest(BaseModel):
    our_adr: Optional[float] = Field(None, ge=1.0, example=175.0)
    competitor_adr: Optional[float] = Field(None, ge=1.0, example=190.0)


class CancellationPredictRequest(BaseModel):
    lead_time: int = Field(..., ge=0, le=1000, description="Days between reservation booking and check-in date", example=45)
    arrival_date: str = Field(..., description="Check-in arrival date in YYYY-MM-DD format", example="2026-08-15")
    adr: float = Field(..., ge=0.0, description="Average daily rate quoted in dollars", example=165.0)
    booking_channel: str = Field("Direct", description="Distribution channel (Direct, Corporate, TA/TO, GDS)", example="Direct")
    customer_type: str = Field("Transient", description="Customer segmentation (Transient, Contract, Transient-Party, Group)", example="Transient")
    deposit_type: str = Field("No Deposit", description="Deposit terms (No Deposit, Non Refund, Refundable)", example="No Deposit")
    stays_in_weekend_nights: int = Field(1, ge=0, le=30, description="Number of weekend nights in stay", example=1)
    stays_in_week_nights: int = Field(2, ge=0, le=60, description="Number of weekday nights in stay", example=2)
    adults: int = Field(2, ge=1, le=20, description="Number of adult guests", example=2)
    children: int = Field(0, ge=0, le=10, description="Number of children", example=0)
    babies: int = Field(0, ge=0, le=10, description="Number of infant guests", example=0)
    is_repeated_guest: int = Field(0, ge=0, le=1, description="Binary flag: 1 if guest has prior completed stays", example=0)
    previous_cancellations: int = Field(0, ge=0, le=50, description="Count of prior cancellations on record for this guest", example=0)
    previous_bookings_not_canceled: int = Field(0, ge=0, le=100, description="Count of prior successful non-canceled stays", example=0)
    booking_changes: int = Field(0, ge=0, le=20, description="Number of changes made to reservation prior to arrival", example=0)
    days_in_waiting_list: int = Field(0, ge=0, le=500, description="Days reservation sat on waiting list", example=0)
    required_car_parking_spaces: int = Field(0, ge=0, le=10, description="Number of vehicle parking spaces requested", example=0)
    total_of_special_requests: int = Field(0, ge=0, le=10, description="Count of special requests submitted (high-intent indicator)", example=1)
    hotel: str = Field("City Hotel", description="Property category (Resort Hotel or City Hotel)", example="City Hotel")
    market_segment: str = Field("Online TA", description="Market classification (Direct, Corporate, Online TA, Offline TA/TO, Groups)", example="Online TA")
    reserved_room_type: str = Field("A", description="Requested room code / category", example="A")
    meal: str = Field("BB", description="Meal plan (BB, HB, FB, SC)", example="BB")


class CancellationThresholdRequest(BaseModel):
    cost_fp: float = Field(45.0, ge=1.0, description="Cost of false positive intervention ($)", example=45.0)
    cost_fn: float = Field(160.0, ge=1.0, description="Gross cost of uncaptured cancellation ($)", example=160.0)
    replacement_rate: float = Field(0.25, ge=0.0, le=1.0, description="Probability of rebooking / replacing room", example=0.25)
    split: str = Field("val", description="Evaluation dataset split ('val' or 'test')", example="val")



# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@router.post("/demand-forecast", summary="Hotel Demand & Room Nights Forecasting")
def forecast_demand(request: DemandForecastRequest):
    """
    Forecasts expected room demand, room nights, and occupancy rate across 7, 30, or 90 days.
    Provides chronological holdout benchmarks against Moving-Average and Seasonal-Naive baselines.
    """
    try:
        result = demand_forecaster.forecast_demand(
            horizon=request.horizon,
            start_date=request.start_date,
        )
        return {"status": "Success", **result}
    except Exception as e:
        logger.error(f"Demand forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prebooking-predict", summary="Pre-Booking Expected Revenue Estimation")
def predict_prebooking(request: PrebookingEstimateRequest):
    """
    Estimates expected reservation value using information available BEFORE booking.
    Strictly prohibits realized variables (no RevPAR, no actual occupancy leakage).
    """
    try:
        result = prebooking_predictor.estimate_prebooking_value(
            check_in_date=request.check_in_date,
            check_out_date=request.check_out_date,
            booking_date=request.booking_date,
            quoted_adr=request.quoted_adr,
            rooms_count=request.rooms_count,
            guests_count=request.guests_count,
            booking_channel=request.booking_channel,
            market_segment=request.market_segment,
            room_type=request.room_type,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Prebooking estimation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prebooking-audit", summary="Booking-Level Schema & Leakage Audit")
def audit_prebooking_data():
    """
    Audits active training dataset to determine whether it contains individual
    reservation records or aggregated operational metrics.
    """
    try:
        active_path = config.MERGED_FOLDER / "active_dataset.csv"
        if not active_path.exists():
            active_path = config.BASE_DIR / "sample_training_dataset.csv"
        df = pd.read_csv(active_path)
        return prebooking_predictor.audit_dataset_for_booking_level(df)
    except Exception as e:
        logger.error(f"Prebooking audit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cancellation-analytics", summary="Cancellation & Slippage Analytics")
def get_cancellation_analytics():
    """
    Computes cancellation rates, no-show rates, and slippage across distribution channels and seasons.
    """
    try:
        result = cancellation_service.analyze_cancellations()
        return result
    except Exception as e:
        logger.error(f"Cancellation analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancellation-predict", summary="Individual Booking Cancellation Risk Prediction")
def predict_cancellation(request: CancellationPredictRequest):
    """
    Predicts probability of reservation cancellation using the trained Random Forest classifier
    validated chronologically on the Antonio et al. (2019) public research dataset.
    """
    try:
        res = cancellation_model_service.predict_cancellation_risk(
            lead_time=request.lead_time,
            arrival_date=request.arrival_date,
            adr=request.adr,
            booking_channel=request.booking_channel,
            customer_type=request.customer_type,
            deposit_type=request.deposit_type,
            stays_in_weekend_nights=request.stays_in_weekend_nights,
            stays_in_week_nights=request.stays_in_week_nights,
            adults=request.adults,
            children=request.children,
            babies=request.babies,
            is_repeated_guest=request.is_repeated_guest,
            previous_cancellations=request.previous_cancellations,
            previous_bookings_not_canceled=request.previous_bookings_not_canceled,
            booking_changes=request.booking_changes,
            days_in_waiting_list=request.days_in_waiting_list,
            required_car_parking_spaces=request.required_car_parking_spaces,
            total_of_special_requests=request.total_of_special_requests,
            hotel=request.hotel,
            market_segment=request.market_segment,
            reserved_room_type=request.reserved_room_type,
            meal=request.meal,
        )
        return res
    except Exception as e:
        logger.error(f"Cancellation prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cancellation-model-performance", summary="Cancellation Classification Model Benchmark Performance")
def get_cancellation_model_performance():
    """
    Returns multi-model benchmark evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score,
    confusion matrix) for the 4 algorithms evaluated on chronological holdout data.
    """
    try:
        res = cancellation_model_service.train_and_benchmark(force_retrain=False)
        return res
    except Exception as e:
        logger.error(f"Cancellation benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cancellation-calibration", summary="Probability Calibration Evaluation & Curves")
def get_cancellation_calibration():
    """
    Returns reliability diagram curves (10 bins), Brier scores, and Expected Calibration Error (ECE)
    comparing uncalibrated probabilities against Platt (Sigmoid) and Isotonic calibration
    evaluated strictly on chronological validation and holdout test splits (zero leakage).
    """
    try:
        return cancellation_model_service.get_calibration_metrics()
    except Exception as e:
        logger.error(f"Cancellation calibration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancellation-threshold-analysis", summary="Cost-Optimal Decision Threshold Analysis")
def evaluate_cancellation_threshold(request: Optional[CancellationThresholdRequest] = None):
    """
    Evaluates decision threshold cost grid (0.05 to 0.95) based on hotel operational cost matrix
    (Cost of False Positive vs Cost of False Negative with room replacement adjustments).
    Returns cost-optimal operating threshold and business savings compared against standard 0.50.
    """
    try:
        req = request or CancellationThresholdRequest()
        return cancellation_model_service.evaluate_threshold_optimization(
            cost_fp=req.cost_fp,
            cost_fn=req.cost_fn,
            replacement_rate=req.replacement_rate,
            split=req.split,
        )
    except Exception as e:
        logger.error(f"Threshold analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancellation-explain", summary="SHAP Explainability for Reservation Cancellation Risk")
def explain_cancellation(request: CancellationPredictRequest):
    """
    Decomposes cancellation probability into exact feature attributions using SHAP TreeExplainer.
    Validates additive consistency (Base Value + sum(SHAP) == Predicted Probability) and provides
    top risk drivers, protective factors, and an associational statistical disclaimer.
    """
    try:
        data = request.dict()
        return cancellation_model_service.explain_cancellation(data)
    except Exception as e:
        logger.error(f"Cancellation SHAP explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancellation-what-if", summary="Cancellation Policy & Commercial What-If Simulator")
def simulate_cancellation_what_if(request: CancellationPredictRequest):
    """
    Simulates changes in cancellation probability across operational levers (deposit terms, lead time,
    booking channel, ADR price adjustments, and special requests) with clear causal disclaimers.
    """
    try:
        data = request.dict()
        return cancellation_model_service.simulate_what_if(data)
    except Exception as e:
        logger.error(f"Cancellation what-if error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cancellation-segment-analysis", summary="Empirical Cancellation Risk by Segment")
def get_cancellation_segment_analysis():
    """
    Returns empirical cancellation rates, volume, and average lead times across key segments
    (lead-time tiers, distribution channels, customer segments, deposit policies, months, hotel types)
    computed from the Antonio et al. (2019) public research dataset.
    """
    try:
        return cancellation_model_service.get_segment_risk_analytics()
    except Exception as e:
        logger.error(f"Cancellation segment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/competitor-analysis", summary="Competitor Price Position Evaluation")
def evaluate_competitor(request: Optional[CompetitorEvalRequest] = None):
    """
    Evaluates Market Price Position Index (MPI) and pricing gap against benchmark competitors.
    """
    try:
        our_adr = request.our_adr if request else None
        comp_adr = request.competitor_adr if request else None
        result = competitor_pricing_service.evaluate_competitive_position(
            our_adr=our_adr,
            competitor_adr=comp_adr,
        )
        return result
    except Exception as e:
        logger.error(f"Competitor analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitor-record", summary="Log Manual Competitor Benchmark Entry")
def add_competitor_record(entry: CompetitorEntryRequest):
    """
    Saves a manual competitor ADR benchmark observation for tracking.
    """
    try:
        res = competitor_pricing_service.add_competitor_entry(
            date_str=entry.date_str,
            competitor_name=entry.competitor_name,
            competitor_adr=entry.competitor_adr,
            our_adr=entry.our_adr,
            room_type=entry.room_type or "Standard Deluxe",
            notes=entry.notes or "",
        )
        return {"status": "Success", "entry": res}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error saving competitor record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitor-records", summary="Get All Stored Competitor Benchmarks")
def get_competitor_records():
    """
    Returns all logged competitor rate benchmark observations.
    """
    try:
        records = competitor_pricing_service.load_benchmarks()
        return {"status": "Success", "count": len(records), "records": records}
    except Exception as e:
        logger.error(f"Error reading competitor records: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business-recommendations", summary="Prioritized Revenue Management Recommendations")
def get_recommendations():
    """
    Generates actionable business recommendations synthesized from forecasting, pricing elasticity,
    operational anomalies, drift tests, and distribution channel metrics.
    """
    try:
        result = business_recommender.generate_recommendations()
        return result
    except Exception as e:
        logger.error(f"Business recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

