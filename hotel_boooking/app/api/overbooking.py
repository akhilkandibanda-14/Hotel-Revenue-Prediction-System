from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.overbooking_optimizer import overbooking_optimizer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/overbooking",
    tags=["Phase 5: Intelligent Overbooking Optimization"],
)


class OverbookingOptimizeRequest(BaseModel):
    target_date: str = Field(..., description="Target arrival date in YYYY-MM-DD format", example="2026-08-15")
    room_category: str = Field("Standard Room", description="Target room category", example="Standard Room")
    custom_adr: Optional[float] = Field(None, ge=10.0, description="Override Average Daily Rate ($)", example=140.0)
    custom_walk_cost: Optional[float] = Field(None, ge=10.0, description="Override guest relocation / walk penalty ($)", example=280.0)
    max_walk_risk_pct: float = Field(2.0, ge=0.1, le=20.0, description="Maximum acceptable walk probability constraint (%)", example=2.0)
    max_additional_bookings: int = Field(30, ge=1, le=100, description="Search horizon of candidate additional bookings", example=30)


@router.get("/forecast", summary="Expected Cancellations & Arrival Probability Distribution")
def get_cancellation_forecast(
    target_date: str = Query(..., description="Target arrival date in YYYY-MM-DD format"),
    room_category: Optional[str] = Query(None, description="Optional room category filter (omit for whole hotel)"),
):
    """
    Computes expected cancellations, variance, and exact Poisson-Binomial arrival distribution
    for confirmed bookings on a specific arrival date.
    """
    try:
        return overbooking_optimizer.compute_cancellation_forecast(
            target_date=target_date,
            room_category=room_category,
        )
    except Exception as e:
        logger.error(f"Error computing cancellation forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", summary="Constrained Overbooking Optimization for Room Category")
def optimize_overbooking(request: OverbookingOptimizeRequest):
    """
    Finds the cost-optimal booking limit B* that maximizes Expected Net Contribution:
        E[ENC(B)] = E[Room Revenue] - E[Relocation Costs]
    Subject to:
        P(Arrivals > Effective Capacity) <= max_walk_risk_pct.
    Returns safe additional bookings, expected gain, and full candidate evaluation grid.
    """
    try:
        return overbooking_optimizer.optimize_overbooking(
            target_date=request.target_date,
            room_category=request.room_category,
            custom_adr=request.custom_adr,
            custom_walk_cost=request.custom_walk_cost,
            max_walk_risk_pct=request.max_walk_risk_pct,
            max_additional_bookings=request.max_additional_bookings,
        )
    except Exception as e:
        logger.error(f"Error optimizing overbooking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenario", summary="Overbooking Policy Scenario Simulator")
def simulate_overbooking_scenario(request: OverbookingOptimizeRequest):
    """
    Simulates overbooking trade-offs under alternative hotel management assumptions
    (varying ADR, relocation costs, and service-level walk risk tolerance).
    """
    try:
        return overbooking_optimizer.simulate_scenario(request.model_dump())
    except Exception as e:
        logger.error(f"Error simulating overbooking scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", summary="Portfolio-Wide Overbooking Recommendations")
def get_portfolio_recommendations(
    target_date: str = Query(..., description="Target arrival date in YYYY-MM-DD format"),
):
    """
    Generates synchronized overbooking limits, safe additional bookings, and expected revenue gains
    across all room categories for a specific date.
    """
    try:
        return overbooking_optimizer.get_portfolio_recommendations(target_date=target_date)
    except Exception as e:
        logger.error(f"Error retrieving portfolio recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", summary="Overbooking Performance Benchmark Across Horizon")
def get_overbooking_performance():
    """
    Simulates the incremental net revenue value of dynamic overbooking vs zero-overbooking
    across a multi-day booking horizon.
    """
    try:
        return overbooking_optimizer.get_overbooking_performance()
    except Exception as e:
        logger.error(f"Error retrieving overbooking performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

