from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.inventory_manager import room_inventory_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/inventory",
    tags=["Phase 5: Room Inventory Management"],
)


class RoomCategoryUpdateRequest(BaseModel):
    room_category: str = Field(..., description="Name of room category", example="Standard Room")
    total_physical_rooms: int = Field(..., ge=1, le=1000, description="Total physical room capacity", example=100)
    rooms_out_of_service: int = Field(0, ge=0, le=500, description="Rooms out of order / maintenance", example=2)
    base_adr: Optional[float] = Field(None, ge=10.0, description="Base average daily rate ($)", example=140.0)
    walk_relocation_cost: Optional[float] = Field(None, ge=10.0, description="Relocation cost if guest is walked ($)", example=280.0)


class IndividualReservationCreateRequest(BaseModel):
    guest_name: str = Field(..., description="Guest or booker name", example="John Doe")
    room_category: str = Field("Standard Room", description="Room category", example="Standard Room")
    arrival_date: str = Field(..., description="Check-in arrival date (YYYY-MM-DD)", example="2026-08-15")
    departure_date: str = Field(..., description="Check-out departure date (YYYY-MM-DD)", example="2026-08-18")
    stay_nights: int = Field(1, ge=1, le=60, description="Duration of stay in nights", example=3)
    booked_adr: float = Field(..., ge=10.0, description="Quoted room rate per night ($)", example=150.0)
    booking_channel: str = Field("Direct", description="Distribution channel (Direct, Online TA, Corporate, GDS)", example="Direct")
    deposit_type: str = Field("No Deposit", description="Deposit terms (No Deposit, Non Refund, Refundable)", example="No Deposit")
    customer_type: str = Field("Transient", description="Customer classification (Transient, Contract, Group)", example="Transient")
    lead_time: int = Field(14, ge=0, le=730, description="Days booked in advance of arrival", example=14)
    adults: int = Field(2, ge=1, le=10, description="Number of adults", example=2)
    children: int = Field(0, ge=0, le=10, description="Number of children", example=0)
    previous_cancellations: int = Field(0, ge=0, le=20, description="Prior cancellations on record", example=0)
    total_of_special_requests: int = Field(0, ge=0, le=10, description="Special requests count", example=1)
    required_car_parking_spaces: int = Field(0, ge=0, le=5, description="Parking spaces needed", example=0)


@router.get("/availability", summary="Room Inventory & Available Capacity by Date")
def get_room_availability(
    target_date: Optional[str] = Query(None, description="Target arrival/stay date in YYYY-MM-DD format (defaults to tomorrow)"),
    room_category: Optional[str] = Query(None, description="Optional room category filter"),
):
    """
    Returns physical rooms, out-of-service maintenance, confirmed bookings, and available rooms
    for a specific date. Avoids double-counting on check-out departure dates.
    """
    try:
        if not target_date:
            target_date = date.today().strftime("%Y-%m-%d")
        return room_inventory_manager.get_availability_by_date(
            target_date_str=target_date,
            room_category=room_category,
        )
    except Exception as e:
        logger.error(f"Error retrieving room availability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rooms", summary="Create or Update Room Category Inventory Configuration")
def update_room_inventory(request: RoomCategoryUpdateRequest):
    """
    Configures total physical rooms, maintenance out-of-service rooms, base rate,
    and walk relocation penalty for a room category.
    """
    try:
        updated = room_inventory_manager.update_room_category(
            room_category=request.room_category,
            total_physical_rooms=request.total_physical_rooms,
            rooms_out_of_service=request.rooms_out_of_service,
            base_adr=request.base_adr,
            walk_relocation_cost=request.walk_relocation_cost,
        )
        return {"status": "Success", "message": f"Inventory for {request.room_category} updated successfully.", "category": updated}
    except Exception as e:
        logger.error(f"Error updating room inventory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/occupancy", summary="Day-by-Day Room Occupancy & Availability Timeline")
def get_occupancy_timeline(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD, defaults to today)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD, defaults to +14 days)"),
    room_category: Optional[str] = Query(None, description="Optional room category filter"),
):
    """
    Computes daily occupancy rates and inventory availability across a date window.
    Strict date boundary prevents double counting on check-out dates.
    """
    try:
        if not start_date:
            start_date = date.today().strftime("%Y-%m-%d")
        if not end_date:
            end_date = (date.today()).strftime("%Y-%m-%d")
        timeline = room_inventory_manager.get_occupancy_timeline(
            start_date_str=start_date,
            end_date_str=end_date,
            room_category=room_category,
        )
        return {
            "status": "Success",
            "start_date": start_date,
            "end_date": end_date,
            "room_category": room_category or "All Categories",
            "timeline_days_count": len(timeline),
            "timeline": timeline,
        }
    except Exception as e:
        logger.error(f"Error retrieving occupancy timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reservations", summary="Active Reservations for a Target Date")
def get_active_reservations(
    target_date: str = Query(..., description="Stay date in YYYY-MM-DD format"),
    room_category: Optional[str] = Query(None, description="Optional room category filter"),
):
    """
    Retrieves individual confirmed reservations occupying rooms on the target date,
    including individual calibrated cancellation probabilities.
    """
    try:
        active = room_inventory_manager.get_active_reservations_for_date(
            target_date_str=target_date,
            room_category=room_category,
            include_calibrated_prob=True,
        )
        return {
            "status": "Success",
            "target_date": target_date,
            "room_category": room_category or "All Categories",
            "confirmed_count": len(active),
            "reservations": active,
        }
    except Exception as e:
        logger.error(f"Error retrieving active reservations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reservation", summary="Create Individual Confirmed Reservation Record")
def create_individual_reservation(request: IndividualReservationCreateRequest):
    """
    Logs an individual guest reservation with full booking covariates to enable
    reservation-level calibrated cancellation risk scoring.
    """
    try:
        record = room_inventory_manager.add_reservation(request.model_dump())
        return {"status": "Success", "message": "Reservation created successfully", "reservation": record}
    except Exception as e:
        logger.error(f"Error creating reservation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/missing-fields-audit", summary="PMS Schema Integration & Missing Fields Audit")
def get_missing_fields_audit():
    """
    Audits the current reservation schema against standard Property Management Systems (Opera, Mews)
    and reports missing production fields (loyalty tier, guarantee codes, room assignments).
    """
    try:
        return room_inventory_manager.get_missing_production_fields_report()
    except Exception as e:
        logger.error(f"Error retrieving missing fields audit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

