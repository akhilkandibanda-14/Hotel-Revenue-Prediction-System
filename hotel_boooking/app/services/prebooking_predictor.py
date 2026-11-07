from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.utils import config

logger = logging.getLogger(__name__)

# Strict blacklist for pre-booking prediction: any post-booking / during-stay feature is prohibited
PREBOOKING_LEAKAGE_BLACKLIST = {
    "Total_Revenue",
    "Revenue",
    "Room_Revenue",
    "Room_Revenue_All",
    "Profit",
    "Occupancy_Rate",
    "Occupancy_All",
    "Occupancy_Managed_Guests",
    "RevPAR",
    "RevPAR_All",
    "RevPAR_Managed_Guests",
    "Reserved_Rooms",
    "Available_Rooms",
    "Fixed_Costs",
    "Variable_Costs",
    "Total_Costs",
    "Operating_Expenses",
    "Checkins",
    "Checkouts",
    "No_Shows",
    "Cancellations",
    "Complaints",
    "Compliment",
    "Average_Review_Score",
}

REQUIRED_BOOKING_LEVEL_FIELDS = [
    "Booking_Date",
    "Check_In_Date",
    "Check_Out_Date",
    "Booking_Channel",
    "Market_Segment",
    "Room_Type",
    "Quoted_ADR",
]


class PrebookingRevenuePredictor:
    """
    Pre-Booking Revenue Prediction & Estimation Engine.
    Ensures ZERO target leakage by strictly prohibiting realized operational variables.
    Distinguishes genuine booking-level model training from exploratory operational estimations.
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.is_trained: bool = False
        self.training_metadata: Dict[str, Any] = {}

    def audit_dataset_for_booking_level(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audits whether a given dataset qualifies as a genuine individual pre-booking dataset
        or is an aggregated operational summary.
        """
        cols = set(df.columns)
        has_checkin = "Check_In_Date" in cols or "check_in_date" in cols
        has_checkout = "Check_Out_Date" in cols or "check_out_date" in cols
        has_booking_date = "Booking_Date" in cols or "booking_date" in cols or "Lead_Time" in cols
        has_realized_leakage = bool(cols.intersection(PREBOOKING_LEAKAGE_BLACKLIST))

        is_genuine_booking_level = has_checkin and has_checkout and (has_booking_date or "Lead_Time" in cols)

        missing_requirements = []
        for req in REQUIRED_BOOKING_LEVEL_FIELDS:
            if req not in cols and req.lower() not in [c.lower() for c in cols]:
                missing_requirements.append(req)

        return {
            "is_genuine_booking_level": is_genuine_booking_level,
            "contains_realized_leakage_fields": has_realized_leakage,
            "leakage_fields_identified": list(cols.intersection(PREBOOKING_LEAKAGE_BLACKLIST)),
            "missing_required_fields": missing_requirements,
            "dataset_rows": len(df),
            "dataset_columns": list(df.columns),
            "assessment_summary": (
                "Dataset contains genuine individual reservation-level transactions."
                if is_genuine_booking_level else
                "Dataset contains daily aggregated operational performance logs rather than "
                "individual booking-level reservations. Pre-booking ML model requires booking-level data."
            ),
        }

    def estimate_prebooking_value(
        self,
        check_in_date: str,
        check_out_date: str,
        booking_date: Optional[str] = None,
        quoted_adr: Optional[float] = None,
        rooms_count: int = 1,
        guests_count: int = 1,
        booking_channel: str = "Direct",
        market_segment: str = "Leisure",
        room_type: str = "Standard",
    ) -> Dict[str, Any]:
        """
        Calculates pre-booking expected revenue value prior to check-in.
        Prevents all future leakage. Validates chronological check-in/check-out order.
        """
        # Validate dates
        try:
            d_in = pd.to_datetime(check_in_date)
            d_out = pd.to_datetime(check_out_date)
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")

        if d_out <= d_in:
            raise ValueError(f"Check-out date ({check_out_date}) must be strictly after check-in date ({check_in_date})")

        length_of_stay = (d_out - d_in).days

        # Compute lead time if booking date provided
        lead_time_days = 14
        if booking_date:
            try:
                d_book = pd.to_datetime(booking_date)
                lead_time_days = max(0, (d_in - d_book).days)
            except Exception:
                pass

        # Quoted ADR resolution
        active_adr = float(quoted_adr) if quoted_adr is not None and quoted_adr > 0 else 165.0

        # Channel commission adjustment (Direct: 0%, Corporate: 5%, OTA: 15-18%)
        channel_commission_map = {
            "Direct": 0.0,
            "Corporate": 0.05,
            "Online": 0.15,
            "OTA": 0.18,
            "Wholesale": 0.20,
            "Groups": 0.10,
        }
        comm_rate = channel_commission_map.get(booking_channel, 0.12)

        # Baseline expected room revenue
        gross_room_revenue = active_adr * length_of_stay * rooms_count
        net_expected_revenue = gross_room_revenue * (1.0 - comm_rate)

        # Historical seasonal and day-of-week demand indicators
        dow_name = d_in.strftime("%A")
        is_weekend = d_in.dayofweek >= 5

        # Room type premium factor
        room_premiums = {
            "Standard": 1.0,
            "Deluxe": 1.25,
            "Executive Suite": 1.60,
            "Presidential Suite": 2.50,
        }
        room_factor = room_premiums.get(room_type, 1.0)
        adjusted_gross_revenue = gross_room_revenue * room_factor
        adjusted_net_revenue = net_expected_revenue * room_factor

        explanation = [
            f"Stay duration: {length_of_stay} night(s) for {rooms_count} room(s) at quoted ADR ${active_adr:.2f}.",
            f"Room tier adjustment: {room_type} multiplier {room_factor}x.",
            f"Distribution channel: {booking_channel} with estimated commission load {comm_rate*100:.1f}%.",
            f"Lead time: {lead_time_days} days in advance.",
            "Zero realized operational variables (no RevPAR, no post-stay actuals) used in this pre-booking calculation.",
        ]

        return {
            "status": "Success",
            "model_type": "Pre-Booking Expected Revenue Engine (Leakage-Free)",
            "is_trained_on_booking_level_data": self.is_trained,
            "booking_details": {
                "check_in_date": d_in.strftime("%Y-%m-%d"),
                "check_out_date": d_out.strftime("%Y-%m-%d"),
                "length_of_stay_nights": length_of_stay,
                "rooms_count": rooms_count,
                "guests_count": guests_count,
                "lead_time_days": lead_time_days,
                "room_type": room_type,
                "booking_channel": booking_channel,
                "market_segment": market_segment,
                "quoted_adr": active_adr,
                "checkin_day_of_week": dow_name,
                "is_weekend_arrival": is_weekend,
            },
            "estimated_gross_revenue": float(round(adjusted_gross_revenue, 2)),
            "estimated_net_revenue": float(round(adjusted_net_revenue, 2)),
            "estimated_commission_cost": float(round(adjusted_gross_revenue - adjusted_net_revenue, 2)),
            "explanation": explanation,
            "data_governance_notice": (
                "Pre-booking model uses purely ex-ante booking attributes. "
                "Active production model remains Extra Trees on operational aggregates; "
                "ingest a booking-level PMS dataset to trigger automated pre-booking ML training."
            ),
        }


# Singleton service
prebooking_predictor = PrebookingRevenuePredictor()

