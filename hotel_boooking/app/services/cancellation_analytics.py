from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.utils import config

logger = logging.getLogger(__name__)


class CancellationAnalyticsService:
    """
    Cancellation and Booking Behavior Analytics Service.
    Computes cancellation rates, no-show rates, and slippage by channel and season.
    Documents exact PMS booking-level data requirements for individual risk modeling.
    """

    def __init__(self):
        pass

    def load_cancellation_data(self) -> Optional[pd.DataFrame]:
        """
        Attempts to load cancellation-rich dataset (Hotel book.csv) from app datasets or web app directory.
        """
        candidates = [
            config.BASE_DIR.parent / "web_app_dashboards" / "web_app" / "Hotel book.csv",
            config.BASE_DIR / "Hotel book.csv",
            config.UPLOADED_FOLDER / "20260630092558388060_Hotel book.csv",
        ]
        for p in candidates:
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if "Cancellations" in df.columns and "Bookings" in df.columns:
                        return df
                except Exception:
                    pass
        return None

    def analyze_cancellations(self) -> Dict[str, Any]:
        """
        Computes aggregate cancellation and no-show statistics across channels and seasons.
        """
        df = self.load_cancellation_data()

        if df is None or len(df) == 0:
            return {
                "status": "Unavailable",
                "message": "No historical dataset with cancellation records (Cancellations, Bookings) found.",
                "dataset_requirements": [
                    "Bookings (count of daily reservations created)",
                    "Cancellations (count of daily cancelled bookings)",
                    "No_Shows (count of guests failing to arrive)",
                    "Checkins and Checkouts",
                ]
            }

        total_bookings = int(df["Bookings"].sum())
        total_cancels = int(df["Cancellations"].sum())
        total_no_shows = int(df["No_Shows"].sum()) if "No_Shows" in df.columns else 0

        overall_cancel_rate = float(round((total_cancels / max(1, total_bookings)) * 100.0, 2))
        overall_noshow_rate = float(round((total_no_shows / max(1, total_bookings)) * 100.0, 2))
        total_slippage_rate = float(round(overall_cancel_rate + overall_noshow_rate, 2))

        # Channel breakdown
        channel_metrics: List[Dict[str, Any]] = []
        if "Booking_Channel" in df.columns:
            for ch, group in df.groupby("Booking_Channel"):
                b = int(group["Bookings"].sum())
                c = int(group["Cancellations"].sum())
                ns = int(group["No_Shows"].sum()) if "No_Shows" in group.columns else 0
                c_rate = float(round((c / max(1, b)) * 100.0, 2))
                ns_rate = float(round((ns / max(1, b)) * 100.0, 2))
                channel_metrics.append({
                    "booking_channel": str(ch),
                    "total_bookings": b,
                    "cancellations": c,
                    "no_shows": ns,
                    "cancellation_rate_pct": c_rate,
                    "no_show_rate_pct": ns_rate,
                    "combined_slippage_pct": float(round(c_rate + ns_rate, 2)),
                })

        # Seasonal breakdown
        season_metrics: List[Dict[str, Any]] = []
        if "Season" in df.columns:
            for s, group in df.groupby("Season"):
                b = int(group["Bookings"].sum())
                c = int(group["Cancellations"].sum())
                c_rate = float(round((c / max(1, b)) * 100.0, 2))
                season_metrics.append({
                    "season": str(s),
                    "total_bookings": b,
                    "cancellations": c,
                    "cancellation_rate_pct": c_rate,
                })

        # Data requirements for guest-level cancellation machine learning
        pms_requirements = [
            "lead_time_days (days between booking creation and check-in)",
            "deposit_type (Non-Refundable, Refundable, No Deposit)",
            "customer_type (Transient, Contract, Group)",
            "previous_cancellations (historical count of cancellations by booker)",
            "days_in_waiting_list",
            "required_car_parking_spaces",
            "total_of_special_requests",
        ]

        return {
            "status": "Success",
            "total_records_analyzed": len(df),
            "summary": {
                "total_bookings": total_bookings,
                "total_cancellations": total_cancels,
                "total_no_shows": total_no_shows,
                "overall_cancellation_rate_pct": overall_cancel_rate,
                "overall_no_show_rate_pct": overall_noshow_rate,
                "combined_revenue_slippage_pct": total_slippage_rate,
            },
            "by_channel": channel_metrics,
            "by_season": season_metrics,
            "guest_level_ml_status": "Descriptive Analytics Active (Awaiting PMS Transactional Schema)",
            "missing_pms_fields_for_individual_ml": pms_requirements,
        }


# Singleton service
cancellation_service = CancellationAnalyticsService()

