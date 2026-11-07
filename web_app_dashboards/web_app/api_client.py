"""
FastAPI Reusable Client Module for Streamlit Frontend.
Handles network calls, timeouts, connection errors, and status deserialization.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT = float(os.getenv("FASTAPI_TIMEOUT", "10.0"))


class HotelRevenueAPIClient:
    """
    Robust API client connecting Streamlit to the FastAPI ML backend.
    """

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def check_health(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Verifies if the FastAPI backend is running and healthy."""
        try:
            resp = requests.get(self._url("/health"), timeout=self.timeout)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            err = f"Cannot connect to FastAPI at {self.base_url}. Ensure the service is running."
            return False, {}, err
        except requests.exceptions.Timeout:
            err = f"Connection to {self.base_url} timed out after {self.timeout}s."
            return False, {}, err
        except Exception as e:
            return False, {}, f"Health check failed: {str(e)}"

    def get_model_info(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves active production model metadata."""
        try:
            resp = requests.get(self._url("/model/info"), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return True, data.get("metadata", {}), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to backend service."
        except Exception as e:
            return False, {}, f"Failed to retrieve model info: {str(e)}"

    def get_model_comparison(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves multi-model benchmark evaluation results across all 5 algorithms."""
        try:
            resp = requests.get(self._url("/model/comparison"), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return True, data.get("comparison", {}), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to backend service."
        except Exception as e:
            return False, {}, f"Failed to retrieve model comparison: {str(e)}"

    def get_model_versions(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves list of available saved model versions."""
        try:
            resp = requests.get(self._url("/model/versions"), timeout=self.timeout)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to backend service."
        except Exception as e:
            return False, {}, f"Failed to retrieve model versions: {str(e)}"

    def rollback_model(self, version: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Rolls back the active production model to an earlier version."""
        try:
            resp = requests.post(self._url(f"/model/rollback/{version}"), timeout=self.timeout)
            if resp.status_code == 404:
                return False, {}, f"Model version '{version}' not found on backend."
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to backend service."
        except Exception as e:
            return False, {}, f"Rollback failed: {str(e)}"

    def predict(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Sends prediction request to POST /predict/.
        Handles full operational features or date-only queries.
        """
        try:
            resp = requests.post(self._url("/predict/"), json=payload, timeout=self.timeout)
            if resp.status_code == 422:
                detail = resp.json().get("detail", "Input validation failed.")
                return False, {}, f"Validation error: {detail}"
            if resp.status_code == 503:
                detail = resp.json().get("detail", "Model not trained yet.")
                return False, {}, f"Service unavailable: {detail}"

            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "Failed":
                return False, {}, data.get("message", "Prediction failed.")
            return True, data, None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to prediction backend."
        except requests.exceptions.Timeout:
            return False, {}, f"Prediction request timed out ({self.timeout}s)."
        except Exception as e:
            return False, {}, f"Prediction failed: {str(e)}"

    def upload_dataset(self, file_bytes: bytes, filename: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Uploads a CSV dataset to POST /dataset/upload for automated background training."""
        try:
            files = {"file": (filename, file_bytes, "text/csv")}
            resp = requests.post(self._url("/dataset/upload"), files=files, timeout=self.timeout)
            if resp.status_code == 400:
                detail = resp.json().get("detail", "Bad upload request.")
                return False, {}, f"Upload error: {detail}"
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to backend for dataset upload."
        except Exception as e:
            return False, {}, f"Dataset upload failed: {str(e)}"

    def get_dataset_status(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves background dataset training scheduler status and processed history."""
        try:
            resp = requests.get(self._url("/dataset/status"), timeout=self.timeout)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to backend service."
        except Exception as e:
            return False, {}, f"Failed to retrieve training status: {str(e)}"

    # --------------------------------------------------------------------------
    # Phase 3: Advanced Analytics & Intelligent Forecasting Methods
    # --------------------------------------------------------------------------
    def get_revenue_forecast(
        self,
        horizon: int = 30,
        start_date: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves chronological revenue forecast with 95% confidence intervals."""
        try:
            payload = {"horizon": horizon}
            if start_date:
                payload["start_date"] = start_date
            resp = requests.post(self._url("/analytics/forecast"), json=payload, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to analytics backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve revenue forecast: {str(e)}"

    def get_occupancy_forecast(
        self,
        horizon: int = 30,
        start_date: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves bounded occupancy forecast with weekday/weekend patterns."""
        try:
            payload = {"horizon": horizon}
            if start_date:
                payload["start_date"] = start_date
            resp = requests.post(self._url("/analytics/occupancy-forecast"), json=payload, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to analytics backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve occupancy forecast: {str(e)}"

    def get_pricing_optimization(
        self,
        base_adr: Optional[float] = None,
        base_occupancy: Optional[float] = None,
        total_rooms: Optional[int] = None,
        fixed_costs: Optional[float] = None,
        var_cost_per_room: Optional[float] = None,
        custom_elasticity: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves what-if pricing matrix and revenue/profit maximizing ADR scenarios."""
        try:
            payload: Dict[str, Any] = {}
            if base_adr is not None:
                payload["base_adr"] = base_adr
            if base_occupancy is not None:
                payload["base_occupancy"] = base_occupancy
            if total_rooms is not None:
                payload["total_rooms"] = total_rooms
            if fixed_costs is not None:
                payload["fixed_costs"] = fixed_costs
            if var_cost_per_room is not None:
                payload["var_cost_per_room"] = var_cost_per_room
            if custom_elasticity is not None:
                payload["custom_elasticity"] = custom_elasticity

            resp = requests.post(self._url("/analytics/pricing-optimizer"), json=payload, timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to analytics backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve pricing optimization: {str(e)}"

    def get_global_shap_importance(self, max_samples: int = 50) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves global SHAP feature importances for active model."""
        try:
            resp = requests.get(
                self._url(f"/analytics/explain/global?max_samples={max_samples}"),
                timeout=self.timeout + 15,
            )
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to analytics backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve global SHAP explanations: {str(e)}"

    def explain_instance(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves local SHAP decomposition for single reservation prediction."""
        try:
            resp = requests.post(self._url("/analytics/explain"), json=payload, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to analytics backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve instance SHAP explanation: {str(e)}"

    def get_anomalies(
        self,
        contamination: float = 0.05,
        z_threshold: float = 2.5,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves historical anomalies detected across revenue, occupancy, and ADR."""
        try:
            resp = requests.get(
                self._url(f"/analytics/anomalies?contamination={contamination}&z_threshold={z_threshold}"),
                timeout=self.timeout + 10,
            )
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to analytics backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve anomalies: {str(e)}"

    def assess_data_drift(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves data drift report comparing incoming distributions against baseline."""
        try:
            resp = requests.post(self._url("/analytics/drift"), json={}, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to analytics backend."
        except Exception as e:
            return False, {}, f"Failed to assess data drift: {str(e)}"

    # --------------------------------------------------------------------------
    # Phase 4: Revenue Intelligence, Demand Prediction & Business Decision Support
    # --------------------------------------------------------------------------
    def get_demand_forecast(
        self,
        horizon: int = 30,
        start_date: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves room demand and room nights forecast with baseline comparison."""
        try:
            payload = {"horizon": horizon}
            if start_date:
                payload["start_date"] = start_date
            resp = requests.post(self._url("/intelligence/demand-forecast"), json=payload, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve demand forecast: {str(e)}"

    def estimate_prebooking_revenue(
        self,
        check_in_date: str,
        check_out_date: str,
        booking_date: Optional[str] = None,
        quoted_adr: Optional[float] = None,
        rooms_count: int = 1,
        guests_count: int = 2,
        booking_channel: str = "Direct",
        market_segment: str = "Leisure",
        room_type: str = "Standard",
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Estimates pre-booking value using purely ex-ante booking attributes (no target leakage)."""
        try:
            payload = {
                "check_in_date": check_in_date,
                "check_out_date": check_out_date,
                "rooms_count": rooms_count,
                "guests_count": guests_count,
                "booking_channel": booking_channel,
                "market_segment": market_segment,
                "room_type": room_type,
            }
            if booking_date:
                payload["booking_date"] = booking_date
            if quoted_adr is not None:
                payload["quoted_adr"] = quoted_adr

            resp = requests.post(self._url("/intelligence/prebooking-predict"), json=payload, timeout=self.timeout + 5)
            if resp.status_code == 422:
                detail = resp.json().get("detail", "Validation failed.")
                return False, {}, f"Input validation error: {detail}"
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to estimate pre-booking revenue: {str(e)}"

    def get_prebooking_audit(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Audits training dataset for booking-level schema vs operational aggregates."""
        try:
            resp = requests.get(self._url("/intelligence/prebooking-audit"), timeout=self.timeout)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve prebooking audit: {str(e)}"

    def get_cancellation_analytics(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves cancellation rates, no-show rates, and slippage by channel and season."""
        try:
            resp = requests.get(self._url("/intelligence/cancellation-analytics"), timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve cancellation analytics: {str(e)}"

    def predict_cancellation(
        self,
        lead_time: int,
        arrival_date: str,
        adr: float,
        booking_channel: str = "Direct",
        customer_type: str = "Transient",
        deposit_type: str = "No Deposit",
        stays_in_weekend_nights: int = 1,
        stays_in_week_nights: int = 2,
        adults: int = 2,
        children: int = 0,
        babies: int = 0,
        is_repeated_guest: int = 0,
        previous_cancellations: int = 0,
        previous_bookings_not_canceled: int = 0,
        booking_changes: int = 0,
        days_in_waiting_list: int = 0,
        required_car_parking_spaces: int = 0,
        total_of_special_requests: int = 0,
        hotel: str = "City Hotel",
        market_segment: str = "Online TA",
        reserved_room_type: str = "A",
        meal: str = "BB",
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Predicts cancellation probability for a booking inquiry via ML model."""
        try:
            payload = {
                "lead_time": lead_time,
                "arrival_date": arrival_date,
                "adr": adr,
                "booking_channel": booking_channel,
                "customer_type": customer_type,
                "deposit_type": deposit_type,
                "stays_in_weekend_nights": stays_in_weekend_nights,
                "stays_in_week_nights": stays_in_week_nights,
                "adults": adults,
                "children": children,
                "babies": babies,
                "is_repeated_guest": is_repeated_guest,
                "previous_cancellations": previous_cancellations,
                "previous_bookings_not_canceled": previous_bookings_not_canceled,
                "booking_changes": booking_changes,
                "days_in_waiting_list": days_in_waiting_list,
                "required_car_parking_spaces": required_car_parking_spaces,
                "total_of_special_requests": total_of_special_requests,
                "hotel": hotel,
                "market_segment": market_segment,
                "reserved_room_type": reserved_room_type,
                "meal": meal,
            }
            resp = requests.post(self._url("/intelligence/cancellation-predict"), json=payload, timeout=self.timeout + 5)
            if resp.status_code == 422:
                return False, {}, resp.json().get("detail", "Validation error")
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to predict cancellation risk: {str(e)}"

    def get_cancellation_model_performance(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves multi-algorithm benchmark metrics on the public research holdout split."""
        try:
            resp = requests.get(self._url("/intelligence/cancellation-model-performance"), timeout=self.timeout + 15)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve cancellation benchmark: {str(e)}"

    def get_cancellation_calibration(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves reliability curves, Brier scores, and ECE for raw vs calibrated models."""
        try:
            resp = requests.get(self._url("/intelligence/cancellation-calibration"), timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve cancellation calibration: {str(e)}"

    def get_cancellation_threshold_analysis(
        self,
        cost_fp: float = 45.0,
        cost_fn: float = 160.0,
        replacement_rate: float = 0.25,
        split: str = "val",
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Evaluates cost-optimal decision threshold across configurable business cost parameters."""
        try:
            payload = {
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
                "replacement_rate": replacement_rate,
                "split": split,
            }
            resp = requests.post(self._url("/intelligence/cancellation-threshold-analysis"), json=payload, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve threshold analysis: {str(e)}"

    def explain_cancellation(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Computes local SHAP attributions and additive consistency for reservation cancellation risk."""
        try:
            resp = requests.post(self._url("/intelligence/cancellation-explain"), json=payload, timeout=self.timeout + 10)
            if resp.status_code == 422:
                return False, {}, resp.json().get("detail", "Validation error")
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve cancellation SHAP explanation: {str(e)}"

    def simulate_cancellation_what_if(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Simulates cancellation probability across operational levers (deposit, lead time, channel, ADR, requests)."""
        try:
            resp = requests.post(self._url("/intelligence/cancellation-what-if"), json=payload, timeout=self.timeout + 10)
            if resp.status_code == 422:
                return False, {}, resp.json().get("detail", "Validation error")
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to simulate cancellation what-if: {str(e)}"

    def get_cancellation_segment_analysis(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves empirical cancellation distributions across lead time tiers, channels, segments, and months."""
        try:
            resp = requests.get(self._url("/intelligence/cancellation-segment-analysis"), timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve cancellation segment analysis: {str(e)}"


    def evaluate_competitor_pricing(
        self,
        our_adr: Optional[float] = None,
        competitor_adr: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Evaluates Market Price Position Index (MPI) against competitor rates."""
        try:
            payload = {}
            if our_adr is not None:
                payload["our_adr"] = our_adr
            if competitor_adr is not None:
                payload["competitor_adr"] = competitor_adr

            resp = requests.post(self._url("/intelligence/competitor-analysis"), json=payload, timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to evaluate competitor pricing: {str(e)}"

    def add_competitor_record(
        self,
        date_str: str,
        competitor_name: str,
        competitor_adr: float,
        our_adr: float,
        room_type: str = "Standard Deluxe",
        notes: str = "",
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Saves a manually entered competitor benchmark observation."""
        try:
            payload = {
                "date_str": date_str,
                "competitor_name": competitor_name,
                "competitor_adr": competitor_adr,
                "our_adr": our_adr,
                "room_type": room_type,
                "notes": notes,
            }
            resp = requests.post(self._url("/intelligence/competitor-record"), json=payload, timeout=self.timeout + 5)
            if resp.status_code == 422:
                return False, {}, resp.json().get("detail", "Validation error")
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to log competitor record: {str(e)}"

    def get_competitor_records(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves all stored competitor rate benchmark observations."""
        try:
            resp = requests.get(self._url("/intelligence/competitor-records"), timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve competitor records: {str(e)}"

    def get_business_recommendations(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves prioritized, actionable business decision recommendations."""
        try:
            resp = requests.get(self._url("/intelligence/business-recommendations"), timeout=self.timeout + 15)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to intelligence backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve business recommendations: {str(e)}"

    # --------------------------------------------------------------------------
    # Phase 5: Room Inventory Management & Intelligent Overbooking Optimization
    # --------------------------------------------------------------------------
    def get_inventory_availability(
        self,
        target_date: Optional[str] = None,
        room_category: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves room inventory, out-of-service rooms, confirmed bookings, and available rooms."""
        try:
            params = {}
            if target_date:
                params["target_date"] = target_date
            if room_category:
                params["room_category"] = room_category
            resp = requests.get(self._url("/inventory/availability"), params=params, timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to inventory backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve inventory availability: {str(e)}"

    def update_room_inventory(
        self,
        room_category: str,
        total_physical_rooms: int,
        rooms_out_of_service: int,
        base_adr: Optional[float] = None,
        walk_relocation_cost: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Updates room category physical capacity and out-of-service maintenance counts."""
        try:
            payload = {
                "room_category": room_category,
                "total_physical_rooms": total_physical_rooms,
                "rooms_out_of_service": rooms_out_of_service,
            }
            if base_adr is not None:
                payload["base_adr"] = base_adr
            if walk_relocation_cost is not None:
                payload["walk_relocation_cost"] = walk_relocation_cost
            resp = requests.post(self._url("/inventory/rooms"), json=payload, timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to inventory backend."
        except Exception as e:
            return False, {}, f"Failed to update room inventory: {str(e)}"

    def get_occupancy_timeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        room_category: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves day-by-day occupancy rates and inventory availability across a date window."""
        try:
            params = {}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            if room_category:
                params["room_category"] = room_category
            resp = requests.get(self._url("/inventory/occupancy"), params=params, timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to inventory backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve occupancy timeline: {str(e)}"

    def get_active_reservations(
        self,
        target_date: str,
        room_category: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves confirmed reservations for a date with calibrated cancellation risks."""
        try:
            params = {"target_date": target_date}
            if room_category:
                params["room_category"] = room_category
            resp = requests.get(self._url("/inventory/reservations"), params=params, timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to inventory backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve active reservations: {str(e)}"

    def get_missing_fields_audit(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves PMS schema integration audit and missing production fields."""
        try:
            resp = requests.get(self._url("/inventory/missing-fields-audit"), timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to inventory backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve missing fields audit: {str(e)}"

    def get_cancellation_forecast(
        self,
        target_date: str,
        room_category: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Computes expected cancellations and Poisson-Binomial arrival probability distribution."""
        try:
            params = {"target_date": target_date}
            if room_category:
                params["room_category"] = room_category
            resp = requests.get(self._url("/overbooking/forecast"), params=params, timeout=self.timeout + 5)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to overbooking backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve cancellation forecast: {str(e)}"

    def optimize_overbooking(
        self,
        target_date: str,
        room_category: str = "Standard Room",
        custom_adr: Optional[float] = None,
        custom_walk_cost: Optional[float] = None,
        max_walk_risk_pct: float = 2.0,
        max_additional_bookings: int = 30,
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Finds the revenue-maximizing booking limit subject to service-level walk risk constraint."""
        try:
            payload = {
                "target_date": target_date,
                "room_category": room_category,
                "max_walk_risk_pct": max_walk_risk_pct,
                "max_additional_bookings": max_additional_bookings,
            }
            if custom_adr is not None:
                payload["custom_adr"] = custom_adr
            if custom_walk_cost is not None:
                payload["custom_walk_cost"] = custom_walk_cost
            resp = requests.post(self._url("/overbooking/optimize"), json=payload, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to overbooking backend."
        except Exception as e:
            return False, {}, f"Failed to optimize overbooking: {str(e)}"

    def simulate_overbooking_scenario(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Simulates overbooking trade-offs under custom hotel management assumptions."""
        try:
            resp = requests.post(self._url("/overbooking/scenario"), json=payload, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to overbooking backend."
        except Exception as e:
            return False, {}, f"Failed to simulate overbooking scenario: {str(e)}"

    def get_portfolio_overbooking_recommendations(self, target_date: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves synchronized overbooking recommendations across all room categories."""
        try:
            params = {"target_date": target_date}
            resp = requests.get(self._url("/overbooking/recommendations"), params=params, timeout=self.timeout + 10)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to overbooking backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve portfolio recommendations: {str(e)}"

    def get_overbooking_performance(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Retrieves multi-day overbooking vs zero-overbooking benchmark comparison."""
        try:
            resp = requests.get(self._url("/overbooking/performance"), timeout=self.timeout + 15)
            resp.raise_for_status()
            return True, resp.json(), None
        except requests.exceptions.ConnectionError:
            return False, {}, "Cannot connect to overbooking backend."
        except Exception as e:
            return False, {}, f"Failed to retrieve overbooking performance: {str(e)}"


