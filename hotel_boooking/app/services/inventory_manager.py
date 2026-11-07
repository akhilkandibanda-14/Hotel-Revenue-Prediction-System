from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.services.cancellation_model import cancellation_model_service
from app.utils import config

logger = logging.getLogger(__name__)

# Dedicated directory for inventory and reservation configurations
INVENTORY_DIR = config.BASE_DIR / "reservations"
INVENTORY_DIR.mkdir(parents=True, exist_ok=True)

INVENTORY_CONFIG_FILE = INVENTORY_DIR / "room_inventory_config.json"
INDIVIDUAL_RESERVATIONS_FILE = INVENTORY_DIR / "individual_reservations.json"

# Room Categories Mapping to Public Antonio et al. room codes for cancellation scoring
ROOM_CATEGORY_CODE_MAP = {
    "Standard Room": "A",
    "Deluxe Room": "D",
    "Executive Suite": "E",
    "Presidential Suite": "G",
}

# Standard Default Demonstration Inventory (Clearly labeled as DEMO_HOTEL_INVENTORY)
DEFAULT_ROOM_CATEGORIES = {
    "Standard Room": {
        "room_category": "Standard Room",
        "room_code": "A",
        "total_physical_rooms": 100,
        "rooms_out_of_service": 2,
        "base_adr": 140.0,
        "walk_relocation_cost": 280.0,
        "description": "Standard Queen/King guest room with city/garden view",
    },
    "Deluxe Room": {
        "room_category": "Deluxe Room",
        "room_code": "D",
        "total_physical_rooms": 60,
        "rooms_out_of_service": 1,
        "base_adr": 195.0,
        "walk_relocation_cost": 390.0,
        "description": "Spacious premium room with balcony and luxury bath",
    },
    "Executive Suite": {
        "room_category": "Executive Suite",
        "room_code": "E",
        "total_physical_rooms": 30,
        "rooms_out_of_service": 1,
        "base_adr": 290.0,
        "walk_relocation_cost": 580.0,
        "description": "Executive suite with dedicated lounge access and living area",
    },
    "Presidential Suite": {
        "room_category": "Presidential Suite",
        "room_code": "G",
        "total_physical_rooms": 10,
        "rooms_out_of_service": 0,
        "base_adr": 650.0,
        "walk_relocation_cost": 1300.0,
        "description": "Top-floor multi-room penthouse with panoramic views and butler service",
    },
}


class RoomInventoryManager:
    """
    Intelligent Room Inventory Management Service (Phase 5.1 & 5.2).
    Manages physical room counts, out-of-service maintenance, individual reservations,
    and calculates non-overlapping daily occupancy and calibrated cancellation probabilities.
    """

    def __init__(self):
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.reservations: List[Dict[str, Any]] = []
        self._load_inventory()

    def _load_inventory(self):
        """Loads room inventory config and individual reservations from disk or initializes defaults."""
        if INVENTORY_CONFIG_FILE.exists():
            try:
                self.categories = json.loads(INVENTORY_CONFIG_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load inventory config: {e}. Using defaults.")
                self.categories = dict(DEFAULT_ROOM_CATEGORIES)
        else:
            self.categories = dict(DEFAULT_ROOM_CATEGORIES)
            self._save_config()

        if INDIVIDUAL_RESERVATIONS_FILE.exists():
            try:
                self.reservations = json.loads(INDIVIDUAL_RESERVATIONS_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load individual reservations: {e}. Initializing seed bookings.")
                self._seed_demonstration_reservations()
        else:
            self._seed_demonstration_reservations()

        # Precompute batch calibrated probabilities for fast O(1) retrieval
        self._batch_score_all_reservations()

    def _batch_score_all_reservations(self):
        """Precomputes calibrated cancellation probability for all un-scored reservations in one fast batch."""
        unscored = [r for r in self.reservations if "calibrated_cancellation_probability" not in r]
        if not unscored or not cancellation_model_service.is_model_ready():
            return

        logger.info(f"Batch scoring {len(unscored)} reservations for cancellation risk...")
        records = []
        for r in unscored:
            try:
                arr_dt = datetime.strptime(r["arrival_date"], "%Y-%m-%d")
                month_num = arr_dt.month
            except Exception:
                month_num = 7
            wknd = 1
            week = max(1, int(r.get("stay_nights", 2)) - 1)
            records.append({
                "lead_time": max(0, int(r.get("lead_time", 30))),
                "stays_in_weekend_nights": wknd,
                "stays_in_week_nights": week,
                "total_stay_nights": wknd + week,
                "adults": max(1, int(r.get("adults", 2))),
                "children": float(r.get("children", 0)),
                "babies": 0,
                "total_guests": max(1.0, float(r.get("adults", 2)) + float(r.get("children", 0))),
                "is_repeated_guest": 0,
                "previous_cancellations": max(0, int(r.get("previous_cancellations", 0))),
                "previous_bookings_not_canceled": 0,
                "booking_changes": max(0, int(r.get("booking_changes", 0))),
                "days_in_waiting_list": 0,
                "adr": max(0.0, float(r.get("booked_adr", 150.0))),
                "required_car_parking_spaces": max(0, int(r.get("required_car_parking_spaces", 0))),
                "total_of_special_requests": max(0, int(r.get("total_of_special_requests", 0))),
                "has_prior_cancellations": 1 if int(r.get("previous_cancellations", 0)) > 0 else 0,
                "arrival_month_num": month_num,
                "hotel": "City Hotel",
                "market_segment": "Online TA" if r.get("booking_channel") == "Online TA" else "Direct",
                "distribution_channel": str(r.get("booking_channel", "Direct")),
                "deposit_type": str(r.get("deposit_type", "No Deposit")),
                "customer_type": str(r.get("customer_type", "Transient")),
                "reserved_room_type": ROOM_CATEGORY_CODE_MAP.get(r.get("room_category", "Standard Room"), "A"),
                "meal": "BB",
            })

        df_batch = pd.DataFrame(records)
        X_trans = cancellation_model_service.preprocessor.transform(df_batch)
        if cancellation_model_service.calibrator_sigmoid:
            probs = cancellation_model_service.calibrator_sigmoid.predict_proba(X_trans)[:, 1]
        else:
            probs = cancellation_model_service.model.predict_proba(X_trans)[:, 1]

        for i, r in enumerate(unscored):
            p = round(float(probs[i]), 4)
            r["calibrated_cancellation_probability"] = p
            r["calibrated_cancellation_pct"] = round(p * 100.0, 1)

        self._save_reservations()
        logger.info(f"Batch scoring completed for {len(unscored)} reservations.")


    def _save_config(self):
        INVENTORY_CONFIG_FILE.write_text(json.dumps(self.categories, indent=2), encoding="utf-8")

    def _save_reservations(self):
        INDIVIDUAL_RESERVATIONS_FILE.write_text(json.dumps(self.reservations, indent=2), encoding="utf-8")

    def _seed_demonstration_reservations(self):
        """
        Seeds structured individual reservation records spanning upcoming high-demand dates.
        Clearly tagged as DEMO_HOTEL_INVENTORY.
        """
        logger.info("Initializing baseline demonstration reservations for Phase 5...")
        today = date.today()
        # Seed reservations for upcoming 14-day window centered on tomorrow + 7 days
        anchor_date = today + timedelta(days=7)

        sample_channels = ["Online TA", "Direct", "Corporate", "Offline TA/TO", "GDS"]
        sample_deposits = ["No Deposit", "Non Refund", "Refundable"]
        sample_customers = ["Transient", "Transient-Party", "Contract", "Group"]

        seed_list = []
        res_counter = 1001

        # Seed bookings for target date range: anchor_date - 2 days to anchor_date + 7 days
        for day_offset in range(-2, 10):
            current_day = anchor_date + timedelta(days=day_offset)
            current_day_str = current_day.strftime("%Y-%m-%d")

            # Determine booking load based on day of week (higher on weekends)
            is_weekend = current_day.weekday() >= 4

            # Standard Room bookings (Capacity 98)
            std_count = 82 if is_weekend else 68
            # Deluxe Room bookings (Capacity 59)
            dlx_count = 52 if is_weekend else 41
            # Executive Suite bookings (Capacity 29)
            ste_count = 25 if is_weekend else 19
            # Presidential Suite bookings (Capacity 10)
            prs_count = 8 if is_weekend else 5

            category_plan = [
                ("Standard Room", std_count, 140.0),
                ("Deluxe Room", dlx_count, 195.0),
                ("Executive Suite", ste_count, 290.0),
                ("Presidential Suite", prs_count, 650.0),
            ]

            for cat_name, count, adr_val in category_plan:
                for idx in range(count):
                    # Stagger stay lengths: 1 to 4 nights
                    stay_nights = 2 if idx % 3 == 0 else (1 if idx % 2 == 0 else 3)
                    dep_day = current_day + timedelta(days=stay_nights)
                    lead_time = 5 + (idx * 4) % 180
                    ch = sample_channels[idx % len(sample_channels)]
                    dep = "Non Refund" if (idx % 11 == 0 and ch == "Online TA") else ("Refundable" if idx % 7 == 0 else "No Deposit")
                    cust = sample_customers[idx % len(sample_customers)]
                    spec_req = 2 if idx % 4 == 0 else (1 if idx % 2 == 0 else 0)
                    parking = 1 if (idx % 5 == 0 and cat_name != "Standard Room") else 0
                    prev_canc = 1 if (idx % 13 == 0) else 0

                    seed_list.append({
                        "booking_id": f"RES-{current_day.strftime('%m%d')}-{res_counter:04d}",
                        "guest_name": f"Guest {res_counter}",
                        "room_category": cat_name,
                        "arrival_date": current_day_str,
                        "departure_date": dep_day.strftime("%Y-%m-%d"),
                        "stay_nights": stay_nights,
                        "booked_adr": adr_val + ((idx % 5) * 10 - 20),
                        "booking_channel": ch,
                        "deposit_type": dep,
                        "customer_type": cust,
                        "lead_time": lead_time,
                        "adults": 2 if idx % 3 == 0 else 1,
                        "children": 1 if idx % 6 == 0 else 0,
                        "babies": 0,
                        "previous_cancellations": prev_canc,
                        "booking_changes": 1 if idx % 8 == 0 else 0,
                        "total_of_special_requests": spec_req,
                        "required_car_parking_spaces": parking,
                        "booking_status": "Confirmed",
                        "data_governance_label": "DEMO_HOTEL_INVENTORY",
                    })
                    res_counter += 1

        self.reservations = seed_list
        self._save_reservations()
        logger.info(f"Initialized {len(self.reservations)} individual demonstration reservations.")

    def get_room_categories(self) -> Dict[str, Dict[str, Any]]:
        """Returns physical room inventory configuration by room category."""
        result = {}
        for cat, data in self.categories.items():
            phys = data.get("total_physical_rooms", 0)
            oos = data.get("rooms_out_of_service", 0)
            result[cat] = {
                **data,
                "effective_capacity": max(0, phys - oos),
            }
        return result

    def update_room_category(
        self,
        room_category: str,
        total_physical_rooms: int,
        rooms_out_of_service: int,
        base_adr: Optional[float] = None,
        walk_relocation_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Creates or updates inventory capacity and operational parameters for a room category."""
        if room_category not in self.categories:
            code = ROOM_CATEGORY_CODE_MAP.get(room_category, "A")
            self.categories[room_category] = {
                "room_category": room_category,
                "room_code": code,
                "total_physical_rooms": int(total_physical_rooms),
                "rooms_out_of_service": int(rooms_out_of_service),
                "base_adr": float(base_adr) if base_adr else 150.0,
                "walk_relocation_cost": float(walk_relocation_cost) if walk_relocation_cost else 300.0,
                "description": f"{room_category} inventory category",
            }
        else:
            self.categories[room_category]["total_physical_rooms"] = int(total_physical_rooms)
            self.categories[room_category]["rooms_out_of_service"] = int(rooms_out_of_service)
            if base_adr is not None:
                self.categories[room_category]["base_adr"] = float(base_adr)
            if walk_relocation_cost is not None:
                self.categories[room_category]["walk_relocation_cost"] = float(walk_relocation_cost)

        self._save_config()
        return self.categories[room_category]

    def get_active_reservations_for_date(
        self,
        target_date_str: str,
        room_category: Optional[str] = None,
        include_calibrated_prob: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all confirmed reservations occupying a room on target_date_str.
        Strict date boundary: arrival_date <= target_date < departure_date (no double counting on checkout).
        Optionally computes calibrated cancellation probability for each reservation.
        """
        try:
            target_dt = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError:
            target_dt = date.today()

        matched = []
        for r in self.reservations:
            if r.get("booking_status", "Confirmed") != "Confirmed":
                continue
            if room_category and r.get("room_category") != room_category:
                continue

            try:
                arr_dt = datetime.strptime(r["arrival_date"], "%Y-%m-%d").date()
                dep_dt = datetime.strptime(r["departure_date"], "%Y-%m-%d").date()
            except Exception:
                continue

            # Strict night occupancy: guest occupies room if arrival <= target < departure
            if arr_dt <= target_dt < dep_dt:
                res_item = dict(r)
                if include_calibrated_prob:
                    # Calculate or retrieve calibrated cancellation probability
                    prob = self._score_reservation_cancellation(res_item)
                    res_item["calibrated_cancellation_probability"] = prob
                    res_item["calibrated_cancellation_pct"] = round(prob * 100.0, 1)
                matched.append(res_item)

        return matched

    def _score_reservation_cancellation(self, res: Dict[str, Any]) -> float:
        """Calls the calibrated cancellation model to compute individual cancellation risk with caching."""
        if "calibrated_cancellation_probability" in res:
            return float(res["calibrated_cancellation_probability"])

        if not cancellation_model_service.is_model_ready():
            return 0.25


        try:
            pred = cancellation_model_service.predict_cancellation_risk(
                lead_time=int(res.get("lead_time", 30)),
                arrival_date=str(res.get("arrival_date", "2026-08-15")),
                adr=float(res.get("booked_adr", 150.0)),
                booking_channel=str(res.get("booking_channel", "Online TA")),
                customer_type=str(res.get("customer_type", "Transient")),
                deposit_type=str(res.get("deposit_type", "No Deposit")),
                stays_in_weekend_nights=1,
                stays_in_week_nights=max(1, int(res.get("stay_nights", 2)) - 1),
                adults=int(res.get("adults", 2)),
                children=int(res.get("children", 0)),
                babies=int(res.get("babies", 0)),
                is_repeated_guest=0,
                previous_cancellations=int(res.get("previous_cancellations", 0)),
                previous_bookings_not_canceled=0,
                booking_changes=int(res.get("booking_changes", 0)),
                days_in_waiting_list=0,
                required_car_parking_spaces=int(res.get("required_car_parking_spaces", 0)),
                total_of_special_requests=int(res.get("total_of_special_requests", 0)),
                hotel="City Hotel",
                market_segment="Online TA" if res.get("booking_channel") == "Online TA" else "Direct",
                reserved_room_type=ROOM_CATEGORY_CODE_MAP.get(res.get("room_category", "Standard Room"), "A"),
                meal="BB",
            )
            return float(pred.get("calibrated_probability", pred.get("cancellation_probability", 0.25)))
        except Exception as e:
            logger.warning(f"Failed to score reservation {res.get('booking_id')}: {e}")
            return 0.25

    def get_availability_by_date(
        self,
        target_date_str: str,
        room_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes total physical rooms, out-of-service, confirmed bookings,
        and available inventory for a specific date across categories.
        """
        active_bookings = self.get_active_reservations_for_date(target_date_str, room_category=room_category, include_calibrated_prob=False)

        cat_configs = self.get_room_categories()
        breakdown = []

        selected_cats = [room_category] if room_category and room_category in cat_configs else list(cat_configs.keys())

        tot_phys = 0
        tot_oos = 0
        tot_eff = 0
        tot_booked = 0

        for cat in selected_cats:
            cfg = cat_configs[cat]
            phys = cfg.get("total_physical_rooms", 0)
            oos = cfg.get("rooms_out_of_service", 0)
            eff = max(0, phys - oos)
            booked = sum(1 for b in active_bookings if b.get("room_category") == cat)
            avail = max(0, eff - booked)
            occ_pct = round((booked / eff) * 100.0, 1) if eff > 0 else 0.0

            tot_phys += phys
            tot_oos += oos
            tot_eff += eff
            tot_booked += booked

            breakdown.append({
                "room_category": cat,
                "room_code": cfg.get("room_code", "A"),
                "total_physical_rooms": phys,
                "rooms_out_of_service": oos,
                "effective_capacity": eff,
                "confirmed_reservations": booked,
                "available_rooms": avail,
                "occupancy_rate_pct": occ_pct,
                "base_adr": cfg.get("base_adr", 150.0),
                "walk_relocation_cost": cfg.get("walk_relocation_cost", 300.0),
            })

        overall_avail = max(0, tot_eff - tot_booked)
        overall_occ_pct = round((tot_booked / tot_eff) * 100.0, 1) if tot_eff > 0 else 0.0

        return {
            "status": "Success",
            "target_date": target_date_str,
            "data_governance_label": "DEMO_HOTEL_INVENTORY",
            "summary": {
                "total_physical_rooms": tot_phys,
                "rooms_out_of_service": tot_oos,
                "total_effective_capacity": tot_eff,
                "total_confirmed_reservations": tot_booked,
                "total_available_rooms": overall_avail,
                "overall_occupancy_rate_pct": overall_occ_pct,
            },
            "category_breakdown": breakdown,
        }

    def get_occupancy_timeline(
        self,
        start_date_str: str,
        end_date_str: str,
        room_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Computes day-by-day occupancy rates and inventory availability across a date range.
        Strict date boundary prevents double counting on check-out dates.
        """
        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except ValueError:
            start_dt = date.today()
            end_dt = start_dt + timedelta(days=14)

        if end_dt < start_dt:
            end_dt = start_dt + timedelta(days=14)

        cat_configs = self.get_room_categories()
        if room_category and room_category in cat_configs:
            cat_eff = cat_configs[room_category]["effective_capacity"]
        else:
            cat_eff = sum(c["effective_capacity"] for c in cat_configs.values())

        timeline = []
        curr = start_dt
        while curr <= end_dt:
            curr_str = curr.strftime("%Y-%m-%d")
            active = self.get_active_reservations_for_date(curr_str, room_category=room_category, include_calibrated_prob=False)
            booked = len(active)
            avail = max(0, cat_eff - booked)
            occ = round((booked / cat_eff) * 100.0, 1) if cat_eff > 0 else 0.0

            timeline.append({
                "date": curr_str,
                "day_of_week": curr.strftime("%A"),
                "is_weekend": curr.weekday() >= 4,
                "effective_capacity": cat_eff,
                "confirmed_reservations": booked,
                "available_rooms": avail,
                "occupancy_rate_pct": occ,
            })
            curr += timedelta(days=1)

        return timeline

    def add_reservation(self, res_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates and logs a new individual confirmed reservation."""
        booking_id = res_data.get("booking_id") or f"RES-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        record = {
            "booking_id": booking_id,
            "guest_name": str(res_data.get("guest_name", "Walk-in / Direct Booker")),
            "room_category": str(res_data.get("room_category", "Standard Room")),
            "arrival_date": str(res_data.get("arrival_date", date.today().strftime("%Y-%m-%d"))),
            "departure_date": str(res_data.get("departure_date", (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"))),
            "stay_nights": int(res_data.get("stay_nights", 1)),
            "booked_adr": float(res_data.get("booked_adr", 150.0)),
            "booking_channel": str(res_data.get("booking_channel", "Direct")),
            "deposit_type": str(res_data.get("deposit_type", "No Deposit")),
            "customer_type": str(res_data.get("customer_type", "Transient")),
            "lead_time": int(res_data.get("lead_time", 14)),
            "adults": int(res_data.get("adults", 2)),
            "children": int(res_data.get("children", 0)),
            "babies": int(res_data.get("babies", 0)),
            "previous_cancellations": int(res_data.get("previous_cancellations", 0)),
            "booking_changes": int(res_data.get("booking_changes", 0)),
            "total_of_special_requests": int(res_data.get("total_of_special_requests", 0)),
            "required_car_parking_spaces": int(res_data.get("required_car_parking_spaces", 0)),
            "booking_status": "Confirmed",
            "data_governance_label": "USER_INPUT_RESERVATION",
            "created_at": datetime.now().isoformat(),
        }
        self.reservations.append(record)
        self._save_reservations()
        return record

    def cancel_reservation(self, booking_id: str) -> bool:
        """Updates booking status to Canceled."""
        for r in self.reservations:
            if r.get("booking_id") == booking_id:
                r["booking_status"] = "Canceled"
                r["canceled_at"] = datetime.now().isoformat()
                self._save_reservations()
                return True
        return False

    def get_missing_production_fields_report(self) -> Dict[str, Any]:
        """
        Audits current schema vs standard PMS schemas (Opera PMS, Mews, Protel).
        Reports missing fields and provides documented integration interface.
        """
        return {
            "status": "Success",
            "data_governance_label": "DEMO_HOTEL_INVENTORY",
            "active_supported_fields": [
                "booking_id", "guest_name", "room_category", "arrival_date",
                "departure_date", "stay_nights", "booked_adr", "booking_channel",
                "deposit_type", "customer_type", "lead_time", "adults", "children",
                "previous_cancellations", "booking_changes", "total_of_special_requests",
                "required_car_parking_spaces", "booking_status",
            ],
            "missing_production_pms_fields": [
                {
                    "field_name": "guest_loyalty_tier",
                    "pms_source": "Guest Profile (Opera / Mews)",
                    "importance": "High",
                    "reason": "Elite loyalty members have near-zero walk tolerance and require walk exclusion flags.",
                },
                {
                    "field_name": "guarantee_type_code",
                    "pms_source": "Reservation Folio",
                    "importance": "High",
                    "reason": "Credit card pre-authorization vs company billing voucher.",
                },
                {
                    "field_name": "assigned_physical_room_number",
                    "pms_source": "Housekeeping / Front Desk",
                    "importance": "Medium",
                    "reason": "Allows real-time lock on specific rooms assigned for early arrival.",
                },
                {
                    "field_name": "corporate_contract_id",
                    "pms_source": "Rate Code Setup",
                    "importance": "Medium",
                    "reason": "Corporate negotiated rate accounts often carry negotiated relocation penalties.",
                },
                {
                    "field_name": "estimated_time_of_arrival_eta",
                    "pms_source": "Pre-Stay Registration",
                    "importance": "Medium",
                    "reason": "Enables intraday walk sequencing (late arrivals walk first).",
                },
            ],
            "integration_interface": {
                "api_endpoint": "POST /inventory/reservation",
                "payload_format": "JSON",
                "ingestion_mode": "Webhook or nightly batch export",
            },
        }


# Global singleton instance
room_inventory_manager = RoomInventoryManager()
