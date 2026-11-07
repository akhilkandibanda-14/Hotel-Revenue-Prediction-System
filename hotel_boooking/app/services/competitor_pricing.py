from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.utils import config

logger = logging.getLogger(__name__)

COMPETITOR_STORAGE_PATH = config.BASE_DIR / "reservations" / "competitor_benchmarks.json"


class CompetitorPricingService:
    """
    Competitive Pricing Intelligence Engine.
    Enables authorized manual entry of competitor rate benchmarks,
    computes Market Price Position Index (MPI), and evaluates competitive variance.
    Does NOT perform unauthorized external web scraping.
    """

    def __init__(self):
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        COMPETITOR_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not COMPETITOR_STORAGE_PATH.exists():
            default_benchmarks = [
                {
                    "id": "comp_001",
                    "date": "2026-07-01",
                    "competitor_name": "Grand Bay Luxury Resort",
                    "competitor_adr": 185.0,
                    "our_adr": 165.0,
                    "room_type": "Standard Deluxe",
                    "notes": "Direct competitor beachfront property",
                    "logged_at": "2026-07-01T10:00:00Z"
                },
                {
                    "id": "comp_002",
                    "date": "2026-07-01",
                    "competitor_name": "City Center Hotel & Suites",
                    "competitor_adr": 150.0,
                    "our_adr": 165.0,
                    "room_type": "Standard Deluxe",
                    "notes": "Midscale business competitor",
                    "logged_at": "2026-07-01T10:00:00Z"
                },
                {
                    "id": "comp_003",
                    "date": "2026-08-15",
                    "competitor_name": "Grand Bay Luxury Resort",
                    "competitor_adr": 220.0,
                    "our_adr": 205.0,
                    "room_type": "Executive Suite",
                    "notes": "High season peak weekend",
                    "logged_at": "2026-08-15T10:00:00Z"
                }
            ]
            COMPETITOR_STORAGE_PATH.write_text(json.dumps(default_benchmarks, indent=2), encoding="utf-8")

    def load_benchmarks(self) -> List[Dict[str, Any]]:
        self._ensure_storage()
        try:
            return json.loads(COMPETITOR_STORAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_benchmarks(self, records: List[Dict[str, Any]]) -> None:
        self._ensure_storage()
        COMPETITOR_STORAGE_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def add_competitor_entry(
        self,
        date_str: str,
        competitor_name: str,
        competitor_adr: float,
        our_adr: float,
        room_type: str = "Standard Deluxe",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Appends a manually entered competitor rate observation.
        """
        if competitor_adr <= 0 or our_adr <= 0:
            raise ValueError("ADR values must be strictly positive.")

        records = self.load_benchmarks()
        new_entry = {
            "id": f"comp_{len(records)+1:03d}",
            "date": date_str,
            "competitor_name": competitor_name.strip(),
            "competitor_adr": float(round(competitor_adr, 2)),
            "our_adr": float(round(our_adr, 2)),
            "room_type": room_type.strip(),
            "notes": notes.strip(),
            "logged_at": datetime.now().isoformat(),
        }
        records.append(new_entry)
        self.save_benchmarks(records)
        return new_entry

    def evaluate_competitive_position(
        self,
        our_adr: Optional[float] = None,
        competitor_adr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates relative price positioning, Market Price Index (MPI), and variance.
        """
        benchmarks = self.load_benchmarks()

        # If explicit values passed, evaluate single comparison
        if our_adr is not None and competitor_adr is not None and competitor_adr > 0:
            mpi = float(round((our_adr / competitor_adr) * 100.0, 2))
            diff_dollars = float(round(our_adr - competitor_adr, 2))
            diff_pct = float(round(((our_adr - competitor_adr) / competitor_adr) * 100.0, 2))

            if diff_pct > 5.0:
                position = "Premium Position (Priced Above Competitor)"
                action = "Justify premium rate with superior amenities/ratings; monitor occupancy resistance."
            elif diff_pct < -5.0:
                position = "Discount Position (Priced Below Competitor)"
                action = "Opportunity to increase ADR if occupancy is high; check rate parity compliance."
            else:
                position = "Parity Position (Priced Within +/- 5%)"
                action = "Rate aligned with competitive market baseline."

            single_comp = {
                "our_adr": our_adr,
                "competitor_adr": competitor_adr,
                "market_price_index_mpi": mpi,
                "dollar_gap": diff_dollars,
                "variance_pct": diff_pct,
                "position_classification": position,
                "suggested_action": action,
            }
        else:
            single_comp = None

        # Historical benchmark summary
        comp_summary = []
        if benchmarks:
            b_df = pd.DataFrame(benchmarks)
            for c_name, grp in b_df.groupby("competitor_name"):
                avg_comp = float(grp["competitor_adr"].mean())
                avg_our = float(grp["our_adr"].mean())
                avg_mpi = float(round((avg_our / max(1.0, avg_comp)) * 100.0, 2))
                comp_summary.append({
                    "competitor_name": str(c_name),
                    "recorded_samples": len(grp),
                    "competitor_average_adr": float(round(avg_comp, 2)),
                    "our_average_adr": float(round(avg_our, 2)),
                    "average_mpi": avg_mpi,
                })

        return {
            "status": "Success",
            "active_comparison": single_comp,
            "historical_benchmarks_count": len(benchmarks),
            "competitors_tracked": comp_summary,
            "recent_records": benchmarks[-10:],
            "data_governance_notice": (
                "Competitor intelligence relies solely on authorized manual benchmark entries. "
                "No automated web scraping of third-party platforms is performed."
            )
        }


# Singleton service
competitor_pricing_service = CompetitorPricingService()

