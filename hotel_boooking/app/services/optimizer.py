from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.utils import config

logger = logging.getLogger(__name__)


def estimate_empirical_elasticity() -> Dict[str, Any]:
    """
    Computes empirical price elasticity of demand (epsilon = d ln(Q) / d ln(P))
    from the historical merged dataset.
    """
    active_path = config.MERGED_FOLDER / "active_dataset.csv"
    if not active_path.exists():
        active_path = config.BASE_DIR / "sample_training_dataset.csv"

    if not active_path.exists():
        # Fallback benchmark elasticity if dataset is missing
        return {
            "overall_elasticity": -0.35,
            "p_value": 0.05,
            "r_squared": 0.10,
            "segment_elasticity": {},
        }

    try:
        df = pd.read_csv(active_path)
        valid = df.dropna(subset=["Occupancy_Rate", "ADR"]).copy()
        valid = valid[(valid["Occupancy_Rate"] > 0) & (valid["ADR"] > 0)]

        if len(valid) < 10:
            return {
                "overall_elasticity": -0.35,
                "p_value": 0.05,
                "r_squared": 0.10,
                "segment_elasticity": {},
            }

        log_occ = np.log(valid["Occupancy_Rate"])
        log_adr = np.log(valid["ADR"])

        res = stats.linregress(log_adr, log_occ)
        overall_eps = float(res.slope)

        # Segment-specific elasticities
        seg_eps: Dict[str, float] = {}
        if "Market_Segment" in valid.columns:
            for seg, group in valid.groupby("Market_Segment"):
                if len(group) >= 10:
                    s_res = stats.linregress(np.log(group["ADR"]), np.log(group["Occupancy_Rate"]))
                    seg_eps[str(seg)] = round(float(s_res.slope), 4)

        return {
            "overall_elasticity": round(overall_eps, 4),
            "p_value": round(float(res.pvalue), 4),
            "r_squared": round(float(res.rvalue ** 2), 4),
            "segment_elasticity": seg_eps,
            "sample_size": len(valid),
        }
    except Exception as e:
        logger.warning(f"Error computing empirical elasticity: {e}")
        return {
            "overall_elasticity": -0.35,
            "p_value": 0.05,
            "r_squared": 0.10,
            "segment_elasticity": {},
        }


class PricingOptimizerService:
    """
    ADR and Revenue Optimization Service.
    Simulates demand shifts under price changes using empirical price elasticity.
    """

    def __init__(self):
        pass

    def get_baseline_defaults(self) -> Dict[str, float]:
        """
        Extracts representative baseline operational values from active dataset.
        """
        active_path = config.MERGED_FOLDER / "active_dataset.csv"
        if not active_path.exists():
            active_path = config.BASE_DIR / "sample_training_dataset.csv"

        if active_path.exists():
            try:
                df = pd.read_csv(active_path)
                return {
                    "base_adr": round(float(df["ADR"].median()), 2) if "ADR" in df.columns else 150.0,
                    "base_occupancy": round(float(df["Occupancy_Rate"].median()), 4) if "Occupancy_Rate" in df.columns else 0.70,
                    "total_rooms": int(df["Available_Rooms"].median()) if "Available_Rooms" in df.columns else 120,
                    "fixed_costs": round(float(df["Fixed_Costs"].median()), 2) if "Fixed_Costs" in df.columns else 2500.0,
                    "var_cost_per_room": round(float((df["Variable_Costs"] / df["Reserved_Rooms"].clip(lower=1)).median()), 2)
                    if "Variable_Costs" in df.columns and "Reserved_Rooms" in df.columns else 15.0,
                }
            except Exception:
                pass

        return {
            "base_adr": 150.0,
            "base_occupancy": 0.70,
            "total_rooms": 120,
            "fixed_costs": 2500.0,
            "var_cost_per_room": 15.0,
        }

    def simulate_pricing_matrix(
        self,
        base_adr: Optional[float] = None,
        base_occupancy: Optional[float] = None,
        total_rooms: Optional[int] = None,
        fixed_costs: Optional[float] = None,
        var_cost_per_room: Optional[float] = None,
        custom_elasticity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generates what-if pricing matrix across +/- 30% ADR change.
        Calculates projected Occupancy, Rooms Sold, Revenue, Costs, and Profit.
        """
        defaults = self.get_baseline_defaults()
        elasticity_meta = estimate_empirical_elasticity()

        b_adr = float(base_adr) if base_adr is not None and base_adr > 0 else defaults["base_adr"]
        b_occ = float(base_occupancy) if base_occupancy is not None and 0 < base_occupancy <= 1.0 else defaults["base_occupancy"]
        t_rooms = int(total_rooms) if total_rooms is not None and total_rooms > 0 else defaults["total_rooms"]
        f_costs = float(fixed_costs) if fixed_costs is not None and fixed_costs >= 0 else defaults["fixed_costs"]
        v_cost = float(var_cost_per_room) if var_cost_per_room is not None and var_cost_per_room >= 0 else defaults["var_cost_per_room"]

        # If user did not provide custom elasticity, use empirical estimate or -0.30 if positive
        if custom_elasticity is not None:
            active_elasticity = float(custom_elasticity)
        else:
            emp = elasticity_meta.get("overall_elasticity", -0.30)
            # In economic demand, price elasticity is typically <= 0. If empirical was weakly positive due to unobserved high-season collinearity, default to -0.30
            active_elasticity = emp if emp < 0 else -0.30

        # Base case
        base_rooms_sold = round(t_rooms * b_occ, 1)
        base_revenue = round(base_rooms_sold * b_adr, 2)
        base_total_costs = round(f_costs + (base_rooms_sold * v_cost), 2)
        base_profit = round(base_revenue - base_total_costs, 2)

        # Scenarios from -30% to +30% in steps of 5%
        delta_pcts = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
        scenarios: List[Dict[str, Any]] = []

        for delta in delta_pcts:
            pct_change = delta / 100.0
            new_adr = round(b_adr * (1.0 + pct_change), 2)

            # Demand adjustment based on elasticity: % dQ = eps * % dP
            expected_occ_change = active_elasticity * pct_change
            new_occ = float(np.clip(b_occ * (1.0 + expected_occ_change), 0.05, 1.0))

            rooms_sold = round(t_rooms * new_occ, 1)
            revenue = round(rooms_sold * new_adr, 2)
            total_costs = round(f_costs + (rooms_sold * v_cost), 2)
            profit = round(revenue - total_costs, 2)
            revpar = round(revenue / max(1, t_rooms), 2)

            rev_lift = round(revenue - base_revenue, 2)
            profit_lift = round(profit - base_profit, 2)

            scenarios.append({
                "adr_change_pct": delta,
                "projected_adr": new_adr,
                "projected_occupancy": round(new_occ, 4),
                "rooms_sold": rooms_sold,
                "projected_revenue": revenue,
                "total_costs": total_costs,
                "projected_profit": profit,
                "projected_revpar": revpar,
                "revenue_lift": rev_lift,
                "profit_lift": profit_lift,
                "is_baseline": bool(delta == 0),
            })

        optimal_revenue_scenario = max(scenarios, key=lambda s: s["projected_revenue"])
        optimal_profit_scenario = max(scenarios, key=lambda s: s["projected_profit"])

        boundary_explanation = (
            f"Microeconomic Boundary Dynamics: Under constant price elasticity ε = {active_elasticity:.2f}, "
            f"marginal revenue dRev/dP = Q * (1 + ε). Since |ε| < 1.0 (inelastic demand), 1 + ε > 0, "
            "meaning revenue monotonically increases with price up to the simulation boundary (+30%). "
            "In real hospitality markets, price sensitivity increases non-linearly at higher rates and competitors "
            "react. Therefore, boundary values represent optimal points within the linear simulation band, "
            "not unconstrained global market optima."
        ) if abs(active_elasticity) < 1.0 else (
            f"Under elastic demand ε = {active_elasticity:.2f} (|ε| > 1.0), rate cuts yield greater room volume and revenue."
        )

        assumptions = [
            f"Assumes active price elasticity of demand epsilon = {active_elasticity:.2f} (a 10% ADR increase shifts occupancy by {active_elasticity * 10:.1f}%).",
            f"Fixed hotel capacity is {t_rooms} rooms with variable cleaning/amenity cost of ${v_cost:.2f} per occupied room.",
            "Ceteris paribus assumption: Competitor pricing, macroeconomic factors, and marketing expenditure remain stationary.",
            "Demand response is bounded between physical minimum 5% occupancy and 100% capacity.",
            boundary_explanation,
        ]

        return {
            "baseline": {
                "base_adr": b_adr,
                "base_occupancy": b_occ,
                "total_rooms": t_rooms,
                "fixed_costs": f_costs,
                "var_cost_per_room": v_cost,
                "base_revenue": base_revenue,
                "base_profit": base_profit,
            },
            "active_elasticity": active_elasticity,
            "boundary_behavior_explanation": boundary_explanation,
            "empirical_elasticity_metadata": elasticity_meta,
            "optimal_revenue_scenario": optimal_revenue_scenario,
            "optimal_profit_scenario": optimal_profit_scenario,
            "scenarios": scenarios,
            "assumptions_and_constraints": assumptions,
        }


# Singleton service instance
optimizer_service = PricingOptimizerService()

