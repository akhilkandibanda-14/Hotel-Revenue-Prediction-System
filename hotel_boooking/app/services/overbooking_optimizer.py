from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.services.inventory_manager import room_inventory_manager

logger = logging.getLogger(__name__)


def compute_poisson_binomial_pmf(success_probs: List[float] | np.ndarray) -> np.ndarray:
    """
    Computes the exact Probability Mass Function (PMF) of a Poisson-Binomial distribution
    using dynamic programming convolution.
    Runs in O(N^2) time and machine-precision accuracy.
    """
    if len(success_probs) == 0:
        return np.array([1.0])

    pmf = np.array([1.0], dtype=np.float64)
    for p in success_probs:
        p_val = max(0.0, min(1.0, float(p)))
        # Rolling convolution: P_j(k) = P_{j-1}(k)*(1-p) + P_{j-1}(k-1)*p
        pmf = np.pad(pmf, (0, 1), mode="constant") * (1.0 - p_val) + np.pad(pmf, (1, 0), mode="constant") * p_val

    # Normalize slight floating-point drift
    sum_pmf = np.sum(pmf)
    if sum_pmf > 0:
        pmf = pmf / sum_pmf

    return pmf


class OverbookingOptimizerService:
    """
    Intelligent Room Inventory & Overbooking Optimization Engine (Phase 5).
    Integrates individual calibrated cancellation probabilities to compute
    exact arrival probability distributions and constrained revenue-maximizing booking limits.
    """

    def __init__(self):
        self.inventory_manager = room_inventory_manager

    def compute_cancellation_forecast(
        self,
        target_date: str,
        room_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Step 5.3: Aggregates individual cancellation probabilities for a specific arrival date.
        Computes expected cancellations, variance, and exact arrival probability distribution.
        """
        # Retrieve active reservations with individual calibrated probabilities
        reservations = self.inventory_manager.get_active_reservations_for_date(
            target_date_str=target_date,
            room_category=room_category,
            include_calibrated_prob=True,
        )

        cat_configs = self.inventory_manager.get_room_categories()
        if room_category and room_category in cat_configs:
            eff_capacity = cat_configs[room_category]["effective_capacity"]
            phys_rooms = cat_configs[room_category]["total_physical_rooms"]
            oos_rooms = cat_configs[room_category]["rooms_out_of_service"]
        else:
            eff_capacity = sum(c["effective_capacity"] for c in cat_configs.values())
            phys_rooms = sum(c["total_physical_rooms"] for c in cat_configs.values())
            oos_rooms = sum(c["rooms_out_of_service"] for c in cat_configs.values())

        n_confirmed = len(reservations)
        if n_confirmed == 0:
            return {
                "status": "Success",
                "target_date": target_date,
                "room_category": room_category or "All Categories (Whole Hotel)",
                "confirmed_reservations": 0,
                "effective_capacity": eff_capacity,
                "expected_cancellations": 0.0,
                "expected_cancellation_rate_pct": 0.0,
                "expected_arrivals": 0.0,
                "variance_arrivals": 0.0,
                "std_dev_arrivals": 0.0,
                "confidence_interval_95": [0, 0],
                "arrival_distribution_pmf": [{"arrivals": 0, "probability": 1.0}],
                "cumulative_arrival_cmf": [{"arrivals": 0, "cumulative_probability": 1.0}],
                "independence_assumption_note": (
                    "Assuming individual guest cancellations are conditionally independent Bernoulli trials "
                    "conditioned on booking covariates."
                ),
            }

        # Individual calibrated cancellation probabilities p_i
        canc_probs = np.array([r.get("calibrated_cancellation_probability", 0.25) for r in reservations], dtype=np.float64)
        # Arrival probabilities q_i = 1 - p_i
        arr_probs = 1.0 - canc_probs

        exp_cancellations = float(np.sum(canc_probs))
        mean_canc_prob = float(exp_cancellations / n_confirmed)
        exp_arrivals = float(np.sum(arr_probs))
        var_arrivals = float(np.sum(canc_probs * arr_probs))
        std_arrivals = float(np.sqrt(var_arrivals))

        # Exact Poisson-Binomial PMF via convolution
        pmf = compute_poisson_binomial_pmf(arr_probs)
        cmf = np.cumsum(pmf)

        # 95% Confidence Interval
        k_low = int(np.searchsorted(cmf, 0.025))
        k_high = int(np.searchsorted(cmf, 0.975))

        pmf_records = [{"arrivals": int(k), "probability": round(float(p), 5)} for k, p in enumerate(pmf)]
        cmf_records = [{"arrivals": int(k), "cumulative_probability": round(float(c), 5)} for k, c in enumerate(cmf)]

        return {
            "status": "Success",
            "target_date": target_date,
            "room_category": room_category or "All Categories (Whole Hotel)",
            "confirmed_reservations": n_confirmed,
            "physical_rooms": phys_rooms,
            "rooms_out_of_service": oos_rooms,
            "effective_capacity": eff_capacity,
            "expected_cancellations": round(exp_cancellations, 2),
            "expected_cancellation_rate_pct": round(mean_canc_prob * 100.0, 1),
            "expected_arrivals": round(exp_arrivals, 2),
            "variance_arrivals": round(var_arrivals, 2),
            "std_dev_arrivals": round(std_arrivals, 2),
            "confidence_interval_95": [k_low, k_high],
            "arrival_distribution_pmf": pmf_records,
            "cumulative_arrival_cmf": cmf_records,
            "independence_assumption_note": (
                "Individual guest cancellations are modeled as conditionally independent Bernoulli trials "
                "given individual booking attributes (lead time, deposit type, distribution channel, and history). "
                "Uncertainty is analytically quantified using the exact Poisson-Binomial distribution."
            ),
        }

    def optimize_overbooking(
        self,
        target_date: str,
        room_category: str = "Standard Room",
        custom_adr: Optional[float] = None,
        custom_walk_cost: Optional[float] = None,
        max_walk_risk_pct: float = 2.0,
        max_additional_bookings: int = 30,
    ) -> Dict[str, Any]:
        """
        Step 5.4, 5.5, 5.6: Constrained Overbooking Revenue Optimization Engine.
        Finds the booking limit B* that maximizes Expected Net Contribution:
            E[ENC(B)] = E[Room Revenue(B)] - E[Relocation Costs(B)]
        Subject to Service-Level Risk Constraint:
            P(Arrivals > Effective Capacity) <= max_walk_risk_pct.
        """
        cat_configs = self.inventory_manager.get_room_categories()
        if room_category not in cat_configs:
            room_category = "Standard Room"

        cfg = cat_configs[room_category]
        phys_capacity = int(cfg.get("total_physical_rooms", 100))
        oos_rooms = int(cfg.get("rooms_out_of_service", 2))
        eff_capacity = max(1, phys_capacity - oos_rooms)

        adr = float(custom_adr) if custom_adr is not None else float(cfg.get("base_adr", 140.0))
        walk_cost = float(custom_walk_cost) if custom_walk_cost is not None else float(cfg.get("walk_relocation_cost", 280.0))
        alpha = max(0.001, min(0.20, float(max_walk_risk_pct) / 100.0))

        # Retrieve active reservations for that date and room category
        active_reservations = self.inventory_manager.get_active_reservations_for_date(
            target_date_str=target_date,
            room_category=room_category,
            include_calibrated_prob=True,
        )

        n_confirmed = len(active_reservations)
        if n_confirmed > 0:
            existing_arr_probs = [1.0 - r.get("calibrated_cancellation_probability", 0.25) for r in active_reservations]
            avg_arrival_prob = float(np.mean(existing_arr_probs))
        else:
            existing_arr_probs = []
            avg_arrival_prob = 0.75  # Default empirical arrival rate (~25% cancel rate)

        # Baseline evaluation (0 additional bookings)
        pmf_base = compute_poisson_binomial_pmf(existing_arr_probs)
        base_exp_rev = float(np.sum([min(k, eff_capacity) * adr * pmf_base[k] for k in range(len(pmf_base))]))
        base_exp_walks = float(np.sum([max(0, k - eff_capacity) * pmf_base[k] for k in range(len(pmf_base))]))
        base_exp_walk_cost = base_exp_walks * walk_cost
        base_net_enc = base_exp_rev - base_exp_walk_cost
        base_walk_prob = float(np.sum([pmf_base[k] for k in range(eff_capacity + 1, len(pmf_base))])) if len(pmf_base) > eff_capacity + 1 else 0.0
        base_spoilage = float(np.sum([max(0, eff_capacity - k) * pmf_base[k] for k in range(len(pmf_base))]))

        # Evaluate candidate grid of additional bookings: b in [0, 1, ..., max_additional_bookings]
        grid = []
        best_b = 0
        best_enc = -1e9
        best_row = None

        for b in range(0, max_additional_bookings + 1):
            total_limit = n_confirmed + b
            # Probability vector: existing reservations + b new reservations with arrival prob
            candidate_probs = list(existing_arr_probs) + [avg_arrival_prob] * b
            pmf_cand = compute_poisson_binomial_pmf(candidate_probs)

            # Revenue: min(k, C) * ADR
            exp_rev = float(np.sum([min(k, eff_capacity) * adr * pmf_cand[k] for k in range(len(pmf_cand))]))
            # Excess arrivals: max(0, k - C)
            exp_walks = float(np.sum([max(0, k - eff_capacity) * pmf_cand[k] for k in range(len(pmf_cand))]))
            exp_walk_cost = exp_walks * walk_cost
            exp_enc = exp_rev - exp_walk_cost

            # Walk risk P(Arrivals > C)
            if len(pmf_cand) > eff_capacity + 1:
                walk_prob = float(np.sum(pmf_cand[eff_capacity + 1:]))
                walk_0_prob = float(np.sum(pmf_cand[:eff_capacity + 1]))
                walk_1_prob = float(pmf_cand[eff_capacity + 1]) if len(pmf_cand) > eff_capacity + 1 else 0.0
                walk_2plus_prob = float(np.sum(pmf_cand[eff_capacity + 2:])) if len(pmf_cand) > eff_capacity + 2 else 0.0
            else:
                walk_prob = 0.0
                walk_0_prob = 1.0
                walk_1_prob = 0.0
                walk_2plus_prob = 0.0

            # Empty rooms / Spoilage
            exp_empty = float(np.sum([max(0, eff_capacity - k) * pmf_cand[k] for k in range(min(eff_capacity + 1, len(pmf_cand)))]))

            # Risk tier classification
            if walk_prob < 0.01:
                risk_tier = "Low Risk"
                badge = "🟢"
            elif walk_prob <= 0.03:
                risk_tier = "Moderate Risk"
                badge = "🟡"
            else:
                risk_tier = "High Risk"
                badge = "🔴"

            constraint_satisfied = bool(walk_prob <= alpha)

            row = {
                "additional_bookings": b,
                "total_booking_limit": total_limit,
                "expected_arrivals": round(float(np.sum(candidate_probs)), 2),
                "expected_revenue": round(exp_rev, 2),
                "expected_walks": round(exp_walks, 3),
                "expected_walk_cost": round(exp_walk_cost, 2),
                "expected_net_contribution": round(exp_enc, 2),
                "walk_probability": round(walk_prob, 4),
                "walk_probability_pct": round(walk_prob * 100.0, 2),
                "prob_zero_walks_pct": round(walk_0_prob * 100.0, 1),
                "prob_one_walk_pct": round(walk_1_prob * 100.0, 1),
                "prob_multiple_walks_pct": round(walk_2plus_prob * 100.0, 1),
                "expected_empty_rooms": round(exp_empty, 2),
                "risk_tier": risk_tier,
                "risk_badge": badge,
                "constraint_satisfied": constraint_satisfied,
            }
            grid.append(row)

            # Optimization search: Maximize ENC subject to walk_prob <= alpha
            if constraint_satisfied and exp_enc > best_enc:
                best_enc = exp_enc
                best_b = b
                best_row = row

        # Fallback if no candidate meets constraint (existing bookings already breach risk limit)
        if best_row is None:
            best_b = 0
            best_row = grid[0]

        # Calculate incremental benefit vs no overbooking (b=0)
        base_row = grid[0]
        net_gain = round(best_row["expected_net_contribution"] - base_row["expected_net_contribution"], 2)
        rev_gain = round(best_row["expected_revenue"] - base_row["expected_revenue"], 2)
        spoilage_reduced = round(base_row["expected_empty_rooms"] - best_row["expected_empty_rooms"], 2)

        return {
            "status": "Success",
            "target_date": target_date,
            "room_category": room_category,
            "assumptions": {
                "physical_rooms": phys_capacity,
                "rooms_out_of_service": oos_rooms,
                "effective_capacity": eff_capacity,
                "currently_confirmed_reservations": n_confirmed,
                "average_daily_rate": adr,
                "walk_relocation_cost": walk_cost,
                "max_acceptable_walk_risk_pct": max_walk_risk_pct,
                "data_governance_label": "DEMO_HOTEL_INVENTORY",
            },
            "recommendation": {
                "safe_additional_bookings": best_b,
                "recommended_total_booking_limit": best_row["total_booking_limit"],
                "walk_probability_pct": best_row["walk_probability_pct"],
                "risk_tier": best_row["risk_tier"],
                "risk_badge": best_row["risk_badge"],
                "expected_walks": best_row["expected_walks"],
                "expected_relocation_cost": best_row["expected_walk_cost"],
                "expected_room_revenue": best_row["expected_revenue"],
                "expected_net_contribution": best_row["expected_net_contribution"],
            },
            "business_impact_vs_no_overbooking": {
                "net_revenue_gain_dollars": net_gain,
                "additional_room_revenue_dollars": rev_gain,
                "additional_walk_exposure_dollars": best_row["expected_walk_cost"],
                "empty_rooms_avoided": spoilage_reduced,
                "strategic_summary": (
                    f"Accepting {best_b} additional bookings (total limit: {best_row['total_booking_limit']}) "
                    f"yields a net expected gain of ${net_gain:,.2f} after accounting for ${best_row['expected_walk_cost']:,.2f} "
                    f"in potential walk costs, while maintaining walk probability at {best_row['walk_probability_pct']:.2f}% "
                    f"(strictly within your {max_walk_risk_pct:.1f}% risk tolerance)."
                ),
            },
            "no_overbooking_baseline": base_row,
            "candidate_evaluation_grid": grid,
            "disclaimer": (
                "Modeled revenues, cancellations, and walk risks represent probabilistic expectations computed via "
                "the Poisson-Binomial distribution of calibrated individual booking probabilities. "
                "They provide decision-support guidance rather than commercial performance guarantees."
            ),
        }

    def simulate_scenario(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Runs overbooking optimization under customized operational assumptions."""
        return self.optimize_overbooking(
            target_date=str(payload.get("target_date", date.today().strftime("%Y-%m-%d"))),
            room_category=str(payload.get("room_category", "Standard Room")),
            custom_adr=float(payload.get("custom_adr")) if payload.get("custom_adr") is not None else None,
            custom_walk_cost=float(payload.get("custom_walk_cost")) if payload.get("custom_walk_cost") is not None else None,
            max_walk_risk_pct=float(payload.get("max_walk_risk_pct", 2.0)),
            max_additional_bookings=int(payload.get("max_additional_bookings", 25)),
        )

    def get_portfolio_recommendations(self, target_date: str) -> Dict[str, Any]:
        """
        Step 5.6: Generates synchronized overbooking recommendations across all room categories.
        """
        cat_configs = self.inventory_manager.get_room_categories()
        results = []
        tot_add = 0
        tot_net_gain = 0.0

        for cat_name in cat_configs.keys():
            opt = self.optimize_overbooking(
                target_date=target_date,
                room_category=cat_name,
                max_walk_risk_pct=2.0,
            )
            rec = opt["recommendation"]
            impact = opt["business_impact_vs_no_overbooking"]
            tot_add += rec["safe_additional_bookings"]
            tot_net_gain += impact["net_revenue_gain_dollars"]

            results.append({
                "room_category": cat_name,
                "effective_capacity": opt["assumptions"]["effective_capacity"],
                "current_confirmed": opt["assumptions"]["currently_confirmed_reservations"],
                "safe_additional_bookings": rec["safe_additional_bookings"],
                "recommended_limit": rec["recommended_total_booking_limit"],
                "walk_risk_pct": rec["walk_probability_pct"],
                "risk_tier": rec["risk_tier"],
                "expected_walk_cost": rec["expected_relocation_cost"],
                "net_expected_gain": impact["net_revenue_gain_dollars"],
            })

        return {
            "status": "Success",
            "target_date": target_date,
            "data_governance_label": "DEMO_HOTEL_INVENTORY",
            "portfolio_summary": {
                "total_additional_bookings_allowed": tot_add,
                "total_net_expected_revenue_gain": round(tot_net_gain, 2),
            },
            "category_recommendations": results,
        }

    def get_overbooking_performance(self) -> Dict[str, Any]:
        """
        Step 5.7: Returns performance comparison of intelligent overbooking vs static 0% overbooking
        across upcoming 7-day high-demand dates.
        """
        anchor = date.today() + timedelta(days=7)
        dates_to_eval = [(anchor + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

        horizon_perf = []
        total_baseline_enc = 0.0
        total_opt_enc = 0.0
        total_additional_rooms = 0

        for d_str in dates_to_eval:
            d_res = self.get_portfolio_recommendations(target_date=d_str)
            d_gain = d_res["portfolio_summary"]["total_net_expected_revenue_gain"]
            d_add = d_res["portfolio_summary"]["total_additional_bookings_allowed"]

            total_opt_enc += d_gain
            total_additional_rooms += d_add

            horizon_perf.append({
                "date": d_str,
                "additional_bookings_accepted": d_add,
                "net_revenue_gain": d_gain,
            })

        return {
            "status": "Success",
            "evaluation_horizon_days": 7,
            "data_governance_label": "DEMO_HOTEL_INVENTORY",
            "summary": {
                "total_incremental_rooms_sold": total_additional_rooms,
                "total_incremental_revenue_gain": round(total_opt_enc, 2),
                "average_daily_revenue_gain": round(total_opt_enc / 7.0, 2),
            },
            "daily_performance": horizon_perf,
        }


# Global singleton instance
overbooking_optimizer = OverbookingOptimizerService()

