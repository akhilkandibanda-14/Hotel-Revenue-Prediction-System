"""
Pricing Optimizer Sensitivity and Microeconomic Boundary Analysis Audit.
"""

import sys
sys.path.insert(0, r"D:\Hotel_Revenue_Project\hotel_boooking")

import numpy as np
import pandas as pd
from scipy import stats
from app.services.optimizer import optimizer_service, estimate_empirical_elasticity

def audit_pricing():
    meta = estimate_empirical_elasticity()
    print("--- EMPIRICAL ELASTICITY METADATA ---")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\n--- SENSITIVITY TO ELASTICITY ASSUMPTIONS ---")
    eps_values = [-0.10, -0.35, -0.60, -0.90, -1.20]

    for eps in eps_values:
        res = optimizer_service.simulate_pricing_matrix(
            base_adr=160.0,
            base_occupancy=0.70,
            total_rooms=100,
            fixed_costs=2000.0,
            var_cost_per_room=15.0,
            custom_elasticity=eps
        )
        opt_rev = res["optimal_revenue_scenario"]
        opt_prof = res["optimal_profit_scenario"]
        print(f"\nAssumed Elasticity ε = {eps:+.2f}:")
        print(f"  Rev-Maximizing:  ADR Change {opt_rev['adr_change_pct']:+3d}% (${opt_rev['projected_adr']:.2f}) -> Revenue: ${opt_rev['projected_revenue']:,.2f} | Occ: {opt_rev['projected_occupancy']*100:.1f}%")
        print(f"  Profit-Maximizing: ADR Change {opt_prof['adr_change_pct']:+3d}% (${opt_prof['projected_adr']:.2f}) -> Profit:  ${opt_prof['projected_profit']:,.2f} | Occ: {opt_prof['projected_occupancy']*100:.1f}%")

if __name__ == "__main__":
    audit_pricing()

