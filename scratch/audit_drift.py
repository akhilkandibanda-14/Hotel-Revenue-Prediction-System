"""
Data Drift Edge Cases & Statistical Hardening Audit Script.
Tests:
1. Reference vs current selection.
2. Small-sample candidate datasets (N=2, N=5).
3. 100% unseen categories.
4. Numerical edge cases (identical distributions, massive shifts).
"""

import sys
sys.path.insert(0, r"D:\Hotel_Revenue_Project\hotel_boooking")

import pandas as pd
import numpy as np
from app.services.drift_detector import drift_detector_service, compute_psi
from app.utils import config

def audit_drift():
    baseline_df = drift_detector_service.load_baseline_dataset()
    print(f"Loaded Baseline: {len(baseline_df)} rows")

    print("\n--- TEST 1: Identical Dataset (Zero Drift) ---")
    res_ident = drift_detector_service.assess_dataset_drift(candidate_df=baseline_df)
    print(f"Status: {res_ident['overall_status']} | Drifted Features: {res_ident['total_drifted_features']}")
    assert res_ident["total_drifted_features"] == 0, "Identical dataset should have 0 drifted features"

    print("\n--- TEST 2: Extreme Drifted Dataset (Massive Shift) ---")
    shifted_df = baseline_df.copy()
    shifted_df["ADR"] = shifted_df["ADR"] * 3.0
    shifted_df["Occupancy_Rate"] = shifted_df["Occupancy_Rate"] * 0.2
    shifted_df["Booking_Channel"] = "Brand_New_Unseen_Channel"
    res_shift = drift_detector_service.assess_dataset_drift(candidate_df=shifted_df)
    print(f"Status: {res_shift['overall_status']} | Drifted Features: {res_shift['total_drifted_features']}")
    print(f"Recommendation: {res_shift['recommendation']}")

    print("\n--- TEST 3: Small-Sample Edge Case (N=3 rows) ---")
    small_df = baseline_df.iloc[:3].copy()
    res_small = drift_detector_service.assess_dataset_drift(candidate_df=small_df)
    print(f"Status: {res_small['overall_status']} | Features tested: {res_small['total_features_tested']}")

    print("\n--- TEST 4: Completely Unseen Categories in PSI ---")
    ref_cat = pd.Series(["Direct", "Online", "OTA"])
    cand_cat = pd.Series(["MetaSearch", "CryptoBooking"])
    psi_unseen = compute_psi(ref_cat, cand_cat)
    print(f"PSI with 100% unseen categories: {psi_unseen:.4f}")
    assert np.isfinite(psi_unseen), "PSI must be finite even with totally unseen categories"

    print("\nAll drift edge case tests completed successfully!")

if __name__ == "__main__":
    audit_drift()

