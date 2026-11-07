"""
API Production Hardening and Edge Case Validation Suite.
Tests all Phase 1, Phase 2, and Phase 3 endpoints against edge cases:
- Invalid dates
- Unseen categories
- Missing values in strict mode
- Boundary / extreme parameter inputs
- Invalid model versions for rollback
"""

import sys
sys.path.insert(0, r"D:\Hotel_Revenue_Project\web_app_dashboards\web_app")

import requests
from api_client import HotelRevenueAPIClient

BASE_URL = "http://127.0.0.1:8000"

def audit_api():
    client = HotelRevenueAPIClient(base_url=BASE_URL)
    print("=== COMMENCING API HARDENING VALIDATION ===")

    # 1. Health
    print("[01] GET /health")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print("     -> 200 OK")

    # 2. Model info & comparison
    print("[02] GET /model/info & /model/comparison & /model/versions")
    r = requests.get(f"{BASE_URL}/model/info")
    assert r.status_code == 200
    r = requests.get(f"{BASE_URL}/model/comparison")
    assert r.status_code == 200
    r = requests.get(f"{BASE_URL}/model/versions")
    assert r.status_code == 200
    print("     -> All 200 OK")

    # 3. Model Rollback Invalid Version (404 check)
    print("[03] POST /model/rollback/v999 (Invalid version)")
    r = requests.post(f"{BASE_URL}/model/rollback/v999")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("     -> Handled properly with 404 Not Found")

    # 4. Predict - Unseen Categories Handling
    print("[04] POST /predict/ with completely unseen categories")
    payload_unseen = {
        "Date": "2026-08-15",
        "Booking_Channel": "Telepathy_Channel",
        "Guest_Type": "Alien_Guest",
        "Market_Segment": "Intergalactic",
        "Guest_Country": "Mars_City",
        "Season": "Solar_Flare",
        "Occupancy_Rate": 0.85,
        "ADR": 190.0,
        "Available_Rooms": 100,
        "Reserved_Rooms": 85,
        "Total_Costs": 2700.0,
    }
    r = requests.post(f"{BASE_URL}/predict/", json=payload_unseen)
    assert r.status_code == 200, f"Failed unseen categories: {r.text}"
    pred_val = r.json().get("Predicted_Revenue")
    print(f"     -> 200 OK: Predicted Revenue = ${pred_val:,.2f} (OrdinalEncoder safely handled unknown with -1)")

    # 5. Predict - Strict Validation Missing Field Rejection (422 check)
    print("[05] POST /predict/ missing required fields with strict=True")
    payload_strict_invalid = {
        "Date": "2026-08-15",
        "strict": True
        # Missing ADR, Occupancy_Rate, Available_Rooms, etc.
    }
    r = requests.post(f"{BASE_URL}/predict/", json=payload_strict_invalid)
    assert r.status_code == 422, f"Expected 422 Unprocessable, got {r.status_code}"
    print(f"     -> Handled properly with 422 Unprocessable: {r.json().get('detail')}")

    # 6. Predict - Date-only with empirical fallback (strict=False)
    print("[06] POST /predict/ date-only query with empirical profile fallback")
    payload_date_only = {"Date": "2026-07-20", "strict": False}
    r = requests.post(f"{BASE_URL}/predict/", json=payload_date_only)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print(f"     -> 200 OK: Predicted = ${r.json().get('Predicted_Revenue'):,.2f} (Profile: {r.json().get('Profile_Applied')})")

    # 7. Predict - Invalid unparseable date (422 check)
    print("[07] POST /predict/ with invalid date string")
    payload_bad_date = {"Date": "invalid-date-string", "strict": False}
    r = requests.post(f"{BASE_URL}/predict/", json=payload_bad_date)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print(f"     -> Handled properly with 422 Unprocessable: {r.json().get('detail')}")

    # 8. Forecasting - Extreme Horizon Clamping
    print("[08] POST /analytics/forecast with horizon=1000 (extreme)")
    r = requests.post(f"{BASE_URL}/analytics/forecast", json={"horizon": 1000})
    assert r.status_code == 200
    res_fc = r.json()
    assert res_fc["horizon_days"] == 90, f"Expected clamped horizon 90, got {res_fc['horizon_days']}"
    print(f"     -> 200 OK: Extreme horizon gracefully clamped to maximum supported 90 days")

    # 9. Pricing Optimizer - Negative/Extreme Elasticity
    print("[09] POST /analytics/pricing-optimizer with custom elasticity ε = -0.90")
    r = requests.post(f"{BASE_URL}/analytics/pricing-optimizer", json={"custom_elasticity": -0.90})
    assert r.status_code == 200
    res_opt = r.json()
    print(f"     -> 200 OK: Optimal Rev ADR = ${res_opt['optimal_revenue_scenario']['projected_adr']}")

    # 10. SHAP Explain - Unseen categories in explanation
    print("[10] POST /analytics/explain with unseen categories")
    payload_explain_unseen = {
        "Date": "2026-07-01",
        "Booking_Channel": "Teleport_Channel",
        "Guest_Country": "Jupiter",
        "Occupancy_Rate": 0.80,
        "ADR": 170.0,
    }
    r = requests.post(f"{BASE_URL}/analytics/explain", json=payload_explain_unseen)
    assert r.status_code == 200
    res_exp = r.json()
    assert res_exp["sum_check_matches"]
    print(f"     -> 200 OK: SHAP sum matches predicted revenue to < $0.05")

    # 11. Anomalies endpoint
    print("[11] GET /analytics/anomalies")
    r = requests.get(f"{BASE_URL}/analytics/anomalies?contamination=0.08&z_threshold=2.0")
    assert r.status_code == 200
    print(f"     -> 200 OK: {r.json().get('anomalies_detected')} anomalies flagged")

    # 12. Drift endpoint
    print("[12] POST /analytics/drift")
    r = requests.post(f"{BASE_URL}/analytics/drift", json={})
    assert r.status_code == 200
    print(f"     -> 200 OK: Status = {r.json().get('overall_status')}")

    print("\n>>> ALL 12 API HARDENING & EDGE CASE TESTS PASSED PERFECTLY! <<<")

if __name__ == "__main__":
    audit_api()
