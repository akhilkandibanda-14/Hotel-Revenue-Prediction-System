"""
Live End-to-End Integration Test for Phase 3 Advanced Analytics & Forecasting.
Verifies all live API endpoints through HotelRevenueAPIClient.
"""

import sys
sys.path.insert(0, r"D:\Hotel_Revenue_Project\web_app_dashboards\web_app")

from api_client import HotelRevenueAPIClient

def test_live_analytics():
    client = HotelRevenueAPIClient(base_url="http://127.0.0.1:8000")

    print("[1/8] Checking Health...")
    ok, h, err = client.check_health()
    assert ok, f"Health check failed: {err}"
    print("      Health OK:", h)

    print("[2/8] Testing 7-Day Revenue Forecast...")
    ok, fc7, err = client.get_revenue_forecast(horizon=7)
    assert ok, f"Forecast failed: {err}"
    assert len(fc7["forecast"]) == 7
    print(f"      7-Day Revenue Forecast OK: Total=${fc7['total_projected_revenue']:,.2f}")

    print("[3/8] Testing 30-Day Occupancy Forecast...")
    ok, occ, err = client.get_occupancy_forecast(horizon=30)
    assert ok, f"Occupancy forecast failed: {err}"
    assert len(occ["forecast"]) == 30
    print(f"      30-Day Occupancy Forecast OK: Mean={occ['mean_projected_occupancy']*100:.1f}%")

    print("[4/8] Testing What-If Pricing Optimization Matrix...")
    ok, opt, err = client.get_pricing_optimization(base_adr=170.0, base_occupancy=0.72)
    assert ok, f"Pricing optimization failed: {err}"
    opt_rev = opt["optimal_revenue_scenario"]
    print(f"      Pricing Optimizer OK: Max Rev ADR=${opt_rev['projected_adr']} -> Revenue=${opt_rev['projected_revenue']:,.2f}")

    print("[5/8] Testing Global SHAP Feature Importance...")
    ok, sh_g, err = client.get_global_shap_importance(max_samples=30)
    assert ok, f"Global SHAP failed: {err}"
    print(f"      Global SHAP OK: Base=${sh_g['base_value']}, Top={sh_g['top_features'][:3]}")

    print("[6/8] Testing Local SHAP Instance Explanation...")
    ok, sh_l, err = client.explain_instance({
        "Date": "2026-07-20",
        "Occupancy_Rate": 0.88,
        "ADR": 210.0,
        "Available_Rooms": 120,
        "Reserved_Rooms": 105,
        "Total_Costs": 3200.0,
    })
    assert ok, f"Local SHAP failed: {err}"
    assert sh_l["sum_check_matches"]
    print(f"      Local SHAP OK: Predicted=${sh_l['predicted_revenue']:,.2f}, Matches={sh_l['sum_check_matches']}")

    print("[7/8] Testing Revenue Anomaly Detection...")
    ok, anom, err = client.get_anomalies(contamination=0.05)
    assert ok, f"Anomaly detection failed: {err}"
    print(f"      Anomalies OK: Found={anom['anomalies_detected']} records across {anom['total_records_analyzed']} total")

    print("[8/8] Testing Data Drift Monitoring...")
    ok, drift, err = client.assess_data_drift()
    assert ok, f"Drift assessment failed: {err}"
    print(f"      Data Drift OK: Status={drift['overall_status']}, Audited={drift['total_features_tested']}")

    print("\n>>> ALL 8 PHASE 3 LIVE INTEGRATION TESTS PASSED PERFECTLY! <<<")

if __name__ == "__main__":
    test_live_analytics()

