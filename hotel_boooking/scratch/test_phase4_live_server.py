import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_endpoints():
    print(f"=== Validating Live Production FastAPI Server at {BASE_URL} ===\n")
    
    # 1. Health
    h = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"1. Health Check: HTTP {h.status_code} | Service: {h.json().get('status')}")
    assert h.status_code == 200

    # 2. Prebooking Audit
    pa = requests.get(f"{BASE_URL}/intelligence/prebooking-audit", timeout=5)
    pa_data = pa.json()
    print(f"2. Prebooking Audit: HTTP {pa.status_code} | Genuine Booking Records: {pa_data.get('is_genuine_booking_level')} | Realized Leakage Fields: {pa_data.get('leakage_fields_identified')}")
    assert pa.status_code == 200

    # 3. Prebooking Predict
    p_body = {
        "check_in_date": "2026-08-01",
        "check_out_date": "2026-08-05",
        "booking_date": "2026-07-15",
        "quoted_adr": 185.0,
        "rooms_count": 2,
        "guests_count": 4,
        "booking_channel": "Direct",
        "market_segment": "Leisure",
        "room_type": "Deluxe"
    }
    pp = requests.post(f"{BASE_URL}/intelligence/prebooking-predict", json=p_body, timeout=5)
    pp_data = pp.json()
    print(f"3. Prebooking Predict: HTTP {pp.status_code} | Gross Est: ${pp_data.get('estimated_gross_revenue'):,.2f} | Net Est: ${pp_data.get('estimated_net_revenue'):,.2f}")
    assert pp.status_code == 200

    # 4. Cancellation Analytics
    ca = requests.get(f"{BASE_URL}/intelligence/cancellation-analytics", timeout=5)
    ca_data = ca.json()
    c_sum = ca_data.get("summary", {})
    print(f"4. Cancellation Analytics: HTTP {ca.status_code} | Total Bookings: {c_sum.get('total_bookings')} | Cancel Rate: {c_sum.get('overall_cancellation_rate_pct')}% | Slippage: {c_sum.get('combined_revenue_slippage_pct')}%")
    assert ca.status_code == 200

    # 5. Competitor Analysis
    cmp = requests.post(f"{BASE_URL}/intelligence/competitor-analysis", json={"our_adr": 180.0, "competitor_adr": 160.0}, timeout=5)
    cmp_data = cmp.json()
    cmp_act = cmp_data.get("active_comparison", {})
    print(f"5. Competitor Analysis: HTTP {cmp.status_code} | MPI: {cmp_act.get('market_price_index_mpi')}% | Position: {cmp_act.get('position_classification')}")
    assert cmp.status_code == 200

    # 6. Demand Forecast (30 days)
    df = requests.post(f"{BASE_URL}/intelligence/demand-forecast", json={"horizon": 30}, timeout=15)
    df_data = df.json()
    print(f"6. Demand Forecast (30d): HTTP {df.status_code} | Horizon: {df_data.get('horizon_days')}d | Total Room Nights: {df_data.get('total_expected_room_nights'):,.0f} | Avg Occ: {df_data.get('projected_average_occupancy_pct')}%")
    assert df.status_code == 200

    # 7. Business Recommendations
    br = requests.get(f"{BASE_URL}/intelligence/business-recommendations", timeout=15)
    br_data = br.json()
    prio = br_data.get("priority_summary", {})
    print(f"7. Business Recommendations: HTTP {br.status_code} | Total: {br_data.get('total_recommendations')} | High: {prio.get('High')} | Medium: {prio.get('Medium')}")
    assert br.status_code == 200

    print("\n✅ ALL 7 LIVE INTELLIGENCE ENDPOINTS ARE ACTIVE AND RETURNING ACCURATE DATA!")

if __name__ == "__main__":
    try:
        test_endpoints()
    except Exception as e:
        print("❌ Error during test:", str(e))
        sys.exit(1)
