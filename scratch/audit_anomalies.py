"""
Anomaly Detection Audit Script.
Examines detected anomalies, severity thresholds, duplicate dates, and business context.
"""

import sys
sys.path.insert(0, r"D:\Hotel_Revenue_Project\hotel_boooking")

import pandas as pd
from app.services.anomaly_detector import anomaly_detector_service
from app.utils import config

def audit_anomalies():
    df = pd.read_csv(config.MERGED_FOLDER / "active_dataset.csv")
    print(f"Total raw rows: {len(df)} | Unique dates: {df['Date'].nunique()}")

    # Check for identical duplicate rows
    dups = df.duplicated()
    print(f"Identical duplicate rows: {dups.sum()}")

    # Run anomaly detection
    res = anomaly_detector_service.detect_anomalies(contamination=0.05, z_threshold=2.5)
    anoms = res["anomalies"]
    print(f"\nTotal Anomalies Detected: {len(anoms)}")
    print(f"Severity Breakdown: {res['counts_by_severity']}")
    print(f"Metric Breakdown: {res['counts_by_metric']}")

    print("\n--- DETAILED ANOMALY LEDGER AUDIT ---")
    anom_df = pd.DataFrame(anoms)

    for i, a in enumerate(anoms):
        # Classify business context
        direction = a["direction"]
        metric = a["metric"]
        dev = a["deviation_pct"]
        sev = a["severity"]
        note = a["diagnostic_note"]

        business_type = "High-Demand Surge (Peak)" if dev > 0 else "Low-Demand Slump / Off-Peak"
        if "Total_Revenue" in metric and dev > 100:
            business_type = "Record Revenue Day / Full Buyout"
        elif "Occupancy_Rate" in metric and dev < -40:
            business_type = "Severe Under-Occupancy Incident"

        print(f"[{i+1:02d}] {a['date']} | {sev:6s} | {metric:15s} | Dev: {dev:+6.1f}% | Context: {business_type}")
        print(f"     Note: {note}")

if __name__ == "__main__":
    audit_anomalies()

