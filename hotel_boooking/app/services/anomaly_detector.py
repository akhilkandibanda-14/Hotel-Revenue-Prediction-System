from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.utils import config

logger = logging.getLogger(__name__)


class AnomalyDetectorService:
    """
    Multi-method hotel anomaly detection service.
    Combines statistical outlier bounds (IQR & Z-score) with IsolationForest ML.
    """

    def __init__(self):
        pass

    def detect_anomalies(
        self,
        contamination: float = 0.05,
        z_threshold: float = 2.5,
    ) -> Dict[str, Any]:
        """
        Analyzes active dataset for historical revenue, occupancy, and ADR anomalies.
        """
        active_path = config.MERGED_FOLDER / "active_dataset.csv"
        if not active_path.exists():
            active_path = config.BASE_DIR / "sample_training_dataset.csv"

        if not active_path.exists():
            raise FileNotFoundError("Historical dataset not found for anomaly detection")

        df = pd.read_csv(active_path)
        if len(df) < 10:
            return {
                "total_records_analyzed": len(df),
                "anomalies_detected": 0,
                "anomaly_rate_pct": 0.0,
                "anomalies": [],
            }

        df["Date_Parsed"] = pd.to_datetime(df.get("Date", pd.Series()), errors="coerce")
        df = df.sort_values("Date_Parsed").reset_index(drop=True)

        target_metrics = ["Total_Revenue", "Occupancy_Rate", "ADR", "Total_Costs", "RevPAR"]
        available_metrics = [m for m in target_metrics if m in df.columns]

        # 1. IsolationForest detection across joint multidimensional space
        iso_cols = [c for c in available_metrics if df[c].notnull().all()]
        iso_anomalies_idx = set()
        if len(iso_cols) >= 2:
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(df[iso_cols])
            iso_anomalies_idx = set(np.where(preds == -1)[0])

        # 2. Statistical IQR and rolling Z-score detection
        anomalies_list: List[Dict[str, Any]] = []

        # Calculate statistics
        for idx, row in df.iterrows():
            is_iso = idx in iso_anomalies_idx

            flagged_reasons: List[str] = []
            highest_severity = "Low"
            primary_metric = "Total_Revenue" if "Total_Revenue" in df.columns else available_metrics[0]
            max_deviation = 0.0
            primary_actual = float(row.get(primary_metric, 0.0))
            primary_expected = float(df[primary_metric].median())

            # Check individual metrics against IQR
            for metric in ("Total_Revenue", "Occupancy_Rate", "ADR"):
                if metric not in df.columns:
                    continue

                val = float(row[metric])
                q1 = float(df[metric].quantile(0.25))
                q3 = float(df[metric].quantile(0.75))
                iqr = q3 - q1
                med = float(df[metric].median())
                std = float(df[metric].std()) if df[metric].std() > 0 else 1.0

                z_score = abs(val - med) / std
                dev_pct = ((val - med) / max(1e-4, abs(med))) * 100.0

                is_iqr_outlier = (val < q1 - 1.5 * iqr) or (val > q3 + 1.5 * iqr)
                is_z_outlier = z_score >= z_threshold

                if is_iqr_outlier or is_z_outlier:
                    if abs(dev_pct) > abs(max_deviation):
                        max_deviation = dev_pct
                        primary_metric = metric
                        primary_actual = val
                        primary_expected = med

                    direction_str = "Spike" if dev_pct > 0 else "Drop"
                    flagged_reasons.append(
                        f"{metric} {direction_str} ({dev_pct:+.1f}% vs median, Z={z_score:.1f})"
                    )

            if is_iso and not flagged_reasons:
                # Multivariate anomaly: calculate operational revenue deviation relative to historical median
                if "Total_Revenue" in df.columns:
                    rev_val = float(row.get("Total_Revenue", df["Total_Revenue"].median()))
                    rev_med = float(df["Total_Revenue"].median())
                    max_deviation = ((rev_val - rev_med) / max(1e-4, rev_med)) * 100.0
                    primary_actual = rev_val
                    primary_expected = rev_med
                    primary_metric = "Total_Revenue"
                    direction_str = "Spike" if max_deviation >= 0 else "Drop"
                    flagged_reasons.append(
                        f"Multivariate anomaly flagged by IsolationForest across joint features (Revenue {direction_str} {max_deviation:+.1f}% vs median)"
                    )
                else:
                    flagged_reasons.append("Multivariate anomaly flagged by IsolationForest across joint hotel features")

            # If either statistical or IsolationForest triggered
            if flagged_reasons or is_iso:
                # Determine severity: High, Medium, Low
                if (is_iso and abs(max_deviation) > 50.0) or (is_iso and len(flagged_reasons) >= 2 and abs(max_deviation) > 35.0):
                    highest_severity = "High"
                elif is_iso or abs(max_deviation) > 25.0:
                    highest_severity = "Medium"
                else:
                    highest_severity = "Low"

                detection_method = "Statistical & IsolationForest" if (is_iso and flagged_reasons) else (
                    "IsolationForest" if is_iso else "Statistical (IQR/Z-Score)"
                )

                direction = "High (Spike)" if max_deviation >= 0 else "Low (Drop)"

                date_str = row["Date_Parsed"].strftime("%Y-%m-%d") if pd.notnull(row["Date_Parsed"]) else str(row.get("Date", f"Row-{idx}"))

                # Construct human-readable diagnostic note
                occ_val = row.get("Occupancy_Rate", 0.0)
                adr_val = row.get("ADR", 0.0)
                rev_val = row.get("Total_Revenue", 0.0)
                diag_note = f"Observed Revenue ${rev_val:,.2f} with {occ_val*100:.0f}% occupancy at ${adr_val:.2f} ADR. " + "; ".join(flagged_reasons)

                anomalies_list.append({
                    "record_index": int(idx),
                    "date": date_str,
                    "metric": primary_metric,
                    "actual_value": round(primary_actual, 2),
                    "expected_value": round(primary_expected, 2),
                    "deviation_pct": round(max_deviation, 2),
                    "direction": direction,
                    "severity": highest_severity,
                    "detection_method": detection_method,
                    "diagnostic_note": diag_note,
                })

        # Sort anomalies by severity (High > Medium > Low) and then by absolute deviation
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        anomalies_list = sorted(
            anomalies_list,
            key=lambda x: (severity_order.get(x["severity"], 3), -abs(x["deviation_pct"])),
        )

        counts_by_severity = {
            "High": sum(1 for a in anomalies_list if a["severity"] == "High"),
            "Medium": sum(1 for a in anomalies_list if a["severity"] == "Medium"),
            "Low": sum(1 for a in anomalies_list if a["severity"] == "Low"),
        }

        counts_by_metric = {}
        for a in anomalies_list:
            m = a["metric"]
            counts_by_metric[m] = counts_by_metric.get(m, 0) + 1

        return {
            "total_records_analyzed": len(df),
            "anomalies_detected": len(anomalies_list),
            "anomaly_rate_pct": round((len(anomalies_list) / len(df)) * 100.0, 2),
            "counts_by_severity": counts_by_severity,
            "counts_by_metric": counts_by_metric,
            "anomalies": anomalies_list,
        }


# Singleton service instance
anomaly_detector_service = AnomalyDetectorService()
