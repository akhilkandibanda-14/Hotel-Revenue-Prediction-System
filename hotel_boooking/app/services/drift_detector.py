from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from app.utils import config

logger = logging.getLogger(__name__)

NUMERICAL_DRIFT_COLS = [
    "Occupancy_Rate",
    "ADR",
    "Available_Rooms",
    "Reserved_Rooms",
    "Fixed_Costs",
    "Variable_Costs",
    "Total_Costs",
    "RevPAR",
    "Total_Revenue",
]

CATEGORICAL_DRIFT_COLS = [
    "Booking_Channel",
    "Guest_Type",
    "Market_Segment",
    "Guest_Country",
    "Season",
    "Month",
]


def compute_psi(reference: pd.Series, candidate: pd.Series, epsilon: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) between two categorical distributions.
    PSI = sum((Actual - Expected) * ln(Actual / Expected))
    """
    ref_clean = reference.dropna().astype(str)
    cand_clean = candidate.dropna().astype(str)

    all_categories = sorted(list(set(ref_clean.unique()).union(set(cand_clean.unique()))))
    if not all_categories:
        return 0.0

    ref_dist = ref_clean.value_counts(normalize=True)
    cand_dist = cand_clean.value_counts(normalize=True)

    psi_total = 0.0
    for cat in all_categories:
        p_ref = float(ref_dist.get(cat, epsilon))
        p_cand = float(cand_dist.get(cat, epsilon))
        psi_total += (p_cand - p_ref) * np.log(p_cand / p_ref)

    return float(max(0.0, psi_total))


class DriftDetectorService:
    """
    Statistical Data Drift Monitoring Service.
    Applies two-sample Kolmogorov-Smirnov tests, Wasserstein distances,
    categorical PSI, and missing-rate tracking against the baseline training distribution.
    """

    def __init__(self):
        pass

    def load_baseline_dataset(self) -> pd.DataFrame:
        active_path = config.MERGED_FOLDER / "active_dataset.csv"
        if not active_path.exists():
            active_path = config.BASE_DIR / "sample_training_dataset.csv"

        if not active_path.exists():
            raise FileNotFoundError("Baseline training dataset not found for drift detection")

        return pd.read_csv(active_path)

    def assess_dataset_drift(
        self,
        candidate_df: Optional[pd.DataFrame] = None,
        candidate_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assesses drift between the baseline reference dataset and a candidate dataset.
        If no candidate provided, performs a temporal split test (e.g. first 70% vs last 30%).
        """
        baseline_df = self.load_baseline_dataset()

        if candidate_df is None:
            if candidate_file_path:
                candidate_df = pd.read_csv(candidate_file_path)
            else:
                # Self-split temporal drift test across historical series
                split_idx = int(len(baseline_df) * 0.70)
                candidate_df = baseline_df.iloc[split_idx:].copy()
                baseline_df = baseline_df.iloc[:split_idx].copy()

        numerical_results: List[Dict[str, Any]] = []
        drifted_num_count = 0

        for col in NUMERICAL_DRIFT_COLS:
            if col in baseline_df.columns and col in candidate_df.columns:
                base_vals = pd.to_numeric(baseline_df[col], errors="coerce").dropna()
                cand_vals = pd.to_numeric(candidate_df[col], errors="coerce").dropna()

                if len(base_vals) >= 5 and len(cand_vals) >= 5:
                    ks_stat = ks_2samp(base_vals, cand_vals)
                    wd_stat = wasserstein_distance(base_vals, cand_vals)

                    is_drifted = bool(ks_stat.pvalue < 0.05)
                    if is_drifted:
                        drifted_num_count += 1

                    numerical_results.append({
                        "feature": col,
                        "ks_statistic": float(round(ks_stat.statistic, 4)),
                        "p_value": float(round(ks_stat.pvalue, 4)),
                        "wasserstein_distance": float(round(wd_stat, 4)),
                        "baseline_mean": float(round(base_vals.mean(), 2)),
                        "candidate_mean": float(round(cand_vals.mean(), 2)),
                        "drift_detected": is_drifted,
                        "status": "Drift Detected" if is_drifted else "Stable",
                    })

        categorical_results: List[Dict[str, Any]] = []
        drifted_cat_count = 0

        for col in CATEGORICAL_DRIFT_COLS:
            if col in baseline_df.columns and col in candidate_df.columns:
                ref_series = baseline_df[col]
                cand_series = candidate_df[col]

                psi_score = compute_psi(ref_series, cand_series)
                ref_cats = set(ref_series.dropna().astype(str).unique())
                cand_cats = set(cand_series.dropna().astype(str).unique())
                unseen = list(cand_cats - ref_cats)

                if psi_score >= 0.20:
                    status = "Significant Drift"
                    drifted_cat_count += 1
                elif psi_score >= 0.10:
                    status = "Moderate Drift"
                    drifted_cat_count += 1
                else:
                    status = "Stable"

                categorical_results.append({
                    "feature": col,
                    "psi_score": float(round(psi_score, 4)),
                    "unseen_categories_count": len(unseen),
                    "unseen_categories": unseen[:5],
                    "status": status,
                })

        # Missing rate drift
        missing_rate_results: List[Dict[str, Any]] = []
        for col in set(baseline_df.columns).intersection(set(candidate_df.columns)):
            ref_miss_pct = float(round((baseline_df[col].isnull().sum() / len(baseline_df)) * 100.0, 2))
            cand_miss_pct = float(round((candidate_df[col].isnull().sum() / len(candidate_df)) * 100.0, 2))
            diff = float(round(cand_miss_pct - ref_miss_pct, 2))
            if abs(diff) > 5.0:
                missing_rate_results.append({
                    "feature": col,
                    "baseline_missing_pct": ref_miss_pct,
                    "candidate_missing_pct": cand_miss_pct,
                    "difference_pct": diff,
                })

        total_features_tested = len(numerical_results) + len(categorical_results)
        total_drifted = drifted_num_count + drifted_cat_count

        # Overall drift status determination
        if total_drifted == 0:
            overall_status = "No Drift"
            alert_color = "green"
            recommendation = "Data distributions match training baseline. No action required."
        elif total_drifted <= 2:
            overall_status = "Moderate Drift"
            alert_color = "yellow"
            recommendation = (
                f"{total_drifted} feature(s) showed moderate distributional shift. "
                "Monitor incoming reservation metrics closely."
            )
        else:
            overall_status = "Severe Drift"
            alert_color = "red"
            recommendation = (
                f"{total_drifted} features showed statistically significant drift. "
                "Retraining the production model on the updated dataset is strongly recommended."
            )

        sample_size_warning = None
        if len(candidate_df) < 10:
            sample_size_warning = (
                f"Candidate sample has only {len(candidate_df)} records (< 10). "
                "Statistical power for two-sample KS and PSI tests is limited; "
                "evaluate larger operational batches to avoid false positives."
            )
            recommendation += f" [Warning: {sample_size_warning}]"

        return {
            "overall_status": overall_status,
            "alert_color": alert_color,
            "recommendation": recommendation,
            "sample_size_warning": sample_size_warning,
            "total_features_tested": total_features_tested,
            "total_drifted_features": total_drifted,
            "baseline_records_count": len(baseline_df),
            "candidate_records_count": len(candidate_df),
            "numerical_drift": numerical_results,
            "categorical_drift": categorical_results,
            "missing_rate_drift": missing_rate_results,
        }


# Singleton service instance
drift_detector_service = DriftDetectorService()

