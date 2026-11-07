from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# Target and leakage blacklist: these columns must NEVER enter the feature matrix X
LEAKAGE_COLUMNS: Set[str] = {
    "Total_Revenue",
    "Profit",
    "Revenue",
    "Room_Revenue",
    "Room_Revenue_All",
    "Revenue_Managed_Guests",
    "RevPAR_Managed_Guests",
    "Occupancy_Managed_Guests",
    "RevPAR_All",
    "Occupancy_All",
    "Operating_Expenses",
}

DEFAULT_CATEGORICAL_COLS = [
    "Month",
    "Booking_Channel",
    "Guest_Type",
    "Market_Segment",
    "Guest_Country",
    "Season",
]

DEFAULT_NUMERICAL_COLS = [
    "Occupancy_Rate",
    "ADR",
    "RevPAR",
    "Available_Rooms",
    "Reserved_Rooms",
    "Fixed_Costs",
    "Variable_Costs",
    "Total_Costs",
]


def sanitize_raw_dataframe(
    df: pd.DataFrame,
    is_training: bool = True,
    target_column: str = "Total_Revenue",
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Cleans raw dataframe: audits duplicates, validates target, and cleans dates.
    In hotel booking systems, identical feature rows can represent multiple valid bookings
    for the same day and room tier. Duplicates are audited and optionally removed.
    """
    cleaned = df.copy()

    if is_training:
        # Audit duplicate rows
        dup_count = int(cleaned.duplicated().sum())
        if dup_count > 0:
            logger.info(f"Audited {dup_count} identical booking records in training dataset.")
            if drop_duplicates:
                cleaned = cleaned.drop_duplicates()
                logger.info(f"Deduplicated dataset: {len(cleaned)} rows remaining.")

        # Target validation
        if target_column not in cleaned.columns:
            raise ValueError(f"Target column '{target_column}' not found in training dataset.")

        # Filter out null, inf, or negative target values
        valid_target_mask = (
            cleaned[target_column].notnull()
            & np.isfinite(cleaned[target_column])
            & (cleaned[target_column] >= 0)
        )
        invalid_count = len(cleaned) - valid_target_mask.sum()
        if invalid_count > 0:
            logger.warning(f"Dropping {invalid_count} rows with invalid/negative '{target_column}'.")
            cleaned = cleaned[valid_target_mask]

        if cleaned.empty:
            raise ValueError("No valid rows remaining after target validation.")

    return cleaned


def extract_date_features(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Extracts chronological and calendar features from Date column.
    """
    res = df.copy()
    if date_column in res.columns:
        parsed_dates = pd.to_datetime(res[date_column], errors="coerce")

        # Fill any unparseable dates with current timestamp
        now = datetime.now()
        parsed_dates = parsed_dates.fillna(pd.Timestamp(now))

        res["Year"] = parsed_dates.dt.year.astype(int)
        res["Day_of_Month"] = parsed_dates.dt.day.astype(int)
        res["Day_of_Week"] = parsed_dates.dt.dayofweek.astype(int)
        res["Is_Weekend"] = res["Day_of_Week"].isin([5, 6]).astype(int)
        res["Quarter"] = parsed_dates.dt.quarter.astype(int)
        res["Month_Num"] = parsed_dates.dt.month.astype(int)

        # If Month is not provided or null, populate it from Date
        if "Month" not in res.columns or res["Month"].isnull().any():
            month_names = parsed_dates.dt.month_name()
            res["Month"] = res.get("Month", month_names).fillna(month_names)

        # If Season is not provided or null, populate based on Month_Num
        if "Season" not in res.columns or res["Season"].isnull().any():
            season_map = {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Autumn", 10: "Autumn", 11: "Autumn",
            }
            seasons = res["Month_Num"].map(season_map).fillna("Summer")
            res["Season"] = res.get("Season", seasons).fillna(seasons)

        res = res.drop(columns=[date_column])

    return res


def handle_numerical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clamps known domain metrics to valid physical intervals.
    """
    res = df.copy()

    # Occupancy Rate: 0.0 to 1.0 (if in percentage 1-100, normalize)
    if "Occupancy_Rate" in res.columns:
        res["Occupancy_Rate"] = pd.to_numeric(res["Occupancy_Rate"], errors="coerce")
        # If user entered percentages > 1.0 up to 100.0, convert to ratio
        high_mask = res["Occupancy_Rate"] > 1.0
        res.loc[high_mask, "Occupancy_Rate"] = res.loc[high_mask, "Occupancy_Rate"] / 100.0
        res["Occupancy_Rate"] = res["Occupancy_Rate"].clip(lower=0.0, upper=1.0)

    # Room counts cannot be negative
    for col in ("Available_Rooms", "Reserved_Rooms"):
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce").clip(lower=0)

    # Costs cannot be negative
    for col in ("Fixed_Costs", "Variable_Costs", "Total_Costs"):
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce").clip(lower=0.0)

    # ADR cannot be negative
    if "ADR" in res.columns:
        res["ADR"] = pd.to_numeric(res["ADR"], errors="coerce").clip(lower=0.0)

    # RevPAR cannot be negative
    if "RevPAR" in res.columns:
        res["RevPAR"] = pd.to_numeric(res["RevPAR"], errors="coerce").clip(lower=0.0)

    return res


class HotelPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade preprocessing pipeline for Hotel Revenue Prediction.
    Ensures identical feature engineering between training and inference.
    Handles unseen categories with safe token -1.
    Prevents target leakage by design.
    """

    def __init__(
        self,
        target_column: str = "Total_Revenue",
        categorical_cols: Optional[List[str]] = None,
    ):
        self.target_column = target_column
        self.categorical_cols = categorical_cols or DEFAULT_CATEGORICAL_COLS
        self.encoder: Optional[OrdinalEncoder] = None
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.feature_columns: List[str] = []
        self.dataset_profile: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def _filter_leakage_and_target(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            c for c in df.columns
            if c in LEAKAGE_COLUMNS or c == self.target_column
        ]
        return df.drop(columns=cols_to_drop, errors="ignore")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HotelPreprocessor":
        # 1. Sanitize & extract date features
        clean_X = self._filter_leakage_and_target(X)
        feat_df = extract_date_features(clean_X)
        feat_df = handle_numerical_outliers(feat_df)

        # 2. Identify and store categorical and numerical columns
        cat_cols = [c for c in self.categorical_cols if c in feat_df.columns]
        num_cols = [c for c in feat_df.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(feat_df[c])]

        # 3. Fit numerical medians
        self.numeric_medians = {}
        for col in num_cols:
            median_val = feat_df[col].dropna().median()
            self.numeric_medians[col] = float(median_val) if not np.isnan(median_val) else 0.0

        # 4. Fit categorical modes
        self.categorical_modes = {}
        for col in cat_cols:
            modes = feat_df[col].dropna().mode()
            self.categorical_modes[col] = str(modes.iloc[0]) if not modes.empty else "Unknown"

        # 5. Fit OrdinalEncoder for categoricals with unknown_value=-1
        if cat_cols:
            filled_cats = feat_df[cat_cols].copy()
            for col in cat_cols:
                filled_cats[col] = filled_cats[col].fillna(self.categorical_modes[col]).astype(str)

            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )
            self.encoder.fit(filled_cats)

        # 6. Record exact feature column list
        self.feature_columns = list(feat_df.columns)

        # 7. Build empirical dataset profile for date-only fallback queries
        self._build_dataset_profile(X)

        self.is_fitted = True
        return self

    def _build_dataset_profile(self, raw_df: pd.DataFrame) -> None:
        """
        Builds operational profiles (overall and grouped by Season/Month)
        so date-only predictions use empirical operational medians rather than zeros.
        """
        try:
            profile = {
                "overall": {
                    "Occupancy_Rate": float(raw_df["Occupancy_Rate"].median()) if "Occupancy_Rate" in raw_df else 0.70,
                    "ADR": float(raw_df["ADR"].median()) if "ADR" in raw_df else 165.0,
                    "RevPAR": float(raw_df["RevPAR"].median()) if "RevPAR" in raw_df else 115.0,
                    "Available_Rooms": int(raw_df["Available_Rooms"].median()) if "Available_Rooms" in raw_df else 100,
                    "Reserved_Rooms": int(raw_df["Reserved_Rooms"].median()) if "Reserved_Rooms" in raw_df else 70,
                    "Fixed_Costs": float(raw_df["Fixed_Costs"].median()) if "Fixed_Costs" in raw_df else 2200.0,
                    "Variable_Costs": float(raw_df["Variable_Costs"].median()) if "Variable_Costs" in raw_df else 550.0,
                    "Total_Costs": float(raw_df["Total_Costs"].median()) if "Total_Costs" in raw_df else 2750.0,
                    "Booking_Channel": str(raw_df["Booking_Channel"].mode().iloc[0]) if "Booking_Channel" in raw_df else "Online",
                    "Guest_Type": str(raw_df["Guest_Type"].mode().iloc[0]) if "Guest_Type" in raw_df else "Transient",
                    "Market_Segment": str(raw_df["Market_Segment"].mode().iloc[0]) if "Market_Segment" in raw_df else "Leisure",
                    "Guest_Country": str(raw_df["Guest_Country"].mode().iloc[0]) if "Guest_Country" in raw_df else "US",
                    "Season": str(raw_df["Season"].mode().iloc[0]) if "Season" in raw_df else "Summer",
                },
                "by_season": {},
            }

            if "Season" in raw_df.columns:
                for season, group in raw_df.groupby("Season"):
                    profile["by_season"][str(season)] = {
                        "Occupancy_Rate": float(group["Occupancy_Rate"].median()) if "Occupancy_Rate" in group else profile["overall"]["Occupancy_Rate"],
                        "ADR": float(group["ADR"].median()) if "ADR" in group else profile["overall"]["ADR"],
                        "RevPAR": float(group["RevPAR"].median()) if "RevPAR" in group else profile["overall"]["RevPAR"],
                        "Reserved_Rooms": int(group["Reserved_Rooms"].median()) if "Reserved_Rooms" in group else profile["overall"]["Reserved_Rooms"],
                    }

            self.dataset_profile = profile
        except Exception as e:
            logger.warning(f"Could not build detailed dataset profile: {e}")
            self.dataset_profile = {"overall": {}}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("HotelPreprocessor must be fitted before calling transform.")

        clean_X = self._filter_leakage_and_target(X)
        feat_df = extract_date_features(clean_X)
        feat_df = handle_numerical_outliers(feat_df)

        # Impute missing numericals with fitted medians
        for col, median_val in self.numeric_medians.items():
            if col in feat_df.columns:
                feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(median_val)
            else:
                feat_df[col] = median_val

        # Encode categoricals with fitted OrdinalEncoder
        cat_cols = [c for c in self.categorical_cols if c in self.feature_columns]
        if self.encoder is not None and cat_cols:
            filled_cats = pd.DataFrame(index=feat_df.index)
            for col in cat_cols:
                if col in feat_df.columns:
                    mode_val = self.categorical_modes.get(col, "Unknown")
                    filled_cats[col] = feat_df[col].fillna(mode_val).astype(str)
                else:
                    filled_cats[col] = self.categorical_modes.get(col, "Unknown")

            encoded_cats = self.encoder.transform(filled_cats)
            for idx, col in enumerate(cat_cols):
                feat_df[col] = encoded_cats[:, idx]

        # Ensure all fitted feature columns exist and align in exact order
        for col in self.feature_columns:
            if col not in feat_df.columns:
                feat_df[col] = self.numeric_medians.get(col, 0.0)

        # Enforce exact column order and coerce remaining non-numerics
        ordered_df = feat_df[self.feature_columns].copy()
        for col in ordered_df.columns:
            if not pd.api.types.is_numeric_dtype(ordered_df[col]):
                ordered_df[col] = pd.to_numeric(ordered_df[col], errors="coerce").fillna(0.0)

        return ordered_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# Backward compatibility wrapper for legacy callers
def preprocess_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backward-compatible preprocessing function.
    Returns:
        transformed dataframe, label_encoders dict
    """
    preprocessor = HotelPreprocessor()
    transformed_X = preprocessor.fit_transform(df)

    # Reconstruct target if present in raw df
    if "Total_Revenue" in df.columns:
        transformed_X["Total_Revenue"] = df["Total_Revenue"].values

    # Build legacy label_encoders dict for compatibility with tests
    legacy_encoders = {}
    if preprocessor.encoder is not None:
        cat_cols = [c for c in preprocessor.categorical_cols if c in preprocessor.feature_columns]
        for idx, col in enumerate(cat_cols):
            classes = list(preprocessor.encoder.categories_[idx])
            # Lightweight mock for classes_ attribute
            class MockEncoder:
                def __init__(self, c_list):
                    self.classes_ = np.array(c_list)
                def transform(self, vals):
                    mapping = {c: i for i, c in enumerate(self.classes_)}
                    return [mapping.get(str(v), 0) for v in vals]
            legacy_encoders[col] = MockEncoder(classes)

    return transformed_X, legacy_encoders
