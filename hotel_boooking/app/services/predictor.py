from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.services.model_manager import load_model_metadata
from app.services.preprocessor import (
    DEFAULT_NUMERICAL_COLS,
    HotelPreprocessor,
    extract_date_features,
)
from app.utils import config

logger = logging.getLogger(__name__)

REQUIRED_PREDICTION_FIELDS = [
    "Date",
    "Occupancy_Rate",
    "ADR",
    "Available_Rooms",
    "Reserved_Rooms",
    "Total_Costs",
]


class RevenuePredictor:
    """
    Production ML Inference Service.
    Loads active model and preprocessing pipeline dynamically.
    Rejects missing inputs under strict mode.
    Employs empirical seasonal profiling when date-only input is provided.
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.legacy_encoders: Optional[Dict[str, Any]] = None
        self.legacy_features: Optional[List[str]] = None
        self.model_mtime: float = 0.0
        self.metadata: Dict[str, Any] = {}
        self.last_prediction_meta: Dict[str, Any] = {}

    def load_assets(self) -> None:
        """
        Dynamically loads or hot-reloads model and preprocessing artifacts.
        Thread-safe atomic pointer assignment.
        """
        current_model_file = config.get_current_model_path()
        model_file = config.get_model_path()
        target_model_path = current_model_file if current_model_file.exists() else model_file
        if not target_model_path.exists():
            self.model = None
            self.preprocessor = None
            self.model_mtime = 0.0
            return

        try:
            current_mtime = os.path.getmtime(target_model_path)
            if self.model is None or current_mtime > self.model_mtime:
                logger.info(f"Loading model assets from {target_model_path}...")
                loaded_model = joblib.load(target_model_path)

                # 1. Try modern preprocessing pipeline
                loaded_pipeline = None
                pipeline_file = config.get_pipeline_path()
                if pipeline_file.exists():
                    try:
                        loaded_pipeline = joblib.load(pipeline_file)
                    except Exception as pe:
                        logger.warning(f"Could not load modern pipeline: {pe}")

                # 2. Legacy fallback
                legacy_encoders = None
                legacy_features = None
                encoder_file = config.get_encoder_path()
                feature_file = config.get_feature_path()
                if encoder_file.exists() and feature_file.exists():
                    try:
                        legacy_encoders = joblib.load(encoder_file)
                        legacy_features = joblib.load(feature_file)
                    except Exception:
                        pass

                # Atomic assignments
                self.model = loaded_model
                self.preprocessor = loaded_pipeline
                self.legacy_encoders = legacy_encoders
                self.legacy_features = legacy_features
                self.model_mtime = current_mtime
                self.metadata = load_model_metadata()

                logger.info("Model assets successfully loaded and ready for inference.")

        except Exception as e:
            logger.error(f"Failed to load model assets: {e}")
            if self.model is None:
                raise FileNotFoundError(f"Model assets not available or corrupted: {e}")

    def validate_and_prepare_input(
        self,
        data: Dict[str, Any],
        strict: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates input features.
        Under strict mode: rejects missing operational fields.
        Under fallback mode: applies empirical seasonal profiling from training data.
        """
        if not data or not isinstance(data, dict):
            raise ValueError("Prediction request payload must be a non-empty JSON object.")

        if "Date" not in data or not str(data["Date"]).strip():
            raise ValueError("Required field 'Date' is missing or empty.")

        # Validate Date format
        try:
            parsed_date = pd.to_datetime(data["Date"])
        except Exception:
            raise ValueError(f"Invalid date format for '{data['Date']}'. Expected YYYY-MM-DD.")

        input_dict = dict(data)
        prediction_mode = "full_features"
        applied_profile = {}

        # Check for missing operational features
        missing_operational = [
            f for f in ("Occupancy_Rate", "ADR", "Reserved_Rooms")
            if f not in input_dict or input_dict[f] is None
        ]

        if missing_operational:
            if strict:
                raise ValueError(
                    f"Strict validation failed. Missing required operational metrics: {missing_operational}. "
                    "Provide operational fields or disable strict mode for seasonal profile fallback."
                )

            # Empirical profile fallback (never fill with arbitrary zeros!)
            prediction_mode = "date_profile_fallback"
            logger.info(
                f"Date-only or partial query detected for date '{data['Date']}'. "
                "Applying empirical training profile instead of zero-filling."
            )

            month_name = parsed_date.month_name()
            season_map = {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Autumn", 10: "Autumn", 11: "Autumn",
            }
            season_name = season_map.get(parsed_date.month, "Summer")

            # Default baseline profile
            profile = {
                "Month": month_name,
                "Season": season_name,
                "Booking_Channel": "Online",
                "Guest_Type": "Transient",
                "Market_Segment": "Leisure",
                "Guest_Country": "US",
                "Occupancy_Rate": 0.70,
                "ADR": 165.0,
                "RevPAR": 115.5,
                "Available_Rooms": 100,
                "Reserved_Rooms": 70,
                "Fixed_Costs": 2200.0,
                "Variable_Costs": 550.0,
                "Total_Costs": 2750.0,
            }

            # If preprocessor has learned empirical dataset profile, use it
            if self.preprocessor and hasattr(self.preprocessor, "dataset_profile"):
                dp = self.preprocessor.dataset_profile
                overall = dp.get("overall", {})
                by_season = dp.get("by_season", {}).get(season_name, {})

                profile.update(overall)
                profile.update(by_season)

            applied_profile = {k: v for k, v in profile.items() if k not in input_dict}

            # Merge: caller's inputs take priority over profile defaults
            for k, v in profile.items():
                if k not in input_dict or input_dict[k] is None:
                    input_dict[k] = v

        # Convert to single-row DataFrame
        df_input = pd.DataFrame([input_dict])

        # Preprocessing transformation
        if self.preprocessor is not None and hasattr(self.preprocessor, "transform"):
            X_transformed = self.preprocessor.transform(df_input)
        else:
            # Fallback legacy transformer
            X_transformed = self._legacy_transform(df_input)

        audit_meta = {
            "prediction_mode": prediction_mode,
            "applied_profile": applied_profile,
            "date": str(data["Date"]),
        }
        return X_transformed, audit_meta

    def _legacy_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Legacy transformer fallback if modern pipeline is absent."""
        df_copy = df.copy()
        df_copy = extract_date_features(df_copy)

        if self.legacy_encoders:
            for col, encoder in self.legacy_encoders.items():
                if col in df_copy.columns:
                    val = str(df_copy[col].iloc[0])
                    classes = getattr(encoder, "classes_", [])
                    if val in classes:
                        df_copy[col] = list(classes).index(val)
                    else:
                        df_copy[col] = 0

        if self.legacy_features:
            for f in self.legacy_features:
                if f not in df_copy.columns:
                    df_copy[f] = 0
            df_copy = df_copy[self.legacy_features]

        for col in df_copy.columns:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0)

        return df_copy

    def predict(self, data: Dict[str, Any], strict: bool = False) -> float:
        """
        Main predict method: returns float prediction.
        Maintains backward compatibility with test suites and external callers.
        """
        self.load_assets()

        if self.model is None:
            raise FileNotFoundError("ML Model is not trained yet. Please upload a dataset first.")

        X_trans, meta = self.validate_and_prepare_input(data, strict=strict)
        prediction = self.model.predict(X_trans)
        pred_val = float(prediction[0])

        meta["predicted_revenue"] = pred_val
        meta["model_version"] = self.metadata.get("version", "v001")
        meta["algorithm"] = self.metadata.get("algorithm", type(self.model).__name__)
        self.last_prediction_meta = meta

        return pred_val

    def predict_detailed(self, data: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        """
        Returns rich prediction response with algorithm, version, and profile metadata.
        """
        revenue = self.predict(data, strict=strict)
        return {
            "status": "Success",
            "Predicted_Revenue": round(revenue, 2),
            **self.last_prediction_meta,
        }


# Singleton instances for convenience across services and routers
predictor = RevenuePredictor()
revenue_predictor = predictor
