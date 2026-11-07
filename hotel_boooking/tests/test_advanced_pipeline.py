from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.utils import config

# Create isolated temporary directory for test artifacts
temp_dir = tempfile.TemporaryDirectory()
temp_path = Path(temp_dir.name)

# Point config to isolated temp directory to protect production assets
config.BASE_DATASET_FOLDER = temp_path / "datasets"
config.UPLOADED_FOLDER = config.BASE_DATASET_FOLDER / "uploaded"
config.MERGED_FOLDER = config.BASE_DATASET_FOLDER / "merged"
config.TRAINING_STATE_FILE = config.BASE_DATASET_FOLDER / "training_state.json"

config.MODEL_FOLDER = temp_path / "models"
config.VERSIONS_FOLDER = config.MODEL_FOLDER / "versions"
config.MODEL_PATH = config.MODEL_FOLDER / config.MODEL_NAME
config.CURRENT_MODEL_PATH = config.MODEL_FOLDER / "current_model.joblib"
config.ENCODER_PATH = config.MODEL_FOLDER / config.ENCODER_NAME
config.FEATURE_PATH = config.MODEL_FOLDER / config.FEATURE_NAME
config.PIPELINE_PATH = config.MODEL_FOLDER / "preprocessing_pipeline.joblib"
config.METADATA_PATH = config.MODEL_FOLDER / "model_metadata.json"
config.COMPARISON_PATH = config.MODEL_FOLDER / "model_comparison.json"

config.RESERVATION_FOLDER = temp_path / "reservations"
config.RESERVATION_PATH = config.RESERVATION_FOLDER / config.RESERVATION_FILE

from app.main import app
from app.services.model_manager import (
    load_model_comparison,
    load_model_metadata,
    rollback_to_version,
)
from app.services.predictor import RevenuePredictor
from app.services.preprocessor import (
    HotelPreprocessor,
    extract_date_features,
    handle_numerical_outliers,
    sanitize_raw_dataframe,
)
from app.services.trainer import compute_metrics, train_model


class TestAdvancedMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = temp_dir
        cls.temp_path = temp_path

        config.MODEL_FOLDER = cls.temp_path / "models"
        config.BASE_DATASET_FOLDER = cls.temp_path / "datasets"
        config.UPLOADED_FOLDER = config.BASE_DATASET_FOLDER / "uploaded"
        config.MERGED_FOLDER = config.BASE_DATASET_FOLDER / "merged"
        config.TRAINING_STATE_FILE = config.BASE_DATASET_FOLDER / "training_state.json"
        config.RESERVATION_FOLDER = cls.temp_path / "reservations"
        config.RESERVATION_PATH = config.RESERVATION_FOLDER / config.RESERVATION_FILE

        config.get_model_folder().mkdir(parents=True, exist_ok=True)
        config.get_versions_folder().mkdir(parents=True, exist_ok=True)
        config.UPLOADED_FOLDER.mkdir(parents=True, exist_ok=True)
        config.MERGED_FOLDER.mkdir(parents=True, exist_ok=True)
        config.RESERVATION_FOLDER.mkdir(parents=True, exist_ok=True)

        # Generate a realistic synthetic dataset
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        np.random.seed(42)
        n = len(dates)

        channels = ["Online", "Direct", "OTA", "Corporate"]
        guest_types = ["Transient", "Group", "Contract", "Transient-Party"]
        segments = ["Leisure", "Corporate", "Wholesale", "Government"]
        countries = ["US", "UK", "DE", "FR", "IN"]
        seasons = ["Winter", "Spring", "Summer", "Autumn"]

        cls.synthetic_df = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Month": dates.month_name(),
            "Booking_Channel": np.random.choice(channels, size=n),
            "Guest_Type": np.random.choice(guest_types, size=n),
            "Market_Segment": np.random.choice(segments, size=n),
            "Guest_Country": np.random.choice(countries, size=n),
            "Season": np.random.choice(seasons, size=n),
            "Occupancy_Rate": np.random.uniform(0.50, 0.95, size=n).round(2),
            "ADR": np.random.uniform(120.0, 220.0, size=n).round(2),
            "Available_Rooms": np.random.randint(80, 120, size=n),
            "Fixed_Costs": np.random.uniform(1800.0, 2500.0, size=n).round(2),
            "Variable_Costs": np.random.uniform(300.0, 700.0, size=n).round(2),
        })

        cls.synthetic_df["Reserved_Rooms"] = (
            cls.synthetic_df["Available_Rooms"] * cls.synthetic_df["Occupancy_Rate"]
        ).astype(int)
        cls.synthetic_df["RevPAR"] = (
            cls.synthetic_df["ADR"] * cls.synthetic_df["Occupancy_Rate"]
        ).round(2)
        cls.synthetic_df["Total_Costs"] = (
            cls.synthetic_df["Fixed_Costs"] + cls.synthetic_df["Variable_Costs"]
        ).round(2)
        cls.synthetic_df["Total_Revenue"] = (
            cls.synthetic_df["Reserved_Rooms"] * cls.synthetic_df["ADR"]
        ).round(2)

        cls.synthetic_csv_path = config.UPLOADED_FOLDER / "synthetic_training_data.csv"
        cls.synthetic_df.to_csv(cls.synthetic_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        temp_dir.cleanup()

    def test_01_preprocessing_and_feature_engineering(self):
        """Verifies calendar feature extraction, outlier handling, and column order."""
        sample = self.synthetic_df.copy()
        sample.loc[0, "Occupancy_Rate"] = 95.0  # Percentage input: should normalize to 0.95
        sample.loc[1, "Reserved_Rooms"] = -5   # Negative rooms: should clamp to 0

        preprocessor = HotelPreprocessor()
        X_trans = preprocessor.fit_transform(sample)

        self.assertIn("Year", preprocessor.feature_columns)
        self.assertIn("Day_of_Month", preprocessor.feature_columns)
        self.assertIn("Day_of_Week", preprocessor.feature_columns)
        self.assertIn("Is_Weekend", preprocessor.feature_columns)
        self.assertIn("Quarter", preprocessor.feature_columns)
        self.assertIn("Month_Num", preprocessor.feature_columns)

        # Verify normalized occupancy rate
        self.assertLessEqual(X_trans["Occupancy_Rate"].max(), 1.0)
        self.assertGreaterEqual(X_trans["Occupancy_Rate"].min(), 0.0)

        # Verify clamped reserved rooms
        self.assertGreaterEqual(X_trans["Reserved_Rooms"].min(), 0)

        # Ensure target is not in feature columns
        self.assertNotIn("Total_Revenue", X_trans.columns)

    def test_02_target_leakage_prevention(self):
        """Ensures Profit, Revenue, Room_Revenue, etc. are strictly purged from features."""
        leaky_df = self.synthetic_df.copy()
        leaky_df["Profit"] = leaky_df["Total_Revenue"] - leaky_df["Total_Costs"]
        leaky_df["Revenue"] = leaky_df["Total_Revenue"]
        leaky_df["Room_Revenue"] = leaky_df["Total_Revenue"]
        leaky_df["Room_Revenue_All"] = leaky_df["Total_Revenue"]

        preprocessor = HotelPreprocessor()
        X_trans = preprocessor.fit_transform(leaky_df)

        forbidden_cols = ["Profit", "Revenue", "Room_Revenue", "Room_Revenue_All", "Total_Revenue"]
        for col in forbidden_cols:
            self.assertNotIn(col, X_trans.columns)

    def test_03_unseen_categorical_handling(self):
        """Verifies unseen categories at inference encode safely as -1 without error."""
        preprocessor = HotelPreprocessor()
        preprocessor.fit(self.synthetic_df)

        unseen_record = pd.DataFrame([{
            "Date": "2026-06-15",
            "Month": "UnknownMonth",
            "Booking_Channel": "Teleportation",
            "Guest_Type": "Alien",
            "Market_Segment": "Intergalactic",
            "Guest_Country": "Mars",
            "Season": "SolarStorm",
            "Occupancy_Rate": 0.80,
            "ADR": 150.0,
            "RevPAR": 120.0,
            "Available_Rooms": 100,
            "Reserved_Rooms": 80,
            "Fixed_Costs": 2000.0,
            "Variable_Costs": 500.0,
            "Total_Costs": 2500.0,
        }])

        X_trans = preprocessor.transform(unseen_record)
        # All unseen categoricals should map to -1.0 token
        cat_cols = ["Month", "Booking_Channel", "Guest_Type", "Market_Segment", "Guest_Country", "Season"]
        for c in cat_cols:
            if c in X_trans.columns:
                self.assertEqual(X_trans[c].iloc[0], -1.0)

    def test_04_multi_model_training_and_auto_selection(self):
        """Verifies all 5 models train, comparison metrics compute, and winner is selected."""
        result = train_model(str(self.synthetic_csv_path))

        self.assertEqual(result["status"], "Training Completed")
        self.assertIn("Selected Model", result)
        self.assertIn("Model Version", result)
        self.assertGreater(result["Training Rows"], 0)
        self.assertGreater(result["R2"], 0.70)

        # Check all 5 models are present in comparison results
        comparison = result["Model Comparison"]
        model_names = [m["model"] for m in comparison]
        self.assertIn("Random Forest", model_names)
        self.assertIn("Decision Tree", model_names)
        self.assertIn("Gradient Boosting", model_names)
        self.assertIn("Extra Trees", model_names)
        self.assertIn("HistGradientBoosting", model_names)

        # Verify artifacts were created in temp dir
        self.assertTrue(config.get_current_model_path().exists())
        self.assertTrue(config.get_pipeline_path().exists())
        self.assertTrue(config.get_metadata_path().exists())
        self.assertTrue(config.get_comparison_path().exists())

    def test_05_versioning_and_rollback(self):
        """Verifies model versioning, metadata storage, and rollback capability."""
        meta1 = load_model_metadata()
        v1 = meta1.get("version", "v001")

        # Train again to create version 2
        result2 = train_model(str(self.synthetic_csv_path))
        meta2 = load_model_metadata()
        v2 = meta2.get("version")
        self.assertNotEqual(v1, v2)

        # Execute rollback to v1
        rolled_meta = rollback_to_version(v1)
        self.assertEqual(rolled_meta["version"], v1)

        active_meta = load_model_metadata()
        self.assertEqual(active_meta["version"], v1)

    def test_06_prediction_pipeline_modes(self):
        """Tests full features prediction, strict rejection, and date-only profile fallback."""
        predictor = RevenuePredictor()

        # 1. Full operational features prediction
        full_payload = {
            "Date": "2026-06-30",
            "Month": "June",
            "Booking_Channel": "Online",
            "Guest_Type": "Transient",
            "Market_Segment": "Leisure",
            "Guest_Country": "US",
            "Season": "Summer",
            "Occupancy_Rate": 0.85,
            "ADR": 160.0,
            "RevPAR": 136.0,
            "Available_Rooms": 100,
            "Reserved_Rooms": 85,
            "Fixed_Costs": 2000.0,
            "Variable_Costs": 500.0,
            "Total_Costs": 2500.0,
        }
        pred_full = predictor.predict(full_payload, strict=True)
        self.assertIsInstance(pred_full, float)
        self.assertGreater(pred_full, 0.0)

        # 2. Strict rejection of missing operational features
        date_only_payload = {"Date": "2026-06-30"}
        with self.assertRaises(ValueError):
            predictor.predict(date_only_payload, strict=True)

        # 3. Date-only prediction with empirical seasonal profile fallback
        pred_date_only = predictor.predict(date_only_payload, strict=False)
        self.assertIsInstance(pred_date_only, float)
        self.assertGreater(pred_date_only, 0.0)
        self.assertEqual(predictor.last_prediction_meta["prediction_mode"], "date_profile_fallback")

    def test_07_fastapi_endpoints(self):
        """Tests FastAPI REST endpoints via TestClient."""
        client = TestClient(app)

        # Test health check
        health_resp = client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.json()["status"], "ok")

        # Test model info endpoint
        info_resp = client.get("/model/info")
        self.assertEqual(info_resp.status_code, 200)
        self.assertEqual(info_resp.json()["status"], "Success")

        # Test model comparison endpoint
        comp_resp = client.get("/model/comparison")
        self.assertEqual(comp_resp.status_code, 200)
        self.assertEqual(comp_resp.json()["status"], "Success")

        # Test model versions endpoint
        ver_resp = client.get("/model/versions")
        self.assertEqual(ver_resp.status_code, 200)
        self.assertEqual(ver_resp.json()["status"], "Success")
        self.assertIn("available_versions", ver_resp.json())

        # Test dataset training status endpoint
        status_resp = client.get("/dataset/status")
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(status_resp.json()["status"], "Success")
        self.assertIn("pending_count", status_resp.json())

        # Test prediction with full features
        pred_payload = {
            "Date": "2026-06-30",
            "Occupancy_Rate": 0.85,
            "ADR": 170.0,
            "RevPAR": 144.5,
            "Available_Rooms": 100,
            "Reserved_Rooms": 85,
            "Total_Costs": 2500.0,
            "Season": "Summer"
        }
        pred_resp = client.post("/predict/", json=pred_payload)
        self.assertEqual(pred_resp.status_code, 200)
        data = pred_resp.json()
        self.assertEqual(data["status"], "Success")
        self.assertGreater(data["Predicted_Revenue"], 0.0)

        # Test date-only prediction (backward-compatible)
        date_resp = client.post("/predict/", json={"Date": "2026-06-30"})
        self.assertEqual(date_resp.status_code, 200)
        date_data = date_resp.json()
        self.assertEqual(date_data["status"], "Success")
        self.assertEqual(date_data["Prediction_Mode"], "date_profile_fallback")

        # Test strict rejection
        strict_resp = client.post("/predict/", json={"Date": "2026-06-30", "strict": True})
        self.assertEqual(strict_resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
