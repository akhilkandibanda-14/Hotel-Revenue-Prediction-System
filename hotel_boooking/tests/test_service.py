import os
import shutil
import tempfile
import unittest
import pandas as pd
from pathlib import Path

# Override config paths before importing app services
from app.utils import config

# Create a temporary directory for tests
temp_dir = tempfile.TemporaryDirectory()
temp_path = Path(temp_dir.name)

# Override all config paths to use the temp directory
config.BASE_DATASET_FOLDER = temp_path / "datasets"
config.UPLOADED_FOLDER = config.BASE_DATASET_FOLDER / "uploaded"
config.MERGED_FOLDER = config.BASE_DATASET_FOLDER / "merged"
config.TRAINING_STATE_FILE = config.BASE_DATASET_FOLDER / "training_state.json"

config.MODEL_FOLDER = temp_path / "models"
config.MODEL_PATH = config.MODEL_FOLDER / config.MODEL_NAME
config.ENCODER_PATH = config.MODEL_FOLDER / config.ENCODER_NAME
config.FEATURE_PATH = config.MODEL_FOLDER / config.FEATURE_NAME

config.RESERVATION_FOLDER = temp_path / "reservations"
config.RESERVATION_PATH = config.RESERVATION_FOLDER / config.RESERVATION_FILE

# Import services now that paths are overridden
from app.services.dataset_manager import (
    ensure_dataset_directories,
    list_pending_datasets,
    merge_datasets,
    load_training_state,
)
from app.services.trainer import train_model
from app.services.predictor import RevenuePredictor
from app.services.reservation_service import save_reservation


class TestHotelRevenueService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ensure_dataset_directories()
        config.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
        config.RESERVATION_FOLDER.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        temp_dir.cleanup()

    def test_end_to_end_flow(self):
        # 1. Test Bootstrapping Behavior (no model yet)
        predictor = RevenuePredictor()
        
        # Checking predict directly fails
        reservation_data = {
            "Date": "2026-06-30",
            "Month": "June",
            "Booking_Channel": "Online",
            "Guest_Type": "Transient",
            "Market_Segment": "Leisure",
            "Guest_Country": "US",
            "Season": "Summer",
            "Occupancy_Rate": 0.85,
            "ADR": 150.0,
            "RevPAR": 127.5,
            "Available_Rooms": 100,
            "Reserved_Rooms": 85,
            "Fixed_Costs": 2000.0,
            "Variable_Costs": 500.0,
            "Total_Costs": 2500.0
        }
        
        with self.assertRaises(FileNotFoundError):
            predictor.predict(reservation_data)

        # Checking save_reservation logs successfully without predicting
        pred, err = save_reservation(reservation_data.copy())
        self.assertIsNone(pred)
        self.assertIsNone(err)
        
        # Verify log file exists and has 1 reservation
        self.assertTrue(config.RESERVATION_PATH.exists())
        df_log = pd.read_csv(config.RESERVATION_PATH)
        self.assertEqual(len(df_log), 1)
        self.assertEqual(df_log.iloc[0]["Month"], "June")
        # In the log, Predicted_Revenue should be empty/NaN
        self.assertTrue(pd.isna(df_log.iloc[0]["Predicted_Revenue"]))

        # 2. Create mock datasets for training
        mock_data_1 = pd.DataFrame([{
            "Date": "2026-06-01",
            "Month": "June",
            "Booking_Channel": "Online",
            "Guest_Type": "Transient",
            "Market_Segment": "Leisure",
            "Guest_Country": "US",
            "Season": "Summer",
            "Occupancy_Rate": 0.80,
            "ADR": 140.0,
            "RevPAR": 112.0,
            "Available_Rooms": 100,
            "Reserved_Rooms": 80,
            "Fixed_Costs": 2000.0,
            "Variable_Costs": 450.0,
            "Total_Costs": 2450.0,
            "Total_Revenue": 11200.0
        }] * 10)  # Make 10 rows
        
        mock_data_2 = pd.DataFrame([{
            "Date": "2026-06-15",
            "Month": "June",
            "Booking_Channel": "Direct",
            "Guest_Type": "Transient",
            "Market_Segment": "Corporate",
            "Guest_Country": "UK",
            "Season": "Summer",
            "Occupancy_Rate": 0.90,
            "ADR": 160.0,
            "RevPAR": 144.0,
            "Available_Rooms": 100,
            "Reserved_Rooms": 90,
            "Fixed_Costs": 2000.0,
            "Variable_Costs": 600.0,
            "Total_Costs": 2600.0,
            "Total_Revenue": 14400.0
        }] * 10)  # Make 10 rows

        # Save mock datasets to UPLOADED_FOLDER
        file_path_1 = config.UPLOADED_FOLDER / "20260630000000000000_dataset1.csv"
        file_path_2 = config.UPLOADED_FOLDER / "20260630000001000000_dataset2.csv"
        mock_data_1.to_csv(file_path_1, index=False)
        mock_data_2.to_csv(file_path_2, index=False)

        # Verify list_pending_datasets detects them
        pending = list_pending_datasets()
        self.assertEqual(len(pending), 2)

        # 3. Test merging and training for dataset 1
        merged_path = merge_datasets(file_path_1)
        self.assertTrue(merged_path.exists())
        self.assertEqual(len(pd.read_csv(merged_path)), 10)

        # Train model
        result = train_model(merged_path)
        self.assertEqual(result["status"], "Training Completed")
        self.assertEqual(result["Training Rows"], 10)

        # Check model files are generated
        self.assertTrue(config.MODEL_PATH.exists())
        self.assertTrue(config.ENCODER_PATH.exists())
        self.assertTrue(config.FEATURE_PATH.exists())

        # 4. Verify predictions now work (loading model dynamically)
        pred = predictor.predict(reservation_data)
        self.assertIsInstance(pred, float)

        date_only_pred = predictor.predict({"Date": "2026-06-30"})
        self.assertIsInstance(date_only_pred, float)

        # Verify save_reservation still just logs the reservation
        pred2, err2 = save_reservation(reservation_data.copy())
        self.assertIsNone(err2)
        self.assertIsNone(pred2)

        # 5. Test continuous training (dataset 2)
        # Verify dataset 1 is marked processed so it isn't listed as pending
        from app.services.dataset_manager import mark_dataset_processed
        mark_dataset_processed(file_path_1, result)
        
        pending_after = list_pending_datasets()
        self.assertEqual(len(pending_after), 1)
        self.assertEqual(pending_after[0].name, file_path_2.name)

        # Merge dataset 2 (should merge dataset 1 and dataset 2)
        merged_path_2 = merge_datasets(file_path_2)
        merged_df_2 = pd.read_csv(merged_path_2)
        # Should contain 20 rows (10 from dataset1 + 10 from dataset2)
        self.assertEqual(len(merged_df_2), 20)

        # Train on merged dataset
        result_2 = train_model(merged_path_2)
        self.assertEqual(result_2["Training Rows"], 20)

        # Verify model file modification time is newer
        # Reloading should happen automatically on next predict
        pred3 = predictor.predict(reservation_data)
        self.assertIsInstance(pred3, float)


if __name__ == "__main__":
    unittest.main()
