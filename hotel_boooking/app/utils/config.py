import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Datasets config
BASE_DATASET_FOLDER = BASE_DIR / "datasets"
UPLOADED_FOLDER = BASE_DATASET_FOLDER / "uploaded"
MERGED_FOLDER = BASE_DATASET_FOLDER / "merged"
TRAINING_STATE_FILE = BASE_DATASET_FOLDER / "training_state.json"

# Models config
MODEL_FOLDER = BASE_DIR / "models"
MODEL_NAME = "hotel_revenue_model.pkl"
ENCODER_NAME = "label_encoders.pkl"
FEATURE_NAME = "feature_columns.pkl"

MODEL_PATH = MODEL_FOLDER / MODEL_NAME
ENCODER_PATH = MODEL_FOLDER / ENCODER_NAME
FEATURE_PATH = MODEL_FOLDER / FEATURE_NAME

# Reservations config
RESERVATION_FOLDER = BASE_DIR / "reservations"
RESERVATION_FILE = "reservation_log.csv"
RESERVATION_PATH = RESERVATION_FOLDER / RESERVATION_FILE

# Scheduler config
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
