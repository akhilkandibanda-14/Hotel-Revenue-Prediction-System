import os
from pathlib import Path

# Suppress harmless Windows Loky warning for physical core lookup
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Datasets config
BASE_DATASET_FOLDER = BASE_DIR / "datasets"
UPLOADED_FOLDER = BASE_DATASET_FOLDER / "uploaded"
MERGED_FOLDER = BASE_DATASET_FOLDER / "merged"
TRAINING_STATE_FILE = BASE_DATASET_FOLDER / "training_state.json"

# Models config
MODEL_FOLDER = BASE_DIR / "models"
VERSIONS_FOLDER = MODEL_FOLDER / "versions"
MODEL_NAME = "hotel_revenue_model.pkl"
ENCODER_NAME = "label_encoders.pkl"
FEATURE_NAME = "feature_columns.pkl"

MODEL_PATH = MODEL_FOLDER / MODEL_NAME
CURRENT_MODEL_PATH = MODEL_FOLDER / "current_model.joblib"
ENCODER_PATH = MODEL_FOLDER / ENCODER_NAME
FEATURE_PATH = MODEL_FOLDER / FEATURE_NAME
PIPELINE_PATH = MODEL_FOLDER / "preprocessing_pipeline.joblib"
METADATA_PATH = MODEL_FOLDER / "model_metadata.json"
COMPARISON_PATH = MODEL_FOLDER / "model_comparison.json"


def get_model_folder() -> Path:
    return MODEL_FOLDER


def get_versions_folder() -> Path:
    return MODEL_FOLDER / "versions"


def get_model_path() -> Path:
    return MODEL_FOLDER / MODEL_NAME


def get_current_model_path() -> Path:
    return MODEL_FOLDER / "current_model.joblib"


def get_encoder_path() -> Path:
    return MODEL_FOLDER / ENCODER_NAME


def get_feature_path() -> Path:
    return MODEL_FOLDER / FEATURE_NAME


def get_pipeline_path() -> Path:
    return MODEL_FOLDER / "preprocessing_pipeline.joblib"


def get_metadata_path() -> Path:
    return MODEL_FOLDER / "model_metadata.json"


def get_comparison_path() -> Path:
    return MODEL_FOLDER / "model_comparison.json"


# Reservations config
RESERVATION_FOLDER = BASE_DIR / "reservations"
RESERVATION_FILE = "reservation_log.csv"
RESERVATION_PATH = RESERVATION_FOLDER / RESERVATION_FILE

# Scheduler config
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))

