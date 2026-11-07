import os
import joblib
import pandas as pd
import logging
from app.utils.config import MODEL_PATH, ENCODER_PATH, FEATURE_PATH

logger = logging.getLogger(__name__)


class RevenuePredictor:

    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.feature_columns = None
        self.model_mtime = 0.0

    def load_assets(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(FEATURE_PATH):
            self.model = None
            self.label_encoders = None
            self.feature_columns = None
            self.model_mtime = 0.0
            return

        try:
            current_mtime = os.path.getmtime(MODEL_PATH)
            if self.model is None or current_mtime > self.model_mtime:
                logger.info("Loading or reloading model assets...")
                model = joblib.load(MODEL_PATH)
                label_encoders = joblib.load(ENCODER_PATH)
                feature_columns = joblib.load(FEATURE_PATH)

                # Atomic assignments to prevent race conditions
                self.model = model
                self.label_encoders = label_encoders
                self.feature_columns = feature_columns
                self.model_mtime = current_mtime
                logger.info("Model assets successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load model assets: {e}")
            if self.model is None:
                raise FileNotFoundError(f"Model assets not available or corrupted: {e}")

    def preprocess_input(self, data):
        df = pd.DataFrame([data])

        # Extract date features if Date column is present
        if "Date" in df.columns:
            try:
                date_col = pd.to_datetime(df["Date"], errors="coerce")
                df["Year"] = date_col.dt.year
                df["Day_of_Month"] = date_col.dt.day
                df["Day_of_Week"] = date_col.dt.dayofweek
                df = df.drop("Date", axis=1)
            except Exception:
                pass

        # Apply label encoders for known categorical columns
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                value = str(df[col].iloc[0])
                if value in encoder.classes_:
                    df[col] = encoder.transform([value])
                else:
                    logger.warning(
                        f"Unseen category '{value}' for column '{col}'. "
                        f"Known: {list(encoder.classes_)}. Using fallback index 0."
                    )
                    df[col] = 0

        # Ensure all expected features are present
        for feature in self.feature_columns:
            if feature not in df.columns:
                df[feature] = 0

        df = df[self.feature_columns]

        # Final safety: coerce any remaining non-numeric columns to 0
        for col in df.select_dtypes(exclude=["number"]).columns:
            logger.warning(
                f"Column '{col}' still has non-numeric dtype '{df[col].dtype}' "
                f"before model predict. Coercing to 0."
            )
            df[col] = 0

        return df

    def predict(self, data):
        self.load_assets()

        if self.model is None:
            raise FileNotFoundError("ML Model is not trained yet. Please upload a dataset first.")

        df = self.preprocess_input(data)
        prediction = self.model.predict(df)
        return float(prediction[0])