from datetime import datetime
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def preprocess_dataset(df):
    """
    Preprocess the uploaded dataset.
    Returns:
        processed dataframe
        label encoders
    """

    label_encoders = {}

    # Extract date features if Date column is present
    if "Date" in df.columns:
        try:
            date_col = pd.to_datetime(df["Date"], errors="coerce")
            df["Year"] = date_col.dt.year
            df["Day_of_Month"] = date_col.dt.day
            df["Day_of_Week"] = date_col.dt.dayofweek

            # Fill NaNs in extracted date features
            current_year = datetime.now().year
            df["Year"] = df["Year"].fillna(current_year)
            df["Day_of_Month"] = df["Day_of_Month"].fillna(1)
            df["Day_of_Week"] = df["Day_of_Week"].fillna(0)

            df = df.drop("Date", axis=1)
        except Exception:
            pass

    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_vals = df[col].mode()
            if not mode_vals.empty:
                df[col] = df[col].fillna(mode_vals[0])
            else:
                df[col] = df[col].fillna("")

    # Encode all non-numeric columns with LabelEncoder
    # Use astype(str) to handle any pandas StringDtype or object dtype uniformly
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            label_encoders[col] = encoder
            logger.debug(f"Encoded column '{col}' with classes: {list(encoder.classes_)}")

    # Final safety: coerce any remaining non-numeric columns to 0
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column '{col}' still non-numeric after encoding. Coercing to 0.")
            df[col] = 0

    return df, label_encoders
