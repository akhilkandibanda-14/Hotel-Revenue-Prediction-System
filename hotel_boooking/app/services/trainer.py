import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from app.services.preprocessor import preprocess_dataset


from app.utils.config import (
    MODEL_FOLDER,
    MODEL_PATH,
    ENCODER_PATH,
    FEATURE_PATH,
)


def train_model(dataset_path):

    # =============================
    # LOAD DATASET
    # =============================

    df = pd.read_csv(dataset_path)

    # =============================
    # PREPROCESS
    # =============================

    df, label_encoders = preprocess_dataset(df)

    # =============================
    # TARGET CHECK
    # =============================

    if "Total_Revenue" not in df.columns:

        raise Exception(
            "Total_Revenue column not found."
        )

    # =============================
    # FEATURES
    # =============================

    X = df.drop(
        "Total_Revenue",
        axis=1
    )

    y = df["Total_Revenue"]

    # Save feature names
    feature_columns = list(X.columns)

    # =============================
    # TRAIN TEST SPLIT
    # =============================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42

    )

    # =============================
    # MODEL
    # =============================

    model = RandomForestRegressor(

        n_estimators=200,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    # =============================
    # PREDICTION
    # =============================

    y_pred = model.predict(

        X_test

    )

    # =============================
    # METRICS
    # =============================

    mae = mean_absolute_error(

        y_test,

        y_pred

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            y_pred

        )

    )

    r2 = r2_score(

        y_test,

        y_pred

    )

    # =============================
    # SAVE MODEL
    # =============================

    os.makedirs(

        MODEL_FOLDER,

        exist_ok=True

    )

    joblib.dump(

        model,

        MODEL_PATH

    )

    joblib.dump(

        label_encoders,

        ENCODER_PATH

    )

    joblib.dump(

        feature_columns,

        FEATURE_PATH

    )

    # =============================
    # RETURN METRICS
    # =============================

    return {

        "status": "Training Completed",

        "MAE": round(mae, 2),

        "RMSE": round(rmse, 2),

        "R2": round(r2, 4),

        "Training Rows": len(df)

    }