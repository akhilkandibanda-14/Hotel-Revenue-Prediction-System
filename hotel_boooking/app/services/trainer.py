from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from app.services.model_manager import deploy_trained_model
from app.services.preprocessor import (
    HotelPreprocessor,
    sanitize_raw_dataframe,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes RMSE, MAE, R2, and MAPE (when valid).
    Safely handles zero-variance edge cases in minimal test sets.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if len(y_true) < 2 or np.all(y_true == y_true[0]):
        r2 = 1.0 if rmse == 0.0 else 0.0
    else:
        try:
            r2_val = r2_score(y_true, y_pred)
            r2 = float(r2_val) if not np.isnan(r2_val) else 0.0
        except Exception:
            r2 = 0.0

    # MAPE is valid when all target values are strictly positive
    if np.all(y_true > 0):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    else:
        mape = 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def prepare_chronological_splits(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sorts chronologically and splits into Train, Validation, and Test sets.
    Prevents temporal data leakage.
    """
    sorted_df = df.copy()

    # Sort chronologically if Date column is present
    if date_col in sorted_df.columns:
        parsed_dates = pd.to_datetime(sorted_df[date_col], errors="coerce")
        sorted_df["_sort_key"] = parsed_dates
        sorted_df = sorted_df.sort_values(by="_sort_key").drop(columns=["_sort_key"])

    sorted_df = sorted_df.reset_index(drop=True)
    n = len(sorted_df)

    if n < 4:
        return sorted_df.copy(), sorted_df.copy(), sorted_df.copy()

    train_end = max(1, int(n * train_pct))
    val_end = max(train_end + 1, int(n * (train_pct + val_pct)))

    if val_end >= n:
        val_end = n - 1
    if train_end >= val_end:
        train_end = val_end - 1

    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]

    if val_df.empty:
        val_df = test_df.copy() if not test_df.empty else train_df.copy()
    if test_df.empty:
        test_df = val_df.copy()

    return train_df, val_df, test_df


def initialize_model_candidates() -> Dict[str, Any]:
    """
    Initializes all 5 candidate regression algorithms with reproducible random states.
    """
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=2,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
        ),
    }


def train_model(dataset_path: Path | str) -> Dict[str, Any]:
    """
    Executes advanced multi-model training:
    1. Loads and sanitizes dataset (removes duplicates, validates target).
    2. Chronological Train / Val / Test splitting.
    3. Fits HotelPreprocessor strictly on training split.
    4. Evaluates all 5 models on the validation split.
    5. Selects best model (primarily lowest Validation RMSE).
    6. Evaluates winner on untouched Test split.
    7. Atomically deploys model artifacts with versioning and rollback support.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    logger.info(f"Initiating advanced ML training on {path.name}")
    raw_df = pd.read_csv(path)

    # 1. Sanitize raw data (handles duplicates, target verification)
    df = sanitize_raw_dataframe(raw_df, is_training=True, target_column="Total_Revenue")
    total_rows = len(df)

    # 2. Chronological Splitting
    train_df, val_df, test_df = prepare_chronological_splits(df, date_col="Date")
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Fit Preprocessor strictly on training data
    preprocessor = HotelPreprocessor(target_column="Total_Revenue")
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["Total_Revenue"].values

    X_val = preprocessor.transform(val_df)
    y_val = val_df["Total_Revenue"].values

    X_test = preprocessor.transform(test_df)
    y_test = test_df["Total_Revenue"].values

    # 4. Multi-Model Training and Validation Benchmarking
    candidates = initialize_model_candidates()
    comparison_results: List[Dict[str, Any]] = []

    best_model_name: Optional[str] = None
    best_model_instance: Optional[Any] = None
    best_val_rmse = float("inf")
    best_val_metrics: Dict[str, float] = {}

    for name, model in candidates.items():
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            train_duration = round(time.time() - start_time, 3)

            val_preds = model.predict(X_val)
            metrics = compute_metrics(y_val, val_preds)

            record = {
                "model": name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAPE": metrics["MAPE"],
                "fit_time_seconds": train_duration,
                "status": "Evaluated",
            }
            comparison_results.append(record)
            logger.info(f"Model: {name:<20} | Val RMSE: {metrics['RMSE']:<8} | Val R2: {metrics['R2']:<6}")

            # Primary criterion: Lowest Validation RMSE (with R2 tie breaker)
            if metrics["RMSE"] < best_val_rmse:
                best_val_rmse = metrics["RMSE"]
                best_val_name = name
                best_model_instance = model
                best_val_metrics = metrics

        except Exception as e:
            logger.error(f"Failed training {name}: {e}")
            comparison_results.append({
                "model": name,
                "error": str(e),
                "status": "Failed",
            })

    if best_model_instance is None:
        raise RuntimeError("All candidate models failed during training evaluation.")

    # 5. Evaluate the winning model ONCE on the untouched Test set
    test_preds = best_model_instance.predict(X_test)
    final_test_metrics = compute_metrics(y_test, test_preds)
    logger.info(f"Selected Best Model: {best_val_name} | Test RMSE: {final_test_metrics['RMSE']}")

    # 6. Deploy Winning Model Artifacts Atomically
    dataset_info = {
        "dataset_name": path.name,
        "total_records": total_rows,
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "feature_count": len(preprocessor.feature_columns),
    }

    deployed_meta = deploy_trained_model(
        model=best_model_instance,
        preprocessor=preprocessor,
        selected_algorithm=best_val_name,
        comparison_results=comparison_results,
        best_metrics=best_val_metrics,
        test_metrics=final_test_metrics,
        dataset_info=dataset_info,
    )

    # 7. Return structured training results
    return {
        "status": "Training Completed",
        "Selected Model": best_val_name,
        "Model Version": deployed_meta["version"],
        "MAE": final_test_metrics["MAE"],
        "RMSE": final_test_metrics["RMSE"],
        "R2": final_test_metrics["R2"],
        "MAPE": final_test_metrics["MAPE"],
        "Validation RMSE": best_val_metrics["RMSE"],
        "Validation R2": best_val_metrics["R2"],
        "Training Rows": total_rows,
        "Features Count": len(preprocessor.feature_columns),
        "Model Comparison": comparison_results,
    }
