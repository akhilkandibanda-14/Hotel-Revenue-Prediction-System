from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.utils import config

logger = logging.getLogger(__name__)


def load_historical_daily_series() -> pd.DataFrame:
    """
    Loads active historical dataset and aggregates to daily hotel operational metrics.
    Sorts chronologically and indexes by Date.
    """
    active_path = config.MERGED_FOLDER / "active_dataset.csv"
    if not active_path.exists():
        # Fallback to sample dataset if merged dataset not created yet
        active_path = config.BASE_DIR / "sample_training_dataset.csv"

    if not active_path.exists():
        raise FileNotFoundError(f"Historical dataset not found at {active_path}")

    df = pd.read_csv(active_path)
    if "Date" not in df.columns:
        raise ValueError("Historical dataset is missing 'Date' column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Ensure required numerical columns
    for col in ("Total_Revenue", "Occupancy_Rate", "ADR", "Available_Rooms", "Reserved_Rooms", "Total_Costs"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate to daily metrics
    agg_rules = {
        "Total_Revenue": "sum",
        "Occupancy_Rate": "mean",
        "ADR": "mean",
    }
    if "Available_Rooms" in df.columns:
        agg_rules["Available_Rooms"] = "sum"
    if "Reserved_Rooms" in df.columns:
        agg_rules["Reserved_Rooms"] = "sum"
    if "Total_Costs" in df.columns:
        agg_rules["Total_Costs"] = "sum"

    daily = df.groupby("Date").agg(agg_rules).reset_index().sort_values("Date")
    return daily


def build_calendar_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extracts cyclical and categorical time-based features preventing data leakage.
    """
    d_series = pd.to_datetime(dates)
    doy = d_series.dt.dayofyear
    dow = d_series.dt.dayofweek

    features = pd.DataFrame({
        "dow": dow,
        "month": d_series.dt.month,
        "day": d_series.dt.day,
        "quarter": d_series.dt.quarter,
        "is_weekend": (dow >= 5).astype(int),
        "sin_doy": np.sin(2.0 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2.0 * np.pi * doy / 365.25),
        "sin_dow": np.sin(2.0 * np.pi * dow / 7.0),
        "cos_dow": np.cos(2.0 * np.pi * dow / 7.0),
    })
    return features


class ForecasterService:
    """
    Chronological forecasting service for Revenue and Occupancy.
    Uses time-based feature modeling and historical backtesting validation.
    """

    def __init__(self):
        pass

    def evaluate_backtest(
        self,
        daily_df: pd.DataFrame,
        target_col: str,
        horizon: int = 7,
    ) -> Tuple[Dict[str, float], float]:
        """
        Chronologically splits historical data to backtest a horizon-step forecast.
        Returns evaluation metrics (RMSE, MAE, MAPE) and residual standard error.
        """
        eval_horizon = min(horizon, max(5, int(len(daily_df) * 0.15)))
        if len(daily_df) <= eval_horizon + 10:
            eval_horizon = max(3, len(daily_df) // 4)

        train_data = daily_df.iloc[:-eval_horizon]
        test_data = daily_df.iloc[-eval_horizon:]

        X_train = build_calendar_features(train_data["Date"])
        y_train = train_data[target_col].values
        X_test = build_calendar_features(test_data["Date"])
        y_test = test_data[target_col].values

        model = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if target_col == "Occupancy_Rate":
            preds = np.clip(preds, 0.0, 1.0)
        else:
            preds = np.clip(preds, 0.0, None)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))

        # Safe MAPE computation avoiding division by zero
        non_zero = y_test > 1e-4
        if np.any(non_zero):
            mape = float(np.mean(np.abs((y_test[non_zero] - preds[non_zero]) / y_test[non_zero])) * 100.0)
        else:
            mape = 0.0

        residuals = y_test - preds
        residual_std = float(np.std(residuals)) if len(residuals) > 1 else rmse

        metrics = {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "backtest_window_days": int(eval_horizon),
        }
        return metrics, residual_std

    def forecast_revenue(
        self,
        horizon: int = 30,
        start_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forecasts daily revenue for next 7, 30, or 90 days with 95% confidence intervals.
        """
        if horizon not in (7, 30, 90):
            if horizon < 1:
                horizon = 7
            elif horizon > 90:
                horizon = 90

        daily_df = load_historical_daily_series()
        if len(daily_df) < 10:
            raise ValueError("Insufficient historical daily data for time-series forecasting (minimum 10 days required)")

        metrics, residual_std = self.evaluate_backtest(daily_df, "Total_Revenue", horizon=horizon)

        # Train on full history
        X_all = build_calendar_features(daily_df["Date"])
        y_all = daily_df["Total_Revenue"].values
        model = HistGradientBoostingRegressor(random_state=42, max_iter=120)
        model.fit(X_all, y_all)

        # Determine forecasting date range
        last_date = daily_df["Date"].max()
        if start_date:
            try:
                base_start = pd.to_datetime(start_date)
            except Exception:
                base_start = last_date + timedelta(days=1)
        else:
            base_start = last_date + timedelta(days=1)

        future_dates = pd.date_range(start=base_start, periods=horizon, freq="D")
        X_future = build_calendar_features(pd.Series(future_dates))
        raw_preds = model.predict(X_future)

        # Construct confidence intervals (95% z=1.96)
        z_multiplier = 1.96
        # Allow uncertainty to widen slightly for longer horizons
        horizon_uncertainty_scale = np.linspace(1.0, 1.4, horizon)

        forecast_points = []
        for i, dt in enumerate(future_dates):
            pred_val = max(0.0, float(raw_preds[i]))
            margin = z_multiplier * residual_std * horizon_uncertainty_scale[i]
            lower = max(0.0, pred_val - margin)
            upper = pred_val + margin

            forecast_points.append({
                "date": dt.strftime("%Y-%m-%d"),
                "predicted_revenue": float(round(pred_val, 2)),
                "lower_bound": float(round(lower, 2)),
                "upper_bound": float(round(upper, 2)),
                "day_of_week": dt.strftime("%A"),
                "is_weekend": bool(dt.dayofweek >= 5),
            })

        total_projected_revenue = sum(p["predicted_revenue"] for p in forecast_points)
        avg_daily_revenue = total_projected_revenue / horizon

        limitations = [
            f"Forecast trained on {len(daily_df)} daily observations spanning {daily_df['Date'].min().strftime('%Y-%m-%d')} to {daily_df['Date'].max().strftime('%Y-%m-%d')}.",
            "Model uses calendar and cyclical time patterns; exogenous shocks (e.g. unexpected travel bans, local weather anomalies) are not modeled.",
            "Confidence intervals are derived from historical out-of-sample backtest residual dispersion and widen for longer horizons.",
        ]

        return {
            "horizon_days": horizon,
            "start_date": future_dates[0].strftime("%Y-%m-%d"),
            "end_date": future_dates[-1].strftime("%Y-%m-%d"),
            "total_projected_revenue": round(total_projected_revenue, 2),
            "average_daily_revenue": round(avg_daily_revenue, 2),
            "backtest_metrics": metrics,
            "forecast": forecast_points,
            "limitations": limitations,
        }

    def forecast_occupancy(
        self,
        horizon: int = 30,
        start_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forecasts daily occupancy rate bounded within [0, 1.0] and provides seasonal analysis.
        """
        if horizon not in (7, 30, 90):
            if horizon < 1:
                horizon = 7
            elif horizon > 90:
                horizon = 90

        daily_df = load_historical_daily_series()
        if len(daily_df) < 10:
            raise ValueError("Insufficient historical daily data for occupancy forecasting")

        metrics, residual_std = self.evaluate_backtest(daily_df, "Occupancy_Rate", horizon=horizon)

        # Train on full history
        X_all = build_calendar_features(daily_df["Date"])
        y_all = daily_df["Occupancy_Rate"].values
        model = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        model.fit(X_all, y_all)

        last_date = daily_df["Date"].max()
        if start_date:
            try:
                base_start = pd.to_datetime(start_date)
            except Exception:
                base_start = last_date + timedelta(days=1)
        else:
            base_start = last_date + timedelta(days=1)

        future_dates = pd.date_range(start=base_start, periods=horizon, freq="D")
        X_future = build_calendar_features(pd.Series(future_dates))
        raw_preds = model.predict(X_future)

        z_multiplier = 1.96
        forecast_points = []
        for i, dt in enumerate(future_dates):
            pred_val = float(np.clip(raw_preds[i], 0.05, 1.0))
            margin = z_multiplier * residual_std
            lower = float(np.clip(pred_val - margin, 0.0, 1.0))
            upper = float(np.clip(pred_val + margin, 0.0, 1.0))

            forecast_points.append({
                "date": dt.strftime("%Y-%m-%d"),
                "predicted_occupancy": float(round(pred_val, 4)),
                "lower_bound": float(round(lower, 4)),
                "upper_bound": float(round(upper, 4)),
                "day_of_week": dt.strftime("%A"),
                "is_weekend": bool(dt.dayofweek >= 5),
            })

        # Historical patterns analysis
        daily_df["dow"] = daily_df["Date"].dt.day_name()
        daily_df["is_weekend"] = daily_df["Date"].dt.dayofweek >= 5
        dow_avg = daily_df.groupby("dow")["Occupancy_Rate"].mean().to_dict()
        weekend_avg = float(daily_df[daily_df["is_weekend"]]["Occupancy_Rate"].mean())
        weekday_avg = float(daily_df[~daily_df["is_weekend"]]["Occupancy_Rate"].mean())

        historical_patterns = {
            "average_historical_occupancy": round(float(daily_df["Occupancy_Rate"].mean()), 4),
            "weekday_average_occupancy": round(weekday_avg, 4),
            "weekend_average_occupancy": round(weekend_avg, 4),
            "day_of_week_distribution": {k: round(v, 4) for k, v in dow_avg.items()},
        }

        return {
            "horizon_days": horizon,
            "start_date": future_dates[0].strftime("%Y-%m-%d"),
            "end_date": future_dates[-1].strftime("%Y-%m-%d"),
            "mean_projected_occupancy": round(float(np.mean([p["predicted_occupancy"] for p in forecast_points])), 4),
            "backtest_metrics": metrics,
            "forecast": forecast_points,
            "historical_patterns": historical_patterns,
        }


# Singleton service instance
forecaster_service = ForecasterService()
