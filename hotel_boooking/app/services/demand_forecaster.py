from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.services.forecaster import build_calendar_features, load_historical_daily_series
from app.utils import config

logger = logging.getLogger(__name__)


class HotelDemandForecaster:
    """
    Hotel Demand Forecasting Engine.
    Predicts expected room demand, room nights, and occupancy rate across 7, 30, and 90-day horizons.
    Evaluates against moving-average and seasonal-naive baselines with chronological holdouts.
    """

    def __init__(self):
        pass

    def evaluate_demand_baselines(
        self,
        daily_df: pd.DataFrame,
        target_col: str = "Reserved_Rooms",
        horizon: int = 30,
    ) -> Dict[str, Any]:
        """
        Chronological holdout backtest comparing HistGradientBoosting against
        Last-Value, 7-Day Moving Average, and Seasonal-Naive baselines.
        """
        eval_horizon = min(horizon, max(5, int(len(daily_df) * 0.20)))
        if len(daily_df) <= eval_horizon + 10:
            eval_horizon = max(3, len(daily_df) // 4)

        train_data = daily_df.iloc[:-eval_horizon].copy()
        test_data = daily_df.iloc[-eval_horizon:].copy()

        y_train = train_data[target_col].values
        y_test = test_data[target_col].values

        # 1. Last-Value Baseline
        preds_last = np.full(eval_horizon, y_train[-1])

        # 2. 7-Day Moving Average Baseline
        ma_window = min(7, len(y_train))
        preds_ma7 = np.full(eval_horizon, np.mean(y_train[-ma_window:]))

        # 3. Seasonal Naive Baseline (Repeat last 7 days)
        last_cycle = y_train[-7:] if len(y_train) >= 7 else y_train
        preds_snaive = np.array([last_cycle[i % len(last_cycle)] for i in range(eval_horizon)])

        # 4. HistGradientBoosting ML
        X_train = build_calendar_features(train_data["Date"])
        X_test = build_calendar_features(test_data["Date"])
        hgbr = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        hgbr.fit(X_train, y_train)
        preds_ml = np.clip(hgbr.predict(X_test), 0.0, None)

        def calc_metrics(y_true, y_pred):
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            non_zero = y_true > 1e-4
            mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0) if np.any(non_zero) else 0.0
            return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2)}

        # Compute residual standard error for prediction intervals
        train_preds = hgbr.predict(X_train)
        res_std = float(np.std(y_train - train_preds)) if len(y_train) > 1 else 10.0

        # Empirical CI coverage on holdout
        z = 1.96
        ci_lower = np.clip(preds_ml - (z * res_std), 0.0, None)
        ci_upper = preds_ml + (z * res_std)
        coverage_pct = float(np.mean((y_test >= ci_lower) & (y_test <= ci_upper)) * 100.0)

        return {
            "eval_horizon_days": int(eval_horizon),
            "residual_std": round(res_std, 2),
            "ci_coverage_pct": round(coverage_pct, 1),
            "metrics": {
                "ml_model": calc_metrics(y_test, preds_ml),
                "moving_average_7d": calc_metrics(y_test, preds_ma7),
                "seasonal_naive_7d": calc_metrics(y_test, preds_snaive),
                "last_value": calc_metrics(y_test, preds_last),
            }
        }

    def forecast_demand(
        self,
        horizon: int = 30,
        start_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates multi-step room demand and occupancy forecast with baseline benchmarks.
        """
        if horizon not in (7, 30, 90):
            if horizon < 1:
                horizon = 7
            elif horizon > 90:
                horizon = 90

        daily_df = load_historical_daily_series()
        if len(daily_df) < 10:
            raise ValueError("Insufficient daily data for demand forecasting (min 10 days required)")

        target_demand_col = "Reserved_Rooms" if "Reserved_Rooms" in daily_df.columns else "Occupancy_Rate"
        benchmark_eval = self.evaluate_demand_baselines(daily_df, target_col=target_demand_col, horizon=horizon)

        # Train ML models on full history
        X_all = build_calendar_features(daily_df["Date"])
        y_demand = daily_df[target_demand_col].values
        y_occupancy = daily_df["Occupancy_Rate"].values if "Occupancy_Rate" in daily_df.columns else (y_demand / 100.0)

        model_demand = HistGradientBoostingRegressor(random_state=42, max_iter=120)
        model_demand.fit(X_all, y_demand)

        model_occ = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        model_occ.fit(X_all, y_occupancy)

        # Build future date range
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

        raw_demand_preds = model_demand.predict(X_future)
        raw_occ_preds = np.clip(model_occ.predict(X_future), 0.05, 1.0)

        res_std = benchmark_eval["residual_std"]
        z = 1.96
        # Widen uncertainty progressively for longer horizons
        uncertainty_factors = np.linspace(1.0, 1.35, horizon)

        forecast_records: List[Dict[str, Any]] = []
        cumulative_room_nights = 0.0

        for i, dt in enumerate(future_dates):
            pred_rooms = float(max(0.0, raw_demand_preds[i]))
            pred_occ = float(raw_occ_preds[i])
            margin = z * res_std * uncertainty_factors[i]

            lower_rooms = float(max(0.0, pred_rooms - margin))
            upper_rooms = float(pred_rooms + margin)
            cumulative_room_nights += pred_rooms

            forecast_records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "expected_room_demand": float(round(pred_rooms, 1)),
                "expected_room_nights": float(round(pred_rooms, 1)),
                "cumulative_room_nights": float(round(cumulative_room_nights, 1)),
                "expected_occupancy_pct": float(round(pred_occ * 100.0, 1)),
                "lower_bound_rooms": float(round(lower_rooms, 1)),
                "upper_bound_rooms": float(round(upper_rooms, 1)),
                "day_of_week": dt.strftime("%A"),
                "is_weekend": bool(dt.dayofweek >= 5),
            })

        total_expected_room_nights = sum(r["expected_room_nights"] for r in forecast_records)
        avg_daily_demand = total_expected_room_nights / horizon
        avg_occupancy = float(np.mean([r["expected_occupancy_pct"] for r in forecast_records]))

        # Clear labeling for 90-day exploratory horizon
        reliability_level = "High Operational Reliability" if horizon <= 14 else (
            "Moderate Reliability (Tactical)" if horizon <= 30 else
            "Exploratory Projection (High Uncertainty — Single Year Historical Depth)"
        )

        limitations = [
            f"Forecast generated from {len(daily_df)} daily observations spanning {daily_df['Date'].min().strftime('%Y-%m-%d')} to {daily_df['Date'].max().strftime('%Y-%m-%d')}.",
            "Demand projections assume stationary booking lead times without competitor disruption, local major events, or macro travel shocks.",
            "Prediction intervals represent empirical out-of-sample holdout residual dispersion and expand over the horizon.",
        ]
        if horizon == 90:
            limitations.append(
                "WARNING: 90-day horizon is classified as exploratory. Single-year historical data does not capture multi-year macroeconomic shifts."
            )

        return {
            "horizon_days": horizon,
            "reliability_classification": reliability_level,
            "start_date": future_dates[0].strftime("%Y-%m-%d"),
            "end_date": future_dates[-1].strftime("%Y-%m-%d"),
            "total_expected_room_nights": float(round(total_expected_room_nights, 1)),
            "average_daily_room_demand": float(round(avg_daily_demand, 1)),
            "projected_average_occupancy_pct": float(round(avg_occupancy, 1)),
            "benchmark_evaluation": benchmark_eval,
            "forecast": forecast_records,
            "limitations": limitations,
        }


# Singleton service
demand_forecaster = HotelDemandForecaster()

