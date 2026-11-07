"""
Forecasting Validation and Baseline Comparison Audit Script.
Compares HistGradientBoosting against Last-Value, Moving-Average, and Seasonal-Naive.
Evaluates 7-day, 30-day, and 90-day holdouts separately.
Checks 95% Confidence Interval coverage.
"""

import sys
sys.path.insert(0, r"D:\Hotel_Revenue_Project\hotel_boooking")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.services.forecaster import load_historical_daily_series, build_calendar_features

def run_forecasting_audit():
    daily_df = load_historical_daily_series()
    print(f"Total Daily Historical Rows: {len(daily_df)}")
    print(f"Date Range: {daily_df['Date'].min().strftime('%Y-%m-%d')} to {daily_df['Date'].max().strftime('%Y-%m-%d')}")

    results = []

    for horizon in [7, 30, 90]:
        print(f"\n{'='*20} EVALUATING HORIZON: {horizon} DAYS {'='*20}")
        if len(daily_df) <= horizon + 14:
            print(f"Warning: Dataset length ({len(daily_df)}) is close to horizon ({horizon}).")

        train_data = daily_df.iloc[:-horizon].copy()
        test_data = daily_df.iloc[-horizon:].copy()

        y_train = train_data["Total_Revenue"].values
        y_test = test_data["Total_Revenue"].values

        # 1. Baseline: Last-Value (Persistence)
        # For multi-step, last known value in train
        last_val = y_train[-1]
        preds_last_val = np.full(horizon, last_val)

        # 2. Baseline: 7-day Moving Average of train
        ma_7 = np.mean(y_train[-7:])
        preds_ma7 = np.full(horizon, ma_7)

        # 3. Baseline: Seasonal Naive (Day of week from last 7 days repeated)
        last_week = y_train[-7:]
        preds_snaive = np.array([last_week[i % 7] for i in range(horizon)])

        # 4. HistGradientBoostingRegressor (ML Model)
        X_train = build_calendar_features(train_data["Date"])
        X_test = build_calendar_features(test_data["Date"])
        hgbr = HistGradientBoostingRegressor(random_state=42, max_iter=120)
        hgbr.fit(X_train, y_train)
        preds_hgbr = np.clip(hgbr.predict(X_test), 0.0, None)

        # Empirical residual std from training fit / one-step errors
        train_preds = hgbr.predict(X_train)
        residual_std = np.std(y_train - train_preds)

        # 95% Confidence Interval for ML
        z = 1.96
        horizon_uncertainty_scale = np.linspace(1.0, 1.4, horizon)
        ci_lower = np.clip(preds_hgbr - (z * residual_std * horizon_uncertainty_scale), 0.0, None)
        ci_upper = preds_hgbr + (z * residual_std * horizon_uncertainty_scale)

        coverage = np.mean((y_test >= ci_lower) & (y_test <= ci_upper)) * 100.0

        models = {
            "Last-Value Baseline": preds_last_val,
            "7-Day Moving Avg Baseline": preds_ma7,
            "Seasonal Naive Baseline": preds_snaive,
            "HistGradientBoosting ML": preds_hgbr
        }

        for name, preds in models.items():
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            non_zero = y_test > 1e-4
            mape = np.mean(np.abs((y_test[non_zero] - preds[non_zero]) / y_test[non_zero])) * 100.0

            row = {
                "Horizon": f"{horizon}d",
                "Model": name,
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "MAPE (%)": round(mape, 2),
                "CI 95% Coverage (%)": f"{coverage:.1f}%" if "ML" in name else "N/A"
            }
            results.append(row)
            print(f"{name:30s} | RMSE: {rmse:10.2f} | MAE: {mae:10.2f} | MAPE: {mape:6.2f}%")

        print(f"ML 95% Confidence Interval Coverage on Actuals: {coverage:.1f}%")

    res_df = pd.DataFrame(results)
    print("\n--- FINAL COMPARISON TABLE ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_forecasting_audit()

