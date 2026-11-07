import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

model_dir = Path("d:/Hotel_Revenue_Project/hotel_boooking/models/cancellation")
base_model = joblib.load(model_dir / "cancellation_model.joblib")
preprocessor = joblib.load(model_dir / "cancellation_preprocessor.joblib")

data_path = Path("d:/Hotel_Revenue_Project/hotel_boooking/datasets/public_research/hotel_bookings_demand_antonio.csv")
df = pd.read_csv(data_path).drop_duplicates().copy()

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'].map(month_map).astype(str).str.zfill(2) + '-' +
    df['arrival_date_day_of_month'].astype(str).str.zfill(2),
    errors='coerce'
)
df = df.sort_values('arrival_date').reset_index(drop=True)

df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['children'] = df['children'].fillna(0)
df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['has_prior_cancellations'] = (df['previous_cancellations'] > 0).astype(int)
df['arrival_month_num'] = df['arrival_date_month'].map(month_map).fillna(1).astype(int)

num_features = [
    'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'total_stay_nights',
    'adults', 'children', 'babies', 'total_guests', 'is_repeated_guest',
    'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests',
    'has_prior_cancellations', 'arrival_month_num'
]
cat_features = [
    'hotel', 'market_segment', 'distribution_channel', 'deposit_type',
    'customer_type', 'reserved_room_type', 'meal'
]
for col in cat_features:
    df[col] = df[col].fillna('Unknown').astype(str)

X = df[num_features + cat_features]
y = df['is_canceled'].astype(int)

# Use test holdout for evaluation
n = len(df)
val_end = int(n * 0.85)
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
X_test_trans = preprocessor.transform(X_test)

probs = base_model.predict_proba(X_test_trans)[:, 1]

def evaluate_threshold_grid(y_true, y_probs, cost_fp=50.0, cost_fn=150.0, replacement_rate=0.20):
    effective_cost_fn = cost_fn * (1.0 - replacement_rate)
    thresholds = np.arange(0.10, 0.95, 0.05)
    rows = []
    for th in thresholds:
        th = round(th, 2)
        preds = (y_probs >= th).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        tot_cost = (fp * cost_fp) + (fn * effective_cost_fn)
        avg_cost = tot_cost / len(y_true)
        rows.append({
            'threshold': th,
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total_cost': round(tot_cost, 2),
            'avg_cost_per_booking': round(avg_cost, 2)
        })
    return pd.DataFrame(rows)

grid_df = evaluate_threshold_grid(y_test.values, probs, cost_fp=45.0, cost_fn=160.0, replacement_rate=0.25)
print("=== THRESHOLD GRID EVALUATION (Sample) ===")
print(grid_df[['threshold', 'precision', 'recall', 'f1', 'fp', 'fn', 'total_cost', 'avg_cost_per_booking']])

opt_row = grid_df.loc[grid_df['total_cost'].idxmin()]
print(f"\nCost-Optimal Threshold: {opt_row['threshold']} | Min Total Cost: ${opt_row['total_cost']:,.2f} (vs Default 0.50: ${grid_df.loc[grid_df['threshold']==0.50, 'total_cost'].values[0]:,.2f})")

