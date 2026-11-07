import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
import joblib

# Load data and preprocessor
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

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

model_dir = Path("d:/Hotel_Revenue_Project/hotel_boooking/models/cancellation")
base_model = joblib.load(model_dir / "cancellation_model.joblib")
preprocessor = joblib.load(model_dir / "cancellation_preprocessor.joblib")

X_val_trans = preprocessor.transform(X_val)
X_test_trans = preprocessor.transform(X_test)

def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1] if i < n_bins-1 else y_prob <= bin_edges[i+1])
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
    return ece

print("=== CALIBRATION BENCHMARK EVALUATION (ON UNTOUCHED TEST SET) ===")

# 1. Uncalibrated Base Model
probs_base = base_model.predict_proba(X_test_trans)[:, 1]
brier_base = brier_score_loss(y_test, probs_base)
roc_base = roc_auc_score(y_test, probs_base)
pr_base = average_precision_score(y_test, probs_base)
ece_base = compute_ece(y_test.values, probs_base)

print(f"1. Uncalibrated Random Forest:")
print(f"   Brier Score: {brier_base:.4f} | ECE: {ece_base:.4f} | ROC-AUC: {roc_base:.4f} | PR-AUC: {pr_base:.4f}")

# 2. Sigmoid (Platt) Calibration fitted on Validation Split
cal_sig = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')
cal_sig.fit(X_val_trans, y_val)
probs_sig = cal_sig.predict_proba(X_test_trans)[:, 1]
brier_sig = brier_score_loss(y_test, probs_sig)
roc_sig = roc_auc_score(y_test, probs_sig)
pr_sig = average_precision_score(y_test, probs_sig)
ece_sig = compute_ece(y_test.values, probs_sig)

print(f"2. Sigmoid (Platt) Calibrated (fit on Val):")
print(f"   Brier Score: {brier_sig:.4f} | ECE: {ece_sig:.4f} | ROC-AUC: {roc_sig:.4f} | PR-AUC: {pr_sig:.4f}")

# 3. Isotonic Calibration fitted on Validation Split
cal_iso = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')
cal_iso.fit(X_val_trans, y_val)
probs_iso = cal_iso.predict_proba(X_test_trans)[:, 1]
brier_iso = brier_score_loss(y_test, probs_iso)
roc_iso = roc_auc_score(y_test, probs_iso)
pr_iso = average_precision_score(y_test, probs_iso)
ece_iso = compute_ece(y_test.values, probs_iso)

print(f"3. Isotonic Calibrated (fit on Val):")
print(f"   Brier Score: {brier_iso:.4f} | ECE: {ece_iso:.4f} | ROC-AUC: {roc_iso:.4f} | PR-AUC: {pr_iso:.4f}")

