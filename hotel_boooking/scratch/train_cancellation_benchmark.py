import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss,
    accuracy_score
)
from sklearn.calibration import calibration_curve

# 1. Load Data
data_path = Path("d:/Hotel_Revenue_Project/hotel_boooking/datasets/public_research/hotel_bookings_demand_antonio.csv")
df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} rows from {data_path}")

# Deduplication note:
# Real hotel transaction callsets have identical booking attributes for group rooms.
# We will drop full exact duplicates to test clean generalizability.
df_clean = df.drop_duplicates().copy()
print(f"After removing identical row duplicates: {len(df_clean):,} rows")

# 2. Chronological Ordering
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df_clean['arrival_date'] = pd.to_datetime(
    df_clean['arrival_date_year'].astype(str) + '-' +
    df_clean['arrival_date_month'].map(month_map).astype(str).str.zfill(2) + '-' +
    df_clean['arrival_date_day_of_month'].astype(str).str.zfill(2),
    errors='coerce'
)
df_clean = df_clean.sort_values('arrival_date').reset_index(drop=True)

# 3. Feature Engineering (Strictly Pre-Arrival Features)
df_clean['total_stay_nights'] = df_clean['stays_in_weekend_nights'] + df_clean['stays_in_week_nights']
df_clean['total_guests'] = df_clean['adults'] + df_clean['children'].fillna(0) + df_clean['babies']
df_clean['is_family'] = ((df_clean['children'].fillna(0) > 0) | (df_clean['babies'] > 0)).astype(int)
df_clean['has_prior_cancellations'] = (df_clean['previous_cancellations'] > 0).astype(int)
df_clean['arrival_month_num'] = df_clean['arrival_date_month'].map(month_map)

# Feature definitions
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

# Impute children nulls with 0
df_clean['children'] = df_clean['children'].fillna(0)

# Fill unknown categories
for col in cat_features:
    df_clean[col] = df_clean[col].fillna('Unknown').astype(str)

X = df_clean[num_features + cat_features]
y = df_clean['is_canceled'].astype(int)

# 4. Chronological Train / Val / Test Split
# Train: 70%, Val: 15%, Test: 15%
n = len(df_clean)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

print(f"\nChronological Split:")
print(f"  Train: {len(X_train):,} ({df_clean['arrival_date'].iloc[0].strftime('%Y-%m-%d')} to {df_clean['arrival_date'].iloc[train_end-1].strftime('%Y-%m-%d')}) | Cancel Rate: {y_train.mean()*100:.1f}%")
print(f"  Val:   {len(X_val):,} ({df_clean['arrival_date'].iloc[train_end].strftime('%Y-%m-%d')} to {df_clean['arrival_date'].iloc[val_end-1].strftime('%Y-%m-%d')}) | Cancel Rate: {y_val.mean()*100:.1f}%")
print(f"  Test:  {len(X_test):,} ({df_clean['arrival_date'].iloc[val_end].strftime('%Y-%m-%d')} to {df_clean['arrival_date'].iloc[-1].strftime('%Y-%m-%d')}) | Cancel Rate: {y_test.mean()*100:.1f}%")

# 5. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_val)
X_test_trans = preprocessor.transform(X_test)

# Baseline: Majority Class Classifier
majority_pred = np.zeros_like(y_val)
majority_prob = np.full_like(y_val, y_train.mean(), dtype=float)

# 6. Candidate Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, n_jobs=-1, class_weight='balanced_subsample', random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, n_jobs=-1, class_weight='balanced', random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150, max_depth=10, min_samples_leaf=20, random_state=42)
}

results = {}
for name, model in models.items():
    t0 = time.time()
    print(f"\nTraining {name}...")
    model.fit(X_train_trans, y_train)
    train_time = round(time.time() - t0, 2)
    
    # Predict on Validation
    val_probs = model.predict_proba(X_val_trans)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    
    prec = precision_score(y_val, val_preds, zero_division=0)
    rec = recall_score(y_val, val_preds, zero_division=0)
    f1 = f1_score(y_val, val_preds, zero_division=0)
    roc_auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)
    brier = brier_score_loss(y_val, val_probs)
    cm = confusion_matrix(y_val, val_preds).tolist()
    
    results[name] = {
        'train_time_sec': train_time,
        'val_precision': round(prec, 4),
        'val_recall': round(rec, 4),
        'val_f1': round(f1, 4),
        'val_roc_auc': round(roc_auc, 4),
        'val_pr_auc': round(pr_auc, 4),
        'val_brier_score': round(brier, 4),
        'val_confusion_matrix': cm
    }
    print(f"  {name} -> Val ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | Brier: {brier:.4f}")

# Evaluate Best Model on Test Set
best_model_name = max(results, key=lambda k: results[k]['val_roc_auc'])
best_model = models[best_model_name]
print(f"\n=======================================================")
print(f"BEST MODEL SELECTED: {best_model_name} (by Validation ROC-AUC)")
print(f"=======================================================")

test_probs = best_model.predict_proba(X_test_trans)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

test_metrics = {
    'test_precision': round(precision_score(y_test, test_preds, zero_division=0), 4),
    'test_recall': round(recall_score(y_test, test_preds, zero_division=0), 4),
    'test_f1': round(f1_score(y_test, test_preds, zero_division=0), 4),
    'test_roc_auc': round(roc_auc_score(y_test, test_probs), 4),
    'test_pr_auc': round(average_precision_score(y_test, test_probs), 4),
    'test_brier_score': round(brier_score_loss(y_test, test_probs), 4),
    'test_confusion_matrix': confusion_matrix(y_test, test_preds).tolist()
}
print(f"Final Test Evaluation on Untouched Holdout Split:")
for k, v in test_metrics.items():
    print(f"  {k:22}: {v}")

