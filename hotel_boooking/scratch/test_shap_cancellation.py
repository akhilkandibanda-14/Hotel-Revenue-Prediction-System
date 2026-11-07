import numpy as np
import pandas as pd
from pathlib import Path
import joblib

try:
    import shap
    print("SHAP is installed! Version:", shap.__version__)
except ImportError:
    print("SHAP is NOT installed.")

model_dir = Path("d:/Hotel_Revenue_Project/hotel_boooking/models/cancellation")
base_model = joblib.load(model_dir / "cancellation_model.joblib")
preprocessor = joblib.load(model_dir / "cancellation_preprocessor.joblib")

# Test transforming a sample input
sample_dict = {
    'lead_time': [120],
    'stays_in_weekend_nights': [1],
    'stays_in_week_nights': [3],
    'total_stay_nights': [4],
    'adults': [2],
    'children': [0.0],
    'babies': [0],
    'total_guests': [2.0],
    'is_repeated_guest': [0],
    'previous_cancellations': [1],
    'previous_bookings_not_canceled': [0],
    'booking_changes': [0],
    'days_in_waiting_list': [0],
    'adr': [160.0],
    'required_car_parking_spaces': [0],
    'total_of_special_requests': [0],
    'has_prior_cancellations': [1],
    'arrival_month_num': [8],
    'hotel': ['City Hotel'],
    'market_segment': ['Online TA'],
    'distribution_channel': ['TA/TO'],
    'deposit_type': ['No Deposit'],
    'customer_type': ['Transient'],
    'reserved_room_type': ['A'],
    'meal': ['BB']
}
sample_df = pd.DataFrame(sample_dict)
X_trans = preprocessor.transform(sample_df)

# Feature names from preprocessor
feature_names = preprocessor.get_feature_names_out()
print("Number of transformed features:", len(feature_names))

# TreeExplainer test
try:
    explainer = shap.TreeExplainer(base_model)
    shap_vals = explainer.shap_values(X_trans)
    print("SHAP values computed successfully!")
    print("SHAP values shape/type:", type(shap_vals), np.array(shap_vals).shape)
    # Check additive consistency
    if isinstance(shap_vals, list):
        # binary classification: shap_vals[1] is for class 1 (canceled)
        sv1 = shap_vals[1][0]
        base_v = explainer.expected_value[1]
    elif len(np.array(shap_vals).shape) == 3:
        sv1 = shap_vals[0, :, 1]
        base_v = explainer.expected_value[1]
    else:
        sv1 = shap_vals[0]
        base_v = explainer.expected_value
    
    pred_prob = base_model.predict_proba(X_trans)[0, 1]
    print(f"Base Value (E[f(X)]): {base_v:.4f}")
    print(f"Sum of SHAP values: {np.sum(sv1):.4f}")
    print(f"Base + Sum: {base_v + np.sum(sv1):.4f}")
    print(f"Actual Model Probability: {pred_prob:.4f}")
    
    # Sort top contributions
    sorted_idx = np.argsort(np.abs(sv1))[::-1]
    print("\nTop 5 contributing features:")
    for idx in sorted_idx[:5]:
        print(f"  {feature_names[idx]}: {sv1[idx]:+.4f}")
except Exception as e:
    print("Error during SHAP test:", e)

