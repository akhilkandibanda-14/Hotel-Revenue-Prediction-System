# Hotel Revenue Prediction Service (Advanced ML Architecture)

A production-grade, continuous machine learning service that ingests reservation feeds, trains multi-model ensembles chronologically upon dataset upload, dynamically selects the best algorithm using validation RMSE, and serves revenue predictions via high-throughput REST APIs.

---

## 1. Setup & Environment

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

---

## 2. Advanced ML Architecture & Pipeline

### A. Preprocessing & Leakage Prevention
- **Strict Target Leakage Shield**: Systematically strips all target synonyms and components (`Profit`, `Revenue`, `Room_Revenue`, `Room_Revenue_All`, `Operating_Expenses`) from features prior to training.
- **Calendar & Temporal Decomposition**: Derives `Year`, `Month_Num`, `Day_of_Month`, `Day_of_Week`, `Is_Weekend`, and `Quarter` from `Date`.
- **Domain Outlier Clamping**: Clamps `Occupancy_Rate` to `[0.0, 1.0]` (normalizing percentage inputs) and enforces non-negative physical bounds on rooms and costs.
- **Safe Unseen Category Handling**: Uses `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` so unseen categories cleanly map to `-1.0` without silent distortion.
- **Strict Split Isolation**: Preprocessing statistics (imputation medians, encoding dictionaries, and seasonal profiles) are fitted strictly on the training partition.

### B. Multi-Model Candidates & Automated Model Selection
On every dataset ingestion, the service trains and compares 5 candidate algorithms:
1. **Random Forest Regressor** (`RandomForestRegressor`) — *Baseline*
2. **Decision Tree Regressor** (`DecisionTreeRegressor`)
3. **Gradient Boosting Regressor** (`GradientBoostingRegressor`)
4. **Extra Trees Regressor** (`ExtraTreesRegressor`)
5. **HistGradientBoosting Regressor** (`HistGradientBoostingRegressor`)

**Selection Policy**: Models are evaluated on the chronological Validation split (15%). The algorithm with the lowest Validation RMSE (with R² and MAE as tie-breakers) is automatically selected and evaluated once on the untouched Test split (15%).

### C. Versioned Artifact Management & Zero-Downtime Rollback
- Promoted models are stored under `models/versions/model_v{version}.joblib`.
- Current production model is deployed atomically to `models/current_model.joblib` and `models/hotel_revenue_model.pkl`.
- Full audit trails are written to `models/model_metadata.json` and `models/model_comparison.json`.
- Any previous version can be instantly restored via `POST /model/rollback/{version}` without restarting the server.

---

## 3. API Usage & Examples

### A. Predict Revenue (Full Operational Features)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/" -Method Post -ContentType "application/json" -Body '{
    "Date": "2026-06-30",
    "Occupancy_Rate": 0.85,
    "ADR": 175.0,
    "Reserved_Rooms": 85,
    "Available_Rooms": 100,
    "Total_Costs": 2600.0,
    "Season": "Summer",
    "Booking_Channel": "Online"
}'
```

### B. Predict Revenue (Date-Only with Empirical Profile Fallback)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/" -Method Post -ContentType "application/json" -Body '{
    "Date": "2026-06-30"
}'
```
*Applies learned empirical seasonal/monthly medians from training data rather than synthetic zeros.*

### C. Strict Mode (Rejects Missing Features with 422)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/" -Method Post -ContentType "application/json" -Body '{
    "Date": "2026-06-30",
    "strict": true
}'
```

### D. Inspect Active Model Metadata
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/info" -Method Get
```

### E. Inspect Latest Multi-Model Comparison Table
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/comparison" -Method Get
```

### F. Roll Back to an Earlier Model Version
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/rollback/v001" -Method Post
```

### G. Post to Reservation Feed
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/reservation/" -Method Post -ContentType "application/json" -Body '{
    "Date": "2026-06-30",
    "Month": "June",
    "Booking_Channel": "Online",
    "Guest_Type": "Transient",
    "Market_Segment": "Leisure",
    "Guest_Country": "US",
    "Season": "Summer",
    "Occupancy_Rate": 0.85,
    "ADR": 150.0,
    "RevPAR": 127.5,
    "Available_Rooms": 100,
    "Reserved_Rooms": 85,
    "Fixed_Costs": 2000.0,
    "Variable_Costs": 500.0,
    "Total_Costs": 2500.0
}'
```

### H. Upload Dataset for Automated Background Retraining
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/dataset/upload" -Method Post -Form @{
    file = [System.IO.File]::Open("D:\path\to\new_data.csv", [System.IO.FileMode]::Open)
}
```

---

## 4. Running the Tests

Execute all unit and pipeline verification tests:
```powershell
python -m unittest discover tests
```
