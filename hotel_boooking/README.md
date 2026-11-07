# Hotel Revenue Prediction Service

This is a background ML service that continuously trains models when new datasets are uploaded, logs reservation feeds via REST APIs, and serves date-based revenue predictions through a separate prediction endpoint.

## 1. Setup & Environment

Ensure you have Python 3.10+ installed and the virtual environment is set up.

- **Activate virtual environment:**
  ```powershell
  .venv\Scripts\activate
  ```

---
/docs to server so that we can examine the beackground serivce....
## 2. Execution

### Running the Automated Tests
To verify the entire service lifecycle (bootstrapping, merging, training, and hot-reloading) in an isolated, temporary environment, execute:
```powershell
.venv\Scripts\python -m unittest tests/test_service.py
```

### Running the Background Application
To start the FastAPI server locally:
```powershell
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Once started, the background training scheduler thread will run in the background and poll for new datasets every 10 seconds.

---

## 3. API Usage Examples

You can interact with the service using PowerShell or any HTTP client (like Postman or curl).

### A. Post to the Reservation Feed
Send new reservation logs to the server. The reservation endpoint only records the request and does not run a prediction.

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

### B. Predict for a Specific Date
Ask the prediction endpoint for revenue on a specific date.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/" -Method Post -ContentType "application/json" -Body '{
  "Date": "2026-06-30"
}'
```

### C. Upload a Dataset for Continuous Training
Upload a CSV dataset. The background scheduler will automatically pick it up, merge it with previous datasets, retrain the Random Forest model, and write the new model to disk. The prediction engine will automatically hot-reload the updated assets on subsequent reservation feed requests.

*Note: Replace `C:\path\to\your\dataset.csv` with the absolute path of your training file.*
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/dataset/upload" -Method Post -Form @{
    file = [System.IO.File]::Open("C:\path\to\your\dataset.csv", [System.IO.FileMode]::Open)
}
```

### D. Check Health
Verify if the service is running:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```
