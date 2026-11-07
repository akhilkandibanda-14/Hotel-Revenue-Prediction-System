# 🏨 Hotel Revenue Prediction System

An end-to-end Machine Learning and Analytics platform for predicting hotel revenue, analyzing booking dynamics, and continuously retraining models with incoming reservation feeds.

---

## 🌟 Architecture & Features

The project is structured into two main sub-systems:

```
Hotel-Revenue-Prediction-System/
├── hotel_boooking/             # FastAPI Backend Service & Continuous ML Engine
│   ├── app/                    # REST API, database, scheduler, predictor & trainer
│   ├── datasets/               # Uploaded, merged, and active datasets
│   ├── models/                 # Serialized model (.pkl), encoders, feature columns
│   ├── reservations/           # Live reservation log store
│   └── tests/                  # Automated test suite
│
├── web_app_dashboards/         # Interactive Streamlit Web App
│   └── web_app/
│       ├── app.py              # Visual dashboard, EDA charts & ad-hoc ML model
│       ├── assets/             # Styling, reports, and assets
│       └── requirements.txt    # Python dependencies
│
├── run_project.py              # Unified single-command launcher
└── README.md
```

### Key Capabilities
- **FastAPI ML Service**:
  - `POST /predict/`: Predicts expected revenue for specific dates.
  - `POST /reservation/`: Records real-time booking logs.
  - `POST /dataset/upload`: Uploads new CSV training datasets.
  - `GET /health`: Health verification endpoint.
  - **Background Scheduler**: Continuously monitors and automatically triggers dataset merging and model retraining.
- **Streamlit Analytics Dashboard**:
  - Interactive exploration of revenue, profit, occupancy rate, and booking channels.
  - Feature importance breakdown from Random Forest regressors.
  - Actual vs. Predicted revenue performance scatter plots.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure Python 3.10+ is installed on your machine.

### 2. Install Dependencies
```bash
pip install fastapi uvicorn streamlit scikit-learn pandas numpy plotly matplotlib seaborn
```

### 3. Launch the Entire System
You can start both the FastAPI backend and Streamlit dashboard with a single command:
```bash
python run_project.py
```

Or run them individually:

- **Run FastAPI Backend**:
  ```bash
  cd hotel_boooking
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
  Swagger Docs will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

- **Run Streamlit Dashboard**:
  ```bash
  cd web_app_dashboards/web_app
  streamlit run app.py --server.port 8501
  ```
  Web UI will open at [http://localhost:8501](http://localhost:8501).

---

## 🧪 Testing

Execute automated unit tests:
```bash
cd hotel_boooking
python -m unittest tests/test_service.py
```

---

## 📄 License
MIT License

