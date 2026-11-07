"""
Hotel Revenue Intelligence & Intelligent Forecasting Platform.
Enterprise SaaS Frontend Application.
"""

import io
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from api_client import HotelRevenueAPIClient
from components.theme import COLORS
from components.ui import render_top_header, render_alert
import views

# 1. Page Configuration
st.set_page_config(
    page_title="Hotel Revenue Intelligence | Enterprise Platform",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. Inject Enterprise CSS
def _load_styles():
    css_path = Path("assets/style.css")
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                return
        except Exception:
            pass

    # Fallback inline CSS tokens
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {COLORS['bg_primary']} !important; }}
        [data-testid="stSidebar"] {{ background-color: {COLORS['sidebar']} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_load_styles()


# 3. Data Utilities & Preprocessing
@st.cache_data(show_spinner=False)
def load_data(uploaded_file: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def create_demo_data(n: int = 240) -> pd.DataFrame:
    rng = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    np.random.seed(42)
    countries = ["US", "UK", "DE", "FR", "IN", "AU", "CA"]
    channels = ["Direct", "OTA", "Corporate", "Online"]
    guest_types = ["Transient", "Group", "Contract", "Transient-Party"]
    segments = ["Leisure", "Corporate", "Wholesale", "Government"]
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    occ = np.random.uniform(0.45, 0.95, size=n).round(2)
    adr = np.random.uniform(110.0, 240.0, size=n).round(2)
    avail = np.random.randint(80, 150, size=n)
    reserved = (avail * occ).astype(int)
    fixed = np.random.uniform(1800.0, 2800.0, size=n).round(2)
    var_c = np.random.uniform(300.0, 750.0, size=n).round(2)
    total_c = (fixed + var_c).round(2)
    total_rev = (reserved * adr).round(2)

    df = pd.DataFrame({
        "Date": rng.strftime("%Y-%m-%d"),
        "Month": rng.month_name(),
        "Booking_Channel": np.random.choice(channels, size=n),
        "Guest_Type": np.random.choice(guest_types, size=n),
        "Market_Segment": np.random.choice(segments, size=n),
        "Guest_Country": np.random.choice(countries, size=n),
        "Season": np.random.choice(seasons, size=n),
        "Occupancy_Rate": occ,
        "ADR": adr,
        "Average_Daily_Rate": adr,
        "Available_Rooms": avail,
        "Reserved_Rooms": reserved,
        "Fixed_Costs": fixed,
        "Variable_Costs": var_c,
        "Total_Costs": total_c,
        "Total_Revenue": total_rev,
        "Profit": (total_rev - total_c).round(2),
    })
    return df


def fill_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    for column in cleaned.columns:
        if cleaned[column].isnull().any():
            col = cleaned[column]
            if col.dtype == "object":
                cleaned[column] = col.fillna("Unknown")
            else:
                try:
                    cleaned[column] = col.fillna(col.median())
                except Exception:
                    cleaned[column] = col.fillna(0)
    return cleaned


def create_profit_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    if {"Total_Revenue", "Total_Costs"}.issubset(enriched.columns):
        enriched["Profit"] = enriched["Total_Revenue"] - enriched["Total_Costs"]
    return enriched


# 4. Navigation Architecture Specification
NAVIGATION = {
    "Overview": {
        "Executive Dashboard": {
            "key": "executive",
            "module": views.executive,
            "title": "Executive Dashboard",
            "subtitle": "Monitor hotel performance, forecast demand and optimize revenue decisions.",
        },
    },
    "Revenue Management": {
        "Revenue Analytics": {
            "key": "revenue_analytics",
            "module": views.revenue_analytics,
            "title": "Revenue Analytics",
            "subtitle": "Temporal revenue trends, segment profitability, and ADR/RevPAR yield dynamics.",
        },
        "Revenue Forecasting": {
            "key": "revenue_forecasting",
            "module": views.revenue_forecasting,
            "title": "Revenue Forecasting",
            "subtitle": "Multi-step 7, 30, and 90-day forward projections with empirical confidence bounds.",
        },
        "ADR Optimization": {
            "key": "adr_optimization",
            "module": views.adr_optimization,
            "title": "ADR & Pricing Optimization",
            "subtitle": "Simulate price elasticity of demand to identify revenue and gross profit maximizing price points.",
        },
        "Guest Segmentation": {
            "key": "guest_segmentation",
            "module": views.guest_segmentation,
            "title": "Guest Segmentation",
            "subtitle": "Channel share, international origin markets, and customer segment profitability.",
        },
        "What-If Prediction": {
            "key": "revenue_whatif",
            "module": views.revenue_whatif,
            "title": "What-If Revenue Predictor",
            "subtitle": "Simulate daily hotel revenue across arbitrary temporal, pricing, and cost parameters.",
        },
        "Demand Forecasting": {
            "key": "demand_forecasting",
            "module": views.demand_forecasting,
            "title": "Room Demand Forecasting",
            "subtitle": "Predict room demand, cumulative room nights, and expected occupancy pacing.",
        },
        "Pre-Booking Estimator": {
            "key": "prebooking",
            "module": views.prebooking,
            "title": "Pre-Booking Estimator",
            "subtitle": "Ex-ante quotation engine estimating stay valuation strictly without realized target leakage.",
        },
        "Competitor Pricing & MPI": {
            "key": "competitor_pricing",
            "module": views.competitor_pricing,
            "title": "Competitor Rate Benchmarking & MPI",
            "subtitle": "Evaluate Market Price Index (MPI) and record authorized competitor rate benchmarks.",
        },
        "Strategic Recommendations": {
            "key": "recommendations",
            "module": views.recommendations,
            "title": "Strategic Recommendations",
            "subtitle": "Cross-module algorithmic decision support synthesizing pricing, demand, and risk signals.",
        },
    },
    "Machine Learning": {
        "Model Governance": {
            "key": "model_governance",
            "module": views.model_governance,
            "title": "Model Governance & Registry",
            "subtitle": "Audit production models, inspect multi-model tournaments, and manage atomic version rollbacks.",
        },
        "Explainable AI (SHAP)": {
            "key": "explainable_ai",
            "module": views.explainable_ai,
            "title": "Explainable AI (SHAP)",
            "subtitle": "Global feature importances and local Shapley value waterfall explanations.",
        },
        "Anomaly Detection": {
            "key": "anomaly_detection",
            "module": views.anomaly_detection,
            "title": "Anomaly Detection",
            "subtitle": "Isolation Forest and statistical outlier auditing for operational excursions in revenue and ADR.",
        },
        "Data Drift Monitoring": {
            "key": "data_drift",
            "module": views.data_drift,
            "title": "Data Drift Monitoring",
            "subtitle": "Track statistical feature shift via Kolmogorov-Smirnov tests and Population Stability Index (PSI).",
        },
    },
    "Cancellation Intelligence": {
        "Cancellation Prediction": {
            "key": "cancellation_prediction",
            "module": views.cancellation_prediction,
            "title": "Cancellation Prediction",
            "subtitle": "Calibrated reservation cancellation probability scoring powered by Antonio et al. research data.",
        },
        "Calibration & Thresholds": {
            "key": "cancellation_calibration",
            "module": views.cancellation_calibration,
            "title": "Calibration & Threshold Optimization",
            "subtitle": "Reliability diagrams, Brier loss scores, and asymmetric cost threshold optimization.",
        },
        "Cancellation Explainability": {
            "key": "cancellation_explainability",
            "module": views.cancellation_explainability,
            "title": "Cancellation Explainability",
            "subtitle": "TreeExplainer SHAP decomposition attributing individual cancellation risk factors.",
        },
        "What-If Policy Simulator": {
            "key": "cancellation_whatif",
            "module": views.cancellation_whatif,
            "title": "What-If Policy Simulator",
            "subtitle": "Simulate 16 counterfactual operational interventions (deposits, lead times, parking, requests).",
        },
        "Segment Risk Analytics": {
            "key": "cancellation_segments",
            "module": views.cancellation_segments,
            "title": "Segment Risk Analytics",
            "subtitle": "Empirical ground-truth cancellation rates across lead times, channels, and customer types.",
        },
    },
    "Inventory Management": {
        "Room Inventory": {
            "key": "room_inventory",
            "module": views.room_inventory,
            "title": "Room Inventory Management",
            "subtitle": "Manage physical room capacities, out-of-order maintenance rooms, and effective capacity.",
        },
        "Occupancy Calendar": {
            "key": "occupancy_calendar",
            "module": views.occupancy_calendar,
            "title": "Occupancy Calendar & Audit",
            "subtitle": "14-day timeline of confirmed reservations vs capacity and active reservation ledger.",
        },
        "Cancellation Forecast": {
            "key": "cancellation_forecast",
            "module": views.cancellation_forecast,
            "title": "Cancellation & Arrival Forecast",
            "subtitle": "Expected cancellation count and exact Poisson-Binomial arrival distribution convolution.",
        },
        "Overbooking Optimizer": {
            "key": "overbooking_optimizer",
            "module": views.overbooking_optimizer,
            "title": "Overbooking Optimizer",
            "subtitle": "Constrained net contribution optimization maximizing revenue subject to hard walk risk limit P(Walk) <= α.",
        },
        "Revenue Impact Simulator": {
            "key": "revenue_simulator",
            "module": views.revenue_simulator,
            "title": "Revenue Impact Simulator",
            "subtitle": "Custom scenario simulator evaluating additional bookings against relocation walk penalties.",
        },
    },
    "Data Management": {
        "Dataset Ingestion": {
            "key": "dataset_ingestion",
            "module": views.dataset_ingestion,
            "title": "Dataset Ingestion & Training",
            "subtitle": "Upload historical reservation CSVs to trigger automated merging and multi-model retraining.",
        },
        "Data Explorer": {
            "key": "data_explorer",
            "module": views.data_explorer,
            "title": "Data Explorer & Inspector",
            "subtitle": "Examine raw and cleaned reservation records, schema metrics, and static export options.",
        },
        "Custom Analytics": {
            "key": "custom_analytics",
            "module": views.custom_analytics,
            "title": "Custom Analytics Builder",
            "subtitle": "Interactive exploratory chart generator across arbitrary dataset dimensions and metrics.",
        },
        "PMS Integration Audit": {
            "key": "pms_integration",
            "module": views.pms_integration,
            "title": "PMS Integration Audit",
            "subtitle": "Technical gap analysis comparing demonstration schema against enterprise PMS platforms.",
        },
    },
}


# 5. Session State Initialization
if "nav_category" not in st.session_state:
    st.session_state.nav_category = "Overview"
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Executive Dashboard"


# 6. Sidebar Construction (Enterprise Dark Theme #111827)
backend_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
api_client = HotelRevenueAPIClient(base_url=backend_url)

with st.sidebar:
    # Branding header
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>HOTEL INTELLIGENCE</h2>
            <p>Enterprise SaaS Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Backend Connection Status
    configured_url = st.text_input(
        "FastAPI Server URL",
        value=backend_url,
        help="Target address of the running FastAPI ML prediction server",
    )
    if configured_url != backend_url:
        api_client = HotelRevenueAPIClient(base_url=configured_url)

    is_online, health_data, health_err = api_client.check_health()
    if is_online:
        st.markdown(
            '<span class="status-pill status-pill-success" style="margin-bottom: 0.75rem;">● FastAPI Server Live (:8000)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-pill-danger" style="margin-bottom: 0.75rem;">○ FastAPI Disconnected</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color: #1F2937; margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

    # Navigation Section Category Selector
    st.markdown('<p class="sidebar-category">MODULE SECTION</p>', unsafe_allow_html=True)
    categories = list(NAVIGATION.keys())
    selected_cat = st.selectbox(
        "Module Category",
        categories,
        index=categories.index(st.session_state.nav_category) if st.session_state.nav_category in categories else 0,
        label_visibility="collapsed",
    )
    if selected_cat != st.session_state.nav_category:
        st.session_state.nav_category = selected_cat
        # Default to first page in the newly selected category
        st.session_state.nav_page = list(NAVIGATION[selected_cat].keys())[0]

    # Sub-pages in Active Section
    st.markdown(f'<p class="sidebar-category">{selected_cat.upper()} PAGES</p>', unsafe_allow_html=True)
    page_options = list(NAVIGATION[selected_cat].keys())
    current_page_idx = page_options.index(st.session_state.nav_page) if st.session_state.nav_page in page_options else 0
    selected_page = st.radio(
        "Select Page",
        page_options,
        index=current_page_idx,
        label_visibility="collapsed",
    )
    st.session_state.nav_page = selected_page

    st.markdown("<hr style='border-color: #1F2937; margin: 1rem 0;'>", unsafe_allow_html=True)

    # Local Data Explorer Source Controls
    st.markdown('<p class="sidebar-category">LOCAL WORKSPACE DATA</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Local CSV",
        type=["csv"],
        key="local_csv_uploader",
        help="Upload CSV to visualize custom booking and revenue dynamics",
    )
    if st.button("Load Demo Dataset (240 Days)", use_container_width=True):
        demo_df = create_demo_data()
        st.session_state.active_df = demo_df
        st.success("Loaded 240-day operational demo dataset!")


# 7. Dataset Ingestion / Fallback Resolution
fallback_path = "Hotel book.csv"
dataframe = None

if "active_df" in st.session_state:
    dataframe = st.session_state.active_df
elif uploaded_file is not None:
    dataframe = load_data(uploaded_file)
    st.session_state.active_df = dataframe
elif Path(fallback_path).exists():
    dataframe = pd.read_csv(fallback_path)
elif Path("assets/demo_data.csv").exists():
    dataframe = pd.read_csv("assets/demo_data.csv")
else:
    dataframe = create_demo_data()

clean_data = fill_missing_values(dataframe)
dashboard_data = create_profit_column(clean_data)


# 8. Render Top Enterprise Header Banner
page_info = NAVIGATION[st.session_state.nav_category][st.session_state.nav_page]
render_top_header(
    title=page_info["title"],
    subtitle=page_info["subtitle"],
    category_name=st.session_state.nav_category,
    is_connected=is_online,
    server_port="8000",
)


# 9. Render Selected View Module
view_module = page_info["module"]
try:
    view_module.render(api_client, dashboard_data)
except Exception as e:
    render_alert(f"<b>View Rendering Error:</b> An unexpected exception occurred while rendering {st.session_state.nav_page}:<br><code>{str(e)}</code>", alert_type="danger")
    st.exception(e)


# 10. Enterprise Footer
st.markdown(
    """
    <div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #E5E7EB; text-align: center; font-size: 0.8rem; color: #94A3B8;">
        Hotel Revenue Intelligence & Intelligent Forecasting Platform &nbsp;|&nbsp; Enterprise SaaS Edition &nbsp;|&nbsp; FastAPI Single Source of Truth
    </div>
    """,
    unsafe_allow_html=True,
)
