import io
import os
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import HotelRevenueAPIClient

# Configure page
st.set_page_config(
    page_title="Hotel Revenue Management & ML Platform",
    page_icon="🏨",
    layout="wide",
)


def _load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                return
        except Exception:
            pass

    # fallback inline CSS
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f7fb 0%, #eef3f8 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #0d233a 0%, #1d4e89 100%);
            color: white;
            box-shadow: 0 16px 36px rgba(13, 35, 58, 0.16);
            margin-bottom: 1.2rem;
        }
        .hero h1, .hero p {
            margin: 0;
        }
        .metric-card {
            padding: 1.1rem 1.2rem;
            border-radius: 16px;
            background: white;
            border: 1px solid rgba(16, 42, 67, 0.08);
            box-shadow: 0 6px 20px rgba(16, 42, 67, 0.05);
            margin-bottom: 0.5rem;
        }
        .status-badge-ok {
            background-color: #d1fae5;
            color: #065f46;
            padding: 0.25rem 0.6rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-badge-err {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 0.25rem 0.6rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_load_css()


@st.cache_data(show_spinner=False)
def load_data(uploaded_file: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def create_demo_data(n=240) -> pd.DataFrame:
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
        "RevPAR": (adr * occ).round(2),
        "Available_Rooms": avail,
        "Reserved_Rooms": reserved,
        "Fixed_Costs": fixed,
        "Variable_Costs": var_c,
        "Total_Costs": total_c,
        "Total_Revenue": total_rev,
    })
    df.to_csv("assets/demo_data.csv", index=False)
    return df


def fill_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype

    for column in cleaned.columns:
        col = cleaned[column]
        if is_numeric_dtype(col):
            try:
                replacement = col.median()
            except Exception:
                replacement = col.mean() if col.dropna().size > 0 else 0
            cleaned[column] = col.fillna(replacement)
        elif is_object_dtype(col) or is_string_dtype(col) or str(col.dtype) == "category":
            mode_values = col.mode(dropna=True)
            replacement = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            cleaned[column] = col.fillna(replacement)
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


def safe_monthly_revenue(dataframe: pd.DataFrame) -> pd.Series | None:
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    if "Month_Name" in dataframe.columns and "Total_Revenue" in dataframe.columns:
        series = dataframe.groupby("Month_Name")["Total_Revenue"].sum().reindex(month_order)
        return series.dropna()

    if "Month" in dataframe.columns and "Total_Revenue" in dataframe.columns:
        series = dataframe.groupby("Month")["Total_Revenue"].sum()
        try:
            return series.reindex(month_order).dropna()
        except Exception:
            return series

    return None


def _load_html_snippets():
    header_path = Path("assets/header.html")
    if header_path.exists():
        try:
            with open(header_path, "r", encoding="utf-8") as f:
                st.markdown(f.read(), unsafe_allow_html=True)
                return
        except Exception:
            pass

    st.markdown(
        """
        <div class="hero">
            <h1>Hotel Revenue Management & ML Platform</h1>
            <p>Unified Analytics Dashboard powered by FastAPI Multi-Model Prediction Service</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


_load_html_snippets()

# Initialize API Client
backend_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
api_client = HotelRevenueAPIClient(base_url=backend_url)

# Sidebar Configuration
with st.sidebar:
    st.header("⚡ Backend Connection")
    configured_url = st.text_input("FastAPI Base URL", value=backend_url, help="Address of the running FastAPI ML server")
    if configured_url != backend_url:
        api_client = HotelRevenueAPIClient(base_url=configured_url)

    is_online, health_data, health_err = api_client.check_health()
    if is_online:
        st.markdown('<span class="status-badge-ok">🟢 FastAPI Connected (Port 8000)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-err">🔴 FastAPI Disconnected</span>', unsafe_allow_html=True)
        st.caption(f"Error: {health_err}")

    st.divider()

    st.header("📁 Local Data Source")
    uploaded_file = st.file_uploader("Upload CSV for Local Exploration", type=["csv"], key="local_explorer_uploader")
    st.caption("Visualizes booking dynamics, channels, seasons, and costs.")
    if st.button("Load Demo Dataset"):
        create_demo_data()
        st.success("Demo dataset generated!")

# Data resolution
fallback_path = "Hotel book.csv"
dataframe = None

if uploaded_file is not None:
    dataframe = load_data(uploaded_file)
elif pd.io.common.file_exists(fallback_path):
    dataframe = pd.read_csv(fallback_path)
elif pd.io.common.file_exists("assets/demo_data.csv"):
    dataframe = pd.read_csv("assets/demo_data.csv")

if dataframe is None:
    st.info("Upload your CSV or click 'Load Demo Dataset' in the sidebar to start.")
    st.stop()

raw_data = dataframe.copy()
clean_data = fill_missing_values(dataframe)
dashboard_data = create_profit_column(clean_data)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><strong>Total Records</strong><br>{dashboard_data.shape[0]:,} rows</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><strong>Features Tracked</strong><br>{dashboard_data.shape[1]:,} columns</div>', unsafe_allow_html=True)
with col3:
    if "Total_Revenue" in dashboard_data.columns:
        st.markdown(f'<div class="metric-card"><strong>Total Revenue</strong><br>${dashboard_data["Total_Revenue"].sum():,.2f}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><strong>Total Revenue</strong><br>N/A</div>', unsafe_allow_html=True)
with col4:
    if "Profit" in dashboard_data.columns:
        st.markdown(f'<div class="metric-card"><strong>Gross Profit</strong><br>${dashboard_data["Profit"].sum():,.2f}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><strong>Gross Profit</strong><br>N/A</div>', unsafe_allow_html=True)

# Main Navigation Tabs
(
    tab_overview,
    tab_revenue,
    tab_segments,
    tab_predict,
    tab_forecast,
    tab_optimizer,
    tab_demand,
    tab_prebooking,
    tab_cancellation,
    tab_overbooking,
    tab_competitor,
    tab_recommendations,
    tab_xai,
    tab_anomalies,
    tab_drift,
    tab_governance,
    tab_upload,
    tab_custom,
    tab_data,
) = st.tabs([
    "📊 Overview",
    "📈 Revenue Trends",
    "👥 Guest Segments",
    "🔮 What-If Predictor",
    "📅 Forecasting",
    "🏷️ ADR Optimization",
    "📈 Demand Forecast",
    "🏷️ Pre-Booking Estimator",
    "❌ Cancellation Analytics",
    "🏨 Inventory & Overbooking",
    "🏢 Competitor Pricing",
    "💡 Business Decisions",
    "🧠 Explainable AI",
    "🚨 Anomaly Detection",
    "📡 Data Drift",
    "🏛️ Model Governance",
    "📤 Dataset Ingestion",
    "🎨 Custom Charts",
    "📋 Data Preview",
])

# ==========================================
# 1. OVERVIEW TAB
# ==========================================
with tab_overview:
    st.subheader("Data Dynamics & Distribution")
    st.dataframe(raw_data.head(15), use_container_width=True)

    left, right = st.columns(2)
    with left:
        if {"Total_Revenue", "Profit"}.issubset(dashboard_data.columns):
            hover_cols = [c for c in ["Date", "Guest_Country", "Booking_Channel"] if c in dashboard_data.columns]
            fig = px.scatter(
                dashboard_data,
                x="Total_Revenue",
                y="Profit",
                hover_data=hover_cols if hover_cols else None,
                title="Revenue vs Profit Relationship",
                color_discrete_sequence=["#1d4e89"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue vs Profit chart requires Total_Revenue and Total_Costs columns.")

    with right:
        if {"Season", "Occupancy_Rate"}.issubset(dashboard_data.columns):
            occ_data = dashboard_data.groupby("Season")["Occupancy_Rate"].mean().reset_index()
            # Handle if occupancy rate is stored as ratio 0-1
            if occ_data["Occupancy_Rate"].max() <= 1.0:
                occ_data["Occupancy_Rate"] = occ_data["Occupancy_Rate"] * 100.0
            fig = px.bar(
                occ_data,
                x="Season",
                y="Occupancy_Rate",
                title="Average Occupancy Rate by Season (%)",
                color="Season",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Occupancy chart requires Season and Occupancy_Rate columns.")

# ==========================================
# 2. REVENUE TRENDS TAB
# ==========================================
with tab_revenue:
    st.subheader("Temporal Revenue Performance")
    monthly_revenue = None
    if "Date" in dashboard_data.columns and "Total_Revenue" in dashboard_data.columns:
        revenue_frame = dashboard_data.copy()
        revenue_frame["Date"] = pd.to_datetime(revenue_frame["Date"], errors="coerce")
        revenue_frame["Month_Name"] = revenue_frame["Date"].dt.month_name()
        monthly_revenue = safe_monthly_revenue(revenue_frame)
    else:
        monthly_revenue = safe_monthly_revenue(dashboard_data)

    if monthly_revenue is not None and not monthly_revenue.empty:
        rev_df = monthly_revenue.reset_index()
        rev_df.columns = ["Month", "Revenue"]
        fig = px.line(
            rev_df,
            x="Month",
            y="Revenue",
            title="Monthly Revenue Trend",
            markers=True,
            line_shape="spline",
        )
        fig.update_traces(line=dict(color="#1d4e89", width=3), marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add a Date or Month column to visualize temporal revenue trends.")

    if "Total_Revenue" in dashboard_data.columns:
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            data=dashboard_data.to_csv(index=False).encode("utf-8"),
            file_name="hotel_revenue_export.csv",
            mime="text/csv",
        )

# ==========================================
# 3. GUEST SEGMENTS TAB
# ==========================================
with tab_segments:
    st.subheader("Channel & Demographic Breakdown")
    left, right = st.columns(2)

    with left:
        if "Booking_Channel" in dashboard_data.columns:
            channel_counts = dashboard_data["Booking_Channel"].value_counts()
            fig = px.pie(
                names=channel_counts.index,
                values=channel_counts.values,
                title="Booking Channel Share",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Booking_Channel column not found.")

    with right:
        if "Guest_Country" in dashboard_data.columns:
            top_countries = dashboard_data["Guest_Country"].value_counts().head(10)
            fig = px.bar(
                x=top_countries.values,
                y=top_countries.index,
                orientation="h",
                title="Top 10 Origin Countries",
                labels={"x": "Total Bookings", "y": "Country"},
                color_discrete_sequence=["#2b6cb0"]
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Guest_Country column not found.")

# ==========================================
# 4. WHAT-IF PREDICTOR TAB (FastAPI Integration)
# ==========================================
with tab_predict:
    st.subheader("🔮 What-If Scenario Revenue Predictor")
    st.write("Interact with the production ML model served by FastAPI to simulate expected daily revenues under operational scenarios.")

    if not is_online:
        st.error("⚠️ FastAPI backend service is offline. Please start the backend service on port 8000 to enable predictions.")
    else:
        with st.form("what_if_prediction_form"):
            st.markdown("#### 1. Temporal & Channel Parameters")
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)

            with p_col1:
                target_date = st.date_input("Target Date", value=date(2026, 7, 15))
            with p_col2:
                season_options = ["Summer", "Autumn", "Winter", "Spring"]
                selected_season = st.selectbox("Season", season_options, index=0)
            with p_col3:
                channel_options = ["Online", "Direct", "OTA", "Corporate"]
                selected_channel = st.selectbox("Booking Channel", channel_options, index=0)
            with p_col4:
                guest_type_options = ["Transient", "Group", "Contract", "Transient-Party"]
                selected_guest_type = st.selectbox("Guest Type", guest_type_options, index=0)

            st.markdown("#### 2. Pricing, Inventory & Costs")
            p_col5, p_col6, p_col7, p_col8 = st.columns(4)

            with p_col5:
                occ_input = st.slider("Occupancy Rate (%)", min_value=10, max_value=100, value=85, step=1)
                occ_rate = occ_input / 100.0
            with p_col6:
                adr_input = st.number_input("Average Daily Rate (ADR $)", min_value=20.0, max_value=1000.0, value=175.0, step=5.0)
            with p_col7:
                avail_rooms = st.number_input("Available Rooms", min_value=10, max_value=1000, value=100, step=5)
            with p_col8:
                auto_reserved = int(avail_rooms * occ_rate)
                reserved_rooms = st.number_input("Reserved Rooms", min_value=0, max_value=int(avail_rooms), value=auto_reserved, step=1)

            p_col9, p_col10, p_col11, p_col12 = st.columns(4)
            with p_col9:
                fixed_costs = st.number_input("Fixed Costs ($)", min_value=0.0, max_value=50000.0, value=2200.0, step=50.0)
            with p_col10:
                var_costs = st.number_input("Variable Costs ($)", min_value=0.0, max_value=20000.0, value=550.0, step=25.0)
            with p_col11:
                total_costs = fixed_costs + var_costs
                st.metric("Total Costs ($)", f"${total_costs:,.2f}")
            with p_col12:
                strict_mode = st.checkbox("Strict Feature Enforcement", value=True, help="Rejects request if required operational fields are missing")

            predict_submitted = st.form_submit_button("🚀 Run Revenue Prediction", use_container_width=True)

        if predict_submitted:
            rev_par = round(adr_input * occ_rate, 2)
            payload = {
                "Date": target_date.strftime("%Y-%m-%d"),
                "Month": target_date.strftime("%B"),
                "Booking_Channel": selected_channel,
                "Guest_Type": selected_guest_type,
                "Market_Segment": "Leisure",
                "Guest_Country": "US",
                "Season": selected_season,
                "Occupancy_Rate": occ_rate,
                "ADR": float(adr_input),
                "RevPAR": float(rev_par),
                "Available_Rooms": int(avail_rooms),
                "Reserved_Rooms": int(reserved_rooms),
                "Fixed_Costs": float(fixed_costs),
                "Variable_Costs": float(var_costs),
                "Total_Costs": float(total_costs),
                "strict": strict_mode,
            }

            with st.spinner("Calling FastAPI Prediction Engine..."):
                ok, pred_res, err = api_client.predict(payload)

            if ok:
                st.success("✅ Prediction Successfully Generated by Backend Service!")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric(
                        "Predicted Daily Revenue",
                        f"${pred_res.get('Predicted_Revenue', 0):,.2f}",
                        delta=f"RevPAR: ${rev_par:,.2f}"
                    )
                with res_col2:
                    st.metric("Serving Algorithm", str(pred_res.get("Algorithm", "Extra Trees")))
                with res_col3:
                    st.metric("Model Version", str(pred_res.get("Model_Version", "v001")))
                with res_col4:
                    st.metric("Inference Mode", str(pred_res.get("Prediction_Mode", "full_features")))

                with st.expander("🔍 View Verified Prediction Payload & Response"):
                    st.json({"request_payload": payload, "backend_response": pred_res})
            else:
                st.error(f"❌ Prediction Failed: {err}")

# ==========================================
# 5. FORECASTING TAB (Phase 3)
# ==========================================
with tab_forecast:
    st.subheader("📅 Intelligent Revenue & Occupancy Forecasting")
    st.write("Multi-step forward forecasting across 7, 30, and 90-day operational horizons with chronological backtesting validation and empirical 95% confidence intervals.")

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service to compute forecasts.")
    else:
        fc_col1, fc_col2, fc_col3 = st.columns([1, 1, 1])
        with fc_col1:
            horizon_choice = st.radio("Forecasting Horizon", [7, 30, 90], index=1, horizontal=True, format_func=lambda x: f"{x} Days")
        with fc_col2:
            forecast_type = st.radio("Target Metric", ["Total Revenue ($)", "Occupancy Rate (%)"], horizontal=True)
        with fc_col3:
            st.write("")
            st.write("")
            run_fc_btn = st.button("🚀 Generate Forecast", key="btn_run_forecast", use_container_width=True)

        if forecast_type == "Total Revenue ($)":
            ok_fc, fc_data, fc_err = api_client.get_revenue_forecast(horizon=horizon_choice)
            if ok_fc and fc_data:
                # KPI Summary Cards
                b_metrics = fc_data.get("backtest_metrics", {})
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.metric("Total Projected Revenue", f"${fc_data.get('total_projected_revenue', 0):,.2f}")
                with kpi2:
                    st.metric("Daily Avg Revenue", f"${fc_data.get('average_daily_revenue', 0):,.2f}")
                with kpi3:
                    st.metric("Backtested RMSE", f"${b_metrics.get('rmse', 0):,.2f}")
                with kpi4:
                    st.metric("Backtested MAPE", f"{b_metrics.get('mape', 0):.1f}%")

                # Plotly time series chart with confidence band
                points = fc_data.get("forecast", [])
                fc_df = pd.DataFrame(points)
                if not fc_df.empty:
                    fig_fc = go.Figure()
                    # 95% CI shaded band
                    fig_fc.add_trace(go.Scatter(
                        x=fc_df["date"],
                        y=fc_df["upper_bound"],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig_fc.add_trace(go.Scatter(
                        x=fc_df["date"],
                        y=fc_df["lower_bound"],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(29, 78, 137, 0.15)',
                        name='95% Confidence Interval',
                    ))
                    # Expected Forecast Line
                    fig_fc.add_trace(go.Scatter(
                        x=fc_df["date"],
                        y=fc_df["predicted_revenue"],
                        mode='lines+markers',
                        line=dict(color='#1d4e89', width=3),
                        marker=dict(size=6),
                        name='Predicted Revenue',
                        hovertemplate='<b>%{x}</b><br>Predicted: $%{y:,.2f}<extra></extra>'
                    ))
                    fig_fc.update_layout(
                        title=f"{horizon_choice}-Day Daily Revenue Forecast with 95% Confidence Bounds",
                        xaxis_title="Date",
                        yaxis_title="Daily Revenue ($)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

                    with st.expander("📋 View Daily Forecast Breakdown Table"):
                        st.dataframe(fc_df, use_container_width=True)

                # Limitations Alert
                with st.expander("ℹ️ Statistical Methodology & Model Caveats"):
                    for lim in fc_data.get("limitations", []):
                        st.markdown(f"- {lim}")
            else:
                st.error(f"Failed to generate revenue forecast: {fc_err}")

        else:
            # Occupancy Rate Forecast
            ok_occ, occ_data, occ_err = api_client.get_occupancy_forecast(horizon=horizon_choice)
            if ok_occ and occ_data:
                b_metrics = occ_data.get("backtest_metrics", {})
                h_pats = occ_data.get("historical_patterns", {})

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.metric("Projected Mean Occupancy", f"{occ_data.get('mean_projected_occupancy', 0)*100:.1f}%")
                with kpi2:
                    st.metric("Weekday Historical Avg", f"{h_pats.get('weekday_average_occupancy', 0)*100:.1f}%")
                with kpi3:
                    st.metric("Weekend Historical Avg", f"{h_pats.get('weekend_average_occupancy', 0)*100:.1f}%")
                with kpi4:
                    st.metric("Backtested MAPE", f"{b_metrics.get('mape', 0):.1f}%")

                points = occ_data.get("forecast", [])
                occ_df = pd.DataFrame(points)
                if not occ_df.empty:
                    fig_occ = go.Figure()
                    fig_occ.add_trace(go.Scatter(
                        x=occ_df["date"],
                        y=occ_df["upper_bound"] * 100,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig_occ.add_trace(go.Scatter(
                        x=occ_df["date"],
                        y=occ_df["lower_bound"] * 100,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(42, 157, 143, 0.2)',
                        name='95% Confidence Bounds',
                    ))
                    fig_occ.add_trace(go.Scatter(
                        x=occ_df["date"],
                        y=occ_df["predicted_occupancy"] * 100,
                        mode='lines+markers',
                        line=dict(color='#2a9d8f', width=3),
                        marker=dict(size=6),
                        name='Projected Occupancy Rate (%)',
                        hovertemplate='<b>%{x}</b><br>Occupancy: %{y:.1f}%<extra></extra>'
                    ))
                    fig_occ.update_layout(
                        title=f"{horizon_choice}-Day Daily Occupancy Forecast (%)",
                        xaxis_title="Date",
                        yaxis_title="Occupancy Rate (%)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_occ, use_container_width=True)

                    with st.expander("📋 View Daily Occupancy Forecast Data"):
                        st.dataframe(occ_df, use_container_width=True)
            else:
                st.error(f"Failed to generate occupancy forecast: {occ_err}")

# ==========================================
# 6. PRICING & ADR OPTIMIZATION TAB (Phase 3)
# ==========================================
with tab_optimizer:
    st.subheader("🏷️ ADR & Dynamic Revenue Optimization")
    st.write("Simulate pricing changes (+/- 30% ADR) with empirical price elasticity of demand to identify revenue and gross profit maximizing price points.")

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service to run pricing simulations.")
    else:
        with st.expander("⚙️ Pricing & Hotel Cost Assumptions", expanded=True):
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                input_base_adr = st.number_input("Base ADR ($)", value=165.0, min_value=50.0, max_value=500.0, step=5.0)
            with p_col2:
                input_base_occ = st.slider("Base Occupancy Rate", min_value=0.20, max_value=1.0, value=0.70, step=0.01)
            with p_col3:
                input_rooms = st.number_input("Total Room Inventory", value=120, min_value=10, max_value=1000, step=10)
            with p_col4:
                input_elasticity = st.slider("Assumed Price Elasticity (ε)", min_value=-1.0, max_value=-0.05, value=-0.35, step=0.05, help="Standard hotel demand elasticity ranges from -0.30 to -0.60.")

            c_col1, c_col2 = st.columns(2)
            with c_col1:
                input_fixed_costs = st.number_input("Daily Fixed Costs ($)", value=2500.0, min_value=0.0, max_value=50000.0, step=100.0)
            with c_col2:
                input_var_cost = st.number_input("Variable Cost per Occupied Room ($)", value=18.0, min_value=0.0, max_value=200.0, step=2.0)

        ok_opt, opt_res, opt_err = api_client.get_pricing_optimization(
            base_adr=input_base_adr,
            base_occupancy=input_base_occ,
            total_rooms=input_rooms,
            fixed_costs=input_fixed_costs,
            var_cost_per_room=input_var_cost,
            custom_elasticity=input_elasticity,
        )

        if ok_opt and opt_res:
            opt_rev = opt_res.get("optimal_revenue_scenario", {})
            opt_prof = opt_res.get("optimal_profit_scenario", {})
            active_eps = opt_res.get("active_elasticity", input_elasticity)

            # Top KPI highlights
            k_col1, k_col2, k_col3 = st.columns(3)
            with k_col1:
                st.metric("Price Elasticity of Demand (ε)", f"{active_eps:.2f}", help="Percentage demand change per 1% price change")
            with k_col2:
                st.metric(
                    "Revenue Maximizing ADR",
                    f"${opt_rev.get('projected_adr', 0):,.2f}",
                    delta=f"{opt_rev.get('adr_change_pct', 0):+d}% ADR | Revenue: ${opt_rev.get('projected_revenue', 0):,.2f}"
                )
            with k_col3:
                st.metric(
                    "Profit Maximizing ADR",
                    f"${opt_prof.get('projected_adr', 0):,.2f}",
                    delta=f"{opt_prof.get('adr_change_pct', 0):+d}% ADR | Profit: ${opt_prof.get('projected_profit', 0):,.2f}"
                )

            # Dual-curve chart: ADR vs Projected Daily Revenue and Daily Profit
            scenarios = opt_res.get("scenarios", [])
            scen_df = pd.DataFrame(scenarios)
            if not scen_df.empty:
                fig_opt = go.Figure()
                fig_opt.add_trace(go.Scatter(
                    x=scen_df["projected_adr"],
                    y=scen_df["projected_revenue"],
                    mode='lines+markers',
                    name='Projected Daily Revenue ($)',
                    line=dict(color='#1d4e89', width=3),
                    marker=dict(size=8),
                ))
                fig_opt.add_trace(go.Scatter(
                    x=scen_df["projected_adr"],
                    y=scen_df["projected_profit"],
                    mode='lines+markers',
                    name='Projected Daily Profit ($)',
                    line=dict(color='#2a9d8f', width=3, dash='dash'),
                    marker=dict(size=8),
                ))
                # Add vertical marker for optimal revenue
                fig_opt.add_vline(x=opt_rev.get('projected_adr', input_base_adr), line_dash="dot", line_color="#e76f51", annotation_text="Max Rev ADR")

                fig_opt.update_layout(
                    title="Revenue & Profit Response across +/- 30% ADR Variations",
                    xaxis_title="Average Daily Rate ($)",
                    yaxis_title="Projected Amount ($)",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_opt, use_container_width=True)

                st.subheader("📊 Scenario Simulation Matrix")
                display_df = scen_df[[
                    "adr_change_pct", "projected_adr", "projected_occupancy",
                    "rooms_sold", "projected_revenue", "total_costs", "projected_profit",
                    "revenue_lift", "profit_lift"
                ]].copy()
                display_df.columns = [
                    "ADR Change (%)", "ADR ($)", "Occupancy",
                    "Rooms Sold", "Revenue ($)", "Total Costs ($)", "Profit ($)",
                    "Revenue Lift ($)", "Profit Lift ($)"
                ]
                st.dataframe(display_df, use_container_width=True)

                st.info("💡 **Simulated vs Observed Notice:** Scenarios display simulated sensitivity projections under a constant-elasticity demand model. In real hospitality markets, boundary recommendations (+30%) should be tested incrementally to observe actual customer price sensitivity.")

                with st.expander("ℹ️ Microeconomic Assumptions, Elasticity & Boundary Dynamics", expanded=True):
                    for asm in opt_res.get("assumptions_and_constraints", []):
                        st.markdown(f"- {asm}")
        else:
            st.error(f"Failed to run pricing optimization: {opt_err}")

# ==========================================
# 6.1. ROOM DEMAND FORECASTING TAB (Phase 4)
# ==========================================
with tab_demand:
    st.subheader("📈 Hotel Room Demand & Night Occupancy Forecasting")
    st.write(
        "Forecast daily room demand, cumulative room nights, and expected occupancy across 7, 30, and 90-day horizons. "
        "Evaluated via chronological backtesting against Moving-Average and Seasonal-Naive baselines."
    )

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service on port 8000 to generate demand forecasts.")
    else:
        d_ctrl1, d_ctrl2 = st.columns([1, 2])
        with d_ctrl1:
            demand_horizon = st.radio("Forecast Horizon", [7, 30, 90], index=1, horizontal=True, key="demand_horizon_choice")
        with d_ctrl2:
            st.caption("Forecasting incorporates day-of-week seasonality, lagged operational momentum, and rolling occupancy trends.")

        ok_dem, dem_data, dem_err = api_client.get_demand_forecast(horizon=demand_horizon)
        if ok_dem and dem_data:
            tot_nights = dem_data.get("total_expected_room_nights", 0)
            avg_demand = dem_data.get("average_daily_room_demand", 0)
            avg_occ = dem_data.get("projected_average_occupancy_pct", 0)
            bench = dem_data.get("benchmark_evaluation", {})
            ml_metrics = bench.get("metrics", {}).get("ml_model", {})
            forecast_points = dem_data.get("forecast", [])

            # KPI Highlights
            dk1, dk2, dk3, dk4 = st.columns(4)
            with dk1:
                st.metric("Avg Daily Room Demand", f"{avg_demand:,.0f} rms/day")
            with dk2:
                st.metric("Cumulative Room Nights", f"{tot_nights:,.0f} r-nts")
            with dk3:
                st.metric("Avg Projected Occupancy", f"{avg_occ:.1f}%")
            with dk4:
                ml_rmse = ml_metrics.get("rmse", 0)
                st.metric("ML Backtest RMSE", f"{ml_rmse:.2f}", help="HistGradientBoosting holdout RMSE")

            # Chart: Demand Forecast with Bounds
            dem_df = pd.DataFrame(forecast_points)
            if not dem_df.empty:
                fig_dem = go.Figure()
                fig_dem.add_trace(go.Scatter(
                    x=dem_df["date"],
                    y=dem_df["upper_bound_rooms"],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_dem.add_trace(go.Scatter(
                    x=dem_df["date"],
                    y=dem_df["lower_bound_rooms"],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(43, 108, 176, 0.15)',
                    name='95% Demand Confidence Bounds',
                ))
                fig_dem.add_trace(go.Scatter(
                    x=dem_df["date"],
                    y=dem_df["expected_room_demand"],
                    mode='lines+markers',
                    line=dict(color='#2b6cb0', width=3),
                    marker=dict(size=6),
                    name='Expected Room Demand (Rooms)',
                    hovertemplate='<b>%{x}</b><br>Demand: %{y:.0f} rooms<extra></extra>'
                ))
                fig_dem.add_trace(go.Scatter(
                    x=dem_df["date"],
                    y=dem_df["expected_occupancy_pct"],
                    mode='lines',
                    yaxis='y2',
                    line=dict(color='#38a169', width=2, dash='dot'),
                    name='Projected Occupancy (%)',
                    hovertemplate='<b>%{x}</b><br>Occupancy: %{y:.1f}%<extra></extra>'
                ))
                fig_dem.update_layout(
                    title=f"{demand_horizon}-Day Room Demand & Occupancy Trajectory",
                    xaxis_title="Date",
                    yaxis=dict(title="Room Demand (Rooms)"),
                    yaxis2=dict(title="Occupancy Rate (%)", overlaying='y', side='right', range=[0, 105]),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_dem, use_container_width=True)

                # Cumulative Room Nights Area Chart
                fig_cum = px.area(
                    dem_df,
                    x="date",
                    y="cumulative_room_nights",
                    title=f"Cumulative Room Nights Delivered over {demand_horizon} Days",
                    labels={"cumulative_room_nights": "Cumulative Room Nights", "date": "Date"},
                    color_discrete_sequence=["#3182ce"]
                )
                st.plotly_chart(fig_cum, use_container_width=True)

                with st.expander("📋 View Daily Demand Breakdown Table"):
                    display_dem = dem_df[[
                        "date", "expected_room_demand", "lower_bound_rooms", "upper_bound_rooms",
                        "expected_room_nights", "cumulative_room_nights", "expected_occupancy_pct"
                    ]].copy()
                    display_dem["expected_occupancy_pct"] = display_dem["expected_occupancy_pct"].round(1).astype(str) + "%"
                    st.dataframe(display_dem, use_container_width=True)

            # Baseline Comparison Table
            with st.expander("🔬 Model Evaluation vs Baselines (Chronological Backtesting)", expanded=True):
                st.write("Validation evaluated strictly on untouched historical holdout test splits to prevent future lookahead leakage:")
                b_metrics = bench.get("metrics", {})
                b_ml = b_metrics.get("ml_model", {})
                b_ma = b_metrics.get("moving_average_7d", {})
                b_sn = b_metrics.get("seasonal_naive_7d", {})
                b_lv = b_metrics.get("last_value", {})

                bench_records = [
                    {"Model / Benchmark": "HistGradientBoosting ML (Demand)", "RMSE": b_ml.get("rmse", 0), "MAE": b_ml.get("mae", 0), "MAPE (%)": b_ml.get("mape", 0)},
                    {"Model / Benchmark": "7-Day Moving Average Baseline", "RMSE": b_ma.get("rmse", 0), "MAE": b_ma.get("mae", 0), "MAPE (%)": b_ma.get("mape", 0)},
                    {"Model / Benchmark": "Seasonal-Naive Baseline (7-Day Lag)", "RMSE": b_sn.get("rmse", 0), "MAE": b_sn.get("mae", 0), "MAPE (%)": b_sn.get("mape", 0)},
                    {"Model / Benchmark": "Last-Value Baseline", "RMSE": b_lv.get("rmse", 0), "MAE": b_lv.get("mae", 0), "MAPE (%)": b_lv.get("mape", 0)},
                ]
                st.table(pd.DataFrame(bench_records))
                for lim in dem_data.get("limitations", []):
                    st.caption(f"• {lim}")
        else:
            st.error(f"Failed to generate demand forecast: {dem_err}")

# ==========================================
# 6.2. PRE-BOOKING ESTIMATOR TAB (Phase 4)
# ==========================================
with tab_prebooking:
    st.subheader("🏷️ Pre-Booking Revenue Estimator (Ex-Ante Pricing Engine)")
    st.write(
        "Estimate the projected financial value of incoming reservation inquiries *before* the stay occurs. "
        "Strictly excludes post-stay realized metrics to ensure zero target leakage."
    )

    st.markdown(
        """
        <div style="background-color: #e6fffa; border-left: 4px solid #319795; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem;">
            <strong>🛡️ Strict Anti-Leakage Protocol Active:</strong><br>
            Post-stay realized metrics (<code>Reserved_Rooms</code>, <code>RevPAR</code>, <code>Actual_Occupancy</code>, <code>Total_Costs</code>) 
            are strictly blacklisted from this pre-booking pipeline.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dataset Audit Banner
    ok_audit, audit_res, _ = api_client.get_prebooking_audit()
    if ok_audit and audit_res:
        with st.expander("🔍 Dataset Booking-Level Audit & PMS Ingestion Status", expanded=False):
            st.markdown(f"**Genuine Booking-Level Records Present:** {audit_res.get('is_genuine_booking_level')}")
            st.markdown(f"**Contains Realized Leakage Fields:** {audit_res.get('contains_realized_leakage_fields')}")
            st.markdown(f"**Identified Realized Fields:** {', '.join(audit_res.get('leakage_fields_identified', []))}")
            st.warning(f"**Missing PMS Transaction Fields:** {', '.join(audit_res.get('missing_required_fields', []))}")
            st.info(f"**Assessment Summary:** {audit_res.get('assessment_summary')}")

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service on port 8000 to enable pre-booking estimates.")
    else:
        with st.form("prebooking_quote_form"):
            st.markdown("#### 1. Stay Schedule & Dates")
            pb_c1, pb_c2, pb_c3 = st.columns(3)
            with pb_c1:
                pb_in = st.date_input("Check-In Date", value=date.today())
            with pb_c2:
                pb_out = st.date_input("Check-Out Date", value=date.fromordinal(date.today().toordinal() + 3))
            with pb_c3:
                pb_book = st.date_input("Booking Creation Date", value=date.today())

            st.markdown("#### 2. Reservation & Channel Attributes")
            pb_c4, pb_c5, pb_c6 = st.columns(3)
            with pb_c4:
                pb_rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=1, step=1)
                pb_guests = st.number_input("Total Guests", min_value=1, max_value=40, value=2, step=1)
            with pb_c5:
                pb_room_type = st.selectbox("Room Category / Tier", ["Standard", "Deluxe", "Suite", "Executive"])
                pb_channel = st.selectbox("Booking Channel", ["Direct", "OTA", "Corporate", "Online"])
            with pb_c6:
                pb_segment = st.selectbox("Market Segment", ["Leisure", "Corporate", "Wholesale", "Government"])
                pb_quoted_adr = st.number_input("Quoted ADR ($ / room / night)", min_value=30.0, max_value=1500.0, value=175.0, step=5.0)

            pb_submit = st.form_submit_button("💡 Calculate Expected Booking Value", use_container_width=True)

        if pb_submit:
            if pb_out <= pb_in:
                st.error("Check-out date must be strictly after check-in date.")
            else:
                ok_pb, pb_data, pb_err = api_client.estimate_prebooking_revenue(
                    check_in_date=pb_in.strftime("%Y-%m-%d"),
                    check_out_date=pb_out.strftime("%Y-%m-%d"),
                    booking_date=pb_book.strftime("%Y-%m-%d"),
                    quoted_adr=float(pb_quoted_adr),
                    rooms_count=int(pb_rooms),
                    guests_count=int(pb_guests),
                    booking_channel=pb_channel,
                    market_segment=pb_segment,
                    room_type=pb_room_type,
                )
                if ok_pb and pb_data:
                    b_det = pb_data.get("booking_details", {})
                    st.success("✅ Ex-Ante Pre-Booking Expected Value Computed")
                    pk1, pk2, pk3, pk4 = st.columns(4)
                    with pk1:
                        st.metric("Estimated Gross Revenue", f"${pb_data.get('estimated_gross_revenue', 0):,.2f}")
                    with pk2:
                        st.metric("Estimated Net Revenue", f"${pb_data.get('estimated_net_revenue', 0):,.2f}")
                    with pk3:
                        st.metric("Length of Stay", f"{b_det.get('length_of_stay_nights', 0)} nights")
                    with pk4:
                        st.metric("Lead Time", f"{b_det.get('lead_time_days', 0)} days")

                    # Breakdown details
                    st.divider()
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        st.markdown("##### 💼 Rate Multipliers & Channel Impact")
                        st.write(f"• **Channel Commission Cost:** ${pb_data.get('estimated_commission_cost', 0):,.2f}")
                        st.write(f"• **Room Tier Selected:** {pb_room_type}")
                        st.write(f"• **Arrival Day of Week:** {b_det.get('checkin_day_of_week')}")
                        st.write(f"• **Weekend Arrival:** {'Yes' if b_det.get('is_weekend_arrival') else 'No'}")
                    with b_col2:
                        st.markdown("##### 🛡️ Governance & Model Status")
                        st.write(f"• **Model Status:** {pb_data.get('model_type')}")
                        st.write(f"• **Trained on Guest-Level PMS:** {'Yes' if pb_data.get('is_trained_on_booking_level_data') else 'No (Ex-Ante Expected Value Model)'}")
                        st.caption(f"Governance Notice: {pb_data.get('data_governance_notice')}")
                        st.info(f"Explanation: {pb_data.get('explanation')}")
                else:
                    st.error(f"Failed to calculate pre-booking estimate: {pb_err}")

# ==========================================
# 6.3. CANCELLATION & SLIPPAGE TAB (Phase 4 & 4.1)
# ==========================================
with tab_cancellation:
    st.subheader("❌ Cancellation Prediction & Revenue Slippage Analytics")
    st.write(
        "Individual guest cancellation risk modeling powered by genuine public research data (Antonio et al., 2019), "
        "paired with operational slippage analytics to support defensive overbooking and cancellation policy adjustments."
    )

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service on port 8000 to inspect cancellation intelligence.")
    else:
        can_mode1, can_mode2 = st.tabs([
            "🤖 Individual Booking Cancellation ML (Public Research Data)",
            "📊 Historical Operational Slippage (Synthetic Demo Data)"
        ])

        with can_mode1:
            st.markdown(
                """
                <div style="background-color: #ebf8ff; border-left: 4px solid #3182ce; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem;">
                    <strong>🏛️ Data Source Governance: PUBLIC RESEARCH DATA</strong><br>
                    Trained and benchmarked on <strong>119,390 genuine hotel booking records</strong> (87,396 deduplicated) from the 
                    <em>Antonio, de Almeida & Nunes (2019) Hotel Booking Demand Dataset</em> (Data in Brief, ScienceDirect). 
                    Chronological validation ensures zero future target leakage.
                </div>
                """,
                unsafe_allow_html=True
            )

            # Section A: Interactive Cancellation Risk Predictor
            st.markdown("#### 1. Real-Time Reservation Cancellation Risk Estimator")
            with st.form("cancellation_predict_form"):
                cp_col1, cp_col2, cp_col3 = st.columns(3)
                with cp_col1:
                    cp_lead = st.slider("Booking Lead Time (Days in Advance)", min_value=0, max_value=365, value=45, step=1,
                                        help="Days elapsed between reservation creation and arrival date.")
                    cp_arr = st.date_input("Arrival Date", value=date.today())
                    cp_adr = st.number_input("Quoted ADR ($ / room / night)", min_value=30.0, max_value=1000.0, value=150.0, step=5.0)
                with cp_col2:
                    cp_channel = st.selectbox("Distribution Channel", ["Direct", "Online TA", "Offline TA/TO", "Corporate", "GDS"])
                    cp_customer = st.selectbox("Customer Segment", ["Transient", "Transient-Party", "Contract", "Group"])
                    cp_deposit = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
                with cp_col3:
                    cp_wknd = st.number_input("Weekend Nights", min_value=0, max_value=14, value=1, step=1)
                    cp_week = st.number_input("Weekday Nights", min_value=0, max_value=30, value=2, step=1)
                    cp_adults = st.number_input("Adult Guests", min_value=1, max_value=10, value=2, step=1)

                with st.expander("⚙️ Additional Booker History & Intent Signals", expanded=False):
                    adv_c1, adv_c2, adv_c3 = st.columns(3)
                    with adv_c1:
                        cp_prev_canc = st.number_input("Prior Cancellations by Booker", min_value=0, max_value=20, value=0, step=1)
                        cp_prev_ok = st.number_input("Prior Completed Stays", min_value=0, max_value=50, value=0, step=1)
                    with adv_c2:
                        cp_changes = st.number_input("Booking Modifications / Changes", min_value=0, max_value=10, value=0, step=1)
                        cp_parking = st.selectbox("Requested Parking Spaces", [0, 1, 2], index=0)
                    with adv_c3:
                        cp_requests = st.slider("Special Requests Count", min_value=0, max_value=5, value=1)
                        cp_hotel_type = st.selectbox("Property Type", ["City Hotel", "Resort Hotel"])

                cp_submit = st.form_submit_button("🔍 Calculate Cancellation Risk", use_container_width=True)

            if cp_submit:
                ch_map = {"Direct": "Direct", "Online TA": "TA/TO", "Offline TA/TO": "TA/TO", "Corporate": "Corporate", "GDS": "GDS"}
                clean_ch = ch_map.get(cp_channel, "Direct")
                seg_map = {"Direct": "Direct", "Online TA": "Online TA", "Offline TA/TO": "Offline TA/TO", "Corporate": "Corporate", "GDS": "Corporate"}
                clean_seg = seg_map.get(cp_channel, "Online TA")

                ok_pred, pred_data, pred_err = api_client.predict_cancellation(
                    lead_time=int(cp_lead),
                    arrival_date=cp_arr.strftime("%Y-%m-%d"),
                    adr=float(cp_adr),
                    booking_channel=clean_ch,
                    customer_type=cp_customer,
                    deposit_type=cp_deposit,
                    stays_in_weekend_nights=int(cp_wknd),
                    stays_in_week_nights=int(cp_week),
                    adults=int(cp_adults),
                    previous_cancellations=int(cp_prev_canc),
                    previous_bookings_not_canceled=int(cp_prev_ok),
                    booking_changes=int(cp_changes),
                    required_car_parking_spaces=int(cp_parking),
                    total_of_special_requests=int(cp_requests),
                    hotel=cp_hotel_type,
                    market_segment=clean_seg
                )
                if ok_pred and pred_data:
                    prob_pct = pred_data.get("cancellation_probability_pct", 0)
                    cal_prob = pred_data.get("calibrated_probability", 0)
                    raw_prob = pred_data.get("raw_probability", 0)
                    cat = pred_data.get("risk_category", "Moderate Risk")
                    badge_color = pred_data.get("risk_color", "#dd6b20")

                    st.markdown("---")
                    res_kpi1, res_kpi2, res_kpi3, res_kpi4 = st.columns(4)
                    with res_kpi1:
                        st.metric(
                            "Calibrated Risk (Sigmoid)",
                            f"{cal_prob*100.0:.1f}%",
                            help="Platt-calibrated probability evaluated strictly on holdout data to eliminate probability distortion.",
                        )
                    with res_kpi2:
                        st.metric(
                            "Raw Model Output",
                            f"{raw_prob*100.0:.1f}%",
                            help="Uncalibrated Random Forest ensemble voting fraction.",
                        )
                    with res_kpi3:
                        st.markdown(
                            f'<div style="background-color: {badge_color}; color: white; padding: 0.6rem 1rem; border-radius: 8px; text-align: center; font-weight: bold; margin-top: 0.5rem;">'
                            f'{cat.upper()}</div>',
                            unsafe_allow_html=True,
                        )
                    with res_kpi4:
                        meta = pred_data.get("model_metadata", {})
                        st.caption(f"**Model:** {meta.get('algorithm')} ({meta.get('version')})")
                        st.caption(f"**Dataset:** {meta.get('training_dataset')}")
                        st.caption(f"**Calibrated:** {'✅ Platt Sigmoid' if meta.get('calibrated') else 'None'}")

                    st.info(f"💡 **Suggested Revenue Action:** {pred_data.get('suggested_operational_action')}")

                    st.markdown("##### 🔍 Identified Contributing Risk Factors")
                    factors = pred_data.get("key_contributing_factors", [])
                    if factors:
                        for f in factors:
                            st.write(f"• {f}")
                    else:
                        st.write("• Booking parameters are aligned with typical historical completion rates.")

                    # Local SHAP Explainability Decomposition
                    st.markdown("##### 🧠 Local SHAP Explainability (TreeExplainer)")
                    req_payload = {
                        "lead_time": int(cp_lead),
                        "arrival_date": cp_arr.strftime("%Y-%m-%d"),
                        "adr": float(cp_adr),
                        "booking_channel": clean_ch,
                        "customer_type": cp_customer,
                        "deposit_type": cp_deposit,
                        "stays_in_weekend_nights": int(cp_wknd),
                        "stays_in_week_nights": int(cp_week),
                        "adults": int(cp_adults),
                        "children": 0,
                        "babies": 0,
                        "is_repeated_guest": 0,
                        "previous_cancellations": int(cp_prev_canc),
                        "previous_bookings_not_canceled": int(cp_prev_ok),
                        "booking_changes": int(cp_changes),
                        "days_in_waiting_list": 0,
                        "required_car_parking_spaces": int(cp_parking),
                        "total_of_special_requests": int(cp_requests),
                        "hotel": cp_hotel_type,
                        "market_segment": clean_seg,
                        "reserved_room_type": "A",
                        "meal": "BB",
                    }
                    ok_exp, exp_data, exp_err = api_client.explain_cancellation(req_payload)
                    if ok_exp and exp_data:
                        base_v = exp_data.get("base_value", 0.5)
                        shap_s = exp_data.get("shap_sum", 0.0)
                        pred_p = exp_data.get("predicted_probability", 0.5)
                        is_cons = exp_data.get("is_additively_consistent", True)

                        st.markdown(
                            f"**Additive Consistency Check:** Baseline Expected Risk: **{base_v*100:.1f}%** | "
                            f"Total SHAP Contribution: **{shap_s*100:+.1f}%** | "
                            f"Model Predicted Probability: **{pred_p*100:.1f}%** "
                            f"({'✅ Mathematically Consistent' if is_cons else '⚠️ Delta Detected'})"
                        )

                        attrs = exp_data.get("all_attributions", [])
                        if attrs:
                            top_attrs = attrs[:10]
                            feat_names = [a["feature"] for a in reversed(top_attrs)]
                            feat_vals = [a["shap_value"] * 100.0 for a in reversed(top_attrs)]
                            bar_colors = ["#e53e3e" if v > 0 else "#38a169" for v in feat_vals]

                            fig_shap = go.Figure(go.Bar(
                                x=feat_vals,
                                y=feat_names,
                                orientation="h",
                                marker=dict(color=bar_colors),
                                text=[f"{v:+.1f}%" for v in feat_vals],
                                textposition="outside",
                            ))
                            fig_shap.update_layout(
                                title="Top Feature Attributions for this Reservation (SHAP Values in % Probability)",
                                xaxis_title="Impact on Cancellation Risk (Percentage Points)",
                                yaxis_title="Feature",
                                height=360,
                                margin=dict(l=20, r=40, t=40, b=30),
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)

                        sh_c1, sh_c2 = st.columns(2)
                        with sh_c1:
                            st.markdown("<strong>🔴 Key Risk-Elevating Factors:</strong>", unsafe_allow_html=True)
                            for d in exp_data.get("top_risk_drivers", []):
                                st.write(f"• **{d['feature']}**: +{d['shap_value']*100:.1f}% pts")
                        with sh_c2:
                            st.markdown("<strong>🟢 Key Protective Factors:</strong>", unsafe_allow_html=True)
                            for p in exp_data.get("top_protective_factors", []):
                                st.write(f"• **{p['feature']}**: {p['shap_value']*100:.1f}% pts")

                        st.caption(f"ℹ️ {exp_data.get('statistical_disclaimer')}")

                    # Counterfactual What-If Policy Simulator Expander
                    with st.expander("🔄 Counterfactual Policy & Commercial What-If Simulator", expanded=False):
                        st.markdown(
                            "Simulate how predicted cancellation probability changes under counterfactual operational policies "
                            "(e.g., non-refundable deposit terms, shorter lead times, direct channel acquisition, rate adjustments)."
                        )
                        ok_wi, wi_data, wi_err = api_client.simulate_cancellation_what_if(req_payload)
                        if ok_wi and wi_data:
                            scens = wi_data.get("scenarios", [])
                            wi_df = pd.DataFrame([
                                {
                                    "Lever": s["lever_category"],
                                    "Scenario Policy": s["scenario_name"],
                                    "Parameter": s["modified_parameter"],
                                    "Simulated Value": s["simulated_value"],
                                    "Simulated Risk (%)": f"{s['simulated_probability_pct']:.1f}%",
                                    "Probability Shift": f"{s['probability_delta_pct_pts']:+.1f}% pts",
                                    "Direction": s["impact_badge"] + " " + s["impact_direction"],
                                    "Risk Tier": s["risk_category"],
                                    "Operational Takeaway": s["operational_takeaway"],
                                }
                                for s in scens
                            ])
                            st.dataframe(wi_df[["Lever", "Scenario Policy", "Simulated Risk (%)", "Probability Shift", "Direction", "Operational Takeaway"]], use_container_width=True)

                            # Plot top shifts
                            scen_names = [s["scenario_name"] for s in scens]
                            deltas = [s["probability_delta_pct_pts"] for s in scens]
                            delta_colors = ["#e53e3e" if d > 0 else "#38a169" for d in deltas]

                            fig_wi = go.Figure(go.Bar(
                                x=deltas,
                                y=scen_names,
                                orientation="h",
                                marker=dict(color=delta_colors),
                                text=[f"{d:+.1f}% pts" for d in deltas],
                                textposition="outside",
                            ))
                            fig_wi.update_layout(
                                title="Simulated Cancellation Probability Shift vs Baseline (Percentage Points)",
                                xaxis_title="Risk Delta (pts)",
                                yaxis_title="Scenario Policy",
                                height=480,
                                margin=dict(l=20, r=40, t=40, b=30),
                            )
                            st.plotly_chart(fig_wi, use_container_width=True)
                            st.caption(f"⚠️ {wi_data.get('causal_disclaimer')}")
                        else:
                            st.warning(f"Could not load what-if simulations: {wi_err}")

                    st.caption(f"ℹ️ {pred_data.get('disclaimer')}")
                else:
                    st.error(f"Failed to calculate cancellation probability: {pred_err}")

            # Section B: Multi-Model Benchmark Comparison Table
            st.divider()
            st.markdown("#### 2. Multi-Algorithm Validation Benchmark (Chronological Holdout Split)")
            ok_bench, bench_data, _ = api_client.get_cancellation_model_performance()
            if ok_bench and bench_data:
                algos = bench_data.get("algorithms_benchmarked", {})
                b_rows = []
                for name, metrics in algos.items():
                    b_rows.append({
                        "Algorithm": name,
                        "Val ROC-AUC": metrics.get("val_roc_auc"),
                        "Val PR-AUC": metrics.get("val_pr_auc"),
                        "Val F1-Score": metrics.get("val_f1"),
                        "Val Precision": metrics.get("val_precision"),
                        "Val Recall": metrics.get("val_recall"),
                        "Brier Loss": metrics.get("val_brier_score"),
                        "Train Time (s)": metrics.get("train_time_sec"),
                    })
                bench_df = pd.DataFrame(b_rows)
                st.dataframe(bench_df, use_container_width=True)

                test_eval = bench_data.get("selected_model_test_evaluation", {})
                best_name = bench_data.get("best_model", "Random Forest")
                st.success(
                    f"🏆 **Selected Production Model:** **{best_name}** | "
                    f"Test ROC-AUC: **{test_eval.get('test_roc_auc', 0):.4f}** | "
                    f"Test Recall: **{test_eval.get('test_recall', 0)*100:.1f}%** | "
                    f"Test F1: **{test_eval.get('test_f1', 0):.4f}**"
                )

            # Section C: Probability Calibration & Reliability Curves (Phase 4.2)
            st.divider()
            st.markdown("#### 3. Probability Calibration & Reliability Assessment")
            ok_cal, cal_data, cal_err = api_client.get_cancellation_calibration()
            if ok_cal and cal_data:
                test_cal = cal_data.get("test_split_holdout", {})
                raw_m = test_cal.get("raw_model", {})
                sig_m = test_cal.get("sigmoid_calibrated", {})
                iso_m = test_cal.get("isotonic_calibrated", {})

                cal_kpi1, cal_kpi2, cal_kpi3, cal_kpi4 = st.columns(4)
                with cal_kpi1:
                    st.metric(
                        "Holdout Expected Calibration Error (ECE)",
                        f"{sig_m.get('ece', 0)*100:.1f}%",
                        delta=f"{(sig_m.get('ece', 0) - raw_m.get('ece', 0))*100:.1f}% vs Raw",
                        delta_color="inverse",
                        help="Expected Calibration Error across 10 probability bins on untouched holdout data.",
                    )
                with cal_kpi2:
                    st.metric(
                        "Holdout Brier Loss Score",
                        f"{sig_m.get('brier_score', 0):.4f}",
                        delta=f"{sig_m.get('brier_score', 0) - raw_m.get('brier_score', 0):.4f} vs Raw",
                        delta_color="inverse",
                        help="Mean squared error of probabilistic predictions (lower is better).",
                    )
                with cal_kpi3:
                    st.metric(
                        "Holdout Discrimination (ROC-AUC)",
                        f"{sig_m.get('roc_auc', 0):.4f}",
                        delta="0.0000 (Preserved)",
                        help="Ranking discrimination is 100% preserved under Platt monotonic transformation.",
                    )
                with cal_kpi4:
                    st.metric(
                        "Calibrator Fitting Split",
                        "Validation Split (N=13,109)",
                        help="Strict chronological anti-leakage guarantee: calibrator never touched the final test split.",
                    )

                # Reliability Diagram
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    name="Perfect Calibration (Ideal)",
                    line=dict(color="#a0aec0", dash="dash"),
                ))

                # Raw Curve
                raw_curve = raw_m.get("curve", {})
                if raw_curve.get("mean_predicted_probabilities"):
                    fig_cal.add_trace(go.Scatter(
                        x=raw_curve["mean_predicted_probabilities"],
                        y=raw_curve["fraction_of_positives"],
                        mode="lines+markers",
                        name=f"Raw Random Forest (ECE: {raw_m.get('ece', 0)*100:.1f}%)",
                        line=dict(color="#3182ce", width=2),
                    ))

                # Sigmoid Curve
                sig_curve = sig_m.get("curve", {})
                if sig_curve.get("mean_predicted_probabilities"):
                    fig_cal.add_trace(go.Scatter(
                        x=sig_curve["mean_predicted_probabilities"],
                        y=sig_curve["fraction_of_positives"],
                        mode="lines+markers",
                        name=f"Sigmoid Platt Calibrated (ECE: {sig_m.get('ece', 0)*100:.1f}%)",
                        line=dict(color="#38a169", width=3),
                    ))

                # Isotonic Curve
                iso_curve = iso_m.get("curve", {})
                if iso_curve.get("mean_predicted_probabilities"):
                    fig_cal.add_trace(go.Scatter(
                        x=iso_curve["mean_predicted_probabilities"],
                        y=iso_curve["fraction_of_positives"],
                        mode="lines+markers",
                        name=f"Isotonic Calibrated (ECE: {iso_m.get('ece', 0)*100:.1f}%)",
                        line=dict(color="#dd6b20", width=2, dash="dot"),
                    ))

                fig_cal.update_layout(
                    title="Reliability Curves (Calibration Curves on Holdout Test Split N=13,110)",
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Empirical Fraction of Positives (Actual Cancellation Rate)",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    height=420,
                    margin=dict(l=20, r=20, t=40, b=30),
                )
                st.plotly_chart(fig_cal, use_container_width=True)
                st.caption(
                    "🛡️ **Leakage Prevention Audit:** Calibrators fitted strictly on chronological Validation split "
                    "(13,109 samples). Holdout Test split (13,110 samples) remained completely untouched until final evaluation."
                )

            # Section D: Operational Decision Threshold & Business Cost Optimizer
            st.divider()
            st.markdown("#### 4. Operational Decision Threshold & Spoilage Cost Optimizer")
            st.markdown(
                "Standard classification defaults to 0.50 threshold, which assumes equal cost for false alarms and missed cancellations. "
                "In hotel revenue management, uncaptured cancellations ($C_{FN}$ - empty room night) are far more damaging than defensive customer follow-ups ($C_{FP}$)."
            )

            th_col1, th_col2, th_col3 = st.columns(3)
            with th_col1:
                ui_cost_fp = st.slider("Cost of False Positive ($C_{FP}$)", min_value=10.0, max_value=150.0, value=45.0, step=5.0,
                                       help="Operational cost of defensive verification inquiries or soft rebooking hold.")
            with th_col2:
                ui_cost_fn = st.slider("Gross Cost of Missed Cancellation ($C_{FN}$)", min_value=50.0, max_value=400.0, value=160.0, step=10.0,
                                       help="Gross lost ADR when an unpredicted cancellation occurs and room sits empty.")
            with th_col3:
                ui_repl = st.slider("Room Re-Booking / Replacement Rate (%)", min_value=0, max_value=80, value=25, step=5,
                                    help="Probability of reselling the canceled room before check-in.")

            ok_th, th_data, th_err = api_client.get_cancellation_threshold_analysis(
                cost_fp=float(ui_cost_fp),
                cost_fn=float(ui_cost_fn),
                replacement_rate=float(ui_repl) / 100.0,
                split="val",
            )
            if ok_th and th_data:
                impact = th_data.get("business_impact", {})
                opt_m = th_data.get("optimal_metrics", {})
                def_m = th_data.get("default_0_50_metrics", {})
                opt_th = th_data.get("optimal_threshold", 0.40)

                th_k1, th_k2, th_k3, th_k4 = st.columns(4)
                with th_k1:
                    st.metric("Cost-Optimal Threshold", f"{opt_th:.2f}", delta=f"{opt_th - 0.50:.2f} vs 0.50")
                with th_k2:
                    st.metric("Total Spoilage Cost Savings", f"${impact.get('net_cost_savings_dollars', 0):,.2f}", delta=f"-{impact.get('cost_reduction_pct', 0):.1f}%")
                with th_k3:
                    st.metric("Uncaptured Cancellations Avoided", f"{impact.get('uncaptured_cancellations_avoided', 0):,}", help="Fewer empty rooms on arrival night.")
                with th_k4:
                    st.metric("Cancellation Recall Rate", f"{opt_m.get('recall', 0)*100:.1f}%", delta=f"+{impact.get('recall_improvement_pct_pts', 0):.1f}% pts")

                st.success(f"📈 **Revenue Management Recommendation:** {impact.get('strategic_recommendation')}")

                # Plot Cost Curve vs Threshold
                grid_data = th_data.get("threshold_grid", [])
                if grid_data:
                    th_vals = [g["threshold"] for g in grid_data]
                    tot_costs = [g["total_cost"] for g in grid_data]
                    recalls = [g["recall"] * 100.0 for g in grid_data]

                    fig_th = go.Figure()
                    fig_th.add_trace(go.Scatter(
                        x=th_vals,
                        y=tot_costs,
                        mode="lines+markers",
                        name="Total Operating Error Cost ($)",
                        line=dict(color="#e53e3e", width=3),
                    ))
                    # Optimal Line
                    fig_th.add_vline(x=opt_th, line_width=2, line_dash="dash", line_color="#38a169", annotation_text=f"Optimal: {opt_th:.2f}")
                    fig_th.add_vline(x=0.50, line_width=1, line_dash="dot", line_color="#a0aec0", annotation_text="Default: 0.50")

                    fig_th.update_layout(
                        title=f"Total Error Cost vs Decision Threshold (Min Cost: ${opt_m.get('total_cost', 0):,.2f} at Threshold {opt_th:.2f})",
                        xaxis_title="Decision Threshold (t)",
                        yaxis_title="Total Operating Error Cost ($)",
                        height=380,
                        margin=dict(l=20, r=20, t=40, b=30),
                    )
                    st.plotly_chart(fig_th, use_container_width=True)

                    # Performance comparison table
                    comp_df = pd.DataFrame([
                        {
                            "Decision Boundary": "Default (0.50)",
                            "Threshold": 0.50,
                            "Recall": f"{def_m.get('recall', 0)*100:.1f}%",
                            "Precision": f"{def_m.get('precision', 0)*100:.1f}%",
                            "Missed Cancellations (FN)": f"{def_m.get('fn', 0):,}",
                            "False Alarms (FP)": f"{def_m.get('fp', 0):,}",
                            "Total Error Cost": f"${def_m.get('total_cost', 0):,.2f}",
                            "Avg Cost / Booking": f"${def_m.get('avg_cost_per_booking', 0):.2f}",
                        },
                        {
                            "Decision Boundary": f"Cost-Optimal ({opt_th:.2f})",
                            "Threshold": opt_th,
                            "Recall": f"{opt_m.get('recall', 0)*100:.1f}%",
                            "Precision": f"{opt_m.get('precision', 0)*100:.1f}%",
                            "Missed Cancellations (FN)": f"{opt_m.get('fn', 0):,}",
                            "False Alarms (FP)": f"{opt_m.get('fp', 0):,}",
                            "Total Error Cost": f"${opt_m.get('total_cost', 0):,.2f}",
                            "Avg Cost / Booking": f"${opt_m.get('avg_cost_per_booking', 0):.2f}",
                        },
                    ])
                    st.dataframe(comp_df, use_container_width=True)

            # Section E: Empirical Segment Risk Dynamics (Live from Antonio et al., 2019 dataset)
            st.divider()
            st.markdown("#### 5. Empirical Cancellation Risk by Segment (Antonio et al., 2019 Research Data)")
            ok_seg, seg_data, seg_err = api_client.get_cancellation_segment_analysis()
            if ok_seg and seg_data:
                segs = seg_data.get("segments", {})

                seg_tab1, seg_tab2, seg_tab3, seg_tab4 = st.tabs([
                    "⏱️ Lead-Time Tiers",
                    "🌐 Distribution Channels",
                    "👥 Customer Segments",
                    "📅 Seasonality (Arrival Months)",
                ])

                with seg_tab1:
                    lt_list = segs.get("by_lead_time_tier", [])
                    if lt_list:
                        lt_df = pd.DataFrame(lt_list)
                        fig_lt = px.bar(
                            lt_df,
                            x="lead_time_tier",
                            y="cancellation_rate_pct",
                            text="cancellation_rate_pct",
                            title="Empirical Cancellation Rate by Booking Lead Time Tier (%)",
                            labels={"lead_time_tier": "Lead Time Tier", "cancellation_rate_pct": "Cancellation Rate (%)"},
                            color="cancellation_rate_pct",
                            color_continuous_scale="Reds",
                        )
                        fig_lt.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        st.plotly_chart(fig_lt, use_container_width=True)
                        st.dataframe(lt_df, use_container_width=True)

                with seg_tab2:
                    ch_list = segs.get("by_distribution_channel", [])
                    if ch_list:
                        ch_df = pd.DataFrame(ch_list)
                        fig_ch = px.bar(
                            ch_df,
                            x="distribution_channel",
                            y="cancellation_rate_pct",
                            text="cancellation_rate_pct",
                            title="Empirical Cancellation Rate by Distribution Channel (%)",
                            labels={"distribution_channel": "Distribution Channel", "cancellation_rate_pct": "Cancellation Rate (%)"},
                            color="cancellation_rate_pct",
                            color_continuous_scale="Oranges",
                        )
                        fig_ch.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        st.plotly_chart(fig_ch, use_container_width=True)
                        st.dataframe(ch_df, use_container_width=True)

                with seg_tab3:
                    cust_list = segs.get("by_customer_type", [])
                    if cust_list:
                        cust_df = pd.DataFrame(cust_list)
                        fig_cust = px.bar(
                            cust_df,
                            x="customer_type",
                            y="cancellation_rate_pct",
                            text="cancellation_rate_pct",
                            title="Empirical Cancellation Rate by Customer Type (%)",
                            labels={"customer_type": "Customer Segment", "cancellation_rate_pct": "Cancellation Rate (%)"},
                            color="cancellation_rate_pct",
                            color_continuous_scale="Purples",
                        )
                        fig_cust.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        st.plotly_chart(fig_cust, use_container_width=True)
                        st.dataframe(cust_df, use_container_width=True)

                with seg_tab4:
                    mo_list = segs.get("by_arrival_month", [])
                    if mo_list:
                        mo_df = pd.DataFrame(mo_list)
                        fig_mo = px.line(
                            mo_df,
                            x="arrival_date_month",
                            y="cancellation_rate_pct",
                            markers=True,
                            title="Empirical Monthly Cancellation Rate Fluctuation (%)",
                            labels={"arrival_date_month": "Arrival Month", "cancellation_rate_pct": "Cancellation Rate (%)"},
                        )
                        fig_mo.update_traces(line_color="#e53e3e", line_width=3)
                        st.plotly_chart(fig_mo, use_container_width=True)
                        st.dataframe(mo_df, use_container_width=True)
            else:
                st.warning(f"Could not load segment risk analytics: {seg_err}")


        with can_mode2:
            st.markdown(
                """
                <div style="background-color: #fffaf0; border-left: 4px solid #dd6b20; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem;">
                    <strong>⚠️ Data Source Governance: SYNTHETIC DEMONSTRATION DATA</strong><br>
                    Derived from <code>Hotel book.csv</code> (109 daily aggregate rows). 
                    The 985 bookings represent the column sum of daily booking counts.
                </div>
                """,
                unsafe_allow_html=True
            )
            # Preserved Phase 4 daily operational slippage charts
            ok_can, can_data, can_err = api_client.get_cancellation_analytics()
            if ok_can and can_data:
                c_sum = can_data.get("summary", {})
                c_chan = can_data.get("by_channel", [])

                ck1, ck2, ck3, ck4 = st.columns(4)
                with ck1:
                    st.metric("Total Observed Bookings", f"{c_sum.get('total_bookings', 0):,}")
                with ck2:
                    st.metric("Overall Cancellation Rate", f"{c_sum.get('overall_cancellation_rate_pct', 0):.1f}%")
                with ck3:
                    st.metric("Overall No-Show Rate", f"{c_sum.get('overall_no_show_rate_pct', 0):.1f}%")
                with ck4:
                    st.metric("Net Revenue Slippage", f"{c_sum.get('combined_revenue_slippage_pct', 0):.1f}%")

                if c_chan:
                    ch_df = pd.DataFrame(c_chan)
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        fig_ch_bar = px.bar(
                            ch_df,
                            x="booking_channel",
                            y=["cancellation_rate_pct", "no_show_rate_pct"],
                            barmode="group",
                            title="Cancellation vs No-Show Rates by Distribution Channel (%)",
                            labels={"booking_channel": "Channel", "value": "Rate (%)", "variable": "Metric"},
                            color_discrete_sequence=["#e53e3e", "#dd6b20"]
                        )
                        st.plotly_chart(fig_ch_bar, use_container_width=True)

                    with col_chart2:
                        fig_slip = px.bar(
                            ch_df,
                            x="booking_channel",
                            y="combined_slippage_pct",
                            title="Combined Slippage Exposure (%) by Booking Channel",
                            labels={"booking_channel": "Channel", "combined_slippage_pct": "Slippage Rate (%)"},
                            color="combined_slippage_pct",
                            color_continuous_scale="Reds"
                        )
                        st.plotly_chart(fig_slip, use_container_width=True)

                    st.subheader("📋 Distribution Channel Attrition Breakdown")
                    st.dataframe(ch_df, use_container_width=True)
            else:
                st.error(f"Failed to retrieve cancellation analytics: {can_err}")

# ==========================================
# 6.3.1. ROOM INVENTORY & OVERBOOKING TAB (Phase 5)
# ==========================================
with tab_overbooking:
    st.subheader("🏨 Room Inventory Management & Intelligent Overbooking Optimization")
    st.write(
        "Combines reservation-level calibrated cancellation probabilities with Poisson-Binomial arrival distributions "
        "to calculate safe additional booking limits, minimize room spoilage, and strictly constrain guest walk risk."
    )

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service on port 8000 to manage inventory and overbooking.")
    else:
        st.markdown(
            """
            <div style="background-color: #ebf8ff; border-left: 4px solid #3182ce; padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem;">
                <strong>🛡️ Data Governance Notice: DEMONSTRATION INVENTORY CONFIGURATION</strong><br>
                Physical room counts (200 rooms across 4 tiers) and confirmed bookings are structured demonstration assets 
                (<code>DEMO_HOTEL_INVENTORY</code>). All arrival probability distributions and net contribution optimizations 
                are computed live by FastAPI from calibrated cancellation models.
            </div>
            """,
            unsafe_allow_html=True,
        )

        ob_tab1, ob_tab2, ob_tab3, ob_tab4 = st.tabs([
            "📊 Inventory & Occupancy Calendar",
            "🎯 Overbooking Optimizer",
            "🔄 Custom Policy Simulator",
            "📋 PMS Integration & Schema Audit",
        ])

        with ob_tab1:
            st.markdown("#### 1. Real-Time Room Category Availability & Capacity")
            io_c1, io_c2 = st.columns([2, 2])
            with io_c1:
                ui_target_date = st.date_input(
                    "Target Stay Date",
                    value=date.today() + timedelta(days=7),
                    help="Target arrival or stay date to evaluate room inventory and capacity.",
                    key="inv_target_date",
                )
            with io_c2:
                ui_cat_filter = st.selectbox(
                    "Room Category Filter",
                    ["All Categories", "Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"],
                    index=0,
                    key="inv_cat_filter",
                )

            filter_arg = None if ui_cat_filter == "All Categories" else ui_cat_filter
            ok_avail, avail_data, avail_err = api_client.get_inventory_availability(
                target_date=ui_target_date.strftime("%Y-%m-%d"),
                room_category=filter_arg,
            )

            if ok_avail and avail_data:
                summ = avail_data.get("summary", {})
                ak1, ak2, ak3, ak4, ak5 = st.columns(5)
                with ak1:
                    st.metric("Total Physical Rooms", f"{summ.get('total_physical_rooms', 0):,}")
                with ak2:
                    st.metric("Out of Service (OOS)", f"{summ.get('rooms_out_of_service', 0):,}", help="Rooms under maintenance")
                with ak3:
                    st.metric("Effective Capacity", f"{summ.get('total_effective_capacity', 0):,}")
                with ak4:
                    st.metric("Confirmed Reservations", f"{summ.get('total_confirmed_reservations', 0):,}")
                with ak5:
                    st.metric("Occupancy Rate", f"{summ.get('overall_occupancy_rate_pct', 0):.1f}%")

                st.markdown("##### 📋 Inventory Breakdown by Room Category")
                cats_list = avail_data.get("category_breakdown", [])
                if cats_list:
                    cat_df = pd.DataFrame(cats_list)
                    st.dataframe(
                        cat_df[[
                            "room_category", "total_physical_rooms", "rooms_out_of_service",
                            "effective_capacity", "confirmed_reservations", "available_rooms",
                            "occupancy_rate_pct", "base_adr", "walk_relocation_cost"
                        ]],
                        use_container_width=True,
                    )

            st.divider()
            st.markdown("#### 2. Multi-Day Occupancy & Capacity Timeline")
            ok_occ, occ_data, occ_err = api_client.get_occupancy_timeline(
                start_date=(ui_target_date - timedelta(days=3)).strftime("%Y-%m-%d"),
                end_date=(ui_target_date + timedelta(days=10)).strftime("%Y-%m-%d"),
                room_category=filter_arg,
            )
            if ok_occ and occ_data:
                occ_timeline = occ_data.get("timeline", [])
                if occ_timeline:
                    occ_df = pd.DataFrame(occ_timeline)
                    fig_occ = px.bar(
                        occ_df,
                        x="date",
                        y="confirmed_reservations",
                        color="occupancy_rate_pct",
                        color_continuous_scale="Blues",
                        title=f"Confirmed Reservations vs Effective Capacity ({ui_cat_filter})",
                        labels={"date": "Date", "confirmed_reservations": "Confirmed Bookings", "occupancy_rate_pct": "Occupancy (%)"},
                        text="confirmed_reservations",
                    )
                    # Add capacity line
                    eff_val = occ_timeline[0]["effective_capacity"]
                    fig_occ.add_hline(y=eff_val, line_dash="dash", line_color="#e53e3e", annotation_text=f"Effective Capacity ({eff_val})")
                    fig_occ.update_traces(textposition="outside")
                    fig_occ.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=30))
                    st.plotly_chart(fig_occ, use_container_width=True)

            st.divider()
            st.markdown("#### 3. Active Confirmed Reservations on Target Date")
            ok_res, res_data, res_err = api_client.get_active_reservations(
                target_date=ui_target_date.strftime("%Y-%m-%d"),
                room_category=filter_arg,
            )
            if ok_res and res_data:
                res_list = res_data.get("reservations", [])
                st.caption(f"Showing **{len(res_list)}** active confirmed reservations with calibrated cancellation probabilities.")
                if res_list:
                    r_sample_df = pd.DataFrame([
                        {
                            "Booking ID": r.get("booking_id"),
                            "Guest": r.get("guest_name"),
                            "Category": r.get("room_category"),
                            "Arrival": r.get("arrival_date"),
                            "Departure": r.get("departure_date"),
                            "Nights": r.get("stay_nights"),
                            "ADR ($)": f"${r.get('booked_adr', 0):.2f}",
                            "Channel": r.get("booking_channel"),
                            "Deposit": r.get("deposit_type"),
                            "Calibrated Cancel Risk": f"{r.get('calibrated_cancellation_pct', 0):.1f}%",
                        }
                        for r in res_list[:25]
                    ])
                    st.dataframe(r_sample_df, use_container_width=True)

        with ob_tab2:
            st.markdown("#### 1. Constrained Overbooking Revenue Optimization")
            opt_c1, opt_c2, opt_c3 = st.columns(3)
            with opt_c1:
                ob_target_date = st.date_input(
                    "Arrival Date",
                    value=date.today() + timedelta(days=7),
                    key="ob_opt_date",
                )
            with opt_c2:
                ob_category = st.selectbox(
                    "Target Room Category",
                    ["Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"],
                    index=0,
                    key="ob_opt_cat",
                )
            with opt_c3:
                ob_walk_risk = st.slider(
                    "Max Acceptable Walk Risk Tolerance (α)",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5,
                    help="Hard service-level risk threshold: probability of arriving guests exceeding room capacity must not exceed α.",
                    key="ob_walk_risk_slider",
                )

            ok_opt, opt_res, opt_err = api_client.optimize_overbooking(
                target_date=ob_target_date.strftime("%Y-%m-%d"),
                room_category=ob_category,
                max_walk_risk_pct=float(ob_walk_risk),
            )

            if ok_opt and opt_res:
                rec = opt_res.get("recommendation", {})
                impact = opt_res.get("business_impact_vs_no_overbooking", {})
                assump = opt_res.get("assumptions", {})

                rk1, rk2, rk3, rk4 = st.columns(4)
                with rk1:
                    st.metric(
                        "Safe Additional Bookings",
                        f"+{rec.get('safe_additional_bookings', 0)} Rooms",
                        help="Additional bookings hotel can accept without violating walk risk constraint.",
                    )
                with rk2:
                    st.metric(
                        "Recommended Booking Limit",
                        f"{rec.get('recommended_total_booking_limit', 0)} Bookings",
                        delta=f"Capacity: {assump.get('effective_capacity', 0)}",
                    )
                with rk3:
                    st.metric(
                        "Walk Probability",
                        f"{rec.get('walk_probability_pct', 0):.2f}%",
                        delta=f"Limit: {ob_walk_risk:.1f}%",
                        delta_color="normal" if rec.get('walk_probability_pct', 0) <= ob_walk_risk else "inverse",
                    )
                with rk4:
                    st.metric(
                        "Expected Net Revenue Gain",
                        f"${impact.get('net_revenue_gain_dollars', 0):,.2f}",
                        delta="After Relocation Costs",
                    )

                st.success(f"📈 **Optimization Recommendation:** {impact.get('strategic_summary')}")

                # Two Visualizations: Expected Arrivals PMF vs Net Contribution Curve
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.markdown("##### 📊 Arrival Probability Distribution (Poisson-Binomial PMF)")
                    ok_fc, fc_data, _ = api_client.get_cancellation_forecast(
                        target_date=ob_target_date.strftime("%Y-%m-%d"),
                        room_category=ob_category,
                    )
                    if ok_fc and fc_data:
                        pmf_list = fc_data.get("arrival_distribution_pmf", [])
                        if pmf_list:
                            pmf_df = pd.DataFrame(pmf_list)
                            pmf_df = pmf_df[pmf_df["probability"] > 0.0001]
                            fig_pmf = px.bar(
                                pmf_df,
                                x="arrivals",
                                y="probability",
                                title=f"Arrival PMF for {ob_category} (E[Arrivals]: {fc_data.get('expected_arrivals', 0):.1f})",
                                labels={"arrivals": "Arriving Guests", "probability": "Probability P(A = k)"},
                            )
                            eff_cap = fc_data.get("effective_capacity", 98)
                            fig_pmf.add_vline(x=eff_cap, line_dash="dash", line_color="#e53e3e", annotation_text=f"Capacity ({eff_cap})")
                            fig_pmf.update_traces(marker_color="#3182ce")
                            fig_pmf.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=30))
                            st.plotly_chart(fig_pmf, use_container_width=True)

                with v_col2:
                    st.markdown("##### 📈 Expected Net Contribution vs Additional Bookings")
                    grid = opt_res.get("candidate_evaluation_grid", [])
                    if grid:
                        grid_df = pd.DataFrame(grid)
                        fig_enc = go.Figure()
                        fig_enc.add_trace(go.Scatter(
                            x=grid_df["additional_bookings"],
                            y=grid_df["expected_net_contribution"],
                            mode="lines+markers",
                            name="Expected Net Contribution ($)",
                            line=dict(color="#38a169", width=3),
                        ))
                        best_add = rec.get("safe_additional_bookings", 0)
                        fig_enc.add_vline(x=best_add, line_dash="dash", line_color="#3182ce", annotation_text=f"Optimal: +{best_add}")
                        fig_enc.update_layout(
                            title=f"Net Contribution Curve (Max: ${rec.get('expected_net_contribution', 0):,.2f})",
                            xaxis_title="Additional Bookings Accepted (b)",
                            yaxis_title="Expected Net Contribution ($)",
                            height=350,
                            margin=dict(l=20, r=20, t=40, b=30),
                        )
                        st.plotly_chart(fig_enc, use_container_width=True)

                st.divider()
                st.markdown("##### 📋 Full Candidate Booking Limit Evaluation Grid")
                if grid:
                    st.dataframe(
                        grid_df[[
                            "additional_bookings", "total_booking_limit", "expected_arrivals",
                            "expected_revenue", "expected_walks", "expected_walk_cost",
                            "expected_net_contribution", "walk_probability_pct", "risk_tier", "constraint_satisfied"
                        ]],
                        use_container_width=True,
                    )

                st.divider()
                st.markdown("#### 2. Portfolio-Wide Synchronized Overbooking Limits")
                ok_port, port_data, _ = api_client.get_portfolio_overbooking_recommendations(target_date=ob_target_date.strftime("%Y-%m-%d"))
                if ok_port and port_data:
                    p_summ = port_data.get("portfolio_summary", {})
                    st.info(
                        f"🏨 **Whole-Hotel Summary on {ob_target_date.strftime('%Y-%m-%d')}:** "
                        f"Safe Additional Bookings: **+{p_summ.get('total_additional_bookings_allowed', 0)} Rooms** | "
                        f"Net Incremental Value: **+${p_summ.get('total_net_expected_revenue_gain', 0):,.2f}**"
                    )
                    p_recs = port_data.get("category_recommendations", [])
                    if p_recs:
                        st.dataframe(pd.DataFrame(p_recs), use_container_width=True)
            else:
                st.error(f"Failed to optimize overbooking: {opt_err}")

        with ob_tab3:
            st.markdown("#### 🔄 Custom Hotel Management Policy & Risk Scenario Simulator")
            st.write(
                "Adjust operational assumptions (rate per night, relocation compensation penalty, risk thresholds) "
                "to evaluate how revenue gains and walk liabilities change."
            )

            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                sim_cat = st.selectbox("Category to Simulate", ["Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"], key="sim_cat")
                sim_date = st.date_input("Simulation Date", value=date.today() + timedelta(days=7), key="sim_date")
            with sim_c2:
                sim_adr = st.number_input("Average Daily Rate ($ / night)", min_value=50.0, max_value=2000.0, value=150.0, step=10.0, key="sim_adr")
                sim_walk_cost = st.number_input("Walk / Relocation Cost ($)", min_value=50.0, max_value=5000.0, value=300.0, step=25.0, help="Alternative room + taxi + compensation voucher", key="sim_walk")
            with sim_c3:
                sim_alpha = st.slider("Risk Tolerance α (%)", min_value=0.2, max_value=15.0, value=1.5, step=0.1, key="sim_alpha")
                sim_max_b = st.slider("Max Search Horizon", min_value=5, max_value=50, value=25, step=5, key="sim_max_b")

            sim_btn = st.button("🚀 Run Scenario Simulation", use_container_width=True)
            if sim_btn:
                ok_sim, sim_res, sim_err = api_client.simulate_overbooking_scenario({
                    "target_date": sim_date.strftime("%Y-%m-%d"),
                    "room_category": sim_cat,
                    "custom_adr": float(sim_adr),
                    "custom_walk_cost": float(sim_walk_cost),
                    "max_walk_risk_pct": float(sim_alpha),
                    "max_additional_bookings": int(sim_max_b),
                })
                if ok_sim and sim_res:
                    s_rec = sim_res.get("recommendation", {})
                    s_base = sim_res.get("no_overbooking_baseline", {})
                    s_imp = sim_res.get("business_impact_vs_no_overbooking", {})

                    st.markdown("---")
                    st.markdown("##### 📊 Scenario Simulation Comparison")
                    sk1, sk2, sk3, sk4 = st.columns(4)
                    with sk1:
                        st.metric("Recommended Additional", f"+{s_rec.get('safe_additional_bookings', 0)} Rooms")
                    with sk2:
                        st.metric("Total Booking Limit", f"{s_rec.get('recommended_total_booking_limit', 0)} Bookings")
                    with sk3:
                        st.metric("Walk Probability", f"{s_rec.get('walk_probability_pct', 0):.2f}%", delta=f"{s_rec.get('risk_tier')}")
                    with sk4:
                        st.metric("Net Incremental Gain", f"${s_imp.get('net_revenue_gain_dollars', 0):,.2f}")

                    st.markdown("##### 🎲 Walk Risk Breakdown")
                    wk_c1, wk_c2, wk_c3, wk_c4 = st.columns(4)
                    with wk_c1:
                        st.metric("P(0 Walks - Safe)", f"{s_rec.get('prob_zero_walks_pct', 98.0):.1f}%")
                    with wk_c2:
                        st.metric("P(Exactly 1 Walk)", f"{s_rec.get('prob_one_walk_pct', 1.8):.1f}%")
                    with wk_c3:
                        st.metric("P(2+ Walks)", f"{s_rec.get('prob_multiple_walks_pct', 0.2):.1f}%")
                    with wk_c4:
                        st.metric("Expected Financial Exposure", f"${s_rec.get('expected_relocation_cost', 0):,.2f}")

                    st.caption(f"ℹ️ {sim_res.get('disclaimer')}")
                else:
                    st.error(f"Failed to simulate scenario: {sim_err}")

        with ob_tab4:
            st.markdown("#### 📋 Property Management System (PMS) Schema & Missing Fields Audit")
            st.write(
                "Audit comparing the current demonstration reservation schema against enterprise PMS platforms "
                "(Oracle Hospitality Opera, Mews, protel) to facilitate genuine production onboarding."
            )
            ok_audit, audit_data, audit_err = api_client.get_missing_fields_audit()
            if ok_audit and audit_data:
                st.markdown("##### ✅ Currently Supported Reservation Fields")
                supported = audit_data.get("active_supported_fields", [])
                st.write(", ".join([f"`{f}`" for f in supported]))

                st.markdown("##### ⚠️ Missing Production PMS Fields Required for Live Integration")
                missing = audit_data.get("missing_production_pms_fields", [])
                if missing:
                    m_df = pd.DataFrame(missing)
                    st.dataframe(m_df, use_container_width=True)

                st.markdown("##### 🔌 Standard Ingestion Interface Specification")
                integ = audit_data.get("integration_interface", {})
                st.json(integ)
            else:
                st.warning(f"Could not load PMS schema audit: {audit_err}")

# ==========================================
# 6.4. COMPETITOR PRICING TAB (Phase 4)
# ==========================================
with tab_competitor:
    st.subheader("🏢 Competitive Rate Benchmarking & Market Price Index (MPI)")
    st.write(
        "Benchmark your daily rates against primary competitive set properties to assess rate elasticity "
        "and market positioning without unauthorized web scraping."
    )

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service on port 8000 to manage competitor intelligence.")
    else:
        comp_tab1, comp_tab2 = st.tabs(["📊 MPI Market Position Calculator", "📝 Record New Comp-Set Rate"])

        with comp_tab1:
            st.markdown("#### Evaluate Price Position Index")
            c_eval1, c_eval2 = st.columns(2)
            with c_eval1:
                eval_our_adr = st.number_input("Our Current ADR ($)", min_value=30.0, max_value=1000.0, value=175.0, step=5.0, key="eval_our_adr")
            with c_eval2:
                eval_comp_adr = st.number_input("Competitor Average ADR ($)", min_value=30.0, max_value=1000.0, value=165.0, step=5.0, key="eval_comp_adr")

            if st.button("📊 Calculate Market Price Index", key="btn_calc_mpi"):
                ok_eval, mpi_data, mpi_err = api_client.evaluate_competitor_pricing(our_adr=eval_our_adr, competitor_adr=eval_comp_adr)
                if ok_eval and mpi_data:
                    act_comp = mpi_data.get("active_comparison", {})
                    mp1, mp2, mp3, mp4 = st.columns(4)
                    with mp1:
                        st.metric("Market Price Index (MPI)", f"{act_comp.get('market_price_index_mpi', 100):.1f}%", help="Our ADR / Comp ADR * 100")
                    with mp2:
                        pos = act_comp.get('position_classification', 'Parity')
                        st.metric("Market Position", pos)
                    with mp3:
                        gap_d = act_comp.get('dollar_gap', 0)
                        st.metric("Rate Gap ($)", f"${gap_d:+,.2f}")
                    with mp4:
                        gap_p = act_comp.get('variance_pct', 0)
                        st.metric("Rate Variance (%)", f"{gap_p:+,.1f}%")

                    st.info(f"💡 **Strategic Position Guidance:** {act_comp.get('suggested_action')}")
                else:
                    st.error(f"Failed to calculate MPI: {mpi_err}")

        with comp_tab2:
            st.markdown("#### Log Authorized Competitor Rate Observation")
            with st.form("add_comp_record_form"):
                cr_c1, cr_c2 = st.columns(2)
                with cr_c1:
                    comp_date = st.date_input("Observation Date", value=date.today())
                    comp_name = st.text_input("Competitor Hotel Name", value="Grand Royal Hotel")
                    comp_room = st.selectbox("Room Tier", ["Standard Deluxe", "Executive Suite", "Ocean View", "City Double"])
                with cr_c2:
                    comp_rate = st.number_input("Competitor Rate ($)", min_value=30.0, max_value=2000.0, value=180.0, step=5.0)
                    our_rate = st.number_input("Our Matching Rate ($)", min_value=30.0, max_value=2000.0, value=175.0, step=5.0)
                    comp_notes = st.text_input("Notes (e.g. includes breakfast, promotional sale)", value="Includes complimentary breakfast")

                submit_comp = st.form_submit_button("💾 Save Competitor Rate Observation", use_container_width=True)

            if submit_comp:
                ok_save, save_res, save_err = api_client.add_competitor_record(
                    date_str=comp_date.strftime("%Y-%m-%d"),
                    competitor_name=comp_name,
                    competitor_adr=float(comp_rate),
                    our_adr=float(our_rate),
                    room_type=comp_room,
                    notes=comp_notes,
                )
                if ok_save:
                    st.success(f"✅ Competitor benchmark for {comp_name} logged successfully! (MPI: {save_res.get('record', {}).get('mpi'):.1f}%)")
                else:
                    st.error(f"Failed to log competitor record: {save_err}")

        # Stored Historical Competitor Benchmarks Table
        st.divider()
        st.subheader("📋 Historical Comp-Set Observations")
        ok_recs, recs_data, _ = api_client.get_competitor_records()
        if ok_recs and recs_data:
            records = recs_data.get("records", []) or recs_data.get("recent_records", [])
            if records:
                rec_df = pd.DataFrame(records)
                st.dataframe(rec_df, use_container_width=True)
            else:
                st.info("No competitor rate records logged yet. Use the form above to record comp-set observations.")

# ==========================================
# 6.5. BUSINESS RECOMMENDATIONS TAB (Phase 4)
# ==========================================
with tab_recommendations:
    st.subheader("💡 Algorithmic Hotel Business Decision Support Engine")
    st.write(
        "Synthesizing multi-model demand forecasts, price elasticity simulations, anomaly alerts, "
        "data drift warnings, and channel slippage metrics into prioritized, high-impact revenue strategies."
    )

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service on port 8000 to generate decision recommendations.")
    else:
        if st.button("🔄 Refresh Business Recommendations", key="btn_refresh_recs"):
            st.rerun()

        ok_rec, rec_data, rec_err = api_client.get_business_recommendations()
        if ok_rec and rec_data:
            prio_summary = rec_data.get("priority_summary", {})
            recs = rec_data.get("recommendations", [])

            # Summary KPIs
            rk1, rk2, rk3, rk4 = st.columns(4)
            with rk1:
                st.metric("Total Strategic Actions", f"{rec_data.get('total_recommendations', 0)}")
            with rk2:
                st.metric("High Priority Decisions", f"{prio_summary.get('High', 0)}")
            with rk3:
                st.metric("Medium Priority", f"{prio_summary.get('Medium', 0)}")
            with rk4:
                st.metric("Synthesis Engine", "Active 🟢")

            st.divider()
            st.markdown("#### 🎯 Prioritized Action Cards")

            for idx, r in enumerate(recs):
                prio = r.get("priority", "Medium")
                badge_color = "#e53e3e" if prio == "High" else ("#dd6b20" if prio == "Medium" else "#3182ce")
                conf_val = r.get("confidence_level", 0.85)

                try:
                    conf_val = float(conf_val)
                except (TypeError, ValueError):
                    conf_val = 0.85

                conf = conf_val * 100 if conf_val <= 1.0 else conf_val

                with st.container():
                    st.markdown(
                        f"""
                        <div style="background: white; border-radius: 12px; border: 1px solid #e2e8f0; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="background: {badge_color}; color: white; padding: 0.2rem 0.6rem; border-radius: 8px; font-weight: bold; font-size: 0.85rem;">
                                    {prio.upper()} PRIORITY
                                </span>
                                <span style="color: #718096; font-size: 0.85rem; font-weight: 500;">
                                    Category: <strong>{r.get('category')}</strong> | Confidence: <strong>{conf:.0f}%</strong>
                                </span>
                            </div>
                            <h4 style="margin: 0.3rem 0; color: #1a202c;">{r.get('title')}</h4>
                            <p style="color: #4a5568; margin-bottom: 0.8rem; font-size: 0.95rem;">{r.get('explanation')}</p>
                            <div style="background: #f7fafc; border-left: 3px solid {badge_color}; padding: 0.6rem 0.8rem; border-radius: 4px;">
                                <strong>Recommended Action:</strong> {r.get('suggested_action')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.caption(f"ℹ️ Engine Disclaimer: {rec_data.get('governance_disclaimer')}")
        else:
            st.error(f"Failed to retrieve business recommendations: {rec_err}")


# ==========================================
# 7. EXPLAINABLE AI TAB (SHAP) (Phase 3)
# ==========================================
with tab_xai:
    st.subheader("🧠 Explainable AI — SHAP Feature Attributions")
    st.write("Understand why the production model makes its predictions using cooperative game theory (SHAP values).")

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service to inspect SHAP attributions.")
    else:
        st.info("ℹ️ SHAP attributions measure statistical feature associations in the trained tree ensemble and do NOT imply direct counterfactual causality.")

        tab_glob, tab_loc = st.tabs(["🌐 Global Feature Importance", "🔍 Single Prediction Explanation"])

        with tab_glob:
            st.write("Mean absolute SHAP value across representative historical booking records:")
            ok_sh_g, glob_shap, err_sh_g = api_client.get_global_shap_importance(max_samples=50)
            if ok_sh_g and glob_shap:
                st.metric("Model Baseline Expectation (E[f(X)])", f"${glob_shap.get('base_value', 0):,.2f}")

                feat_imps = glob_shap.get("feature_importances", [])
                f_df = pd.DataFrame(feat_imps)
                if not f_df.empty:
                    f_df = f_df.sort_values("importance", ascending=True)
                    fig_shap = px.bar(
                        f_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Global SHAP Feature Importance (|Mean SHAP Value|)",
                        labels={"importance": "Mean |SHAP Value| ($ Revenue Impact)", "feature": "Operational Feature"},
                        color="importance",
                        color_continuous_scale="Viridis",
                    )
                    fig_shap.update_layout(showlegend=False, height=500)
                    st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.error(f"Failed to fetch global SHAP importances: {err_sh_g}")

        with tab_loc:
            st.write("Deconstruct how individual operational features push a specific prediction above or below the baseline:")
            with st.form("xai_local_form"):
                x_col1, x_col2, x_col3 = st.columns(3)
                with x_col1:
                    x_date = st.date_input("Date", value=datetime(2026, 7, 15), key="x_date")
                    x_occ = st.slider("Occupancy Rate", 0.05, 1.0, 0.85, 0.01, key="x_occ")
                with x_col2:
                    x_adr = st.number_input("ADR ($)", value=195.0, min_value=30.0, max_value=800.0, step=5.0, key="x_adr")
                    x_avail = st.number_input("Available Rooms", value=120, min_value=1, max_value=1000, key="x_avail")
                with x_col3:
                    x_res = st.number_input("Reserved Rooms", value=102, min_value=0, max_value=1000, key="x_res")
                    x_costs = st.number_input("Total Costs ($)", value=3100.0, min_value=0.0, step=50.0, key="x_costs")

                xai_submit = st.form_submit_button("🔍 Explain This Prediction", use_container_width=True)

            if xai_submit:
                payload = {
                    "Date": x_date.strftime("%Y-%m-%d"),
                    "Occupancy_Rate": float(x_occ),
                    "ADR": float(x_adr),
                    "Available_Rooms": int(x_avail),
                    "Reserved_Rooms": int(x_res),
                    "Total_Costs": float(x_costs),
                }
                ok_x_loc, loc_res, err_x_loc = api_client.explain_instance(payload)
                if ok_x_loc and loc_res:
                    res_kpi1, res_kpi2, res_kpi3 = st.columns(3)
                    with res_kpi1:
                        st.metric("Baseline Model Expectation", f"${loc_res.get('base_value', 0):,.2f}")
                    with res_kpi2:
                        st.metric("Net SHAP Adjustment", f"{loc_res.get('net_shap_adjustment', 0):+,.2f}")
                    with res_kpi3:
                        st.metric("Final Predicted Revenue", f"${loc_res.get('predicted_revenue', 0):,.2f}")

                    # Top drivers
                    pos_col, neg_col = st.columns(2)
                    with pos_col:
                        st.markdown("##### 🟢 Features Increasing Revenue")
                        for p in loc_res.get("top_positive_features", [])[:4]:
                            st.success(f"**{p['feature']}** ({p['feature_value']}): +${p['shap_value']:,.2f}")
                    with neg_col:
                        st.markdown("##### 🔴 Features Decreasing Revenue")
                        for n in loc_res.get("top_negative_features", [])[:4]:
                            st.error(f"**{n['feature']}** ({n['feature_value']}): -${abs(n['shap_value']):,.2f}")

                    # Full contributions bar chart
                    all_c = loc_res.get("all_feature_contributions", [])
                    if all_c:
                        c_df = pd.DataFrame(all_c)
                        c_df["color"] = c_df["shap_value"].apply(lambda v: "#2a9d8f" if v > 0 else "#e76f51")
                        c_df = c_df.sort_values("shap_value", ascending=True)
                        fig_loc = go.Figure(go.Bar(
                            x=c_df["shap_value"],
                            y=c_df["feature"],
                            orientation="h",
                            marker_color=c_df["color"],
                        ))
                        fig_loc.update_layout(
                            title="Feature Contributions to Predicted Revenue ($)",
                            xaxis_title="SHAP Value ($ Difference from Baseline)",
                            yaxis_title="Feature",
                            height=450,
                        )
                        st.plotly_chart(fig_loc, use_container_width=True)
                else:
                    st.error(f"Failed to explain instance: {err_x_loc}")

# ==========================================
# 8. ANOMALY DETECTION TAB (Phase 3)
# ==========================================
with tab_anomalies:
    st.subheader("🚨 Historical Revenue, Occupancy & ADR Anomaly Detection")
    st.write("Scan operational logs to detect statistically abnormal days using combined Interquartile Range (IQR), rolling Z-Scores, and Isolation Forest ML.")

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service to inspect anomalies.")
    else:
        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            contam_slider = st.slider("IsolationForest Contamination Rate", min_value=0.01, max_value=0.15, value=0.05, step=0.01)
        with ctrl2:
            severity_filter = st.selectbox("Filter by Severity", ["All Severities", "High", "Medium", "Low"])

        ok_anom, anom_res, anom_err = api_client.get_anomalies(contamination=contam_slider)
        if ok_anom and anom_res:
            c_sev = anom_res.get("counts_by_severity", {})
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.metric("Total Records Scanned", f"{anom_res.get('total_records_analyzed', 0):,}")
            with a2:
                st.metric("Anomalies Flagged", f"{anom_res.get('anomalies_detected', 0):,} ({anom_res.get('anomaly_rate_pct', 0)}%)")
            with a3:
                st.metric("High Severity Anom.", f"{c_sev.get('High', 0):,}")
            with a4:
                st.metric("Medium Severity Anom.", f"{c_sev.get('Medium', 0):,}")

            anom_list = anom_res.get("anomalies", [])
            if severity_filter != "All Severities":
                anom_list = [a for a in anom_list if a["severity"] == severity_filter]

            anom_df = pd.DataFrame(anom_list)
            if not anom_df.empty:
                # Scatter visualization
                fig_anom = px.scatter(
                    anom_df,
                    x="date",
                    y="actual_value",
                    color="severity",
                    size=anom_df["deviation_pct"].abs(),
                    hover_data=["metric", "actual_value", "expected_value", "deviation_pct", "direction"],
                    title="Detected Operational Anomalies Timeline",
                    color_discrete_map={"High": "#e63946", "Medium": "#f4a261", "Low": "#457b9d"},
                )
                st.plotly_chart(fig_anom, use_container_width=True)

                st.subheader("📋 Anomaly Diagnostics Ledger")
                cols_to_show = ["date", "metric", "actual_value", "expected_value", "deviation_pct", "direction", "severity", "detection_method", "diagnostic_note"]
                cols_avail = [c for c in cols_to_show if c in anom_df.columns]
                st.dataframe(anom_df[cols_avail], use_container_width=True)
            else:
                st.success("No anomalies found matching the selected filter criteria.")
        else:
            st.error(f"Failed to scan anomalies: {anom_err}")

# ==========================================
# 9. DATA DRIFT MONITORING TAB (Phase 3)
# ==========================================
with tab_drift:
    st.subheader("📡 Data Drift & Distributional Stability Monitoring")
    st.write("Track statistical distribution drift between active reference training data and incoming booking distributions using Kolmogorov-Smirnov tests and Categorical Population Stability Index (PSI).")

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service to inspect data drift.")
    else:
        if st.button("🔄 Execute Real-Time Data Drift Audit", key="btn_run_drift", use_container_width=True):
            with st.spinner("Computing Kolmogorov-Smirnov statistics and categorical PSI..."):
                ok_dr, drift_res, err_dr = api_client.assess_data_drift()
                if ok_dr and drift_res:
                    status = drift_res.get("overall_status", "Unknown")
                    if status == "No Drift":
                        st.success(f"✅ **Overall Status: {status}** — {drift_res.get('recommendation')}")
                    elif status == "Moderate Drift":
                        st.warning(f"⚠️ **Overall Status: {status}** — {drift_res.get('recommendation')}")
                    else:
                        st.error(f"🚨 **Overall Status: {status}** — {drift_res.get('recommendation')}")

                    d_kpi1, d_kpi2, d_kpi3 = st.columns(3)
                    with d_kpi1:
                        st.metric("Features Audited", drift_res.get("total_features_tested", 0))
                    with d_kpi2:
                        st.metric("Drifted Features Count", drift_res.get("total_drifted_features", 0))
                    with d_kpi3:
                        st.metric("Reference Dataset Size", f"{drift_res.get('baseline_records_count', 0):,} rows")

                    # Numerical Drift Table
                    st.markdown("##### 🔢 Numerical Features Drift (Kolmogorov-Smirnov & Wasserstein)")
                    num_drift = drift_res.get("numerical_drift", [])
                    if num_drift:
                        nd_df = pd.DataFrame(num_drift)
                        st.dataframe(nd_df, use_container_width=True)

                    # Categorical Drift Table
                    st.markdown("##### 🏷️ Categorical Features Stability (Population Stability Index - PSI)")
                    cat_drift = drift_res.get("categorical_drift", [])
                    if cat_drift:
                        cd_df = pd.DataFrame(cat_drift)
                        st.dataframe(cd_df, use_container_width=True)
                else:
                    st.error(f"Failed to assess data drift: {err_dr}")
        else:
            st.info("Click 'Execute Real-Time Data Drift Audit' to compute statistical drift tests across numerical and categorical hotel features.")

# ==========================================
# 10. MODEL GOVERNANCE TAB (FastAPI Integration)
# ==========================================
with tab_governance:
    st.subheader("🏛️ Production Model Governance & Registry")
    st.write("Audit the active model deployed in production, inspect multi-model benchmark results, and manage zero-downtime version rollbacks.")

    if not is_online:
        st.warning("⚠️ FastAPI backend is offline. Start the service to inspect live model governance.")
    else:
        # Fetch model metadata and comparisons
        ok_info, meta, err_info = api_client.get_model_info()
        ok_comp, comp, err_comp = api_client.get_model_comparison()
        ok_vers, vers_data, err_vers = api_client.get_model_versions()

        if ok_info and meta:
            # High level model KPI metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Active Algorithm", str(meta.get("algorithm", "Unknown")))
            with m_col2:
                st.metric("Active Version", str(meta.get("version", "Unknown")))
            with m_col3:
                st.metric("Feature Dimensions", f"{meta.get('feature_count', 0)} columns")
            with m_col4:
                dep_time = meta.get("deployed_at", "")[:19].replace("T", " ")
                st.metric("Deployed At", dep_time if dep_time else "N/A")

            # Evaluation metrics comparison: Validation vs Test
            st.markdown("#### 🎯 Performance Evaluation Metrics")
            val_m = meta.get("validation_metrics", {})
            test_m = meta.get("test_metrics", {})

            e_col1, e_col2, e_col3, e_col4 = st.columns(4)
            with e_col1:
                st.metric("Validation RMSE", f"${val_m.get('RMSE', 0):,.2f}", help="Primary selection metric")
            with e_col2:
                st.metric("Validation R²", f"{val_m.get('R2', 0):.4f}")
            with e_col3:
                st.metric("Untouched Test RMSE", f"${test_m.get('RMSE', 0):,.2f}")
            with e_col4:
                st.metric("Untouched Test R²", f"{test_m.get('R2', 0):.4f}")

            # 5-Model Benchmark Comparison Table
            st.markdown("#### 🏆 Multi-Model Benchmark Comparison")
            if ok_comp and comp.get("models"):
                comp_df = pd.DataFrame(comp["models"])
                # Clean columns and order
                cols_order = [c for c in ["model", "RMSE", "MAE", "R2", "MAPE", "fit_time_seconds", "status"] if c in comp_df.columns]
                comp_df = comp_df[cols_order].sort_values("RMSE", ascending=True)

                st.dataframe(comp_df, use_container_width=True)

                # Comparison Bar Chart
                fig_comp = px.bar(
                    comp_df,
                    x="model",
                    y="RMSE",
                    color="model",
                    title="Validation RMSE Across All 5 Algorithms (Lower is Better)",
                    labels={"RMSE": "Validation RMSE ($)", "model": "Candidate Algorithm"},
                    text="RMSE"
                )
                fig_comp.update_traces(texttemplate='$%{text:.1f}', textposition='outside')
                st.plotly_chart(fig_comp, use_container_width=True)

            # Version History & Rollback Interface
            st.divider()
            st.markdown("#### 🔄 Model Version History & Zero-Downtime Rollback")
            available_versions = vers_data.get("available_versions", [meta.get("version", "v001")]) if ok_vers else [meta.get("version", "v001")]
            current_ver = meta.get("version", "v001")

            r_col1, r_col2 = st.columns([2, 1])
            with r_col1:
                selected_version = st.selectbox(
                    "Select Target Version to Roll Back To:",
                    available_versions,
                    index=0,
                    help="Rolls back the active model file to a prior snapshot atomically."
                )
                confirm_rollback = st.checkbox(f"⚠️ I confirm rolling back production model from {current_ver} to {selected_version}")

            with r_col2:
                st.write("")
                st.write("")
                if st.button("Execute Model Rollback", disabled=not confirm_rollback):
                    with st.spinner(f"Rolling back to {selected_version}..."):
                        ok_rb, rb_res, rb_err = api_client.rollback_model(selected_version)
                    if ok_rb:
                        st.success(f"✅ Successfully rolled back active production model to {selected_version}!")
                        st.rerun()
                    else:
                        st.error(f"❌ Rollback failed: {rb_err}")

        else:
            st.info("No deployed model metadata found. Upload a training dataset to trigger the first training run.")

# ==========================================
# 6. DATASET INGESTION & SCHEDULER MONITOR
# ==========================================
with tab_upload:
    st.subheader("📤 Dataset Ingestion & Background Training Monitor")
    st.write("Upload new historical reservation CSV datasets to automatically trigger dataset merging, multi-model retraining, and automated best-model deployment.")

    up_col1, up_col2 = st.columns([1, 1])
    with up_col1:
        st.markdown("#### Upload Dataset to FastAPI Service")
        new_training_file = st.file_uploader(
            "Upload Training Dataset (CSV)",
            type=["csv"],
            key="api_training_dataset_uploader"
        )
        if new_training_file is not None:
            if st.button("⚡ Upload to Background ML Pipeline", use_container_width=True):
                with st.spinner("Uploading dataset to FastAPI..."):
                    file_bytes = new_training_file.getvalue()
                    ok_up, up_res, err_up = api_client.upload_dataset(file_bytes, new_training_file.name)
                if ok_up:
                    st.success(f"✅ Dataset accepted! Filename: {up_res.get('filename')}. Background scheduler will merge and retrain within 10 seconds.")
                else:
                    st.error(f"❌ Upload failed: {err_up}")

    with up_col2:
        st.markdown("#### Continuous Scheduler Status")
        if st.button("🔄 Refresh Scheduler Status"):
            st.rerun()

        ok_st, status_data, err_st = api_client.get_dataset_status()
        if ok_st and status_data:
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric("Pending Ingestion", f"{status_data.get('pending_count', 0)} files")
            with s_col2:
                st.metric("Processed Datasets", f"{status_data.get('processed_count', 0)} files")
            with s_col3:
                st.metric("Failed Ingestions", f"{status_data.get('failed_count', 0)} files")

            latest = status_data.get("latest_training")
            if latest:
                st.markdown("##### 📌 Last Processed Training Event")
                st.write(f"**Dataset:** `{latest.get('filename')}`")
                st.write(f"**Processed Timestamp:** `{latest.get('processed_at')}`")
                res_obj = latest.get("result", {})
                if res_obj:
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("Winner Model", str(res_obj.get("Selected Model", res_obj.get("status", "N/A"))))
                    with r2:
                        st.metric("RMSE", f"{res_obj.get('RMSE', 0):,.2f}")
                    with r3:
                        st.metric("R² Score", f"{res_obj.get('R2', 0)}")
        else:
            st.info("Training scheduler status not available.")

# ==========================================
# 7. CUSTOM CHART BUILDER TAB
# ==========================================
with tab_custom:
    st.subheader("🎨 Custom Chart Builder")
    st.write("Generate custom interactive dashboards across any variables in the loaded dataset.")

    all_cols = dashboard_data.columns.tolist()
    numeric_cols = dashboard_data.select_dtypes(include=[np.number]).columns.tolist()

    if "custom_x_axis" not in st.session_state:
        st.session_state.custom_x_axis = all_cols[0] if all_cols else None
    if "custom_y_axis" not in st.session_state:
        st.session_state.custom_y_axis = numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else None)
    if "custom_chart_type" not in st.session_state:
        st.session_state.custom_chart_type = "Scatter"

    col_x, col_y, col_type, col_btn = st.columns(4)
    with col_x:
        x_axis = st.selectbox("📊 X-axis Column", all_cols, index=all_cols.index(st.session_state.custom_x_axis) if st.session_state.custom_x_axis in all_cols else 0)
        st.session_state.custom_x_axis = x_axis
    with col_y:
        y_axis = st.selectbox("📈 Y-axis Column", all_cols, index=all_cols.index(st.session_state.custom_y_axis) if st.session_state.custom_y_axis in all_cols else (1 if len(all_cols) > 1 else 0))
        st.session_state.custom_y_axis = y_axis
    with col_type:
        chart_type = st.selectbox(
            "📉 Chart Type",
            ["Scatter", "Line", "Bar", "Box", "Histogram", "Area", "Density", "Violin", "Strip", "Funnel", "Pie", "Sunburst", "Treemap"],
            index=["Scatter", "Line", "Bar", "Box", "Histogram", "Area", "Density", "Violin", "Strip", "Funnel", "Pie", "Sunburst", "Treemap"].index(st.session_state.custom_chart_type) if st.session_state.custom_chart_type in ["Scatter", "Line", "Bar", "Box", "Histogram", "Area", "Density", "Violin", "Strip", "Funnel", "Pie", "Sunburst", "Treemap"] else 0
        )
        st.session_state.custom_chart_type = chart_type
    with col_btn:
        st.write("")
        st.write("")
        generate_btn = st.button("🔄 Render Chart", use_container_width=True)

    if x_axis and y_axis:
        try:
            plot_data = dashboard_data[[x_axis, y_axis]].dropna()
            if plot_data.empty:
                st.warning("⚠️ No valid non-null rows for selected columns.")
            else:
                if chart_type == "Scatter":
                    fig = px.scatter(plot_data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                elif chart_type == "Line":
                    if pd.api.types.is_numeric_dtype(plot_data[x_axis]):
                        plot_data = plot_data.sort_values(x_axis)
                    fig = px.line(plot_data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}", markers=True)
                elif chart_type == "Bar":
                    if plot_data[x_axis].dtype == "object":
                        agg_data = plot_data.groupby(x_axis)[y_axis].mean().reset_index()
                        fig = px.bar(agg_data, x=x_axis, y=y_axis, title=f"Average {y_axis} by {x_axis}")
                    else:
                        fig = px.bar(plot_data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                elif chart_type == "Box":
                    fig = px.box(plot_data, x=x_axis if plot_data[x_axis].dtype == "object" else None, y=y_axis, title=f"Distribution of {y_axis} by {x_axis}")
                elif chart_type == "Histogram":
                    fig = px.histogram(plot_data, x=y_axis, nbins=30, title=f"Distribution of {y_axis}")
                elif chart_type == "Area":
                    if pd.api.types.is_numeric_dtype(plot_data[x_axis]):
                        plot_data = plot_data.sort_values(x_axis)
                    fig = px.area(plot_data, x=x_axis, y=y_axis, title=f"Area Chart: {y_axis} vs {x_axis}")
                elif chart_type == "Density":
                    fig = px.density_contour(plot_data, x=x_axis, y=y_axis, title=f"Density: {y_axis} vs {x_axis}")
                elif chart_type == "Violin":
                    fig = px.violin(plot_data, x=x_axis, y=y_axis, box=True, points="outliers", title=f"Violin: {y_axis} by {x_axis}")
                elif chart_type == "Strip":
                    fig = px.strip(plot_data, x=x_axis, y=y_axis, jitter=True, title=f"Strip Plot: {y_axis} by {x_axis}")
                elif chart_type == "Funnel":
                    agg = plot_data.groupby(x_axis)[y_axis].sum().reset_index().sort_values(y_axis, ascending=False)
                    fig = px.funnel(agg, x=y_axis, y=x_axis, title=f"Funnel: {y_axis} by {x_axis}")
                elif chart_type == "Pie":
                    agg = plot_data.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.pie(agg, values=y_axis, names=x_axis, title=f"Pie: {y_axis} by {x_axis}")
                elif chart_type == "Sunburst":
                    agg = plot_data.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.sunburst(agg, names=x_axis, values=y_axis, title=f"Sunburst: {y_axis} by {x_axis}")
                elif chart_type == "Treemap":
                    agg = plot_data.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.treemap(agg, labels=x_axis, values=y_axis, title=f"Treemap: {y_axis} by {x_axis}")

                st.plotly_chart(fig, use_container_width=True, key=f"chart_{x_axis}_{y_axis}_{chart_type}")

                # Quick Summary Statistics
                st.divider()
                s1, s2, s3 = st.columns(3)
                if pd.api.types.is_numeric_dtype(plot_data[y_axis]):
                    with s1:
                        st.metric("Mean", f"{plot_data[y_axis].mean():.2f}")
                    with s2:
                        st.metric("Median", f"{plot_data[y_axis].median():.2f}")
                    with s3:
                        st.metric("Std Dev", f"{plot_data[y_axis].std():.2f}")

        except Exception as ce:
            st.error(f"❌ Chart generation error: {str(ce)}")

# ==========================================
# 8. DATA PREVIEW TAB
# ==========================================
with tab_data:
    st.subheader("Data Inspector & Static Report Generation")
    st.dataframe(dashboard_data, use_container_width=True)

    d_col1, d_col2 = st.columns([1, 1])
    with d_col1:
        csv_full = dashboard_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Cleaned Dataset (CSV)",
            data=csv_full,
            file_name="hotel_dataset_cleaned.csv",
            mime="text/csv",
        )
    with d_col2:
        if st.button("📄 Generate Static HTML Report"):
            try:
                subprocess.run(["python", "export_static.py"], check=True)
                st.success("✅ Static report generated at assets/hotel_report.html")
            except Exception as se:
                st.error(f"Failed to generate static report: {se}")

st.caption("Hotel Revenue Management System — Phase 3 Advanced Analytics & Intelligent Forecasting Platform.")
