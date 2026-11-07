"""
Demand Forecasting View - Room Nights & Capacity Pacing.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders room demand forecasting and pacing analytics."""
    
    d_ctrl1, d_ctrl2 = st.columns([1, 2])
    with d_ctrl1:
        demand_horizon = st.radio(
            "Forecast Horizon",
            [7, 30, 90],
            index=1,
            horizontal=True,
            key="demand_horizon_choice",
            format_func=lambda x: f"{x} Days",
        )
    with d_ctrl2:
        st.caption("Forecasting model accounts for day-of-week seasonality, historical pacing momentum, and rolling occupancy patterns.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    ok_dem, dem_data, dem_err = api_client.get_demand_forecast(horizon=demand_horizon)
    if ok_dem and dem_data:
        tot_nights = dem_data.get("total_expected_room_nights", 0)
        avg_demand = dem_data.get("average_daily_room_demand", 0)
        avg_occ = dem_data.get("projected_average_occupancy_pct", 0)
        bench = dem_data.get("benchmark_evaluation", {})
        ml_metrics = bench.get("metrics", {}).get("ml_model", {})
        forecast_points = dem_data.get("forecast", [])

        # KPI Row
        dk1, dk2, dk3, dk4 = st.columns(4)
        with dk1:
            render_kpi_card(
                "Daily Room Demand",
                f"{avg_demand:,.0f} rms/day",
                subtext="Expected average room nights",
            )
        with dk2:
            render_kpi_card(
                "Cumulative Room Nights",
                f"{tot_nights:,.0f} r-nts",
                subtext=f"{demand_horizon}-day total volume",
            )
        with dk3:
            render_kpi_card(
                "Projected Occupancy",
                f"{avg_occ:.1f}%",
                subtext="Expected mean occupancy rate",
            )
        with dk4:
            ml_rmse = ml_metrics.get("rmse", 0)
            render_kpi_card(
                "Holdout Test RMSE",
                f"{ml_rmse:.2f}",
                subtext="HistGradientBoosting holdout error",
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Main Chart: Demand & Occupancy Trajectory
        dem_df = pd.DataFrame(forecast_points)
        if not dem_df.empty:
            render_section_header(
                f"{demand_horizon}-Day Room Demand & Occupancy Trajectory",
                "Projected daily room demand and dual-axis occupancy rate percentage",
            )
            fig_dem = go.Figure()
            # 95% Confidence Bounds
            fig_dem.add_trace(go.Scatter(
                x=dem_df["date"].tolist() + dem_df["date"].tolist()[::-1],
                y=dem_df["upper_bound_rooms"].tolist() + dem_df["lower_bound_rooms"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% Demand Bounds",
            ))
            fig_dem.add_trace(go.Scatter(
                x=dem_df["date"],
                y=dem_df["expected_room_demand"],
                mode="lines+markers",
                name="Expected Room Demand",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=5, color="#1D4ED8"),
            ))
            fig_dem.add_trace(go.Scatter(
                x=dem_df["date"],
                y=dem_df["expected_occupancy_pct"],
                mode="lines",
                yaxis="y2",
                line=dict(color="#0F766E", width=2, dash="dot"),
                name="Projected Occupancy (%)",
            ))

            fig_dem.update_layout(
                yaxis2=dict(
                    title=dict(text="Occupancy Rate (%)", font=dict(color="#0F766E", size=12)),
                    overlaying="y",
                    side="right",
                    range=[0, 105],
                    showgrid=False,
                    tickfont=dict(color="#0F766E", size=11),
                )
            )
            apply_saas_theme(fig_dem, height=360, x_title="Date", y_title="Room Demand (Rooms)")
            st.plotly_chart(fig_dem, use_container_width=True)

            # Cumulative Room Nights Chart
            c1, c2 = st.columns([1, 1])
            with c1:
                render_section_header("Cumulative Room Nights Delivered")
                fig_cum = px.area(
                    dem_df,
                    x="date",
                    y="cumulative_room_nights",
                    color_discrete_sequence=["#2563EB"],
                )
                apply_saas_theme(fig_cum, height=280, x_title="Date", y_title="Cumulative Room Nights")
                st.plotly_chart(fig_cum, use_container_width=True)

            with c2:
                render_section_header("Model Evaluation vs Historical Baselines")
                b_metrics = bench.get("metrics", {})
                b_ml = b_metrics.get("ml_model", {})
                b_ma = b_metrics.get("moving_average_7d", {})
                b_sn = b_metrics.get("seasonal_naive_7d", {})
                b_lv = b_metrics.get("last_value", {})

                bench_records = [
                    {"Model / Benchmark": "HistGradientBoosting ML", "RMSE": b_ml.get("rmse", 0), "MAE": b_ml.get("mae", 0), "MAPE (%)": b_ml.get("mape", 0)},
                    {"Model / Benchmark": "7-Day Moving Average Baseline", "RMSE": b_ma.get("rmse", 0), "MAE": b_ma.get("mae", 0), "MAPE (%)": b_ma.get("mape", 0)},
                    {"Model / Benchmark": "Seasonal-Naive (7-Day Lag)", "RMSE": b_sn.get("rmse", 0), "MAE": b_sn.get("mae", 0), "MAPE (%)": b_sn.get("mape", 0)},
                    {"Model / Benchmark": "Last-Value Baseline", "RMSE": b_lv.get("rmse", 0), "MAE": b_lv.get("mae", 0), "MAPE (%)": b_lv.get("mape", 0)},
                ]
                st.dataframe(pd.DataFrame(bench_records), use_container_width=True)

            with st.expander("Daily Demand Breakdown Table", expanded=False):
                display_dem = dem_df[[
                    "date", "expected_room_demand", "lower_bound_rooms", "upper_bound_rooms",
                    "expected_room_nights", "cumulative_room_nights", "expected_occupancy_pct"
                ]].copy()
                st.dataframe(display_dem, use_container_width=True)
    else:
        render_alert(f"Failed to generate demand forecast: {dem_err}", alert_type="danger")

