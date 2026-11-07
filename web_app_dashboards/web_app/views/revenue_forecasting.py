"""
Revenue & Occupancy Forecasting View - Multi-Horizon ML Projections.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders 7, 30, and 90-day revenue and occupancy forecasting workspace."""
    
    st.markdown("#### Forecast Configuration")
    fc_col1, fc_col2, fc_col3 = st.columns([1, 1, 1])
    
    with fc_col1:
        horizon_choice = st.radio(
            "Forecasting Horizon",
            [7, 30, 90],
            index=1,
            horizontal=True,
            format_func=lambda x: f"{x} Days",
        )
    with fc_col2:
        forecast_type = st.radio(
            "Forecast Metric",
            ["Total Revenue ($)", "Occupancy Rate (%)"],
            horizontal=True,
        )
    with fc_col3:
        st.write("")
        st.caption("Algorithm: HistGradientBoostingRegressor with chronological backtesting.")
        
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    if forecast_type == "Total Revenue ($)":
        ok_fc, fc_data, fc_err = api_client.get_revenue_forecast(days=horizon_choice)
        if ok_fc and fc_data:
            b_metrics = fc_data.get("backtest_metrics", {})
            
            # KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card(
                    "Total Projected Revenue",
                    f"${fc_data.get('total_projected_revenue', 0):,.0f}",
                    subtext=f"{horizon_choice}-day cumulative revenue",
                )
            with k2:
                render_kpi_card(
                    "Daily Avg Projected",
                    f"${fc_data.get('average_daily_revenue', 0):,.2f}",
                    subtext="Expected daily average revenue",
                )
            with k3:
                rmse_val = b_metrics.get("rmse", 0)
                render_kpi_card(
                    "Backtested RMSE",
                    f"${rmse_val:,.2f}",
                    subtext="Historical test holdout error",
                )
            with k4:
                mape_val = b_metrics.get("mape", 0)
                render_kpi_card(
                    "Backtested MAPE",
                    f"{mape_val:.1f}%",
                    subtext="Mean absolute percentage error",
                )
                
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            
            # Chart
            render_section_header(
                f"{horizon_choice}-Day Daily Revenue Forecast",
                "Point predictions with empirical 95% confidence intervals",
            )
            points = fc_data.get("forecast", [])
            fc_df = pd.DataFrame(points)
            if not fc_df.empty:
                fig_fc = go.Figure()
                
                # 95% Confidence Interval band
                upper_col = "upper_bound" if "upper_bound" in fc_df.columns else "ci_upper"
                lower_col = "lower_bound" if "lower_bound" in fc_df.columns else "ci_lower"
                pred_col = "predicted_revenue" if "predicted_revenue" in fc_df.columns else "prediction"
                
                if upper_col in fc_df.columns and lower_col in fc_df.columns:
                    fig_fc.add_trace(go.Scatter(
                        x=fc_df["date"].tolist() + fc_df["date"].tolist()[::-1],
                        y=fc_df[upper_col].tolist() + fc_df[lower_col].tolist()[::-1],
                        fill="toself",
                        fillcolor="rgba(37, 99, 235, 0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        name="95% Confidence Interval",
                    ))
                    
                fig_fc.add_trace(go.Scatter(
                    x=fc_df["date"],
                    y=fc_df[pred_col],
                    mode="lines+markers",
                    name="Projected Daily Revenue",
                    line=dict(color="#2563EB", width=2.5),
                    marker=dict(size=5, color="#1D4ED8"),
                ))
                apply_saas_theme(fig_fc, height=360, y_title="Daily Revenue ($)")
                st.plotly_chart(fig_fc, use_container_width=True)
                
                # Daily Forecast Breakdown Table
                with st.expander("Detailed Forecast Breakdown Table", expanded=False):
                    st.dataframe(fc_df, use_container_width=True)
                    
            # Limitations & Assumptions
            limits = fc_data.get("limitations", [])
            if limits:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                render_section_header("Forecast Assumptions & Model Boundaries")
                limit_text = "<br>".join([f"• {lim}" for lim in limits])
                render_alert(limit_text, alert_type="info")
        else:
            render_alert(f"Failed to generate revenue forecast: {fc_err}", alert_type="danger")
            
    else:
        # Occupancy Rate Forecast
        ok_occ, occ_data, occ_err = api_client.get_occupancy_forecast(days=horizon_choice)
        if ok_occ and occ_data:
            b_metrics = occ_data.get("backtest_metrics", {})
            h_pats = occ_data.get("historical_patterns", {})
            
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card(
                    "Mean Projected Occupancy",
                    f"{occ_data.get('mean_projected_occupancy', 0)*100:.1f}%",
                    subtext=f"{horizon_choice}-day forward horizon",
                )
            with k2:
                render_kpi_card(
                    "Weekday Historical Avg",
                    f"{h_pats.get('weekday_average_occupancy', 0)*100:.1f}%",
                    subtext="Historical Monday–Friday mean",
                )
            with k3:
                render_kpi_card(
                    "Weekend Historical Avg",
                    f"{h_pats.get('weekend_average_occupancy', 0)*100:.1f}%",
                    subtext="Historical Saturday–Sunday mean",
                )
            with k4:
                render_kpi_card(
                    "Backtested MAPE",
                    f"{b_metrics.get('mape', 0):.1f}%",
                    subtext="Mean absolute percentage error",
                )
                
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            
            render_section_header(
                f"{horizon_choice}-Day Daily Occupancy Forecast",
                "Point estimates with 95% confidence intervals against capacity limits",
            )
            points = occ_data.get("forecast", [])
            occ_df = pd.DataFrame(points)
            if not occ_df.empty:
                fig_occ = go.Figure()
                upper_col = "upper_bound" if "upper_bound" in occ_df.columns else "ci_upper"
                lower_col = "lower_bound" if "lower_bound" in occ_df.columns else "ci_lower"
                pred_col = "predicted_occupancy" if "predicted_occupancy" in occ_df.columns else "prediction"
                
                if upper_col in occ_df.columns and lower_col in occ_df.columns:
                    fig_occ.add_trace(go.Scatter(
                        x=occ_df["date"].tolist() + occ_df["date"].tolist()[::-1],
                        y=(occ_df[upper_col] * 100).tolist() + (occ_df[lower_col] * 100).tolist()[::-1],
                        fill="toself",
                        fillcolor="rgba(15, 118, 110, 0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        name="95% Confidence Bounds",
                    ))
                    
                fig_occ.add_trace(go.Scatter(
                    x=occ_df["date"],
                    y=occ_df[pred_col] * 100,
                    mode="lines+markers",
                    name="Projected Occupancy (%)",
                    line=dict(color="#0F766E", width=2.5),
                    marker=dict(size=5, color="#0F766E"),
                ))
                apply_saas_theme(fig_occ, height=360, y_title="Occupancy Rate (%)")
                st.plotly_chart(fig_occ, use_container_width=True)
                
                with st.expander("Detailed Occupancy Forecast Breakdown Table", expanded=False):
                    st.dataframe(occ_df, use_container_width=True)
        else:
            render_alert(f"Failed to generate occupancy forecast: {occ_err}", alert_type="danger")

