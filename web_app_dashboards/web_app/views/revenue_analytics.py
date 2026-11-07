"""
Revenue Analytics View - Deep Dive into Portfolio Revenue Dynamics.
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
    """Renders comprehensive revenue trend and yield analytics."""
    
    # Check date column
    date_col = "Date" if "Date" in df.columns else None
    working_df = df.copy()
    
    if date_col:
        working_df["ParsedDate"] = pd.to_datetime(working_df[date_col], errors="coerce")
        working_df = working_df.dropna(subset=["ParsedDate"]).sort_values("ParsedDate")
        
    # Interactive Filters Row
    st.markdown("#### Portfolio Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        if date_col and not working_df.empty:
            min_date = working_df["ParsedDate"].min().date()
            max_date = working_df["ParsedDate"].max().date()
            date_range = st.date_input("Date Range", value=(min_date, max_date))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                working_df = working_df[
                    (working_df["ParsedDate"].dt.date >= date_range[0]) &
                    (working_df["ParsedDate"].dt.date <= date_range[1])
                ]
        else:
            st.caption("Date filtering unavailable without Date column.")
            
    with f_col2:
        seg_col = "Market_Segment" if "Market_Segment" in df.columns else None
        if seg_col:
            segments = ["All Segments"] + sorted(list(df[seg_col].dropna().unique()))
            chosen_seg = st.selectbox("Market Segment", segments)
            if chosen_seg != "All Segments":
                working_df = working_df[working_df[seg_col] == chosen_seg]
        else:
            st.caption("No segment column found.")
            
    with f_col3:
        chan_col = "Booking_Channel" if "Booking_Channel" in df.columns else None
        if chan_col:
            channels = ["All Channels"] + sorted(list(df[chan_col].dropna().unique()))
            chosen_chan = st.selectbox("Booking Channel", channels)
            if chosen_chan != "All Channels":
                working_df = working_df[working_df[chan_col] == chosen_chan]
        else:
            st.caption("No booking channel column found.")
            
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # Revenue KPI Summary for Filtered Range
    tot_rev = working_df["Total_Revenue"].sum() if "Total_Revenue" in working_df.columns else 0.0
    tot_cost = working_df["Total_Costs"].sum() if "Total_Costs" in working_df.columns else 0.0
    gross_profit = tot_rev - tot_cost if ("Total_Revenue" in working_df.columns and "Total_Costs" in working_df.columns) else 0.0
    
    adr_col = "Average_Daily_Rate" if "Average_Daily_Rate" in working_df.columns else ("ADR" if "ADR" in working_df.columns else None)
    avg_adr = working_df[adr_col].mean() if adr_col else 0.0
    avg_occ = working_df["Occupancy_Rate"].mean() if "Occupancy_Rate" in working_df.columns else 0.0
    revpar = avg_adr * avg_occ
    
    rk1, rk2, rk3, rk4, rk5 = st.columns(5)
    with rk1:
        render_kpi_card("Period Revenue", f"${tot_rev:,.0f}", subtext=f"{len(working_df):,} total records")
    with rk2:
        render_kpi_card("Gross Profit", f"${gross_profit:,.0f}", subtext=f"Margin: {(gross_profit/tot_rev*100) if tot_rev>0 else 0:.1f}%")
    with rk3:
        render_kpi_card("Realized ADR", f"${avg_adr:.2f}", subtext="Per occupied room")
    with rk4:
        render_kpi_card("RevPAR", f"${revpar:.2f}", subtext="Yield per available room")
    with rk5:
        render_kpi_card("Average Occupancy", f"{avg_occ*100:.1f}%", subtext="Portfolio capacity utilization")
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Visualizations Section
    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Temporal Revenue Progression", "Daily realized revenue with 7-day moving average trend")
        if date_col and "Total_Revenue" in working_df.columns and not working_df.empty:
            working_df["MA7"] = working_df["Total_Revenue"].rolling(7, min_periods=1).mean()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=working_df["ParsedDate"],
                y=working_df["Total_Revenue"],
                mode="lines",
                name="Daily Revenue",
                line=dict(color="rgba(37, 99, 235, 0.4)", width=1.5),
            ))
            fig_trend.add_trace(go.Scatter(
                x=working_df["ParsedDate"],
                y=working_df["MA7"],
                mode="lines",
                name="7-Day Moving Avg",
                line=dict(color="#2563EB", width=2.5),
            ))
            apply_saas_theme(fig_trend, height=340, y_title="Revenue ($)")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Insufficient temporal data for progression chart.")
            
    with c2:
        render_section_header("ADR & RevPAR Yield Comparison", "Rate realization versus total inventory yield")
        if date_col and adr_col and "Occupancy_Rate" in working_df.columns and not working_df.empty:
            working_df["ComputedRevPAR"] = working_df[adr_col] * working_df["Occupancy_Rate"]
            fig_yield = go.Figure()
            fig_yield.add_trace(go.Scatter(
                x=working_df["ParsedDate"],
                y=working_df[adr_col],
                mode="lines",
                name="ADR ($)",
                line=dict(color="#2563EB", width=2),
            ))
            fig_yield.add_trace(go.Scatter(
                x=working_df["ParsedDate"],
                y=working_df["ComputedRevPAR"],
                mode="lines",
                name="RevPAR ($)",
                line=dict(color="#0F766E", width=2),
            ))
            apply_saas_theme(fig_yield, height=340, y_title="Rate / Yield ($)")
            st.plotly_chart(fig_yield, use_container_width=True)
        else:
            st.info("Insufficient rate/occupancy data for yield comparison.")
            
    # Breakdown Analysis Row
    b1, b2 = st.columns(2)
    with b1:
        render_section_header("Revenue Contribution by Market Segment", "Total revenue generated per segment")
        if seg_col and "Total_Revenue" in working_df.columns:
            seg_rev = working_df.groupby(seg_col)["Total_Revenue"].sum().reset_index()
            fig_s = px.bar(
                seg_rev,
                x=seg_col,
                y="Total_Revenue",
                color=seg_col,
                color_discrete_sequence=CHART_PALETTE,
            )
            apply_saas_theme(fig_s, height=300, y_title="Revenue ($)")
            fig_s.update_layout(showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("No segment revenue data available.")
            
    with b2:
        render_section_header("Occupancy Distribution", "Distribution of daily occupancy rates across selected period")
        if "Occupancy_Rate" in working_df.columns:
            fig_dist = px.histogram(
                working_df,
                x="Occupancy_Rate",
                nbins=20,
                color_discrete_sequence=["#0F766E"],
            )
            apply_saas_theme(fig_dist, height=300, x_title="Occupancy Rate", y_title="Days")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No occupancy rate data available.")
            
    # Data Table & Export
    render_section_header("Detailed Records Table", "Sample records matching active filters")
    show_cols = [c for c in ["Date", "Month", "Market_Segment", "Booking_Channel", "Occupancy_Rate", "Average_Daily_Rate", "Total_Revenue", "Total_Costs", "Profit"] if c in working_df.columns]
    st.dataframe(working_df[show_cols].head(50), use_container_width=True)
    
    st.download_button(
        "Download Filtered Data (CSV)",
        data=working_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_revenue_data.csv",
        mime="text/csv",
    )

