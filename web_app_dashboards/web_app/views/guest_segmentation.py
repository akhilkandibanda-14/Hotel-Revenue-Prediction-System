"""
Guest Segmentation View - Demographic & Channel Distribution.
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
    """Renders channel, geographic, and guest demographic segmentation analytics."""
    
    # Overview metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        n_channels = df["Booking_Channel"].nunique() if "Booking_Channel" in df.columns else 0
        render_kpi_card("Active Channels", str(n_channels), subtext="Direct, OTA, Corporate, etc.")
    with k2:
        n_countries = df["Guest_Country"].nunique() if "Guest_Country" in df.columns else 0
        render_kpi_card("Origin Markets", str(n_countries), subtext="International guest origin countries")
    with k3:
        n_segments = df["Market_Segment"].nunique() if "Market_Segment" in df.columns else 0
        render_kpi_card("Market Segments", str(n_segments), subtext="Commercial segmentation tiers")
    with k4:
        top_chan = df["Booking_Channel"].mode()[0] if "Booking_Channel" in df.columns and not df["Booking_Channel"].empty else "—"
        render_kpi_card("Primary Channel", str(top_chan), subtext="Highest volume booking source")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Row 1: Booking Channels & Origin Countries
    left, right = st.columns(2)
    with left:
        render_section_header("Booking Channel Share", "Volume distribution across acquisition channels")
        if "Booking_Channel" in df.columns:
            chan_counts = df["Booking_Channel"].value_counts().reset_index()
            chan_counts.columns = ["Channel", "Bookings"]
            fig_chan = px.pie(
                chan_counts,
                names="Channel",
                values="Bookings",
                hole=0.55,
                color_discrete_sequence=CHART_PALETTE,
            )
            fig_chan.update_traces(textposition="inside", textinfo="percent+label")
            apply_saas_theme(fig_chan, height=330)
            st.plotly_chart(fig_chan, use_container_width=True)
        else:
            st.info("Booking_Channel column not found.")

    with right:
        render_section_header("Top Origin Markets", "Leading feeder countries by booking volume")
        if "Guest_Country" in df.columns:
            top_countries = df["Guest_Country"].value_counts().head(10).reset_index()
            top_countries.columns = ["Country", "Bookings"]
            fig_ctry = px.bar(
                top_countries,
                x="Bookings",
                y="Country",
                orientation="h",
                color_discrete_sequence=["#2563EB"],
            )
            fig_ctry.update_layout(yaxis=dict(autorange="reversed"))
            apply_saas_theme(fig_ctry, height=330, x_title="Total Bookings", y_title="Country")
            st.plotly_chart(fig_ctry, use_container_width=True)
        else:
            st.info("Guest_Country column not found.")

    # Row 2: Market Segment Revenue & Guest Type
    b_left, b_right = st.columns(2)
    with b_left:
        render_section_header("Market Segment Yield", "Revenue generation by market segment")
        if "Market_Segment" in df.columns and "Total_Revenue" in df.columns:
            seg_rev = df.groupby("Market_Segment")["Total_Revenue"].sum().reset_index()
            fig_sr = px.bar(
                seg_rev,
                x="Market_Segment",
                y="Total_Revenue",
                color="Market_Segment",
                color_discrete_sequence=CHART_PALETTE,
            )
            apply_saas_theme(fig_sr, height=320, y_title="Revenue ($)")
            fig_sr.update_layout(showlegend=False)
            st.plotly_chart(fig_sr, use_container_width=True)
        elif "Market_Segment" in df.columns:
            seg_counts = df["Market_Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig_sc = px.bar(seg_counts, x="Segment", y="Count", color_discrete_sequence=["#0F766E"])
            apply_saas_theme(fig_sc, height=320, y_title="Count")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Market_Segment column not found.")

    with b_right:
        render_section_header("Guest Type Profile", "Distribution across customer classifications")
        if "Guest_Type" in df.columns:
            gt_counts = df["Guest_Type"].value_counts().reset_index()
            gt_counts.columns = ["Guest_Type", "Count"]
            fig_gt = px.pie(
                gt_counts,
                names="Guest_Type",
                values="Count",
                hole=0.55,
                color_discrete_sequence=["#0F766E", "#2563EB", "#D97706", "#64748B"],
            )
            fig_gt.update_traces(textposition="inside", textinfo="percent+label")
            apply_saas_theme(fig_gt, height=320)
            st.plotly_chart(fig_gt, use_container_width=True)
        else:
            st.info("Guest_Type column not found.")

