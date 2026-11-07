"""
Data Explorer View - Schema & Raw Data Inspection.
"""

import subprocess
from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders interactive data explorer, summary statistics, and static export options."""
    
    render_section_header("Data Explorer & Schema Inspector", "Review dataset dimensions, missing values, and column schemas")

    # High level dataset KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total Rows", f"{len(df):,}", subtext="Observation records")
    with k2:
        render_kpi_card("Total Columns", f"{df.shape[1]:,}", subtext="Tracked feature dimensions")
    with k3:
        null_count = df.isnull().sum().sum()
        render_kpi_card("Missing Values", f"{null_count:,}", delta_color="positive" if null_count == 0 else "warning", subtext="Total null cells")
    with k4:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        render_kpi_card("Memory Footprint", f"{mem_mb:.2f} MB", subtext="In-memory dataframe size")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Search & Filtering
    filter_txt = st.text_input("Filter records by search term (case-insensitive across string columns)", value="")
    display_df = df
    if filter_txt:
        str_cols = display_df.select_dtypes(include=["object"]).columns
        mask = False
        for c in str_cols:
            mask = mask | display_df[c].astype(str).str.contains(filter_txt, case=False, na=False)
        display_df = display_df[mask]

    st.dataframe(display_df, use_container_width=True)

    # Export & Report Actions
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    render_section_header("Export & Static Reporting")
    
    d_col1, d_col2 = st.columns([1, 1])
    with d_col1:
        csv_full = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered Dataset (CSV)",
            data=csv_full,
            file_name="hotel_dataset_export.csv",
            mime="text/csv",
        )
    with d_col2:
        if st.button("Generate Static HTML Report"):
            try:
                subprocess.run(["python", "export_static.py"], check=True)
                render_alert("Static report generated successfully at assets/hotel_report.html", alert_type="success")
            except Exception as se:
                render_alert(f"Failed to generate static report: {se}", alert_type="danger")

