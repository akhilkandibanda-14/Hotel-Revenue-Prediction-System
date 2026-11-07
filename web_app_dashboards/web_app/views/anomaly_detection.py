"""
Anomaly Detection View - Isolation Forest & Operational Outlier Auditing.
"""

from typing import Any, Dict
import pandas as pd
import plotly.express as px
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders operational outlier and anomaly detection workspace."""
    
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        contam_slider = st.slider("Isolation Forest Contamination Rate", min_value=0.01, max_value=0.15, value=0.05, step=0.01)
    with ctrl2:
        severity_filter = st.selectbox("Severity Filter", ["All Severities", "High", "Medium", "Low"])

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    ok_anom, anom_res, anom_err = api_client.get_anomalies(contamination=contam_slider)
    if ok_anom and anom_res:
        c_sev = anom_res.get("counts_by_severity", {})
        
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            render_kpi_card("Records Scanned", f"{anom_res.get('total_records_analyzed', 0):,}", subtext="Operational dataset rows")
        with a2:
            rate_pct = anom_res.get("anomaly_rate_pct", 0)
            render_kpi_card("Anomalies Flagged", f"{anom_res.get('anomalies_detected', 0):,}", delta=f"{rate_pct}% of total", delta_color="neutral", subtext="Outliers detected")
        with a3:
            render_kpi_card("High Severity", f"{c_sev.get('High', 0):,}", delta_color="negative", subtext="Large revenue / ADR excursions")
        with a4:
            render_kpi_card("Medium Severity", f"{c_sev.get('Medium', 0):,}", delta_color="neutral", subtext="Moderate statistical deviations")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        anom_list = anom_res.get("anomalies", [])
        if severity_filter != "All Severities":
            anom_list = [a for a in anom_list if a["severity"] == severity_filter]

        anom_df = pd.DataFrame(anom_list)
        if not anom_df.empty:
            render_section_header("Detected Operational Anomalies Timeline", "Scatter visualization of anomalies sized by deviation magnitude")
            fig_anom = px.scatter(
                anom_df,
                x="date",
                y="actual_value",
                color="severity",
                size=anom_df["deviation_pct"].abs().clip(lower=5, upper=100),
                hover_data=["metric", "actual_value", "expected_value", "deviation_pct", "direction"],
                color_discrete_map={"High": "#DC2626", "Medium": "#D97706", "Low": "#2563EB"},
            )
            apply_saas_theme(fig_anom, height=360, x_title="Date", y_title="Observed Value")
            st.plotly_chart(fig_anom, use_container_width=True)

            render_section_header("Anomaly Diagnostics Ledger", "Tabular record of flagged events with diagnostic attribution")
            cols_to_show = ["date", "metric", "actual_value", "expected_value", "deviation_pct", "direction", "severity", "detection_method", "diagnostic_note"]
            cols_avail = [c for c in cols_to_show if c in anom_df.columns]
            st.dataframe(anom_df[cols_avail], use_container_width=True)
        else:
            render_alert("No anomalies found matching the selected filter criteria.", alert_type="success")
    else:
        render_alert(f"Failed to scan anomalies: {anom_err}", alert_type="danger")

