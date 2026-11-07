"""
Data Drift Monitoring View - Kolmogorov-Smirnov & PSI Statistical Testing.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders statistical distribution drift monitoring using KS-tests and Categorical PSI."""
    
    render_section_header(
        "Data Drift & Distributional Stability Monitoring",
        "Track distribution shift between training reference data and incoming booking traffic via KS tests and PSI",
    )

    if st.button("Execute Real-Time Data Drift Audit", key="btn_run_drift", use_container_width=True):
        with st.spinner("Computing Kolmogorov-Smirnov statistics and categorical PSI across feature distributions..."):
            ok_dr, drift_res, err_dr = api_client.assess_data_drift()
            if ok_dr and drift_res:
                status = drift_res.get("overall_status", "No Drift")
                rec_text = drift_res.get("recommendation", "Features stable")
                
                if status == "No Drift":
                    render_alert(f"<b>Overall Status: {status}</b> — {rec_text}", alert_type="success")
                elif status == "Moderate Drift":
                    render_alert(f"<b>Overall Status: {status}</b> — {rec_text}", alert_type="warning")
                else:
                    render_alert(f"<b>Overall Status: {status}</b> — {rec_text}", alert_type="danger")

                d_kpi1, d_kpi2, d_kpi3 = st.columns(3)
                with d_kpi1:
                    render_kpi_card("Features Audited", str(drift_res.get("total_features_tested", 0)), subtext="Columns tested")
                with d_kpi2:
                    drift_count = drift_res.get("total_drifted_features", 0)
                    render_kpi_card("Drifted Features Count", str(drift_count), delta_color="negative" if drift_count > 0 else "positive", subtext="Features with p < 0.05 / PSI > 0.1")
                with d_kpi3:
                    render_kpi_card("Reference Baseline Size", f"{drift_res.get('baseline_records_count', 0):,} rows", subtext="Training distribution baseline")

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                # Numerical Drift Table
                render_section_header("Numerical Features Drift (Kolmogorov-Smirnov & Wasserstein Distance)")
                num_drift = drift_res.get("numerical_drift", [])
                if num_drift:
                    nd_df = pd.DataFrame(num_drift)
                    st.dataframe(nd_df, use_container_width=True)

                # Categorical Drift Table
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                render_section_header("Categorical Feature Stability (Population Stability Index - PSI)")
                cat_drift = drift_res.get("categorical_drift", [])
                if cat_drift:
                    cd_df = pd.DataFrame(cat_drift)
                    st.dataframe(cd_df, use_container_width=True)
            else:
                render_alert(f"Failed to assess data drift: {err_dr}", alert_type="danger")
    else:
        render_alert("Click <b>'Execute Real-Time Data Drift Audit'</b> above to evaluate statistical drift tests across active hotel feature distributions.", alert_type="info")

