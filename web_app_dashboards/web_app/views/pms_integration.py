"""
PMS Integration & Schema Audit View.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders PMS schema audit comparing demo schema with enterprise PMS requirements."""
    
    render_section_header(
        "Property Management System (PMS) Schema Audit",
        "Technical gap analysis comparing active reservation schema against enterprise PMS platforms (Oracle Opera, Mews, protel, Cloudbeds)",
    )

    ok_audit, audit_data, audit_err = api_client.get_missing_fields_audit()
    if ok_audit and audit_data:
        supported = audit_data.get("active_supported_fields", [])
        missing = audit_data.get("missing_production_pms_fields", [])
        
        # Summary KPIs
        k1, k2, k3 = st.columns(3)
        with k1:
            render_kpi_card("Supported Schema Fields", str(len(supported)), delta="Active", delta_color="positive", subtext="Verified reservation fields")
        with k2:
            render_kpi_card("Missing Production Fields", str(len(missing)), delta="Required", delta_color="warning", subtext="Fields required for live PMS sync")
        with k3:
            readiness = 100 * len(supported) / (len(supported) + len(missing)) if (len(supported) + len(missing)) > 0 else 0
            render_kpi_card("Schema Readiness Score", f"{readiness:.1f}%", delta_color="neutral", subtext="Enterprise integration parity")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        render_section_header("Currently Supported Active Fields")
        st.write(", ".join([f"`{f}`" for f in supported]))

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_section_header("Missing Production PMS Fields Required for Live Integration", "Fields needed to transition from ex-ante demonstration to direct PMS webhook synchronization")
        if missing:
            m_df = pd.DataFrame(missing)
            st.dataframe(m_df, use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_section_header("Standard Ingestion Interface Specification")
        integ = audit_data.get("integration_interface", {})
        st.json(integ)
    else:
        render_alert(f"Could not load PMS schema audit: {audit_err}", alert_type="danger")

