"""
Model Governance & Registry View - Zero-Downtime Deployment & Version Rollback.
"""

from typing import Any, Dict
import pandas as pd
import plotly.express as px
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders enterprise ML model governance, benchmark tournament results, and version rollback."""
    
    ok_info, meta, err_info = api_client.get_model_info()
    ok_comp, comp, err_comp = api_client.get_model_comparison()
    ok_vers, vers_data, err_vers = api_client.get_model_versions()

    if ok_info and meta:
        # High Level Model KPIs
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            render_kpi_card("Active Algorithm", str(meta.get("algorithm", "Unknown")), subtext="Production champion model")
        with m_col2:
            render_kpi_card("Active Version", str(meta.get("version", "Unknown")), delta="Production", delta_color="positive", subtext="Atomic deployed artifact tag")
        with m_col3:
            render_kpi_card("Feature Dimensions", f"{meta.get('feature_count', 0)} cols", subtext="Pipeline inputs verified")
        with m_col4:
            dep_time = meta.get("deployed_at", "")[:19].replace("T", " ")
            render_kpi_card("Deployed At", dep_time if dep_time else "N/A", subtext="UTC timestamp")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Performance Evaluation Metrics: Validation vs Test Holdout
        render_section_header("Performance Evaluation Metrics", "Evaluated on untouched test split with no future lookahead leakage")
        val_m = meta.get("validation_metrics", {})
        test_m = meta.get("test_metrics", {})

        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            render_kpi_card("Validation RMSE", f"${val_m.get('RMSE', 0):,.2f}", subtext="Primary tournament metric")
        with e_col2:
            render_kpi_card("Validation R²", f"{val_m.get('R2', 0):.4f}", subtext="Variance explained")
        with e_col3:
            render_kpi_card("Untouched Test RMSE", f"${test_m.get('RMSE', 0):,.2f}", subtext="Out-of-sample holdout error")
        with e_col4:
            render_kpi_card("Untouched Test R²", f"{test_m.get('R2', 0):.4f}", subtext="Generalization performance")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Multi-Model Benchmark Comparison
        render_section_header("5-Model Tournament Benchmark", "Automated model selection across regressors trained on identical splits")
        if ok_comp and comp.get("models"):
            comp_df = pd.DataFrame(comp["models"])
            cols_order = [c for c in ["model", "RMSE", "MAE", "R2", "MAPE", "fit_time_seconds", "status"] if c in comp_df.columns]
            comp_df = comp_df[cols_order].sort_values("RMSE", ascending=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(comp_df, use_container_width=True)

            with c2:
                fig_comp = px.bar(
                    comp_df,
                    x="model",
                    y="RMSE",
                    color="model",
                    color_discrete_sequence=CHART_PALETTE,
                    text="RMSE",
                )
                fig_comp.update_traces(texttemplate="$%{text:.1f}", textposition="outside")
                apply_saas_theme(fig_comp, height=280, x_title="Candidate Model", y_title="Validation RMSE ($)")
                fig_comp.update_layout(showlegend=False)
                st.plotly_chart(fig_comp, use_container_width=True)

        # Version History & Rollback Interface
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_section_header("Version History & Zero-Downtime Rollback", "Atomically switch active model artifact with rollback governance")
        
        available_versions = vers_data.get("available_versions", [meta.get("version", "v001")]) if ok_vers else [meta.get("version", "v001")]
        current_ver = meta.get("version", "v001")

        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            selected_version = st.selectbox(
                "Target Deployment Version:",
                available_versions,
                index=0,
                help="Rolls back the active model file to a prior snapshot atomically.",
            )
            confirm_rollback = st.checkbox(f"I confirm promoting / rolling back production model from {current_ver} to {selected_version}")

        with r_col2:
            st.write("")
            st.write("")
            if st.button("Execute Atomic Rollback", disabled=not confirm_rollback):
                with st.spinner(f"Promoting {selected_version} to production..."):
                    ok_rb, rb_res, rb_err = api_client.rollback_model(selected_version)
                if ok_rb:
                    render_alert(f"Successfully rolled back active production model to {selected_version}!", alert_type="success")
                    st.rerun()
                else:
                    render_alert(f"Rollback failed: {rb_err}", alert_type="danger")

    else:
        render_alert("No deployed model metadata found. Please verify the FastAPI backend connection.", alert_type="warning")

