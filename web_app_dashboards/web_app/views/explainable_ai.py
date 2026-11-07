"""
Explainable AI (XAI) View - Global and Local SHAP Attributions.
"""

from datetime import datetime
from typing import Any, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders SHAP feature attribution explanations (Global & Single Prediction)."""
    
    render_alert(
        "<b>Interpretability Notice:</b> SHAP attributions quantify statistical contribution to tree ensemble predictions "
        "via cooperative game theory (Shapley values) and do not represent counterfactual causal proof.",
        alert_type="info",
    )

    tab_glob, tab_loc = st.tabs(["Global Feature Importance", "Single Prediction Explanation"])

    with tab_glob:
        render_section_header("Global SHAP Feature Importance", "Mean absolute SHAP value across representative historical booking records")
        ok_sh_g, glob_shap, err_sh_g = api_client.get_global_shap_importance(max_samples=50)
        if ok_sh_g and glob_shap:
            base_val = glob_shap.get("base_value", 0)
            render_kpi_card("Model Expected Value E[f(X)]", f"${base_val:,.2f}", subtext="Baseline prediction across dataset")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            feat_imps = glob_shap.get("feature_importances", [])
            f_df = pd.DataFrame(feat_imps)
            if not f_df.empty:
                f_df = f_df.sort_values("importance", ascending=True)
                fig_shap = px.bar(
                    f_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale=["#93C5FD", "#2563EB", "#1E3A8A"],
                )
                fig_shap.update_layout(coloraxis_showscale=False)
                apply_saas_theme(fig_shap, height=450, x_title="Mean |SHAP Value| ($ Revenue Impact)", y_title="Operational Feature")
                st.plotly_chart(fig_shap, use_container_width=True)
        else:
            render_alert(f"Failed to fetch global SHAP importances: {err_sh_g}", alert_type="danger")

    with tab_loc:
        render_section_header("Single Prediction Waterfall Decomposition", "Deconstruct how specific feature values push a prediction above or below the baseline")
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

            xai_submit = st.form_submit_button("Explain This Prediction", use_container_width=True)

        if xai_submit:
            payload = {
                "Date": x_date.strftime("%Y-%m-%d"),
                "Occupancy_Rate": float(x_occ),
                "ADR": float(x_adr),
                "Available_Rooms": int(x_avail),
                "Reserved_Rooms": int(x_res),
                "Total_Costs": float(x_costs),
            }

            with st.spinner("Computing local TreeExplainer SHAP values..."):
                ok_x_loc, loc_res, err_x_loc = api_client.explain_instance(payload)

            if ok_x_loc and loc_res:
                res_kpi1, res_kpi2, res_kpi3 = st.columns(3)
                with res_kpi1:
                    render_kpi_card("Baseline Expectation", f"${loc_res.get('base_value', 0):,.2f}", subtext="E[f(X)] benchmark")
                with res_kpi2:
                    adj = loc_res.get("net_shap_adjustment", 0)
                    render_kpi_card("Net SHAP Adjustment", f"{adj:+,.2f}", delta_color="positive" if adj >= 0 else "negative", subtext="Sum of feature attributions")
                with res_kpi3:
                    render_kpi_card("Predicted Revenue", f"${loc_res.get('predicted_revenue', 0):,.2f}", subtext="Final model inference point")

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                
                # Positive and Negative Drivers
                pos_col, neg_col = st.columns(2)
                with pos_col:
                    render_section_header("Drivers Increasing Revenue")
                    for p in loc_res.get("top_positive_features", [])[:4]:
                        render_alert(f"<b>{p['feature']}</b> ({p['feature_value']}): +${p['shap_value']:,.2f}", alert_type="success")
                with neg_col:
                    render_section_header("Drivers Decreasing Revenue")
                    for n in loc_res.get("top_negative_features", [])[:4]:
                        render_alert(f"<b>{n['feature']}</b> ({n['feature_value']}): -${abs(n['shap_value']):,.2f}", alert_type="danger")

                # Full contributions bar chart
                all_c = loc_res.get("all_feature_contributions", [])
                if all_c:
                    render_section_header("Feature Contribution Breakdown ($)")
                    c_df = pd.DataFrame(all_c)
                    c_df["color"] = c_df["shap_value"].apply(lambda v: "#16A34A" if v > 0 else "#DC2626")
                    c_df = c_df.sort_values("shap_value", ascending=True)
                    fig_loc = go.Figure(go.Bar(
                        x=c_df["shap_value"],
                        y=c_df["feature"],
                        orientation="h",
                        marker_color=c_df["color"],
                    ))
                    apply_saas_theme(fig_loc, height=380, x_title="SHAP Value ($ Difference from Baseline)", y_title="Feature")
                    st.plotly_chart(fig_loc, use_container_width=True)
            else:
                render_alert(f"Failed to explain instance: {err_x_loc}", alert_type="danger")

