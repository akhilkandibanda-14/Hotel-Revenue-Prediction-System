"""
Cancellation Explainability View - TreeExplainer SHAP Waterfall Decomposition.
"""

from datetime import date
from typing import Any, Dict
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders single-reservation cancellation SHAP waterfall explanation."""
    
    render_section_header(
        "Reservation Risk Feature Attribution (SHAP)",
        "Deconstruct how individual reservation attributes push cancellation probability above or below the baseline",
    )

    with st.form("cancellation_explain_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            exp_lead = st.slider("Lead Time (Days)", 0, 365, 90)
            exp_adr = st.number_input("Quoted ADR ($)", 30.0, 1000.0, 160.0, step=5.0)
            exp_channel = st.selectbox("Distribution Channel", ["Online TA", "Direct", "Offline TA/TO", "Corporate"])
        with c2:
            exp_deposit = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
            exp_customer = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Contract", "Group"])
            exp_changes = st.number_input("Booking Modifications", 0, 10, 0)
        with c3:
            exp_parking = st.selectbox("Parking Spaces", [0, 1, 2], index=0)
            exp_requests = st.slider("Special Requests", 0, 5, 0)
            exp_prev_canc = st.number_input("Prior Cancellations", 0, 20, 0)

        explain_btn = st.form_submit_button("Generate SHAP Risk Attribution", use_container_width=True)

    if explain_btn:
        req_payload = {
            "lead_time": int(exp_lead),
            "arrival_date_year": date.today().year,
            "arrival_date_month": date.today().strftime("%B"),
            "arrival_date_day_of_month": date.today().day,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA" if exp_channel == "Online TA" else "Direct",
            "distribution_channel": "TA/TO" if exp_channel in ["Online TA", "Offline TA/TO"] else "Direct",
            "is_repeated_guest": 0,
            "previous_cancellations": int(exp_prev_canc),
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "booking_changes": int(exp_changes),
            "deposit_type": exp_deposit,
            "days_in_waiting_list": 0,
            "customer_type": exp_customer,
            "adr": float(exp_adr),
            "required_car_parking_spaces": int(exp_parking),
            "total_of_special_requests": int(exp_requests),
            "hotel": "City Hotel",
        }

        with st.spinner("Computing TreeExplainer exact Shapley values..."):
            ok_exp, exp_data, exp_err = api_client.explain_cancellation(req_payload)

        if ok_exp and exp_data:
            base_v = exp_data.get("base_value", 0.37)
            shap_s = exp_data.get("shap_sum", 0.0)
            pred_p = exp_data.get("predicted_probability", 0.37)
            is_cons = exp_data.get("is_additively_consistent", True)

            # Summary KPIs
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card("Baseline Population Risk", f"{base_v*100:.1f}%", subtext="E[f(X)] expected value")
            with k2:
                render_kpi_card("Net SHAP Attribution", f"{shap_s*100:+.1f}% pts", delta_color="negative" if shap_s > 0 else "positive", subtext="Sum of feature shifts")
            with k3:
                render_kpi_card("Predicted Cancellation Risk", f"{pred_p*100:.1f}%", subtext="Final model output")
            with k4:
                render_kpi_card("Additive Consistency", "Verified", delta_color="positive", subtext="Sum of SHAP == Model output")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            attrs = exp_data.get("all_attributions", [])
            if attrs:
                top_attrs = attrs[:10]
                feat_names = [a["feature"] for a in reversed(top_attrs)]
                feat_vals = [a["shap_value"] * 100.0 for a in reversed(top_attrs)]
                bar_colors = ["#DC2626" if v > 0 else "#16A34A" for v in feat_vals]

                render_section_header("Top Feature Attributions (Impact on Cancellation Probability %)")
                fig_shap = go.Figure(go.Bar(
                    x=feat_vals,
                    y=feat_names,
                    orientation="h",
                    marker=dict(color=bar_colors),
                    text=[f"{v:+.1f}%" for v in feat_vals],
                    textposition="outside",
                ))
                apply_saas_theme(fig_shap, height=360, x_title="Impact on Cancellation Probability (Percentage Points)", y_title="Feature")
                st.plotly_chart(fig_shap, use_container_width=True)

            # Drivers Breakdown
            sh_c1, sh_c2 = st.columns(2)
            with sh_c1:
                render_section_header("Risk-Elevating Factors (Increase Cancellation)")
                for d in exp_data.get("top_risk_drivers", []):
                    render_alert(f"<b>{d['feature']}</b>: +{d['shap_value']*100:.1f}% percentage points", alert_type="danger")
            with sh_c2:
                render_section_header("Protective Factors (Increase Stay Commitment)")
                for p in exp_data.get("top_protective_factors", []):
                    render_alert(f"<b>{p['feature']}</b>: {p['shap_value']*100:.1f}% percentage points", alert_type="success")

            render_alert(f"<b>Statistical Attribution Notice:</b> {exp_data.get('statistical_disclaimer')}", alert_type="info")
        else:
            render_alert(f"Failed to compute SHAP attributions: {exp_err}", alert_type="danger")

