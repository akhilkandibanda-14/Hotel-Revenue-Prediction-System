"""
What-If Policy Simulator View - Counterfactual Intervention Analysis.
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
    """Renders 16-scenario counterfactual policy evaluator across operational levers."""
    
    render_section_header(
        "Counterfactual Policy What-If Simulator",
        "Simulate how policy adjustments (deposits, lead time limits, parking, requests) alter cancellation risk",
    )

    with st.form("whatif_base_form"):
        w_c1, w_c2, w_c3 = st.columns(3)
        with w_c1:
            base_lead = st.slider("Base Lead Time (Days)", 0, 365, 60)
            base_deposit = st.selectbox("Base Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
        with w_c2:
            base_channel = st.selectbox("Base Channel", ["Online TA", "Direct", "Offline TA/TO", "Corporate"])
            base_customer = st.selectbox("Base Customer Segment", ["Transient", "Transient-Party", "Contract", "Group"])
        with w_c3:
            base_adr = st.number_input("Base ADR ($)", 30.0, 1000.0, 150.0, step=5.0)
            base_requests = st.slider("Base Special Requests", 0, 5, 0)

        run_sim = st.form_submit_button("Simulate 16 Operational Policy Scenarios", use_container_width=True)

    if run_sim:
        req_payload = {
            "lead_time": int(base_lead),
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
            "market_segment": "Online TA" if base_channel == "Online TA" else "Direct",
            "distribution_channel": "TA/TO" if base_channel in ["Online TA", "Offline TA/TO"] else "Direct",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "booking_changes": 0,
            "deposit_type": base_deposit,
            "days_in_waiting_list": 0,
            "customer_type": base_customer,
            "adr": float(base_adr),
            "required_car_parking_spaces": 0,
            "total_of_special_requests": int(base_requests),
            "hotel": "City Hotel",
        }

        with st.spinner("Simulating counterfactual policies through Random Forest calibrator..."):
            ok_wi, wi_data, wi_err = api_client.simulate_cancellation_what_if(req_payload)

        if ok_wi and wi_data:
            base_p = wi_data.get("baseline_probability", 0.35)
            scens = wi_data.get("scenarios", [])

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card("Baseline Policy Risk", f"{base_p*100:.1f}%", subtext="Starting reservation probability")
            with k2:
                render_kpi_card("Scenarios Evaluated", str(len(scens)), subtext="Discrete counterfactual runs")
            with k3:
                max_red = min([s["probability_delta_pct_pts"] for s in scens], default=0)
                render_kpi_card("Max Risk Reduction", f"{max_red:+.1f}% pts", delta_color="positive", subtext="Best mitigating intervention")
            with k4:
                render_kpi_card("Causal Verification", "Statistical Association", subtext="Observational ML model output")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Chart of Simulated Shifts
            scen_names = [s["scenario_name"] for s in scens]
            deltas = [s["probability_delta_pct_pts"] for s in scens]
            delta_colors = ["#DC2626" if d > 0 else "#16A34A" for d in deltas]

            render_section_header("Policy Sensitivity Impact (Percentage Point Shift in Cancellation Risk)")
            fig_wi = go.Figure(go.Bar(
                x=deltas,
                y=scen_names,
                orientation="h",
                marker=dict(color=delta_colors),
                text=[f"{d:+.1f}% pts" for d in deltas],
                textposition="outside",
            ))
            apply_saas_theme(fig_wi, height=480, x_title="Risk Delta (pts)", y_title="Policy Scenario")
            st.plotly_chart(fig_wi, use_container_width=True)

            # Tabular breakdown
            render_section_header("Comprehensive Scenario Breakdown Matrix")
            wi_df = pd.DataFrame([
                {
                    "Lever": s["lever_category"],
                    "Policy Scenario": s["scenario_name"],
                    "Simulated Risk (%)": f"{s['simulated_probability_pct']:.1f}%",
                    "Probability Shift": f"{s['probability_delta_pct_pts']:+.1f}% pts",
                    "Direction": s["impact_badge"] + " " + s["impact_direction"],
                    "Operational Takeaway": s["operational_takeaway"],
                }
                for s in scens
            ])
            st.dataframe(wi_df, use_container_width=True)

            render_alert(f"<b>Causal Limitation Notice:</b> {wi_data.get('causal_disclaimer')}", alert_type="info")
        else:
            render_alert(f"Could not load what-if simulations: {wi_err}", alert_type="danger")
    else:
        render_alert("Click <b>'Simulate 16 Operational Policy Scenarios'</b> above to test policy adjustments.", alert_type="info")

