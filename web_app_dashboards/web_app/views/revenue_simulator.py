"""
Revenue Impact Simulator View - Interactive Custom Policy & Risk Sensitivity.
"""

from datetime import date, timedelta
from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders custom management policy scenario simulator for room overbooking and relocation costs."""
    
    render_section_header(
        "Custom Hotel Management Policy & Risk Scenario Simulator",
        "Adjust operational parameters (rate per night, relocation penalty, risk thresholds) to evaluate net contribution vs liabilities",
    )

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_cat = st.selectbox("Category to Simulate", ["Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"], key="sim_cat")
        sim_date = st.date_input("Simulation Date", value=date.today() + timedelta(days=7), key="sim_date")
    with sim_c2:
        sim_adr = st.number_input("Average Daily Rate ($ / night)", min_value=50.0, max_value=2000.0, value=150.0, step=10.0, key="sim_adr")
        sim_walk_cost = st.number_input("Walk / Relocation Cost ($)", min_value=50.0, max_value=5000.0, value=300.0, step=25.0, help="Alternative room + transportation + compensation voucher", key="sim_walk")
    with sim_c3:
        sim_alpha = st.slider("Risk Tolerance α (%)", min_value=0.2, max_value=15.0, value=1.5, step=0.1, key="sim_alpha")
        sim_max_b = st.slider("Max Search Horizon", min_value=5, max_value=50, value=25, step=5, key="sim_max_b")

    sim_btn = st.button("Run Custom Scenario Simulation", use_container_width=True)
    if sim_btn:
        with st.spinner("Simulating custom policy across arrival convolution grid..."):
            ok_sim, sim_res, sim_err = api_client.simulate_overbooking_scenario({
                "target_date": sim_date.strftime("%Y-%m-%d"),
                "room_category": sim_cat,
                "custom_adr": float(sim_adr),
                "custom_walk_cost": float(sim_walk_cost),
                "max_walk_risk_pct": float(sim_alpha),
                "max_additional_bookings": int(sim_max_b),
            })
        if ok_sim and sim_res:
            s_rec = sim_res.get("recommendation", {})
            s_base = sim_res.get("no_overbooking_baseline", {})
            s_imp = sim_res.get("business_impact_vs_no_overbooking", {})

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            render_section_header("Scenario Simulation Comparison")
            
            sk1, sk2, sk3, sk4 = st.columns(4)
            with sk1:
                render_kpi_card("Recommended Additional", f"+{s_rec.get('safe_additional_bookings', 0)} Rooms", subtext="Safe booking allowance")
            with sk2:
                render_kpi_card("Total Booking Limit", f"{s_rec.get('recommended_total_booking_limit', 0)} Bookings", subtext="Maximum reservations to accept")
            with sk3:
                wp = s_rec.get('walk_probability_pct', 0)
                render_kpi_card("Walk Probability", f"{wp:.2f}%", delta=f"{s_rec.get('risk_tier')}", delta_color="positive" if wp <= sim_alpha else "negative", subtext=f"Tolerance: {sim_alpha:.1f}%")
            with sk4:
                render_kpi_card("Net Incremental Gain", f"${s_imp.get('net_revenue_gain_dollars', 0):,.2f}", delta="After relocation costs", delta_color="positive", subtext="Incremental profit")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            render_section_header("Walk Risk & Relocation Liability Breakdown")
            
            wk_c1, wk_c2, wk_c3, wk_c4 = st.columns(4)
            with wk_c1:
                render_kpi_card("P(0 Walks - Safe)", f"{s_rec.get('prob_zero_walks_pct', 98.0):.1f}%", subtext="Capacity sufficient")
            with wk_c2:
                render_kpi_card("P(1 Walk)", f"{s_rec.get('prob_one_walk_pct', 1.8):.2f}%", subtext="Minor spillover")
            with wk_c3:
                render_kpi_card("Expected Walks 𝔼[W]", f"{s_rec.get('expected_walks', 0):.3f}", subtext="Mean relocation incidents")
            with wk_c4:
                render_kpi_card("Expected Walk Cost", f"${s_rec.get('expected_walk_cost', 0):,.2f}", subtext="Relocation expense liability")

            render_alert(f"<b>Takeaway:</b> {s_imp.get('strategic_summary')}", alert_type="success")
        else:
            render_alert(f"Failed to simulate custom scenario: {sim_err}", alert_type="danger")
    else:
        render_alert("Click <b>'Run Custom Scenario Simulation'</b> above to evaluate custom pricing and walk penalties.", alert_type="info")

