"""
Cancellation & Arrival Probability Distribution View - Poisson-Binomial Convolution.
"""

from datetime import date, timedelta
from typing import Any, Dict
import pandas as pd
import plotly.express as px
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders date-specific expected cancellation aggregation and exact Poisson-Binomial arrival distribution."""
    
    render_section_header("Expected Cancellations & Arrival Probability Distribution", "Exact convolution of non-identical individual arrival probabilities")
    
    c1, c2 = st.columns(2)
    with c1:
        target_date = st.date_input("Target Arrival Date", value=date.today() + timedelta(days=7), key="cf_target_date")
    with c2:
        category = st.selectbox(
            "Room Category",
            ["Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"],
            index=0,
            key="cf_category",
        )

    ok_fc, fc_data, fc_err = api_client.get_cancellation_forecast(
        target_date=target_date.strftime("%Y-%m-%d"),
        room_category=category,
    )

    if ok_fc and fc_data:
        # High Level Forecast KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("Confirmed Bookings", f"{fc_data.get('total_confirmed_reservations', 0):,}", subtext=f"Total booked on {target_date}")
        with k2:
            render_kpi_card("Expected Cancellations 𝔼[C]", f"{fc_data.get('expected_cancellations', 0):.1f} rooms", subtext=f"Cancel rate: {fc_data.get('expected_cancellation_rate_pct', 0):.1f}%")
        with k3:
            render_kpi_card("Expected Arrivals 𝔼[A]", f"{fc_data.get('expected_arrivals', 0):.1f} guests", subtext="Mean arriving reservations")
        with k4:
            eff_cap = fc_data.get("effective_capacity", 50)
            render_kpi_card("Effective Capacity", f"{eff_cap} rooms", subtext="Physical capacity minus OOS")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Poisson-Binomial PMF Chart
        render_section_header(
            f"Exact Arrival PMF for {category}",
            f"Probability mass function computed via exact convolution of individual Bernoulli parameters (𝔼[A]: {fc_data.get('expected_arrivals', 0):.1f})",
        )
        pmf_list = fc_data.get("arrival_distribution_pmf", [])
        if pmf_list:
            pmf_df = pd.DataFrame(pmf_list)
            pmf_df = pmf_df[pmf_df["probability"] > 0.0001]
            fig_pmf = px.bar(
                pmf_df,
                x="arrivals",
                y="probability",
                color_discrete_sequence=["#2563EB"],
            )
            fig_pmf.add_vline(x=eff_cap, line_dash="dash", line_color="#DC2626", annotation_text=f"Capacity ({eff_cap})")
            apply_saas_theme(fig_pmf, height=360, x_title="Arriving Guests", y_title="Probability P(A = k)")
            st.plotly_chart(fig_pmf, use_container_width=True)

        # Individual Reservation Roster with calibrated probabilities
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_section_header("Active Reservation Probabilities Ledger", "Calibrated cancellation probabilities per guest")
        res_list = fc_data.get("reservations", [])
        if res_list:
            r_df = pd.DataFrame([
                {
                    "Booking ID": r.get("booking_id"),
                    "Guest Name": r.get("guest_name"),
                    "Arrival": r.get("arrival_date"),
                    "Departure": r.get("departure_date"),
                    "Booked ADR": f"${r.get('booked_adr', 0):.2f}",
                    "Channel": r.get("booking_channel"),
                    "Calibrated Cancel Risk": f"{r.get('calibrated_cancellation_pct', 0):.1f}%",
                    "Arrival Probability": f"{(1 - r.get('calibrated_cancellation_pct', 0)/100)*100:.1f}%",
                }
                for r in res_list
            ])
            st.dataframe(r_df, use_container_width=True)
    else:
        render_alert(f"Failed to fetch cancellation forecast: {fc_err}", alert_type="danger")

