"""
Occupancy Calendar & Reservation Audit View.
"""

from datetime import date, timedelta
from typing import Any, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders 14-day multi-day occupancy timeline and active reservation roster."""
    
    c1, c2 = st.columns([2, 2])
    with c1:
        cal_date = st.date_input("Anchor Date", value=date.today(), key="occ_cal_date")
    with c2:
        cal_cat = st.selectbox(
            "Room Category Filter",
            ["All Categories", "Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"],
            index=0,
            key="occ_cal_cat",
        )

    filter_arg = None if cal_cat == "All Categories" else cal_cat

    # 14-Day Timeline
    render_section_header("Multi-Day Occupancy & Capacity Timeline", "Confirmed reservations vs effective capacity across a 14-day window")
    
    start_d = cal_date - timedelta(days=2)
    ok_occ, occ_data, occ_err = api_client.get_occupancy_timeline(
        start_date=start_d.strftime("%Y-%m-%d"),
        days=14,
        room_category=filter_arg,
    )

    if ok_occ and occ_data:
        occ_timeline = occ_data.get("timeline", [])
        if occ_timeline:
            occ_df = pd.DataFrame(occ_timeline)
            fig_occ = px.bar(
                occ_df,
                x="date",
                y="confirmed_reservations",
                color="occupancy_rate_pct",
                color_continuous_scale=["#93C5FD", "#2563EB", "#1E3A8A"],
                text="confirmed_reservations",
            )
            # Add capacity benchmark line
            eff_val = occ_timeline[0].get("effective_capacity", 50)
            fig_occ.add_hline(y=eff_val, line_dash="dash", line_color="#DC2626", annotation_text=f"Capacity ({eff_val})")
            fig_occ.update_traces(textposition="outside")
            fig_occ.update_layout(coloraxis_showscale=False)
            apply_saas_theme(fig_occ, height=350, x_title="Date", y_title="Confirmed Bookings")
            st.plotly_chart(fig_occ, use_container_width=True)

    # Active Confirmed Reservations on Selected Date
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    render_section_header(f"Active Confirmed Reservations ({cal_date.strftime('%Y-%m-%d')})", "Reservation ledger with individual calibrated cancellation probabilities")
    
    ok_res, res_data, res_err = api_client.get_active_reservations(
        target_date=cal_date.strftime("%Y-%m-%d"),
        room_category=filter_arg,
    )

    if ok_res and res_data:
        res_list = res_data.get("reservations", [])
        st.caption(f"Displaying **{len(res_list)}** active confirmed reservations.")
        if res_list:
            r_sample_df = pd.DataFrame([
                {
                    "Booking ID": r.get("booking_id"),
                    "Guest Name": r.get("guest_name"),
                    "Room Category": r.get("room_category"),
                    "Check-In": r.get("arrival_date"),
                    "Check-Out": r.get("departure_date"),
                    "Stay Nights": r.get("stay_nights"),
                    "Booked ADR ($)": f"${r.get('booked_adr', 0):.2f}",
                    "Channel": r.get("booking_channel"),
                    "Deposit": r.get("deposit_type"),
                    "Calibrated Cancel Risk": f"{r.get('calibrated_cancellation_pct', 0):.1f}%",
                }
                for r in res_list
            ])
            st.dataframe(r_sample_df, use_container_width=True)
        else:
            render_alert("No active reservations found for the selected date and category.", alert_type="info")
    else:
        render_alert(f"Failed to fetch active reservations: {res_err}", alert_type="danger")

