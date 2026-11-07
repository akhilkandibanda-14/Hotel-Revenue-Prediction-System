"""
Pre-Booking Revenue Estimator View - Ex-Ante Pricing Engine.
"""

from datetime import date, timedelta
from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders ex-ante pre-booking inquiry evaluation engine with zero target leakage."""
    
    render_alert(
        "<b>Strict Anti-Leakage Protocol Active:</b> Post-stay realized metrics "
        "(<code>Reserved_Rooms</code>, <code>RevPAR</code>, <code>Actual_Occupancy</code>, <code>Total_Costs</code>) "
        "are strictly blacklisted from this inquiry quotation pipeline.",
        alert_type="info",
    )

    # Dataset Audit status
    ok_audit, audit_res, _ = api_client.get_prebooking_audit()
    if ok_audit and audit_res:
        with st.expander("Dataset Booking-Level Audit & PMS Ingestion Status", expanded=False):
            st.markdown(f"**Genuine Booking-Level Records Present:** {audit_res.get('is_genuine_booking_level')}")
            st.markdown(f"**Contains Realized Leakage Fields:** {audit_res.get('contains_realized_leakage_fields')}")
            st.markdown(f"**Identified Realized Fields:** {', '.join(audit_res.get('leakage_fields_identified', []))}")
            st.warning(f"**Missing PMS Transaction Fields:** {', '.join(audit_res.get('missing_required_fields', []))}")
            st.caption(f"Assessment Summary: {audit_res.get('assessment_summary')}")

    render_section_header("Inquiry Reservation Parameters", "Enter prospective guest booking specifications")
    with st.form("prebooking_quote_form"):
        st.markdown("##### 1. Stay Schedule & Timeline")
        pb_c1, pb_c2, pb_c3 = st.columns(3)
        with pb_c1:
            pb_in = st.date_input("Check-In Date", value=date.today())
        with pb_c2:
            pb_out = st.date_input("Check-Out Date", value=date.today() + timedelta(days=3))
        with pb_c3:
            pb_book = st.date_input("Booking Creation Date", value=date.today())

        st.markdown("##### 2. Reservation & Channel Attributes")
        pb_c4, pb_c5, pb_c6 = st.columns(3)
        with pb_c4:
            pb_rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=1, step=1)
            pb_guests = st.number_input("Total Guests", min_value=1, max_value=40, value=2, step=1)
        with pb_c5:
            pb_room_type = st.selectbox("Room Category / Tier", ["Standard", "Deluxe", "Suite", "Executive"])
            pb_channel = st.selectbox("Booking Channel", ["Direct", "OTA", "Corporate", "Online"])
        with pb_c6:
            pb_segment = st.selectbox("Market Segment", ["Leisure", "Corporate", "Wholesale", "Government"])
            pb_quoted_adr = st.number_input("Quoted ADR ($ / room / night)", min_value=30.0, max_value=1500.0, value=175.0, step=5.0)

        pb_submit = st.form_submit_button("Calculate Expected Booking Value", use_container_width=True)

    if pb_submit:
        if pb_out <= pb_in:
            render_alert("Check-out date must be strictly after check-in date.", alert_type="danger")
        else:
            with st.spinner("Computing ex-ante booking expected valuation..."):
                ok_pb, pb_data, pb_err = api_client.estimate_prebooking_revenue(
                    check_in_date=pb_in.strftime("%Y-%m-%d"),
                    check_out_date=pb_out.strftime("%Y-%m-%d"),
                    booking_date=pb_book.strftime("%Y-%m-%d"),
                    quoted_adr=float(pb_quoted_adr),
                    rooms_count=int(pb_rooms),
                    guests_count=int(pb_guests),
                    booking_channel=pb_channel,
                    market_segment=pb_segment,
                    room_type=pb_room_type,
                )
            if ok_pb and pb_data:
                b_det = pb_data.get("booking_details", {})
                
                # Result KPIs
                pk1, pk2, pk3, pk4 = st.columns(4)
                with pk1:
                    render_kpi_card(
                        "Estimated Gross Revenue",
                        f"${pb_data.get('estimated_gross_revenue', 0):,.2f}",
                        subtext="Ex-ante quoted stay total",
                    )
                with pk2:
                    render_kpi_card(
                        "Estimated Net Revenue",
                        f"${pb_data.get('estimated_net_revenue', 0):,.2f}",
                        subtext="Net of channel acquisition costs",
                    )
                with pk3:
                    render_kpi_card(
                        "Length of Stay",
                        f"{b_det.get('length_of_stay_nights', 0)} nights",
                        subtext="Duration of reservation",
                    )
                with pk4:
                    render_kpi_card(
                        "Lead Time",
                        f"{b_det.get('lead_time_days', 0)} days",
                        subtext="Advance booking window",
                    )

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                
                # Breakdown columns
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    render_section_header("Rate Multipliers & Channel Impact")
                    st.write(f"• **Channel Commission Cost:** ${pb_data.get('estimated_commission_cost', 0):,.2f}")
                    st.write(f"• **Room Tier Selected:** {pb_room_type}")
                    st.write(f"• **Arrival Day of Week:** {b_det.get('checkin_day_of_week')}")
                    st.write(f"• **Weekend Arrival:** {'Yes' if b_det.get('is_weekend_arrival') else 'No'}")
                with b_col2:
                    render_section_header("Governance & Model Attribution")
                    st.write(f"• **Model Status:** {pb_data.get('model_type')}")
                    st.write(f"• **Trained on Guest-Level PMS:** {'Yes' if pb_data.get('is_trained_on_booking_level_data') else 'No (Ex-Ante Expected Value Model)'}")
                    st.caption(f"Governance Notice: {pb_data.get('data_governance_notice')}")
            else:
                render_alert(f"Failed to calculate pre-booking estimate: {pb_err}", alert_type="danger")

