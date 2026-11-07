"""
Room Inventory View - Physical Capacity & Maintenance Tracking.
"""

from datetime import date, timedelta
from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders room capacity management, out-of-order tracking, and category configurations."""
    
    render_section_header("Real-Time Room Category Availability & Capacity", "Physical room capacities and out-of-order maintenance tracking")
    
    io_c1, io_c2 = st.columns([2, 2])
    with io_c1:
        ui_target_date = st.date_input(
            "Target Stay Date",
            value=date.today() + timedelta(days=7),
            key="inv_target_date",
        )
    with io_c2:
        ui_cat_filter = st.selectbox(
            "Room Category Filter",
            ["All Categories", "Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"],
            index=0,
            key="inv_cat_filter",
        )

    filter_arg = None if ui_cat_filter == "All Categories" else ui_cat_filter
    ok_avail, avail_data, avail_err = api_client.get_inventory_availability(
        target_date=ui_target_date.strftime("%Y-%m-%d"),
        room_category=filter_arg,
    )

    if ok_avail and avail_data:
        summ = avail_data.get("summary", {})
        
        # Capacity KPIs
        ak1, ak2, ak3, ak4, ak5 = st.columns(5)
        with ak1:
            render_kpi_card("Physical Rooms", f"{summ.get('total_physical_rooms', 0):,}", subtext="Built inventory")
        with ak2:
            oos = summ.get('rooms_out_of_service', 0)
            render_kpi_card("Out of Service (OOS)", f"{oos:,}", delta_color="warning" if oos > 0 else "positive", subtext="Under maintenance")
        with ak3:
            render_kpi_card("Effective Capacity", f"{summ.get('total_effective_capacity', 0):,}", subtext="Saleable rooms")
        with ak4:
            render_kpi_card("Confirmed Bookings", f"{summ.get('total_confirmed_reservations', 0):,}", subtext="Active reservations")
        with ak5:
            occ_p = summ.get('overall_occupancy_rate_pct', 0)
            render_kpi_card("Current Occupancy", f"{occ_p:.1f}%", subtext="Confirmed / Effective")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_section_header("Inventory Breakdown by Room Category", "Capacity, active bookings, baseline ADR, and relocation walk costs per room tier")
        
        cats_list = avail_data.get("category_breakdown", [])
        if cats_list:
            cat_df = pd.DataFrame(cats_list)
            st.dataframe(
                cat_df[[
                    "room_category", "total_physical_rooms", "rooms_out_of_service",
                    "effective_capacity", "confirmed_reservations", "available_rooms",
                    "occupancy_rate_pct", "base_adr", "walk_relocation_cost"
                ]],
                use_container_width=True,
            )

        # Inventory Adjustment Form
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_section_header("Update Room Inventory Configuration", "Adjust total physical capacity or maintenance out-of-order count")
        
        with st.form("update_inv_form"):
            u_c1, u_c2, u_c3 = st.columns(3)
            with u_c1:
                edit_cat = st.selectbox("Category to Update", ["Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"])
            with u_c2:
                edit_phys = st.number_input("New Physical Capacity", min_value=1, max_value=500, value=50, step=1)
            with u_c3:
                edit_oos = st.number_input("Rooms Out of Service (OOS)", min_value=0, max_value=50, value=2, step=1)

            submit_inv = st.form_submit_button("Update Room Category Inventory", use_container_width=True)

        if submit_inv:
            with st.spinner("Updating room category configuration..."):
                ok_up, up_res, up_err = api_client.update_room_inventory(
                    room_category=edit_cat,
                    physical_capacity=int(edit_phys),
                    out_of_service=int(edit_oos),
                )
            if ok_up:
                render_alert(f"Successfully updated inventory for {edit_cat}! Effective capacity: {edit_phys - edit_oos} rooms.", alert_type="success")
                st.rerun()
            else:
                render_alert(f"Failed to update room inventory: {up_err}", alert_type="danger")
    else:
        render_alert(f"Failed to fetch room inventory: {avail_err}", alert_type="danger")

