"""
Competitor Rate Benchmarking & Market Price Index (MPI) View.
"""

from datetime import date
from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders competitive pricing analysis, MPI positioning, and rate observation logging."""
    
    comp_tab1, comp_tab2 = st.tabs(["MPI Market Position Calculator", "Record New Comp-Set Rate"])

    with comp_tab1:
        render_section_header("Evaluate Competitive Position Index", "Calculate MPI and rate variance relative to your primary comp-set")
        c_eval1, c_eval2 = st.columns(2)
        with c_eval1:
            eval_our_adr = st.number_input("Our Current ADR ($)", min_value=30.0, max_value=1000.0, value=175.0, step=5.0, key="eval_our_adr")
        with c_eval2:
            eval_comp_adr = st.number_input("Competitor Average ADR ($)", min_value=30.0, max_value=1000.0, value=165.0, step=5.0, key="eval_comp_adr")

        if st.button("Calculate Market Price Index", key="btn_calc_mpi"):
            ok_eval, mpi_data, mpi_err = api_client.evaluate_competitor_pricing(our_adr=eval_our_adr, competitor_adr=eval_comp_adr)
            if ok_eval and mpi_data:
                act_comp = mpi_data.get("active_comparison", {})
                mp1, mp2, mp3, mp4 = st.columns(4)
                with mp1:
                    mpi_val = act_comp.get("market_price_index_mpi", 100)
                    render_kpi_card("Market Price Index (MPI)", f"{mpi_val:.1f}%", subtext="Our ADR / Comp ADR * 100")
                with mp2:
                    pos = act_comp.get("position_classification", "Parity")
                    render_kpi_card("Market Position", str(pos), subtext="Rate tier positioning")
                with mp3:
                    gap_d = act_comp.get("dollar_gap", 0)
                    render_kpi_card("Rate Gap ($)", f"${gap_d:+,.2f}", subtext="Dollar differential")
                with mp4:
                    gap_p = act_comp.get("variance_pct", 0)
                    render_kpi_card("Rate Variance (%)", f"{gap_p:+,.1f}%", subtext="Percentage spread")

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                render_alert(f"<b>Strategic Position Guidance:</b> {act_comp.get('suggested_action')}", alert_type="info")
            else:
                render_alert(f"Failed to calculate MPI: {mpi_err}", alert_type="danger")

    with comp_tab2:
        render_section_header("Log Authorized Competitor Rate Observation", "Record comp-set spot rates for historical tracking")
        with st.form("add_comp_record_form"):
            cr_c1, cr_c2 = st.columns(2)
            with cr_c1:
                comp_date = st.date_input("Observation Date", value=date.today())
                comp_name = st.text_input("Competitor Hotel Name", value="Grand Royal Hotel")
                comp_room = st.selectbox("Room Tier", ["Standard Deluxe", "Executive Suite", "Ocean View", "City Double"])
            with cr_c2:
                comp_rate = st.number_input("Competitor Rate ($)", min_value=30.0, max_value=2000.0, value=180.0, step=5.0)
                our_rate = st.number_input("Our Matching Rate ($)", min_value=30.0, max_value=2000.0, value=175.0, step=5.0)
                comp_notes = st.text_input("Notes (e.g. includes breakfast, promotional sale)", value="Includes complimentary breakfast")

            submit_comp = st.form_submit_button("Save Competitor Rate Observation", use_container_width=True)

        if submit_comp:
            ok_save, save_res, save_err = api_client.add_competitor_record(
                date_str=comp_date.strftime("%Y-%m-%d"),
                competitor_name=comp_name,
                competitor_adr=float(comp_rate),
                our_adr=float(our_rate),
                room_type=comp_room,
                notes=comp_notes,
            )
            if ok_save:
                render_alert(f"Competitor benchmark for {comp_name} logged successfully! (MPI: {save_res.get('record', {}).get('mpi'):.1f}%)", alert_type="success")
            else:
                render_alert(f"Failed to log competitor record: {save_err}", alert_type="danger")

    # Stored Historical Competitor Benchmarks Table
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    render_section_header("Historical Comp-Set Rate Observations", "Previously recorded competitor benchmarks")
    ok_recs, recs_data, _ = api_client.get_competitor_records()
    if ok_recs and recs_data:
        records = recs_data.get("records", []) or recs_data.get("recent_records", [])
        if records:
            rec_df = pd.DataFrame(records)
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.info("No competitor rate records logged yet. Use the form above to record comp-set observations.")

