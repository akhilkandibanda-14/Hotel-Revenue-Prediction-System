"""
ADR & Price Elasticity Optimization View.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders dynamic ADR price sensitivity and revenue optimization workspace."""
    
    render_section_header("Hotel Cost & Base Operational Assumptions", "Configure pricing parameters and cost structures for sensitivity modeling")
    
    # Defaults from dataset
    def_adr = 165.0
    if "Average_Daily_Rate" in df.columns:
        def_adr = float(df["Average_Daily_Rate"].mean())
    elif "ADR" in df.columns:
        def_adr = float(df["ADR"].mean())
        
    def_occ = 0.70
    if "Occupancy_Rate" in df.columns:
        def_occ = float(df["Occupancy_Rate"].mean())
        
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        input_base_adr = st.number_input("Base ADR ($)", value=round(def_adr, 1), min_value=30.0, max_value=600.0, step=5.0)
    with p_col2:
        input_base_occ = st.slider("Base Occupancy Rate", min_value=0.10, max_value=1.0, value=round(def_occ, 2), step=0.01)
    with p_col3:
        input_rooms = st.number_input("Total Room Inventory", value=125, min_value=10, max_value=1000, step=5)
    with p_col4:
        input_elasticity = st.slider(
            "Price Elasticity (ε)",
            min_value=-1.0,
            max_value=-0.05,
            value=-0.35,
            step=0.05,
            help="Hospitality empirical demand elasticity typically ranges from -0.30 to -0.60.",
        )

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        input_fixed_costs = st.number_input("Daily Fixed Costs ($)", value=2500.0, min_value=0.0, max_value=50000.0, step=100.0)
    with c_col2:
        input_var_cost = st.number_input("Variable Cost per Occupied Room ($)", value=18.0, min_value=0.0, max_value=200.0, step=2.0)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    ok_opt, opt_res, opt_err = api_client.get_pricing_optimization(
        base_adr=input_base_adr,
        base_occupancy=input_base_occ,
        total_rooms=input_rooms,
        fixed_costs=input_fixed_costs,
        var_cost_per_room=input_var_cost,
        custom_elasticity=input_elasticity,
    )

    if ok_opt and opt_res:
        opt_rev = opt_res.get("optimal_revenue_scenario", {})
        opt_prof = opt_res.get("optimal_profit_scenario", {})
        active_eps = opt_res.get("active_elasticity", input_elasticity)

        # KPI Summary Row
        k1, k2, k3 = st.columns(3)
        with k1:
            render_kpi_card(
                "Active Elasticity (ε)",
                f"{active_eps:.2f}",
                subtext="Percentage demand change per 1% price change",
            )
        with k2:
            rev_change = opt_rev.get("adr_change_pct", 0)
            render_kpi_card(
                "Revenue Maximizing ADR",
                f"${opt_rev.get('projected_adr', 0):,.2f}",
                delta=f"{rev_change:+d}% ADR change",
                delta_color="positive" if rev_change >= 0 else "neutral",
                subtext=f"Expected Daily Revenue: ${opt_rev.get('projected_revenue', 0):,.0f}",
            )
        with k3:
            prof_change = opt_prof.get("adr_change_pct", 0)
            render_kpi_card(
                "Profit Maximizing ADR",
                f"${opt_prof.get('projected_adr', 0):,.2f}",
                delta=f"{prof_change:+d}% ADR change",
                delta_color="positive" if prof_change >= 0 else "neutral",
                subtext=f"Expected Daily Gross Profit: ${opt_prof.get('projected_profit', 0):,.0f}",
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Dual Curve Chart: ADR vs Projected Daily Revenue and Daily Profit
        render_section_header("Price Sensitivity Curves", "Revenue and profit response curves across +/- 30% ADR variations")
        scenarios = opt_res.get("scenarios", [])
        scen_df = pd.DataFrame(scenarios)
        if not scen_df.empty:
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=scen_df["projected_adr"],
                y=scen_df["projected_revenue"],
                mode="lines+markers",
                name="Projected Daily Revenue ($)",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=6, color="#2563EB"),
            ))
            fig_opt.add_trace(go.Scatter(
                x=scen_df["projected_adr"],
                y=scen_df["projected_profit"],
                mode="lines+markers",
                name="Projected Daily Profit ($)",
                line=dict(color="#0F766E", width=2.5, dash="dash"),
                marker=dict(size=6, color="#0F766E"),
            ))
            
            # Marker for optimal revenue ADR
            opt_adr_val = opt_rev.get("projected_adr", input_base_adr)
            fig_opt.add_vline(x=opt_adr_val, line_dash="dot", line_color="#D97706", annotation_text="Revenue Max ADR")

            apply_saas_theme(fig_opt, height=360, x_title="Average Daily Rate ($)", y_title="Projected Amount ($)")
            st.plotly_chart(fig_opt, use_container_width=True)

            # Scenario Simulation Matrix
            render_section_header("Scenario Simulation Matrix", "Comprehensive breakdown across simulated price adjustments")
            display_df = scen_df[[
                "adr_change_pct", "projected_adr", "projected_occupancy",
                "rooms_sold", "projected_revenue", "total_costs", "projected_profit",
                "revenue_lift", "profit_lift"
            ]].copy()
            display_df.columns = [
                "ADR Change (%)", "ADR ($)", "Occupancy",
                "Rooms Sold", "Revenue ($)", "Total Costs ($)", "Profit ($)",
                "Revenue Lift ($)", "Profit Lift ($)"
            ]
            st.dataframe(display_df, use_container_width=True)

            # Important Economic Assumptions & Disclaimers
            render_alert(
                "<b>Model Assumption Notice:</b> Sensitivity projections assume constant price elasticity within the local demand curve. "
                "Elasticity estimates reflect historical correlation and should not be interpreted as absolute causal proof. "
                "Large boundary price changes (+30%) should be implemented incrementally with A/B rate testing.",
                alert_type="warning",
            )

            with st.expander("Microeconomic Assumptions, Elasticity & Boundary Dynamics", expanded=False):
                for asm in opt_res.get("assumptions_and_constraints", []):
                    st.markdown(f"• {asm}")
    else:
        render_alert(f"Failed to run pricing optimization: {opt_err}", alert_type="danger")

