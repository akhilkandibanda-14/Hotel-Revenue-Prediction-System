"""
Constrained Overbooking Optimizer View - Revenue Maximization with Hard Service-Level Walk Risk Control.
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
    """Renders constrained overbooking optimizer and portfolio-wide recommendations."""
    
    render_section_header(
        "Constrained Overbooking Revenue Optimization",
        "Maximize expected net contribution subject to hard walk probability tolerance constraint P(Walk) <= α",
    )

    opt_c1, opt_c2, opt_c3 = st.columns(3)
    with opt_c1:
        ob_target_date = st.date_input(
            "Arrival Date",
            value=date.today() + timedelta(days=7),
            key="ob_opt_date",
        )
    with opt_c2:
        ob_category = st.selectbox(
            "Target Room Category",
            ["Standard Room", "Deluxe Room", "Executive Suite", "Presidential Suite"],
            index=0,
            key="ob_opt_cat",
        )
    with opt_c3:
        ob_walk_risk = st.slider(
            "Max Walk Risk Tolerance (α %)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Hard service-level risk threshold: probability of arriving guests exceeding room capacity must not exceed α.",
            key="ob_walk_risk_slider",
        )

    ok_opt, opt_res, opt_err = api_client.optimize_overbooking(
        target_date=ob_target_date.strftime("%Y-%m-%d"),
        room_category=ob_category,
        max_walk_risk_pct=float(ob_walk_risk),
    )

    if ok_opt and opt_res:
        rec = opt_res.get("recommendation", {})
        impact = opt_res.get("business_impact_vs_no_overbooking", {})
        assump = opt_res.get("assumptions", {})

        # Primary KPIs
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            add_b = rec.get("safe_additional_bookings", 0)
            render_kpi_card(
                "Safe Additional Bookings",
                f"+{add_b} Rooms",
                delta="Optimal quota",
                delta_color="positive",
                subtext="Within walk risk limit",
            )
        with rk2:
            render_kpi_card(
                "Recommended Booking Limit",
                f"{rec.get('recommended_total_booking_limit', 0)} Bookings",
                subtext=f"Effective capacity: {assump.get('effective_capacity', 0)} rooms",
            )
        with rk3:
            walk_p = rec.get("walk_probability_pct", 0)
            render_kpi_card(
                "Walk Probability",
                f"{walk_p:.2f}%",
                delta=f"Limit: {ob_walk_risk:.1f}%",
                delta_color="positive" if walk_p <= ob_walk_risk else "negative",
                subtext="P(Arriving Guests > Capacity)",
            )
        with rk4:
            net_gain = impact.get("net_revenue_gain_dollars", 0)
            render_kpi_card(
                "Expected Net Revenue Gain",
                f"${net_gain:,.2f}",
                delta="Net of walk costs",
                delta_color="positive",
                subtext="Incremental profit contribution",
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        render_alert(f"<b>Strategic Recommendation:</b> {impact.get('strategic_summary')}", alert_type="success")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Dual Visualizations: PMF vs Net Contribution Curve
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            render_section_header("Arrival PMF Distribution")
            ok_fc, fc_data, _ = api_client.get_cancellation_forecast(
                target_date=ob_target_date.strftime("%Y-%m-%d"),
                room_category=ob_category,
            )
            if ok_fc and fc_data:
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
                    eff_cap = fc_data.get("effective_capacity", 50)
                    fig_pmf.add_vline(x=eff_cap, line_dash="dash", line_color="#DC2626", annotation_text=f"Cap ({eff_cap})")
                    apply_saas_theme(fig_pmf, height=330, x_title="Arriving Guests", y_title="Probability P(A = k)")
                    st.plotly_chart(fig_pmf, use_container_width=True)

        with v_col2:
            render_section_header("Expected Net Contribution Curve")
            grid = opt_res.get("candidate_evaluation_grid", [])
            if grid:
                grid_df = pd.DataFrame(grid)
                fig_enc = go.Figure()
                fig_enc.add_trace(go.Scatter(
                    x=grid_df["additional_bookings"],
                    y=grid_df["expected_net_contribution"],
                    mode="lines+markers",
                    name="Expected Net Contribution ($)",
                    line=dict(color="#16A34A", width=2.5),
                    marker=dict(size=6, color="#16A34A"),
                ))
                best_add = rec.get("safe_additional_bookings", 0)
                fig_enc.add_vline(x=best_add, line_dash="dash", line_color="#2563EB", annotation_text=f"Optimal: +{best_add}")
                apply_saas_theme(fig_enc, height=330, x_title="Additional Bookings Accepted", y_title="Net Contribution ($)")
                st.plotly_chart(fig_enc, use_container_width=True)

        # Full Candidate Booking Limit Evaluation Grid
        render_section_header("Candidate Booking Limit Evaluation Grid", "Detailed evaluation across additional booking increments")
        if grid:
            st.dataframe(
                grid_df[[
                    "additional_bookings", "total_booking_limit", "expected_arrivals",
                    "expected_revenue", "expected_walks", "expected_walk_cost",
                    "expected_net_contribution", "walk_probability_pct", "risk_tier", "constraint_satisfied"
                ]],
                use_container_width=True,
            )

        # Portfolio-Wide Recommendations
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_section_header(f"Portfolio-Wide Overbooking Limits ({ob_target_date.strftime('%Y-%m-%d')})")
        ok_port, port_data, _ = api_client.get_portfolio_overbooking_recommendations(target_date=ob_target_date.strftime("%Y-%m-%d"))
        if ok_port and port_data:
            port_recs = port_data.get("portfolio_recommendations", [])
            if port_recs:
                port_df = pd.DataFrame([
                    {
                        "Room Category": r["room_category"],
                        "Capacity": r["effective_capacity"],
                        "Confirmed": r["confirmed_reservations"],
                        "Expected Cancels": f"{r['expected_cancellations']:.1f}",
                        "Safe Overbook (Quota)": f"+{r['recommended_additional_bookings']} rooms",
                        "Total Booking Limit": r["recommended_booking_limit"],
                        "Walk Risk": f"{r['walk_probability_pct']:.2f}%",
                        "Net Revenue Lift": f"${r['expected_net_gain']:+,.2f}",
                    }
                    for r in port_recs
                ])
                st.dataframe(port_df, use_container_width=True)
    else:
        render_alert(f"Failed to optimize overbooking: {opt_err}", alert_type="danger")

