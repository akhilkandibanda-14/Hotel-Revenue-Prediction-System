"""
Strategic Business Recommendations Engine View.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders cross-module algorithmic revenue and pricing recommendations."""
    
    ok_rec, rec_data, rec_err = api_client.get_business_recommendations()
    if ok_rec and rec_data:
        prio_summary = rec_data.get("priority_summary", {})
        recs = rec_data.get("recommendations", [])

        # Summary KPIs
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            render_kpi_card("Strategic Actions", str(rec_data.get("total_recommendations", 0)), subtext="Total active advisory actions")
        with rk2:
            render_kpi_card("High Priority", str(prio_summary.get("High", 0)), delta_color="negative", subtext="Immediate revenue impact")
        with rk3:
            render_kpi_card("Medium Priority", str(prio_summary.get("Medium", 0)), subtext="Tactical yield improvements")
        with rk4:
            render_kpi_card("Advisory Engine", "Active", delta_color="positive", subtext="Continuous rule synthesis")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_section_header("Prioritized Action Directives", "Synthesized across pricing models, demand pacing, cancellation risk, and drift alerts")

        for idx, r in enumerate(recs):
            prio = r.get("priority", "Medium")
            badge_class = "status-pill-danger" if prio == "High" else ("status-pill-warning" if prio == "Medium" else "status-pill-neutral")
            conf_val = r.get("confidence_level", 0.85)
            conf = conf_val * 100 if conf_val <= 1.0 else conf_val

            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span class="status-pill {badge_class}">
                            {prio.upper()} PRIORITY
                        </span>
                        <span style="color: #64748B; font-size: 0.8rem; font-weight: 500;">
                            Category: <strong style="color: #111827;">{r.get('category')}</strong> &nbsp;|&nbsp; Confidence: <strong style="color: #111827;">{conf:.0f}%</strong>
                        </span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #111827; margin: 0.35rem 0;">{r.get('title')}</div>
                    <p style="color: #475569; margin-bottom: 0.75rem; font-size: 0.88rem; line-height: 1.45;">{r.get('explanation')}</p>
                    <div style="background: #F8FAFC; border-left: 3px solid #2563EB; padding: 0.6rem 0.85rem; border-radius: 4px; font-size: 0.85rem; color: #1E293B;">
                        <strong>Recommended Execution:</strong> {r.get('suggested_action')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_alert(f"<b>Engine Notice:</b> {rec_data.get('governance_disclaimer')}", alert_type="info")
    else:
        render_alert(f"Failed to retrieve business recommendations: {rec_err}", alert_type="danger")

