"""
Cancellation Segment Risk Analytics View - Empirical Ground Truth Distribution.
"""

from typing import Any, Dict
import pandas as pd
import plotly.express as px
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders empirical cancellation rates and vulnerability profiling across market segments."""
    
    render_section_header(
        "Empirical Segment Cancellation Dynamics",
        "Empirical ground-truth cancellation rates derived from 119,390 genuine hotel reservation records (Antonio et al., 2019)",
    )

    ok_seg, seg_data, seg_err = api_client.get_cancellation_segment_analysis()
    if ok_seg and seg_data:
        segs = seg_data.get("segments", {})

        seg_tab1, seg_tab2, seg_tab3, seg_tab4 = st.tabs([
            "Lead-Time Tiers",
            "Distribution Channels",
            "Customer Classifications",
            "Seasonal Arrival Patterns",
        ])

        with seg_tab1:
            render_section_header("Lead Time Horizon Risk Curve")
            lt_list = segs.get("by_lead_time_tier", [])
            if lt_list:
                lt_df = pd.DataFrame(lt_list)
                fig_lt = px.bar(
                    lt_df,
                    x="lead_time_tier",
                    y="cancellation_rate_pct",
                    text="cancellation_rate_pct",
                    color="cancellation_rate_pct",
                    color_continuous_scale=["#16A34A", "#D97706", "#DC2626"],
                )
                fig_lt.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_lt.update_layout(coloraxis_showscale=False)
                apply_saas_theme(fig_lt, height=340, x_title="Lead Time Horizon", y_title="Cancellation Rate (%)")
                st.plotly_chart(fig_lt, use_container_width=True)
                st.dataframe(lt_df, use_container_width=True)

        with seg_tab2:
            render_section_header("Distribution Channel Risk Comparison")
            ch_list = segs.get("by_distribution_channel", [])
            if ch_list:
                ch_df = pd.DataFrame(ch_list)
                fig_ch = px.bar(
                    ch_df,
                    x="distribution_channel",
                    y="cancellation_rate_pct",
                    text="cancellation_rate_pct",
                    color="cancellation_rate_pct",
                    color_continuous_scale=["#16A34A", "#D97706", "#DC2626"],
                )
                fig_ch.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_ch.update_layout(coloraxis_showscale=False)
                apply_saas_theme(fig_ch, height=340, x_title="Channel", y_title="Cancellation Rate (%)")
                st.plotly_chart(fig_ch, use_container_width=True)
                st.dataframe(ch_df, use_container_width=True)

        with seg_tab3:
            render_section_header("Customer Classification Breakdown")
            cust_list = segs.get("by_customer_type", [])
            if cust_list:
                cust_df = pd.DataFrame(cust_list)
                fig_cust = px.bar(
                    cust_df,
                    x="customer_type",
                    y="cancellation_rate_pct",
                    text="cancellation_rate_pct",
                    color="cancellation_rate_pct",
                    color_continuous_scale=["#16A34A", "#D97706", "#DC2626"],
                )
                fig_cust.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_cust.update_layout(coloraxis_showscale=False)
                apply_saas_theme(fig_cust, height=340, x_title="Customer Type", y_title="Cancellation Rate (%)")
                st.plotly_chart(fig_cust, use_container_width=True)
                st.dataframe(cust_df, use_container_width=True)

        with seg_tab4:
            render_section_header("Seasonal Arrival Month Distribution")
            m_list = segs.get("by_month", [])
            if m_list:
                m_df = pd.DataFrame(m_list)
                fig_m = px.bar(
                    m_df,
                    x="arrival_month",
                    y="cancellation_rate_pct",
                    text="cancellation_rate_pct",
                    color_discrete_sequence=["#2563EB"],
                )
                fig_m.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                apply_saas_theme(fig_m, height=340, x_title="Month", y_title="Cancellation Rate (%)")
                st.plotly_chart(fig_m, use_container_width=True)
                st.dataframe(m_df, use_container_width=True)
    else:
        render_alert(f"Failed to fetch segment analytics: {seg_err}", alert_type="danger")

