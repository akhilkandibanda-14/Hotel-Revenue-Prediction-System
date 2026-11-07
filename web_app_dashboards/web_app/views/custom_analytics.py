"""
Custom Analytics Chart Builder View.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders interactive custom chart builder across all dataset variables."""
    
    render_section_header("Interactive Custom Chart Builder", "Build exploratory visualizations across any variables in the loaded dataset")

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "custom_x_axis" not in st.session_state:
        st.session_state.custom_x_axis = all_cols[0] if all_cols else None
    if "custom_y_axis" not in st.session_state:
        st.session_state.custom_y_axis = numeric_cols[0] if numeric_cols else (all_cols[1] if len(all_cols) > 1 else None)
    if "custom_chart_type" not in st.session_state:
        st.session_state.custom_chart_type = "Scatter"

    col_x, col_y, col_type = st.columns(3)
    with col_x:
        x_axis = st.selectbox(
            "X-axis Dimension",
            all_cols,
            index=all_cols.index(st.session_state.custom_x_axis) if st.session_state.custom_x_axis in all_cols else 0,
        )
        st.session_state.custom_x_axis = x_axis
    with col_y:
        y_axis = st.selectbox(
            "Y-axis Metric",
            all_cols,
            index=all_cols.index(st.session_state.custom_y_axis) if st.session_state.custom_y_axis in all_cols else (1 if len(all_cols) > 1 else 0),
        )
        st.session_state.custom_y_axis = y_axis
    with col_type:
        chart_type = st.selectbox(
            "Chart Geometry",
            ["Scatter", "Line", "Bar", "Box", "Histogram", "Area"],
            index=["Scatter", "Line", "Bar", "Box", "Histogram", "Area"].index(st.session_state.custom_chart_type) if st.session_state.custom_chart_type in ["Scatter", "Line", "Bar", "Box", "Histogram", "Area"] else 0,
        )
        st.session_state.custom_chart_type = chart_type

    if x_axis and y_axis:
        try:
            plot_data = df[[x_axis, y_axis]].dropna()
            if plot_data.empty:
                render_alert("No valid non-null rows for selected columns.", alert_type="warning")
            else:
                if chart_type == "Scatter":
                    fig = px.scatter(plot_data, x=x_axis, y=y_axis, color_discrete_sequence=["#2563EB"])
                elif chart_type == "Line":
                    if pd.api.types.is_numeric_dtype(plot_data[x_axis]):
                        plot_data = plot_data.sort_values(x_axis)
                    fig = px.line(plot_data, x=x_axis, y=y_axis, markers=True, color_discrete_sequence=["#2563EB"])
                elif chart_type == "Bar":
                    if plot_data[x_axis].dtype == "object":
                        agg_data = plot_data.groupby(x_axis)[y_axis].mean().reset_index()
                        fig = px.bar(agg_data, x=x_axis, y=y_axis, color_discrete_sequence=["#2563EB"])
                    else:
                        fig = px.bar(plot_data, x=x_axis, y=y_axis, color_discrete_sequence=["#2563EB"])
                elif chart_type == "Box":
                    fig = px.box(plot_data, x=x_axis, y=y_axis, color_discrete_sequence=["#0F766E"])
                elif chart_type == "Histogram":
                    fig = px.histogram(plot_data, x=x_axis, y=y_axis, color_discrete_sequence=["#2563EB"])
                elif chart_type == "Area":
                    fig = px.area(plot_data, x=x_axis, y=y_axis, color_discrete_sequence=["#2563EB"])
                else:
                    fig = px.scatter(plot_data, x=x_axis, y=y_axis, color_discrete_sequence=["#2563EB"])

                apply_saas_theme(fig, title=f"{y_axis} vs {x_axis}", height=420)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            render_alert(f"Failed to generate visualization: {str(e)}", alert_type="danger")

