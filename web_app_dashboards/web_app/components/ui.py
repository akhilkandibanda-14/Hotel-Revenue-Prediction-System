"""
Reusable UI Components for Hotel Revenue Management Enterprise Dashboard.
"""

from typing import Optional
import streamlit as st


def render_top_header(
    title: str,
    subtitle: str,
    category_name: str,
    is_connected: bool,
    server_port: str = "8000",
) -> None:
    """Renders the top enterprise header bar with breadcrumb path and live backend status badge."""
    status_pill = (
        f'<span class="status-pill status-pill-success">● Connected (Port {server_port})</span>'
        if is_connected
        else '<span class="status-pill status-pill-danger">○ Disconnected</span>'
    )

    st.markdown(
        f"""
        <div class="top-header-banner">
            <div class="header-title-wrap">
                <div class="breadcrumb-nav">
                    <span>HOTEL INTELLIGENCE</span>
                    <span class="separator">/</span>
                    <span>{category_name.upper()}</span>
                    <span class="separator">/</span>
                    <span class="active-page">{title.upper()}</span>
                </div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div>
                {status_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "positive",  # positive, negative, neutral
    subtext: Optional[str] = None,
    unavailable: bool = False,
) -> None:
    """Renders a clean, enterprise KPI card."""
    display_value = "—" if unavailable else value
    delta_html = ""

    if delta and not unavailable:
        delta_class = f"kpi-delta kpi-delta-{delta_color}"
        arrow = "↑" if delta_color == "positive" else ("↓" if delta_color == "negative" else "•")
        delta_html = f'<div class="{delta_class}">{arrow} {delta}</div>'

    subtext_html = f'<div class="kpi-subtext">{subtext}</div>' if subtext else ""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-header">
                <span class="kpi-title">{title}</span>
                {delta_html}
            </div>
            <div class="kpi-value">{display_value}</div>
            {subtext_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, desc: Optional[str] = None) -> None:
    """Renders a subtle, clean section divider with heading and description."""
    desc_html = f'<p class="section-desc">{desc}</p>' if desc else ""
    st.markdown(
        f"""
        <div class="section-header">
            <h3 class="section-title">{title}</h3>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert(message: str, alert_type: str = "info") -> None:
    """Renders an enterprise-styled alert notice."""
    st.markdown(
        f"""
        <div class="enterprise-alert enterprise-alert-{alert_type}">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, message: str) -> None:
    """Renders a clean empty-state message when data is unavailable."""
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 10px; padding: 2.5rem; text-align: center; margin: 1rem 0;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #111827; margin-bottom: 0.35rem;">{title}</div>
            <div style="font-size: 0.85rem; color: #64748B; max-width: 480px; margin: 0 auto;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

