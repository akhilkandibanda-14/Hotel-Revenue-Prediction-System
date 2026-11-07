"""
Enterprise SaaS Theme Tokens and Plotly Styler for Hotel Revenue Intelligence Platform.
"""

from typing import Optional
import plotly.graph_objects as go

# Design System Color Tokens
COLORS = {
    "bg_primary": "#F7F8FA",
    "surface": "#FFFFFF",
    "sidebar": "#111827",
    "sidebar_hover": "#1F2937",
    "sidebar_active": "#1E293B",
    "primary_accent": "#2563EB",
    "secondary_accent": "#0F766E",
    "text_primary": "#111827",
    "text_secondary": "#64748B",
    "text_muted": "#94A3B8",
    "border": "#E5E7EB",
    "border_dark": "#374151",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
    "muted_bg": "#F1F5F9",
}

# Standardized Enterprise Chart Palette
CHART_PALETTE = [
    "#2563EB",  # Blue (Primary)
    "#0F766E",  # Teal (Secondary)
    "#D97706",  # Amber
    "#64748B",  # Slate
    "#16A34A",  # Emerald Green
    "#8B5CF6",  # Purple
    "#0284C7",  # Sky Blue
    "#DC2626",  # Red / Danger
]


def apply_saas_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 400,
    show_legend: bool = True,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
) -> go.Figure:
    """Applies clean enterprise SaaS styling (Stripe/Linear aesthetic) to any Plotly figure."""
    layout_update = dict(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            size=12,
            color="#64748B",
        ),
        margin=dict(l=40, r=30, t=50 if title else 25, b=45),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
            bordercolor="#374151",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="#E5E7EB",
            tickfont=dict(color="#64748B", size=11),
            title=dict(text=x_title, font=dict(color="#475569", size=12)) if x_title else None,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="#E5E7EB",
            tickfont=dict(color="#64748B", size=11),
            title=dict(text=y_title, font=dict(color="#475569", size=12)) if y_title else None,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#475569", size=11),
            bgcolor="rgba(255,255,255,0.7)",
        ) if show_legend else dict(visible=False),
    )

    if title:
        layout_update["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(color="#111827", size=14),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        )

    fig.update_layout(**layout_update)
    return fig

