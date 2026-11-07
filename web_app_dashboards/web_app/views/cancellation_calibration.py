"""
Cancellation Calibration & Threshold Optimization View.
"""

from typing import Any, Dict
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import COLORS, CHART_PALETTE, apply_saas_theme
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders probability reliability assessment and asymmetric cost threshold optimization."""
    
    cal_tab1, cal_tab2 = st.tabs(["Probability Calibration Curves", "Decision Threshold Optimizer"])

    with cal_tab1:
        render_section_header("Reliability Assessment (Calibration Curves)", "Comparison of uncalibrated Random Forest against Platt Sigmoid and Isotonic Regression on untouched holdout test data")
        
        ok_cal, cal_data, cal_err = api_client.get_cancellation_calibration()
        if ok_cal and cal_data:
            test_cal = cal_data.get("test_split_holdout", {})
            raw_m = test_cal.get("raw_model", {})
            sig_m = test_cal.get("sigmoid_calibrated", {})
            iso_m = test_cal.get("isotonic_calibrated", {})

            cal_kpi1, cal_kpi2, cal_kpi3, cal_kpi4 = st.columns(4)
            with cal_kpi1:
                render_kpi_card(
                    "Holdout ECE (Sigmoid)",
                    f"{sig_m.get('ece', 0)*100:.1f}%",
                    delta=f"{(sig_m.get('ece', 0) - raw_m.get('ece', 0))*100:.1f}% vs raw",
                    delta_color="positive",
                    subtext="Expected Calibration Error across 10 bins",
                )
            with cal_kpi2:
                render_kpi_card(
                    "Brier Loss Score",
                    f"{sig_m.get('brier_score', 0):.4f}",
                    delta=f"{sig_m.get('brier_score', 0) - raw_m.get('brier_score', 0):.4f} vs raw",
                    delta_color="positive",
                    subtext="Mean squared probabilistic loss",
                )
            with cal_kpi3:
                render_kpi_card(
                    "Discrimination (ROC-AUC)",
                    f"{sig_m.get('roc_auc', 0):.4f}",
                    subtext="100% ranking preserved under Platt transform",
                )
            with cal_kpi4:
                render_kpi_card(
                    "Fitting Split",
                    "Validation (N=13,109)",
                    subtext="Zero future leakage guarantee",
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Reliability Diagram
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Perfect Calibration (Ideal)",
                line=dict(color="#94A3B8", dash="dash"),
            ))

            raw_curve = raw_m.get("curve", {})
            if raw_curve:
                fig_cal.add_trace(go.Scatter(
                    x=raw_curve.get("mean_predicted_probabilities", []),
                    y=raw_curve.get("fraction_of_positives", []),
                    mode="lines+markers",
                    name=f"Raw Random Forest (Brier: {raw_m.get('brier_score', 0):.4f})",
                    line=dict(color="#DC2626", width=2),
                ))

            sig_curve = sig_m.get("curve", {})
            if sig_curve:
                fig_cal.add_trace(go.Scatter(
                    x=sig_curve.get("mean_predicted_probabilities", []),
                    y=sig_curve.get("fraction_of_positives", []),
                    mode="lines+markers",
                    name=f"Platt Sigmoid (Brier: {sig_m.get('brier_score', 0):.4f}) [ACTIVE]",
                    line=dict(color="#2563EB", width=2.5),
                ))

            iso_curve = iso_m.get("curve", {})
            if iso_curve:
                fig_cal.add_trace(go.Scatter(
                    x=iso_curve.get("mean_predicted_probabilities", []),
                    y=iso_curve.get("fraction_of_positives", []),
                    mode="lines+markers",
                    name=f"Isotonic Regression (Brier: {iso_m.get('brier_score', 0):.4f})",
                    line=dict(color="#0F766E", width=2),
                ))

            apply_saas_theme(fig_cal, height=380, x_title="Mean Predicted Probability", y_title="Fraction of Observed Cancellations")
            st.plotly_chart(fig_cal, use_container_width=True)
        else:
            render_alert(f"Failed to load calibration metrics: {cal_err}", alert_type="danger")

    with cal_tab2:
        render_section_header("Optimal Decision Threshold Tuning", "Tune operational classification threshold under asymmetric error costs")
        
        cost_col1, cost_col2 = st.columns(2)
        with cost_col1:
            cost_fp = st.number_input("Cost of False Positive ($ / incident)", value=150.0, step=10.0,
                                      help="Cost of treating a staying guest as canceled (guest dissatisfaction, unnecessary walk).")
        with cost_col2:
            cost_fn = st.number_input("Cost of False Negative ($ / incident)", value=50.0, step=10.0,
                                      help="Cost of treating a canceled guest as staying (unoccupied room, lost spoilage revenue).")

        th_slider = st.slider("Classification Threshold τ", min_value=0.05, max_value=0.95, value=0.40, step=0.05)

        ok_th, th_data, th_err = api_client.get_cancellation_threshold_analysis(
            threshold=th_slider,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        if ok_th and th_data:
            opt_th = th_data.get("optimal_threshold", 0.40)
            curr_metrics = th_data.get("active_threshold_metrics", {})
            conf_matrix = curr_metrics.get("confusion_matrix", {})

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            tk1, tk2, tk3, tk4 = st.columns(4)
            with tk1:
                render_kpi_card("Optimal Threshold τ*", f"{opt_th:.2f}", subtext="Total cost minimizing cutoff")
            with tk2:
                render_kpi_card("Total Expected Cost", f"${curr_metrics.get('total_cost', 0):,.0f}", subtext=f"At active τ = {th_slider:.2f}")
            with tk3:
                render_kpi_card("Sensitivity (Recall)", f"{curr_metrics.get('recall', 0)*100:.1f}%", subtext="Cancellations caught")
            with tk4:
                render_kpi_card("Specificity", f"{curr_metrics.get('specificity', 0)*100:.1f}%", subtext="Stays correctly protected")

            # Cost vs Threshold Curve
            grid = th_data.get("threshold_cost_curve", [])
            if grid:
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                render_section_header("Expected Operational Cost across Thresholds ($)")
                grid_df = pd.DataFrame(grid)
                fig_cost = go.Figure()
                fig_cost.add_trace(go.Scatter(
                    x=grid_df["threshold"],
                    y=grid_df["total_cost"],
                    mode="lines",
                    name="Total Misclassification Cost ($)",
                    line=dict(color="#2563EB", width=2.5),
                ))
                fig_cost.add_vline(x=opt_th, line_dash="dash", line_color="#16A34A", annotation_text=f"Optimal τ* = {opt_th:.2f}")
                apply_saas_theme(fig_cost, height=320, x_title="Decision Cutoff (τ)", y_title="Expected Cost ($)")
                st.plotly_chart(fig_cost, use_container_width=True)

            # Confusion Matrix Table
            if conf_matrix:
                render_section_header("Resulting Operational Confusion Matrix")
                cm_data = [
                    {"Actual State": "Actual Stay (Negative)", "Predicted Stay": f"{conf_matrix.get('TN', 0):,} (TN)", "Predicted Cancel": f"{conf_matrix.get('FP', 0):,} (FP)"},
                    {"Actual State": "Actual Cancel (Positive)", "Predicted Stay": f"{conf_matrix.get('FN', 0):,} (FN)", "Predicted Cancel": f"{conf_matrix.get('TP', 0):,} (TP)"},
                ]
                st.dataframe(pd.DataFrame(cm_data), use_container_width=True)
        else:
            render_alert(f"Failed to compute threshold analysis: {th_err}", alert_type="danger")

