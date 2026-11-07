"""
Dataset Ingestion & Continuous ML Pipeline View.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from components.theme import COLORS, CHART_PALETTE
from components.ui import render_kpi_card, render_section_header, render_alert
from api_client import HotelRevenueAPIClient


def render(api_client: HotelRevenueAPIClient, df: pd.DataFrame) -> None:
    """Renders server-side dataset upload, deduplication ingestion, and continuous ML training status."""
    
    render_section_header(
        "Dataset Ingestion & Background Training Monitor",
        "Upload new historical reservation CSV datasets to trigger continuous dataset merging and multi-model retraining",
    )

    up_col1, up_col2 = st.columns([1, 1])
    with up_col1:
        render_section_header("Upload Dataset to FastAPI Service")
        new_training_file = st.file_uploader(
            "Upload Training Dataset (CSV)",
            type=["csv"],
            key="api_training_dataset_uploader",
        )
        if new_training_file is not None:
            if st.button("Upload to Background ML Pipeline", use_container_width=True):
                with st.spinner("Uploading dataset to FastAPI..."):
                    file_bytes = new_training_file.getvalue()
                    ok_up, up_res, err_up = api_client.upload_dataset(file_bytes, new_training_file.name)
                if ok_up:
                    render_alert(f"Dataset accepted! Filename: {up_res.get('filename')}. Background scheduler will merge and retrain automatically.", alert_type="success")
                else:
                    render_alert(f"Upload failed: {err_up}", alert_type="danger")

    with up_col2:
        render_section_header("Continuous Training Scheduler Status")
        ok_st, status_data, err_st = api_client.get_dataset_status()
        if ok_st and status_data:
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                render_kpi_card("Pending Ingestion", f"{status_data.get('pending_count', 0)} files", subtext="Queue waiting")
            with s_col2:
                render_kpi_card("Processed Ingestions", f"{status_data.get('processed_count', 0)} files", delta_color="positive", subtext="Successfully merged")
            with s_col3:
                fail_c = status_data.get('failed_count', 0)
                render_kpi_card("Failed Ingestions", f"{fail_c} files", delta_color="negative" if fail_c > 0 else "positive", subtext="Validation rejections")

            latest = status_data.get("latest_training")
            if latest:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                render_section_header("Last Processed Training Event")
                st.write(f"• **Dataset Filename:** `{latest.get('filename')}`")
                st.write(f"• **Processed Timestamp:** `{latest.get('processed_at')}`")
                res_obj = latest.get("result", {})
                if res_obj:
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        render_kpi_card("Tournament Winner", str(res_obj.get("Selected Model", res_obj.get("status", "N/A"))), subtext="Deployed to production")
                    with r2:
                        render_kpi_card("Validation RMSE", f"${res_obj.get('RMSE', 0):,.2f}", subtext="Holdout error")
                    with r3:
                        render_kpi_card("Validation R²", f"{res_obj.get('R2', 0)}", subtext="Variance explained")
        else:
            render_alert("Training scheduler status not currently available.", alert_type="info")

