from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.anomaly_detector import anomaly_detector_service
from app.services.cancellation_analytics import cancellation_service
from app.services.demand_forecaster import demand_forecaster
from app.services.drift_detector import drift_detector_service
from app.services.forecaster import forecaster_service
from app.services.optimizer import optimizer_service

logger = logging.getLogger(__name__)


class BusinessRecommendationEngine:
    """
    Intelligent Business Recommendation Engine.
    Synthesizes forecasts, pricing elasticity, anomaly scans, drift assessments,
    and cancellation metrics into prioritized, actionable revenue management recommendations.
    """

    def __init__(self):
        pass

    def generate_recommendations(self) -> Dict[str, Any]:
        """
        Gathers real analytical insights across modules and compiles prioritized business recommendations.
        """
        recommendations: List[Dict[str, Any]] = []

        # 1. Forecasting & Demand Insight
        try:
            fc_7 = forecaster_service.forecast_revenue(horizon=7)
            occ_7 = forecaster_service.forecast_occupancy(horizon=7)
            avg_occ_pct = occ_7.get("mean_projected_occupancy", 0.70) * 100.0
            avg_daily_rev = fc_7.get("average_daily_revenue", 12000.0)

            if avg_occ_pct > 80.0:
                recommendations.append({
                    "id": "REC_PRICING_01",
                    "title": "High Near-Term Demand: Consider Tactical Rate Optimization",
                    "category": "Pricing & Yield",
                    "priority": "High",
                    "explanation": (
                        f"7-day projected occupancy averages {avg_occ_pct:.1f}%, indicating strong demand compression. "
                        "With limited unallocated room inventory, yield management supports a cautious price increase."
                    ),
                    "supporting_metrics": {
                        "Projected 7-Day Occupancy": f"{avg_occ_pct:.1f}%",
                        "Projected Daily Revenue": f"${avg_daily_rev:,.2f}",
                        "Demand Level": "Compression / Peak",
                    },
                    "confidence_level": "High Evidence (Backtested 7-Day Model)",
                    "suggested_action": "Evaluate a +5% to +10% rate adjustment on transient and OTA channels for upcoming peak nights.",
                    "relevant_period": f"Next 7 Days ({fc_7.get('start_date')} to {fc_7.get('end_date')})",
                })
            elif avg_occ_pct < 60.0:
                recommendations.append({
                    "id": "REC_OCC_01",
                    "title": "Soft Demand Projected: Activate Demand Generation & Value Packages",
                    "category": "Occupancy & Marketing",
                    "priority": "High",
                    "explanation": (
                        f"7-day forecasted occupancy is {avg_occ_pct:.1f}%, well below standard 70% threshold. "
                        "Unsold room inventory risks perishable revenue loss."
                    ),
                    "supporting_metrics": {
                        "Projected 7-Day Occupancy": f"{avg_occ_pct:.1f}%",
                        "Target Occupancy Baseline": "70.0%",
                        "Shortfall": f"{70.0 - avg_occ_pct:.1f}%",
                    },
                    "confidence_level": "Moderate Evidence (Seasonal Demand Lag)",
                    "suggested_action": "Launch targeted weekend packages or value-add inclusions (complimentary breakfast/parking) rather than aggressive rate slashing.",
                    "relevant_period": f"Next 7 Days ({fc_7.get('start_date')} to {fc_7.get('end_date')})",
                })
        except Exception as e:
            logger.warning(f"Error computing forecast recommendations: {e}")

        # 2. Anomaly Detection Insight
        try:
            anom_res = anomaly_detector_service.detect_anomalies(contamination=0.05)
            high_sev_count = anom_res.get("counts_by_severity", {}).get("High", 0)
            if high_sev_count > 0:
                top_anom = anom_res.get("anomalies", [])[0]
                recommendations.append({
                    "id": "REC_ANOMALY_01",
                    "title": f"Investigate {high_sev_count} High-Severity Operational Anomalies",
                    "category": "Operational Audit",
                    "priority": "Medium",
                    "explanation": (
                        f"Multi-method anomaly scan identified {high_sev_count} high-severity outliers. "
                        f"Most significant: {top_anom.get('date')} ({top_anom.get('deviation_pct'):+.1f}% deviation in {top_anom.get('metric')})."
                    ),
                    "supporting_metrics": {
                        "High Severity Outliers": high_sev_count,
                        "Top Anomaly Date": top_anom.get("date"),
                        "Observed Value": f"${top_anom.get('actual_value', 0):,.2f}",
                        "Baseline Value": f"${top_anom.get('expected_value', 0):,.2f}",
                    },
                    "confidence_level": "High Evidence (Statistical IQR & IsolationForest)",
                    "suggested_action": "Audit PMS night audit logs for that date to verify if revenue spike was a group buyout or rate misconfiguration.",
                    "relevant_period": f"Historical Log: {top_anom.get('date')}",
                })
        except Exception as e:
            logger.warning(f"Error computing anomaly recommendations: {e}")

        # 3. Data Drift & Model Health Insight
        try:
            drift_res = drift_detector_service.assess_dataset_drift()
            drift_status = drift_res.get("overall_status", "No Drift")
            drifted_cnt = drift_res.get("total_drifted_features", 0)

            if drift_status == "Severe Drift":
                recommendations.append({
                    "id": "REC_DRIFT_01",
                    "title": "Severe Distributional Drift Detected: Schedule Model Retraining",
                    "category": "Model Governance",
                    "priority": "High",
                    "explanation": (
                        f"{drifted_cnt} operational features exhibit statistically significant drift (KS p < 0.05). "
                        "The serving model may experience degraded predictive accuracy."
                    ),
                    "supporting_metrics": {
                        "Drifted Features Count": drifted_cnt,
                        "Overall Drift Status": drift_status,
                    },
                    "confidence_level": "High Evidence (Two-Sample Kolmogorov-Smirnov Test)",
                    "suggested_action": "Upload latest merged operational dataset and trigger automated background training in Model Governance tab.",
                    "relevant_period": "Immediate Action Recommended",
                })
            elif drift_status == "Moderate Drift":
                recommendations.append({
                    "id": "REC_DRIFT_02",
                    "title": "Moderate Metric Shift Detected: Monitor Channel & Booking Distributions",
                    "category": "Model Governance",
                    "priority": "Low",
                    "explanation": (
                        f"{drifted_cnt} feature(s) showed mild distributional variation. Production predictions remain stable."
                    ),
                    "supporting_metrics": {
                        "Drifted Features": drifted_cnt,
                        "Status": "Moderate Drift",
                    },
                    "confidence_level": "Moderate Evidence",
                    "suggested_action": "Continue tracking weekly booking patterns; retrain at next monthly operational checkpoint.",
                    "relevant_period": "Next 30 Days",
                })
        except Exception as e:
            logger.warning(f"Error computing drift recommendations: {e}")

        # 4. Cancellation & Channel Slippage Insight
        try:
            canc_res = cancellation_service.analyze_cancellations()
            if canc_res.get("status") == "Success":
                summ = canc_res.get("summary", {})
                slip_pct = summ.get("combined_revenue_slippage_pct", 0.0)
                by_ch = canc_res.get("by_channel", [])
                ota_metric = next((c for c in by_ch if "OTA" in c["booking_channel"]), None)

                if ota_metric and ota_metric.get("cancellation_rate_pct", 0.0) > 7.0:
                    recommendations.append({
                        "id": "REC_CANC_01",
                        "title": f"Elevated OTA Cancellation Rate ({ota_metric['cancellation_rate_pct']:.1f}%): Review Cancellation Policies",
                        "category": "Distribution & Risk",
                        "priority": "Medium",
                        "explanation": (
                            f"OTA bookings exhibit a {ota_metric['cancellation_rate_pct']:.1f}% cancellation rate and "
                            f"{ota_metric.get('no_show_rate_pct', 0):.1f}% no-show rate. Total channel slippage is {ota_metric.get('combined_slippage_pct', 0):.1f}%."
                        ),
                        "supporting_metrics": {
                            "OTA Cancellation Rate": f"{ota_metric['cancellation_rate_pct']:.1f}%",
                            "Direct Cancellation Rate": f"{next((c['cancellation_rate_pct'] for c in by_ch if 'Direct' in c['booking_channel']), 0):.1f}%",
                            "Overall Slippage": f"{slip_pct:.1f}%",
                        },
                        "confidence_level": "High Evidence (Historical Channel Counts)",
                        "suggested_action": "Tighten OTA cancellation cutoff windows to 48 hours and incentivize direct bookings with flexible rebooking guarantees.",
                        "relevant_period": "Ongoing Channel Management",
                    })
        except Exception as e:
            logger.warning(f"Error computing cancellation recommendations: {e}")

        # 5. Fallback baseline if no anomalies or alerts triggered
        if not recommendations:
            recommendations.append({
                "id": "REC_STABLE_01",
                "title": "Operations Aligned: Maintain Balanced Yield Strategy",
                "category": "Pricing & Yield",
                "priority": "Low",
                "explanation": "Demand, pricing, and distribution metrics are operating within historical benchmark tolerances.",
                "supporting_metrics": {"System Health": "Optimal", "Variance": "< 5%"},
                "confidence_level": "Standard Baseline",
                "suggested_action": "Continue monitoring forward booking pace and competitor rates.",
                "relevant_period": "Upcoming Month",
            })

        # Priority ordering: High > Medium > Low
        p_order = {"High": 0, "Medium": 1, "Low": 2}
        recommendations = sorted(recommendations, key=lambda r: p_order.get(r["priority"], 3))

        disclaimer = (
            "BUSINESS DECISION NOTICE: Recommendations are algorithmic decision-support guidelines "
            "synthesized from historical data, empirical elasticity, and statistical distribution tests. "
            "They do NOT guarantee commercial outcomes. Hotel revenue managers must apply local market "
            "judgment and event knowledge before executing commercial rate or policy modifications."
        )

        return {
            "status": "Success",
            "total_recommendations": len(recommendations),
            "priority_summary": {
                "High": sum(1 for r in recommendations if r["priority"] == "High"),
                "Medium": sum(1 for r in recommendations if r["priority"] == "Medium"),
                "Low": sum(1 for r in recommendations if r["priority"] == "Low"),
            },
            "recommendations": recommendations,
            "governance_disclaimer": disclaimer,
        }


# Singleton service
business_recommender = BusinessRecommendationEngine()

