from __future__ import annotations

import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services.anomaly_detector import anomaly_detector_service
from app.services.drift_detector import drift_detector_service
from app.services.explainer import explainer_service
from app.services.forecaster import forecaster_service
from app.services.optimizer import optimizer_service
from app.utils import config


class TestPhase3Analytics(unittest.TestCase):
    """
    Comprehensive verification suite for Phase 3:
    Advanced Hotel Revenue Analytics & Intelligent Forecasting.
    """

    @classmethod
    def setUpClass(cls):
        # Ensure paths point to production assets
        config.MODEL_FOLDER = config.BASE_DIR / "models"
        config.CURRENT_MODEL_PATH = config.MODEL_FOLDER / "current_model.joblib"
        config.PIPELINE_PATH = config.MODEL_FOLDER / "preprocessing_pipeline.joblib"
        config.BASE_DATASET_FOLDER = config.BASE_DIR / "datasets"
        config.MERGED_FOLDER = config.BASE_DATASET_FOLDER / "merged"
        config.METADATA_PATH = config.MODEL_FOLDER / "model_metadata.json"
        config.COMPARISON_PATH = config.MODEL_FOLDER / "model_comparison.json"

        # Force predictor reload
        from app.services.predictor import revenue_predictor
        revenue_predictor.model = None
        revenue_predictor.model_mtime = 0.0
        revenue_predictor.load_assets()

        cls.client = TestClient(app)

    def test_01_forecaster_revenue_horizons(self):
        """Test revenue forecasting across 7, 30, and 90 day horizons with backtest metrics."""
        for horizon in (7, 30, 90):
            res = forecaster_service.forecast_revenue(horizon=horizon)
            self.assertEqual(res["horizon_days"], horizon)
            self.assertEqual(len(res["forecast"]), horizon)
            self.assertIn("backtest_metrics", res)
            self.assertIn("rmse", res["backtest_metrics"])
            self.assertIn("mae", res["backtest_metrics"])
            self.assertIn("mape", res["backtest_metrics"])
            self.assertGreater(res["total_projected_revenue"], 0.0)

            # Verify confidence intervals
            first_day = res["forecast"][0]
            self.assertIn("predicted_revenue", first_day)
            self.assertIn("lower_bound", first_day)
            self.assertIn("upper_bound", first_day)
            self.assertGreaterEqual(first_day["upper_bound"], first_day["predicted_revenue"])
            self.assertLessEqual(first_day["lower_bound"], first_day["predicted_revenue"])
            self.assertGreaterEqual(first_day["lower_bound"], 0.0)

    def test_02_forecaster_occupancy(self):
        """Test occupancy rate forecasting bounded in [0, 1] with pattern analysis."""
        res = forecaster_service.forecast_occupancy(horizon=14)
        self.assertEqual(len(res["forecast"]), 14)
        self.assertIn("historical_patterns", res)
        self.assertIn("weekend_average_occupancy", res["historical_patterns"])
        self.assertIn("weekday_average_occupancy", res["historical_patterns"])

        for pt in res["forecast"]:
            occ = pt["predicted_occupancy"]
            self.assertGreaterEqual(occ, 0.0)
            self.assertLessEqual(occ, 1.0)
            self.assertGreaterEqual(pt["upper_bound"], occ)
            self.assertLessEqual(pt["lower_bound"], occ)

    def test_03_pricing_optimizer_scenarios(self):
        """Test what-if pricing matrix across +/- 30% ADR change with elasticity."""
        res = optimizer_service.simulate_pricing_matrix(
            base_adr=160.0,
            base_occupancy=0.75,
            total_rooms=100,
            fixed_costs=2000.0,
            var_cost_per_room=20.0,
        )
        self.assertIn("optimal_revenue_scenario", res)
        self.assertIn("optimal_profit_scenario", res)
        self.assertEqual(len(res["scenarios"]), 13)

        baseline_scen = next(s for s in res["scenarios"] if s["is_baseline"])
        self.assertEqual(baseline_scen["adr_change_pct"], 0)
        self.assertEqual(baseline_scen["projected_adr"], 160.0)
        self.assertEqual(baseline_scen["revenue_lift"], 0.0)

        for s in res["scenarios"]:
            self.assertGreaterEqual(s["projected_occupancy"], 0.05)
            self.assertLessEqual(s["projected_occupancy"], 1.0)
            self.assertGreater(s["projected_revenue"], 0.0)

    def test_04_shap_explainer(self):
        """Test SHAP global feature importances and local single prediction decomposition."""
        glob = explainer_service.get_global_importance(max_background_samples=30)
        self.assertIn("base_value", glob)
        self.assertIn("feature_importances", glob)
        self.assertGreater(len(glob["feature_importances"]), 5)
        self.assertIn("causal_disclaimer", glob)

        # Test local instance
        sample_input = {
            "Date": "2026-06-20",
            "Occupancy_Rate": 0.80,
            "ADR": 175.0,
            "Available_Rooms": 110,
            "Reserved_Rooms": 88,
            "Total_Costs": 2600.0,
        }
        loc = explainer_service.explain_instance(sample_input)
        self.assertIn("predicted_revenue", loc)
        self.assertIn("base_value", loc)
        self.assertIn("top_positive_features", loc)
        self.assertIn("top_negative_features", loc)
        self.assertTrue(loc["sum_check_matches"])

    def test_05_anomaly_detection(self):
        """Test multi-method anomaly detection (IQR/Z-Score + IsolationForest)."""
        res = anomaly_detector_service.detect_anomalies(contamination=0.05)
        self.assertGreater(res["total_records_analyzed"], 0)
        self.assertGreater(res["anomalies_detected"], 0)
        self.assertIn("counts_by_severity", res)

        top_anom = res["anomalies"][0]
        self.assertIn("severity", top_anom)
        self.assertIn("metric", top_anom)
        self.assertIn("diagnostic_note", top_anom)

    def test_06_drift_detector(self):
        """Test Kolmogorov-Smirnov and categorical PSI data drift assessment."""
        res = drift_detector_service.assess_dataset_drift()
        self.assertIn("overall_status", res)
        self.assertIn(res["overall_status"], ["No Drift", "Moderate Drift", "Severe Drift"])
        self.assertIn("numerical_drift", res)
        self.assertIn("categorical_drift", res)
        self.assertGreater(len(res["numerical_drift"]), 0)
        self.assertGreater(len(res["categorical_drift"]), 0)

    def test_07_api_analytics_endpoints(self):
        """Test full integration with FastAPI /analytics/ endpoints."""
        # 1. Forecast endpoint
        fc_resp = self.client.post("/analytics/forecast", json={"horizon": 7})
        self.assertEqual(fc_resp.status_code, 200)
        self.assertEqual(fc_resp.json()["status"], "Success")
        self.assertEqual(len(fc_resp.json()["forecast"]), 7)

        # 2. Occupancy forecast endpoint
        occ_resp = self.client.post("/analytics/occupancy-forecast", json={"horizon": 7})
        self.assertEqual(occ_resp.status_code, 200)
        self.assertEqual(occ_resp.json()["status"], "Success")

        # 3. Pricing optimizer endpoint
        opt_resp = self.client.post("/analytics/pricing-optimizer", json={"base_adr": 150.0, "base_occupancy": 0.70})
        self.assertEqual(opt_resp.status_code, 200)
        self.assertEqual(opt_resp.json()["status"], "Success")

        # 4. Global SHAP endpoint
        shap_g_resp = self.client.get("/analytics/explain/global?max_samples=25")
        self.assertEqual(shap_g_resp.status_code, 200)
        self.assertEqual(shap_g_resp.json()["status"], "Success")

        # 5. Local SHAP endpoint
        shap_l_resp = self.client.post("/analytics/explain", json={"Date": "2026-07-01", "Occupancy_Rate": 0.75, "ADR": 160.0})
        self.assertEqual(shap_l_resp.status_code, 200)
        self.assertEqual(shap_l_resp.json()["status"], "Success")

        # 6. Anomalies endpoint
        anom_resp = self.client.get("/analytics/anomalies")
        self.assertEqual(anom_resp.status_code, 200)
        self.assertEqual(anom_resp.json()["status"], "Success")

        # 7. Drift endpoint
        drift_resp = self.client.post("/analytics/drift", json={})
        self.assertEqual(drift_resp.status_code, 200)
        self.assertEqual(drift_resp.json()["status"], "Success")


if __name__ == "__main__":
    unittest.main()
