import unittest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.cancellation_model import cancellation_model_service


class TestPhase42CancellationAdvanced(unittest.TestCase):
    """
    Unit and integration test suite for Phase 4.2:
    Probability Calibration, Threshold Optimization, SHAP Explainability,
    Counterfactual What-If Simulation, and Segment Risk Analytics.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Ensure model is ready
        if not cancellation_model_service.is_model_ready():
            cancellation_model_service.train_and_benchmark()

    def test_01_model_is_ready_and_calibrators_loaded(self):
        """Verifies that model, preprocessor, and both Sigmoid and Isotonic calibrators are loaded."""
        self.assertTrue(cancellation_model_service.is_model_ready())
        self.assertIsNotNone(cancellation_model_service.calibrator_sigmoid)
        self.assertIsNotNone(cancellation_model_service.calibrator_isotonic)

    def test_02_probability_calibration_metrics(self):
        """Verifies calibration metrics, reliability curves, Brier scores, and ECE values."""
        cal_data = cancellation_model_service.get_calibration_metrics()
        self.assertEqual(cal_data.get("status"), "Success")

        # Test Holdout Split Check
        holdout = cal_data.get("test_split_holdout", {})
        self.assertIn("raw_model", holdout)
        self.assertIn("sigmoid_calibrated", holdout)
        self.assertIn("isotonic_calibrated", holdout)

        raw_ece = holdout["raw_model"]["ece"]
        sig_ece = holdout["sigmoid_calibrated"]["ece"]

        # Platt scaling should strictly reduce ECE compared to raw Random Forest
        self.assertLess(sig_ece, raw_ece)
        self.assertLessEqual(sig_ece, 0.08)  # ECE under 8%

        # Check ROC-AUC preservation under Platt scaling
        raw_auc = holdout["raw_model"]["roc_auc"]
        sig_auc = holdout["sigmoid_calibrated"]["roc_auc"]
        self.assertAlmostEqual(raw_auc, sig_auc, places=3)

        # Check reliability curve data points
        curve = holdout["sigmoid_calibrated"]["curve"]
        self.assertTrue(len(curve["mean_predicted_probabilities"]) > 0)
        self.assertTrue(len(curve["fraction_of_positives"]) > 0)

    def test_03_decision_threshold_optimization_math(self):
        """Verifies cost-optimal threshold calculations and savings logic."""
        res = cancellation_model_service.evaluate_threshold_optimization(
            cost_fp=45.0,
            cost_fn=160.0,
            replacement_rate=0.25,
            split="val",
        )
        self.assertEqual(res.get("status"), "Success")
        opt_th = res.get("optimal_threshold")
        self.assertGreater(opt_th, 0.05)
        self.assertLess(opt_th, 0.95)

        opt_m = res.get("optimal_metrics", {})
        def_m = res.get("default_0_50_metrics", {})

        # Optimal threshold must achieve lower or equal total cost than default 0.50
        self.assertLessEqual(opt_m["total_cost"], def_m["total_cost"])

        impact = res.get("business_impact", {})
        self.assertGreaterEqual(impact["net_cost_savings_dollars"], 0.0)
        self.assertIn("strategic_recommendation", impact)

    def test_04_shap_additive_consistency(self):
        """Validates that Base Value + sum(SHAP) strictly equals predicted probability within 1e-4."""
        test_booking = {
            "lead_time": 60,
            "arrival_date": "2026-08-15",
            "adr": 175.0,
            "booking_channel": "Online TA",
            "customer_type": "Transient",
            "deposit_type": "No Deposit",
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "previous_cancellations": 1,
            "previous_bookings_not_canceled": 0,
            "booking_changes": 0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
            "hotel": "City Hotel",
            "market_segment": "Online TA",
        }
        exp = cancellation_model_service.explain_cancellation(test_booking)
        self.assertEqual(exp.get("status"), "Success")
        self.assertTrue(exp.get("is_additively_consistent"))
        self.assertLess(exp.get("additive_discrepancy"), 1e-4)

        # Attributions check
        self.assertTrue(len(exp.get("top_risk_drivers")) > 0)
        self.assertIn("statistical_disclaimer", exp)

    def test_05_counterfactual_what_if_simulation(self):
        """Verifies multi-lever what-if simulation matrix and probability deltas."""
        test_booking = {
            "lead_time": 45,
            "arrival_date": "2026-09-01",
            "adr": 150.0,
            "booking_channel": "Direct",
            "deposit_type": "No Deposit",
            "total_of_special_requests": 1,
        }
        wi = cancellation_model_service.simulate_what_if(test_booking)
        self.assertEqual(wi.get("status"), "Success")
        base_p = wi.get("baseline_probability")
        self.assertGreaterEqual(base_p, 0.0)
        self.assertLessEqual(base_p, 1.0)

        scenarios = wi.get("scenarios", [])
        self.assertGreaterEqual(len(scenarios), 10)

        for s in scenarios:
            self.assertIn("simulated_probability", s)
            self.assertIn("probability_delta", s)
            self.assertIn("impact_direction", s)
            self.assertIn("operational_takeaway", s)
            self.assertAlmostEqual(s["simulated_probability"] - base_p, s["probability_delta"], places=3)

        self.assertIn("causal_disclaimer", wi)

    def test_06_empirical_segment_risk_analytics(self):
        """Verifies empirical segment statistics from public research dataset."""
        seg = cancellation_model_service.get_segment_risk_analytics()
        self.assertEqual(seg.get("status"), "Success")
        self.assertEqual(seg.get("dataset_total_records"), 87396)

        segs = seg.get("segments", {})
        self.assertIn("by_lead_time_tier", segs)
        self.assertIn("by_distribution_channel", segs)
        self.assertIn("by_customer_type", segs)
        self.assertIn("by_deposit_type", segs)
        self.assertIn("by_arrival_month", segs)
        self.assertIn("by_hotel", segs)

        # Lead time tiers: rates should escalate with longer lead times
        lt_tiers = segs["by_lead_time_tier"]
        tier_dict = {t["lead_time_tier"]: t["cancellation_rate_pct"] for t in lt_tiers}
        self.assertLess(tier_dict["0-7 days"], tier_dict["180+ days"])

    def test_07_api_endpoint_cancellation_calibration(self):
        """Integration test for GET /intelligence/cancellation-calibration."""
        resp = self.client.get("/intelligence/cancellation-calibration")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "Success")
        self.assertIn("test_split_holdout", data)

    def test_08_api_endpoint_cancellation_threshold_analysis(self):
        """Integration test for POST /intelligence/cancellation-threshold-analysis."""
        payload = {"cost_fp": 50.0, "cost_fn": 180.0, "replacement_rate": 0.20, "split": "val"}
        resp = self.client.post("/intelligence/cancellation-threshold-analysis", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "Success")
        self.assertIn("optimal_threshold", data)
        self.assertIn("business_impact", data)

    def test_09_api_endpoint_cancellation_explain(self):
        """Integration test for POST /intelligence/cancellation-explain."""
        payload = {
            "lead_time": 30,
            "arrival_date": "2026-07-20",
            "adr": 140.0,
            "booking_channel": "Direct",
            "customer_type": "Transient",
            "deposit_type": "No Deposit",
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "booking_changes": 0,
            "days_in_waiting_list": 0,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 2,
            "hotel": "Resort Hotel",
            "market_segment": "Direct",
            "reserved_room_type": "A",
            "meal": "BB",
        }
        resp = self.client.post("/intelligence/cancellation-explain", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "Success")
        self.assertTrue(data.get("is_additively_consistent"))

    def test_10_api_endpoint_cancellation_what_if(self):
        """Integration test for POST /intelligence/cancellation-what-if."""
        payload = {
            "lead_time": 30,
            "arrival_date": "2026-07-20",
            "adr": 140.0,
            "booking_channel": "Direct",
            "customer_type": "Transient",
            "deposit_type": "No Deposit",
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "booking_changes": 0,
            "days_in_waiting_list": 0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 1,
            "hotel": "City Hotel",
            "market_segment": "Direct",
            "reserved_room_type": "A",
            "meal": "BB",
        }
        resp = self.client.post("/intelligence/cancellation-what-if", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "Success")
        self.assertIn("scenarios", data)
        self.assertIn("causal_disclaimer", data)

    def test_11_api_endpoint_cancellation_segment_analysis(self):
        """Integration test for GET /intelligence/cancellation-segment-analysis."""
        resp = self.client.get("/intelligence/cancellation-segment-analysis")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "Success")
        self.assertEqual(data.get("dataset_total_records"), 87396)


if __name__ == "__main__":
    unittest.main()

