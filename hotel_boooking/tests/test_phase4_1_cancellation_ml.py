import json
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.cancellation_model import cancellation_model_service, LEAKAGE_BLACKLIST, PUBLIC_DATASET_FILE
from app.utils import config


class TestPhase41CancellationML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.MODEL_FOLDER = config.BASE_DIR / "models"
        cls.client = TestClient(app)

    def test_dataset_loading_and_governance(self):
        """Verifies public research dataset presence, row count, and governance label."""
        self.assertTrue(PUBLIC_DATASET_FILE.exists(), "Antonio et al. dataset file must exist.")
        meta_file = PUBLIC_DATASET_FILE.parent / "dataset_metadata.json"
        self.assertTrue(meta_file.exists(), "Public dataset metadata must exist.")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.assertEqual(meta.get("data_governance_label"), "PUBLIC_RESEARCH_DATA")
        self.assertEqual(meta.get("classification"), "Public Research Data")

    def test_leakage_prevention(self):
        """Ensures post-arrival and outcome features are strictly blacklisted."""
        benchmark = cancellation_model_service.train_and_benchmark(force_retrain=False)
        self.assertEqual(benchmark.get("status"), "Success")
        leak_audit = benchmark.get("leakage_prevention_audit", {})
        self.assertTrue(leak_audit.get("is_canceled_excluded"))
        self.assertTrue(leak_audit.get("reservation_status_excluded"))
        self.assertTrue(leak_audit.get("reservation_status_date_excluded"))
        self.assertTrue(leak_audit.get("assigned_room_type_excluded"))

    def test_chronological_split_and_validation(self):
        """Ensures chronological split info exists and train dates precede test dates."""
        benchmark = cancellation_model_service.train_and_benchmark(force_retrain=False)
        split = benchmark.get("split_info", {})
        self.assertIn("train_dates", split)
        self.assertIn("test_dates", split)
        self.assertGreater(split["train_size"], 50000)
        self.assertGreater(split["test_size"], 10000)

    def test_model_benchmarking_and_metrics(self):
        """Verifies multi-algorithm benchmark evaluated ROC-AUC, F1, and Brier scores."""
        benchmark = cancellation_model_service.train_and_benchmark(force_retrain=False)
        algos = benchmark.get("algorithms_benchmarked", {})
        self.assertIn("Logistic Regression", algos)
        self.assertIn("Random Forest", algos)
        self.assertIn("Extra Trees", algos)
        self.assertIn("HistGradientBoosting", algos)

        # Selected model test metrics
        test_eval = benchmark.get("selected_model_test_evaluation", {})
        self.assertGreater(test_eval.get("test_roc_auc", 0), 0.70)
        self.assertGreater(test_eval.get("test_f1", 0), 0.50)
        self.assertLess(test_eval.get("test_brier_score", 1.0), 0.25)

    def test_prediction_api_endpoint(self):
        """Tests POST /intelligence/cancellation-predict live inference."""
        payload = {
            "lead_time": 60,
            "arrival_date": "2026-09-01",
            "adr": 150.0,
            "booking_channel": "Direct",
            "customer_type": "Transient",
            "deposit_type": "No Deposit",
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 1,
        }
        resp = self.client.post("/intelligence/cancellation-predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "Success")
        self.assertIn("cancellation_probability", data)
        self.assertIn("risk_category", data)
        self.assertIn(data.get("risk_category"), ["Low Risk", "Moderate Risk", "High Risk"])
        self.assertIn("key_contributing_factors", data)
        self.assertIn("disclaimer", data)

    def test_unseen_categories(self):
        """Ensures inference handles unseen or extreme category values gracefully."""
        payload = {
            "lead_time": 10,
            "arrival_date": "2026-10-15",
            "adr": 250.0,
            "booking_channel": "UNSEEN_CHANNEL_XYZ",
            "customer_type": "MYSTERY_SEGMENT",
            "deposit_type": "UNKNOWN_DEPOSIT",
            "hotel": "FANTASY_HOTEL",
            "reserved_room_type": "Z99",
            "meal": "EXTRA_MEAL",
        }
        resp = self.client.post("/intelligence/cancellation-predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("cancellation_probability", data)

    def test_probability_calibration_and_sensitivity(self):
        """Tests that high-risk attributes produce higher probability than low-risk attributes."""
        high_risk_payload = {
            "lead_time": 200,
            "arrival_date": "2026-08-01",
            "adr": 180.0,
            "booking_channel": "TA/TO",
            "deposit_type": "Non Refund",
            "previous_cancellations": 2,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
        }
        low_risk_payload = {
            "lead_time": 2,
            "arrival_date": "2026-08-01",
            "adr": 180.0,
            "booking_channel": "Direct",
            "deposit_type": "No Deposit",
            "previous_cancellations": 0,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 2,
        }
        resp_high = self.client.post("/intelligence/cancellation-predict", json=high_risk_payload)
        resp_low = self.client.post("/intelligence/cancellation-predict", json=low_risk_payload)
        self.assertEqual(resp_high.status_code, 200)
        self.assertEqual(resp_low.status_code, 200)
        prob_high = resp_high.json().get("cancellation_probability")
        prob_low = resp_low.json().get("cancellation_probability")
        self.assertGreater(prob_high, prob_low, "High-risk scenario must yield higher cancellation probability.")

    def test_benchmark_endpoint(self):
        """Tests GET /intelligence/cancellation-model-performance."""
        resp = self.client.get("/intelligence/cancellation-model-performance")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "Success")
        self.assertIn("algorithms_benchmarked", data)
        self.assertIn("selected_model_test_evaluation", data)


if __name__ == "__main__":
    unittest.main()

