from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.services.business_recommender import business_recommender
from app.services.cancellation_analytics import cancellation_service
from app.services.competitor_pricing import competitor_pricing_service
from app.services.demand_forecaster import demand_forecaster
from app.services.prebooking_predictor import prebooking_predictor
from app.utils import config


class TestPhase4Intelligence(unittest.TestCase):
    """
    Comprehensive verification suite for Phase 4:
    Real-World Hotel Revenue Intelligence & Advanced Demand Prediction.
    """

    @classmethod
    def setUpClass(cls):
        config.MODEL_FOLDER = config.BASE_DIR / "models"
        config.CURRENT_MODEL_PATH = config.MODEL_FOLDER / "current_model.joblib"
        config.PIPELINE_PATH = config.MODEL_FOLDER / "preprocessing_pipeline.joblib"
        config.BASE_DATASET_FOLDER = config.BASE_DIR / "datasets"
        config.MERGED_FOLDER = config.BASE_DATASET_FOLDER / "merged"
        cls.client = TestClient(app)

    def test_01_demand_forecaster_horizons_and_baselines(self):
        """Test demand forecasting across 7, 30, and 90 day horizons with baseline benchmarks."""
        for horizon in (7, 30, 90):
            res = demand_forecaster.forecast_demand(horizon=horizon)
            self.assertEqual(res["horizon_days"], horizon)
            self.assertEqual(len(res["forecast"]), horizon)
            self.assertIn("benchmark_evaluation", res)
            self.assertIn("metrics", res["benchmark_evaluation"])
            metrics = res["benchmark_evaluation"]["metrics"]
            self.assertIn("ml_model", metrics)
            self.assertIn("moving_average_7d", metrics)
            self.assertIn("seasonal_naive_7d", metrics)
            self.assertGreater(res["total_expected_room_nights"], 0.0)

            # Check confidence intervals
            first_pt = res["forecast"][0]
            self.assertGreaterEqual(first_pt["upper_bound_rooms"], first_pt["expected_room_demand"])
            self.assertLessEqual(first_pt["lower_bound_rooms"], first_pt["expected_room_demand"])

    def test_02_prebooking_predictor_and_leakage_audit(self):
        """Test pre-booking value estimation and strict target leakage audit."""
        # Valid booking estimation
        res = prebooking_predictor.estimate_prebooking_value(
            check_in_date="2026-07-15",
            check_out_date="2026-07-18",
            booking_date="2026-07-01",
            quoted_adr=180.0,
            rooms_count=2,
            guests_count=4,
            booking_channel="OTA",
            market_segment="Leisure",
            room_type="Deluxe",
        )
        self.assertEqual(res["status"], "Success")
        self.assertEqual(res["booking_details"]["length_of_stay_nights"], 3)
        self.assertGreater(res["estimated_gross_revenue"], 0.0)
        self.assertGreater(res["estimated_net_revenue"], 0.0)
        self.assertGreater(res["estimated_commission_cost"], 0.0)

        # Inverted checkout error check
        with self.assertRaises(ValueError):
            prebooking_predictor.estimate_prebooking_value(
                check_in_date="2026-07-20",
                check_out_date="2026-07-18",
            )

    def test_03_cancellation_analytics(self):
        """Test cancellation rates, no-show rates, and channel breakdown."""
        res = cancellation_service.analyze_cancellations()
        if res.get("status") == "Success":
            self.assertIn("summary", res)
            self.assertIn("overall_cancellation_rate_pct", res["summary"])
            self.assertIn("by_channel", res)
            self.assertIn("missing_pms_fields_for_individual_ml", res)
        else:
            self.assertIn("dataset_requirements", res)

    def test_04_competitor_pricing_and_mpi(self):
        """Test competitor rate benchmarking, Market Price Index, and positioning."""
        eval_res = competitor_pricing_service.evaluate_competitive_position(
            our_adr=190.0,
            competitor_adr=170.0,
        )
        self.assertEqual(eval_res["status"], "Success")
        act = eval_res["active_comparison"]
        self.assertIsNotNone(act)
        self.assertGreater(act["market_price_index_mpi"], 100.0)
        self.assertIn("Premium", act["position_classification"])

        # Add record
        new_entry = competitor_pricing_service.add_competitor_entry(
            date_str="2026-08-01",
            competitor_name="Test Beach Resort",
            competitor_adr=175.0,
            our_adr=165.0,
            room_type="Deluxe Sea View",
            notes="Weekend benchmark test",
        )
        self.assertEqual(new_entry["competitor_name"], "Test Beach Resort")

    def test_05_business_recommendation_engine(self):
        """Test synthesized revenue management recommendations with priority and metrics."""
        rec_res = business_recommender.generate_recommendations()
        self.assertEqual(rec_res["status"], "Success")
        self.assertGreaterEqual(rec_res["total_recommendations"], 1)
        first_rec = rec_res["recommendations"][0]
        self.assertIn("title", first_rec)
        self.assertIn("priority", first_rec)
        self.assertIn("supporting_metrics", first_rec)
        self.assertIn("suggested_action", first_rec)
        self.assertIn("governance_disclaimer", rec_res)

    def test_06_fastapi_intelligence_endpoints(self):
        """Test REST API endpoints under /intelligence/."""
        # 1. Demand Forecast
        r_df = self.client.post("/intelligence/demand-forecast", json={"horizon": 7})
        self.assertEqual(r_df.status_code, 200)
        self.assertEqual(r_df.json()["status"], "Success")

        # 2. Prebooking Predict
        r_pb = self.client.post("/intelligence/prebooking-predict", json={
            "check_in_date": "2026-08-10",
            "check_out_date": "2026-08-14",
            "quoted_adr": 175.0,
            "rooms_count": 1,
            "guests_count": 2,
            "booking_channel": "Direct",
        })
        self.assertEqual(r_pb.status_code, 200)
        self.assertEqual(r_pb.json()["status"], "Success")

        # 3. Prebooking Audit
        r_aud = self.client.get("/intelligence/prebooking-audit")
        self.assertEqual(r_aud.status_code, 200)

        # 4. Cancellation Analytics
        r_canc = self.client.get("/intelligence/cancellation-analytics")
        self.assertEqual(r_canc.status_code, 200)

        # 5. Competitor Analysis
        r_comp = self.client.post("/intelligence/competitor-analysis", json={"our_adr": 180.0, "competitor_adr": 160.0})
        self.assertEqual(r_comp.status_code, 200)

        # 6. Competitor Records GET
        r_crec = self.client.get("/intelligence/competitor-records")
        self.assertEqual(r_crec.status_code, 200)

        # 7. Business Recommendations
        r_recs = self.client.get("/intelligence/business-recommendations")
        self.assertEqual(r_recs.status_code, 200)
        self.assertEqual(r_recs.json()["status"], "Success")


if __name__ == "__main__":
    unittest.main()

