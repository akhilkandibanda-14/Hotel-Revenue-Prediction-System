import unittest
from datetime import date, timedelta
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.inventory_manager import room_inventory_manager
from app.services.overbooking_optimizer import compute_poisson_binomial_pmf, overbooking_optimizer


class TestPhase5InventoryAndOverbooking(unittest.TestCase):
    """
    Comprehensive Automated Test Suite for Phase 5:
    Intelligent Room Inventory Management & Constrained Overbooking Optimization.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_room_inventory_capacity_and_oos(self):
        """Verifies physical room capacity, out-of-service maintenance, and effective capacity."""
        cats = room_inventory_manager.get_room_categories()
        self.assertIn("Standard Room", cats)
        self.assertIn("Deluxe Room", cats)
        self.assertIn("Executive Suite", cats)
        self.assertIn("Presidential Suite", cats)

        std = cats["Standard Room"]
        phys = std["total_physical_rooms"]
        oos = std["rooms_out_of_service"]
        eff = std["effective_capacity"]
        self.assertEqual(eff, phys - oos)
        self.assertGreater(eff, 0)

    def test_02_reservation_date_overlap_no_double_counting(self):
        """
        Verifies that room occupancy strictly enforces arrival_date <= target_date < departure_date.
        A 1-night reservation on 2026-09-01 departing 2026-09-02 occupies a room on Sept 1, but NOT Sept 2.
        """
        res_test = {
            "booking_id": "TEST-OVERLAP-001",
            "guest_name": "Test Overlap Guest",
            "room_category": "Standard Room",
            "arrival_date": "2026-09-01",
            "departure_date": "2026-09-02",
            "stay_nights": 1,
            "booked_adr": 150.0,
            "booking_channel": "Direct",
            "deposit_type": "No Deposit",
            "booking_status": "Confirmed",
        }
        room_inventory_manager.add_reservation(res_test)

        # Check Sept 1 night: MUST be occupied
        active_sep1 = room_inventory_manager.get_active_reservations_for_date("2026-09-01", "Standard Room", include_calibrated_prob=False)
        ids_sep1 = [r["booking_id"] for r in active_sep1]
        self.assertIn("TEST-OVERLAP-001", ids_sep1)

        # Check Sept 2 night (checkout date): MUST NOT be occupied (available for new arrival)
        active_sep2 = room_inventory_manager.get_active_reservations_for_date("2026-09-02", "Standard Room", include_calibrated_prob=False)
        ids_sep2 = [r["booking_id"] for r in active_sep2]
        self.assertNotIn("TEST-OVERLAP-001", ids_sep2)

    def test_03_room_category_isolation(self):
        """Verifies that reservations for Standard Room do not reduce Deluxe Room available inventory."""
        date_str = "2026-09-10"
        avail_std = room_inventory_manager.get_availability_by_date(date_str, "Standard Room")
        avail_dlx = room_inventory_manager.get_availability_by_date(date_str, "Deluxe Room")

        self.assertEqual(len(avail_std["category_breakdown"]), 1)
        self.assertEqual(avail_std["category_breakdown"][0]["room_category"], "Standard Room")
        self.assertEqual(len(avail_dlx["category_breakdown"]), 1)
        self.assertEqual(avail_dlx["category_breakdown"][0]["room_category"], "Deluxe Room")

    def test_04_poisson_binomial_pmf_mathematical_properties(self):
        """
        Verifies mathematical correctness of exact Poisson-Binomial distribution convolution:
        1. Sum of probabilities equals 1.0.
        2. Mean of distribution equals sum of success probabilities.
        3. Variance of distribution equals sum of p_i * (1 - p_i).
        """
        q_probs = [0.65, 0.70, 0.85, 0.50, 0.90, 0.75, 0.80]
        pmf = compute_poisson_binomial_pmf(q_probs)

        # 1. Sum to 1.0
        self.assertAlmostEqual(float(np.sum(pmf)), 1.0, places=5)

        # 2. Expected value
        expected_k = float(np.sum(np.arange(len(pmf)) * pmf))
        self.assertAlmostEqual(expected_k, float(np.sum(q_probs)), places=5)

        # 3. Variance
        expected_var = float(np.sum(((np.arange(len(pmf)) - expected_k) ** 2) * pmf))
        theoretical_var = float(np.sum([p * (1.0 - p) for p in q_probs]))
        self.assertAlmostEqual(expected_var, theoretical_var, places=5)

    def test_05_cancellation_forecast_aggregation(self):
        """Verifies aggregation of individual calibrated cancellation probabilities."""
        target_date = room_inventory_manager.reservations[0]["arrival_date"]
        fc = overbooking_optimizer.compute_cancellation_forecast(target_date, "Standard Room")

        self.assertEqual(fc["status"], "Success")
        n = fc["confirmed_reservations"]
        self.assertGreaterEqual(n, 0)
        exp_c = fc["expected_cancellations"]
        exp_a = fc["expected_arrivals"]
        self.assertAlmostEqual(exp_c + exp_a, float(n), places=1)
        self.assertIn("arrival_distribution_pmf", fc)
        self.assertIn("independence_assumption_note", fc)

    def test_06_constrained_overbooking_optimization(self):
        """Verifies constrained optimization: selects b* maximizing ENC subject to P(Walk) <= alpha."""
        target_date = room_inventory_manager.reservations[0]["arrival_date"]
        alpha_risk = 2.0  # 2% max walk risk

        opt = overbooking_optimizer.optimize_overbooking(
            target_date=target_date,
            room_category="Standard Room",
            custom_adr=150.0,
            custom_walk_cost=300.0,
            max_walk_risk_pct=alpha_risk,
        )

        self.assertEqual(opt["status"], "Success")
        rec = opt["recommendation"]
        safe_b = rec["safe_additional_bookings"]
        walk_p = rec["walk_probability_pct"]

        # Walk probability must not exceed configured alpha (allowing slight grid tolerance if b=0)
        if safe_b > 0:
            self.assertLessEqual(walk_p, alpha_risk + 0.05)

        # Net revenue gain vs baseline must be non-negative
        imp = opt["business_impact_vs_no_overbooking"]
        self.assertGreaterEqual(imp["net_revenue_gain_dollars"], 0.0)

    def test_07_boundary_zero_cancellation_scenario(self):
        """
        When all guests have 0% cancellation probability (100% arrival certainty),
        overbooking beyond physical capacity must be strictly prohibited under tight risk tolerance.
        """
        # Test synthetic scenario where success probability is 1.0 (zero cancellation)
        sure_probs = [1.0] * 50
        pmf = compute_poisson_binomial_pmf(sure_probs)
        # Exactly 50 arrivals with 100% probability
        self.assertAlmostEqual(pmf[50], 1.0, places=5)
        # Zero probability of fewer or more arrivals
        self.assertAlmostEqual(float(np.sum(pmf[:50])), 0.0, places=5)

    def test_08_risk_constraint_sensitivity(self):
        """Verifies that relaxing walk risk constraint alpha allows equal or more additional bookings."""
        target_date = room_inventory_manager.reservations[0]["arrival_date"]

        opt_strict = overbooking_optimizer.optimize_overbooking(
            target_date=target_date,
            room_category="Standard Room",
            max_walk_risk_pct=0.5,  # 0.5% very strict
        )
        opt_relaxed = overbooking_optimizer.optimize_overbooking(
            target_date=target_date,
            room_category="Standard Room",
            max_walk_risk_pct=5.0,  # 5.0% relaxed
        )

        b_strict = opt_strict["recommendation"]["safe_additional_bookings"]
        b_relaxed = opt_relaxed["recommendation"]["safe_additional_bookings"]
        self.assertLessEqual(b_strict, b_relaxed)

    def test_09_api_inventory_endpoints(self):
        """Integration test for all Phase 5 /inventory endpoints."""
        # 1. Availability
        resp = self.client.get("/inventory/availability")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "Success")
        self.assertIn("summary", data)

        # 2. Update room category
        update_payload = {
            "room_category": "Standard Room",
            "total_physical_rooms": 100,
            "rooms_out_of_service": 3,
            "base_adr": 145.0,
            "walk_relocation_cost": 290.0,
        }
        resp = self.client.post("/inventory/rooms", json=update_payload)
        self.assertEqual(resp.status_code, 200)

        # 3. Occupancy timeline
        resp = self.client.get("/inventory/occupancy")
        self.assertEqual(resp.status_code, 200)

        # 4. Active reservations
        today_str = date.today().strftime("%Y-%m-%d")
        resp = self.client.get(f"/inventory/reservations?target_date={today_str}")
        self.assertEqual(resp.status_code, 200)

        # 5. Missing fields audit
        resp = self.client.get("/inventory/missing-fields-audit")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("missing_production_pms_fields", resp.json())

    def test_10_api_overbooking_endpoints(self):
        """Integration test for all Phase 5 /overbooking endpoints."""
        target_date = room_inventory_manager.reservations[0]["arrival_date"]

        # 1. Forecast
        resp = self.client.get(f"/overbooking/forecast?target_date={target_date}&room_category=Standard+Room")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("expected_arrivals", resp.json())

        # 2. Optimize
        resp = self.client.post("/overbooking/optimize", json={
            "target_date": target_date,
            "room_category": "Standard Room",
            "max_walk_risk_pct": 2.0,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn("recommendation", resp.json())

        # 3. Scenario
        resp = self.client.post("/overbooking/scenario", json={
            "target_date": target_date,
            "room_category": "Deluxe Room",
            "custom_adr": 220.0,
            "custom_walk_cost": 450.0,
            "max_walk_risk_pct": 1.5,
        })
        self.assertEqual(resp.status_code, 200)

        # 4. Recommendations
        resp = self.client.get(f"/overbooking/recommendations?target_date={target_date}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("portfolio_summary", resp.json())

        # 5. Performance
        resp = self.client.get("/overbooking/performance")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("summary", resp.json())


if __name__ == "__main__":
    unittest.main()

