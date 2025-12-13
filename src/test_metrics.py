import unittest
import numpy as np

from utils_v2 import compute_metrics   # <-- import your refined function here


class TestDetectionMetrics(unittest.TestCase):

    def setUp(self):
        """
        Create a small synthetic dataset of predictions + GT
        for two classes (car=0, pedestrian=1).
        """

        # ---------- Ground Truth ----------
        self.gt = [
            {
                "boxes": np.array([
                    [0, 0, 0, 2, 4, 1.5, 0],      # GT car
                    [10, 10, 0, 0.8, 0.8, 1.7, 0] # GT pedestrian
                ]),
                "labels": np.array([0, 1])
            }
        ]

        # ---------- Case 1: Perfect Predictions (same positions & labels) ----------
        self.pred_perfect = [
            {
                "boxes": np.array([
                    [0, 0, 0, 2, 4, 1.5, 0],      # correct car
                    [10, 10, 0, 0.8, 0.8, 1.7, 0] # correct pedestrian
                ]),
                "scores": np.array([0.9, 0.85]),
                "labels": np.array([0, 1])
            }
        ]

        # ---------- Case 2: Wrong Predictions (far away → no matches) ----------
        self.pred_wrong = [
            {
                "boxes": np.array([
                    [100, 100, 0, 2, 4, 1.5, 0],  # too far
                    [200, 200, 0, 0.8, 1, 1.7, 0]
                ]),
                "scores": np.array([0.9, 0.8]),
                "labels": np.array([0, 1])
            }
        ]

        # ---------- Case 3: Missing Predictions ----------
        self.pred_none = [
            {
                "boxes": np.zeros((0, 7)),
                "scores": np.zeros((0,)),
                "labels": np.zeros((0,))
            }
        ]

    # ------------------------------------------------------
    # TEST 1 — Perfect prediction → AP = 1 for both classes
    # ------------------------------------------------------
    def test_perfect_predictions(self):
        metrics = compute_metrics(self.pred_perfect, self.gt)

        self.assertAlmostEqual(metrics["mAP"], 1.0, places=5)
        self.assertAlmostEqual(metrics["AP_per_class"]["car"], 1.0)
        self.assertAlmostEqual(metrics["AP_per_class"]["pedestrian"], 1.0)

        # NDS should be high because ATE=0, ASE=0, AOE=0
        self.assertTrue(metrics["NDS"] > 0.9)

    # ------------------------------------------------------
    # TEST 2 — Completely wrong predictions → AP = 0
    # ------------------------------------------------------
    def test_wrong_predictions(self):
        metrics = compute_metrics(self.pred_wrong, self.gt)

        self.assertAlmostEqual(metrics["mAP"], 0.0, places=5)
        self.assertAlmostEqual(metrics["AP_per_class"]["car"], 0.0)
        self.assertAlmostEqual(metrics["AP_per_class"]["pedestrian"], 0.0)

        # Since all errors are huge, NDS should be low
        self.assertTrue(metrics["NDS"] < 0.5)

    # ------------------------------------------------------
    # TEST 3 — No predictions → mAP = 0
    # ------------------------------------------------------
    def test_no_predictions(self):
        metrics = compute_metrics(self.pred_none, self.gt)

        self.assertAlmostEqual(metrics["mAP"], 0.0)
        self.assertAlmostEqual(metrics["AP_per_class"]["car"], 0.0)
        self.assertAlmostEqual(metrics["AP_per_class"]["pedestrian"], 0.0)

        # ATE, ASE, AOE are undefined but default high, so NDS low
        self.assertTrue(metrics["NDS"] < 0.5)


if __name__ == "__main__":
    unittest.main()
