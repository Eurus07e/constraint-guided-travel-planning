import unittest

from scripts.analyze_verifier_calibration import confusion_counts, confusion_metrics


class VerifierCalibrationTest(unittest.TestCase):
    def test_confusion_counts(self):
        counts = confusion_counts(
            [True, True, False, False],
            [True, False, True, False],
        )
        self.assertEqual(counts, {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual(
            confusion_metrics(counts),
            {
                "precision": 0.5,
                "recall": 0.5,
                "specificity": 0.5,
                "negative_predictive_value": 0.5,
                "accuracy": 0.5,
            },
        )

    def test_rejects_misaligned_vectors(self):
        with self.assertRaises(ValueError):
            confusion_counts([True], [True, False])


if __name__ == "__main__":
    unittest.main()

