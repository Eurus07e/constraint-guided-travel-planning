import unittest
from scripts.verify_frozen import verify

class FrozenEvidenceTests(unittest.TestCase):
    def test_published_predictions_and_outcomes_are_consistent(self):
        report = verify()
        self.assertEqual(report['instances'], 180)
        self.assertEqual(report['methods'], 13)
        self.assertEqual(report['final_pass_counts']['seeded_repair_probe'], 88)
