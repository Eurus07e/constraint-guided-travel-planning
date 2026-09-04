import unittest
from scripts.analyze_evidence import paired_stats

class EvidenceTests(unittest.TestCase):
    def test_no_changes_have_zero_interval_and_unit_p(self):
        report=paired_stats([True,False],[True,False],bootstrap_samples=100)
        self.assertEqual(report['gains'],0);self.assertEqual(report['losses'],0)
        self.assertEqual(report['paired_bootstrap_95_ci_percentage_points'],[0,0])
        self.assertEqual(report['mcnemar_exact_two_sided_p'],1)

    def test_paired_counts_and_exact_significance(self):
        report=paired_stats([False]*6,[True]*6,bootstrap_samples=100)
        self.assertEqual(report['gains'],6);self.assertEqual(report['delta_percentage_points'],100)
        self.assertEqual(report['mcnemar_exact_two_sided_p'],0.03125)
        self.assertEqual(report['paired_bootstrap_95_ci_percentage_points'],[100,100])
        with self.assertRaises(ValueError):paired_stats([True],[])
