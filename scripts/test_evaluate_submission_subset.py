import unittest

from scripts.evaluate_submission_subset import summarize_cases


class EvaluateSubmissionSubsetTest(unittest.TestCase):
    def test_summary_uses_subset_denominator(self):
        base = {
            "delivered": True,
            "commonsense_micro_passed": 7,
            "commonsense_micro_total": 8,
            "hard_micro_passed": 2,
            "hard_micro_total": 3,
        }
        cases = [
            {**base, "commonsense_pass": True, "hard_pass": True, "final_pass": True},
            {**base, "commonsense_pass": False, "hard_pass": True, "final_pass": False},
        ]
        result = summarize_cases(cases)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["final_pass_count"], 1)
        self.assertEqual(result["final_pass_rate"], 0.5)
        self.assertEqual(result["hard_macro_rate"], 1.0)
        self.assertEqual(result["commonsense_micro_rate"], 14 / 16)


if __name__ == "__main__":
    unittest.main()

