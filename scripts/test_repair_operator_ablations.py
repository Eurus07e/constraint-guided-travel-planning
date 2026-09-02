import unittest

from scripts.run_repair_operator_ablations import CONFIGS


class RepairOperatorAblationsTest(unittest.TestCase):
    def test_configs_cover_singles_leave_one_out_and_full(self):
        self.assertEqual(CONFIGS["accommodation_only"], ("accommodation",))
        self.assertEqual(CONFIGS["cuisine_only"], ("cuisine",))
        self.assertEqual(CONFIGS["budget_only"], ("budget",))
        self.assertEqual(set(CONFIGS["all_operators"]), {"accommodation", "cuisine", "budget"})
        self.assertEqual(len(CONFIGS["without_accommodation"]), 2)
        self.assertEqual(len(CONFIGS["without_cuisine"]), 2)
        self.assertEqual(len(CONFIGS["without_budget"]), 2)


if __name__ == "__main__":
    unittest.main()

