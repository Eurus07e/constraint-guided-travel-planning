import unittest
import os
from pathlib import Path

from scripts.heldout_query_adapter import recover_evaluator_fields, validate_against_split


class HeldoutQueryAdapterTest(unittest.TestCase):
    def test_recover_explicit_constraints(self):
        record = {
            "org": "A",
            "dest": "B",
            "days": 5,
            "query": (
                "Plan a 3-day trip for a duo with a budget of $1,200. "
                "Visit 2 different cities. We require private rooms, "
                "pet-friendly lodging, Italian and Chinese food, and no flights."
            ),
        }
        recovered = recover_evaluator_fields(record)
        self.assertEqual(recovered["budget"], 1200)
        self.assertEqual(recovered["people_number"], 2)
        self.assertEqual(recovered["visiting_city_number"], 2)
        self.assertEqual(
            recovered["local_constraint"],
            "{'house rule': 'pets', 'cuisine': ['Italian', 'Chinese'], 'room type': 'private room', 'transportation': 'no flight'}",
        )

    def test_city_count_uses_benchmark_horizon(self):
        recovered = recover_evaluator_fields(
            {
                "org": "A",
                "dest": "B",
                "days": 7,
                "query": "Plan a seven-day trip for two with a budget of $4,000.",
            }
        )
        self.assertEqual(recovered["visiting_city_number"], 3)

    @unittest.skipUnless(os.getenv("TP_INTEGRATION") == "1", "requires the official benchmark data")
    def test_validation_split_recovery_is_exact(self):
        try:
            from datasets import Dataset, load_dataset
        except ImportError:
            self.skipTest("datasets is not installed")
        cache_paths = sorted(
            Path.home().glob(
                ".cache/huggingface/datasets/osunlp___travel_planner/validation/*/*/travel_planner-validation.arrow"
            )
        )
        records = (
            Dataset.from_file(str(cache_paths[-1]))
            if cache_paths
            else load_dataset(
                "osunlp/TravelPlanner",
                "validation",
                download_mode="reuse_cache_if_exists",
            )["validation"]
        )
        report = validate_against_split(records)
        self.assertEqual(report["totals"], {"records": 180, "budget": 0, "people_number": 0, "visiting_city_number": 0, "local_constraint": 0})
        self.assertEqual(report["mismatches"], [])


if __name__ == "__main__":
    unittest.main()
