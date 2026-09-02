import unittest
from unittest.mock import patch

import pandas as pd

import strong_baseline_runner as runner


class StrongBaselineRunnerTest(unittest.TestCase):
    def test_chat_completions_url_accepts_versioned_base(self):
        self.assertEqual(
            runner.chat_completions_url("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )

    def test_chat_completions_url_adds_version_to_unversioned_base(self):
        self.assertEqual(
            runner.chat_completions_url("https://api.deepseek.com"),
            "https://api.deepseek.com/v1/chat/completions",
        )

    def test_select_rows_excludes_one_based_pilot_ids(self):
        frame = pd.DataFrame({"value": [1, 2, 3, 4]})
        with patch.dict("os.environ", {"EXCLUDE_IDS": "1,3"}, clear=False):
            selected = runner.select_rows(frame)
        self.assertEqual(selected.index.tolist(), [1, 3])

    def test_submission_path_can_be_overridden_for_independent_runs(self):
        custom_path = "evaluation/qwen_r2.jsonl"
        with patch.dict("os.environ", {"SUBMISSION_FILE": custom_path}, clear=False):
            path = runner.submission_file_path(
                "test", "qwen3.5-plus", "constraint_direct_json", "sole-planning"
            )
        self.assertEqual(str(path), custom_path)


if __name__ == "__main__":
    unittest.main()
