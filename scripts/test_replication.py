import json
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.run_replication import validate_config
import strong_baseline_runner as runner
from utils.paths import ROOT

class ReplicationTests(unittest.TestCase):
    def test_replication_matrix_requires_independent_runs_and_labeled_split(self):
        config=json.loads((ROOT/'configs/replication.json').read_text())
        self.assertEqual(validate_config(config)['repeats'],3)
        with self.assertRaises(ValueError):validate_config({**config,'repeats':1})
        with self.assertRaises(ValueError):validate_config({**config,'set_type':'test'})

    def test_malformed_direct_response_does_not_import_an_old_seed(self):
        with patch.object(runner,'call_llm',return_value='not a plan'),patch.object(runner,'load_direct_baseline_plan',side_effect=AssertionError('must not use historical fallback')),patch.object(runner,'direct_prompt',return_value=('system','user')):
            result=runner.run_direct({'days':3},1)
        self.assertEqual(result['plan'],[])
        self.assertEqual(result['fallback'],'schema_quality_warning')
