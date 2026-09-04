import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.runner_inputs import load_seed_submission, validate_runner_settings
import seeded_multi_agent_planner as seeded


class RunnerInputTests(unittest.TestCase):
    def test_seed_identity_and_container_errors_report_source_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'seeds.jsonl'
            for content in ['null\n', '{"idx":1,"plan":null}\n', '{"idx":true,"plan":[]}\n',
                            '{"idx":1,"plan":[null]}\n', 'invalid\n']:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'seeds.jsonl:1'):
                    load_seed_submission(path)
            path.write_text('{"idx":2,"plan":[]}\n{"idx":1,"plan":[]}\n')
            self.assertEqual([row['idx'] for row in load_seed_submission(path)], [1,2])
            for rows in [[], [{'idx':2,'plan':[]}], [{'idx':1,'plan':[]},{'idx':1,'plan':[]}]]:
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.assertRaisesRegex(ValueError, 'contiguous'):
                    load_seed_submission(path)

    def test_corrupt_debug_record_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(seeded, 'DEBUG_DIR', Path(directory)):
            path=Path(directory)/'debug_1.json'
            for content in ['[]','42','null','truncated']:
                path.write_text(content)
                self.assertIsNone(seeded.load_debug(1))

    def test_invalid_configuration_is_rejected(self):
        for namespace in [{'ROUNDS':-1}, {'CANDIDATES':0}, {'MAX_WORKERS':0}, {'MAX_PLAN_TOKENS':0}]:
            with self.subTest(namespace=namespace), self.assertRaises(ValueError):
                validate_runner_settings(namespace)
        with patch.dict(os.environ, {'TP_AUDIT_VERSION':'typo'}), self.assertRaisesRegex(ValueError,'TP_AUDIT_VERSION'):
            validate_runner_settings({})
