import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import strong_baseline_runner as baseline
from utils.paths import ROOT
from utils.runner_inputs import select_rows
from utils.run_state import configure_run


class RunnerRecoveryTests(unittest.TestCase):
    def load_runner(self, name):
        spec = importlib.util.spec_from_file_location('isolated_' + name, ROOT / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_invalid_selection_cannot_alias_or_duplicate_tasks(self):
        frame = pd.DataFrame({'query': ['a', 'b', 'c']})
        for env in [{'IDS':'0'}, {'IDS':'1,1'}, {'IDS':'4'}, {'IDS':''},
                    {'LIMIT':'0'}, {'IDS':'1', 'EXCLUDE_IDS':'1'}]:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    select_rows(frame)
        with patch.dict(os.environ, {'IDS':'3,1', 'EXCLUDE_IDS':'1'}, clear=True):
            self.assertEqual(select_rows(frame).index.tolist(), [2])

    def test_explicit_draft_is_used_and_invalid_source_does_not_trigger_new_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'validation').mkdir(); (root / 'debug').mkdir()
            output = root / 'validation/generated_plan_1.json'
            key = f'{baseline.MODEL_NAME}_constraint_direct_json_{baseline.MODE}_parsed_results'
            plan = [{'days':1}]
            output.write_text(json.dumps([{key:plan}]))
            (root / 'debug/debug_1.json').write_text('{"status":"completed"}')
            with patch.dict(os.environ, {'CONSTRAINT_DIRECT_OUTPUT_DIR':directory}), \
                 patch.object(baseline, 'CONSTRAINT_DIRECT_OUTPUT_DIR', root), \
                 patch.object(baseline, 'SET_TYPE', 'validation'):
                self.assertEqual(baseline.load_constraint_direct_draft(1)['plan'], plan)
                output.write_text('not json')
                with patch.object(baseline, 'run_direct', side_effect=AssertionError('must not change inputs')):
                    with self.assertRaisesRegex(ValueError, 'Cannot reuse explicit Direct draft'):
                        baseline.run_self_refine({'days':1}, 1)

    def test_explicit_draft_content_changes_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'
            (source / 'validation').mkdir(parents=True)
            output = source / 'validation/generated_plan_1.json'
            output.write_text('[{"plan":[]}]')
            env = {'CONSTRAINT_DIRECT_OUTPUT_DIR':str(source), 'OUT_ROOT':str(root / 'runs')}
            def namespace():
                return {'__file__':str(ROOT / 'strong_baseline_runner.py'), '__name__':'test',
                        'STRATEGY':'test', 'SET_TYPE':'validation', 'CONSTRAINT_DIRECT_OUTPUT_DIR':source}
            with patch.dict(os.environ, env, clear=True), patch('utils.run_state.DATABASE', root / 'empty'), contextlib.redirect_stdout(io.StringIO()):
                before = namespace(); configure_run(before)
                output.write_text('[{"plan":[{"days":1}]}]')
                after = namespace(); configure_run(after)
            self.assertNotEqual(before['RUN_FINGERPRINT'], after['RUN_FINGERPRINT'])

    def test_failed_cases_remain_visible_and_resume_retries_only_failures(self):
        for name in ['strong_baseline_runner', 'contract_multi_agent_repair']:
            with self.subTest(runner=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                frame = pd.DataFrame({'query':['a','b']})
                env = {'OUT_ROOT':str(root / 'runs'), 'MODEL_NAME':'offline-test', 'MAX_WORKERS':'1'}
                with patch.dict(os.environ, env, clear=True), patch('utils.run_state.DATABASE', root / 'no-database'), contextlib.redirect_stdout(io.StringIO()):
                    calls = []
                    def execute(fail_second):
                        module = self.load_runner(name)
                        def run_one(row, idx, *seeds):
                            calls.append(idx)
                            if idx == 2 and fail_second:
                                raise RuntimeError('simulated unavailable provider')
                            plan = [{'days':1, **{key:'-' for key in ['current_city','transportation','breakfast','lunch','dinner','attraction','accommodation']}}]
                            (module.DEBUG_DIR / f'debug_{idx}.json').write_text('{"status":"completed"}')
                            return plan, {'status':'completed'}
                        with patch.object(module, 'run_one', side_effect=run_one), contextlib.ExitStack() as stack:
                            if name == 'strong_baseline_runner':
                                stack.enter_context(patch.object(module, 'load_records_dataframe', return_value=frame))
                            else:
                                stack.enter_context(patch.object(module.pd, 'read_csv', return_value=frame))
                                stack.enter_context(patch.object(module, 'load_jsonl', return_value=[{'idx':1,'plan':[]},{'idx':2,'plan':[]}]))
                            if fail_second:
                                with self.assertRaises(SystemExit): module.main()
                            else:
                                module.main()
                        return module
                    failed = execute(True)
                    saved = [json.loads(line) for line in failed.SUBMISSION_FILE.read_text().splitlines()]
                    self.assertEqual([row['idx'] for row in saved], [1,2])
                    self.assertEqual(saved[1]['plan'], [])
                    self.assertEqual(json.loads((failed.OUT_ROOT / 'status.json').read_text())['failed_ids'], [2])
                    calls.clear()
                    resumed = execute(False)
                    self.assertEqual(resumed.OUT_ROOT, failed.OUT_ROOT)
                    self.assertEqual(calls, [2])
                    self.assertEqual(json.loads((resumed.OUT_ROOT / 'status.json').read_text())['status'], 'completed')

    def test_malformed_direct_response_does_not_import_old_submission(self):
        with patch.object(baseline, 'call_llm', return_value='not a plan'), \
             patch.object(baseline, 'load_direct_baseline_plan', side_effect=AssertionError('must not substitute old data')), \
             patch.object(baseline, 'direct_prompt', return_value=('system','user')):
            result = baseline.run_direct({'days':3}, 1)
        self.assertEqual(result['plan'], [])
        self.assertEqual(result['fallback'], 'schema_quality_warning')
