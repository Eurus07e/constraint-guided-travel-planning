"""Exercise the existing entry points with real benchmark data and fake API responses."""
import contextlib
import io
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from utils.paths import DATABASE, FROZEN, ROOT


@unittest.skipUnless(os.getenv('TP_INTEGRATION') == '1', 'requires the official benchmark data')
class RunnerIntegrationTests(unittest.TestCase):
    def test_three_entry_points_save_and_resume_without_external_requests(self):
        seeds = [json.loads(line)['plan'] for line in (FROZEN / 'predictions/direct.jsonl').read_text().splitlines()]
        runners = [('strong_baseline_runner.py','constraint_direct_json'),
                   ('seeded_multi_agent_planner.py','multi_agent_seeded_r2_n3'),
                   ('contract_multi_agent_repair.py','cc_mar_r3')]
        for filename, strategy in runners:
            with self.subTest(runner=filename), tempfile.TemporaryDirectory() as directory:
                calls = []
                def respond(*args, **kwargs):
                    self.assertEqual(filename, 'strong_baseline_runner.py')
                    idx = len(calls); calls.append(idx)
                    response = requests.Response(); response.status_code = 200
                    response._content = json.dumps({'model':'offline-test', 'id':f'local-{idx}',
                        'usage':{'total_tokens':10}, 'choices':[{'message':{
                            'content':json.dumps({'plan':seeds[idx]})}, 'finish_reason':'stop'}]}).encode()
                    return response
                env = {'TP_DATABASE_DIR':str(DATABASE), 'OUT_ROOT':directory, 'STRATEGY':strategy,
                       'LIMIT':'3', 'RESUME':'1', 'MODEL_NAME':'offline-test', 'ALLOW_LLM':'0',
                       'ROUNDS':'0', 'MAX_WORKERS':'1', 'OPENAI_API_KEY':'local-test-only',
                       'OPENAI_API_BASE':'https://offline.invalid/v1',
                       'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1'}
                with patch.dict(os.environ, env, clear=True), patch('requests.post', side_effect=respond), contextlib.redirect_stdout(io.StringIO()):
                    first = runpy.run_path(str(ROOT / filename), run_name='__main__')
                    first_calls = len(calls)
                    resumed = runpy.run_path(str(ROOT / filename), run_name='__main__')
                self.assertEqual(first['OUT_ROOT'], resumed['OUT_ROOT'])
                self.assertEqual(len(calls), first_calls)
                self.assertEqual(first_calls, 3 if filename == 'strong_baseline_runner.py' else 0)
                status = json.loads((resumed['OUT_ROOT'] / 'status.json').read_text())
                rows = [json.loads(line) for line in resumed['SUBMISSION_FILE'].read_text().splitlines()]
                self.assertEqual(status['status'], 'completed')
                self.assertEqual([row['idx'] for row in rows], [1,2,3])
                self.assertTrue(all(isinstance(row['plan'], list) and row['plan'] for row in rows))

    def test_failed_model_calls_mark_cases_retryable_in_all_runners(self):
        seed=json.loads((FROZEN/'predictions/direct.jsonl').read_text().splitlines()[0])['plan']
        for filename,strategy in [('strong_baseline_runner.py','constraint_direct_json'),
                                  ('seeded_multi_agent_planner.py','multi_agent_seeded_r2_n3'),
                                  ('contract_multi_agent_repair.py','cc_mar_r3')]:
            with self.subTest(runner=filename), tempfile.TemporaryDirectory() as directory:
                env={'TP_DATABASE_DIR':str(DATABASE),'OUT_ROOT':directory,'STRATEGY':strategy,
                     'IDS':'1','RESUME':'1','MODEL_NAME':'offline-test','ALLOW_LLM':'1',
                     'ROUNDS':'1','CANDIDATES':'2','MAX_WORKERS':'1','OPENAI_API_KEY':'local-test-only',
                     'OPENAI_API_BASE':'https://offline.invalid/v1','MAX_RETRIES':'1'}
                rejected=requests.Response();rejected.status_code=401;rejected._content=b'{}'
                with patch.dict(os.environ,env,clear=True), contextlib.redirect_stdout(io.StringIO()):
                    with patch('requests.post',return_value=rejected) as post, self.assertRaises(SystemExit):
                        runpy.run_path(str(ROOT/filename),run_name='__main__')
                    self.assertEqual(post.call_count,1)
                    run_root=next(Path(directory).iterdir())
                    self.assertEqual(json.loads((run_root/'status.json').read_text())['failed_ids'],[1])
                    self.assertEqual(json.loads((run_root/'submission.jsonl').read_text())['plan'],[])
                    recovered=requests.Response();recovered.status_code=200
                    recovered._content=json.dumps({'model':'offline-test','choices':[{'message':{'content':json.dumps({
                        'plan':seed,'repairable':False,'judgments':[]})}}]}).encode()
                    with patch('requests.post',return_value=recovered) as post:
                        resumed=runpy.run_path(str(ROOT/filename),run_name='__main__')
                    self.assertGreater(post.call_count,0)
                    self.assertEqual(resumed['OUT_ROOT'],run_root.resolve())
                    self.assertEqual(json.loads((run_root/'status.json').read_text())['status'],'completed')

    def test_distance_conversion_preserves_all_stored_numeric_costs(self):
        import ast
        import pandas as pd
        from tools.googleDistanceMatrix.apis import distance_km
        distances=pd.read_csv(DATABASE/'googleDistanceMatrix/distance.csv')['distance'].dropna()
        for value in distances:
            historical=ast.literal_eval(value.replace('km','').replace(',','').strip())
            parsed=distance_km(value)
            self.assertEqual(int(parsed),int(historical))
            self.assertEqual(int(parsed*0.05),int(historical*0.05))

    def test_final_evaluator_handles_pass_and_fail_cases_in_clean_environment(self):
        import pandas as pd
        from evaluation.protocol import evaluate_rows
        manifest=json.loads((FROZEN/'manifest.json').read_text())
        records=pd.read_csv(DATABASE/'validation.csv').to_dict('records')
        for method in ['direct','seeded_repair_probe']:
            spec=manifest['methods'][method]
            labels=spec['final_pass']
            indices=[labels.index(False),labels.index(True)]
            predictions=[json.loads(line) for line in (FROZEN/spec['prediction_file']).read_text().splitlines()]
            before=os.getcwd()
            with contextlib.redirect_stdout(io.StringIO()):
                report=evaluate_rows([predictions[idx] for idx in indices],records)
            self.assertEqual(os.getcwd(),before)
            self.assertEqual(report['summary']['final_pass_count'],1)
            for case in report['cases']:
                self.assertEqual(case['final_pass'],labels[case['idx']-1])
            json.dumps(report)
