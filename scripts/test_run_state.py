import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
import requests
from utils.run_state import ensure_manifest,atomic_write_text
from utils.llm_client import complete

class RunStateTests(unittest.TestCase):
    def test_changed_configuration_cannot_resume_old_run(self):
        with tempfile.TemporaryDirectory() as folder:
            ensure_manifest(folder,{'rounds':2,'seed_hash':'a'})
            ensure_manifest(folder,{'rounds':2,'seed_hash':'a'})
            with self.assertRaisesRegex(ValueError,'mismatch'):ensure_manifest(folder,{'rounds':3,'seed_hash':'a'})

    def test_atomic_write_preserves_old_file_if_replace_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json';atomic_write_text(path,'old')
            with patch('utils.run_state.os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):atomic_write_text(path,'new')
            self.assertEqual(path.read_text(),'old')
            self.assertEqual(list(Path(folder).iterdir()),[path])

    def response(self):
        r=Mock();r.status_code=200;r.headers={};r.json.return_value={'model':'provider-model','id':'response-id','choices':[{'message':{'content':'{}'},'finish_reason':'stop'}],'usage':{'total_tokens':12}};return r

    def test_cache_isolated_by_endpoint_and_run_and_works_offline(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'OPENAI_API_KEY':'test-key','OPENAI_API_BASE':'https://a.example/v1','TP_RUN_ID':'r1'}),patch('utils.llm_client.requests.post',return_value=self.response()) as post:
            args=dict(cache_dir=Path(folder)/'cache',model='m')
            complete('s','u',**args)
            os.environ.pop('OPENAI_API_KEY')
            complete('s','u',**args)
            self.assertEqual(post.call_count,1)
            os.environ['OPENAI_API_KEY']='test-key';os.environ['TP_RUN_ID']='r2'
            complete('s','u',**args)
            os.environ['OPENAI_API_BASE']='https://b.example/v1';complete('s','u',**args)
            self.assertEqual(post.call_count,3)
            events=[json.loads(x) for x in (Path(folder)/'requests.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['event']=='success' for e in events),3)
            self.assertNotIn('test-key',(Path(folder)/'requests.jsonl').read_text())

    def test_auth_failure_is_not_retried(self):
        r=self.response();r.status_code=401;r.raise_for_status.side_effect=requests.HTTPError('unauthorized')
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'OPENAI_API_KEY':'test-key'}),patch('utils.llm_client.requests.post',return_value=r) as post:
            with self.assertRaisesRegex(RuntimeError,'HTTP 401'):complete('s','u',cache_dir=Path(folder)/'cache',model='m')
            self.assertEqual(post.call_count,1)

    def test_budget_counts_attempts_and_prevents_another_request(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'OPENAI_API_KEY':'test-key','TP_MAX_REQUESTS':'1'}),patch('utils.llm_client.requests.post',return_value=self.response()) as post:
            args=dict(cache_dir=Path(folder)/'cache',model='m')
            complete('s','u',**args)
            with self.assertRaisesRegex(RuntimeError,'exhausted'):complete('s','different',**args)
            self.assertEqual(post.call_count,1)
