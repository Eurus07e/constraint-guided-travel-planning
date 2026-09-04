import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
import contract_multi_agent_repair as cc
import seeded_multi_agent_planner as seeded
from utils.llm_client import ModelRequestError, RequestBudgetExceeded, complete


class ModelFailureTests(unittest.TestCase):
    def response(self, payload=None, status=200):
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload if payload is not None else {
            'choices':[{'message':{'content':'{}'}}]}).encode()
        return response

    def test_corrupt_cache_containers_are_replaced(self):
        for content in ['null', '[]', '42', '{"content":"   "}', 'truncated']:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {'OPENAI_API_KEY':'local-test'}, clear=True), \
                 patch('utils.llm_client.requests.post', return_value=self.response()) as post:
                cache = Path(directory) / 'cache'
                complete('s', 'u', cache_dir=cache, model='test')
                next(cache.glob('*.json')).write_text(content)
                self.assertEqual(complete('s', 'u', cache_dir=cache, model='test'), '{}')
                self.assertEqual(post.call_count, 2)

    def test_malformed_provider_envelopes_are_logged_as_failures(self):
        for payload in [[], {'choices':None}, {'choices':[None]}, {'choices':[{'message':None}]}]:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {'OPENAI_API_KEY':'local-test'}, clear=True), \
                 patch('utils.llm_client.requests.post', return_value=self.response(payload)) as post:
                with self.assertRaises(ModelRequestError):
                    complete('s', 'u', cache_dir=Path(directory)/'cache', model='test')
                self.assertEqual(post.call_count, 1)
                events = [json.loads(line) for line in (Path(directory)/'requests.jsonl').read_text().splitlines()]
                self.assertEqual([e['event'] for e in events], ['attempt','failure'])

    def test_invalid_limits_fail_before_any_request_or_output(self):
        for name, value in [('MAX_RETRIES','0'), ('REQUEST_TIMEOUT','-1'), ('TP_MAX_REQUESTS','-2')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {name:value,'OPENAI_API_KEY':'local-test'}, clear=True), \
                 patch('utils.llm_client.requests.post') as post:
                with self.assertRaisesRegex(ValueError, name):
                    complete('s', 'u', cache_dir=Path(directory)/'cache', model='test')
                post.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_transient_failure_retries_with_accounted_attempts(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'OPENAI_API_KEY':'local-test','MAX_RETRIES':'2'}, clear=True), \
             patch('utils.llm_client.requests.post', side_effect=[self.response({},503),self.response()]) as post, \
             patch('utils.llm_client.time.sleep'):
            self.assertEqual(complete('s', 'u', cache_dir=Path(directory)/'cache', model='test'), '{}')
            self.assertEqual(post.call_count, 2)
            events = [json.loads(line)['event'] for line in (Path(directory)/'requests.jsonl').read_text().splitlines()]
            self.assertEqual(events, ['attempt','failure','attempt','success'])

    def test_cc_agent_and_critic_do_not_swallow_request_failures(self):
        with patch.object(cc, 'call_llm', side_effect=ModelRequestError('HTTP 401')), \
             patch.object(cc, 'agent_prompt', return_value=('s','u')), \
             patch.object(cc, 'critic_prompt', return_value=('s','u')), \
             patch.object(cc, 'DISABLE_CRITIC', False):
            with self.assertRaises(ModelRequestError):
                cc.propose_from_agent('route_agent', {'days':1}, [], {}, 1, {}, 1)
            with self.assertRaises(ModelRequestError):
                cc.critic_judgments({}, [], {}, [], 1, 1)

    def test_seeded_stages_do_not_swallow_exhausted_budget(self):
        candidate = {'candidate_id':'a','seed_source':'direct','stage':'seed','summary':{},
                     'plan':[], 'audit':{'score':0,'fatal_count':1}}
        candidates = [candidate, {**candidate,'candidate_id':'b'}]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(seeded, 'ALLOW_LLM', True))
            stack.enter_context(patch.object(seeded, 'call_deepseek_json', side_effect=RequestBudgetExceeded('budget exhausted')))
            for name in ['planner_prompt','verifier_prompt','reflector_prompt','reviser_prompt','selector_prompt']:
                stack.enter_context(patch.object(seeded, name, return_value=('s','u')))
            stack.enter_context(patch.object(seeded, 'deterministic_reflection', return_value={'instructions':[]}))
            stack.enter_context(patch.object(seeded, 'deterministic_repair', return_value={'improved':False}))
            stack.enter_context(patch.object(seeded, 'select_candidates_for_llm_verification', return_value={'a'}))
            stack.enter_context(patch.object(seeded, 'rank_candidates', return_value=candidates))
            stack.enter_context(patch.object(seeded, 'get_base_seed_best', return_value=candidate))
            stack.enter_context(patch.object(seeded, 'should_promote_candidate', return_value=True))
            actions = [lambda:seeded.generate_scratch_candidate({'days':1},1,candidates),
                       lambda:seeded.verify_candidates({},1,[candidate],'test'),
                       lambda:seeded.reflect_on_candidates({},1,[candidate]),
                       lambda:seeded.revise_candidate({'days':1},1,candidate,{}),
                       lambda:seeded.choose_final_candidate({},1,candidates)]
            for number, action in enumerate(actions):
                with self.subTest(stage=number), self.assertRaises(RequestBudgetExceeded):
                    action()
