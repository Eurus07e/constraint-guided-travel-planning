import unittest
from unittest.mock import patch
from copy import deepcopy
from utils.plan_audit import strict_issues
import contract_multi_agent_repair as cc

class StrictAuditTests(unittest.TestCase):
    def test_missing_information_and_repeated_meals_block_early_stop(self):
        day={'days':1,'current_city':'Boston','transportation':'-','breakfast':'Cafe, Boston','lunch':'Cafe, Boston','dinner':'-','attraction':'-','accommodation':'-'}
        contracts={x['contract'] for x in strict_issues({'days':2},[day])}
        self.assertIn('complete_information',contracts)
        self.assertIn('restaurant_diversity',contracts)

    def test_city_name_is_not_a_substring_match(self):
        day={'days':1,'current_city':'York','transportation':'-','breakfast':'Cafe, New York','lunch':'B, York','dinner':'C, York','attraction':'A, York;','accommodation':'-'}
        self.assertIn('current_city_alignment',{x['contract'] for x in strict_issues({'days':1},[day])})

    def test_patch_is_atomic_and_role_scoped(self):
        plan=[{'days':1,'current_city':'A','accommodation':'Hotel, A'}];before=deepcopy(plan)
        with self.assertRaises(ValueError):cc.apply_patches(plan,[{'day':1,'field':'days','value':'2'}])
        self.assertEqual(plan,before)
        with patch.object(cc,'GENERIC_AGENTS',False):
            with self.assertRaises(ValueError):cc.validate_role_changes('lodging_agent',plan,[{**plan[0],'current_city':'B'}])

    def test_strict_constraint_regressions_are_rejected(self):
        self.assertIn('regressed',cc.protected_regression({'strict_checks':{'restaurant_diversity':True}},{'strict_checks':{'restaurant_diversity':False}}))

    def test_critic_invalid_proposal_is_not_scored_or_promoted(self):
        proposal={'proposal_id':'p','agent_id':'route_agent','repairable':True,'plan':[{'days':1}]}
        with patch.object(cc,'audit_plan',side_effect=AssertionError('must not score invalid proposal')):
            _,_,accepted,decisions=cc.mediate({},[],{},[proposal],{'p':{'status':'invalid'}})
        self.assertIsNone(accepted)
        self.assertEqual(decisions[0]['reason'],'critic_invalid')

    def test_database_matching_preserves_case_sensitive_eligibility(self):
        import pandas as pd
        from utils.plan_audit import strict_entity_eligible
        strict_entity_eligible.cache_clear()
        rows=pd.DataFrame([{'NAME':'private spacious bedroom','city':'Seattle'}])
        with patch('utils.plan_audit._lookup_accommodation',return_value=rows):
            self.assertFalse(strict_entity_eligible('Private spacious bedroom, Seattle','accommodation'))
            self.assertTrue(strict_entity_eligible('private spacious bedroom, Seattle','accommodation'))
        strict_entity_eligible.cache_clear()
