import unittest
from evaluation.protocol import evaluate_rows, validate_submission


def task():
    return {'days':1,'level':'easy','local_constraint':{'house rule':None,'cuisine':None,'room type':None,'transportation':None}}


def plan():
    return [{'days':1,**{k:'-' for k in ['current_city','transportation','breakfast','attraction','lunch','dinner','accommodation']}}]


class IndexedEvaluationTests(unittest.TestCase):
    def test_out_of_order_is_joined_by_id_and_missing_full_ids_rejected(self):
        records=[task(),task()]
        rows=[{'idx':2,'plan':[]},{'idx':1,'plan':[]}]
        self.assertEqual([r['idx'] for r in validate_submission(rows,records,True)],[1,2])
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            validate_submission(rows[:1],records,True)
        for bad in [[rows[0],rows[0]],[{'idx':0,'plan':[]}],[{'idx':True,'plan':[]}],[{'idx':1,'plan':None}]]:
            with self.assertRaises(ValueError):validate_submission(bad,records)

    def test_missing_output_stays_in_micro_denominator(self):
        common={k:(True,None) for k in ['is_not_absent','is_valid_information_in_sandbox','a','b','c','d','e','f']}
        hard={'valid_cost':(True,None)}
        result=evaluate_rows([{'idx':1,'plan':plan()},{'idx':2,'plan':[]}],[task(),task()],True,(lambda *_:common,lambda *_:hard))
        self.assertEqual(result['summary']['commonsense_micro_rate'],0.5)
        self.assertEqual(result['summary']['hard_micro_rate'],0.5)
        self.assertEqual(result['summary']['final_pass_rate'],0.5)

    def test_malformed_days_fail_without_invoking_scorer(self):
        result=evaluate_rows([{'idx':1,'plan':[{'days':1}]}],[task()],False,(lambda *_:self.fail(),lambda *_:self.fail()))
        self.assertFalse(result['cases'][0]['schema_ok'])
        self.assertFalse(result['cases'][0]['final_pass'])
