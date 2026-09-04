import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scripts.prepare_data import REQUIRED, check_database
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix, distance_km
from utils.func import get_city_list


class DataToolTests(unittest.TestCase):
    def test_distance_text_is_numeric_data(self):
        self.assertEqual(distance_km('1,234 km'), 1234)
        self.assertEqual(distance_km('12.5 km'), 12.5)
        self.assertEqual(distance_km('500 m'), 0.5)
        for value in ['1+2 km', '__import__("os") km', 'nan km', '-5 km', '1,23 km']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                distance_km(value)

    def test_ground_transport_costs_and_missing_values(self):
        rows=pd.DataFrame([{'origin':'A','destination':'B','duration':'2 hours','distance':'1,234 km'},
                           {'origin':'C','destination':'D','duration':float('nan'),'distance':float('nan')}])
        with patch('tools.googleDistanceMatrix.apis.pd.read_csv', return_value=rows):
            tool=GoogleDistanceMatrix()
        self.assertEqual(tool.run_for_evaluation('A','B','self-driving')['cost'], 61)
        self.assertEqual(tool.run_for_evaluation('A','B','taxi')['cost'], 1234)
        self.assertIsNone(tool.run_for_evaluation('C','D')['cost'])
        self.assertEqual(tool.run('C','D'), 'No valid information.')

    def test_city_lookup_uses_configured_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'background').mkdir()
            (root/'background/citySet_with_states.txt').write_text('A\tState\nB\tState\n')
            with patch('utils.func.DATABASE', root):
                self.assertEqual(get_city_list(5,'A','State'), ['A','B(State)'])

    def test_preparation_rejects_missing_columns_used_by_runners(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for filename, columns in REQUIRED.items():
                path=root/filename;path.parent.mkdir(parents=True,exist_ok=True)
                if columns:pd.DataFrame(columns=columns).to_csv(path,index=False)
                else:path.write_text('A\tState\n')
            self.assertEqual(len(check_database(root)), len(REQUIRED))
            path=root/'validation.csv'
            pd.DataFrame(columns=[column for column in REQUIRED['validation.csv'] if column!='query']).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'query'):
                check_database(root)

    def test_online_distance_request_has_finite_retries_and_timeout(self):
        from requests.exceptions import SSLError
        with patch('tools.googleDistanceMatrix.apis.pd.read_csv', return_value=pd.DataFrame()):
            tool=GoogleDistanceMatrix()
        with patch('tools.googleDistanceMatrix.apis.requests.get',side_effect=SSLError('failed')) as get, \
             patch('tools.googleDistanceMatrix.apis.time.sleep'):
            with self.assertRaises(SSLError):
                tool.run_online('A','B','taxi')
            self.assertEqual(get.call_count,3)
            self.assertEqual(get.call_args.kwargs['timeout'],30)
            self.assertEqual(get.call_args.kwargs['params']['mode'],'driving')
