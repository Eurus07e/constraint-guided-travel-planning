"""Check the official database, or obtain the official validation task table."""
import argparse
import hashlib
import json
from pathlib import Path
from utils.paths import DATABASE, FROZEN

REQUIRED = {
    'validation.csv': ['org', 'dest', 'days', 'people_number', 'budget', 'local_constraint', 'reference_information'],
    'accommodations/clean_accommodations_2022.csv': ['NAME', 'price', 'minimum nights', 'city'],
    'attractions/attractions.csv': ['Name', 'City'],
    'flights/clean_Flights_2022.csv': ['Flight Number', 'Price', 'OriginCityName', 'DestCityName'],
    'restaurants/clean_restaurant_2022.csv': ['Name', 'Average Cost', 'City'],
    'googleDistanceMatrix/distance.csv': ['origin', 'destination', 'distance'],
    'background/citySet_with_states.txt': [],
}


def check_database(root, require_frozen=False):
    import pandas as pd
    expected = json.loads((FROZEN / 'manifest.json').read_text())['database_sha256']
    report = {}
    for name, columns in REQUIRED.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. See docs/reproduction.md for official data setup.')
        if columns:
            actual = pd.read_csv(path, nrows=0).columns
            missing = set(columns) - set(actual)
            if missing:
                raise ValueError(f'{name}: missing columns {sorted(missing)}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = digest == expected.get(name)
        if require_frozen and not matches:
            raise ValueError(f'{name}: data hash differs from the historical reproduction data')
        report[name] = {'sha256': digest, 'matches_historical_data': matches}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DATABASE)
    parser.add_argument('--fetch-validation', action='store_true')
    parser.add_argument('--require-frozen-hashes', action='store_true')
    args = parser.parse_args()
    if args.fetch_validation:
        from datasets import load_dataset
        target = args.database / 'validation.csv'
        if target.exists():
            raise FileExistsError(f'Refusing to replace {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        load_dataset('osunlp/TravelPlanner', 'validation')['validation'].to_pandas().to_csv(target, index=False)
    print(json.dumps(check_database(args.database, args.require_frozen_hashes), indent=2))


if __name__ == '__main__':
    main()
