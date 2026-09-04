"""Check the official database, or obtain the official validation task table."""
import argparse
import hashlib
import json
from pathlib import Path
from utils.paths import DATABASE, FROZEN

REQUIRED = {
    'validation.csv': ['org', 'dest', 'days', 'visiting_city_number', 'date', 'people_number', 'budget',
                       'query', 'level', 'local_constraint', 'reference_information'],
    'accommodations/clean_accommodations_2022.csv': ['NAME', 'price', 'room type', 'house_rules', 'minimum nights',
                                                   'maximum occupancy', 'review rate number', 'city'],
    'attractions/attractions.csv': ['Name', 'Latitude', 'Longitude', 'Address', 'Phone', 'Website', 'City'],
    'flights/clean_Flights_2022.csv': ['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime',
                                      'FlightDate', 'OriginCityName', 'DestCityName', 'Distance'],
    'restaurants/clean_restaurant_2022.csv': ['Name', 'Average Cost', 'Cuisines', 'Aggregate Rating', 'City'],
    'googleDistanceMatrix/distance.csv': ['origin', 'destination', 'distance', 'duration'],
    'background/citySet_with_states.txt': [],
}


def check_database(root, require_frozen=False):
    import pandas as pd
    root = Path(root).expanduser().resolve()
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
