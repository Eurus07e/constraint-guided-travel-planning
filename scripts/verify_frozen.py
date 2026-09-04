"""Verify frozen file hashes, sample identities and published outcome counts offline."""
import hashlib
import json
from utils.paths import FROZEN


def verify(root=FROZEN):
    manifest = json.loads((root / 'manifest.json').read_text())
    counts = {}
    for method, spec in manifest['methods'].items():
        path = root / spec['prediction_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError(f'Prediction hash mismatch: {method}')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if [r['idx'] for r in rows] != list(range(1, manifest['instances'] + 1)):
            raise ValueError(f'Invalid sample identities: {method}')
        if any(set(r) != {'idx', 'plan'} or not isinstance(r['plan'], list) for r in rows):
            raise ValueError(f'Unexpected prediction fields: {method}')
        outcomes = spec['final_pass']
        if len(outcomes) != len(rows) or any(type(x) is not bool for x in outcomes):
            raise ValueError(f'Invalid outcomes: {method}')
        count = sum(outcomes)
        if abs(100 * count / len(rows) - spec['expected_scores']['final_pass_rate']) > 0.0051:
            raise ValueError(f'Headline score mismatch: {method}')
        counts[method] = count
    return {'instances': manifest['instances'], 'methods': len(counts), 'final_pass_counts': counts}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
