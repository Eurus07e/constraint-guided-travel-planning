"""Validate row selection and cached plan containers before runners use them."""
import os
import json
from pathlib import Path


def validate_runner_settings(namespace):
    for name, minimum in [('ROUNDS', 0), ('CANDIDATES', 1), ('MAX_WORKERS', 1),
                          ('MAX_REFERENCE_CHARS', 1), ('MAX_PLAN_TOKENS', 1)]:
        if name in namespace and (type(namespace[name]) is not int or namespace[name] < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if os.getenv('TP_AUDIT_VERSION', 'strict-v2') not in {'strict-v2', 'legacy-v1'}:
        raise ValueError('TP_AUDIT_VERSION must be strict-v2 or legacy-v1')


def load_seed_submission(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Seed submission not found: {path}')
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f'{path}:{line_number}: invalid JSON') from exc
        if not isinstance(row, dict) or type(row.get('idx')) is not int or not valid_plan_container(row.get('plan')):
            raise ValueError(f'{path}:{line_number}: expected an integer idx and a list of plan-day objects')
        rows.append(row)
    if not rows or sorted(row['idx'] for row in rows) != list(range(1, len(rows) + 1)):
        raise ValueError(f'Seed submission must contain unique contiguous one-based IDs: {path}')
    return sorted(rows, key=lambda row: row['idx'])


def valid_plan_container(value):
    return isinstance(value, list) and all(isinstance(day, dict) for day in value)


def select_rows(df):
    def parse_ids(value, name):
        try:
            ids = [int(part.strip()) for part in value.split(',')]
        except ValueError as exc:
            raise ValueError(f'{name} must contain comma-separated integer IDs') from exc
        if len(ids) != len(set(ids)):
            raise ValueError(f'{name} contains duplicate IDs')
        if any(idx < 1 or idx > len(df) for idx in ids):
            raise ValueError(f'{name} IDs must be between 1 and {len(df)}')
        return ids

    if os.getenv('IDS') is not None:
        selected = df.iloc[[idx - 1 for idx in parse_ids(os.environ['IDS'], 'IDS')]]
    elif os.getenv('LIMIT') is not None:
        limit = int(os.environ['LIMIT'])
        if limit < 1:
            raise ValueError('LIMIT must be positive')
        selected = df.head(limit)
    else:
        selected = df
    if os.getenv('EXCLUDE_IDS'):
        excluded = {idx - 1 for idx in parse_ids(os.environ['EXCLUDE_IDS'], 'EXCLUDE_IDS')}
        selected = selected.loc[[idx for idx in selected.index if idx not in excluded]]
    if selected.empty:
        raise ValueError('No rows selected')
    return selected
