"""Indexed TravelPlanner evaluation with task-defined micro denominators.

Criterion implementations retain the published local scorer semantics. Skipped
criteria contribute zero passes and remain in the applicable denominator.
"""
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOL_VERSION = 'travelplanner-local-v1-indexed'
CONSTRAINT_KEYS = ('house rule', 'cuisine', 'room type', 'transportation')


def local_constraints(record):
    value = record.get('local_constraint', {})
    return ast.literal_eval(value) if isinstance(value, str) else dict(value)


def validate_submission(rows, records, require_complete=False):
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get('idx')) is not int:
            raise ValueError('Every submission row must have an integer idx')
        idx = row['idx']
        if idx in indexed:
            raise ValueError(f'Duplicate submission idx: {idx}')
        if not 1 <= idx <= len(records):
            raise ValueError(f'Submission idx outside dataset: {idx}')
        plan = row.get('plan')
        if not isinstance(plan, list):
            raise ValueError(f'idx={idx}: plan must be a list; use [] for failed delivery')
        if any(not isinstance(day, dict) for day in plan):
            raise ValueError(f'idx={idx}: every plan day must be an object')
        indexed[idx] = row
    if not indexed:
        raise ValueError('No submission rows')
    if require_complete and set(indexed) != set(range(1, len(records)+1)):
        missing = sorted(set(range(1, len(records)+1)) - set(indexed))
        raise ValueError(f'Incomplete full submission; missing idx: {missing[:20]}')
    return [indexed[idx] for idx in sorted(indexed)]


def criterion_pass(info):
    if not info:
        return False, ['not_evaluated'], 0, 0
    values = {name: raw[0] for name, raw in info.items() if raw[0] is not None}
    failures = [name for name, value in values.items() if not bool(value)]
    return not failures, failures, sum(bool(v) for v in values.values()), len(values)


def summarize_cases(cases):
    n = len(cases)
    if not n:
        raise ValueError('No cases to summarize')
    common_total = sum(c['commonsense_micro_total'] for c in cases)
    hard_total = sum(c['hard_micro_total'] for c in cases)
    count = sum(c['final_pass'] for c in cases)
    return {
        'n': n,
        'delivery_rate': sum(c['delivered'] for c in cases)/n,
        'commonsense_macro_rate': sum(c['commonsense_pass'] for c in cases)/n,
        'hard_macro_rate': sum(c['hard_pass'] for c in cases)/n,
        'final_pass_rate': count/n, 'final_pass_count': count,
        'commonsense_micro_rate': sum(c['commonsense_micro_passed'] for c in cases)/common_total if common_total else None,
        'hard_micro_rate': sum(c['hard_micro_passed'] for c in cases)/hard_total if hard_total else None,
        'commonsense_micro_total': common_total, 'hard_micro_total': hard_total,
    }


def evaluate_rows(rows, records, require_complete=False, evaluators=None, strict_schema=False):
    rows = validate_submission(rows, records, require_complete)
    if evaluators is None:
        from evaluation.commonsense_constraint import evaluation as common_eval
        from evaluation.hard_constraint import evaluation as hard_eval
    else:
        common_eval, hard_eval = evaluators
    cases = []
    required = {'days','current_city','transportation','breakfast','attraction','lunch','dinner','accommodation'}
    for row in rows:
        idx = row['idx']; record = dict(records[idx-1]); plan = row['plan']
        record['local_constraint'] = local_constraints(record)
        missing = any(not required.issubset(day) for day in plan)
        invalid_values = any(type(day['days']) is not int or any(not isinstance(day[k], str) for k in required-{'days'}) for day in plan) if not missing else True
        schema_ok = bool(plan) and len(plan) == int(record['days']) and not missing and not invalid_values
        if schema_ok:
            schema_ok = [day['days'] for day in plan] == list(range(1, int(record['days'])+1))
        scoreable = bool(plan) and all((required-{'days'}).issubset(day) and all(isinstance(day[k], str) for k in required-{'days'}) for day in plan)
        common = common_eval(record, plan) if scoreable and (schema_ok or not strict_schema) else None
        hard = None
        if common and common['is_not_absent'][0] and common['is_valid_information_in_sandbox'][0]:
            hard = hard_eval(record, plan)
        cp, cf, cm, _ = criterion_pass(common); hp, hf, hm, _ = criterion_pass(hard)
        cases.append({'idx':idx, 'level':record['level'], 'days':int(record['days']),
            'delivered':bool(plan), 'schema_ok':schema_ok,
            'commonsense_pass':cp, 'hard_pass':hp, 'final_pass':cp and hp,
            'commonsense_failures':cf, 'hard_failures':hf,
            'commonsense_micro_passed':cm, 'commonsense_micro_total':8,
            'hard_micro_passed':hm, 'hard_micro_total':1+sum(record['local_constraint'].get(k) is not None for k in CONSTRAINT_KEYS),
            'constraint_results':{'commonsense':common,'hard':hard}})
    by_level = defaultdict(list); by_days = defaultdict(list); failures = Counter()
    for c in cases:
        by_level[c['level']].append(c);by_days[str(c['days'])].append(c)
        failures.update('commonsense:'+name for name in c['commonsense_failures'])
        failures.update('hard:'+name for name in c['hard_failures'])
    return {'protocol':PROTOCOL_VERSION,'summary':summarize_cases(cases),'cases':cases,
        'by_level':{k:summarize_cases(v) for k,v in sorted(by_level.items())},
        'by_days':{k:summarize_cases(v) for k,v in sorted(by_days.items())},'failure_counts':dict(failures)}


def evaluate_file(input_path, set_type, require_complete=False):
    from utils.local_data import load_travelplanner_records
    rows = [json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
    result = evaluate_rows(rows, load_travelplanner_records(set_type), require_complete)
    result['set_type'] = set_type
    return result


def official_scores(summary):
    return {name:summary[key] for name,key in {
        'Delivery Rate':'delivery_rate','Commonsense Constraint Micro Pass Rate':'commonsense_micro_rate',
        'Commonsense Constraint Macro Pass Rate':'commonsense_macro_rate','Hard Constraint Micro Pass Rate':'hard_micro_rate',
        'Hard Constraint Macro Pass Rate':'hard_macro_rate','Final Pass Rate':'final_pass_rate'}.items()}
