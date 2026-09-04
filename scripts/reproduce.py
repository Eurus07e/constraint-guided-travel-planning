"""Reconstruct the historical selector and deterministic repair without model calls."""
import argparse
import json
from pathlib import Path
from utils.paths import DATABASE, ROOT


def reproduce(limit=None):
    import pandas as pd
    import seeded_multi_agent_planner as planner
    rows = pd.read_csv(DATABASE / 'validation.csv')
    direct = planner.load_jsonl(planner.DIRECT_SUBMISSION_FILE)
    program = planner.load_jsonl(planner.PROGRAM_SUBMISSION_FILE)
    if len(direct) != len(rows) or len(program) != len(rows):
        raise ValueError('Seed files must cover the complete validation table')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    output = []
    for i, row in rows.head(limit).iterrows() if limit else rows.iterrows():
        idx = i + 1
        task = planner.build_task_context(row)
        task["_audit_version"] = "legacy-v1"
        seeds = planner.load_seed_candidates(idx, task, direct, program)
        candidates = list(seeds)
        for seed in seeds:
            result = planner.deterministic_repair(task, seed)
            if result['improved']:
                candidates.append(planner.build_candidate(seed['candidate_id']+'_det_repaired', seed['seed_source'], 'revised', 1, task, result['plan'], planner.render_plan_text(result['plan'])))
        output.append({'idx': idx, 'plan': planner.rank_candidates(candidates)[0]['plan']})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/reproduced_repair.jsonl')
    args = parser.parse_args()
    rows = reproduce(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    print(f'Reconstructed {len(rows)} plans: {args.output}')


if __name__ == '__main__':
    main()
