"""Re-score every frozen prediction and check the published table and case labels."""
import json
from pathlib import Path
from utils.paths import FROZEN, ROOT
from evaluation.protocol import evaluate_file

SCORE_KEYS={'delivery_rate':'delivery_rate','commonsense_micro':'commonsense_micro_rate','commonsense_macro':'commonsense_macro_rate','hard_micro':'hard_micro_rate','hard_macro':'hard_macro_rate','final_pass_rate':'final_pass_rate'}


def main():
    manifest=json.loads((FROZEN/'manifest.json').read_text());reports={}
    for method,spec in manifest['methods'].items():
        result=evaluate_file(FROZEN/spec['prediction_file'],'validation',True)
        discrepancies={key:{'expected':value,'actual':result['summary'][SCORE_KEYS[key]]*100} for key,value in spec['expected_scores'].items() if abs(result['summary'][SCORE_KEYS[key]]*100-value)>0.0051}
        changed=[c['idx'] for c,label in zip(result['cases'],spec['final_pass']) if c['final_pass']!=label]
        reports[method]={'summary':result['summary'],'discrepancies':discrepancies,'changed_final_pass_ids':changed}
        target=ROOT/'runs'/'frozen_replay'/f'{method}.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
        print(method,result['summary']['final_pass_count'],'discrepancies',discrepancies,'changed_ids',changed,flush=True)
    target=ROOT/'results'/'replay_verification.json';target.write_text(json.dumps({'protocol':result['protocol'],'methods':reports},indent=2)+'\n')
    if any(r['discrepancies'] or r['changed_final_pass_ids'] for r in reports.values()):raise SystemExit('Historical replay discrepancies require investigation')


if __name__=='__main__':main()
