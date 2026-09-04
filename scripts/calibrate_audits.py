"""Compare versioned internal audit decisions against frozen external outcomes."""
import json
from collections import Counter
import pandas as pd
from utils.paths import DATABASE, FROZEN, ROOT
from utils.plan_audit import audit_plan
from scripts.analyze_verifier_calibration import confusion_counts, confusion_metrics
from seeded_multi_agent_planner import build_task_context, normalize_plan_list


def main():
    records=pd.read_csv(DATABASE/'validation.csv');manifest=json.loads((FROZEN/'manifest.json').read_text());report={}
    for method in ['direct','verifier_hybrid_agent','seeded_repair_probe','multi_agent_seeded_r2_n3','cc_mar_r3']:
        spec=manifest['methods'][method];rows=[json.loads(x) for x in (FROZEN/spec['prediction_file']).read_text().splitlines()]
        per_version={};failures=Counter();examples=[]
        for version in ['legacy-v1','strict-v2']:
            predicted=[]
            for i,row in enumerate(rows):
                task=build_task_context(records.iloc[i]);plan=normalize_plan_list(row['plan'],task['days'])
                # Match the historical calibration: it audited raw frozen plans.
                audited=audit_plan(task,row['plan'],version)
                predicted.append(bool(audited['likely_pass']))
                if version=='strict-v2' and not spec['final_pass'][i]:
                    failures.update(x['contract'] for x in audited.get('strict_issues',[]))
                if version=='strict-v2' and audited['likely_pass']!=spec['final_pass'][i]:
                    examples.append({'idx':row['idx'],'audit_pass':bool(audited['likely_pass']),'final_pass':spec['final_pass'][i]})
            counts=confusion_counts(predicted,spec['final_pass'])
            per_version[version]={'confusion':counts,'metrics':confusion_metrics(counts),'audit_likely_pass':sum(predicted)}
        report[method]={'n':len(rows),'versions':per_version,'strict_failure_categories':dict(failures),'strict_disagreements':examples}
        print(method,{k:v['confusion'] for k,v in per_version.items()},flush=True)
    (ROOT/'results/audit_comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
