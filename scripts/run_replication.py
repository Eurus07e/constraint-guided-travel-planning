"""Run independent, fingerprinted model replications with shared per-case seeds.

Credentials come from the environment. Run once per model/provider. Dry-run
validates configuration and reports the experiment size without model requests.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from utils.paths import ROOT,DATABASE,FROZEN
from utils.run_state import atomic_write_text,digest,file_digest,ensure_manifest,public_endpoint


def _run_case(job):
    config,model,repeat,idx,root_text=job
    root=Path(root_text)/f'r{repeat}'/f'case_{idx}'
    checkpoint=root/'case.json'
    if checkpoint.exists():
        existing=json.loads(checkpoint.read_text())
        if existing.get('status')=='completed':return existing
    os.environ.update(MODEL_NAME=model,TP_RUN_ID=f'{model}-r{repeat}-case{idx}',TP_AUDIT_VERSION=config['audit_version'],
        MAX_RETRIES='2',REQUEST_TIMEOUT='180',TP_MAX_REQUESTS=str(config['max_requests_per_method']),
        MAX_REFERENCE_CHARS=str(config['reference_chars']),MAX_PLAN_TOKENS=str(config['max_plan_tokens']))
    import pandas as pd
    import strong_baseline_runner as direct
    import seeded_multi_agent_planner as seeded
    import contract_multi_agent_repair as cc
    from evaluation.protocol import evaluate_rows
    from scripts.summarize_requests import summarize
    for runner in (direct,seeded,cc):
        runner.MODEL_NAME=model
        runner.MAX_REFERENCE_CHARS=config['reference_chars']
    direct.MAX_PLAN_TOKENS=config['max_plan_tokens']
    cc.ROUNDS=config['cc_mar_rounds'];seeded.ROUNDS=config['seeded_rounds'];seeded.CANDIDATES=config['seeded_candidates']
    records=pd.read_csv(DATABASE/'validation.csv').to_dict('records');record=records[idx-1]
    task=seeded.build_task_context(record)
    direct_task=direct.build_task(record)
    program=seeded.load_jsonl(FROZEN/'predictions/program_v23_best.jsonl')
    def paths(runner,method):
        runner.CACHE_DIR=root/method/'cache';runner.DEBUG_DIR=root/method/'debug';runner.OUTPUT_DIR=root/method/'outputs'
        for path in [runner.CACHE_DIR,runner.DEBUG_DIR,runner.OUTPUT_DIR]:path.mkdir(parents=True,exist_ok=True)
    results={};errors={};draft=None
    try:
        paths(direct,'direct');draft=direct.run_direct(direct_task,idx);results['direct']=draft['plan']
    except Exception as exc:
        errors['direct']={'type':type(exc).__name__,'message':str(exc)}
    if draft is not None:
        direct_rows=[{'plan':[]} for _ in records];direct_rows[idx-1]={'plan':draft['plan']}
        seeds=seeded.load_seed_candidates(idx,task,direct_rows,program)
        if 'selector' in config['methods']:results['selector']=seeded.rank_candidates(seeds)[0]['plan']
        if 'deterministic' in config['methods']:
            candidates=list(seeds)
            for seed in seeds:
                repaired=seeded.deterministic_repair(task,seed)
                if repaired['improved']:
                    candidates.append(seeded.build_candidate(seed['candidate_id']+'_repaired',seed['seed_source'],'revised',1,task,repaired['plan'],seeded.render_plan_text(repaired['plan'])))
            results['deterministic']=seeded.rank_candidates(candidates)[0]['plan']
        for method in config['methods']:
            try:
                if method=='self_refine':
                    paths(direct,method)
                    original=direct.load_constraint_direct_draft
                    try:
                        direct.load_constraint_direct_draft=lambda _:draft
                        results[method]=direct.run_self_refine(direct_task,idx)['plan']
                    finally:direct.load_constraint_direct_draft=original
                elif method=='cc_mar':
                    paths(cc,method);results[method]=cc.run_one(record,idx,direct_rows,program)[0]
                elif method=='seeded':
                    paths(seeded,method);seeded.RESUME=False
                    results[method]=seeded.run_one(record,idx,direct_rows,program)[1]
            except Exception as exc:errors[method]={'type':type(exc).__name__,'message':str(exc)}
    scores={}
    for method in config['methods']:
        scores[method]=evaluate_rows([{'idx':idx,'plan':results.get(method,[])}],records)['cases'][0]
    request_reports={}
    for method in config['methods']:
        log=root/method/'requests.jsonl'
        if log.exists():request_reports[method]=summarize(log)
    result={'idx':idx,'repeat':repeat,'model':model,'status':'failed' if errors else 'completed','errors':errors,
        'plans':results,'scores':scores,'requests':request_reports,'shared_direct_seed_hash':digest(draft['plan']) if draft else None,
        'program_seed_hash':digest(program[idx-1]['plan'])}
    atomic_write_text(checkpoint,json.dumps(result,indent=2)+'\n')
    return result



def run_case(job):
    try:
        return _run_case(job)
    except Exception as exc:
        config, model, repeat, idx, root_text = job
        import pandas as pd
        from evaluation.protocol import evaluate_rows
        records = pd.read_csv(DATABASE / 'validation.csv').to_dict('records')
        empty = evaluate_rows([{'idx':idx,'plan':[]}], records, evaluators=(None,None))['cases'][0]
        result = {'idx':idx,'repeat':repeat,'model':model,'status':'failed',
            'errors':{'case':{'type':type(exc).__name__,'message':str(exc)}},'plans':{},
            'scores':{method:empty for method in config['methods']},'requests':{}}
        atomic_write_text(Path(root_text)/f'r{repeat}'/f'case_{idx}'/'case.json',json.dumps(result,indent=2)+'\n')
        return result


def validate_config(config):
    supported={'direct','selector','deterministic','self_refine','cc_mar','seeded'}
    if config.get('set_type')!='validation':raise ValueError('This paired protocol currently requires labeled validation records')
    if config.get('repeats',0)<3:raise ValueError('Replication requires at least three runs')
    if not config.get('methods') or set(config['methods'])-supported or 'direct' not in config['methods']:raise ValueError('Unsupported method matrix')
    if config.get('workers',0)<1 or config.get('max_requests_per_method',0)<1:raise ValueError('Positive worker and request limits required')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/replication.json')
    parser.add_argument('--model',default=os.getenv('MODEL_NAME'))
    parser.add_argument('--limit',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();config=validate_config(json.loads(args.config.read_text()))
    import pandas as pd
    records=pd.read_csv(DATABASE/'validation.csv');ids=list(range(1,len(records)+1))
    if args.limit is not None:
        if args.limit<1:raise ValueError('limit must be positive')
        ids=ids[:args.limit]
    if not args.model:raise ValueError('Specify --model or MODEL_NAME')
    config={**config,'model':args.model,'ids':ids,'endpoint':public_endpoint(os.getenv('OPENAI_API_BASE','https://api.deepseek.com')),
        'data_hash':file_digest(DATABASE/'validation.csv'),'program_seed_hash':file_digest(FROZEN/'predictions/program_v23_best.jsonl'),
        'code':{str(p.relative_to(ROOT)):file_digest(p) for p in [Path(__file__).resolve(),ROOT/'strong_baseline_runner.py',ROOT/'seeded_multi_agent_planner.py',ROOT/'contract_multi_agent_repair.py',*sorted((ROOT/'utils').glob('*.py')),*sorted((ROOT/'evaluation').glob('*.py'))]}}
    root=ROOT/'runs/replication'/digest(config)[:20]
    if args.dry_run:
        print(json.dumps({'model':args.model,'cases_per_run':len(ids),'repeats':config['repeats'],'methods':config['methods'],'run_directory':str(root),'config_fingerprint':digest(config)},indent=2));return
    ensure_manifest(root,config)
    # Fail before dispatching any batch if credentials/model access is invalid.
    from utils.llm_client import complete
    complete('Return JSON.','Return {"ok":true}.',cache_dir=root/'preflight/cache',model=args.model,max_tokens=64,json_mode=True)
    jobs=[(config,args.model,repeat,idx,str(root)) for repeat in range(1,config['repeats']+1) for idx in ids]
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        for result in pool.map(run_case,jobs):print(f"r{result['repeat']} idx={result['idx']} {result['status']}",flush=True)
    from evaluation.protocol import summarize_cases
    from scripts.analyze_evidence import paired_stats
    from statistics import mean,stdev
    report={'config':config,'runs':{},'run_level':{},'status':'completed'}
    for repeat in range(1,config['repeats']+1):
        cases=[json.loads((root/f'r{repeat}'/f'case_{idx}'/'case.json').read_text()) for idx in ids]
        if any(c['status']!='completed' for c in cases):report['status']='incomplete'
        report['runs'][str(repeat)]={method:summarize_cases([c['scores'][method] for c in cases]) for method in config['methods']}
        report['runs'][str(repeat)]['paired_direct_to_deterministic']=paired_stats([c['scores']['direct']['final_pass'] for c in cases],[c['scores']['deterministic']['final_pass'] for c in cases]) if 'deterministic' in config['methods'] else None
    for method in config['methods']:
        rates=[r[method]['final_pass_rate'] for r in report['runs'].values()]
        report['run_level'][method]={'mean_final_pass_rate':mean(rates),'sample_standard_deviation':stdev(rates),'runs':len(rates)}
    atomic_write_text(root/'summary.json',json.dumps(report,indent=2)+'\n')
    print(root/'summary.json')
    if report['status']!='completed':raise SystemExit('Some cases failed; inspect checkpoints and resume')


if __name__=='__main__':main()
