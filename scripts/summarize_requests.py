"""Summarize actual network attempts separately from cache reuse."""
import argparse,json
from pathlib import Path


def summarize(path):
    events=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    usage={};latency=[]
    for event in events:
        if event['event']=='success':
            latency.append(event.get('latency_seconds',0))
            for key,value in (event.get('usage') or {}).items():
                if isinstance(value,(int,float)):usage[key]=usage.get(key,0)+value
    return {'attempts':sum(e['event']=='attempt' for e in events),'successes':sum(e['event']=='success' for e in events),
        'failures':sum(e['event']=='failure' for e in events),'cache_hits':sum(e['event']=='cache_hit' for e in events),
        'usage':usage,'successful_request_seconds':sum(latency),'mean_success_latency_seconds':sum(latency)/len(latency) if latency else None}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);args=p.parse_args();print(json.dumps(summarize(args.log),indent=2))
