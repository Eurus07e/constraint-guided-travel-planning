"""Content-addressed run manifests and atomic checkpoints."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from utils.paths import ROOT, DATABASE


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def public_endpoint(endpoint):
    parts=urlsplit(endpoint)
    return urlunsplit((parts.scheme,parts.netloc.rsplit('@',1)[-1],parts.path,'',''))


def atomic_write_text(path, text, encoding='utf-8'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding=encoding) as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return len(text)


def ensure_manifest(root, config):
    root=Path(root);fingerprint=digest(config)
    path=root/'manifest.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get('fingerprint')!=fingerprint:
            raise ValueError('Run manifest mismatch; select a new run directory')
    else:
        atomic_write_text(path,json.dumps({'schema_version':1,'fingerprint':fingerprint,'config':config},indent=2)+'\n')
    return fingerprint


def configure_run(namespace):
    settings={k:v for k,v in namespace.items() if k.isupper() and isinstance(v,(int,float,str,bool)) and 'KEY' not in k and k not in {'MODE'}}
    settings.update({'audit_version':os.getenv('TP_AUDIT_VERSION','strict-v2'),
        'endpoint':public_endpoint(os.getenv('OPENAI_API_BASE','https://api.deepseek.com')),
        'run_id':os.getenv('TP_RUN_ID','default'),'ids':os.getenv('IDS'),'limit':os.getenv('LIMIT'),
        'exclude_ids':os.getenv('EXCLUDE_IDS'),'max_requests':os.getenv('TP_MAX_REQUESTS'),
        'set_type':namespace.get('SET_TYPE','validation')})
    sources=[Path(namespace['__file__']).resolve(),*sorted((ROOT/'utils').glob('*.py'))]
    if namespace.get('__name__')=='__main__':
        sources.extend(ROOT/name for name in ['seeded_multi_agent_planner.py','contract_multi_agent_repair.py','strong_baseline_runner.py'])
    config={'settings':settings,'code':{p.relative_to(ROOT).as_posix():file_digest(p) for p in sources},'inputs':{}}
    for p in sorted(DATABASE.rglob('*')):
        if p.is_file() and p.suffix in {'.csv','.txt'}:
            config['inputs']['database/'+p.relative_to(DATABASE).as_posix()]=file_digest(p)
    for name in ['DIRECT_SUBMISSION_FILE','PROGRAM_SUBMISSION_FILE']:
        p=namespace.get(name)
        if p and Path(p).is_file():config['inputs'][name]=file_digest(p)
    fingerprint=digest(config)
    base=Path(os.getenv('OUT_ROOT',str(ROOT/'runs'/namespace['STRATEGY'])))
    run_root=base/fingerprint[:20]
    ensure_manifest(run_root,config)
    split=settings['set_type']
    namespace.update({'OUT_ROOT':run_root,'OUTPUT_DIR':run_root/split,'DEBUG_DIR':run_root/'debug',
        'CACHE_DIR':run_root/'llm_cache','RUN_FINGERPRINT':fingerprint})
    output=Path(os.getenv('SUBMISSION_FILE',str(run_root/'submission.jsonl')))
    owner=output.with_suffix(output.suffix+'.manifest.json')
    if output.exists() and (not owner.exists() or json.loads(owner.read_text()).get('fingerprint')!=fingerprint):
        raise ValueError(f'Output belongs to another run or has no manifest: {output}')
    atomic_write_text(owner,json.dumps({'fingerprint':fingerprint})+'\n')
    namespace['SUBMISSION_FILE']=output
    if 'SUBMISSION_DIR' in namespace:namespace['SUBMISSION_DIR']=output.parent
    print(f'run directory: {run_root}',flush=True)
    return config
