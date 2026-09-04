"""OpenAI-compatible requests with namespaced caches and per-attempt accounting."""
import fcntl
import json
import os
import time
from pathlib import Path
import requests
from utils.run_state import atomic_write_text, digest, public_endpoint


def chat_completions_url(base):
    base=str(base).rstrip('/')
    if base.endswith('/chat/completions'):return base
    return base+'/chat/completions' if base.endswith('/v1') else base+'/v1/chat/completions'


def event(path, payload, reserve=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+',encoding='utf-8') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        if reserve:
            maximum=int(os.getenv('TP_MAX_REQUESTS','0'))
            if maximum:
                stream.seek(0)
                used=sum(json.loads(line).get('event')=='attempt' for line in stream if line.strip())
                if used>=maximum:raise RuntimeError('TP_MAX_REQUESTS exhausted; increase the budget in a new run')
        stream.seek(0,2);stream.write(json.dumps(payload,ensure_ascii=False)+'\n');stream.flush()
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def complete(system_prompt,user_prompt,*,cache_dir,model,max_tokens=4096,temperature=0,cache_tag='default',json_mode=False):
    base=os.getenv('OPENAI_API_BASE','https://api.deepseek.com')
    url=chat_completions_url(base);cache_dir=Path(cache_dir)
    payload={'model':model,'messages':[{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}],
        'temperature':temperature,'max_tokens':max_tokens}
    if json_mode:payload['response_format']={'type':'json_object'}
    if 'dashscope.aliyuncs.com' in url:payload['enable_thinking']=os.getenv('ENABLE_THINKING','0')=='1'
    key=digest({'endpoint':url,'run_id':os.getenv('TP_RUN_ID','default'),'tag':cache_tag,'payload':payload})
    cache=cache_dir/(key+'.json');log=cache_dir.parent/'requests.jsonl'
    common={'cache_key':key,'tag':cache_tag,'model':model,'endpoint':public_endpoint(url),'time':time.time()}
    if cache.exists():
        try:
            saved=json.loads(cache.read_text())
            if not isinstance(saved.get('content'),str) or not saved['content']:raise ValueError('empty cached response')
            event(log,{**common,'event':'cache_hit'})
            return saved['content']
        except (ValueError,KeyError):pass
    api_key=os.getenv('OPENAI_API_KEY','')
    if not api_key:raise RuntimeError('OPENAI_API_KEY is not set')
    retries=int(os.getenv('MAX_RETRIES','3'));timeout=int(os.getenv('REQUEST_TIMEOUT','180'))
    for attempt in range(1,retries+1):
        event(log,{**common,'event':'attempt','attempt':attempt},reserve=True)
        started=time.monotonic();response=None
        try:
            response=requests.post(url,headers={'Authorization':'Bearer '+api_key,'Content-Type':'application/json'},json=payload,timeout=timeout)
            response.raise_for_status()
            obj=response.json();choice=obj['choices'][0];content=choice['message'].get('content') or ''
            if not isinstance(content,str) or not content.strip():raise ValueError('Provider returned no usable text content')
            metadata={**common,'event':'success','attempt':attempt,'model':obj.get('model',model),
                'request_id':obj.get('id'),'usage':obj.get('usage'),'finish_reason':choice.get('finish_reason'),
                'latency_seconds':time.monotonic()-started}
            atomic_write_text(cache,json.dumps({'content':content,'metadata':metadata},ensure_ascii=False))
            event(log,metadata)
            return content
        except (requests.RequestException,ValueError,KeyError,IndexError) as exc:
            status=response.status_code if response is not None else None
            event(log,{**common,'event':'failure','attempt':attempt,'status':status,'error_type':type(exc).__name__, 'latency_seconds':time.monotonic()-started})
            retryable=status in {408,429,500,502,503,504} or isinstance(exc,(requests.Timeout,requests.ConnectionError))
            if not retryable or attempt==retries:
                raise RuntimeError(f'Model request failed: {type(exc).__name__}, HTTP {status}; see requests.jsonl') from None
            retry_after=response.headers.get('Retry-After','') if response is not None else ''
            delay=min(60,float(retry_after)) if retry_after.isdigit() else min(30,2**attempt)
            time.sleep(delay)
    raise RuntimeError('MAX_RETRIES must be positive')
