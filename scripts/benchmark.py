"""Explicit single-request benchmark; never called by deployment commands."""
import argparse,hashlib,json,time,urllib.request,uuid
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['code','prose','algorithm','prefill8k','prefill32k','prefill197k'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--key-file',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:8888');p.add_argument('--policy',default='fnc7:0.3');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve previous measurements; choose a new --out')
    prompts=json.loads((Path(__file__).resolve().parents[1]/'evidence/prompts.json').read_text())
    is_prefill=a.case.startswith('prefill');tokens=32 if is_prefill else 512
    if is_prefill:
        size={'prefill8k':8192,'prefill32k':32768,'prefill197k':196608}[a.case]
        filler='The quick brown fox jumps over the lazy dog. This is a sentence about systems, memory, and computation. '
        prompt=f'recipe-cold-{uuid.uuid4().hex}-{size}: '+filler*(size//22)+'\n'+prompts['zh_prose']
    else:prompt=prompts[{'code':'mia_code','prose':'mia_prose','algorithm':'logic_code'}[a.case]]
    key=a.key_file.expanduser().read_text().strip();op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def get(path):return json.load(op.open(urllib.request.Request(a.url+path,headers={'Authorization':'Bearer '+key}),timeout=10))
    before=get('/health')
    if not before.get('ok') or before.get('busy') or before.get('requests_running'):raise RuntimeError('Service must be healthy and idle')
    model=get('/v1/models')['data'][0]['id']
    body={'model':model,'messages':[{'role':'user','content':prompt}],'seed':a.seed,'temperature':1,'top_p':.95,'top_k':20,'min_p':0,'max_tokens':tokens,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True},'tf_policy':a.policy,'chat_template_kwargs':{'enable_thinking':False}}
    req=urllib.request.Request(a.url+'/v1/chat/completions',json.dumps(body).encode(),{'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    first=None;stats={};usage={};text=[];start=time.perf_counter()
    with op.open(req,timeout=1800) as resp:
        for line in resp:
            if not line.startswith(b'data:'):continue
            data=line[5:].strip()
            if data==b'[DONE]':break
            e=json.loads(data)
            if e.get('error'):raise RuntimeError(e['error'])
            stats=e.get('tensorfold') or stats;usage=e.get('usage') or usage
            for c in e.get('choices',[]):
                d=c.get('delta',{});part=d.get('content') or d.get('reasoning_content')
                if part:
                    if first is None:first=time.perf_counter()
                    text.append(part)
    wall=time.perf_counter()-start;after=get('/health')
    cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',stats.get('cached',0))
    isolated=after.get('requests_total')==before.get('requests_total',0)+1 and not after.get('busy')
    row={'case':a.case,'model':model,'seed':a.seed,'policy':a.policy,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'output':''.join(text),'usage':usage,'stats':stats,'ttft_s':None if first is None else first-start,'wall_s':wall,'isolated':isolated,'cached_tokens':cached}
    complete=usage.get('completion_tokens')==tokens
    row['valid']=isolated and complete and (not is_prefill or cached==0)
    if row['valid']:
        row['decode_tok_s']=(tokens-1)/stats['decode_s']
        if is_prefill:row['cold_prefill_tok_s']=usage['prompt_tokens']/stats['prefill_s']
    a.out.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k in ('case','valid','ttft_s','decode_tok_s','cold_prefill_tok_s','cached_tokens')}))
if __name__=='__main__':main()
