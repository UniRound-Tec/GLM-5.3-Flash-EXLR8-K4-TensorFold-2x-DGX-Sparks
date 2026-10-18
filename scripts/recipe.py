"""Explicit local/LAN deployment stages. Never stop an unrelated container."""
import argparse,hashlib,json,os,pathlib,secrets,shlex,shutil,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
OWNER='glm53.exlr8.recipe'
DRAFTS=('7d74cdd881ed7e32c31175984a67823127b66cfe','dc77ff1c99eeb2df044ee3d4f0094eb033fee410','bf582e4eacc1810f76656d1811693ff6c6737d2a')
def run(argv,peer=None,capture=False,check=True,input=None):
    if peer:argv=['ssh','-o','BatchMode=yes',peer,shlex.join(list(map(str,argv)))]
    return subprocess.run(list(map(str,argv)),text=True,input=input,capture_output=capture,check=check)
def py(code,args=(),peer=None):return run(['python3','-c',code,*args],peer,True).stdout
def config(path):
    c=json.loads(path.read_text())
    if not c['worker'] or c['worker'].startswith('-'):raise ValueError('Invalid SSH peer')
    for key in ('image','container'):
        if not c[key] or c[key].startswith('-'):raise ValueError('Invalid '+key)
    for key in ('api_port','backend_port','master_port'):
        if type(c[key]) is not int or not 1<=c[key]<=65535:raise ValueError('Invalid '+key)
    if len({c[k] for k in ('api_port','backend_port','master_port')})!=3:raise ValueError('Ports must differ')
    return c
def paths(c,rank):
    side=c['peer' if rank else 'head'];peer=c['worker'] if rank else None
    if not side['model_dir'] or not side['draft_dir']:raise ValueError('Set model_dir and draft_dir on both nodes')
    return json.loads(py('import json,pathlib,sys;d=json.loads(sys.argv[1]);print(json.dumps({k:str(pathlib.Path(v).expanduser().absolute()) if v else "" for k,v in d.items()}))',[json.dumps(side)],peer))
def fp(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def inspect(c,rank):
    peer=c['worker'] if rank else None
    fmt='{"labels":{{json .Config.Labels}},"image":{{json .Image}},"running":{{.State.Running}},"oom":{{.State.OOMKilled}},"restarts":{{.RestartCount}}}'
    r=run(['docker','inspect','--format',fmt,c['container']],peer,True,False)
    if r.returncode:
        run(['docker','info','--format','{{.OSType}}'],peer,True);return None
    d=json.loads(r.stdout)
    if (d['labels'] or {}).get(OWNER)!='1':raise RuntimeError('Name belongs to another deployment; use a new container name')
    return d
def prepare(c):
    for rank in (0,1):
        state=inspect(c,rank)
        if state and state['running']:raise RuntimeError('Stop this recipe before rewriting its runtime configuration')
    for rank in (0,1):
        peer=c['worker'] if rank else None;p=paths(c,rank);data=p['data_dir']
        if c['vision'] and rank==0 and not p['vision_dir']:raise ValueError('Set head.vision_dir for the existing BF16 vision source')
        py('import pathlib,sys; p=pathlib.Path(sys.argv[1]);assert p!=pathlib.Path("/");[(p/x).mkdir(parents=True,exist_ok=True) for x in ("scripts","cache","spec-data","source-speed/decode30","private")]',[data],peer)
        for src,dest in [(ROOT/'scripts/prepare_view.py','scripts/prepare_view.py'),(ROOT/'config/runtime.env','runtime.env'),(ROOT/'config/defaults.json','source-speed/decode30/defaults.json'),(ROOT/'config/policy.txt','source-speed/empirical-policy-default.txt')]:
            if peer:run(['scp',str(src),peer+':'+shlex.quote(data+'/'+dest)])
            else:shutil.copy2(src,data+'/'+dest)
        cmd=['python3',data+'/scripts/prepare_view.py','--source',p['model_dir'],'--output',data+'/view']
        if c['vision'] and rank==0:cmd+=['--vision',p['vision_dir']]
        report=json.loads(run(cmd,peer,True).stdout)
        py('import pathlib,sys; p=pathlib.Path(sys.argv[1]);assert p.name==sys.argv[2];assert (p/"config.json").is_file() and (p/"model.safetensors").is_file(),"The pinned day-31 draft is required"',[p['draft_dir'],DRAFTS[-1]],peer)
        if rank==0:
            key=pathlib.Path(data)/'private/api-key';key.parent.chmod(0o700)
            if not key.exists():
                text=pathlib.Path(c['api_key_file']).expanduser().read_text().strip() if c['api_key_file'] else secrets.token_urlsafe(32)
                if not text:raise ValueError('Empty key file')
                with os.fdopen(os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:f.write(text+'\n')
            key.chmod(0o600)
        print(json.dumps({'rank':rank,'prepared':True,'logical_tensors':report['logical_tensors'],'vision_tensors':report['vision_tensors']}))
def mount_root(path):
    p=pathlib.PurePosixPath(path)
    for a in (p,*p.parents):
        if a.name.startswith('models--') and a.parent.name=='hub':return str(a)
    return str(p.parent.parent if p.parent.name=='snapshots' else p.parent)
def command(c,rank,p,roots):
    data=p['data_dir'];side='worker' if rank else 'head'
    cmd=['docker','run','-d','--name',c['container'],'--label',OWNER+'=1','--label',OWNER+'.config='+fp(c),'--restart=no','--gpus','all','--network','host','--ipc','host','--device','/dev/infiniband','--cap-add','IPC_LOCK','--ulimit','memlock=-1:-1','--ulimit','stack=67108864:67108864','--shm-size','16g','--env-file',data+'/runtime.env']
    for root in sorted(set(roots+[mount_root(p['draft_dir'])])):
        if root=='/':raise ValueError('Refusing root filesystem model mount')
        cmd+=['--mount',f'type=bind,src={root},dst={root},readonly']
    for src,dst,ro in [(data+'/view','/exlr8-checkpoint',True),(data+'/cache','/cache',False),(data+'/spec-data','/spec-data',False),(data+'/source-speed','/source-speed',False)]:
        cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    if rank==0:cmd+=['--mount',f'type=bind,src={data}/private/api-key,dst=/run/secrets/api-key,readonly']
    env={'RANK':rank,'MASTER_ADDR':c['head_rdma_ip'],'MASTER_PORT':c['master_port'],'MODEL_SOURCE':p['model_dir'],'DRAFT_SOURCE':p['draft_dir'],'MODEL_NAME':c['model_name'],'API_HOST':c['api_host'],'API_PORT':c['api_port'],'BACKEND_PORT':c['backend_port'],'CONTEXT':c['context'],'PARALLEL':c['parallel'],'MAX_TOKENS':c['max_tokens'],'VISION':int(c['vision']),'NCCL_SOCKET_IFNAME':c[side+'_interface'],'NCCL_IB_HCA':c[side+'_hca'],'NCCL_IB_GID_INDEX':c['gid_index']}
    for k,v in env.items():cmd+=['-e',f'{k}={v}']
    return cmd+[c['image']]
def start(c):
    prepared=[];image_ids=[]
    for rank in (0,1):
        peer=c['worker'] if rank else None;p=paths(c,rank);s=inspect(c,rank)
        image=run(['docker','image','inspect','--format','{{.Id}}',c['image']],peer,True).stdout.strip();image_ids.append(image)
        if s and (s['running'] or s['labels'].get(OWNER+'.config')!=fp(c) or s['image']!=image):raise RuntimeError('Container is running or has different config/image; nothing was stopped')
        ports=[c['master_port']] if rank else [c['api_port'],c['backend_port'],c['master_port']]
        py('import os,platform,socket,sys;assert platform.machine()=="aarch64";assert os.path.isdir("/dev/infiniband");assert os.path.exists("/sys/class/net/"+sys.argv[1]);[(lambda s,p:(s.bind(("0.0.0.0",p)),s.close()))(socket.socket(),int(p)) for p in sys.argv[2:]]',[c['worker_interface' if rank else 'head_interface'],*map(str,ports)],peer)
        roots=json.loads(py('import pathlib,sys;print((pathlib.Path(sys.argv[1])/"view/recipe-mounts.json").read_text())',[p['data_dir']],peer))['mount_roots']
        py('import pathlib,sys;p=pathlib.Path(sys.argv[1]);assert all((p/x).is_file() for x in ("runtime.env","source-speed/decode30/defaults.json","source-speed/empirical-policy-default.txt"));assert sys.argv[2]!="0" or bool((p/"private/api-key").read_text().strip())',[p['data_dir'],str(rank)],peer)
        prepared.append((rank,peer,p,s,roots))
    if len(set(image_ids))!=1:raise RuntimeError('Image IDs differ; use sync-image')
    for rank,peer,p,s,roots in reversed(prepared):
        run(['docker','start',c['container']] if s else command(c,rank,p,roots),peer)
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        for rank in (0,1):
            s=inspect(c,rank)
            if not s or not s['running'] or s['oom']:raise RuntimeError('Startup failed; retained containers for logs. No fallback was started')
        try:
            h=json.load(op.open(f'http://127.0.0.1:{c["backend_port"]}/health',timeout=5))
            if h.get('ok'):print(json.dumps({'ready':True,'pool_tokens':h.get('pool_tokens'),'api_port':c['api_port'],'model':c['model_name']}));return
        except OSError:pass
        time.sleep(5)
    raise RuntimeError('Startup observation timed out; inspect existing containers before retrying')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','build','prepare','sync-image','start','stop','status']);ap.add_argument('--config',default=str(ROOT/'config/local.json'));ap.add_argument('--base-image',default=os.environ.get('BASE_IMAGE'))
    a=ap.parse_args();path=pathlib.Path(a.config)
    if a.action=='init':
        if path.exists():raise FileExistsError('Config already exists')
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'config/example.json',path);print('Edit config/local.json, then build and prepare.');return
    c=config(path)
    if a.action=='build':
        base=a.base_image or json.loads((ROOT/'sources.lock.json').read_text())['base_image']
        run(['docker','image','inspect',base],capture=True)
        run(['docker','build','--network','none','--pull=false','--build-arg','BASE_IMAGE='+base,'--build-arg','INSTALL_DEPS=0','-t',c['image'],str(ROOT)])
    elif a.action=='prepare':prepare(c)
    elif a.action=='sync-image':
        producer=subprocess.Popen(['docker','save',c['image']],stdout=subprocess.PIPE)
        consumer=subprocess.run(['ssh','-o','BatchMode=yes',c['worker'],'docker load'],stdin=producer.stdout)
        producer.stdout.close();rc=producer.wait()
        if consumer.returncode or rc:raise RuntimeError('Image sync failed')
    elif a.action=='start':start(c)
    elif a.action=='stop':
        for rank in (0,1):
            s=inspect(c,rank)
            if s and s['running']:run(['docker','stop','--time','30',c['container']],c['worker'] if rank else None)
    else:
        for rank in (0,1):print(json.dumps({'rank':rank,'state':inspect(c,rank)}))
if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError,subprocess.CalledProcessError) as e:raise SystemExit(str(e))
