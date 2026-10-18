"""Save at most sixteen real prompt activation rows per selected projection, once."""
import json,os
from pathlib import Path
registry={};root=None
def register(w):
    global root
    path=os.environ.get('TF_QUANT_AUDIT_DIR','')
    if not path:return
    root=Path(path);root.mkdir(exist_ok=True)
    layers=[l for l in w.layers if l.kind=='kda'];selected=[layers[0],layers[len(layers)//2],layers[-1]]
    def add(q,name):registry[id(q)]=(name,q)
    for l in selected:
        for name in ('proj','fb','gb','o'):add(getattr(l.kda,name),f'L{l.index}.kda.{name}')
    layers=[l for l in w.layers if l.kind=='dsa']
    for l in (layers[0],layers[-1]):
        for name in ('proj','q_b','o'):add(getattr(l.dsa,name),f'L{l.index}.dsa.{name}')
    add(w.head,'head')
    (root/'matrix-manifest.json').write_text(json.dumps([{'name':name,'type':type(q).__module__+'.'+type(q).__name__,'n':q.n,'k':q.k} for name,q in registry.values()],indent=2))
def capture(b,x,q,f32):
    if root is None or not b.prefill or id(q) not in registry:return
    import torch
    if torch.cuda.is_current_stream_capturing():return
    name,_=registry[id(q)];p=root/(name+'.pt')
    if p.exists():return
    # Warm-up has no external marker. The first actual test request enables capture.
    if not (root/'enabled').exists():return
    count=min(16,len(x));idx=torch.linspace(0,len(x)-1,count,device=x.device).long()
    torch.save({'name':name,'x':x[idx].detach().cpu(),'row_indices':idx.cpu(),'original_rows':len(x),'f32':f32},p)
    del registry[id(q)]
