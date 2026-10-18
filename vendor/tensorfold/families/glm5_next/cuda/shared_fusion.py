"""Original-K4 shared expert joins the routed batch for 2..8-row verification.

The old routed view and separate shared view alias one combined allocation.
Prefill and original graphs retain their own paths and arithmetic.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import torch
from . import exl3_mm
from tensorfold.cuda import experts as grouped

BANKS={}
ledger=[]

def eligible(shared):
    from .zip_k4 import ZipK4
    from .kda_source import K4Output
    if shared is None or not isinstance(shared.gu,ZipK4) or not isinstance(shared.down,ZipK4):return False
    qs=shared.gu.source
    return (isinstance(qs,list) and len(qs)==2 and all(isinstance(q,K4Output) and (q.n,q.k)==(1024,4096) for q in qs)
            and isinstance(shared.down.source,K4Output) and (shared.down.source.n,shared.down.source.k)==(4096,1024))

def source(shared,proj):
    assert eligible(shared)
    return {'gate_proj':shared.gu.source[0],'up_proj':shared.gu.source[1],'down_proj':shared.down.source}[proj]

def append(ts,us,vs,shared,proj):
    q=source(shared,proj)
    ts.append(q.trellis[0]);us.append(q.suh);vs.append(q.svh)

def finish(parts,shared,count):
    assert eligible(shared)
    for proj,(t,u,v) in parts.items():
        q=source(shared,proj)
        assert len(t)==len(u)==len(v)==count+1
        assert torch.equal(q.trellis,t[-1:]) and torch.equal(q.suh,u[-1]) and torch.equal(q.svh,v[-1])
        # Replace the old standalone allocations before any graph/prefetch table is built.
        q.trellis=t[-1:];q.suh=u[-1];q.svh=v[-1]
    (gt,sg,vg),(ut,su,vu),(dt,sd,vd)=(parts[p] for p in ('gate_proj','up_proj','down_proj'))
    ex=exl3_mm.Exl3Experts(gt[:-1],ut[:-1],dt[:-1],sg[:-1],su[:-1],vg[:-1],vu[:-1],sd[:-1],vd[:-1],count,vg.shape[1],vd.shape[1])
    BANKS[id(ex)]=SimpleNamespace(gt=gt,ut=ut,dt=dt,sg=sg,su=su,vg=vg,vu=vu,sd=sd,vd=vd)
    ledger.append({'experts':count,'shared_aliases':True,'extra_weight_bytes':0})
    return ex

@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda.build import load
    p=Path(__file__).parent
    return load(name='glm_shared_merged_v1',sources=[str(p/'merged.cpp'),str(p/'merged.cu')],extra_cuda_cflags=['-O3'],verbose=False)

def enabled(m,b,R,done):
    from . import trial_control
    return bool(trial_control.ACTIVE.get('shared_fusion',False) and 2<=R<=8 and not b.prefill and done is None and id(m.experts) in BANKS)

def run(x,pick,plan,ex,s,y,R,limit):
    assert 2<=R<=8 and plan.tile==16 and s.slots==9 and ex.dims==4096 and ex.width==1024
    q=BANKS[id(ex)];mod=extension();P=s.rows*s.slots;items=grouped.max_items(R*s.slots,plan.experts)
    mod.rot_in(x,x.stride(0),pick,q.sg,q.su,s.xg,s.xu,R,4096,9)
    mod.dec(s.xg,s.xu,q.gt,q.ut,plan.items,plan.counts,plan.members,s.z,2,4096,1024,P,4,items,9,0,False,q.vg,q.vu,q.sd,s.xd,float(limit),s.done,1)
    mod.gateup_epilogue(s.z,pick,q.vg,q.vu,q.sd,s.xd,R,P,1024,4,9,float(limit))
    mod.dec(s.xd,s.xd,q.dt,q.dt,plan.items,plan.counts,plan.members,s.z,1,1024,4096,P,1,items,9,1,False,q.vd,q.vd,q.vd,y,0.,s.done,1)
