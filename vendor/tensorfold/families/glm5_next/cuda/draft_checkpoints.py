"""Pinned DFlash2 weight banks. Target weights, verification and sampling are untouched."""
import gc,json,os,time
from pathlib import Path

REVISIONS={27:'7d74cdd881ed7e32c31175984a67823127b66cfe',28:'dc77ff1c99eeb2df044ee3d4f0094eb033fee410',31:'bf582e4eacc1810f76656d1811693ff6c6737d2a'}
FIELDS=('fc','layers','hidden_norm','norm','hproj','pred','succ')

def snapshot(d):return {k:getattr(d,k) for k in FIELDS}
def apply(d,bank):
    assert set(bank)==set(FIELDS)
    for k in FIELDS:setattr(d,k,bank[k])

def preload(d,current_path):
    """Load before target KV allocation; only one MSE representation per historical revision."""
    from . import dflash2,qmm
    import torch
    parent=Path(current_path).parent
    assert Path(current_path).name==REVISIONS[31]
    cfg=json.loads((Path(current_path)/'config.json').read_text())
    d.checkpoint_weights={}
    original=dflash2._quantize4
    before=torch.cuda.memory_allocated();t0=time.monotonic()
    try:
        # Per-row MSE calculation is unchanged; small chunks bound temporary allocations.
        dflash2._quantize4=lambda w:qmm.quantize4_mse(w,chunk=256)
        for day in (27,28):
            path=parent/REVISIONS[day]
            if not (path/'model.safetensors').is_file():
                continue  # Retired historical checkpoint; current day 31 remains required.
            assert json.loads((path/'config.json').read_text())==cfg,'Incompatible draft configuration'
            extra=dflash2.Drafter(path,d.w,block=d.block,capacity=d.capacity,tap_rows=d.tap_rows,ring=bool(d.ring))
            for k in ('D','hd','heads','kvh','inter','block','top_k','tap_layers','window','ring','cap','causal','eps'):
                assert getattr(extra,k)==getattr(d,k),(day,k)
            d.checkpoint_weights[day]=snapshot(extra)
            del extra;gc.collect();torch.cuda.empty_cache()
            # Release clean source file pages, retaining the actual CPU selector arrays.
            with (path/'model.safetensors').open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            print('TF_DRAFT_CHECKPOINT_LOADED '+json.dumps({'rank':d.rank,'day':day,'revision':REVISIONS[day]}),flush=True)
    finally:dflash2._quantize4=original
    torch.cuda.synchronize()
    print('TF_DRAFT_CHECKPOINT_MEMORY '+json.dumps({'rank':d.rank,'gpu_extra_bytes':torch.cuda.memory_allocated()-before,'cpu_codebooks_extra_bytes':sum(v[k].nbytes for v in d.checkpoint_weights.values() for k in ('pred','succ')),'load_s':time.monotonic()-t0}),flush=True)

def prepare(m):
    import torch
    d,md=m.drafts.d,m.drafts
    base=snapshot(d);m.checkpoint_banks={};m.checkpoint_graphs={}
    for quant in ('minmax','mse'):
        bank=dict(base);bank['fc'],bank['layers'],_=m.trial_weights[quant]
        m.checkpoint_banks[31,quant]=bank
        m.checkpoint_graphs[31,quant]=m.trial_drafts[quant]
    for day,bank in d.checkpoint_weights.items():
        apply(d,bank);m.w.draft_head=m.trial_weights['mse'][2]
        md.block_graphs,md.tap_graphs={},{}
        md.capture()
        m.checkpoint_banks[day,'mse']=bank
        m.checkpoint_graphs[day,'mse']=(md.block_graphs,md.tap_graphs,md.pool)
    activate(m,31,'minmax');torch.cuda.synchronize()
    print('TF_DRAFT_CHECKPOINT_READY '+json.dumps({'rank':d.rank,'banks':[str(k) for k in m.checkpoint_banks],'graphs':{str(k):[len(v[0]),len(v[1])] for k,v in m.checkpoint_graphs.items()}}),flush=True)

def activate(m,day,quant):
    assert not m.lanes,'Checkpoint switch requires an empty scheduler'
    d,md=m.drafts.d,m.drafts
    apply(d,m.checkpoint_banks[day,quant])
    m.w.draft_head=m.trial_weights[quant][2]
    md.block_graphs,md.tap_graphs,md.pool=m.checkpoint_graphs[day,quant]
