"""Decode prefetch tables must not include the lossless prefill-only ZipK4 backup.

All three modes reuse the graph's original table pointer and item counts. Mode 0
is the original table, 1 excludes prefill backups, 2 prefetches only a 16-byte
dummy. Switching occurs once between requests on both TP ranks, never in a round.
"""
import json
from pathlib import Path
import torch
from . import l2pf

def decode_object(q):
    from .zip_k4 import ZipK4
    return q.source if isinstance(q,ZipK4) else q

def overlap(spans,tensors):
    ranges=[(t.data_ptr(),t.data_ptr()+t.numel()*t.element_size()) for t in tensors]
    return sum(max(0,min(a+n,z)-max(a,b)) for a,n in spans for b,z in ranges)

def install(pf,rank):
    if pf is None or hasattr(pf,'speed_tables'):return
    original=pf.table.detach().cpu().clone();filtered=original.clone()
    pf.speed_dummy=torch.zeros(16,dtype=torch.uint8,device=pf.table.device)
    dummy=[pf.speed_dummy.data_ptr(),16];assert dummy[0]%16==0
    report=[];layers=list(pf.w.layers)
    from .zip_k4 import ZipK4
    for i,layer in enumerate(layers):
        nxt=layers[i+1] if i+1<len(layers) else None
        plans={'a':pf._ffn(layer),'f':pf._next(nxt),'o':pf._out(layer)}
        for name,groups in plans.items():
            key=pf.sites.get((layer.index,name))
            if key is None:continue
            first,count,grid=key
            raw=original[first:first+count].tolist()
            packed=[t for q in groups if isinstance(q,ZipK4) for t in q.packed]
            pieces=l2pf.pieces(l2pf.ranges([decode_object(q) for q in groups],int(pf.s.mb*(1<<20))),pf.s.piece)
            used=pieces[:count]
            assert all(a%16==0 and n>0 and n%16==0 for a,n in used)
            assert overlap(used,packed)==0,'Prefill-only backup remains in decode prefetch'
            dropped=sum(n for _,n in pieces[count:])
            filtered[first:first+count]=torch.tensor(used+[dummy]*(count-len(used)),dtype=torch.int64)
            report.append({'layer':layer.index,'site':name,'items':count,'useful_items':len(used),
               'original_bytes':sum(n for _,n in raw),'original_prefill_only_bytes':overlap(raw,packed),
               'filtered_bytes':sum(n for _,n in used),'dropped_tail_bytes':dropped})
    idle=torch.tensor(dummy,dtype=torch.int64).expand_as(original).contiguous()
    pf.speed_tables={0:original.to(pf.table.device),1:filtered.to(pf.table.device),2:idle.to(pf.table.device)}
    pf.speed_mode=0;pf.speed_report=report
    p=Path('/source-speed')
    if p.exists():
        (p/f'prefetch-ledger-r{rank}.json').write_text(json.dumps({'sites':report,'table_bytes':pf.table.numel()*8,
            'modes':{'0':'original','1':'decode tensors only','2':'16-byte dummy per item'}},indent=2))

def activate(pf,mode):
    if pf is None:
        if mode:raise RuntimeError('Prefetch table is unavailable')
        return
    assert mode in (0,1,2) and hasattr(pf,'speed_tables')
    if pf.speed_mode!=mode:
        pf.table.copy_(pf.speed_tables[mode]);torch.cuda.synchronize();pf.speed_mode=mode
