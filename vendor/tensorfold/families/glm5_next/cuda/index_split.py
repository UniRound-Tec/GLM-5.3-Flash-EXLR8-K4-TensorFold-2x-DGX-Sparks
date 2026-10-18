"""Prefill query-row partition, preserving TensorFold's selection arithmetic.

The zigzag quarter split follows Enntity/atlas 72588541 (AGPL-3.0-only).
No expert, attention-head or KV partition changes. Decode stays replicated.
"""
import torch
import triton
import triton.language as tl
from . import sparse

@triton.jit
def _expand(POOL,TOK,CNT,POS):
    r=tl.program_id(0);i=tl.arange(0,4096)
    p=tl.load(POOL+r*512+i//4,mask=i<2048,other=0)
    pos=tl.load(POS)
    left=(pos+r+1)%4;start=(pos+r+1)//4*4
    out=tl.where(i<2048,p+i%4,tl.where(i-2048<left,start+i-2048,-1))
    tl.store(TOK+r*2051+i,out,mask=i<2051)
    tl.store(CNT+r,2048+left)

def quarters(rows, rank):
    q=rows//4
    own=(0,3) if rank==0 else (1,2)
    peer=(1,2) if rank==0 else (0,3)
    return [(i*q,(i+1)*q) for i in own],[(i*q,(i+1)*q) for i in peer],4*q

def select(comm,qi,wts,pk,pos,rows,np_max,pos_dev):
    if comm.world!=2 or pos is None or pos<32768 or rows<4096:
        return sparse.select_tokens(qi,wts,pk,pos,rows,np_max,pos_dev)
    own,other,tail=quarters(rows,comm.rank)
    tokens=torch.empty((rows,sparse.TOKENS),dtype=torch.int32,device=qi.device)
    counts=torch.empty(rows,dtype=torch.int32,device=qi.device)
    for lo,hi in own:
        at=pos_dev if lo==0 else pos_dev+lo
        ids,num=sparse.select_tokens(qi[lo:hi],wts[lo:hi],pk,pos+lo,hi-lo,np_max,at)
        tokens[lo:hi].copy_(ids);counts[lo:hi].copy_(num)
    sends=[];recvs=[]
    for (lo,hi),(pl,ph) in zip(own,other):
        # Four consecutive tokens per selected pool; only exchange the first.
        sends.append(tokens[lo:hi,:2048:4].contiguous())
        recvs.append(torch.empty((ph-pl,512),dtype=torch.int32,device=qi.device))
    comm.exchange(sends,recvs,1-comm.rank)
    for compact,(lo,hi) in zip(recvs,other):
        _expand[(hi-lo,)](compact,tokens[lo:hi],counts[lo:hi],pos_dev+lo,num_warps=4)
    if tail<rows:
        ids,num=sparse.select_tokens(qi[tail:],wts[tail:],pk,pos+tail,rows-tail,np_max,pos_dev+tail)
        tokens[tail:].copy_(ids);counts[tail:].copy_(num)
    return tokens,counts
