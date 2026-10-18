"""Qualified large source-K4 projections on the existing unrolled decode kernel.

No weights change. SK changes for K=6144/8192, so these shapes have FP32
rounding differences; the local relative-L2 gate is 1e-5 versus the old kernel.
"""
import math
import torch
from . import kda_source as ks
# (N,K): (split-K, 16-byte prefetch distance selector). Small shared matrices stay on the old path.
SHAPES={(4096,4096):(4,3),(6144,4096):(4,3),(4096,8192):(8,2),(4096,6144):(6,3)}
_buffers={}

def eligible(x,q):return len(x)<=32 and (q.n,q.k) in SHAPES

def scratch(m,n,k,device):
    key=(m,n,k,str(device));sk,_=SHAPES[(n,k)]
    if key not in _buffers:
        assert not torch.cuda.is_current_stream_capturing(),'K4 fast shape must be warmed'
        count=math.ceil(m/16)
        _buffers[key]=(torch.empty((m,k),dtype=torch.float16,device=device),
            torch.empty((sk*m*n,),dtype=torch.float32,device=device),
            torch.tensor([[0,j*16,min(16,m-j*16)] for j in range(count)],dtype=torch.int32,device=device),
            torch.tensor([count],dtype=torch.int32,device=device),
            torch.arange(m,dtype=torch.int32,device=device),torch.zeros(1,dtype=torch.int32,device=device))
    return _buffers[key]

def matmul(x,q,*,out=None,f32=False):
    assert eligible(x,q) and x.shape[1]==q.k and x.dtype==torch.bfloat16
    m=len(x);sk,ld=SHAPES[(q.n,q.k)]
    if out is None:out=torch.empty((m,q.n),device=x.device,dtype=torch.float32 if f32 else torch.bfloat16)
    assert out.is_contiguous() and out.shape==(m,q.n)
    xh,z,items,counts,members,done=scratch(m,q.n,q.k,x.device);ext=ks._ext()
    ext.rot_rows(x,x.stride(0),q.suh,xh,m,q.k)
    ext.dec(xh,xh,q.trellis,q.trellis,items,counts,members,z,1,q.k,q.n,m,sk,len(items),1,0,False,
            q.svh,q.svh,q.svh,out,0.,done,ld)
    ks._post[(m,q.n//128)](z,q.svh,out,m,q.n,sk,f32,num_warps=4)
    return out
