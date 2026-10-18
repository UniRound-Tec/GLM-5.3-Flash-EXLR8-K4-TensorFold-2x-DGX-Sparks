import torch,triton
import triton.language as tl

@triton.jit
def gemv(X,W,S,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,F32:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    scale=tl.load(S+n,n<N,other=0)
    acc=tl.zeros((BN,BK),tl.float32)
    for offset in range(K//BK):
        ix=offset*BK+k
        x=tl.load(X+ix).to(tl.float32)
        code=tl.load(W+n[:,None].to(tl.int64)*K+ix[None,:],n[:,None]<N,other=0)
        v=code.to(tl.float8e4nv,bitcast=True).to(tl.float32)
        w=(v*scale[:,None]).to(tl.bfloat16).to(tl.float32)
        acc=acc+w*x[None,:]
    y=tl.sum(acc,axis=1)
    if not F32:y=y.to(tl.bfloat16)
    tl.store(Y+n,y,n<N)

def matmul(x,q,out=None,f32=False,config=(8,256,4)):
    assert x.shape==(1,q.k) and q.scale.dtype==torch.float32 and q.block==q.k
    if out is None:out=torch.empty((1,q.n),device=x.device,dtype=torch.float32 if f32 else torch.bfloat16)
    bn,bk,nw=config
    gemv[(triton.cdiv(q.n,bn),)](x,q.weight.view(torch.uint8),q.scale,out,N=q.n,K=q.k,BN=bn,BK=bk,F32=f32,num_warps=nw)
    return out
