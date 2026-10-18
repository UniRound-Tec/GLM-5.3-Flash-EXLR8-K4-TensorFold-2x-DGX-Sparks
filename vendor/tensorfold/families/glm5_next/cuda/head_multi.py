"""Small row batch, one compressed head tile shared across rows; no new quantization."""
import torch,triton
import triton.language as tl

@triton.jit
def gemv(X,W,S,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,XS:tl.constexpr,
         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,F32:tl.constexpr):
    r=tl.arange(0,BM)
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    scale=tl.load(S+n,n<N,other=0)
    acc=tl.zeros((BM,BN,BK),tl.float32)
    for offset in range(K//BK):
        ix=offset*BK+k
        x=tl.load(X+r[:,None]*XS+ix[None,:],r[:,None]<M,other=0).to(tl.float32)
        code=tl.load(W+n[:,None].to(tl.int64)*K+ix[None,:],n[:,None]<N,other=0)
        v=code.to(tl.float8e4nv,bitcast=True).to(tl.float32)
        w=(v*scale[:,None]).to(tl.bfloat16).to(tl.float32)
        acc=acc+w[None,:,:]*x[:,None,:]
    y=tl.sum(acc,axis=2)
    if not F32:y=y.to(tl.bfloat16)
    tl.store(Y+r[:,None]*N+n[None,:],y,(r[:,None]<M)&(n[None,:]<N))

def matmul(x,q,out=None,f32=False,config=(4,256,4)):
    m,k=x.shape
    assert 2<=m<=8 and k==q.k and q.block==k and q.scale.dtype==torch.float32
    if out is None:out=torch.empty((m,q.n),device=x.device,dtype=torch.float32 if f32 else torch.bfloat16)
    bn,bk,nw=config
    gemv[(triton.cdiv(q.n,bn),)](x,q.weight.view(torch.uint8),q.scale,out,M=m,N=q.n,K=k,XS=x.stride(0),BM=triton.next_power_of_2(m),BN=bn,BK=bk,F32=f32,num_warps=nw)
    return out
