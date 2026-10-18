"""Selective KDA source formats. QKV and output gate MXFP8; decay/beta Q4, output original EXLR8 K4.

K4 uses the deployed E4M3-codebook EXL3-compatible FP16 MMA kernel, with
original trellis/scales; no weight requantization and no resident BF16 copy.
"""
from dataclasses import dataclass
import re
import torch
import triton
import triton.language as tl
from . import qmm, kda_mxfp8 as sf
from .exl3_mm import _ext

@dataclass
class MixedInput:
    qkv: object
    gates: object
    ga: object
    n: int
    k: int
    def nbytes(self): return self.qkv.nbytes()+self.gates.nbytes()+self.ga.nbytes()

@dataclass
class K4Output:
    trellis: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    n: int
    k: int
    def nbytes(self): return sum(t.numel()*t.element_size() for t in (self.trellis,self.suh,self.svh))

def load_output(rd,name,device):
    from .split import rule,rank_cut,AXIS
    s=name.removeprefix('model.language_model.').removesuffix('.weight')
    m=re.fullmatch(r'layers\.(\d+)\.(.*)',s)
    f=rd.safe(rd.root/f'dense/exlr8-dense-{int(m[1]):03d}.safetensors');key=m[2]
    fmt,n,k,width,side,*_=f.get(key+'.record').tolist()
    assert fmt==1 and width==4 and key=='self_attn.o_proj'
    # Reorder only, preserving every packed trellis word and scale bit.
    a=f.get(key+'.trellis').view(torch.int32)
    if side==2:a=a.reshape(k//128,n//128,8,8,32).permute(0,2,1,3,4).reshape(k//16,n//16,32)
    else:a=a.reshape(n//128,k//16,8,32).permute(1,0,2,3).reshape(k//16,n//16,32)
    su=f.get(key+'.suh');sv=f.get(key+'.svh')
    assert su.dtype==sv.dtype==torch.float16
    kind=rule(name);assert AXIS[kind]==1
    lo,hi=rank_cut(rd.layout,name,[n,k],kind) or (rd.rank*(k//2),(rd.rank+1)*(k//2))
    assert lo%128==hi%128==0
    return K4Output(a[lo//16:hi//16].contiguous().unsqueeze(0).to(device),su[lo:hi].contiguous().to(device),sv.contiguous().to(device),n,hi-lo)

@triton.jit
def _join(A,B,G,Y,M:tl.constexpr,NA:tl.constexpr,NB:tl.constexpr,NG:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);n=NA+NB+NG;r=i//n;c=i%n
    # Preserve [QKV | f_a (128) | g_a (128) | beta], with g_a read from its source.
    cg=c-NA-128;isg=(cg>=0)&(cg<NG)
    cb=c-NA-tl.where(c>=NA+128+NG,NG,0)
    a=tl.load(A+r*NA+c,(r<M)&(c<NA),other=0)
    b=tl.load(B+r*NB+cb,(r<M)&(c>=NA)&(~isg),other=0)
    g=tl.load(G+r*NG+cg,(r<M)&isg,other=0)
    tl.store(Y+r*S+c,tl.where(c<NA,a,tl.where(isg,g,b)),r<M)

@triton.jit
def _post(Z,S,Y,M:tl.constexpr,N:tl.constexpr,SK:tl.constexpr,F32:tl.constexpr):
    r=tl.program_id(0);base=tl.program_id(1)*128;i=tl.arange(0,128)
    v=tl.full((128,),0,tl.float32)
    for s in tl.static_range(SK):v+=tl.load(Z+(s*M+r)*N+base+i)
    for d in tl.static_range(7):
        j=i^(1<<d);other=tl.gather(v,j,0)
        v=tl.where((i&(1<<d))==0,v+other,other-v)
    v=v*0.08838834764831845*tl.load(S+base+i).to(tl.float32)
    tl.store(Y+r*N+base+i,v if F32 else v.to(tl.bfloat16))

_scratch={}
def scratch(m,k,n,device):
    key=(m,k,n,str(device))
    if key not in _scratch:
        assert not torch.cuda.is_current_stream_capturing(),'K4 shape must be warmed before capture'
        count=(m+15)//16
        items=torch.tensor([[0,j*16,min(16,m-j*16)] for j in range(count)],dtype=torch.int32,device=device)
        buffers=(torch.empty((m,k),dtype=torch.float16,device=device),
            torch.empty((4*m*n,),dtype=torch.float32,device=device),items,
            torch.tensor([count],dtype=torch.int32,device=device),torch.arange(m,dtype=torch.int32,device=device))
        if m>64:return buffers  # Variable prompt lengths must not accumulate persistent scratch.
        _scratch[key]=buffers
    return _scratch[key]

def weight_estimate(base,raw,cfg):
    def transform(name,info):
        total,extra=base(name,info)
        m=re.search(r'layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj|g_a_proj|g_b_proj)\.weight$',name)
        if not m or int(m[1])>=cfg.layers or cfg.kinds[int(m[1])]!='kda':return total,extra
        values=raw(name,info)[0]//2
        if m[2]=='o_proj':
            n=cfg.hidden;k=values//n
            return values//2+(n+k)*2,extra
        resident=values+values//16  # Original FP8 bytes plus uint16 E8M0 scale storage.
        if int(m[1])==0 and m[2]=='q_proj':resident+=512<<20  # Shared decode/unpack and peak prompt scratch.
        return resident,extra
    return transform

def matmul(x,q,xs=None,*,out=None,f32=False,part=None,prefill=False):
    m=x.shape[0]
    if isinstance(q,sf.SourceF8):return sf.matmul(x,q,out=out,f32=f32,part=part)
    if isinstance(q,MixedInput):
        a=sf.matmul(x,q.qkv,f32=f32,part=part)
        if prefill:
            from tensorfold.cuda.kernels import qmm as shared
            b=shared.prefill_matmul(x,q.gates,f32=f32)
        else:
            from tensorfold.cuda.kernels import qmm as shared
            # Preserve the former fused matrix's split/reduction configuration.
            b=shared.matmul(x,q.gates,qmm.group_sums(x) if xs is None else xs,
                sk=qmm.split_k(q.n,q.k),f32=f32,
                variant=qmm.dec_tile(q.n,q.k) if m<=16 else None)
        if out is None:out=torch.empty((m,q.n),device=x.device,dtype=a.dtype)
        ga=sf.matmul(x,q.ga,f32=f32,part=part)
        _join[(triton.cdiv(m*q.n,1024),)](a,b,ga,out,m,q.qkv.n,q.gates.n,q.ga.n,out.stride(0),1024)
        return out
    assert isinstance(q,K4Output) and x.shape[1]==q.k and x.dtype==torch.bfloat16
    if out is None:out=torch.empty((m,q.n),device=x.device,dtype=torch.float32 if f32 else torch.bfloat16)
    assert out.is_contiguous() and out.shape==(m,q.n)
    xh,z,items,counts,members=scratch(m,q.k,q.n,x.device)
    ext=_ext();sk=4
    ext.rot_rows(x,x.stride(0),q.suh,xh,m,q.k)
    ext.grouped(xh,xh,q.trellis,q.trellis,items,counts,members,z,1,q.k,q.n,m,sk,len(items),8,4,m>=64,True,False)
    _post[(m,q.n//128)](z,q.svh,out,m,q.n,sk,f32,num_warps=4)
    return out
