"""Two independent KDA projections and copy-only tap packing, with fixed buffers."""
from functools import lru_cache
from pathlib import Path
import torch
import triton
import triton.language as tl

@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda.build import load
    here=Path(__file__).parent
    return load(name='glm_tf_q4_pair_20261006',sources=[str(here/'q4_pair.cpp'),str(here/'q4_pair.cu')],extra_cuda_cflags=['-O3'],verbose=False)

@triton.jit
def _sums_pair(X0,X1,S0,S1,STRIDE:tl.constexpr):
    r=tl.program_id(0);pair=tl.program_id(1)
    g=tl.arange(0,16);c=tl.arange(0,64)
    x=X0 if pair==0 else X1;s=S0 if pair==0 else S1
    v=tl.load(x+r*STRIDE+g[:,None]*64+c[None,:],mask=g[:,None]<2,other=0.0)
    tl.store(s+r*2+g,tl.sum(v.to(tl.float32),1),mask=g<2)

def kda_pair(x0,x1,q0,q1,s0,s1,y0,y1):
    assert x0.shape==x1.shape and q0.k==q1.k==128 and q0.n==q1.n
    _sums_pair[(x0.shape[0],2)](x0,x1,s0,s1,STRIDE=x0.stride(0),num_warps=2)
    extension().pair(x0,x1,s0,s1,q0.weight,q0.scales,q0.biases,q1.weight,q1.scales,q1.biases,y0,y1,q0.n)

class TapParts:
    def __init__(self,parts):
        self.parts=tuple(parts);assert len(self.parts)==5
        self.shape=(self.parts[0].shape[0],sum(t.shape[1] for t in self.parts))
    def __getitem__(self,rows):return TapParts([t[rows] for t in self.parts])

@triton.jit
def _pack_taps(A,B,C,D,E,O,SA:tl.constexpr,SB:tl.constexpr,SC:tl.constexpr,SD:tl.constexpr,SE:tl.constexpr,
               WIDTH:tl.constexpr,OFF:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK);ok=j<WIDTH
    tl.store(O+(OFF+row)*5*WIDTH+j,tl.load(A+row*SA+j,mask=ok,other=0),mask=ok)
    tl.store(O+(OFF+row)*5*WIDTH+WIDTH+j,tl.load(B+row*SB+j,mask=ok,other=0),mask=ok)
    tl.store(O+(OFF+row)*5*WIDTH+2*WIDTH+j,tl.load(C+row*SC+j,mask=ok,other=0),mask=ok)
    tl.store(O+(OFF+row)*5*WIDTH+3*WIDTH+j,tl.load(D+row*SD+j,mask=ok,other=0),mask=ok)
    tl.store(O+(OFF+row)*5*WIDTH+4*WIDTH+j,tl.load(E+row*SE+j,mask=ok,other=0),mask=ok)

def copy_taps(dst,off,part):
    if isinstance(part,TapParts):
        ts=part.parts;w=ts[0].shape[1]
        assert all(t.dtype==dst.dtype and t.stride(1)==1 and t.shape[1]==w for t in ts)
        _pack_taps[(part.shape[0],triton.cdiv(w,256))](*ts,dst,*(t.stride(0) for t in ts),WIDTH=w,OFF=off,BLOCK=256,num_warps=4)
    else:dst[off:off+part.shape[0]].copy_(part)
