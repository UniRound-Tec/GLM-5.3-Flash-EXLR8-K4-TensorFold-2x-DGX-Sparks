"""Prompt-only FP32 weighted atomic reduction; reuse expert-output backing storage."""
from functools import lru_cache
from pathlib import Path
import importlib.util
import torch
import triton
import triton.language as tl
from tensorfold.cuda import experts as grouped
from . import exl3_mm

@lru_cache(maxsize=1)
def extension():
    spec=importlib.util.spec_from_file_location('glm_prefill2000_expert_atomic',Path(__file__).with_name('prefill_atomic_kernel.so'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def output(ey, rows, dims):
    return ey.view(-1)[:rows*dims].view(rows,dims)

def routed(x,pick,plan,ex,s,ey,wts,R,limit):
    assert plan.tile==64 and ex.shared_suh is None
    base=exl3_mm._ext();d,n=ex.dims,ex.width;slots=s.slots
    out=output(ey,R,d);out.zero_()
    base.rot_in(x,x.stride(0),pick,ex.suh_g,ex.suh_u,s.xg,s.xu,R,d,slots)
    extension().prompt_atomic(s.xg,s.xu,ex.gt,ex.ut,ex.dt,plan.items,plan.counts,plan.members,ex.svh_g,ex.svh_u,ex.suh_d,ex.svh_d,s.xd,out,d,n,ex.count,grouped.max_items(R*slots,plan.experts,64),float(limit),64,slots,False,s.order if exl3_mm.item_order() else None,wts)

@triton.jit
def _add_shared(X,S,W,Y,D:tl.constexpr,SLOTS:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0);d=tl.program_id(1)*B+tl.arange(0,B)
    value=tl.load(X+r*D+d)+tl.load(S+r*D+d)*tl.load(W+r*SLOTS+SLOTS-1)
    tl.store(Y+r*D+d,value)

def add_shared(x,sy,wts,out):
    r,d=out.shape
    assert x.is_contiguous() and sy.is_contiguous() and wts.is_contiguous() and out.is_contiguous()
    _add_shared[(r,d//1024)](x,sy,wts,out,D=d,SLOTS=wts.shape[1],B=1024,num_warps=4)
