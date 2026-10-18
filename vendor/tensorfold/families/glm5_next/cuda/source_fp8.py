"""Source FP8 storage with exact BF16 reconstruction before the existing BF16 dot chain.
No activation quantization; scale is inside weight reconstruction, never moved after dot.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from tensorfold.families.glm5_next.cuda import qmm

@dataclass
class SourceF8(qmm.F8):
    block: int = 32

    def nbytes(self):return self.weight.numel()*self.weight.element_size()+self.scale.numel()*self.scale.element_size()

def from_reader(rd, name, *, device='cuda', rows=None):
    from tensorfold.families.glm5_next.cuda.split import rule, rank_cut, AXIS
    import re
    s=name.removeprefix('model.language_model.')
    m=re.match(r'layers\.(\d+)\.(.*)',s)
    f=rd.safe(rd.root/f'dense/exlr8-dense-{int(m[1]):03d}.safetensors') if m else rd.safe(rd.root/'dense/exlr8-vocab.safetensors')
    key=m[2] if m else s
    key=key.removesuffix('.weight')
    if key+'.record' not in f.h:return None
    fmt,n,k,*_=f.get(key+'.record').tolist()
    if fmt not in (3,5):return None
    w=f.get(key+'.fp8').view(torch.float8_e4m3fn)
    if fmt==3:
        scale=f.get(key+'.e8m0').to(torch.uint16);block=32
    else:
        scale=f.get(key+'.scale').float().reshape(n,1);block=k
    kind=rule(name)
    if kind!='rep':
        axis=AXIS[kind];lo,hi=rank_cut(rd.layout,name,[n,k],kind) or (rd.rank*([n,k][axis]//2),(rd.rank+1)*([n,k][axis]//2))
        w=w.narrow(axis,lo,hi-lo)
        if axis==0:scale=scale[lo:hi]
        elif block!=k:
            assert lo%block==0 and hi%block==0
            scale=scale[:,lo//block:hi//block]
        else:block=hi-lo
    if rows is not None:w=w[rows[0]:rows[1]];scale=scale[rows[0]:rows[1]]
    n,k=w.shape
    q=SourceF8(w.contiguous().view(torch.uint16).to(device),scale.contiguous().to(device),n,k,block)
    return q

def reconstruct(q):
    if getattr(q,"native_mx",False):return unpack(q).clone()
    scale=torch.pow(2.,q.scale.float()-127) if q.scale.dtype==torch.uint16 else q.scale
    return (q.weight.view(torch.float8_e4m3fn).float()*scale.repeat_interleave(q.block,1)).bfloat16()

def cat(parts):
    assert all(p.block==parts[0].block and p.k==parts[0].k for p in parts)
    return SourceF8(torch.cat([p.weight for p in parts]),torch.cat([p.scale for p in parts]),sum(p.n for p in parts),parts[0].k,parts[0].block)

@triton.jit
def _weight(w, ws, POW2:tl.constexpr):
    v=w.to(tl.uint8).to(tl.float8e4nv,bitcast=True).to(tl.bfloat16)
    if POW2:
        bits=v.to(tl.uint16,bitcast=True).to(tl.int32)
        shifted=bits+((ws.to(tl.int32)-127)<<7)
        return tl.where((bits&32767)!=0,shifted,bits).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    return (v.to(tl.float32)*ws).to(tl.bfloat16)

@triton.jit
def _packed_mm(X, W, S, OUT, PART, M, x_stride, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr, SEQ: tl.constexpr = False, SCALE_BLOCK: tl.constexpr = 32, POW2: tl.constexpr = False):
    """Multiply bf16 rows over a fixed K slice in order with fp32 sums, independently of other rows; SEQ: one program
    runs every slice and sums them as _reduce does (the same bits, without the fp32 partials' round trip)."""

    PER: tl.constexpr = K // SK
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    if SEQ:
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SK):
            part = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
            for k0 in range(s * PER, s * PER + PER, BK):
                x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
                w = tl.load(W.to(tl.pointer_type(tl.uint8)) + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0)
                if POW2:
                    sw = tl.load(S + rn[:, None].to(tl.int64) * (K // 32) + (k0 // 32 + tl.arange(0, BK // 32))[None, :], mask=n_ok[:, None], other=127)
                    ws = tl.broadcast_to(sw[:, :, None], (BLOCK_N, BK // 32, 32)).reshape(BLOCK_N, BK)
                else:
                    ws = tl.load(S + rn, mask=n_ok, other=0)[:, None]
                w = _weight(w, ws, POW2)
                part = part + tl.dot(x, tl.trans(w))
            if s == 0:
                acc = part
            else:
                acc = acc + part
    else:
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for k0 in range(pid_s * PER, pid_s * PER + PER, BK):
            x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W.to(tl.pointer_type(tl.uint8)) + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0)
            if POW2:
                sw = tl.load(S + rn[:, None].to(tl.int64) * (K // 32) + (k0 // 32 + tl.arange(0, BK // 32))[None, :], mask=n_ok[:, None], other=127)
                ws = tl.broadcast_to(sw[:, :, None], (BLOCK_N, BK // 32, 32)).reshape(BLOCK_N, BK)
            else:
                ws = tl.load(S + rn, mask=n_ok, other=0)[:, None]
            w = _weight(w, ws, POW2)
            acc = acc + tl.dot(x, tl.trans(w))
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1 or SEQ:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


TUNING=(64,4,1)

def _direct(x,q,*,out=None,f32=False,part=None):
    from . import trial_control
    if (trial_control.ACTIVE.get('d30',0) & 2 and x.shape[0] == 1
            and q.n == 77440 and q.k == 4096 and q.block == q.k
            and q.scale.dtype == torch.float32):
        from .head_gemv import matmul as single_head
        return single_head(x,q,out=out,f32=f32,config=(8,512,4))
    # Used only during capture of the small verify bank; eager/prefill remain unchanged.
    if (trial_control.VERIFY8_CAPTURE and 2 <= x.shape[0] <= 4
            and q.n == 77440 and q.k == 4096 and q.block == q.k
            and q.scale.dtype == torch.float32):
        from .head_multi import matmul as multi_head
        return multi_head(x,q,out=out,f32=f32,config=(4,512,4))
    if getattr(q,"native_mx",False):
        from . import source_mx
        return source_mx.matmul(x,q,out=out,f32=f32,part=part)
    m,k=x.shape;bm=qmm.bucket(min(m,128));bn=qmm.B16_BN
    warps,stages=qmm.B16_CONFIG[bm];sk=qmm.b16_split_k(q.n,q.k);seq=qmm._seq(sk,m,q.n,q.k)
    bn,warps,stages=TUNING if m<=32 else (bn,warps,stages)
    if out is None:out=torch.empty((m,q.n),device=x.device,dtype=torch.float32 if f32 else torch.bfloat16)
    if sk>1 and not seq:
        if part is None or part.numel()<sk*m*q.n:part=torch.empty((sk*m*q.n,),device=x.device,dtype=torch.float32)
    _packed_mm[(triton.cdiv(m,bm),triton.cdiv(q.n,bn),1 if seq else sk)](x,q.weight,q.scale,out,part if sk>1 and not seq else out,m,x.stride(0),N=q.n,K=k,SK=sk,BM=bm,BLOCK_N=bn,BK=qmm.B16_BK,F32=f32,SEQ=seq,SCALE_BLOCK=q.block,POW2=q.scale.dtype==torch.uint16,num_warps=warps,num_stages=stages)
    if sk>1 and not seq:qmm._reduce[(triton.cdiv(m*q.n,1024),)](part,out,m*q.n,SK=sk,BLOCK=1024,F32=f32,num_warps=4)
    return out

# One reusable BF16 buffer for large windows; no persistent expanded copy per matrix.
_fallback=None
_prefill_depth=0
_owner=None
cache_hits=0
unpack_calls=0
ledger=[]
def prepare(q):
    global _fallback
    need=q.n*q.k
    if _fallback is None or _fallback.numel()<need:
        _fallback=torch.empty(need,device=q.weight.device,dtype=torch.bfloat16)

@triton.jit
def _unpack(W,S,Y,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr,POW2:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    p=tl.load(W+i//2,i<N*K,other=0).to(tl.int32)
    code=(p>>((i%2)*8))&255
    scale=tl.load(S+(i//K)*(K//B)+(i%K)//B,i<N*K,other=0)
    v=_weight(code,scale,POW2)
    tl.store(Y+i,v,i<N*K)

def unpack(q):
    global _owner, cache_hits, unpack_calls
    capture=torch.cuda.is_current_stream_capturing()
    if _prefill_depth and not capture and _owner is q:
        cache_hits+=1
        return _fallback[:q.n*q.k].view(q.n,q.k)
    assert _fallback is not None and _fallback.numel()>=q.n*q.k
    y=_fallback[:q.n*q.k].view(q.n,q.k)
    if getattr(q,"native_mx",False):
        from . import source_mx
        source_mx.unpack(q,y)
    else:
        _unpack[(triton.cdiv(q.n*q.k,1024),)](q.weight,q.scale,y,q.n,q.k,q.block,q.scale.dtype==torch.uint16,1024,num_warps=4)
    unpack_calls+=1
    _owner=q if _prefill_depth and not capture else None
    return y

def qualify(q,reference,label):
    prepare(q)
    assert torch.equal(unpack(q).view(torch.int16),reference.contiguous().view(torch.int16)),label
    if q.block==32:
        from . import source_mx
        source_mx.pack(q)
        assert torch.equal(unpack(q).view(torch.int16),reference.contiguous().view(torch.int16)),label+" tiled"
    ledger.append(dict(name=label,shape=[q.n,q.k],original_bytes=reference.numel()*2,packed_bytes=q.nbytes()))
    return q

def matmul(x,q,*,out=None,f32=False,part=None):
    if x.shape[0]>32:
        from . import trial_control, mx_prefill
        if trial_control.ACTIVE.get('mx_prefill',False) and mx_prefill.eligible(x,q,f32):
            return mx_prefill.matmul(x,q,out=out)
        w=unpack(q)
        return qmm._matmul_b16(x,qmm.B16(w,q.n,q.k),out=out,f32=f32,part=part)
    return _direct(x,q,out=out,f32=f32,part=part)

from contextlib import contextmanager
@contextmanager
def prefill_scope():
    # No captured decode graph (>32 rows) may mutate this buffer. This engine's
    # verify bank is bounded at 32; scopes are also reset on generator close/error.
    global _prefill_depth,_owner
    _owner=None
    _prefill_depth+=1
    try:
        yield
    finally:
        _owner=None
        _prefill_depth-=1
