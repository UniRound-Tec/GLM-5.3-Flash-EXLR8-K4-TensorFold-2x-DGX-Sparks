"""Original K4 for small windows; lossless BF16 exponent maps for prompt GEMM.

The optional BF16 representation replaces, rather than accompanies, the full
BF16 tensor. Both representations together must fit inside its original bytes.
One existing source_fp8 scratch buffer is shared across all matrices.
"""
from dataclasses import dataclass
import numpy as np
import torch
import triton
import triton.language as tl
from tensorfold.families.glm5_next.cuda import qmm,source_fp8
try:
    from . import source_k4
except ImportError:
    import source_k4

ledger=[]

@dataclass
class ZipK4(qmm.B16):
    source: object
    packed: tuple
    start: int
    escape_fraction: float
    def nbytes(self):
        sources=self.source if isinstance(self.source,list) else [self.source]
        return sum(q.nbytes() for q in sources)+sum(t.numel()*t.element_size() for t in self.packed)

def pack_cpu(w):
    a=np.ascontiguousarray(w,dtype=np.uint16).reshape(-1,64)
    hist=np.zeros(256,dtype=np.int64)
    for i in range(0,len(a),65536):
        hist+=np.bincount(((a[i:i+65536]>>7)&255).ravel(),minlength=256)
    start=int(np.convolve(hist,np.ones(7,dtype=np.int64),'valid').argmax())
    sm=[];full=[];maps=[];offsets=[];count=0;shifts=np.arange(64,dtype=np.uint64)
    for i in range(0,len(a),65536):
        z=a[i:i+65536];ex=(z>>7)&255
        code=np.where((ex>=start)&(ex<start+7),ex-start,7).astype(np.uint8)
        esc=code==7;cnt=esc.sum(1)
        offsets.append((count+np.r_[0,np.cumsum(cnt)[:-1]]).astype(np.int32));count+=int(cnt.sum())
        maps.append(np.stack([(((code>>b)&1).astype(np.uint64)<<shifts).sum(1,dtype=np.uint64) for b in range(3)]))
        sm.append(((z&127)|((z>>8)&128))[~esc].astype(np.uint8))
        full.append(z[esc].astype(np.uint16))
    arrays=(np.concatenate(sm),np.concatenate(full),np.concatenate(maps,axis=1).view(np.int64),np.concatenate(offsets))
    return arrays,start,count/a.size

@triton.jit
def _decompress(S,F,B,O,Y,TOTAL:tl.constexpr,START:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    block=i//64;bit=i%64;count:tl.constexpr=TOTAL//64
    b0=tl.load(B+block,i<TOTAL,other=0).to(tl.uint64)
    b1=tl.load(B+count+block,i<TOTAL,other=0).to(tl.uint64)
    b2=tl.load(B+2*count+block,i<TOTAL,other=0).to(tl.uint64)
    code=((b0>>bit)&1)|(((b1>>bit)&1)<<1)|(((b2>>bit)&1)<<2)
    before=(b0&b1&b2)&((tl.full((),1,tl.uint64)<<bit)-1)
    ec=tl.inline_asm_elementwise('popc.b64 $0, $1;',constraints='=r,l',args=[before],dtype=tl.int32,is_pure=True,pack=1)
    fo=tl.load(O+block,i<TOTAL,other=0)
    so=block*64-fo
    sm=tl.load(S+so+bit-ec,(i<TOTAL)&(code!=7),other=0).to(tl.uint16)
    full=tl.load(F+fo+ec,(i<TOTAL)&(code==7),other=0).to(tl.uint16)
    value=(sm&127)|((sm&128)<<8)|((START+code.to(tl.uint16))<<7)
    tl.store(Y+i,tl.where(code==7,full,value).to(tl.uint16).to(tl.bfloat16,bitcast=True),i<TOTAL)

def prepare(q):
    need=q.n*q.k
    if source_fp8._fallback is None or source_fp8._fallback.numel()<need:
        source=q.source[0] if isinstance(q.source,list) else q.source
        source_fp8._fallback=torch.empty(need,dtype=torch.bfloat16,device=source.trellis.device)
    source_fp8._owner=None

def unpack(q):
    capture=torch.cuda.is_current_stream_capturing()
    if source_fp8._prefill_depth and not capture and source_fp8._owner is q:
        return source_fp8._fallback[:q.n*q.k].view(q.n,q.k)
    y=source_fp8._fallback[:q.n*q.k].view(q.n,q.k)
    _decompress[(triton.cdiv(q.n*q.k,512),)](*q.packed,y,q.n*q.k,q.start,512,num_warps=4)
    source_fp8._owner=q if source_fp8._prefill_depth and not capture else None
    return y

def make(rd,name,raw,device='cuda'):
    src=[source_k4.load(rd,n,device=device) for n in name] if isinstance(name,list) else source_k4.load(rd,name,device=device)
    n=sum(q.n for q in src) if isinstance(src,list) else src.n
    k=src[0].k if isinstance(src,list) else src.k
    if isinstance(src,list):assert all(q.k==k for q in src)
    arrays,start,fraction=pack_cpu(raw.cpu().contiguous().view(torch.uint16).numpy())
    packed=tuple(torch.from_numpy(a if a.size else np.zeros(1,dtype=a.dtype)).to(device) for a in arrays)
    q=ZipK4(torch.empty(0,device=device),n,k,src,packed,start,fraction)
    prepare(q)
    reference=raw.to(device)
    assert torch.equal(unpack(q).view(torch.int16),reference.contiguous().view(torch.int16)),name
    assert q.nbytes()<=raw.numel()*raw.element_size(),('Hybrid representation exceeds BF16 bytes',name,q.nbytes())
    ledger.append({'name':name,'shape':[q.n,q.k],'original_bytes':raw.numel()*raw.element_size(),
                   'hybrid_bytes':q.nbytes(),'escape_fraction':fraction})
    return q

def matmul(x,q,*,out=None,f32=False,part=None):
    if len(x)<=32:
        if isinstance(q.source,list):
            parts=[source_k4.matmul(x,p,f32=f32) for p in q.source]
            return torch.cat(parts,dim=1,out=out)
        return source_k4.matmul(x,q.source,out=out,f32=f32)
    return qmm._matmul_b16(x,qmm.B16(unpack(q),q.n,q.k),out=out,f32=f32,part=part)
