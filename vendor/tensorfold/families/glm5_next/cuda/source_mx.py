"""Source MXFP8, tiled without requantizing weights; native small-window W8A8.
Large windows reconstruct the exact source BF16 matrix into the existing shared buffer.
"""
import ctypes
from pathlib import Path
import torch,triton
import triton.language as tl
from tensorfold.families.glm5_next.cuda import qmm
LIB=ctypes.CDLL(str(Path(__file__).with_name('source_mx_kernel.so')))
LIB.launch_tiled.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*4+[ctypes.c_void_p]
LIB.launch_tiled.restype=ctypes.c_int

@triton.jit
def _pack(W,S,P,PS,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);total=tl.cdiv(N,64)*4*(K//32)*128
    lane=i%32;reg=(i//32)%4;tile=i//128;kb=tile%(K//32);nb=tile//(K//32)
    n=nb*16+lane//4+(reg%2)*8;k=kb*32+(lane%4)*4+(reg//2)*16
    v=tl.load(W.to(tl.pointer_type(tl.int32))+n.to(tl.int64)*(K//4)+k//4,(i<total)&(n<N),other=0)
    tl.store(P+i,v,i<total)
    st=i//32;sl=i%32;sn=(st//(K//32))*16+sl//4+(sl%2)*8
    sv=tl.load(S+sn.to(tl.int64)*(K//32)+st%(K//32),(i<total//4)&(sn<N),other=127)
    tl.store(PS+i,sv.to(tl.uint8),i<total//4)

def pack(q):
    assert q.block==32 and not getattr(q,'native_mx',False)
    count=triton.cdiv(q.n,64)*4*(q.k//32)*128
    w=torch.empty(count,device=q.weight.device,dtype=torch.int32)
    s=torch.empty(count//4,device=q.weight.device,dtype=torch.uint8)
    before=q.nbytes()
    _pack[(triton.cdiv(count,512),)](q.weight,q.scale,w,s,q.n,q.k,512)
    q.weight=w;q.scale=s;q.native_mx=True
    return q.nbytes()-before

@triton.jit
def _unpack(W,S,Y,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);n=i//K;k=i%K
    tile=(n//16)*(K//32)+k//32
    reg=(n%16)//8+2*((k%32)//16)
    lane=(n%8)*4+(k%16)//4
    word=tl.load(W+tile.to(tl.int64)*128+reg*32+lane,i<N*K,other=0).to(tl.uint32)
    b=((word>>((k%4)*8))&255).to(tl.uint8).to(tl.float8e4nv,bitcast=True).to(tl.bfloat16)
    scale=tl.load(S+tile.to(tl.int64)*32+(n%8)*4+(n%16)//8,i<N*K,other=127).to(tl.int32)
    bits=b.to(tl.uint16,bitcast=True).to(tl.int32)
    value=tl.where((bits&32767)!=0,bits+((scale-127)<<7),bits).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    tl.store(Y+i,value,i<N*K)

def unpack(q,y):
    _unpack[(triton.cdiv(q.n*q.k,1024),)](q.weight,q.scale,y,q.n,q.k,1024)
    return y

@triton.jit
def _quant(X,A,S,M:tl.constexpr,K:tl.constexpr,STRIDE:tl.constexpr):
    row=tl.program_id(0);block=tl.program_id(1);k=block*32+tl.arange(0,32)
    x=tl.load(X+row*STRIDE+k).to(tl.float32);amax=tl.max(tl.abs(x),0)
    e=tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(amax/448.,2.**-126))),-126.),127.)
    quant=tl.minimum(tl.maximum(x/tl.exp2(e),-448.),448.).to(tl.float8e4nv)
    tl.store(A+row*K+k,quant);tl.store(S+row*(K//32)+block,(e+127).to(tl.uint8))

_scratch={}
def scratch(m,n,k,device):
    key=(m,n,k,str(device))
    if key not in _scratch:
        assert not torch.cuda.is_current_stream_capturing(),'Native MX shape must be warmed'
        _scratch[key]=(torch.empty((m,k),device=device,dtype=torch.float8_e4m3fn),
                       torch.empty((m,k//32),device=device,dtype=torch.uint8),
                       torch.empty((4*m*n,),device=device,dtype=torch.float32))
    return _scratch[key]

def matmul(x,q,out=None,f32=False,part=None):
    m,k=x.shape;assert m<=32 and k==q.k and x.dtype==torch.bfloat16 and x.stride(1)==1
    if out is None:out=torch.empty((m,q.n),device=x.device,dtype=torch.float32 if f32 else torch.bfloat16)
    assert out.is_contiguous() and out.shape==(m,q.n)
    a,s,p=scratch(m,q.n,k,x.device)
    _quant[(m,k//32)](x,a,s,m,k,x.stride(0),num_warps=1)
    error=LIB.launch_tiled(a.data_ptr(),s.data_ptr(),q.weight.data_ptr(),q.scale.data_ptr(),p.data_ptr(),
                           m,q.n,k,4,torch.cuda.current_stream().cuda_stream)
    if error:raise RuntimeError(f'Native MX launch error {error}')
    qmm._reduce[(triton.cdiv(m*q.n,1024),)](p,out,m*q.n,SK=4,BLOCK=1024,F32=f32,num_warps=4)
    return out
