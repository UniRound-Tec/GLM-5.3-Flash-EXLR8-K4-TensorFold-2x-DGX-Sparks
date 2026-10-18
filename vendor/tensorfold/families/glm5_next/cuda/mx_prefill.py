"""Native prefill input projections, sharing the existing BF16 fallback storage.

Only >=1024 rows of the two previously measured input shapes are eligible.
Decode, all other matrices, and small prefill tails retain their old dispatch.
"""
import ctypes
from pathlib import Path
import torch
_lib=None
calls=0
rows=0

def library():
    global _lib
    if _lib is None:
        lib=ctypes.CDLL(str(Path(__file__).with_name('mx_prefill_kernel.so')))
        u=ctypes.c_uint64;i=ctypes.c_int
        lib.mxp_load.argtypes=[];lib.mxp_load.restype=i
        lib.mxp_bytes.argtypes=[i,i];lib.mxp_bytes.restype=u
        lib.mxp_repack.argtypes=[u,u,u,u,i,u];lib.mxp_repack.restype=i
        lib.mxp_run.argtypes=[u,u,u,u,u,u,i,i,u];lib.mxp_run.restype=i
        rc=lib.mxp_load()
        if rc:raise RuntimeError(f'MX prefill module load: {rc}')
        _lib=lib
    return _lib

def eligible(x,q,f32=False):
    return (not f32 and 1024<=x.shape[0]<=4096 and q.k==4096
            and q.n in (12576,2048) and getattr(q,'native_mx',False)
            and x.dtype==torch.bfloat16 and x.is_contiguous())

def matmul(x,q,*,out=None):
    global calls,rows
    from . import source_fp8
    assert eligible(x,q) and not torch.cuda.is_current_stream_capturing()
    if out is None:out=torch.empty((len(x),q.n),device=x.device,dtype=torch.bfloat16)
    assert out.dtype==torch.bfloat16 and out.is_contiguous() and out.shape==(len(x),q.n)
    buf=source_fp8._fallback
    # Every write invalidates both the BF16 and ZipK4 unpack cache. No additional
    # resident weight layout or persistent activation/output buffer is created.
    source_fp8._owner=None
    lib=library();need=lib.mxp_bytes(len(x),q.n)
    assert buf is not None and buf.numel()*buf.element_size()>=need
    rc=lib.mxp_run(x.data_ptr(),q.weight.data_ptr(),q.scale.data_ptr(),out.data_ptr(),
          buf.data_ptr(),buf.numel()*buf.element_size(),len(x),q.n,torch.cuda.current_stream().cuda_stream)
    if rc:raise RuntimeError(f'MX prefill execution: {rc}')
    calls+=1;rows+=len(x)
    return out

def prepare():
    from . import source_fp8
    library()
    assert source_fp8._fallback.numel()*2>=library().mxp_bytes(4096,12576)
