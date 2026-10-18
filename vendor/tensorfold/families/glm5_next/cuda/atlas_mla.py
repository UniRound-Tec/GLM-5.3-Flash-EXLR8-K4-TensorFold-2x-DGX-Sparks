"""Atlas's pipelined sparse MLA over existing row-scaled FP8 cache."""
import ctypes
from functools import lru_cache
from pathlib import Path
import torch

@lru_cache(None)
def library():
    lib=ctypes.CDLL(str(Path(__file__).with_name('libatlas_mla.so')))
    lib.atlas_mla.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_uint64]
    lib.atlas_mla.restype=ctypes.c_int
    rc=lib.atlas_mla_init()
    if rc:raise RuntimeError(f'Atlas MLA shared-memory initialization failed: {rc}')
    return lib

def run(q,cache,tokens,counts,out,scale):
    assert q.shape[1:]==(32,512) and q.dtype==out.dtype==torch.bfloat16
    assert cache.dtype==torch.uint8 and cache.shape[1]==528
    assert tokens.dtype==counts.dtype==torch.int32
    assert all(x.is_contiguous() for x in (q,cache,tokens,counts,out))
    pointers=(ctypes.c_uint64*5)(*[x.data_ptr() for x in (q,cache,tokens,counts,out)])
    rc=library().atlas_mla(pointers,q.shape[0],tokens.shape[1],scale,torch.cuda.current_stream().cuda_stream)
    if rc:raise RuntimeError(f'Atlas MLA launch failed: {rc}')
    return out
