"""Atlas K128W adapter: existing QAD weights, shared scratch, prefill only."""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Weights:
    w13: torch.Tensor
    s13: torch.Tensor
    alpha13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    alpha2: torch.Tensor
    gs1: torch.Tensor
    gs2: torch.Tensor

def prepare(owner):
    from b12x.moe.fused_moe._impl import _tp4_tile_major_lookup
    w=owner.prepared._impl
    assert _tp4_tile_major_lookup(w.w1_fp4,w.w2_fp4) is not None
    scales=w.nvfp4_csf.inline_scales
    def table(t,stride):
        return torch.tensor([t.data_ptr()+i*stride for i in range(288)],dtype=torch.uint64,device=t.device)
    return Weights(table(w.w1_fp4,2048*2048),table(scales[0].storage,0),w.w1_alphas,
                   table(w.w2_fp4,4096*512),table(scales[1].storage,0),w.w2_alphas,w.a1_gscale,w.a2_gscale)

def scratch_bytes(capacity):
    r=capacity*8
    specs=[(288,4),(289,4),(288,4),(r,4),(r,4),(r,4),(578,4),
           (r*2048,1),(r*256,1),(r*512,1),(r*64,1),(r*4096,2)]
    return sum(((n*b+255)//256)*256 for n,b in specs)

class Workspace:
    def __init__(self,scratch,capacity=4096,library=None,modes=None,prefill_mode=0):
        self.capacity=capacity
        self.decode_modes=modes or {}
        self.prefill_mode=prefill_mode
        path=Path(library) if library else Path(__file__).with_name('libatlas_experts.so')
        self.lib=ctypes.CDLL(str(path))
        self.lib.atlas_run.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_int,ctypes.c_uint64,ctypes.c_int]
        self.lib.atlas_run.restype=ctypes.c_int
        self.lib.atlas_run_prefill.argtypes=self.lib.atlas_run.argtypes
        self.lib.atlas_run_prefill.restype=ctypes.c_int
        r=capacity*8
        specs=[(288,torch.int32),(289,torch.int32),(288,torch.int32),(r,torch.int32),(r,torch.int32),
               (r,torch.int32),(578,torch.int32),(r*2048,torch.uint8),(r*256,torch.uint8),
               (r*512,torch.uint8),(r*64,torch.uint8),(r*4096,torch.bfloat16)]
        assert scratch.numel()>=scratch_bytes(capacity)
        offset=0;self.tensors=[]
        for n,dtype in specs:
            size=n*torch.empty((),dtype=dtype).element_size()
            self.tensors.append(scratch[offset:offset+size].view(dtype))
            offset+=(size+255)//256*256
        self.scratch=scratch
    def run(self,weights,x,ids,probabilities,output,mode=None):
        assert 0<x.shape[0]<=self.capacity and x.dtype==output.dtype==torch.bfloat16
        assert ids.dtype==torch.int32 and probabilities.dtype==torch.float32
        assert all(t.is_contiguous() for t in (x,ids,probabilities,output))
        tensors=[x,ids,probabilities,output,*vars(weights).values(),*self.tensors]
        ptrs=(ctypes.c_uint64*24)(*[t.data_ptr() for t in tensors])
        if mode is None:
            mode=self.decode_modes.get(x.shape[0],self.prefill_mode)
            if x.shape[0]>=128:
                from .trial_control import ACTIVE
                mode=64 if ACTIVE['chunk8k'] and x.shape[0]>=8192 else ACTIVE['prefill']
        fn=self.lib.atlas_run_prefill if mode>=256 else self.lib.atlas_run
        code=fn(ptrs,x.shape[0],torch.cuda.current_stream().cuda_stream,mode)
        if code:raise RuntimeError(f'Atlas prefill launch error {code}')
        return output
