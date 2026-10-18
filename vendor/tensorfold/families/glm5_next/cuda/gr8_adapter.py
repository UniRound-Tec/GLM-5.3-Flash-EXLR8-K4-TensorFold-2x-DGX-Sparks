"""gr8 E4 K4 kernels, Flash TP2 single-row bridge; original expert storage retained."""
from pathlib import Path
import os
import torch
_lib=None
def extension():
 global _lib
 if _lib is None:
  from torch.utils.cpp_extension import load
  os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.1');os.environ.setdefault('MAX_JOBS','2')
  _lib=load(name='glm_gr8_tp2',sources=[str(Path(__file__).with_name('gr8_bind.cu'))],extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','-gencode=arch=compute_121,code=sm_121'],verbose=False)
 return _lib
class Scratch:
 def __init__(self,dev):
  self.x=torch.empty((1,4096),device=dev,dtype=torch.float16)
  self.zero=torch.zeros(64,device=dev,dtype=torch.int32)
  self.slots=torch.arange(8,device=dev,dtype=torch.int32)
  self.gp=torch.zeros(8,8,2,16,8,128,device=dev)
  self.z=torch.zeros(8,8,8,4096,device=dev,dtype=torch.float16)
  self.yp=torch.zeros(1,4096,device=dev,dtype=torch.float16)
  self.fctl=torch.zeros(515,device=dev,dtype=torch.int32)
  self.out=torch.empty(1,4096,device=dev,dtype=torch.float32)
def weights(ex):
 if not hasattr(ex,'gr8_pack'):
  assert ex.dims==4096 and ex.width==1024
  tab=torch.tensor([[e*2097152,e*2097152,e*8,8,4,0] for e in range(ex.count)],device=ex.gt.device,dtype=torch.int64)
  v2=torch.stack([ex.svh_g.reshape(ex.count,8,128),ex.svh_u.reshape(ex.count,8,128),ex.suh_d.reshape(ex.count,8,128)],dim=2).contiguous()
  ex.gr8_pack=(tab,ex.gt,ex.ut,ex.dt,ex.suh_g,ex.suh_u,v2,ex.svh_d)
 return ex.gr8_pack
def run(x,pick,wts,ex,s):
 assert x.shape==(1,4096) and pick.shape[1]==9
 s.x.copy_(x)
 extension().run([s.x,pick[:1].view(-1),wts[:1].view(-1),*weights(ex),s.zero,s.slots,s.gp,s.z,s.yp,s.out,s.fctl,s.x])
 return s.out
