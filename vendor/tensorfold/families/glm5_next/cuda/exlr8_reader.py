"""Flash EXLR8 K4 to existing GLM tensor interfaces, without re-quantizing experts.

Codec/layout derived from TensorFold exlr8_codec/exlr8_experts (Apache-2.0).
Weights retain the source model's terms. Dense matrices reconstruct into BF16.
"""
import json,struct,re,os
from pathlib import Path
import numpy as np
import torch

DT={'U8':torch.uint8,'I8':torch.int8,'I16':torch.int16,'F16':torch.float16,
    'BF16':torch.bfloat16,'F32':torch.float32,'I32':torch.int32,'I64':torch.int64,'U64':torch.uint64}

class Safe:
 def __init__(self,p):
  self.p=Path(p)
  with self.p.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];self.h=json.loads(f.read(n));self.base=n+8
 def raw(self,a,n):
  with self.p.open('rb') as f:
   f.seek(a);b=f.read(n);assert len(b)==n
   os.posix_fadvise(f.fileno(),a,n,os.POSIX_FADV_DONTNEED)
  return b
 def get(self,k):
  d=self.h[k];a,b=d['data_offsets']
  return torch.frombuffer(bytearray(self.raw(self.base+a,b-a)),dtype=DT[d['dtype']]).reshape(d['shape'])

BOOK=None
def codebook():
 global BOOK
 if BOOK is None:
  s=np.arange(65536,dtype=np.uint32)*np.uint32(0xCBAC1FED)
  z=(s & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
  a=(z & 65535).astype(np.uint16).view(np.float16)
  b=(z>>16).astype(np.uint16).view(np.float16)
  BOOK=torch.from_numpy((a+b).astype(np.float16)).to(torch.float8_e4m3fn).float().numpy()
 return BOOK

def tile_values(words):
 """Independent circular bitstream decoder; output tiles as [tile,input16,output16]."""
 w=np.asarray(words,dtype=np.uint32).reshape(-1,32)
 p=np.arange(256);start=((p+1)*4+1024-16)%1024;ix=start//32;off=start%32
 pair=(w[:,ix].astype(np.uint64)<<32)|w[:,(ix+1)%32].astype(np.uint64)
 state=((pair << off.astype(np.uint64))>>48).astype(np.uint16)
 val=codebook()[state]
 lane=p//8;j=p%8;r=2*(lane%4)+(j%2)+8*((j//2)%2);c=lane//4+8*(j//4)
 out=np.empty((len(w),16,16),np.float32);out[:,r,c]=val
 return out

def fwht(x,axis):
 y=np.moveaxis(x,axis,-1).copy();assert y.shape[-1]==128
 for d in (1,2,4,8,16,32,64):
  z=y.reshape(*y.shape[:-1],128//(2*d),2,d)
  a=z[...,0,:].copy();b=z[...,1,:].copy();z[...,0,:]=a+b;z[...,1,:]=a-b
 return np.moveaxis(y,-1,axis)

def reconstruct(words,suh,svh):
 """Legacy [input/16,output/16,32] tile layout -> HF [output,input] BF16.
 Block Hadamards use fp32 sums, exact here before scaling because codebook is E4M3.
 """
 a=np.asarray(words,dtype=np.uint32);ki,no,_=a.shape
 out=np.empty((no*16,ki*16),np.float32)
 su=np.asarray(suh,dtype=np.float32);sv=np.asarray(svh,dtype=np.float32)
 for i in range(0,ki,8):
  # All output blocks for one input block, bounded temporary memory.
  t=tile_values(a[i:i+8]).reshape(8,no,16,16).transpose(0,2,1,3).reshape(128,no*16)
  blocks=t.reshape(128,no//8,128).transpose(1,0,2)
  blocks=fwht(fwht(blocks,1),2)/128.0
  # f64 scale product matches the exact codec reference before final BF16 rounding.
  blocks=blocks.astype(np.float64)
  blocks*=su[i*16:i*16+128][None,:,None]
  blocks*=sv.reshape(no//8,128)[:,None,:]
  out[:,i*16:i*16+128]=blocks.transpose(0,2,1).reshape(no*16,128)
 return torch.from_numpy(out).to(torch.bfloat16)

class RankReader:
 def __init__(self,model_dir,rank,world=2):
  assert world==2
  self.dir=Path(model_dir);self.root=Path(json.loads((self.dir/'exlr8-source.json').read_text())['path'])
  self.rank=rank;self.world=world;self.files={};self.index={};self.ahead={};self._idx=None;self._layer=None
  from .split import layout_for
  self.layout=layout_for(model_dir,rank,world)
 def safe(self,p):
  key=str(p)
  if key not in self.files:self.files[key]=Safe(p)
  return self.files[key]
 def prefetch(self,*args,**kwargs):pass
 def close(self):self.files.clear();self._idx=None
 def idx(self,layer):
  if self._layer!=layer:
   f=self.safe(self.root/f'experts/exlr8-layer-{layer:03d}.index.safetensors')
   self._idx={k:f.get(k) for k in f.h if k.startswith(('k4.','shared.k4.','signs.'))};self._layer=layer
  return self._idx
 def expert(self,layer,e,proj,part,shared=False):
  ix=self.idx(layer);pre='shared.k4.' if shared else 'k4.';e=0 if shared else e
  channel=ix[pre+proj+('.suh' if proj=='down' else '.svh')][e]
  block=ix[pre+proj+('.sv_block' if proj=='down' else '.su_block')][e]
  sign=ix['signs.out' if proj=='down' else 'signs.in']
  hidden=(block.repeat_interleave(128)*sign).to(torch.float16)
  lo=self.rank*1024;hi=lo+1024
  if part=='suh':return (channel[lo:hi] if proj=='down' else hidden).contiguous()
  if part=='svh':return (hidden if proj=='down' else channel[lo:hi]).contiguous()
  assert part=='trellis'
  pi={'gate':0,'up':1,'down':2}[proj]
  entries=ix[pre+'index'][e,pi,lo//128:hi//128].numpy()
  p=self.root/f'experts/exlr8-layer-{layer:03d}.k4.safetensors'
  tiles=[]
  with p.open('rb') as f:
   for off,n in entries:
    assert int(n)==4096*128//2
    f.seek(int(off));b=f.read(int(n));assert len(b)==int(n)
    w=np.frombuffer(b,np.uint32)
    if proj=='down':w=w.reshape(32,8,8,32).transpose(1,0,2,3).reshape(8,256,32)
    else:w=w.reshape(256,8,32)
    tiles.append(w)
   os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  w=np.concatenate(tiles,axis=0 if proj=='down' else 1).copy()
  return torch.from_numpy(w.view(np.int16)).reshape(w.shape[0],w.shape[1],64)
 def dense(self,f,key):
  rec=f.get(key+'.record').tolist();fmt,n,k,width,side,*_=rec
  if fmt==1:
   assert width==4
   a=f.get(key+'.trellis').numpy().view(np.uint32)
   if side==2:a=a.reshape(k//128,n//128,8,8,32).transpose(0,2,1,3,4).reshape(k//16,n//16,32)
   else:a=a.reshape(n//128,k//16,8,32).transpose(1,0,2,3).reshape(k//16,n//16,32)
   return reconstruct(a,f.get(key+'.suh').numpy(),f.get(key+'.svh').numpy())
  if fmt==6:return f.get(key+'.bf16')
  if fmt==7:return f.get(key+'.fp32')
  a=f.get(key+'.fp8').view(torch.float8_e4m3fn).float()
  if fmt==3:a*=torch.pow(2.,f.get(key+'.e8m0').float()-127).repeat_interleave(32,dim=1)
  elif fmt==4:a*=f.get(key+'.scale').repeat_interleave(128,0).repeat_interleave(128,1)[:n,:k]
  elif fmt==5:a*=f.get(key+'.scale')[:,None]
  else:raise ValueError(('unsupported dense format',fmt,key))
  return a.to(torch.bfloat16)
 def get(self,name):
  from .split import rule,rank_cut,AXIS
  prefix='model.language_model.'
  s=name[len(prefix):] if name.startswith(prefix) else name
  m=re.match(r'layers\.(\d+)\.(.*)',s)
  if m:
   layer=int(m[1]);key=m[2]
   ex=re.match(r'mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(trellis|suh|svh)$',key)
   if ex:return self.expert(layer,int(ex[1]),ex[2],ex[3])
   sh=re.match(r'mlp\.shared_experts\.(gate|up|down)_proj\.weight$',key)
   if sh:
    proj=sh[1];w=self.expert(layer,0,proj,'trellis',True).view(torch.int32).numpy()
    return reconstruct(w,self.expert(layer,0,proj,'suh',True).numpy(),self.expert(layer,0,proj,'svh',True).numpy())
   f=self.safe(self.root/f'dense/exlr8-dense-{layer:03d}.safetensors')
  else:f=self.safe(self.root/'dense/exlr8-vocab.safetensors');key=s
  if key in f.h:x=f.get(key)
  elif key.endswith('.weight') and key[:-7]+'.record' in f.h:x=self.dense(f,key[:-7])
  else:raise KeyError((name,list(f.h)))
  kind=rule(name)
  if kind=='rep':return x
  cut=rank_cut(self.layout,name,list(x.shape),kind)
  axis=AXIS[kind]
  lo,hi=cut or (self.rank*(x.shape[axis]//2),(self.rank+1)*(x.shape[axis]//2))
  return x.narrow(axis,lo,hi-lo).contiguous()

def headers(model_dir):
 """Logical runtime tensors for existing admission, separate from packed source files."""
 return json.loads((Path(model_dir)/'exlr8-runtime-headers.json').read_text())


def checkpoint_contract(model_dir):
    """Validate explicit EXLR8 provenance and Flash geometry; never masquerade as mcg EXL3."""
    path = Path(model_dir)
    cfg = json.loads((path / "config.json").read_text())
    quant = cfg.get("quantization_config") or {}
    source = json.loads((path / "exlr8-source.json").read_text())
    if quant.get("source_format") != "exlr8-e4m3-v2" or quant.get("bits") != 4:
        raise ValueError("Expected EXLR8 E4M3 K4 compatibility checkpoint")
    if not source.get("revision") or quant.get("source_revision") != source["revision"]:
        raise ValueError("EXLR8 source revision mismatch")
    root = Path(source["path"])
    original = json.loads((root / "config.json").read_text())
    expected = {"hidden_size":4096,"num_hidden_layers":45,"n_routed_experts":288,"moe_intermediate_size":2048}
    for config in (cfg, original):
        text = config.get("text_config") or config
        if any(text.get(k) != v for k,v in expected.items()):
            raise ValueError("Unsupported EXLR8 Flash geometry")
    for layer in range(3,45):
        index = Safe(root / f"experts/exlr8-layer-{layer:03d}.index.safetensors")
        if index.h.get("k4.index",{}).get("shape") != [288,3,16,2]:
            raise ValueError(f"Bad EXLR8 K4 index geometry at layer {layer}")
        if not (root / f"experts/exlr8-layer-{layer:03d}.k4.safetensors").is_file():
            raise ValueError(f"Missing EXLR8 K4 layer {layer}")
    return {"family":"glm53_flash_exlr8_k4","expert_compute":"fp16","dense_compute":"bf16"}
