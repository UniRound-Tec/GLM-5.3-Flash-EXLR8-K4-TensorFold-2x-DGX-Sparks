"""Generalize the qualified original-K4 linear wrapper to other source K4 dense matrices."""
import re,torch,os
FAST=os.environ.get("TF_GLM_K4_DECODE_FAST")=="1"
from tensorfold.families.glm5_next.cuda import kda_source as ks
from tensorfold.families.glm5_next.cuda.split import rule,rank_cut,AXIS

def load(rd,name,device='cuda'):
    s=name.removeprefix('model.language_model.').removesuffix('.weight')
    m=re.fullmatch(r'layers\.(\d+)\.(.*)',s);assert m
    shared=re.fullmatch(r'mlp\.shared_experts\.(gate|up|down)_proj',m[2])
    if shared:
        layer=int(m[1]);proj=shared[1]
        a=rd.expert(layer,0,proj,'trellis',True).view(torch.int32).contiguous().unsqueeze(0).to(device)
        su=rd.expert(layer,0,proj,'suh',True).contiguous().to(device)
        sv=rd.expert(layer,0,proj,'svh',True).contiguous().to(device)
        return ks.K4Output(a,su,sv,len(sv),len(su))
    f=rd.safe(rd.root/f'dense/exlr8-dense-{int(m[1]):03d}.safetensors');key=m[2]
    fmt,n,k,width,side,*_=f.get(key+'.record').tolist()
    assert fmt==1 and width==4,(key,fmt,width)
    a=f.get(key+'.trellis').view(torch.int32)
    if side==2:a=a.reshape(k//128,n//128,8,8,32).permute(0,2,1,3,4).reshape(k//16,n//16,32)
    else:a=a.reshape(n//128,k//16,8,32).permute(1,0,2,3).reshape(k//16,n//16,32)
    su=f.get(key+'.suh');sv=f.get(key+'.svh');kind=rule(name)
    assert su.dtype==sv.dtype==torch.float16
    if kind!='rep':
        axis=AXIS[kind];lo,hi=rank_cut(rd.layout,name,[n,k],kind) or (rd.rank*([n,k][axis]//2),(rd.rank+1)*([n,k][axis]//2))
        assert lo%128==hi%128==0
        if axis==1:a=a[lo//16:hi//16];su=su[lo:hi];k=hi-lo
        else:a=a[:,lo//16:hi//16];sv=sv[lo:hi];n=hi-lo
    return ks.K4Output(a.contiguous().unsqueeze(0).to(device),su.contiguous().to(device),sv.contiguous().to(device),n,k)

def matmul(x,q,out=None,f32=False):
    if FAST:
        from . import k4_fast
        if k4_fast.eligible(x,q):return k4_fast.matmul(x,q,out=out,f32=f32)
    return ks.matmul(x,q,out=out,f32=f32)
