"""Real-source MXFP8 large-window probe. No engine/service modification."""
import gc,hashlib,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,'/b12x-source')
import torch,triton
import cutlass.cute as cute
P=Path('/probe');torch.set_num_threads(2)
assert not (P/'results.json').exists(),'Preserve previous results'
# Export actual compiled code for a post-run instruction audit, outside timing.
original_compile=cute.compile
exports=[]
def export_compile(*args,**kwargs):
    fn=original_compile(*args,**kwargs)
    name=f'mx_{len(exports)}';(P/'aot').mkdir(exist_ok=True)
    fn.export_to_c(str(P/'aot'),name,name)
    exports.append(dict(name=name,program=str(args[0])))
    return fn
cute.compile=export_compile
from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm._shared.wo_mxfp8 import pack_mxfp8_scales_for_dense_gemm
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from tensorfold.families.glm5_next.cuda import source_fp8,qmm
from tensorfold.families.glm5_next.cuda.exlr8_reader import RankReader

CASES=[('kda',0),('dsa',3),('kda',22),('dsa',43),('kda',44)]
TILES=[(64,64),(64,128),(128,128)]
result=dict(complete=False,started=time.time(),screening=[],rows=[],checks=[],exports=exports,
    protocol=dict(rows=[1024,2048,4096],repetitions=5,cache_flush_MiB=64,
                  weights='Original EXLR8 dense MXFP8; lossless layout change only',
                  activations='Stored same-site decode activations repeated to large M; not a new prefill capture',
                  hardware='Both TP slices on Spark01 GPU',generations=0,reloads=0,
                  load_path='cpasync',new_quantization='BF16 activations to block32 E4M3 + UE8M0',
                  reference_guard='Quantized mathematical-reference relative L2 <0.005 (implementation guard, not a quality certificate)'))
flush=torch.empty(64<<20,device='cuda',dtype=torch.uint8)

def save():
    tmp=P/'results.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(P/'results.json')
def emit(row):save();print(json.dumps(row),flush=True)
def err(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30))
def finite(t):return bool(torch.isfinite(t).all())
def timing(calls):
    for fn in calls.values():fn()
    torch.cuda.synchronize();times={k:[] for k in calls}
    for rep in range(5):
        keys=list(calls);keys=keys[rep%len(keys):]+keys[:rep%len(keys)]
        for key in keys:
            flush.fill_(rep)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();calls[key]();b.record();b.synchronize()
            times[key].append(a.elapsed_time(b))
    return {k:dict(mean_ms=statistics.mean(v),min_ms=min(v),max_ms=max(v),samples_ms=v) for k,v in times.items()}

def bf16_tuned(x,w,out,part):
    m,k=x.shape;n=w.shape[0];bm,bn,warps,stages=(64,128,8,3)
    sk=qmm.b16_split_k(n,k);seq=qmm._seq(sk,m,n,k)
    qmm._bmm[(triton.cdiv(m,bm),triton.cdiv(n,bn),1 if seq else sk)](x,w,out,part,m,x.stride(0),
        N=n,K=k,SK=sk,BM=bm,BLOCK_N=bn,BK=64,F32=False,SEQ=seq,num_warps=warps,num_stages=stages)
    if sk>1 and not seq:
        qmm._reduce[(triton.cdiv(m*n,1024),)](part,out,m*n,SK=sk,BLOCK=1024,F32=False,num_warps=4)
    return out

def native_buffers(m,n,k):
    npad=triton.cdiv(n,128)*128
    aq=torch.empty((m,k),device='cuda',dtype=torch.float8_e4m3fn)
    ar=torch.empty((m,k//32),device='cuda',dtype=torch.uint8)
    am=pack_mxfp8_scales_for_dense_gemm(torch.full_like(ar,127),m=m,k=k)
    y=torch.empty((m,npad,1),device='cuda',dtype=torch.bfloat16)
    final=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    return aq,ar,am,y,final

def quantize(x,b):quantize_mxfp8_rows_cute(x,b[0],b[1],b[2])
def gemm(x,b,w,s,tile):
    dense_gemm((b[0].unsqueeze(-1),b[2]),(w.unsqueeze(-1),s),out=b[3],
        ab_dtype='float8_e4m3fn',sf_dtype='float8_e8m0fnu',c_dtype='bfloat16',sf_vec_size=32,
        load_path='cpasync',swap_ab=False,mma_tiler_mn=tuple(tile),_split_k_slices_override=1,_tile_k_override=128)
    b[4].copy_(b[3][:,:b[4].shape[1],0])
    return b[4]
def native(x,b,w,s,tile):
    quantize(x,b);return gemm(x,b,w,s,tile)

def diagnostics(x,ref,b,raw,segments):
    assert finite(b[4]) and finite(ref),'Nonfinite output'
    # Decode the exact quantizer output, then use an independent BF16/FP32 matmul.
    xq=(b[0].float()*torch.exp2(b[1].float()-127).repeat_interleave(32,1)).bfloat16()
    qref=qmm.matmul(xq,qmm.make_b16(raw))
    implementation_error=err(b[4],qref)
    assert implementation_error<.005,('Native/quantized-reference mismatch',implementation_error)
    parts={name:err(b[4][:,lo:hi],ref[:,lo:hi]) for name,lo,hi in segments}
    return dict(output_relative_l2=err(b[4],ref),activation_relative_l2=err(xq,x),
                native_vs_quantized_reference_l2=implementation_error,
                components_relative_l2=parts,finite=True)

try:
    result['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,
        gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability())
    result['source_hashes']={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (source_fp8,qmm)}
    result['b12x_dense_hash']=hashlib.sha256(Path('/b12x-source/b12x/_lib/dense_gemm.py').read_bytes()).hexdigest()
    winners={}
    for rank in (0,1):
        rd=RankReader('/exlr8-checkpoint',rank)
        for kind,layer in CASES:
            label=f'L{layer}.{kind}.proj';print(json.dumps(dict(phase='load',rank=rank,label=label)),flush=True)
            names=['self_attn.'+p+'_proj' for p in ('q','k','v','f_a','g_a','b')] if kind=='kda' else ['self_attn.q_a_proj','self_attn.kv_a_proj_with_mqa']
            names=[f'model.language_model.layers.{layer}.{n}.weight' for n in names]
            raw=torch.cat([rd.get(n) for n in names]).cuda()
            parts=[source_fp8.from_reader(rd,n) for n in names]
            segments=[];pos=0
            for name,p in zip(names,parts):segments.append((name.split('self_attn.')[1].removesuffix('.weight'),pos,pos+p.n));pos+=p.n
            q=source_fp8.cat(parts);del parts
            n,k=raw.shape;npad=triton.cdiv(n,128)*128
            w=torch.zeros((npad,k),device='cuda',dtype=torch.float8_e4m3fn)
            scales=torch.full((npad,k//32),127,device='cuda',dtype=torch.uint8)
            w[:n].copy_(q.weight.view(torch.float8_e4m3fn));scales[:n].copy_(q.scale.to(torch.uint8))
            restored=(w[:n].float()*torch.exp2(scales[:n].float()-127).repeat_interleave(32,1)).bfloat16()
            assert torch.equal(restored.view(torch.int16),raw.view(torch.int16)),'Weight repack not exact'
            del restored
            sf=pack_mxfp8_scales_for_dense_gemm(scales,m=npad,k=k)
            source_fp8.qualify(q,raw,label)
            xs=torch.load(Path(f'/quality-captures/captures-r{rank}/{label}.pt'),weights_only=True)['x'].cuda()
            def input_rows(m):return xs.repeat((m+len(xs)-1)//len(xs),1)[:m].contiguous()
            if kind not in winners:
                x=input_rows(2048);b=native_buffers(len(x),n,k);ref=source_fp8.matmul(x,q)
                stats={};checks={}
                for tile in TILES:
                    print(json.dumps(dict(phase='compile',kind=kind,tile=tile)),flush=True)
                    native(x,b,w,sf,tile)
                    checks[str(tile)]=diagnostics(x,ref,b,raw,segments)
                stats=timing({str(tile):lambda tile=tile:native(x,b,w,sf,tile) for tile in TILES})
                winner=min(TILES,key=lambda tile:stats[str(tile)]['mean_ms']);winners[kind]=winner
                result['screening'].append(dict(kind=kind,rank=rank,label=label,shape=[n,k],times=stats,checks=checks,winner=winner))
                emit(dict(phase='selected',kind=kind,tile=winner,times_ms={key:v['mean_ms'] for key,v in stats.items()}))
                del x,b,ref
            tile=winners[kind]
            # Irregular row count and changed inputs check the actual output buffer behavior.
            x=input_rows(257);gen=torch.Generator(device='cuda');gen.manual_seed(1800+layer+rank)
            x.add_(torch.randn(x.shape,device='cuda',generator=gen,dtype=torch.bfloat16)*.015625)
            b=native_buffers(len(x),n,k);ref=source_fp8.matmul(x,q)
            native(x,b,w,sf,tile);diag=diagnostics(x,ref,b,raw,segments)
            result['checks'].append(dict(rank=rank,label=label,m=257,changed_input=True,**diag));save()
            del x,b,ref
            for m in (1024,2048,4096):
                x=input_rows(m);b=native_buffers(m,n,k)
                base=torch.empty((m,n),device='cuda',dtype=torch.bfloat16);tuned=torch.empty_like(base)
                sk=qmm.b16_split_k(n,k);seq=qmm._seq(sk,m,n,k)
                part=torch.empty(sk*m*n if sk>1 and not seq else 1,device='cuda',dtype=torch.float32)
                def baseline():return source_fp8.matmul(x,q,out=base,part=part)
                def optimized():return bf16_tuned(x,source_fp8.unpack(q),tuned,part)
                def candidate():return native(x,b,w,sf,tile)
                baseline();optimized();candidate()
                assert torch.equal(base.view(torch.int16),tuned.view(torch.int16)),'BF16 tuned reference changed bits'
                diag=diagnostics(x,base,b,raw,segments)
                times=timing({'bf16_original_full':baseline,'bf16_tuned_full':optimized,'mxfp8_full':candidate,
                              'activation_quant_only':lambda:quantize(x,b),'mxfp8_gemm_and_copy':lambda:gemm(x,b,w,sf,tile)})
                row=dict(rank=rank,label=label,kind=kind,shape=[n,k],n_padded=npad,m=m,tile=tile,times=times,**diag,
                    speedup_vs_original=times['bf16_original_full']['mean_ms']/times['mxfp8_full']['mean_ms']-1,
                    speedup_vs_tuned=times['bf16_tuned_full']['mean_ms']/times['mxfp8_full']['mean_ms']-1,
                    bf16_buffer_bytes=raw.numel()*2,native_weight_bytes=w.numel()+sf.numel(),
                    original_native_weight_bytes=q.nbytes(),activation_and_output_scratch_bytes=sum(v.numel()*v.element_size() for v in b),
                    peak_gpu_bytes=torch.cuda.max_memory_allocated())
                result['rows'].append(row);emit({key:row[key] for key in ('rank','label','m','speedup_vs_tuned','output_relative_l2')})
                del x,b,base,tuned,part,baseline,optimized,candidate
            del q,raw,w,scales,sf,xs
            source_fp8._fallback=None;source_fp8._owner=None;source_fp8.ledger.clear()
            gc.collect();torch.cuda.empty_cache()
        rd.close();del rd
    result.update(complete=True,finished=time.time(),winners=winners);save()
except BaseException as e:
    result.update(error=repr(e),finished=time.time());save();raise
