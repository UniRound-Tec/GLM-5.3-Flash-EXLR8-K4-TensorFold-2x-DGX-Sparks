"""Measured SM121 GLM prefill tiles; no serving-time tuning."""
from dataclasses import replace
import os

TILES={128:16,256:32,512:32,1024:32,2048:64,4096:128,8192:128}

def enabled():
    return os.environ.get('TF_GLM_PREFILL_VARIANT','')=='fused64'

def counts(max_tokens,fixed):
    extra=tuple(n for n in TILES if n<=max_tokens) if enabled() else ()
    return tuple(sorted({max_tokens,*fixed,*extra}))

def select(q,config):
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    if (enabled() and q.num_tokens in TILES
        and (q.num_experts,q.hidden_size,q.intermediate_size,q.top_k)==(288,4096,1024,8)
        and q.quant_mode=='nvfp4' and q.activation=='silu' and q.io_dtype=='bfloat16'
        and q.deterministic_output is True and not q.apply_router_weight_on_input):
        return MoeDecodeConfig(backend='dynamic',route_planner='internal',max_active_clusters=None,
            dynamic_tile_m=TILES[q.num_tokens],dynamic_route_mode='grouped',
            nvfp4_share_input=False,nvfp4_inline_scales=q.nvfp4_inline_scales)
    return config

def apply(plan,*,geometry,device):
    import torch
    if not enabled() or tuple(geometry)!=(288,4096,1024,8) or torch.cuda.get_device_capability(device)!=(12,1):
        return plan
    return replace(plan,variants={n:replace(p,override=select(p.query,p.override))
        if n in TILES else p for n,p in plan.variants.items()})
