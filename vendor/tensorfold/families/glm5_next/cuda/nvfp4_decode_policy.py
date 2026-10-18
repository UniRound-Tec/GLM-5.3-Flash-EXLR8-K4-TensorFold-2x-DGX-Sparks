"""Measured SM121 GLM A4 decode plans; preserve the larger-window planner."""
from dataclasses import replace


def apply(plan, *, geometry, device):
    import torch
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    if tuple(geometry) != (288,4096,1024,8) or torch.cuda.get_device_capability(device) != (12,1):
        return plan
    variants={}
    for rows,child in plan.variants.items():
        q=child.query
        if (1 <= rows <= 7 and q.quant_mode=='nvfp4' and q.activation=='silu'
                and q.io_dtype=='bfloat16' and q.deterministic_output is True
                and not q.apply_router_weight_on_input):
            # Real TP shards: native A4 + inline scales beats direct micro.
            # Preserve per-expert activation scales and fixed-order reduction.
            config=MoeDecodeConfig(backend='dynamic',route_planner='internal',
                max_active_clusters=None,dynamic_tile_m=16,dynamic_route_mode='grouped',
                nvfp4_share_input=False,nvfp4_inline_scales=q.nvfp4_inline_scales)
            child=replace(child,override=config)
        variants[rows]=child
    return replace(plan,variants=variants)
