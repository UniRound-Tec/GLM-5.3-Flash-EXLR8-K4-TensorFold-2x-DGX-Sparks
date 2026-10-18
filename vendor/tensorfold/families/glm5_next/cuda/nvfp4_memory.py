"""Bounded NVFP4 loading on unified-memory hosts; no serving-time polling."""
import json
import math
import os
from pathlib import Path

GIB = 1 << 30
_enabled = False
_limit = 0
_floor = 0


def host_available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024


def configure():
    global _enabled, _limit, _floor
    import torch
    if _enabled:
        return
    value = os.environ.get('TF_GLM_CUDA_LIMIT_GIB', '')
    if not value:
        raise RuntimeError('NVFP4 loading requires an explicit TF_GLM_CUDA_LIMIT_GIB')
    cap = float(value)
    floor = float(os.environ.get('TF_GLM_HOST_FLOOR_GIB', '12'))
    if not math.isfinite(cap) or not math.isfinite(floor) or cap <= 0 or floor < 6:
        raise ValueError('Invalid NVFP4 memory cap or host reserve')
    _floor = int(floor*GIB)
    # Explicit host floor plus a separate CPU/driver allowance.
    # Runtime allowance inside the CUDA cap is configured independently.
    driver = int(float(os.environ.get('TF_GLM_DRIVER_RESERVE_GIB', '2'))*GIB)
    if driver < 2*GIB:
        raise ValueError('Driver reserve must be at least 2 GiB')
    _limit = min(int(cap*GIB), torch.cuda.memory_reserved()+host_available()-_floor-driver)
    total = torch.cuda.get_device_properties(0).total_memory
    if _limit <= 0 or _limit >= total:
        raise RuntimeError('No safe NVFP4 CUDA allocation budget')
    torch.cuda.set_per_process_memory_fraction(_limit/total, 0)
    _enabled = True
    checkpoint('allocator_limit_set')


def checkpoint(stage, additional=0):
    if not _enabled:
        return
    import torch
    available = host_available()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    record = dict(stage=stage, host_available=available, allocated=allocated,
                  reserved=reserved, allocator_limit=_limit, host_floor=_floor,
                  additional=int(additional))
    print('TF_NVFP4_MEMORY '+json.dumps(record), flush=True)
    if available < _floor+additional or allocated+additional > _limit:
        raise RuntimeError('NVFP4 memory guard refused stage '+stage)


def scratch_budget(*, experts, hidden, intermediate, top_k, max_tokens, mode, limit, device='cuda:0'):
    """Pinned B12X default planning, metadata only; same inputs as preparation."""
    from types import SimpleNamespace
    import torch
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe._preparation import _query, _control_snapshot, _lower_caps
    from b12x.moe.fused_moe._tuning import _default_config
    from b12x.moe.fused_moe._impl import plan_tp_moe_scratch
    from b12x.preparation import FrozenMapping
    from b12x.preparation.device import detect_device

    plan = moe.plan_weights(source=moe.PackedSource(format='modelopt_nvfp4', w13_layout='w13'),
        activation=moe.ActivationSpec(mode=mode, nonlinearity='silu', io_dtype=torch.bfloat16,
                                     swiglu_limit=limit),
        geometry=moe.MoEGeometry(num_experts=experts, hidden_size=hidden, intermediate_size=intermediate))
    metadata = SimpleNamespace(plan=plan, num_experts=experts, hidden_size=hidden,
        intermediate_size=intermediate, _impl=SimpleNamespace(can_share_input=lambda **kw: True))
    identity = detect_device(device).identity
    if identity is None:
        raise RuntimeError('NVFP4 scratch planning requires device identity')
    size = 0
    from .nvfp4_prefill_policy import counts, select, enabled
    from dataclasses import replace
    for rows in counts(max_tokens, range(1,65)):
        capacity = moe.ExecutionCapacity(max_tokens=rows, top_k=top_k,
                                        warmup_token_counts=(rows,), route_num_experts=0)
        query = _query(metadata, capacity, rows,
            moe.RoutingSpec(apply_router_weight_on_input=False, deterministic_output=True),
            _control_snapshot(), FrozenMapping())
        if enabled():
            query = replace(query,nvfp4_inline_scales=True,shared_input_scales=False)
        config = select(query,_default_config(query, identity))
        caps = _lower_caps(query, config, plan._impl, torch.device(device))
        size = max(size, sum(s.nbytes for s in plan_tp_moe_scratch(caps, prewarm_launches=False).scratch_specs()))
    return size + max_tokens*(4*hidden + 8*top_k)


def serving_available(torch):
    """Admission must fit the real allocator ceiling and runtime headroom."""
    if not _enabled:
        return None
    reserve = int(float(os.environ.get('TF_GLM_RUNTIME_RESERVE_GIB', '4'))*GIB)
    if reserve < 2*GIB:
        raise ValueError('Runtime reserve must be at least 2 GiB')
    return max(0, _limit - int(torch.cuda.memory_reserved()) - reserve)


def serving_stats():
    """Allocation counters only: no CUDA allocation, sync or user data."""
    if not _enabled:
        return {}
    import torch
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    return dict(allocator_limit_bytes=_limit, allocated_bytes=allocated,
                reserved_bytes=reserved, allocator_room_bytes=max(0, _limit-reserved),
                host_available_bytes=host_available(),
                runtime_reserve_bytes=int(float(os.environ.get('TF_GLM_RUNTIME_RESERVE_GIB', '4'))*GIB))
