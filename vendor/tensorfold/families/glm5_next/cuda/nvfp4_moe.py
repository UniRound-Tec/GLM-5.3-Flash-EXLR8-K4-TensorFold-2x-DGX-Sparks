"""Standalone B12X MoE adapter following vLLM's prepared NVFP4 execution path.

QAD routing stays in TensorFold. This adapter consumes its already normalized,
scaled top-k probabilities. The model owns the CSF expansion planes; every
forward buffer owns its route copies and output. Target MoE launches are serial
on TensorFold's compute stream, so they share one execution scratch allocation,
just as they share the CSF expansion planes.
"""
from dataclasses import dataclass
import json
import os
import torch

from .nvfp4_csf_loader import read_nvfp4_csf_layer
from .nvfp4_memory import checkpoint


@dataclass
class WarmupStorage:
    x: torch.Tensor
    output: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    scratch: torch.Tensor

    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in vars(self).values())


@dataclass
class Nvfp4Experts:
    prepared: object
    layer: int
    plan: object = None
    atlas: object = None

    def tensors(self):
        """Expose owned storage to TensorFold's memory report, without cycles."""
        from dataclasses import fields, is_dataclass
        seen = set()
        def walk(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                yield value
            elif is_dataclass(value):
                for field in fields(value):
                    if field.name != 'plan':
                        yield from walk(getattr(value, field.name))
            elif isinstance(value, dict):
                for child in value.values():yield from walk(child)
            elif isinstance(value, (list,tuple)):
                for child in value:yield from walk(child)
        yield from walk(self.prepared)
        yield from walk(self.atlas)


class Nvfp4Runtime:
    def __init__(self, *, experts, hidden, intermediate, top_k, device, mode='a4', limit=10.0):
        from b12x.moe import fused_moe as moe
        from b12x.preparation import PreparationSession
        if mode not in ('a4', 'a16'):
            raise ValueError('NVFP4 activation mode must be a4 or a16')
        self.experts, self.hidden, self.intermediate, self.top_k = experts, hidden, intermediate, top_k
        self.device = torch.device(device)
        self.weight_plan = moe.plan_weights(
            source=moe.PackedSource(format='modelopt_nvfp4', w13_layout='w13'),
            activation=moe.ActivationSpec(mode=mode, nonlinearity='silu',
                io_dtype=torch.bfloat16, swiglu_limit=limit),
            geometry=moe.MoEGeometry(num_experts=experts, hidden_size=hidden,
                                    intermediate_size=intermediate))
        self.scales = (
            torch.empty((experts, 2*intermediate, hidden//16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((experts, hidden, intermediate//16), dtype=torch.float8_e4m3fn, device=device))
        self.session = PreparationSession(device=device, autotune=False, compile_workers=0)
        self.layers = []
        self.capacity = 0
        self.scratch = None
        self.warmup = None
        self.atlas = None

    def load_layer(self, root, layer, *, rank, world):
        from b12x.moe import fused_moe as moe
        raw = read_nvfp4_csf_layer(root, layer, num_experts=self.experts,
            hidden_size=self.hidden, intermediate_size=self.intermediate*world,
            tp_rank=rank, tp_size=world, device=self.device,
            w13_scale_scratch=self.scales[0], w2_scale_scratch=self.scales[1])
        return self.add_weights(raw, layer=layer)

    def add_weights(self, raw, *, layer):
        from b12x.moe import fused_moe as moe
        result = Nvfp4Experts(moe.prepare_weights(plan=self.weight_plan, weights=raw), layer)
        self.layers.append(result)
        return result

    def prepare(self, max_tokens, fixed_counts):
        from b12x.moe import fused_moe as moe
        from .nvfp4_prefill_policy import counts as prefill_counts
        counts = prefill_counts(max_tokens, fixed_counts)
        if not self.layers or min(counts) < 1 or max(counts) > max_tokens:
            raise ValueError('Invalid NVFP4 preparation capacity')
        if self.scratch is not None:
            raise RuntimeError('NVFP4 preparation is single-use')
        self.capacity = max_tokens
        for owner in self.layers:
            owner.plan = moe.plan_execution(experts=owner.prepared,
                capacity=moe.ExecutionCapacity(max_tokens=max_tokens, top_k=self.top_k,
                    warmup_token_counts=counts, route_num_experts=0),
                routing=moe.RoutingSpec(apply_router_weight_on_input=False,
                                        deterministic_output=True))
        from .nvfp4_decode_policy import apply as apply_decode_policy
        for owner in self.layers:
            owner.plan = apply_decode_policy(owner.plan,
                geometry=(self.experts,self.hidden,self.intermediate,self.top_k),device=self.device)
        from .nvfp4_prefill_policy import apply as apply_prefill_policy
        for owner in self.layers:
            owner.plan = apply_prefill_policy(owner.plan,
                geometry=(self.experts,self.hidden,self.intermediate,self.top_k),device=self.device)
        # scratch_specs() sizes the default recipe without materializing it.
        # autotune=False ensures preparation uses that same recipe. All MoEs
        # execute serially; retained PreparedCall owners must share storage too.
        size = max(sum(s.nbytes for s in owner.plan.scratch_specs()) for owner in self.layers)
        io_bytes = max_tokens*(4*self.hidden + 8*self.top_k)
        checkpoint('nvfp4_warmup_allocate', additional=size+io_bytes)
        self.scratch = torch.empty(size, device=self.device, dtype=torch.uint8)
        self.warmup = WarmupStorage(
            torch.full((max_tokens, self.hidden), 0.125, dtype=torch.bfloat16, device=self.device),
            torch.empty((max_tokens, self.hidden), dtype=torch.bfloat16, device=self.device),
            torch.arange(max_tokens*self.top_k, dtype=torch.int32, device=self.device)
                 .remainder_(self.experts).reshape(max_tokens, self.top_k),
            torch.full((max_tokens, self.top_k), 1/self.top_k, device=self.device, dtype=torch.float32),
            self.scratch)
        for owner in self.layers:
            checkpoint(f'nvfp4_warmup_layer_{owner.layer}', additional=256 << 20)
            factories = {n: self._factory(n) for n in owner.plan.token_counts}
            self.session.prepare((owner.plan.request(name=f'glm_nvfp4_layer_{owner.layer}',
                                                     prepare_calls=factories),))
        self.session.freeze()
        print('TF_NVFP4_SHARED_MEMORY '+json.dumps({'layers':len(self.layers), 'variants':len(counts),
              'scratch_bytes':size, 'warmup_and_execution_bytes':self.warmup.nbytes}), flush=True)
        from b12x.moe.fused_moe._impl import _tp4_tile_major_lookup
        layouts = [_tp4_tile_major_lookup(o.prepared._impl.w1_fp4, o.prepared._impl.w2_fp4) is not None for o in self.layers]
        print('TF_NVFP4_LAYOUT '+json.dumps({'name':'tile_major_v1', 'layers':len(layouts),
              'tile_major_layers':sum(layouts), 'weight_values_changed':False}),flush=True)
        first = self.layers[0].plan
        selected = {str(n): {'backend': p.selection.config.backend,
                    'planner': p.selection.config.route_planner,
                    'inline_scales': p.selection.config.nvfp4_inline_scales}
                    for n,p in first.variants.items() if n in (1,7,8,16,32,64,max_tokens)}
        print('TF_NVFP4_DECODE_POLICY '+json.dumps({'name':'sm121_glm_v1',
              'geometry':[self.experts,self.hidden,self.intermediate,self.top_k],
              'shapes':selected}),flush=True)
        print('TF_NVFP4_PREFILL_POLICY '+json.dumps({'name':'sm121_prefill_v1',
              'capacity':max_tokens,'shapes':{str(n):{'tile_m':p.selection.config.dynamic_tile_m,
              'inline_scales':p.selection.config.nvfp4_inline_scales}
              for n,p in first.variants.items() if n>64}}),flush=True)
        if os.environ.get('TF_GLM_ATLAS_PREFILL') == '1':
            from .atlas_adapter import Workspace,prepare,scratch_bytes
            if (self.experts,self.hidden,self.intermediate,self.top_k)!=(288,4096,1024,8):
                raise RuntimeError('Atlas prefill requires the tested GLM TP2 geometry')
            if max_tokens>8192 or scratch_bytes(max_tokens)>self.scratch.numel():
                raise RuntimeError('Atlas prefill exceeds the admitted shared scratch')
            from .atlas_selection import DECODE_MODES,PREFILL_MODE
            self.atlas=Workspace(self.scratch,max_tokens,modes=DECODE_MODES,prefill_mode=PREFILL_MODE)
            for owner in self.layers:owner.atlas=prepare(owner)
            self.atlas.run(self.layers[0].atlas,self.warmup.x,self.warmup.ids,
                           self.warmup.weights,self.warmup.output)
            print('TF_ATLAS_DECODE '+json.dumps({'layers':len(self.layers),'decode_modes':DECODE_MODES,'prefill_mode':PREFILL_MODE,'weight_copy_bytes':0}),flush=True)
            print('TF_ATLAS_PREFILL '+json.dumps({'upstream':'72588541a466',
                'layers':len(self.layers),'capacity':max_tokens,'minimum_rows':128,
                'shared_scratch_bytes':scratch_bytes(max_tokens),'weight_copy_bytes':0}),flush=True)
        checkpoint('nvfp4_warmup_complete')

    def _factory(self, rows):
        from b12x.preparation import PreparedCall
        def factory(state):
            shared = self.warmup
            needed = sum(s.nbytes for s in state.scratch.scratch_specs())
            if needed > shared.scratch.numel():
                raise RuntimeError('Prepared NVFP4 scratch exceeds its admitted shared allocation')
            x, out = shared.x[:rows], shared.output[:rows]
            ids, weights = shared.ids[:rows], shared.weights[:rows]
            binding = state.bind(a=x, topk_ids=ids, topk_weights=weights, output=out,
                                 scratch=shared.scratch, input_scales_static=True)
            return PreparedCall(run=lambda: state.run(binding), output=out,
                                owners=(shared, binding))
        return factory

    def buffers(self, rows):
        return Nvfp4Buffers(self, rows)

    def close(self):
        self.session.close()


class Nvfp4Buffers:
    def __init__(self, runtime, rows):
        if rows > runtime.capacity or any(x.plan is None for x in runtime.layers):
            raise ValueError('Prepare all NVFP4 layers before allocating execution buffers')
        self.runtime, self.rows = runtime, rows
        self.ids = torch.empty((rows,runtime.top_k), device=runtime.device, dtype=torch.int32)
        self.probabilities = torch.empty((rows,runtime.top_k), device=runtime.device, dtype=torch.float32)
        self.output = torch.empty((rows,runtime.hidden), device=runtime.device, dtype=torch.bfloat16)
        self.scratch = runtime.scratch

    def run(self, owner, x, ids, probabilities):
        from b12x.moe import fused_moe as moe
        rows = x.shape[0]
        if rows > self.rows or x.shape[1] != self.runtime.hidden:
            raise ValueError('NVFP4 input exceeds its allocated buffer')
        if x.dtype != torch.bfloat16 or ids.shape != probabilities.shape or ids.shape != (rows,self.runtime.top_k):
            raise ValueError('Unexpected NVFP4 input or route contract')
        self.ids[:rows].copy_(ids)
        self.probabilities[:rows].copy_(probabilities)
        if self.runtime.atlas is not None and (rows>=128 or rows in self.runtime.atlas.decode_modes):
            return self.runtime.atlas.run(owner.atlas,x,self.ids[:rows],
                                         self.probabilities[:rows],self.output[:rows])
        binding = moe.bind(owner.plan, scratch=self.scratch, a=x, experts=owner.prepared,
            topk_ids=self.ids[:rows], topk_weights=self.probabilities[:rows],
            output=self.output[:rows], input_scales_static=True)
        moe.run(binding=binding)
        return self.output[:rows]
