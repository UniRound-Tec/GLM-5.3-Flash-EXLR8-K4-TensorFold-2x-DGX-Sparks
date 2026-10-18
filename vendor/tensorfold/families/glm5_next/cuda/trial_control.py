"""Bounded A/B banks: switch only between requests, mirrored by the TP protocol."""
import copy
import json
import os
from pathlib import Path

BOOLS = ('gr8', 'kda', 'taps', 'asyncmsg', 'sampler', 'prefill_taps', 'index_split', 'chunk8k', 'verify8', 'shared_fusion', 'mx_prefill')
BASE = {'gr8': False, 'quant': 'minmax', 'kda': False, 'taps': False, 'prefill': 0,
        'asyncmsg': False, 'sampler': False, 'prefill_taps': False,
        'index_split': False, 'chunk8k': False, 'mla_variant': 0, 'spec': 0, 'draft_noise': 0.7, 'pf': 0, 'd30': 0, 'draft_checkpoint': 31, 'verify8': False, 'rrs_k': 0, 'expert_order': 0, 'rrs_mode': 0, 'fnc_zero': 0, 'shared_fusion': False, 'mx_prefill': False}
ACTIVE = dict(BASE)
VERIFY8_CAPTURE = False
_alternates = {}
_quant_bytes = 0

def enabled():
    return os.environ.get('TF_GLM_TRIAL_BANKS', '0') == '1'

def parse(value=None):
    if value is None:
        path = os.environ.get('TF_GLM_TRIAL_DEFAULT_FILE')
        value = json.loads(Path(path).read_text()) if path else BASE
    if not isinstance(value, dict) or set(value) - set(BASE):
        raise ValueError('tf_trials has unknown fields')
    out = {**BASE, **value}
    if out['quant'] not in ('minmax', 'mse') or any(type(out[k]) is not bool for k in BOOLS):
        raise ValueError('tf_trials quant is minmax or mse; kda and taps are booleans')
    if type(out['prefill']) is not int or out['prefill'] not in (0, 273, 274, 289, 290, 272, 288, 257, 258):
        raise ValueError('invalid prefill variant')
    if type(out['mla_variant']) is not int or out['mla_variant'] not in (0, 1):
        raise ValueError('invalid MLA variant')
    if type(out['spec']) is not int or out['spec'] not in (0,1,2):
        raise ValueError('spec: 0 off, 1 timing, 2 timing and candidates')
    if type(out['pf']) is not int or out['pf'] not in (0,1,2):
        raise ValueError('pf: 0 original, 1 decode tensors only, 2 dummy prefetch')
    if type(out['d30']) is not int or out['d30'] not in range(8):
        raise ValueError('d30 must be an integer mask 0..7')
    if type(out['draft_checkpoint']) is not int or out['draft_checkpoint'] not in (27,28,31):
        raise ValueError('draft_checkpoint must be 27, 28 or 31')
    if out['draft_checkpoint'] != 31 and out['quant'] != 'mse':
        raise ValueError('Historical checkpoint banks use MSE Q4 only')
    if out['verify8'] and (out['d30'] != 7 or out['gr8'] or out['kda']):
        raise ValueError('verify8 requires d30=7, gr8=false and kda=false')
    if type(out['rrs_k']) is not int or out['rrs_k'] not in range(8):
        raise ValueError('rrs_k must be 0..7')
    if type(out['expert_order']) is not int or out['expert_order'] not in (0,1,2):
        raise ValueError('expert_order: 0 old, 1 N first, 2 expert first')
    if out['expert_order'] and not out['verify8']:
        raise ValueError('expert_order requires verify8')
    if out['rrs_k'] and (out['spec'] or out['draft_noise'] <= 0):
        raise ValueError('RRS requires spec=0 and positive draft_noise')
    if type(out['rrs_mode']) is not int or out['rrs_mode'] not in (0,1,2,3):
        raise ValueError('rrs_mode: 0 fixed, 1 inclusive, 2 pilot, 3 inclusive native edge')
    if out['rrs_mode'] and not out['rrs_k']:
        raise ValueError('rrs_mode needs positive rrs_k')
    if type(out['fnc_zero']) is not int or out['fnc_zero'] not in (0,100,150,200,250,300,350):
        raise ValueError('fnc_zero must be 0/100/150/200/250/300/350 thousandths')
    if out['fnc_zero'] and out['rrs_k']:
        raise ValueError('fnc_zero only applies to original FNC')
    if out['shared_fusion'] and (not out['verify8'] or out['expert_order'] != 2):
        raise ValueError('shared_fusion requires verify8 and expert_order=2')
    noise = out['draft_noise']
    if type(noise) not in (float, int) or not 0 <= noise <= 1.5 or abs(noise * 1000 - round(noise * 1000)) > 1e-8:
        raise ValueError('draft_noise: 0..1.5 in steps of .001')
    out['draft_noise'] = round(noise * 1000) / 1000
    if not enabled() and out != BASE:
        raise ValueError('trial banks are not enabled')
    return out

def pack(value):
    value = parse(value)
    return [int(value['quant'] == 'mse'), int(value['kda']), int(value['taps']),
            value['prefill'], int(value['asyncmsg']), int(value['sampler']), int(value['prefill_taps']), int(value['index_split']), int(value['chunk8k']), value['mla_variant'], value['spec'], int(round(value['draft_noise'] * 1000)), int(value['gr8']), value['pf'], value['d30'], value['draft_checkpoint'], int(value['verify8']), value['rrs_k'], value['expert_order'], value['rrs_mode'], value['fnc_zero'], int(value['shared_fusion']), int(value['mx_prefill'])]

def unpack(values):
    if len(values) == 3:
        values = [*values, 0, 0, 0, 0]
    if len(values) == 7:
        values = [*values, 0, 0, 0]
    if len(values) == 10:
        values = [*values, 0]
    if len(values) == 11:
        values = [*values, 700]
    if len(values) == 12:
        values = [*values, 0]
    if len(values) == 13:
        values = [*values, 0]
    if len(values) == 14:
        values = [*values,0]
    if len(values) == 15:
        values = [*values,31]
    if len(values) == 16:
        values = [*values,0]
    if len(values) == 17:
        values = [*values,0,0]
    if len(values) == 19:
        values = [*values,0,0]
    if len(values) == 21:
        values = [*values,0]
    if len(values) == 22:
        values = [*values,0]
    if len(values) != 23 or any(v not in (0, 1) for i,v in enumerate(values) if i not in (3,10,11,13,14,15,17,18,19,20)) or values[10] not in (0,1,2) or type(values[11]) is not int or not 0 <= values[11] <= 1500:
        raise ValueError('invalid TRIAL payload')
    return parse(dict(quant='mse' if values[0] else 'minmax', kda=bool(values[1]), taps=bool(values[2]),
                      prefill=values[3], asyncmsg=bool(values[4]), sampler=bool(values[5]), prefill_taps=bool(values[6]), index_split=bool(values[7]),
                      chunk8k=bool(values[8]), mla_variant=values[9], spec=values[10], draft_noise=values[11] / 1000, gr8=bool(values[12]), pf=values[13], d30=values[14], draft_checkpoint=values[15], verify8=bool(values[16]), rrs_k=values[17], expert_order=values[18], rrs_mode=values[19], fnc_zero=values[20], shared_fusion=bool(values[21]), mx_prefill=bool(values[22])))

def dual_quantize(w):
    global _quant_bytes
    from .qmm import quantize4, quantize4_mse
    base = quantize4(w)
    alt = quantize4_mse(w)
    _alternates[id(base)] = alt
    _quant_bytes += sum(t.numel() * t.element_size() for t in (alt.weight, alt.scales, alt.biases))
    return base

def _alt(q):
    from .qmm import Q4
    if not isinstance(q, Q4):
        return q
    if id(q) not in _alternates:
        raise RuntimeError('a draft matrix has no MSE bank')
    return _alternates[id(q)]

def _set_weights(m, bank):
    d = m.drafts.d
    d.fc, d.layers, m.w.draft_head = bank

def prepare(m):
    """Capture both paths before admission; their persistent buffers never change."""
    if not enabled():
        return
    global VERIFY8_CAPTURE
    import torch
    from . import trial_kernels
    assert not m.lanes and not m.tune.lone
    assert m.drafts is not None and m.verify.graphs
    from . import prefetch_filter
    prefetch_filter.install(m.w.meta.get('l2pf'),m.rank)
    d = m.drafts.d
    base = (d.fc, d.layers, m.w.draft_head)
    alternate_layers = []
    for layer in d.layers:
        alt = copy.copy(layer)
        for name, q in vars(layer).items():
            setattr(alt, name, _alt(q))
        alternate_layers.append(alt)
    alternate = (_alt(d.fc), alternate_layers, _alt(m.w.draft_head))
    m.trial_weights = {'minmax': base, 'mse': alternate}
    md = m.drafts
    m.trial_drafts = {'minmax': (md.block_graphs, md.tap_graphs, md.pool)}
    _set_weights(m, alternate)
    md.block_graphs, md.tap_graphs = {}, {}
    md.capture()
    m.trial_drafts['mse'] = (md.block_graphs, md.tap_graphs, md.pool)
    _set_weights(m, base)
    md.block_graphs, md.tap_graphs, md.pool = m.trial_drafts['minmax']
    m.trial_verify = {False: m.verify.graphs}
    # Only the KDA projection pair changes graph computation. Tap packing is outside it.
    trial_kernels.extension()
    ACTIVE['kda'] = True
    m.verify.graphs = {}
    m.verify.capture()
    m.trial_verify[True] = m.verify.graphs
    m.trial_gr8 = {}
    from . import gr8_adapter
    gr8_adapter.extension()
    for kda in (False,True):
        ACTIVE.update(kda=kda,gr8=True)
        m.verify.graphs=dict(m.trial_verify[kda])
        m.verify.capture(rows=[1])
        m.trial_gr8[kda]=m.verify.graphs
    # Only the one-row graph changes. All larger verification windows retain their original graph.
    m.trial_d30 = {}
    for mask in (0,2,4,6):
        ACTIVE.update(BASE)
        ACTIVE['d30'] = mask
        m.verify.graphs = dict(m.trial_verify[False])
        if mask:
            m.verify.capture(rows=[1])
        regular = m.verify.graphs[1]
        m.verify.capture(rows=[1], select=False)
        short = m.verify.graphs[1]
        m.verify.graphs[1] = regular
        m.verify.graphs[-1] = short
        m.trial_d30[mask] = m.verify.graphs
    # One candidate bank extends eligible d30 work to 2..8 rows. Original R1 and R9+ graphs stay shared.
    before_mem = torch.cuda.memory_allocated()
    ACTIVE.update(BASE)
    ACTIVE.update(d30=7, verify8=True)
    VERIFY8_CAPTURE = True
    try:
        m.verify.graphs = dict(m.trial_d30[6])
        m.verify.capture(rows=range(2,9))
        regular = {r:m.verify.graphs[r] for r in range(2,9)}
        m.verify.capture(rows=range(2,9), select=False)
        for r in range(2,9):
            m.verify.graphs[-r] = m.verify.graphs[r]
            m.verify.graphs[r] = regular[r]
        m.trial_verify8 = m.verify.graphs
    finally:
        VERIFY8_CAPTURE = False
    torch.cuda.synchronize()
    print('TF_VERIFY8_READY '+json.dumps({'rank':m.rank,'rows':list(range(2,9)),
          'head_rows':[2,3,4],'extra_allocated_bytes':torch.cuda.memory_allocated()-before_mem,
          'graphs':len(m.trial_verify8),'short_guard':'all row positions < SPARSE_FROM'}),flush=True)
    # Small proven bitwise candidates, captured in the same load for request A/B.
    from . import expert_order
    expert_order.extension(1)
    expert_order.extension(2)
    m.trial_expert_order = {0: m.trial_verify8}
    before_order = torch.cuda.memory_allocated()
    VERIFY8_CAPTURE = True
    try:
        for order in (1,2):
            ACTIVE.update(d30=7, verify8=True, expert_order=order)
            m.verify.graphs = dict(m.trial_verify8)
            m.verify.capture(rows=range(2,9))
            regular = {r:m.verify.graphs[r] for r in range(2,9)}
            m.verify.capture(rows=range(2,9), select=False)
            for r in range(2,9):
                m.verify.graphs[-r] = m.verify.graphs[r]
                m.verify.graphs[r] = regular[r]
            m.trial_expert_order[order] = m.verify.graphs
    finally:
        VERIFY8_CAPTURE = False
    torch.cuda.synchronize()
    print('TF_EXPERT_ORDER_READY '+json.dumps({'rank':m.rank,'orders':[0,1,2],
          'extra_allocated_bytes':torch.cuda.memory_allocated()-before_order}),flush=True)
    from . import shared_fusion
    shared_fusion.extension()
    before_shared = torch.cuda.memory_allocated()
    ACTIVE.update(d30=7,verify8=True,expert_order=2,shared_fusion=True)
    VERIFY8_CAPTURE = True
    try:
        m.verify.graphs = dict(m.trial_expert_order[2])
        m.verify.capture(rows=range(2,9))
        regular = {r:m.verify.graphs[r] for r in range(2,9)}
        m.verify.capture(rows=range(2,9),select=False)
        for r in range(2,9):
            m.verify.graphs[-r] = m.verify.graphs[r]
            m.verify.graphs[r] = regular[r]
        m.trial_shared_fusion = m.verify.graphs
    finally:
        VERIFY8_CAPTURE = False
    torch.cuda.synchronize()
    print('TF_SHARED_FUSION_READY '+json.dumps({'rank':m.rank,'layers':len(shared_fusion.ledger),
       'extra_weight_bytes':sum(x['extra_weight_bytes'] for x in shared_fusion.ledger),
       'extra_allocated_bytes':torch.cuda.memory_allocated()-before_shared,'rows':list(range(2,9))}),flush=True)
    ACTIVE.update(BASE)
    m.verify.graphs = m.trial_verify[False]
    from . import draft_checkpoints
    draft_checkpoints.prepare(m)
    from . import mx_prefill
    mx_prefill.prepare()
    print('TF_MX_PREFILL_READY '+json.dumps({'rank':m.rank,'rows':[1024,4096],'extra_resident_weight_bytes':0,'extra_scratch_bytes':0}),flush=True)
    m.trial_active = dict(BASE)
    torch.cuda.synchronize()
    print('TF_TRIAL_READY '+json.dumps({'rank': m.rank, 'draft_extra_bytes': _quant_bytes,
          'verify_graphs': {str(k): len(v) for k,v in m.trial_verify.items()},
          'draft_graphs': {k: [len(v[0]),len(v[1])] for k,v in m.trial_drafts.items()}}), flush=True)

def activate(m, value):
    value = parse(value)
    if value == getattr(m, 'trial_active', BASE):
        return
    if m.lanes:
        raise RuntimeError('trial changes require an empty scheduler')
    import torch
    torch.cuda.synchronize()
    from . import prefetch_filter
    prefetch_filter.activate(m.w.meta.get('l2pf'),value['pf'])
    _set_weights(m, m.trial_weights[value['quant']])
    from . import draft_checkpoints
    draft_checkpoints.activate(m,value['draft_checkpoint'],value['quant'])
    m.verify.graphs = (m.trial_gr8 if value['gr8'] else m.trial_verify)[value['kda']]
    if value['d30'] and not value['gr8'] and not value['kda']:
        m.verify.graphs = m.trial_d30[value['d30'] & 6]
    if value['verify8']:
        m.verify.graphs = m.trial_expert_order[value['expert_order']]
    if value['shared_fusion']:
        m.verify.graphs = m.trial_shared_fusion
    if value['mla_variant']:
        from .atlas_mla import library
        library()
    if value['chunk8k'] and m.e.pbuf.rows < 8192:
        raise RuntimeError('8192-row buffers were not admitted')
    m.e.prefill_rows = 8192 if value['chunk8k'] else 4096
    ACTIVE.update(value)
    m.trial_active = dict(value)

def settings(s):
    return parse((getattr(s, 'glm', None) or {}).get('trial'))

def tag():
    return ACTIVE['quant'] + ':pf4:' + ':'.join(str(ACTIVE[k]) for k in ('index_split', 'mla_variant', 'chunk8k', 'gr8', 'd30', 'draft_checkpoint', 'verify8', 'rrs_k', 'expert_order', 'rrs_mode', 'fnc_zero', 'shared_fusion', 'mx_prefill'))

def projections(b, fa, ga, fb, gb, R, fallback):
    from . import qmm
    if ACTIVE['kda'] and R <= 32 and isinstance(fb, qmm.Q4) and isinstance(gb, qmm.Q4):
        from .trial_kernels import kda_pair
        kda_pair(fa, ga, fb, gb, b.xs_fa[:R], b.xs_ga[:R], b.ka[:R], b.kg[:R])
    else:
        fallback(b, fa, fb, qmm.group_sums_for(fb, fa, b.xs_fa[:R]), b.ka[:R])
        fallback(b, ga, gb, qmm.group_sums_for(gb, ga, b.xs_ga[:R]), b.kg[:R])

def taps(parts):
    if ACTIVE['taps']:
        from .trial_kernels import TapParts
        return TapParts(parts)
    import torch
    return torch.cat(parts, dim=1)
