"""Request-opt-in bounded DFlash diagnostics. No recording for ordinary requests."""
import json
import os
from pathlib import Path
import time

CURRENT = None
MAX_ROUNDS = 512

def begin(m, lanes):
    global CURRENT
    from .trial_control import ACTIVE
    mode = ACTIVE.get('spec', 0)
    CURRENT = None
    if not mode or len(lanes) != 1:
        return
    lane = lanes[0]
    records = getattr(lane, 'spec_records', None)
    if records is None:
        records = lane.spec_records = []
    if len(records) >= MAX_ROUNDS:
        return
    CURRENT = dict(mode=mode, t0=time.perf_counter(), times={}, choices=[], lane=lane, rank=m.rank)

def mark(name):
    if CURRENT is not None:
        CURRENT['times'][name] = (time.perf_counter() - CURRENT['t0']) * 1000

def block_start():
    if CURRENT is not None:
        import torch
        CURRENT['events'] = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        CURRENT['events'][0].record()
        mark('block_submit')

def block_end():
    if CURRENT is not None:
        CURRENT['events'][1].record()

def transfer_end():
    if CURRENT is not None:
        CURRENT['events'][2].record()

def synced():
    if CURRENT is not None:
        ev = CURRENT['events']
        # The original candidate-ready synchronization already completed these events.
        CURRENT['gpu_block_ms'] = ev[0].elapsed_time(ev[1])
        CURRENT['gpu_copy_ms'] = ev[1].elapsed_time(ev[2])
        mark('candidates_ready')

def choice(tokens, values, edge, score, noise, picked):
    if CURRENT is not None and CURRENT['mode'] == 2:
        CURRENT['choices'].append(dict(tokens=tokens.tolist(), unary=values.tolist(), edge=edge.tolist(),
              score=score.tolist(), noise=None if noise is None else noise.tolist(), picked=int(picked)))

def end(m, lanes, segments, sampled, keeps, copied):
    global CURRENT
    c = CURRENT
    CURRENT = None
    if c is None:
        return
    c['times']['end'] = (time.perf_counter()-c['t0'])*1000
    seg, rows, keep = segments[0], sampled[0], keeps[0]
    row = dict(times_ms=c['times'], gpu_block_ms=c.get('gpu_block_ms'), gpu_copy_ms=c.get('gpu_copy_ms'),
               copied=bool(copied[0]), proposed=len(seg.tokens)-1, keep=int(keep),
               choices=c['choices'], sampled=list(rows), drafts=list(seg.tokens[1:]))
    # Only samples through the first mismatch have the actual accepted prefix.
    row['valid_observations'] = min(int(keep), len(rows), len(c['choices']))
    c['lane'].spec_records.append(row)

def finish(m, lane):
    records = getattr(lane, 'spec_records', None)
    if not records:
        return
    p = Path(os.environ.get('TF_GLM_SPEC_DIR', '/tmp/glm-spec'))
    p.mkdir(parents=True, exist_ok=True)
    name = f'r{m.rank}-s{lane.sid}-{time.time_ns()}.json'
    data = dict(rank=m.rank, sid=lane.sid, mode=getattr(lane.s, 'glm_trial', {}).get('spec'),
                prompt_tokens=len(lane.s.prompt), policy=lane.spec, costs=m.g.costs, records=records)
    (p/name).write_text(json.dumps(data, separators=(',', ':')))
    lane.s.glm_spec = dict(file=name, rounds=len(records), limit=MAX_ROUNDS)

