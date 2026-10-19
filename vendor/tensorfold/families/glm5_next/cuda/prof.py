"""TF_GLM_PROFILE=1: time (with device syncs) and memory of each block kind of every prompt chunk, printed after each prefill; never inside decode or capture."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch

ENABLED = os.environ.get("TF_GLM_PROFILE", "0") == "1"
active = False                     # set only around prefill chunks
totals: dict[str, float] = {}
peaks: dict[str, int] = {}         # the most memory a block allocated on top of what was live before it


@contextmanager
def timed(name: str):
    if event_active:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            event_pairs.append((name,start,end))
        return
    if not (ENABLED and active):
        yield
        return
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    yield
    torch.cuda.synchronize()
    totals[name] = totals.get(name, 0.0) + time.perf_counter() - t
    peaks[name] = max(peaks.get(name, 0), torch.cuda.max_memory_allocated() - before)


def report(tokens: int) -> None:
    if not ENABLED or not totals:
        return
    total = sum(v for k, v in totals.items() if ":" not in k)          # "dsa: x" parts are inside "dsa (total)"
    parts = ", ".join(f"{k} {v:.2f}s ({100 * v / total:.0f}%)" for k, v in sorted(totals.items(), key=lambda kv: -kv[1]))
    print(f"[tensorfold] prefill profile, {tokens} tokens in {total:.1f}s timed ({tokens / total:.0f} tok/s): {parts}",
          flush=True)
    big = ", ".join(f"{k} {v / 2**30:.2f}" for k, v in sorted(peaks.items(), key=lambda kv: -kv[1])[:6])
    print(f"[tensorfold] prefill memory (GiB): allocated {torch.cuda.memory_allocated() / 2**30:.1f}, reserved "
          f"{torch.cuda.memory_reserved() / 2**30:.1f}; most a block added: {big}", flush=True)
    totals.clear()
    peaks.clear()


# Diagnostic only: opt-in per request, main-stream events, one synchronization per chunk.
# Nested intervals overlap; these measurements are not used as benchmark throughput.
event_active = False
event_pairs = []

@contextmanager
def event_scope(rank, rows):
    global event_active
    from . import trial_control
    if not trial_control.ACTIVE.get('prefill_profile', False):
        yield
        return
    if event_active:
        raise RuntimeError('nested prefill profile scope')
    event_pairs.clear()
    event_active = True
    try:
        yield
    finally:
        event_active = False
        torch.cuda.synchronize()
        values = {}
        for name,start,end in event_pairs:
            v = values.setdefault(name, {'ms':0.0,'count':0})
            v['ms'] += start.elapsed_time(end)
            v['count'] += 1
        import json
        print('TF_PREFILL_EVENTS '+json.dumps({'rank':rank,'rows':rows,'schedule':trial_control.ACTIVE.get('prefill_schedule',0),'parts':values}),flush=True)
        event_pairs.clear()
