"""Metadata-only planning for bounded, TP-consistent prompt snapshots."""
from __future__ import annotations

import math
import os


def read_budget(value: str | None = None) -> int:
    value = os.environ.get('TF_GLM_SNAPSHOT_GIB', '0') if value is None else value
    try:
        gib = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('TF_GLM_SNAPSHOT_GIB must be finite and nonnegative') from exc
    if not math.isfinite(gib) or gib < 0:
        raise ValueError('TF_GLM_SNAPSHOT_GIB must be finite and nonnegative')
    return int(gib * (1 << 30))


def max_copy_bytes(st, ctx, head_bytes: int) -> int:
    """Maximum held tensor bytes; uses tensor shape metadata only."""
    # A snapshot retains one recurrent parity and the whole convolution window.
    size = st.rec.numel() * st.rec.element_size() // 2
    size += st.conv.numel() * st.conv.element_size() + int(head_bytes)
    # A prompt may stop at any of the three incomplete pooling positions.
    for trio in (getattr(st, 'index', None) or [])[:len(st.kc)]:
        ik, ig, _ = trio
        if ik is None or ig is None:
            raise ValueError('Snapshot index ring metadata is missing')
        size += 3 * 8  # int64 ring slots
        size += 3 * (math.prod(ik.shape[1:]) * ik.element_size()
                     + math.prod(ig.shape[1:]) * ig.element_size())
    if ctx is not None:
        for c in list(ctx.kc) + list(ctx.vc):
            rows = min(c.shape[1], max(0, ctx.window + 1))
            size += c.shape[0] * rows * math.prod(c.shape[2:]) * c.element_size()
    return size


def common_charge(local_maxima: list[int]) -> int:
    """Round the largest rank's requirement; all ranks charge the same amount."""
    if not local_maxima or min(local_maxima) <= 0:
        raise ValueError('Invalid snapshot size')
    block = 2 << 20
    return ((max(local_maxima) + block - 1) // block) * block


def pre_evictions(kept, candidate, entries: int, budget: int, charge: int, victim):
    """Apply the existing victim order with the virtual new state before cloning."""
    if entries < 1 or budget < 1 or charge < 1:
        raise ValueError('Invalid snapshot admission limits')
    working = list(kept) + [candidate]
    dropped = []
    while len(working) > entries or len(working) * charge > budget:
        chosen = victim(working)
        working = [c for c in working if c is not chosen]
        if chosen is candidate:
            return dropped, False
        dropped.append(chosen)
    return dropped, True
