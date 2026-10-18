"""Multi-stream decode windows: several streams' rows back to back ([s1: rows][s2: rows]...), each stream's per-token
DSA state in its own extent of shared arenas (latents [P, 512], pooled keys [P / 4 + 2, 128], bf16 or TF_GLM_KV=fp8's
kv8 rows; extents aligned to EXTENT tokens) and its index keys and gates in its own part of [Q, 128] tensors: its
extent (Q = P, token t at base + t) or, given ``ring``, its stream slot's ring of ``ring`` rows (token t at ring_base
+ t % ring, as the single-stream rings of forward.index_ring; ``ring`` >= a segment's rows + 3). ``SegRows`` holds
the per-row and per-segment device tables the segmented kernels read (latent.seg_*, sparse.seg_*); ``SelectScratch``
the token selection's buffers. Static buffers, so one CUDA graph per window size covers every mix of positions and
context lengths."""

from __future__ import annotations

from typing import Sequence

import torch

from .sparse import POOL, SPARSE_FROM, TOKENS, split_scratch

MAX_SEGS = 8          # streams a window holds at most (--parallel up to engine.PARALLEL_MOST)
SEGS_DEFAULT = 4      # a window's tables hold at least this many segments: --parallel 2 to 4 keep the four they had
EXTENT = 2048         # a stream's extent starts at a multiple of this many tokens (its pooled keys at base / 4)


def seg_count(streams: int) -> int:
    """Segments a batched window's tables (and the launch grids its CUDA graphs capture) hold for ``streams`` stream
    slots: SEGS_DEFAULT up to four streams (the tables and grids of v1.4), one a stream past that, at most MAX_SEGS.
    Every segmented kernel loops over the table's segments and skips one of 0 rows, so a larger table only adds
    programs that exit at once: each segment's bits are those of its stream's window alone at any size."""

    streams = int(streams)
    if not 1 <= streams <= MAX_SEGS:
        raise ValueError(f"a batched verify window holds 1 to {MAX_SEGS} streams, not {streams}")
    return max(SEGS_DEFAULT, streams)


class SegRows:
    """Device tables of a window of up to ``rows`` rows in up to ``max_segs`` segments, one int32 tensor:

    per row: ``pos`` (the row's position in its stream), ``base`` (its stream's first arena row), ``seg`` (segment
    index), ``sparse`` (1: past the dense limit, attends to its selected tokens; 0: dense over keys 0 .. pos),
    ``ibase`` (its stream's first row in the index key and gate arenas: its ring's, or ``base`` without rings);
    per segment: ``seg_start`` (first row), ``seg_rows`` (0: unused), ``seg_pbase`` (pooled-key base, base / 4),
    ``seg_pools`` (pools its last row sees when it has sparse rows, else 0: what the selection scores).

    ``ring``: rows of each stream's index key and gate ring (None: no rings, token t at base + t); a segment then
    names its ring's first row (``set``'s fourth field: one ring a stream slot, slot x ring, the same for the
    stream in every window)."""

    NO_RING = 2 ** 31 - 1      # the kernels' ring size without rings (positions stay below it: base + pos)

    def __init__(self, rows: int, device, max_segs: int = SEGS_DEFAULT, ring: int | None = None) -> None:
        if ring is not None and ring < 4:
            raise ValueError(f"an index ring of {ring} rows")
        if not 1 <= max_segs <= MAX_SEGS:
            raise ValueError(f"tables of {max_segs} segments: 1 to {MAX_SEGS}")
        self.rows, self.max_segs = rows, max_segs
        self.ring = ring
        self.ring_rows = ring if ring is not None else self.NO_RING
        n = 5 * rows + 4 * max_segs
        pin = torch.cuda.is_available() and torch.device(device).type == "cuda"
        self.host = torch.zeros((n,), dtype=torch.int32, pin_memory=pin)
        self.dev = torch.zeros((n,), dtype=torch.int32, device=device)
        self.pos, self.base, self.seg, self.sparse, self.ibase = (self.dev[i * rows:(i + 1) * rows] for i in range(5))
        o = 5 * rows
        self.seg_start, self.seg_rows, self.seg_pbase, self.seg_pools = (
            self.dev[o + i * max_segs:o + (i + 1) * max_segs] for i in range(4))
        self.staged = torch.cuda.Event() if pin else None
        self.R = 0
        self.segments: list[tuple[int, int, int]] = []

    def fill(self, segments: Sequence[tuple[int, ...]], *, sparse_from: int | None = SPARSE_FROM):
        """The host tables for ``segments`` [(base, pos, rows[, ring_base])] in window order (numpy int32, the device
        layout); ring_base (with rings, required): the stream's ring's first row.
        ``sparse_from``: the first position that attends sparsely (None: no index, every row dense)."""

        if not 1 <= len(segments) <= self.max_segs:
            raise ValueError(f"a window of {len(segments)} segments, at most {self.max_segs}")
        R = sum(seg[2] for seg in segments)
        if R > self.rows:
            raise ValueError(f"a window of {R} rows, the tables hold {self.rows}")
        rows, S = self.rows, self.max_segs
        t = torch.zeros((5 * rows + 4 * S,), dtype=torch.int32).numpy()
        r = 0
        for s, seg in enumerate(segments):
            base, pos, n = seg[:3]
            if n < 1 or pos < 0 or base < 0 or base % EXTENT:
                raise ValueError(f"segment {s}: base {base} (a multiple of {EXTENT}), position {pos}, {n} rows")
            if self.ring is None:
                ibase = base
            else:
                if len(seg) < 4 or seg[3] < 0:
                    raise ValueError(f"segment {s}: index rings need the stream's ring base")
                ibase = seg[3]
                if n + 3 > self.ring:
                    raise ValueError(f"segment {s}: {n} rows and the 3 before them past a ring of {self.ring}")
            for i in range(n):
                q = pos + i
                t[r + i] = q
                t[rows + r + i] = base
                t[2 * rows + r + i] = s
                t[3 * rows + r + i] = int(sparse_from is not None and q >= sparse_from)
                t[4 * rows + r + i] = ibase
            last = pos + n - 1
            o = 5 * rows
            t[o + s] = r
            t[o + S + s] = n
            t[o + 2 * S + s] = base // POOL
            t[o + 3 * S + s] = (last + 1) // POOL if sparse_from is not None and last >= sparse_from else 0
            r += n
        return t, R

    def set(self, segments: Sequence[tuple[int, ...]], *, sparse_from: int | None = SPARSE_FROM) -> int:
        """Stage ``segments`` [(base, pos, rows[, ring_base])] into the device tables (a pinned copy, ordered on the current
        stream like forward.stage's token ids); returns the window's rows."""

        t, R = self.fill(segments, sparse_from=sparse_from)
        if self.staged is not None:
            self.staged.synchronize()          # the previous copy has left the pinned buffer
        self.host.numpy()[:] = t
        self.dev.copy_(self.host, non_blocking=True)
        if self.staged is not None:
            self.staged.record()
        self.R, self.segments = R, list(segments)
        return R

    def any_sparse(self) -> bool:
        """Host view of the last ``set``: does any row attend sparsely."""
        return bool(self.host[3 * self.rows:3 * self.rows + self.R].any())


class SelectScratch:
    """The segmented selection's buffers: fp32 scores [rows, pools] (a stream's pools at most: its largest context
    / 4), tokens [rows, 2051] int32, counts [rows] int32, the split selection's scratch."""

    def __init__(self, rows: int, max_context: int, device) -> None:
        self.rows = rows
        pools = max(1, -(-max_context // POOL))
        self.scores = torch.empty((rows, pools), dtype=torch.float32, device=device)
        self.tokens = torch.full((rows, TOKENS), -1, dtype=torch.int32, device=device)
        self.counts = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.split = split_scratch(rows, pools, device)
