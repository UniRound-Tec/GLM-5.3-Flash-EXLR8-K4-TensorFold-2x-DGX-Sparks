"""TF_GLM_MULTI_PREFILL=1 (--parallel): one prompt chunk holds the next rows of several waiting prompts, each prompt's
rows getting the bits they get in a chunk of their own.

Without it, concurrent requests' prompts are prefilled one stream at a time (``multi.MultiDecoder``: the oldest
filling stream's chunk, then a decode round, then the next chunk ...), so the last of four short prompts that arrive
together waits for eight forwards (a 130-token prompt is two chunks: up to its 64-token grid point, where its state
is kept, then its tail) and the decode rounds between them. A forward's cost is mostly fixed at short lengths (~0.12 s
on two Sparks, the weights read once a layer whatever the rows), so a chunk of all four prompts costs little more than
one of them.

A multi-prompt chunk lays its pieces back to back ([s1 rows][s2 rows]...), as a batched verify window does, and runs
one forward over them on the prompt buffers (``Engine.pbuf``):

- the row-independent work runs once over every row: embedding, hyper-connection glue (split between the ranks,
  ``hcsplit``, by the chunk's rows), the blocks' fronts (KDA input projections; DSA projections, norms and query
  expansion), the output projections, the MoE router and routed experts (each pair one fp32 chain: one expert read
  for every prompt), the shared expert, the final norm. These are the kernels a prompt chunk already relies on to
  give a row the same bits whatever chunk it came in (``forward``'s module docstring, ``hcsplit``'s fronts);
- what reads or writes a stream's own state runs piece by piece on that stream's ``forward.State``, through the very
  calls a chunk of its own makes, with the piece's rows as a view of the buffers: the KDA recurrence, its state commit
  and conv shift (``forward.kda_rows``, the chunked or serial kernel at the stream's position), and DSA's latent
  write, indexer update, absorb, dense / sparse attention and expand (``forward.dsa_rows``, at the stream's position
  with its own extent, index ring and chunk count);
- after the forward, per piece in order: the head's logits of its last row when the piece ends its prompt (one row,
  as alone), its DFlash2 taps (its own context), its commit.

So a piece's bits are those of the same rows run as a chunk of their own, from the same position: a valid chunking of
its prompt (pieces start where its own chunks would, on the prompt grid, and end at its kept points or its end).

Image prompts (rows replaced by the vision tower's output) and replays of a kept prompt's head are not grouped; they
fill alone as before. The latent cache is required (``latent.ENABLED``, the default)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch


def enabled(value: str | None = None) -> bool:
    """TF_GLM_MULTI_PREFILL: 1 groups waiting prompts into one chunk (rank 0 decides; rank 1 follows its messages),
    0 (default) fills them one stream at a time."""

    value = (os.environ.get("TF_GLM_MULTI_PREFILL", "") if value is None else value).strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"TF_GLM_MULTI_PREFILL: 0 or 1, not {value!r}")
    return value == "1"


def wait_ms(value: str | None = None) -> float:
    """TF_GLM_MULTI_PREFILL_WAIT_MS: with TF_GLM_MULTI_PREFILL=1, how long an idle server waits after a request
    arrives for more to arrive with it (so their prompts share the first chunk), ending as soon as every slot is
    taken; 0 (default): no wait."""

    value = (os.environ.get("TF_GLM_MULTI_PREFILL_WAIT_MS", "") if value is None else value).strip() or "0"
    try:
        ms = float(value)
    except ValueError:
        ms = -1.0
    if not 0 <= ms <= 1000:
        raise ValueError(f"TF_GLM_MULTI_PREFILL_WAIT_MS: milliseconds from 0 to 1000, not {value!r}")
    return ms


@dataclass(eq=False)
class Piece:
    """A stream's rows of a multi-prompt chunk: ``tokens`` its prompt from ``st.pos`` on; ``drafter`` its DFlash2
    context (taps added after the forward) or None; ``head``: the piece ends its prompt (its last row's logits)."""

    st: Any
    tokens: list[int]
    drafter: Any = None
    head: bool = False
    lo: int = field(default=0, init=False)       # its first row in the chunk

    @property
    def rows(self) -> int:
        return len(self.tokens)


def layout(pieces: Sequence[Piece], rows_max: int) -> int:
    """Each piece's first row (back to back, in order); the chunk's rows. Pieces need rows, distinct stream slots and
    disjoint extents; the chunk at most ``rows_max`` rows."""

    if not pieces:
        raise ValueError("a multi-prompt chunk needs a piece")
    slots = [p.st.slot for p in pieces]
    if len(set(slots)) != len(slots):
        raise ValueError(f"a multi-prompt chunk's pieces share a stream slot: {slots}")
    spans = sorted((p.st.base, p.st.base + p.st.capacity) for p in pieces)
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("a multi-prompt chunk's pieces have overlapping extents")
    at = 0
    for p in pieces:
        if p.rows < 1:
            raise ValueError("a multi-prompt chunk's piece without rows")
        p.lo = at
        at += p.rows
    if at > rows_max:
        raise ValueError(f"a multi-prompt chunk of {at} rows, the prompt buffers hold {rows_max}")
    return at


class PieceMixer:
    """The layers' KDA / DSA blocks of a multi-prompt chunk (``forward.compute``'s ``mixer``, ``split_layers``'
    ``mixer(layer, done)``): fronts and output projection over every row, the stream-state work piece by piece."""

    def __init__(self, w, b, pieces: Sequence[Piece], R: int) -> None:
        self.w, self.b, self.pieces, self.R = w, b, list(pieces), R

    def __call__(self, layer, done: tuple[int, int] | None = None):
        from . import prof
        from .forward import chunks_for, dsa_front, dsa_rows, kda_front, kda_rows, out_proj, undone
        from . import qmm

        w, b, R = self.w, self.b, self.R
        if layer.kind == "kda":
            k = layer.kda
            with prof.timed("kda"):
                with prof.timed("kda: projections"):
                    for lo, hi in undone(R, done):
                        kda_front(layer, b, lo, hi)
                for p in self.pieces:
                    kda_rows(layer, w, p.st, b, p.lo, p.rows)
                with prof.timed("kda: out + all-gather"):
                    return out_proj(w, b, b.kscratch.out[:R], k.o, None, R, site=(layer.index, "a"))
        a, c = layer.dsa, w.cfg
        if a.absorb is None:
            raise ValueError("multi-prompt chunks need the latent cache (TF_GLM_LATENT=1)")
        with prof.timed("dsa (total)"):
            for lo, hi in undone(R, done):
                dsa_front(layer, w, b, lo, hi)
            for p in self.pieces:
                st = p.st
                di = st.dsa_index[layer.index]
                dsa_rows(a, w, st.kc[di], st.pos_dev, b, p.rows, chunks_for(st, p.rows),
                         st.index[di] if st.index is not None else None, st.pos, None, layer.index, lo=p.lo)
            o = b.vn[:R].view(R, a.heads * c.v_dim)
            return out_proj(w, b, o, a.o, qmm.group_sums(o, b.xs_ao[:R]), R, site=(layer.index, "a"))


def _stage(w, b, pieces: Sequence[Piece], R: int) -> None:
    """Every piece's room checked, the chunk's token ids into the prompt buffers (``forward.stage``'s pinned copy)."""

    from .forward import check_room

    for p in pieces:
        check_room(w, p.st, p.rows)
    tokens = [int(t) for p in pieces for t in p.tokens]
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = tokens
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()


@torch.no_grad()
def prefill_pieces(e, pieces: Sequence[Piece]) -> list[torch.Tensor | None]:
    """One prompt chunk over every piece (``decode.prefill_chunk`` for several streams at once): the forward, then per
    piece its head's logits row (a copy, [1, V/world]) when it ends its prompt (else None), its DFlash2 taps and its
    commit (each stream's position advances by its rows)."""

    from . import latent, trial_control
    from .forward import commit, compute, mm

    if not latent.ENABLED:
        raise ValueError("multi-prompt chunks need the latent cache (TF_GLM_LATENT=1)")
    w, b = e.w, e.pbuf
    R = layout(pieces, b.rows)
    _stage(w, b, pieces, R)
    compute(w, None, b, R, head=False, mixer=PieceMixer(w, b, pieces, R))     # leaves b.fnormed's rows
    heads: list[torch.Tensor | None] = []
    for p in pieces:
        if p.head:                       # the head on the piece's last row: the one-row matmul a chunk of its own runs
            r = p.lo + p.rows - 1
            heads.append(mm(b, b.fnormed[r:r + 1], w.head, b.fxs[r:r + 1], b.logits[:1]).clone())
        else:
            heads.append(None)
    for p in pieces:
        if p.drafter is not None:
            if trial_control.ACTIVE['prefill_taps']:
                e.add_prefill_taps(p.drafter, p.rows, b, offset=p.lo)
            else:
                p.drafter.add_taps(torch.cat([t[p.lo:p.lo + p.rows] for t in b.taps], dim=1))
    for p in pieces:
        commit(w, p.st, b, p.rows, p.rows)
    return heads


def prefill_piece_steps(e, pieces: Sequence[Piece]):
    """Sliced counterpart of prefill_pieces; no position/tap publication before the final slice."""
    from . import latent, trial_control
    from .forward import commit, compute_steps, mm

    if not latent.ENABLED:
        raise ValueError("multi-prompt chunks need the latent cache (TF_GLM_LATENT=1)")
    w, b = e.w, e.pbuf
    R = layout(pieces, b.rows)
    _stage(w, b, pieces, R)
    yield from compute_steps(w, None, b, R, head=False, mixer=PieceMixer(w, b, pieces, R))
    heads = []
    for p in pieces:
        if p.head:
            r = p.lo + p.rows - 1
            heads.append(mm(b, b.fnormed[r:r + 1], w.head, b.fxs[r:r + 1], b.logits[:1]).clone())
        else:
            heads.append(None)
    for p in pieces:
        if p.drafter is not None:
            if trial_control.ACTIVE['prefill_taps']:
                e.add_prefill_taps(p.drafter, p.rows, b, offset=p.lo)
            else:
                p.drafter.add_taps(torch.cat([t[p.lo:p.lo + p.rows] for t in b.taps], dim=1))
    for p in pieces:
        commit(w, p.st, b, p.rows, p.rows)
    return heads
