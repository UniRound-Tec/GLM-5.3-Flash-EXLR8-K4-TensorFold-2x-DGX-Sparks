"""CUDA decode accepts drafts only when they match the serial keyed sample; all ranks sample identical gathered candidates without a broadcast."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.sampling import comm_gather, nucleus_rows, one_rank
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import glue, prof, qmm, tp
from .forward import Buffers, State, chunks_for, commit, compute, stage
from .mtp import mtp_compute, mtp_forward, mtp_stage
from .sparse import pool_bucket
from .weights import Weights


class StopVote:
    """A serial run's stop (issue #38): rank 0's ``on_tokens`` asks for it (the client has gone, a stop string, a
    gate's cut), and every rank learns it from the next verify sample's all-gather, which carries one float32 word
    more for it (``sample_rows``' ``vote``), so all ranks end the run after the same round with no exchange of its
    own. Wraps ``on_tokens``: a true result is this rank's wish (rank 1's callback never wishes); ``stop`` is the
    agreed decision, alike on every rank. Only when the run ends changes: the tokens sampled never do."""

    def __init__(self, on_tokens=None) -> None:
        self.fn = on_tokens
        self.mine = False               # this rank's on_tokens asked to stop (sticky)
        self.agreed = False             # some rank asked, as of the last verify sample (sticky)
        self.flags: torch.Tensor | None = None      # the last sample's gathered words, read when ``stop`` is asked

    def __call__(self, tokens) -> bool:
        if self.fn is not None and self.fn(tokens):
            self.mine = True
        return self.mine

    def gather(self, gather):
        """``gather`` (every rank's float32 words, [world, n]) with this rank's wish as one word more on its first
        call (a sample's first exchange), the results given back without it."""

        first = [True]

        def voting(words: torch.Tensor) -> torch.Tensor:
            if not first[0]:
                return gather(words)
            first[0] = False
            flag = torch.full((1,), 1.0 if self.mine else 0.0, dtype=words.dtype, device=words.device)
            got = gather(torch.cat([words.reshape(-1), flag]))
            self.flags = got[:, -1]
            return got[:, :-1].contiguous()

        return voting

    @property
    def stop(self) -> bool:
        if self.flags is not None:
            self.agreed = self.agreed or bool((self.flags > 0).any())
            self.flags = None
        return self.agreed


def stopped(e) -> bool:
    """Whether every rank agreed to end this run (``StopVote``); False outside a voting run."""

    vote = getattr(e, "vote", None)
    return vote is not None and vote.stop


def sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                offset: int | None = None, probs: list[float] | None = None,
                vote: StopVote | None = None) -> list[int]:
    """Rows of (this rank's vocabulary slice of) logits at their absolute positions -> tokens, same on all ranks;
    ``vote``: a ``StopVote`` whose wish rides on the sample's (first) all-gather."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    if not greedy and not sampling.top_k:           # top_k off: the shared nucleus rule over every rank's shard
        gather = one_rank if w.comm is None else comm_gather(w.comm)
        return nucleus_rows(logits, positions, sampling, offset=w.vocab_offset if offset is None else offset,
                            gather=gather if vote is None else vote.gather(gather), probs=probs)
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if probs is not None and greedy:
        k = min(logits.shape[1], 20 + MARGIN)       # the draft's confidence needs its competitors too
    vals, ids = torch.topk(logits.float(), k, dim=-1)
    ids = (ids + (w.vocab_offset if offset is None else offset)).to(torch.int32)
    if w.comm is None:
        values = vals.cpu().numpy().astype(np.float32)
        tokens = ids.cpu().numpy().astype(np.int64)
        if vote is not None:
            vote.gather(one_rank)(vals.new_zeros((0,)))
    else:
        packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous()
        if vote is None:
            got = torch.empty((w.world * packed.numel(),), dtype=torch.float32, device=logits.device)
            w.comm.all_gather(packed.view(-1), got)
        else:
            got = vote.gather(comm_gather(w.comm))(packed.view(-1))
        g = got.reshape(w.world, R, 2 * k).cpu()
        values = torch.cat([g[r, :, :k] for r in range(w.world)], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(w.world)], dim=1).numpy()
        tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if probs is not None:
        probs.extend(_probability(values, tokens, chosen, sampling))
    return chosen


def _probability(values: np.ndarray, tokens: np.ndarray, chosen: list[int], sampling: Sampling | None) -> list[float]:
    """Return each chosen token's top-k/top-p probability, using temperature 1 for greedy draft confidence."""

    temp = sampling.temperature if sampling is not None and sampling.temperature > 0 else 1.0
    top_p = sampling.top_p if sampling is not None else 1.0
    top_k = sampling.top_k if sampling is not None and sampling.top_k else values.shape[1]
    out = []
    for i, tok in enumerate(chosen):
        order = np.lexsort((tokens[i], -values[i]))[:top_k]
        v = values[i][order].astype(np.float64) / temp
        p = np.exp(v - v.max())
        p /= p.sum()
        if 0.0 < top_p < 1.0:
            keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
            p = p[:keep] / p[:keep].sum()
            order = order[:keep]
        ids = tokens[i][order]
        hit = np.nonzero(ids == tok)[0]
        out.append(float(p[hit[0]]) if len(hit) else 0.0)
    return out


PREFILL_ROWS = 2048      # rows of a prompt chunk


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and for prompt chunks."""

    def __init__(self, w: Weights, *, capacity: int = 2560, max_rows: int = 8, prefill_rows: int = PREFILL_ROWS,
                 graphs: bool = False, graph_rows: tuple[int, ...] = (1, 2, 3, 4), long_context: bool = False,
                 taps: tuple[int, ...] = (), grid: int = 0, split=None, mtp_rows: int | None = None,
                 streams: int = 1, pool_rows: int | None = None, kv: str | None = None, display: int = 0) -> None:
        """``kv``: the DSA caches' format (``kv8``; default TF_GLM_KV, bf16 unless set: fp8 halves the latent cache and
        the indexer's pooled keys); ``grid``: prompt chunks start on multiples of it (TF_GLM_PROMPT_GRID; 0:
        anywhere); ``split``: a ``hcsplit.SplitSettings`` (TF_GLM_HC_SPLIT, TF_GLM_PREFILL_OVERLAP) for prompt
        chunks, None or off: unsplit; ``mtp_rows``: rows of an MTP step (its absorb chunks; default ``max_rows``):
        draft chains take one row a step, so the MTP buffers need not be as wide as the verify windows; ``streams``:
        stream slots (``forward.Slots``) and ``pool_rows``: tokens of the shared cache pool (``forward.Caches``,
        default ``capacity``) for concurrent streams (``multi``); ``display``: bytes of the GB10's display reservation
        the pool was sized with (``display_kv``, TF_GLM_DISPLAY_KV_MIB). ``st`` is the state the single-stream paths run on: ``home`` (slot 0, the pool's first
        ``capacity`` tokens), the one the CUDA graphs were captured on, unless ``use`` points it at another."""

        if grid and prefill_rows % grid:
            raise ValueError(f"prompt chunks of {prefill_rows} rows do not tile a {grid}-row grid")
        self.w = w
        self.grid = grid
        w.meta["long_context"] = long_context
        from . import l2pf

        pf = l2pf.Settings.from_env()                        # TF_GLM_L2PF: installed before any graph capture
        if pf.on and torch.cuda.is_available() and w.meta.get("l2pf") is None:
            w.meta["l2pf"] = l2pf.Prefetch(w, pf)
            if w.rank == 0:
                print(f"[tensorfold] {w.meta['l2pf'].summary()}", flush=True)
        self.rows, self.prefill_rows = max_rows, prefill_rows
        if any(l.kind == "kda" for l in w.layers):           # the wide KDA chain's scratch, before any capture
            from . import kda as kda_mod

            kda_mod.reserve(max(max_rows, prefill_rows), tp.of(w).lin_heads, w.device)
        self.buf = Buffers(w, max_rows, capacity)
        # the row split's pad rows past a full chunk (``hcsplit.pad_rows``: 0 at two and four ranks, 1 at three)
        from .hcsplit import buffer_rows

        pad = buffer_rows(prefill_rows, w.world, split) - prefill_rows
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True, pad=pad)
        if taps:
            self.buf.set_taps(tuple(taps), w.cfg.hidden)         # before any graph capture
            self.pbuf.set_taps(tuple(taps), w.cfg.hidden)
        if split is not None and split.split:
            from .hcsplit import HcSplit

            self.pbuf.split = HcSplit(w, self.pbuf, split)
        self.mbuf = Buffers(w, min(mtp_rows or max_rows, max_rows), capacity) if w.mtp is not None else None
        from .forward import Caches, Slots, index_ring

        # the indexer's key and gate rings, one a slot, hold the widest window (a prompt chunk, an MTP absorb chunk, a
        # verify window, a stream's rows of a batched window) and the 3 rows before it
        from . import kv8

        self.caches = Caches(w, pool_rows or capacity, streams=streams,
                             ring=index_ring(pool_rows or capacity, max(max_rows, prefill_rows)),
                             kv=kv8.kv_kind() if kv is None else kv, display=display)
        self.slots = Slots(w, streams, max_rows)
        self.home = self.st = State(w, capacity, max_rows, caches=self.caches, slots=self.slots)
        self.last_hidden: torch.Tensor | None = None
        self.head: torch.Tensor | None = None       # the last prompt's end logits row (``prefill``'s keep_head)
        self.constraint = self.window = None            # a request's grammar, and the next sample's rows under it
        self.vote: StopVote | None = None               # a serial run's stop vote, carried by each verify sample
        self.draft_n = w.head.n
        self.graphs = None
        self.replays = {"main": 0, "sparse": 0, "mtp": 0, "sparse_mtp": 0, "eager": 0}   # steps by path
        if graphs:
            from .graphs import Graphs

            mtp_graphs = tuple(r for r in graph_rows if self.mbuf is None or r <= self.mbuf.rows)
            self.graphs = Graphs(self, tuple(r for r in graph_rows if r <= max_rows), mtp_graphs)
            self.reset()

    def reset(self) -> None:
        self.st.reset()

    def use(self, st: State) -> State:
        """Run the single-stream paths (prefill, forward, commit, snapshots) on ``st``; returns the previous state."""

        prev, self.st = self.st, st
        return prev

    def graphed(self, st: State | None = None) -> bool:
        """Whether the captured graphs run ``st`` (default the current state): the same slot and extent base as the
        state they were captured on (and the pool's views, not a clone's)."""

        st = self.st if st is None else st
        return (self.graphs is not None and st.caches is self.caches and st.slots is self.slots
                and st.graph_key == self.home.graph_key)

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A step's forward (a CUDA graph when one was captured for its shape and state): logits [R, V/world]."""

        R = stage(self.w, self.st, self.buf, tokens)
        dense = self.st.pos + R <= self.w.cfg.dense_limit
        g, kind = None, "main"
        if not self.graphed():
            pass
        elif self.graphs is not None and dense:
            g = self.graphs.main.get((R, self.st.parity))
        elif self.graphs is not None and self.st.pos >= self.w.cfg.dense_limit and self.st.index is not None:
            # every row past the dense limit: the sparse graph for this pool bucket (same kernels as eager)
            # the pool buckets the graphs were captured for: the home view's (a stream at home with a smaller
            # extent scores pools past its own too, all past its visible ones: -inf, the same selection)
            bucket = pool_bucket(self.st.pos, R, self.home.index[0][2].shape[0] - 2)
            g, kind = self.graphs.sparse.get((R, self.st.parity, bucket)), "sparse"
        if g is not None:
            self.replays[kind] += 1
            g.replay()
            return self.buf.logits[:R]
        self.replays["eager"] += 1
        return compute(self.w, self.st, self.buf, R, nch=chunks_for(self.st, R), host_pos=self.st.pos)

    def mtp(self, next_tokens: Sequence[int], hidden: torch.Tensor) -> torch.Tensor:
        """The MTP head on rows (hidden, next token): logits of the last row [1, V/world]."""

        n = mtp_stage(self.w, self.st, self.mbuf, next_tokens, hidden)
        dense = self.st.mtp_len + n <= self.w.cfg.dense_limit
        g, kind = None, "mtp"
        if not self.graphed():
            pass
        elif self.graphs is not None and not self.mbuf.zero_first and dense:
            g = self.graphs.mtp.get(n)
        elif (self.graphs is not None and not self.mbuf.zero_first and self.st.index is not None
              and self.st.mtp_len >= self.w.cfg.dense_limit):
            bucket = pool_bucket(self.st.mtp_len, n, self.home.index[-1][2].shape[0] - 2)
            g, kind = self.graphs.sparse_mtp.get((n, bucket)), "sparse_mtp"
        if g is not None:
            self.replays[kind] += 1
            g.replay()
            return self.mbuf.logits[:1, :self.draft_n]
        from .attention import CHUNK

        return mtp_compute(self.w, self.st, self.mbuf, n, nch=-(-(self.st.mtp_len + n) // CHUNK),
                           host_pos=self.st.mtp_len)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False, probs: list[float] | None = None) -> list[int]:
        if not draft and self.constraint is not None and self.window is not None:
            self.constraint.mask(logits, self.window, self.w.vocab_offset)   # this rank's vocabulary columns
            self.window = None
        vote = None if draft else getattr(self, "vote", None)
        return sample_rows(self.w, logits, positions, sampling, None, probs, vote)

    def verify_window(self, tokens: list[int]) -> list[int]:
        """The window a reply's grammar keeps (a chain cut at its first rejected draft), masked at the next sample."""

        if self.constraint is None:
            return tokens
        self.window = self.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
        return self.window.tokens

    def follow(self, tokens: Sequence[int]) -> None:
        if self.constraint is not None:
            self.constraint.advance(tokens)

    def tap_rows(self, n: int, b: Buffers | None = None) -> torch.Tensor:
        """The last forward's first n rows of DFlash2 taps, concatenated in layer order: [n, taps * D]."""

        return torch.cat([t[:n] for t in (b or self.buf).taps], dim=1)

    def add_prefill_taps(self, drafter, n: int, b: Buffers, offset: int = 0) -> None:
        """Copy layer slices into the existing draft input; avoid a full-chunk cat.

        Both solo and pooled drafts consume this input on the same CUDA stream.
        add_taps retains its original launch sizes, order and BF16 values.
        """
        owner = getattr(drafter, "owner", drafter)
        buf = owner.tap_in
        if n < 0 or offset < 0 or any(t.shape[0] < offset + n for t in b.taps):
            raise ValueError("prefill tap rows outside the forward buffers")
        if sum(t.shape[1] for t in b.taps) != buf.shape[1]:
            raise ValueError("prefill and draft tap widths disagree")
        for start in range(0, n, buf.shape[0]):
            take = min(n - start, buf.shape[0])
            target = buf[:take]
            torch.cat([t[offset + start:offset + start + take] for t in b.taps], dim=1, out=target)
            drafter.add_taps(target)

    def main_hidden(self, rows: slice) -> torch.Tensor:
        """Return the final-normed main-model rows that the MTP head reads after a forward with logits."""

        return self.buf.fnormed[rows]

    def draft_hidden(self, row: int) -> torch.Tensor:
        """The MTP head's own output row a chained draft reads (after an MTP step): its shared_head.norm output."""

        return self.mbuf.fnormed[0:1]


# -- MTP drafts ---------------------------------------------------------------------------------------------------
def absorb(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """Absorb hidden rows and their next tokens into the MTP cache in independent chunks; return the last logits."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    step = e.mbuf.rows
    logits = None
    for s0 in range(0, len(next_tokens), step):
        part = list(next_tokens[s0:s0 + step])
        logits = e.mtp(part, hidden[s0:s0 + step])
        st.set_mtp_len(st.mtp_len + len(part))
    return logits


def draft(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb kept positions and chain drafts until cumulative confidence fails, always keeping the first draft."""

    st = e.st
    logits = absorb(e, hidden, next_tokens)
    drafts: list[int] = []
    n = len(next_tokens)
    chain = 1.0
    for j in range(count):
        probs: list[float] = []
        d = e.sample(logits[:1], [position + j], sampling, draft=True, probs=probs if confidence > 0 else None)[0]
        if confidence > 0 and j > 0 and chain * probs[0] < confidence:
            break
        drafts.append(d)
        if confidence > 0:
            chain *= probs[0]
            if chain < confidence:            # a further draft could not pass either: skip its MTP step
                break
        if j + 1 < count:
            prev = e.draft_hidden(n - 1 if j == 0 else 0)
            logits = e.mtp([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
    return drafts


# -- prefix snapshots ---------------------------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Copy committed KDA state and retain attention caches whose valid prefixes survive resume; pending MTP rows await next tokens from the new prompt."""

    ids: list[int]
    rec: torch.Tensor
    conv: torch.Tensor
    pending: torch.Tensor | None
    mtp_len: int
    drafter_end: int
    rows: list | None = None      # the attention rows of ids, saved when another conversation took the live caches
    nbytes: int = 0
    drafter_rows: list | None = None  # a ring drafter's window rows before drafter_end, copied when taken
    tail: list | None = None      # the indexer rings' rows of the pools still filling at ids' end (``_ring_tail``)
    # the head's logits row at ids' end (this rank's vocabulary slice, before any grammar mask), kept with a prompt's
    # own end state: a request of exactly these ids resumes here and samples its first token from it, prefilling
    # nothing (``prefill``); None for states short of a prompt's end (which no head ran for)
    head: torch.Tensor | None = None


def _ring_tail(st, n: int, m: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(slots, rows) of the indexer's key and gate rings a snapshot of n tokens (m in the MTP head) needs besides the
    pools: the tokens of each layer's pool still filling (n % 4 of them, the last ones), which later tokens' windows
    overwrite in the ring (``forward.index_ring``). slots: ring rows (positions mod the ring); rows: [2, k, 128]
    (keys, gates), cloned."""
    index, kc = getattr(st, "index", None) or [], getattr(st, "kc", [])
    out = []

    def take(trio, end):
        ik, ig, _ = trio
        slots = torch.arange(end - end % 4, end, device=ik.device) % ik.shape[0]
        out.append((slots, torch.stack((ik[slots], ig[slots]))))

    for trio in index[:len(kc)]:
        if n % 4:
            take(trio, n)
    if m > 0 and m % 4 and len(index) > len(kc):
        take(index[-1], m)
    return out


def put_ring_tail(st, snap: Snapshot) -> None:
    """A kept snapshot's ring tail (``_ring_tail``) into ``st``'s rings (restore, and a multi-stream stream taking
    over a kept prompt in any slot)."""
    index, kc = getattr(st, "index", None) or [], getattr(st, "kc", [])
    idx = list(index[:len(kc)]) if len(snap.ids) % 4 else []
    if snap.mtp_len > 0 and snap.mtp_len % 4 and len(index) > len(kc):
        idx.append(index[-1])
    tail = snap.tail or []
    if len(tail) != len(idx):
        raise ValueError("a snapshot's indexer ring rows do not match the caches they go back to")
    for (ik, ig, _), (slots, rows) in zip(idx, tail):
        ik.index_copy_(0, slots % ik.shape[0], rows[0])
        ig.index_copy_(0, slots % ig.shape[0], rows[1])


def take_snapshot(e: Engine, ids: Sequence[int], pending: torch.Tensor | None, *, mtp: bool,
                  drafter=None) -> Snapshot:
    """A ring drafter's window rows are copied now (``_ring_window``): its next rows overwrite them in the ring; so
    are the indexer rings' rows of the pools still filling (``_ring_tail``)."""
    st = e.st
    rec = st.rec[st.cur[0]].clone() if st.cur else st.rec[0].clone()
    mtp_len = st.mtp_len - st.mtp_drafted if mtp and pending is not None else -1
    snap = Snapshot(list(ids), rec, st.conv.clone(), pending.clone() if pending is not None else None,
                    mtp_len, drafter.context_end if drafter is not None else -1,
                    tail=_ring_tail(st, len(ids), max(mtp_len, 0)))
    if drafter is not None and getattr(drafter, "ring", 0) and snap.drafter_end == len(snap.ids):
        snap.drafter_rows = _ring_window(drafter, len(snap.ids))
    return snap


def _ring_slots(drafter, n: int) -> torch.Tensor:
    """The ring slots of the window rows a block pass at context end n reads (n - window - 1 .. n - 1)."""
    lo = max(0, n - drafter.window - 1)
    return torch.arange(lo, n, device=drafter.kc[0].device) % drafter.ring


def _ring_window(drafter, n: int) -> list[torch.Tensor]:
    """A copy of a ring drafter's window rows before n, in position order."""
    idx = _ring_slots(drafter, n)
    return [c.index_select(1, idx) for c in drafter.kc] + [c.index_select(1, idx) for c in drafter.vc]


def _put_ring_window(drafter, n: int, rows: list[torch.Tensor]) -> None:
    idx = _ring_slots(drafter, n)
    caches = list(drafter.kc) + list(drafter.vc)
    if len(rows) != len(caches) or any(r.shape[1] != idx.numel() for r in rows):
        raise ValueError("a kept state's DFlash2 window does not match the drafter's ring")
    for c, r in zip(caches, rows):
        c.index_copy_(1, idx, r)


def _row_views(st, n: int, m: int) -> list[torch.Tensor]:
    """Views of the attention rows a snapshot of n tokens (m in the MTP head) depends on: latents or keys and values,
    and the indexer's pooled keys (its per-token keys and gates live in rings: the snapshot's ``tail``)."""
    views = [kc[:n] for kc in st.kc] + [vc[:n] for vc in st.vc if vc is not None]
    idx = st.index or []
    main_idx = idx[:len(st.kc)]
    for _, _, pk in main_idx:
        views.append(pk[:n // 4 + 1])
    if m > 0 and hasattr(st, "mtp_kc"):
        views.append(st.mtp_kc[:m])
        if getattr(st, "mtp_vc", None) is not None:
            views.append(st.mtp_vc[:m])
        if len(idx) > len(st.kc):
            views.append(idx[-1][2][:m // 4 + 1])
    return views


def _drafter_views(drafter, n: int) -> list[torch.Tensor]:
    """DFlash2's context rows a snapshot of n tokens depends on: its sliding window's keys and values before n (every
    later query sits at n or past it, and masks older rows), per draft layer."""
    lo = max(0, n - drafter.window - 1)
    return [c[:, lo:n] for c in drafter.kc] + [c[:, lo:n] for c in drafter.vc]


def _saved_views(e: Engine, snap: Snapshot, drafter=None) -> list[torch.Tensor]:
    """The rows ``save_rows`` copies: the attention rows, and with ``drafter`` (whose context the snapshot ends at)
    DFlash2's window rows."""
    n = len(snap.ids)
    views = _row_views(e.st, n, max(snap.mtp_len, 0))
    if drafter is not None and snap.drafter_end == n and not getattr(drafter, "ring", 0):   # a ring's: drafter_rows
        views += _drafter_views(drafter, n)
    return views


def save_rows(e: Engine, snap: Snapshot, drafter=None) -> None:
    """Copy a snapshot's attention rows out of the live caches before another conversation overwrites them; DFlash2's
    context goes with them only given ``drafter`` (TF_GLM_SHARED_PREFIX), otherwise the snapshot no longer fits it."""
    views = _saved_views(e, snap, drafter)
    snap.rows = [v.clone() for v in views]
    snap.nbytes = sum(r.numel() * r.element_size() for r in snap.rows)
    if drafter is None or snap.drafter_end != len(snap.ids):
        snap.drafter_end = -1
        snap.drafter_rows = None


def row_bytes(e: Engine, snap: Snapshot, drafter=None) -> int:
    """What ``save_rows`` would copy for this snapshot."""
    return sum(v.numel() * v.element_size() for v in _saved_views(e, snap, drafter))


def snapshot_bytes(snap: Snapshot) -> int:
    """Device memory a kept snapshot holds: its KDA states, conv windows, pending MTP rows, a ring drafter's window,
    the indexer rings' tail and any saved rows."""
    held = [snap.rec, snap.conv] + ([snap.pending] if snap.pending is not None else []) + (snap.drafter_rows or [])
    held += [snap.head] if snap.head is not None else []
    held += [t for pair in (snap.tail or []) for t in pair]
    return sum(t.numel() * t.element_size() for t in held) + (snap.nbytes if snap.rows is not None else 0)


def load_rows(e: Engine, snap: Snapshot, drafter=None) -> None:
    """Put a saved snapshot's attention rows (and DFlash2's, when it saved them: the same ``drafter``) back."""
    views = _saved_views(e, snap, drafter)
    if len(views) != len(snap.rows):
        raise ValueError("a saved snapshot's rows do not match the caches they go back to")
    for dst, src in zip(views, snap.rows):
        dst.copy_(src)


def restore(e: Engine, snap: Snapshot, drafter=None) -> None:
    st = e.st
    put_ring_tail(st, snap)
    if st.cur:
        st.rec[st.cur[0]].copy_(snap.rec)
    st.conv.copy_(snap.conv)
    st.set_pos(len(snap.ids))
    st.set_mtp_len(max(snap.mtp_len, 0))
    st.mtp_drafted = 0
    if drafter is not None:
        if getattr(drafter, "ring", 0):
            if snap.drafter_rows is None or snap.drafter_end != len(snap.ids):
                raise ValueError("this snapshot kept no DFlash2 window for the drafter's ring")
            _put_ring_window(drafter, snap.drafter_end, snap.drafter_rows)
        drafter.context_end = snap.drafter_end
        drafter.pos_dev.fill_(snap.drafter_end)


# -- prefill ----------------------------------------------------------------------------------------------------
def whole(snap: Snapshot | None, prompt: Sequence[int]) -> bool:
    """Whether ``snap`` is a state of exactly ``prompt`` that kept its head's logits row (a replay resumes there)."""

    return snap is not None and snap.head is not None and len(snap.ids) == len(prompt)


@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True, drafter=None,
            resume: Snapshot | None = None, feed=None, sample: bool = True, keep_head: bool = False) -> int | None:
    """Commit the prompt in chunks and sample its first token; a resumed prompt ends in a fresh prefill's state; ``feed``: an image prompt's feature rows (``engine.VisionFeed``); ``sample=False``: commit only (no head, no first token, None), leaving the state a prompt of this length leaves before its sample.

    ``resume`` may also be a state of the whole prompt that kept its head's logits row (``whole``, a replay): the
    state goes back and the first token is sampled from that row, nothing prefilled. ``keep_head``: the head's
    logits row of the prompt's end, as sampled from (before a grammar's mask), is left in ``e.head`` for a snapshot."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    w, st, b = e.w, e.st, e.pbuf
    use_mtp = mtp and w.mtp is not None
    begin = 0
    replay = sample and whole(resume, prompt)
    e.head = None
    if resume is None:
        e.reset()
        if drafter is not None:
            drafter.reset()
    else:
        begin = len(resume.ids)
        if e.grid and begin % e.grid:
            raise ValueError(f"a resumed prefill starts on the {e.grid}-row grid, not at {begin}")
        if (begin >= len(prompt) and not replay) or list(prompt[:begin]) != resume.ids:
            raise ValueError("a resumed prefill needs a snapshot of a strict prefix of the prompt (or of the whole "
                             "prompt with its head's logits row)")
        if (use_mtp and resume.mtp_len < 0) or (drafter is not None and resume.drafter_end != begin):
            raise ValueError("this snapshot's draft caches do not fit the request")
        restore(e, resume, drafter)
        if use_mtp and replay:
            # the prompt's last hidden row waits for the first token, as a fresh prefill leaves it (the MTP cache
            # holds every row before it: mtp_len n - 1)
            e.last_hidden = resume.pending[-1:].clone()
        elif use_mtp:
            k = resume.pending.shape[0]
            _absorb_rows(e, resume.pending, list(prompt[begin - k + 1:begin + 1]))
    last = None
    if replay:
        last = resume.head.clone()                       # a grammar's mask below writes into the rows it samples
    else:
        prof.active = True
        for start in range(begin, len(prompt), e.prefill_rows):
            last = prefill_chunk(e, prompt, start, min(e.prefill_rows, len(prompt) - start), drafter=drafter,
                                 feed=feed, mtp=use_mtp, head=sample)
        prof.active = False
        prof.report(len(prompt) - begin)
    if not sample:
        return None
    if keep_head:
        e.head = last.clone()
    if e.constraint is not None:                         # the first token's row, under the reply's grammar
        e.window = e.constraint.window([0], [-1])
    first = e.sample(last, [len(prompt)], sampling)[0]
    e.follow([first])
    return first


def prefill_chunk(e: Engine, prompt: Sequence[int], start: int, R: int, *, drafter=None, feed=None,
                  mtp: bool = False, head: bool = True) -> torch.Tensor | None:
    """Commit prompt rows start .. start + R (at e.st.pos == start): the chunk's forward, DFlash2's taps, the MTP
    head's rows (``mtp``: the next tokens are the prompt's), the commit; a copy of the head's logits of its last
    row, taken before anything else reuses the buffers, with ``head``, else None. ``prefill`` runs a prompt as these;
    so do concurrent streams' prompt steps (``multi``), one chunk at a time."""

    w, st, b = e.w, e.st, e.pbuf
    chunk = list(prompt[start:start + R])
    inject = feed.chunk(start, R) if feed is not None else None
    out = compute(w, st, b, stage(w, st, b, chunk), nch=chunks_for(st, R), host_pos=st.pos, inject=inject,
                  head=head)
    out = out.clone() if head else None
    e.last_hidden = b.fnormed[R - 1:R].clone()
    if drafter is not None:
        e.add_prefill_taps(drafter, R, b)
    if mtp:
        nxt = list(prompt[start + 1:start + R + 1])
        if nxt:
            with prof.timed("mtp absorb"):
                _absorb_rows(e, b.fnormed[:len(nxt)], nxt)
    with prof.timed("commit"):
        commit(w, st, b, R, R)
    return out


def prefill_chunk_steps(e: Engine, prompt: Sequence[int], start: int, R: int, *, drafter=None, feed=None,
                        head: bool = True):
    """The concurrent prompt path, with a yield per layer and commit only after the final slice."""
    from .forward import compute_steps

    w, st, b = e.w, e.st, e.pbuf
    chunk = list(prompt[start:start + R])
    inject = feed.chunk(start, R) if feed is not None else None
    out = yield from compute_steps(w, st, b, stage(w, st, b, chunk), nch=chunks_for(st, R),
                                   host_pos=st.pos, inject=inject, head=head)
    out = out.clone() if head else None
    e.last_hidden = b.fnormed[R - 1:R].clone()
    if drafter is not None:
        e.add_prefill_taps(drafter, R, b)
    with prof.timed("commit"):
        commit(w, st, b, R, R)
    return out


def _absorb_rows(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> None:
    """A prompt's rows into the MTP cache through the prefill buffers (the prefill arithmetic, like the prompt)."""

    st = e.st
    mtp_forward(e.w, st, e.pbuf, next_tokens, hidden)
    st.set_mtp_len(st.mtp_len + len(next_tokens))


# -- decode loops -----------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    # seconds by stage: "forward" the verify windows' GPU time (CUDA events), "sample" from a window's launch to its
    # tokens (waiting for the window included: no sync of its own), "draft" and "commit" host time
    stages: dict[str, float] = field(default_factory=dict)
    depths: list[int] = field(default_factory=list)
    keeps: list[int] = field(default_factory=list)
    arms: str = ""                          # auto_decode: each round's drafter, "m" (MTP), "f" (DFlash2) or "c" (copy)
    pending: torch.Tensor | None = None     # auto_decode: MTP input rows of committed positions not absorbed yet
    copy_rounds: int = 0                    # rounds whose drafts were copied from the context (``copy_drafts``)
    copy_drafted: int = 0
    copy_accepted: int = 0

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


def _sync(w: Weights) -> None:
    torch.cuda.synchronize()


class _Clock:
    """A verify window's GPU time between two CUDA events, read after the round's sample has waited for the window
    anyway (so timing adds no host-device sync); wall time where there is no GPU (the CPU tests' fake engines)."""

    def __init__(self) -> None:
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.a = torch.cuda.Event(enable_timing=True)
            self.b = torch.cuda.Event(enable_timing=True)
        self.t = self.dt = 0.0

    def start(self) -> None:
        if self.cuda:
            self.a.record()
        else:
            self.t = time.perf_counter()

    def stop(self) -> None:
        if self.cuda:
            self.b.record()
        else:
            self.dt = time.perf_counter() - self.t

    def seconds(self) -> float:
        """Only after something has waited for the work queued before ``stop`` (the sample's device-to-host copy)."""
        return self.a.elapsed_time(self.b) / 1e3 if self.cuda else self.dt


class _Late:
    """``on_tokens`` one round late: a round's tokens go to the caller once the next verify window is queued on the
    GPU (``flush`` right after ``Engine.forward``), so the server's work on them (detokenizing, stop strings, the
    stream write) runs while the GPU verifies instead of between rounds, where the GPU would wait for it. The loops
    never read on_tokens' result themselves: a stop it asks for reaches every rank through ``StopVote`` (the next
    verify sample's all-gather), and the loops end after that round (``stopped``). Only when the tokens arrive
    changes: one round later, all of them before the loop returns (``flush`` again at its end)."""

    def __init__(self, on_tokens) -> None:
        self.fn, self.held = on_tokens, None

    def __call__(self, tokens) -> None:
        if self.fn is not None:
            self.flush()
            self.held = list(tokens)

    def flush(self) -> None:
        if self.held is not None:
            held, self.held = self.held, None
            self.fn(held)


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *,
                  stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    stages = dict(forward=0.0, sample=0.0, commit=0.0)
    clock, late = _Clock(), _Late(on_tokens)
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos) and not stopped(e):
        clock.start()
        logits = e.forward(e.verify_window([out[-1]]))
        clock.stop()
        late.flush()                        # the last step's token, while this step runs
        t1 = time.perf_counter()
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        e.follow([tok])
        t2 = time.perf_counter()
        commit(w, st, b, 1, 1)
        t3 = time.perf_counter()
        stages["forward"] += clock.seconds()
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        out.append(tok)
        late([tok])
    late.flush()
    _sync(w)
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, stages=stages)


class DepthPolicy:
    """Choose a fixed draft count or adapt it to running acceptance. DFlash2 stop rules within a round: the product of
    the chain's confidences holds ``confidence``; with ``beta`` > 0 those confidences are noise-aware
    (``dflash2.noisy_confidence``), and with ``round_ms`` too (ms of a round of k drafts, k = 0..) the chain is cut
    where E[tokens] / ms is largest instead. MTP chains ignore ``beta`` and ``round_ms``."""

    def __init__(self, most: int = 3, fixed: bool = False, low: float = 0.8, high: float = 0.9,
                 confidence: float = 0.0, beta: float = 0.0, round_ms: tuple[float, ...] | None = None) -> None:
        self.most, self.fixed, self.low, self.high = most, fixed, low, high
        self.confidence = confidence
        self.beta, self.round_ms = beta, round_ms
        self.rate = 0.8

    @property
    def chain_rule(self) -> dict:
        """Drafter.propose's keyword arguments for the noise-aware rules (none for the others)."""

        return {"beta": self.beta, "round_ms": self.round_ms} if self.beta > 0 else {}

    def next(self, drafted: int, accepted: int) -> int:
        if self.fixed:
            return self.most
        if drafted:
            self.rate = 0.875 * self.rate + 0.125 * (accepted / drafted)
        return max(1, min(self.most, 1 if self.rate < self.low else 2 if self.rate < self.high else 3))


def copy_room(copies, count: int, out: list[int]) -> int:
    """Copy drafts a round may propose: the rows still wanted past the pending token (0 without copy drafts)."""

    return 0 if copies is None else count - len(out) - 1


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, policy: DepthPolicy | None = None,
               stop_eos: bool = False, on_tokens=None, copies=None) -> DecodeResult:
    """Verify pending and MTP draft rows through the first mismatch, starting with every prompt position except the last in the MTP cache; ``copies``: a ``copy_drafts.CopyDrafts`` of prompt and pending token, whose proposals replace the MTP chain (the kept rows are still absorbed)."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy()
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    c_rounds = c_drafted = c_accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    clock, late = _Clock(), _Late(on_tokens)
    _sync(w)
    start = time.perf_counter()
    t0 = time.perf_counter()
    copied = copies.propose(copy_room(copies, count, out)) if copies is not None else []
    if copied:
        absorb(e, e.last_hidden, [pending])
        drafts = copied
    else:
        depth = min(policy.next(0, 0), count - len(out))
        drafts = draft(e, e.last_hidden, [pending], st.pos + 1, depth, sampling, policy.confidence) if depth > 0 else []
    stages["draft"] += time.perf_counter() - t0
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos) and not stopped(e):
        tokens = e.verify_window([out[-1]] + drafts)
        drafts = tokens[1:]
        R = len(tokens)
        clock.start()
        logits = e.forward(tokens)
        clock.stop()
        late.flush()                        # the last round's tokens, while this window runs
        t1 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t2 = time.perf_counter()
        commit(w, st, b, R, keep)
        t3 = time.perf_counter()
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        if copied:
            c_rounds, c_drafted, c_accepted = c_rounds + 1, c_drafted + len(drafts), c_accepted + keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        e.follow(sampled[:keep])
        out.extend(sampled[:keep])
        if copies is not None:
            copies.extend(sampled[:keep])
        late(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["forward"] += clock.seconds()
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos) or stopped(e):
            break
        t4 = time.perf_counter()
        was_copy = bool(copied)
        copied = copies.propose(copy_room(copies, count, out)) if copies is not None else []
        if copied:                          # the MTP head still takes the kept rows; it drafts nothing this round
            absorb(e, e.main_hidden(slice(0, keep)), sampled[:keep])
            drafts = copied
        else:                               # a copied round's acceptance does not move the MTP depth
            depth = min(policy.next(*((0, 0) if was_copy else (len(drafts), keep - 1))), count - len(out))
            drafts = (draft(e, e.main_hidden(slice(0, keep)), sampled[:keep], st.pos + 1, depth, sampling,
                            policy.confidence) if depth > 0 else [])
        stages["draft"] += time.perf_counter() - t4
    late.flush()
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps,
                        copy_rounds=c_rounds, copy_drafted=c_drafted, copy_accepted=c_accepted)


@torch.no_grad()
def dflash_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                  policy: DepthPolicy | None = None, stop_eos: bool = False, on_tokens=None,
                  copies=None) -> DecodeResult:
    """Verify pending and DFlash2 draft rows through the first mismatch and absorb kept taps, starting from prefill's drafter state; ``copies``: a ``copy_drafts.CopyDrafts`` of prompt and pending token, whose proposals replace the round's DFlash2 block (the kept taps are still absorbed)."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy(3, fixed=True)
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    c_rounds = c_drafted = c_accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    clock, late = _Clock(), _Late(on_tokens)
    _sync(w)
    start = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos) and not stopped(e):
        t0 = time.perf_counter()
        copied = copies.propose(copy_room(copies, count, out)) if copies is not None else []
        if copied:
            drafts = copied
        else:
            drafts = (drafter.propose(out[-1], depth, sampling, policy.confidence, **policy.chain_rule)
                      if depth > 0 else [])
        t1 = time.perf_counter()
        tokens = e.verify_window([out[-1]] + drafts)
        drafts = tokens[1:]
        R = len(tokens)
        clock.start()
        logits = e.forward(tokens)
        clock.stop()
        late.flush()                        # the last round's tokens, while this window runs
        t2 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t3 = time.perf_counter()
        commit(w, st, b, R, keep)
        t4 = time.perf_counter()
        drafter.add_taps(e.tap_rows(keep))
        t5 = time.perf_counter()
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        if copied:
            c_rounds, c_drafted, c_accepted = c_rounds + 1, c_drafted + len(drafts), c_accepted + keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        e.follow(sampled[:keep])
        out.extend(sampled[:keep])
        if copies is not None:
            copies.extend(sampled[:keep])
        late(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += clock.seconds()
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
        # a copied round's acceptance does not move DFlash2's depth
        depth = min(policy.next(*((0, 0) if copied else (len(drafts), keep - 1))), count - len(out))
    late.flush()
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps,
                        copy_rounds=c_rounds, copy_drafted=c_drafted, copy_accepted=c_accepted)
