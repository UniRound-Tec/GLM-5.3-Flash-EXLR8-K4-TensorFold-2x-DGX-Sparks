"""GLM-5.3-Flash's tensor-parallel forward and commit; kernels keep rows apart, so row r has the serial step's bits."""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.geometry import MLA_PROMPT_ATT_ROWS as PROMPT_ATT_ROWS   # a dense latent call's prompt rows
from tensorfold.cuda.kernels import prefill_attention, qmm as shared

from . import glue, kda as kda_mod, kv8, l2pf, latent, prof, qmm, sparse, tp, trial_control
from .attention import AttnScratch, attention, kv_write
from .weights import LayerW, Weights


# rows a prompt chunk's dense latent attention runs at once (its partials' rows; the startup estimate's)
from tensorfold.cuda.geometry import MLA_PROMPT_ATT_ROWS as PROMPT_ATT_ROWS  # noqa: E402


def prompt_split_k(w: Weights) -> bool:
    """Whether a prompt chunk's projections take split-K partials: an EXL3 checkpoint's BF16 / FP8 ones do; 4-bit
    ones (an MLX checkpoint, TF_GLM_DENSE=q4, read off the first layer's input projection) run without."""

    if w.cfg.quant not in ("exl3", "nvfp4_csf"):
        return False
    first = next((l.kda.proj if l.kda is not None else l.dsa.proj for l in w.layers
                  if getattr(l, "kda", None) is not None or getattr(l, "dsa", None) is not None), None)
    from .kda_source import MixedInput
    return not isinstance(first, (qmm.Q4,MixedInput))


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    def __init__(self, w: Weights, rows: int, capacity: int = 2560, *, prefill: bool = False, pad: int = 0) -> None:
        c = w.cfg
        dev = w.device
        bf, f32 = torch.bfloat16, torch.float32
        D, S = c.hidden, c.streams
        HL = tp.of(w).heads
        LL = tp.of(w).lin_heads
        self.rows, self.prefill = rows, prefill
        # ``pad``: rows held past the largest window, for the row split's pad rows (TF_GLM_HC_SPLIT past two ranks: a
        # chunk rounded up to whole shares a rank, ``hcsplit.pad_rows``); every buffer below holds them
        self.alloc_rows = rows = rows + int(pad)
        head_rows = 1 if prefill else rows
        self.world = w.world
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        if latent.ENABLED:
            # Dense attention only ever covers contexts up to the dense limit; longer rows go sparse.
            self.attn = None
            # a prompt chunk's dense pass runs PROMPT_ATT_ROWS rows at a time (``dense_attention``): its fp32
            # partials hold that many rows, not the chunk's
            self.lat_s = latent.LatentScratch(rows, HL, latent.chunks_for(min(capacity, 2560) + rows), dev,
                                              lw=c.kv_lora, part_rows=PROMPT_ATT_ROWS if prefill else rows)
            self.att_starts = (torch.arange(0, rows, self.lat_s.part_rows, dtype=torch.int32, device=dev)
                               if prefill else None)
        else:
            self.attn = AttnScratch(1 if prefill else rows, HL, c.qk_dim, capacity, dev)
        if prefill:
            kda_layers = [l for l in w.layers if l.kind == "kda"]
            width = kda_layers[0].kda.proj.n if kda_layers else 0
            self.kproj = torch.zeros((1, rows, width), dtype=bf, device=dev)
            self.kscratch = kda_mod.KDAScratch(rows, LL, dev)
        self.hin = torch.empty((rows, c.hidden), dtype=torch.bfloat16, device=dev)      # MTP input rows
        self.zero_first = False          # MTP: this step starts at position 0 (its embedding is zeroed)
        self.x = torch.empty((rows, S * D), dtype=bf, device=dev)
        self.normed = torch.empty((rows, D), dtype=bf, device=dev)
        self.xs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.post = torch.empty((rows, S), dtype=f32, device=dev)
        self.comb = torch.empty((rows, S * S), dtype=f32, device=dev)
        self.hcpart = torch.empty((rows, glue.HC_BLOCKS, 32), dtype=f32, device=dev)
        # KDA
        self.ka = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.kg = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.xs_fa = torch.empty((rows, 2), dtype=f32, device=dev)
        self.xs_ga = torch.empty((rows, 2), dtype=f32, device=dev)
        self.kxs = torch.empty((rows, LL * 128 // 64), dtype=f32, device=dev)
        # DSA
        self.dp = torch.empty((rows, c.q_lora + c.kv_lora), dtype=bf, device=dev)
        self.qr = torch.empty((rows, c.q_lora), dtype=bf, device=dev)
        self.xs_qr = torch.empty((rows, c.q_lora // 64), dtype=f32, device=dev)
        self.lat = torch.empty((rows, c.kv_lora), dtype=bf, device=dev)
        self.xs_lat = torch.empty((rows, c.kv_lora // 64), dtype=f32, device=dev)
        self.q = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.kn = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.vn = torch.empty((rows, HL, c.v_dim), dtype=bf, device=dev)
        self.xs_ao = torch.empty((rows, HL * c.v_dim // 64), dtype=f32, device=dev)
        # DSA indexer (long contexts)
        self.ikr = torch.empty((rows, c.index_dim + c.index_heads), dtype=bf, device=dev)
        self.igr = torch.empty((rows, c.index_dim), dtype=f32, device=dev)
        self.qi = torch.empty((rows, c.index_heads * c.index_dim), dtype=bf, device=dev)
        # dense MLP
        dl = tp.of(w).dense
        self.gu = torch.empty((rows, 2 * dl), dtype=bf, device=dev)
        self.act = torch.empty((rows, dl), dtype=bf, device=dev)
        self.xs_act = torch.empty((rows, dl // 64), dtype=f32, device=dev)
        # MoE
        slots = c.top_k + 1
        ml = tp.of(w).moe
        self.mlog = torch.empty((rows, c.experts), dtype=f32, device=dev)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=dev)
        self.wts = torch.empty((rows, slots), dtype=f32, device=dev)
        self.eact = None if c.quant == "nvfp4_csf" else torch.empty((rows * slots, ml), dtype=bf, device=dev)
        exl3 = c.quant == "exl3"
        self.ey = None if c.quant == "nvfp4_csf" else torch.empty((rows, slots, D), dtype=bf if prefill and not exl3 else f32, device=dev)
        self.exl3 = None
        prompt = False
        if c.quant == "exl3":            # EXL3 routed experts, and the shared expert as a BF16 MLP
            from .exl3_mm import Scratch, prompt_kernels, prompt_pass

            prompt = prefill and prompt_kernels()
            sl = tp.of(w).shared
            self.exl3 = Scratch(rows, slots, D, ml, dev, prompt=prompt)
            self.sgu = torch.empty((rows, 2 * sl), dtype=bf, device=dev)
            self.sact = torch.empty((rows, sl), dtype=bf, device=dev)
            self.sxs = torch.empty((rows, sl // 64), dtype=f32, device=dev)
            self.sy = torch.empty((rows, D), dtype=f32, device=dev)
        self.nvfp4 = None
        if c.quant == "nvfp4_csf":
            sl = tp.of(w).shared
            self.sgu = torch.empty((rows, 2 * sl), dtype=bf, device=dev)
            self.sact = torch.empty((rows, sl), dtype=bf, device=dev)
            self.sxs = torch.empty((rows, sl // 64), dtype=f32, device=dev)
            self.sy = torch.empty((rows, D), dtype=f32, device=dev)
            self.nvfp4 = w.nvfp4.buffers(rows)
        # a prompt window's EXL3 experts take items of 64 pairs (the prompt kernels' pass), decode windows 16
        self.plan = None if c.quant == "nvfp4_csf" else grouped.Plan(rows, slots, c.experts + 1, dev, prefill=prefill and (not exl3 or prompt),
                                 tile=prompt_pass() if prompt else None)
        # rank partials
        self.part = torch.empty((rows, D), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        # split-K partials of BF16 / FP8 projections (``mm``): a prompt chunk's 4-bit ones run the prompt matmul, which
        # keeps none, so an MLX checkpoint's and TF_GLM_DENSE=q4's prompt buffers hold none (the FP8 head's one row
        # allocates its few itself); a partials buffer holds scratch only, so where it lives never changes a bit
        self.sk = torch.empty((1 if prefill and not prompt_split_k(w) else 8 * rows * 16384,), dtype=f32, device=dev)
        # final
        self.hidden = torch.empty((rows, D), dtype=bf, device=dev)
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        self.fxs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.logits = torch.empty((head_rows, w.head.n), dtype=bf, device=dev)
        # MTP
        self.me = torch.empty((rows, D), dtype=bf, device=dev)
        self.mcat = torch.empty((rows, 2 * D), dtype=bf, device=dev)
        self.mxs = torch.empty((rows, 2 * D // 64), dtype=f32, device=dev)
        self.mx = torch.empty((rows, D), dtype=bf, device=dev)
        self._parents: dict[int, torch.Tensor] = {}
        # DFlash2 taps: the mean of the streams after chosen layers (``set_taps``), filled by every forward
        self.taps: list[torch.Tensor] = []
        self.tap_at: dict[int, list[int]] = {}
        self.experts = c.experts
        self.top_k = c.top_k
        # TF_GLM_HC_SPLIT: a prompt buffer's row split (``hcsplit.HcSplit``), set by the engine; None runs unsplit
        self.split = None

    def set_taps(self, layers: tuple[int, ...], hidden: int) -> None:
        self.tap_at = {}
        for i, layer in enumerate(layers):
            self.tap_at.setdefault(layer, []).append(i)
        self.taps = [torch.empty((self.alloc_rows, hidden), dtype=torch.bfloat16, device=self.ids.device)
                     for _ in layers]

    def parents(self, R: int) -> torch.Tensor:
        p = self._parents.get(R)
        if p is None:
            p = torch.arange(-1, R - 1, dtype=torch.int32, device=self.ids.device)
            self._parents[R] = p
        return p


def index_ring(capacity: int, rows: int) -> int:
    """Rows of the indexer's key and gate rings (``sparse.index_update``) for windows of up to ``rows`` rows: those
    and the 3 before them, rounded up to 64; the context when that is less (then no row wraps)."""
    return min(capacity, -(-(rows + 3) // 64) * 64)


class Caches:
    """Every per-token cache of the model over ``rows`` tokens of a shared pool (``pool.Arena``): per DSA layer its
    latents (or keys and values without the latent cache); the MTP layer's; and with long contexts per indexed layer
    (the DSA layers, then the MTP layer) its index keys, gates and pooled keys (a row a pool of 4 tokens, 2 rows of
    pad). Element types and layouts are decided here; a ``State`` views them over its extent (``bind``).

    ``ring``: the index keys and gates (read only by the pool a token falls in, while it fills) are not arena planes
    but one ring of ``ring`` rows per stream slot (``streams`` of them, ``index_ring``): token t of the stream in slot
    s at row s x ring + t % ring. They stay with the slot when its extent moves; a kept prompt carries the rows of its
    pools still filling (``decode.Snapshot.tail``). None: arena planes (token t at its extent's row).

    ``kv``: the latent planes' and pooled keys' format (TF_GLM_KV, ``kv8``): bf16, or fp8 (a uint8 row a token /
    pool: e4m3 codes, an fp32 power-of-two scale, pad), with the latent cache only.

    ``display``: bytes of the GB10's display reservation the pool was sized with (``display_kv``); the first DSA
    layers' latent planes are then carved from one span that holds them."""

    def __init__(self, w: Weights, rows: int, *, streams: int = 1, ring: int | None = None,
                 kv: str = "bf16", display: int = 0) -> None:
        from .pool import Arena, Plane

        c = w.cfg
        dev = w.device
        HL = tp.of(w).heads
        dsa = len([l for l in w.layers if l.kind == "dsa"])
        self.latent = latent.ENABLED
        if kv not in kv8.KINDS or (kv != "bf16" and not self.latent):
            raise ValueError(f"TF_GLM_KV={kv}: bf16, or fp8 with the latent cache (TF_GLM_LATENT=1)")
        self.kv = kv
        planes: list[Plane] = []
        spanned: list[torch.Tensor] = []
        if display:
            if not self.latent:
                raise ValueError("TF_GLM_DISPLAY_KV_MIB needs the latent cache (TF_GLM_LATENT=1)")
            from . import display_kv

            spanned = display_kv.latent_planes(dsa, rows, c.kv_lora, kv, dev, display)

        def add(shape, div: int = 1, pad: int = 0, fmt: str = "bf16", tensor=None) -> int:
            n = rows // div + pad
            t = tensor if tensor is not None else (kv8.zeros(n, shape[0], fmt, dev) if fmt != "bf16" else
                                                   torch.zeros((n, *shape), dtype=torch.bfloat16, device=dev))
            planes.append(Plane(t, div, pad, spanned=tensor is not None))
            return len(planes) - 1

        def attention_planes() -> tuple[int, int | None]:
            if self.latent:          # one 512-wide latent a token and layer (kc), no separate values (vc)
                return add((c.kv_lora,), fmt=kv, tensor=spanned.pop(0) if spanned else None), None
            return add((HL, c.qk_dim)), add((HL, c.v_dim))

        self.kc, self.vc = [], []
        for _ in range(dsa):
            k, v = attention_planes()
            self.kc.append(k)
            self.vc.append(v)
        self.mtp = attention_planes() if w.mtp is not None else None
        # DSA indexer caches (long contexts only): per layer (and the MTP layer, last) keys, gates, pool keys; with a
        # ring, index[i] = (None, None, pool plane) and rings[i] = (keys, gates) [streams x ring, 128]
        self.index = None
        self.rings = None
        self.ring = None if ring is None else int(ring)
        if self.ring is not None and self.ring < 4:
            raise ValueError(f"index rings of {self.ring} rows")
        if w.meta.get("long_context"):
            n_idx = dsa + (1 if w.mtp is not None else 0)
            if self.ring is None:
                self.index = [(add((c.index_dim,)), add((c.index_dim,)), add((c.index_dim,), 4, 2, kv))
                              for _ in range(n_idx)]
            else:
                self.index = [(None, None, add((c.index_dim,), 4, 2, kv)) for _ in range(n_idx)]
                ring_rows = int(streams) * self.ring
                self.rings = [tuple(torch.zeros((ring_rows, c.index_dim), dtype=torch.bfloat16, device=dev)
                                    for _ in range(2)) for _ in range(n_idx)]
        self.arena = Arena(rows, planes)
        self.rows = rows

    def index_tensors(self, i: int) -> tuple:
        """Indexed layer i's (keys, gates, pooled keys) over every stream: the rings (or planes) and the pool plane."""

        planes = self.arena.planes
        pk = planes[self.index[i][2]].tensor
        if self.rings is None:
            return planes[self.index[i][0]].tensor, planes[self.index[i][1]].tensor, pk
        return self.rings[i][0], self.rings[i][1], pk

    def ring_base(self, slot: int) -> int:
        """The first ring row of stream slot ``slot`` (index rows at ring_base + t % ring)."""
        return 0 if self.ring is None else int(slot) * self.ring

    def ring_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for pair in (self.rings or []) for t in pair)

    def bind(self, st: "State", base: int, size: int) -> None:
        """Point ``st``'s caches at tokens [base, base + size) of the pool."""

        view = lambda i: None if i is None else self.arena.view(i, base, size)     # noqa: E731
        st.kc = [view(i) for i in self.kc]
        st.vc = [view(i) for i in self.vc]
        if self.mtp is not None:
            st.mtp_kc, st.mtp_vc = view(self.mtp[0]), view(self.mtp[1])
        if self.index is None:
            st.index = None
        elif self.rings is None:
            st.index = [tuple(view(i) for i in trio) for trio in self.index]
        else:                        # the slot's rings (index_update's ring: their rows) and the extent's pools
            r0 = self.ring_base(st.slot)
            st.index = [(ik[r0:r0 + self.ring], ig[r0:r0 + self.ring], view(trio[2]))
                        for (ik, ig), trio in zip(self.rings, self.index)]


class Slots:
    """What each of ``count`` concurrent streams holds whatever its length: KDA recurrent states [count, 2, layers,
    H, 128, 128] (two parities) and conv windows [count, layers, 3, C], the decode window's KDA projections and replay
    scratch (so a window's commit can come after another stream's forward), and the device positions a captured graph
    reads. A ``State`` in slot s views slot s."""

    def __init__(self, w: Weights, count: int, rows: int) -> None:
        c = w.cfg
        dev = w.device
        LL = tp.of(w).lin_heads
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        n = len(kda_layers)
        width = kda_layers[0].kda.proj.n if kda_layers else 0
        self.count, self.rows, self.layers = count, rows, n
        self.conv = torch.zeros((count, n, c.conv - 1, 3 * LL * 128), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((count, 2, n, LL, 128, 128), dtype=torch.float32, device=dev)
        self.proj = torch.zeros((count, n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch = [kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None for _ in range(count)]
        self.pos_dev = torch.zeros((count, 1), dtype=torch.int32, device=dev)
        self.mtp_pos_dev = torch.zeros((count, 1), dtype=torch.int32, device=dev)

    def nbytes(self) -> int:
        t = [self.conv, self.rec, self.proj, self.pos_dev, self.mtp_pos_dev]
        s = [x for set_ in self.scratch if set_ is not None for x in (set_.out, set_.k, set_.v, set_.g, set_.b)]
        return sum(x.numel() * x.element_size() for x in t + s)


def slot_bytes(w: Weights, rows: int) -> int:
    """Device bytes one more stream slot takes (``Slots``), without allocating it."""

    c = w.cfg
    LL = tp.of(w).lin_heads
    kda_layers = [l for l in w.layers if l.kind == "kda"]
    n = len(kda_layers)
    width = kda_layers[0].kda.proj.n if kda_layers else 0
    DK = DV = 128
    scratch = rows * LL * DV * 2 + rows * LL * DK * 4 + rows * LL * DV * 2 + rows * LL * DK * 4 + rows * LL * 4
    return (n * (c.conv - 1) * 3 * LL * 128 * 2 + 2 * n * LL * 128 * 128 * 4 + n * rows * width * 2 + n * scratch
            + 16)


class State:
    """Committed caches of one sequence (and of the MTP head's attention layer): views of its extent [base, base +
    capacity) of the pool's ``Caches`` and of its stream slot (``Slots``). Without ``caches`` / ``slots`` it makes its
    own (one sequence of ``capacity`` tokens, one slot), as it always held tensors of its own."""

    def __init__(self, w: Weights, capacity: int, rows: int, *, caches: Caches | None = None, base: int = 0,
                 slots: Slots | None = None, slot: int = 0, kv: str = "bf16") -> None:
        self.caches = caches if caches is not None else Caches(w, capacity, kv=kv)
        self.slots = slots if slots is not None else Slots(w, 1, rows)
        self.slot = slot
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        dsa_layers = [l for l in w.layers if l.kind == "dsa"]
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.dsa_index = {l.index: i for i, l in enumerate(dsa_layers)}
        n = len(kda_layers)
        sl = self.slots
        self.pos = 0
        self.pos_dev = sl.pos_dev[slot]
        self.mtp_pos_dev = sl.mtp_pos_dev[slot]
        self.conv = sl.conv[slot]
        self.rec = sl.rec[slot]
        self.cur = [0] * n
        self.proj = sl.proj[slot]
        self.scratch_set = sl.scratch[slot]
        self.scratch = self.scratch_set.views if n else []
        self.latent = latent.ENABLED
        self.mtp_len = 0
        self.mtp_drafted = 0
        self.bind(base, capacity)

    def bind(self, base: int, capacity: int) -> None:
        """View the pool's tokens [base, base + capacity) (a new extent, grown or moved: the rows are the caller's)."""

        self.base, self.capacity = int(base), int(capacity)
        self.caches.bind(self, self.base, self.capacity)

    @property
    def kv(self) -> str:
        """The DSA caches' format (TF_GLM_KV): the pool's."""
        return self.caches.kv

    @property
    def graph_key(self) -> tuple[int, int]:
        """What a captured graph baked in besides the shared buffers: the slot and the extent's base."""
        return self.slot, self.base

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0] * len(self.cur)
        self.set_pos(0)
        self.set_mtp_len(0)
        self.mtp_drafted = 0

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n
        self.mtp_pos_dev.fill_(n)

    @property
    def parity(self) -> int:
        return self.cur[0] if self.cur else 0

    def clone(self) -> "State":
        """A copy with tensors of its own (no longer the pool's or the slot's views; not for captured graphs)."""
        import copy

        other = copy.copy(self)
        other.proj = self.proj.clone()
        other.conv = self.conv.clone()
        other.rec = self.rec.clone()
        other.cur = list(self.cur)
        other.pos_dev = self.pos_dev.clone()
        other.mtp_pos_dev = self.mtp_pos_dev.clone()
        other.kc = [x.clone() for x in self.kc]
        other.vc = [x.clone() if x is not None else None for x in self.vc]
        if self.index is not None:
            other.index = [tuple(x.clone() for x in trio) for trio in self.index]
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_vc = self.mtp_vc.clone() if self.mtp_vc is not None else None
        return other


# -- blocks ---------------------------------------------------------------------------------------------------
def gather(w: Weights, b: Buffers, R: int) -> torch.Tensor:
    """Every rank's fp32 partial b.part[:R] in rank order: [world, R, D] (summed rank 0 first by the consumer)."""

    d = b.part.shape[1]
    if w.comm is None:
        return b.part[:R].view(1, R, d)
    out = b.gath[:b.world * R * d]
    w.comm.all_gather(b.part[:R].reshape(-1), out)
    return out.view(b.world, R, d)


def mm(b: Buffers, x: torch.Tensor, q, xs: torch.Tensor | None, out: torch.Tensor, f32: bool = False) -> torch.Tensor:
    """A projection: 4-bit ones of a prompt chunk on the shared prefill matmul, the rest on ``qmm.matmul``."""

    from . import quant_audit
    quant_audit.capture(b,x,q,f32)
    from . import kda_source as ks
    if isinstance(q,(ks.MixedInput,ks.K4Output,ks.sf.SourceF8)):
        return ks.matmul(x,q,xs,out=out,f32=f32,part=b.sk,prefill=b.prefill)
    if b.prefill and isinstance(q, qmm.Q4):
        return shared.prefill_matmul(x, q, f32=f32, out=out)
    return qmm.matmul(x, q, xs, out=out, f32=f32, part=b.sk)


def partials(w: Weights, b: Buffers, R: int, fill, label: str | None = None, site: tuple | None = None
             ) -> torch.Tensor | None:
    """``fill(lo, hi)`` writes rows lo .. hi of this rank's fp32 partial b.part; returns every rank's partials
    [world, R, D] (``gather``), or None while a prompt chunk runs row-split (TF_GLM_HC_SPLIT), which exchanges them
    and runs hc_post itself (``hcsplit.HcSplit``)."""

    sp = b.split
    if sp is not None and sp.active:
        sp.partial(fill)
        return None
    fill(0, R)
    if site is not None and l2pf.ACTIVE is not None:      # TF_GLM_L2PF: the next reads into L2 during the gather
        l2pf.ACTIVE.site(*site)
    if label is None:
        return gather(w, b, R)
    with prof.timed(label):
        return gather(w, b, R)


def out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, R: int,
             site: tuple | None = None) -> torch.Tensor | None:
    def fill(lo: int, hi: int) -> None:          # rows are independent: any row range gives the same bits
        mm(b, x[lo:hi], q, None if xs is None else xs[lo:hi], b.part[lo:hi], f32=True)

    return partials(w, b, R, fill, site=site)


def undone(R: int, done: tuple[int, int] | None) -> list[tuple[int, int]]:
    """The row ranges of [0, R) outside ``done`` (a block's front already run on those rows, TF_GLM_PREFILL_OVERLAP=2),
    in row order; every row when ``done`` is None."""

    if done is None:
        return [(0, R)]
    lo, hi = done
    return [(a, z) for a, z in ((0, min(lo, R)), (max(hi, 0), R)) if z > a]


def kda_front(layer: LayerW, b: Buffers, lo: int, hi: int) -> None:
    """A prompt chunk's KDA input projections of rows lo .. hi (row-independent: the prompt matmul's bits for any
    row range; BF16 / FP8 projections of an EXL3 checkpoint keep rows apart too)."""

    k = layer.kda
    p = b.kproj[0, lo:hi]
    mm(b, b.normed[lo:hi], k.proj, b.xs[lo:hi], p)
    mm(b, p[:, k.fa_off:k.fa_off + 128], k.fb, None, b.ka[lo:hi])
    mm(b, p[:, k.ga_off:k.ga_off + 128], k.gb, None, b.kg[lo:hi])


def kda_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, done: tuple[int, int] | None = None
              ) -> torch.Tensor:
    """``done``: rows whose projections ``kda_front`` already wrote (a prompt chunk, TF_GLM_PREFILL_OVERLAP=2)."""

    c = w.cfg
    k = layer.kda
    li = st.kda_index[layer.index]
    p = b.kproj[0, :R] if b.prefill else st.proj[li, :R]
    pre = b.prefill
    with prof.timed("kda: projections"):
        if pre:
            for lo, hi in undone(R, done):
                kda_front(layer, b, lo, hi)
        else:
            mm(b, b.normed[:R], k.proj, b.xs[:R], p)
            fa = p[:, k.fa_off:k.fa_off + 128]
            ga = p[:, k.ga_off:k.ga_off + 128]
            from .trial_control import projections
            projections(b, fa, ga, k.fb, k.gb, R, mm)
            l2pf.site(layer.index, "o")
    if pre:                              # a prompt chunk keeps every row: the layer commits now
        out = kda_rows(layer, w, st, b, 0, R)
    else:
        cur = st.cur[li]
        with prof.timed("kda: recurrence"):
            out = kda_mod.chain(p, k.b_off, b.ka[:R], b.kg[:R], st.conv[li], k.conv, st.rec[cur, li], k.a_log,
                                k.dt_bias, k.norm, c.eps, c.lower, R, st.scratch[li], st.rec[1 - cur, li])
    with prof.timed("kda: out + all-gather"):
        return out_proj(w, b, out, k.o, None if pre else qmm.group_sums_for(k.o, out, b.kxs[:R]), R, site=(layer.index, "a"))


def _scratch_rows(s, lo: int):
    """A prompt buffer's KDA scratch from row ``lo`` on (the same storage)."""

    if lo == 0:
        return s
    from types import SimpleNamespace

    return SimpleNamespace(out=s.out[lo:], k=s.k[lo:], v=s.v[lo:], g=s.g[lo:], b=s.b[lo:])


def kda_rows(layer: LayerW, w: Weights, st: State, b: Buffers, lo: int, n: int) -> torch.Tensor:
    """A prompt chunk's KDA recurrence for rows lo .. lo + n of the prompt buffers (their projections done) on ``st``
    from its position: the chain into b.kscratch.out[lo:lo + n] (returned), the layer's state committed and its conv
    window shifted. ``kda_block`` runs a chunk as rows 0 .. R; a chunk of several streams' prompts
    (``multi_prefill``) runs each stream's rows through this on that stream's state: the call they get alone."""

    c = w.cfg
    k = layer.kda
    li = st.kda_index[layer.index]
    rs = slice(lo, lo + n)
    cur = st.cur[li]
    with prof.timed("kda: recurrence"):
        out = kda_mod.chain(b.kproj[0, rs], k.b_off, b.ka[rs], b.kg[rs], st.conv[li], k.conv, st.rec[cur, li],
                            k.a_log, k.dt_bias, k.norm, c.eps, c.lower, n, _scratch_rows(b.kscratch, lo),
                            st.rec[1 - cur, li], pos=st.pos)
    st.cur[li] = 1 - cur
    _shift_conv(st.conv[li:li + 1], b.kproj[:, rs], n)
    return out


def kda_segments(layer: LayerW, w: Weights, b: Buffers, R: int, seg: torch.Tensor, slots: "Slots",
                 proj: torch.Tensor, scratch, li: int) -> torch.Tensor:
    """kda_block's decode path for a multi-stream window (``seg``: kda.segment_table, set): the projections on every
    row at once (``proj`` [layers, rows, width]: the window's, which the commit's conv shift reads), the chain per
    segment from its slot's state (``kda.chain_segments``: each segment's bits alone; always the wide path, as
    decode windows run by default), ``scratch``: the window's replay scratch (a KDAScratchSet)."""

    c = w.cfg
    k = layer.kda
    p = proj[li, :R]
    with prof.timed("kda: projections"):
        mm(b, b.normed[:R], k.proj, b.xs[:R], p)
        fa = p[:, k.fa_off:k.fa_off + 128]
        ga = p[:, k.ga_off:k.ga_off + 128]
        from .trial_control import projections
        projections(b, fa, ga, k.fb, k.gb, R, mm)
        l2pf.site(layer.index, "o")
    with prof.timed("kda: recurrence"):
        out = kda_mod.chain_segments(seg, p, k.b_off, b.ka[:R], b.kg[:R], slots.conv[:, li], k.conv,
                                     slots.rec[:, :, li], k.a_log, k.dt_bias, k.norm, c.eps, c.lower, R,
                                     scratch.views[li])
    with prof.timed("kda: out + all-gather"):
        return out_proj(w, b, out, k.o, qmm.group_sums_for(k.o, out, b.kxs[:R]), R, site=(layer.index, "a"))


def dsa_front(layer: LayerW, w: Weights, b: Buffers, lo: int, hi: int) -> None:
    """DSA's projection, query / latent norms and query expansion of rows lo .. hi (row-independent kernels)."""

    c = w.cfg
    a = layer.dsa
    mm(b, b.normed[lo:hi], a.proj, b.xs[lo:hi], b.dp[lo:hi])
    glue.rmsnorm(b.dp[lo:hi, :c.q_lora], a.q_norm, c.eps, b.qr[lo:hi], b.xs_qr[lo:hi])
    glue.rmsnorm(b.dp[lo:hi, c.q_lora:], a.kv_norm, c.eps, b.lat[lo:hi], b.xs_lat[lo:hi])
    mm(b, b.qr[lo:hi], a.q_b, b.xs_qr[lo:hi], b.q[lo:hi].view(hi - lo, a.heads * c.qk_dim))


def dsa_block(layer: LayerW, w: Weights, kc: torch.Tensor, vc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers,
              R: int, nch: int | None, index=None, host_pos: int | None = None,
              sparse_np: int | None = None, done: tuple[int, int] | None = None) -> torch.Tensor:
    """Write every index key and pool; rows past the dense limit attend to their top-512 pools (host_pos eager, or sparse_np in a captured graph). ``done``: rows ``dsa_front`` already ran on (TF_GLM_PREFILL_OVERLAP=2)."""

    c = w.cfg
    a = layer.dsa
    for lo, hi in undone(R, done):
        dsa_front(layer, w, b, lo, hi)
    HL = a.heads
    if a.absorb is not None:
        return _dsa_latent(a, w, kc, pos_dev, b, R, nch, index, host_pos, sparse_np, layer=layer.index)
    if sparse_np is not None:
        raise ValueError("sparse CUDA graphs need the latent cache (TF_GLM_LATENT=1)")
    mm(b, b.lat[:R], a.kv_k, b.xs_lat[:R], b.kn[:R].view(R, HL * c.qk_dim))
    mm(b, b.lat[:R], a.kv_v, b.xs_lat[:R], b.vn[:R].view(R, HL * c.v_dim))
    kv_write(b.kn[:R], b.vn[:R], kc, vc, pos_dev)
    sparse_rows = index is not None and host_pos is not None and host_pos + R - 1 >= c.dense_limit
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
        glue.router(b.normed[:R], ix.gate, b.igr[:R])
        sparse.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    if sparse_rows and host_pos >= c.dense_limit:     # every row sparse: the dense pass is skipped
        o = torch.empty((R, HL, c.v_dim), dtype=torch.bfloat16, device=b.q.device) if b.prefill else b.attn.out[:R]
    elif b.prefill:
        o = prefill_attention.attention(b.q[:R], kc, vc, host_pos, scale=c.qk_dim ** -0.5)
    else:
        o = attention(b.q[:R], kc, vc, pos_dev, b.attn, scale=c.qk_dim ** -0.5, nch=nch)
    if sparse_rows:
        mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
        tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, host_pos, R,
                                              pk.shape[0] - 2, pos_dev)
        sparse.sparse_attention(b.q[:R], kc, vc, tokens, counts, o, c.qk_dim ** -0.5)
    o = o.view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, None if b.prefill else qmm.group_sums_for(a.o, o, b.xs_ao[:R]), R)


def dense_rows(c, R: int, host_pos: int | None) -> int:
    """Rows of a window from ``host_pos`` that keep dense attention: those before the first position select_tokens
    gives a sparse count (sparse_attention overwrites every row from there). All R when the position is not known
    on the host or the config's dense limit is not the selection's."""

    if host_pos is None or c.dense_limit != sparse.SPARSE_FROM:
        return R
    return min(R, max(0, sparse.SPARSE_FROM - host_pos))


def dense_attention(qa: torch.Tensor, lc: torch.Tensor, pos_dev: torch.Tensor, s, *, scale: float, nch: int,
                    out: torch.Tensor, hb: int, starts: torch.Tensor | None = None) -> torch.Tensor:
    """``latent.attention`` of qa's rows in blocks of the scratch's ``part_rows`` (block i's queries at device
    position pos_dev + starts[i], ``starts`` = arange(0, rows, part_rows)): every row gets the bits of one call over
    all of them, since a row's programs read only its query, keys up to its position and its own partials, and
    ``hb`` and ``nch`` stay the window's (``latent.attention``)."""

    R, step = qa.shape[0], s.part_rows
    if R <= step:
        return latent.attention(qa, lc, pos_dev, s, scale=scale, nch=nch, out=out, hb=hb)
    if starts is None or starts.numel() < -(-R // step):
        raise ValueError(f"dense attention of {R} rows in blocks of {step} needs the blocks' start offsets")
    at = pos_dev + starts                                  # each block's first row's position, on the device
    for i, r0 in enumerate(range(0, R, step)):
        r1 = min(R, r0 + step)
        latent.attention(qa[r0:r1], lc, at[i:i + 1], s, scale=scale, nch=nch, out=out[r0:r1], hb=hb)
    return out


def _dsa_latent(a, w: Weights, lc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers, R: int, nch: int | None,
                index, host_pos: int | None, sparse_np: int | None = None, layer: int = -1) -> torch.Tensor:
    """DSA on the latent cache: the same indexer and selection, attention over latents with kv_b's key blocks absorbed into the query."""

    o = dsa_rows(a, w, lc, pos_dev, b, R, nch, index, host_pos, sparse_np, layer)
    return out_proj(w, b, o, a.o, qmm.group_sums_for(a.o, o, b.xs_ao[:R]), R, site=(layer, "a"))


def dsa_rows(a, w: Weights, lc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers, R: int, nch: int | None,
             index, host_pos: int | None, sparse_np: int | None = None, layer: int = -1, lo: int = 0) -> torch.Tensor:
    """``_dsa_latent`` up to its output projection, for rows lo .. lo + R of the buffers (their front done) on one
    stream's caches (``lc``, ``index``, ``pos_dev``, ``host_pos``: its first row's position): the latent write,
    indexer update, absorb, attention and expand; returns the rows' attention output [R, heads * v_dim] (b.vn's rows).
    A chunk of several streams' prompts (``multi_prefill``) runs each stream's rows through this: the call they get
    alone."""

    c = w.cfg
    HL = a.heads
    s = b.lat_s
    rs = slice(lo, lo + R)
    with prof.timed("dsa: latent write"):
        latent.latent_write(b.lat[rs], lc, pos_dev)
    # sparse_np: every row is past the dense limit (a captured graph); else the host position decides
    all_sparse = sparse_np is not None or (host_pos is not None and host_pos >= c.dense_limit)
    sparse_rows = index is not None and (all_sparse or (host_pos is not None and host_pos + R - 1 >= c.dense_limit))
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        with prof.timed("dsa: indexer update"):
            mm(b, b.normed[rs], ix.kw, b.xs[rs], b.ikr[rs])
            glue.router(b.normed[rs], ix.gate, b.igr[rs])
            sparse.index_update(b.ikr[rs, :c.index_dim], b.igr[rs], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    with prof.timed("dsa: absorb"):
        qa = latent.absorb_q(b.q[rs], a.absorb, s.qa[rs], prompt=b.prefill and latent.PROMPT_MMA)
    l2pf.site(layer, "o")
    ol = s.ol[rs]
    scale = c.qk_dim ** -0.5
    if not all_sparse:
        # Rows past the dense limit are recomputed sparsely below, so the dense pass needs only the chunks up to it
        # and only the rows before it (a row's dense bits do not depend on the rows after it: latent.attention).
        dense = dense_rows(c, R, host_pos) if sparse_rows else R
        with prof.timed("dsa: dense attention"):
            dense_attention(qa[:dense], lc, pos_dev, s, scale=scale, nch=min(nch or s.nch, s.nch), out=ol[:dense],
                            hb=latent.head_block(R), starts=b.att_starts if b.prefill else None)
    if sparse_rows:
        with prof.timed("dsa: select tokens"):
            mm(b, b.qr[rs], ix.qb, b.xs_qr[rs], b.qi[rs])
            if b.prefill and sparse_np is None and trial_control.ACTIVE['index_split']:
                from .index_split import select
                tokens, counts = select(w.comm, b.qi[rs], b.ikr[rs, c.index_dim:], pk, host_pos, R,
                                        pk.shape[0] - 2, pos_dev)
            else:
                tokens, counts = sparse.select_tokens(b.qi[rs], b.ikr[rs, c.index_dim:], pk, host_pos, R,
                                                      pk.shape[0] - 2, pos_dev, bucket=sparse_np)
        with prof.timed("dsa: sparse attention"):
            if b.prefill and trial_control.ACTIVE['mla_variant'] == 1 and qa.shape[1:] == (32, 512) and lc.dtype == torch.uint8:
                from .atlas_mla import run
                run(qa, lc, tokens, counts, ol, scale)
            elif b.prefill and latent.SPARSE_ONEPASS:     # prompt rows: one pass a row, no partials (their own bits)
                latent.sparse_onepass(qa, lc, tokens, counts, ol, scale)
            else:
                latent.sparse_attention(qa, lc, tokens, counts, ol, scale)
    with prof.timed("dsa: expand"):
        return latent.expand_v(ol, a.absorb, b.vn[rs], prompt=b.prefill and latent.PROMPT_MMA).view(R, HL * c.v_dim)


def dsa_segments(layer: LayerW, w: Weights, lc: torch.Tensor, b: Buffers, R: int, rows, sel=None, index=None, *,
                 select: bool = True) -> torch.Tensor | None:
    """dsa_block on the latent cache for a multi-stream window (``rows``: segments.SegRows, already set): several
    streams' rows back to back, each stream's latents (lc), index keys, gates and pooled keys (index: arenas) in
    its own extent. Each segment's rows get the bits of that segment alone through dsa_block (the projections,
    absorb and expand keep rows apart; the attention kernels are latent.seg_* / sparse.seg_*). Graphs: one per R
    covers every position mix. ``select`` False skips the token selection (eager windows with no sparse row:
    ``rows.any_sparse()``); ``sel``: segments.SelectScratch, needed with an index."""

    c = w.cfg
    a = layer.dsa
    if a.absorb is None:
        raise ValueError("segmented windows need the latent cache (TF_GLM_LATENT=1)")
    mm(b, b.normed[:R], a.proj, b.xs[:R], b.dp[:R])
    glue.rmsnorm(b.dp[:R, :c.q_lora], a.q_norm, c.eps, b.qr[:R], b.xs_qr[:R])
    glue.rmsnorm(b.dp[:R, c.q_lora:], a.kv_norm, c.eps, b.lat[:R], b.xs_lat[:R])
    HL = a.heads
    mm(b, b.qr[:R], a.q_b, b.xs_qr[:R], b.q[:R].view(R, HL * c.qk_dim))
    s = b.lat_s
    with prof.timed("dsa: latent write"):
        latent.seg_latent_write(b.lat[:R], lc, rows)
    tokens = counts = None
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        with prof.timed("dsa: indexer update"):
            mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
            glue.router(b.normed[:R], ix.gate, b.igr[:R])
            sparse.seg_index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, rows)
        if select:
            with prof.timed("dsa: select tokens"):
                mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
                tokens, counts = sparse.seg_select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, rows, sel)
    with prof.timed("dsa: absorb"):
        qa = latent.absorb_q(b.q[:R], a.absorb, s.qa[:R])
    l2pf.site(layer.index, "o")
    with prof.timed("dsa: attention"):
        ol = latent.seg_attention(qa, lc, rows, tokens, counts, s, scale=c.qk_dim ** -0.5, out=s.ol[:R])
    with prof.timed("dsa: expand"):
        o = latent.expand_v(ol, a.absorb, b.vn[:R]).view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, qmm.group_sums_for(a.o, o, b.xs_ao[:R]), R, site=(layer.index, "a"))


def mlp_front(layer: LayerW, w: Weights, b: Buffers, lo: int, hi: int) -> None:
    m = layer.mlp
    mm(b, b.normed[lo:hi], m.gu, b.xs[lo:hi], b.gu[lo:hi])
    glue.swiglu(b.gu[lo:hi], b.act[lo:hi], b.xs_act[lo:hi], w.cfg.limit)


def mlp_block(layer: LayerW, w: Weights, b: Buffers, R: int, done: tuple[int, int] | None = None) -> torch.Tensor:
    m = layer.mlp
    for lo, hi in undone(R, done):
        mlp_front(layer, w, b, lo, hi)
    return out_proj(w, b, b.act[:R], m.down, b.xs_act[:R], R, site=(layer.index, "f"))


def shared_front(layer: LayerW, w: Weights, b: Buffers, lo: int, hi: int) -> None:
    """The MoE's shared expert (an EXL3 checkpoint's, BF16 / FP8 / 4-bit matmuls) on rows lo .. hi: b.sy."""

    s, c = layer.moe.shared, w.cfg
    with prof.timed("moe: shared expert"):
        mm(b, b.normed[lo:hi], s.gu, b.xs[lo:hi], b.sgu[lo:hi])
        glue.swiglu(b.sgu[lo:hi], b.sact[lo:hi], b.sxs[lo:hi], c.limit)
        mm(b, b.sact[lo:hi], s.down, b.sxs[lo:hi], b.sy[lo:hi], f32=True)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int, done: tuple[int, int] | None = None) -> torch.Tensor:
    """``done``: rows whose shared expert ``shared_front`` already ran (TF_GLM_PREFILL_OVERLAP=2)."""

    c = w.cfg
    m = layer.moe
    with prof.timed("moe: route"):
        glue.router(b.normed[:R], m.router, b.mlog[:R])
        glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts, c.routed_scale, c.norm_topk)
        if c.quant != "nvfp4_csf":
            grouped.route(b.pick[:R], b.plan, b.plan.tile)   # the plan keeps its pass (TF_GLM_EXL3_PASS)
    if c.quant == "nvfp4_csf":
        with prof.timed("moe: routed (nvfp4)"):
            routed = b.nvfp4.run(m.experts, b.normed[:R], b.pick[:R, :c.top_k], b.wts[:R, :c.top_k])

        def fill(lo: int, hi: int) -> None:
            for a, z in undone(hi, done):
                if max(a, lo) < z:
                    shared_front(layer, w, b, max(a, lo), z)
            # B12X already multiplies routed probabilities and reduces experts.
            # Shared experts and the existing rank reduction remain in TensorFold.
            torch.add(routed[lo:hi], b.sy[lo:hi], out=b.part[lo:hi])

        return partials(w, b, R, fill, "moe: all-gather", site=(layer.index, "f"))
    if m.shared is not None:
        from . import shared_fusion
        if shared_fusion.enabled(m,b,R,done):
            shared_fusion.run(b.normed[:R],b.pick,b.plan,m.experts,b.exl3,b.ey.view(-1,c.hidden),R,c.limit)
            def fill_merged(lo,hi):
                glue.combine(b.ey[lo:hi],b.wts[lo:hi],b.part[lo:hi])
            return partials(w,b,R,fill_merged,"moe: all-gather",site=(layer.index,"f"))
        # EXL3: the routed slots through the trellis kernels, the shared expert (last slot) through BF16 matmuls
        from . import exl3_mm

        if l2pf.ACTIVE is not None and done is None:
            # TF_GLM_L2PF: the shared expert before the routed experts (disjoint buffers, the same kernels: the same
            # bits), while the L2 still holds what site "a" prefetched for it
            shared_front(layer, w, b, 0, R)
            done = (0, R)
        atomic_active = b.prefill and R >= 1024 and m.experts.shared_suh is None and trial_control.ACTIVE.get('prefill_atomic',False)
        gr8_active = R == 1 and not b.prefill and trial_control.ACTIVE.get('gr8',False)
        with prof.timed("moe: routed (gr8)" if gr8_active else "moe: routed (exl3)"):
            if gr8_active:
                from . import gr8_adapter
                if not hasattr(b,'gr8_scratch'): b.gr8_scratch=gr8_adapter.Scratch(b.normed.device)
                gr8_routed=gr8_adapter.run(b.normed[:R],b.pick,b.wts,m.experts,b.gr8_scratch)
            elif atomic_active:
                from . import prefill_atomic
                prefill_atomic.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey, b.wts[:R], R, c.limit)
            else:
                exl3_mm.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey.view(-1, c.hidden), R, c.limit)

        def fill(lo: int, hi: int) -> None:
            for a, z in undone(hi, done):     # rows lo .. hi the front has not run on
                if max(a, lo) < z:
                    shared_front(layer, w, b, max(a, lo), z)
            with prof.timed("moe: combine"):
                # the shared row read from sy where it is (the copy's bits, one pass less); decode windows too unless
                # TF_GLM_COMBINE_SY=0
                if gr8_active:
                    torch.add(gr8_routed[lo:hi],b.sy[lo:hi],out=b.part[lo:hi])
                elif atomic_active:
                    from . import prefill_atomic
                    prefill_atomic.add_shared(prefill_atomic.output(b.ey, R, c.hidden)[lo:hi], b.sy[lo:hi], b.wts[lo:hi], b.part[lo:hi])
                elif b.prefill or __import__("os").environ.get("TF_GLM_COMBINE_SY", "1") != "0":
                    glue.combine_shared(b.ey[lo:hi], b.sy[lo:hi], b.wts[lo:hi], b.part[lo:hi])
                else:
                    b.ey[lo:hi, c.top_k].copy_(b.sy[lo:hi])
                    glue.combine(b.ey[lo:hi], b.wts[lo:hi], b.part[lo:hi])

        return partials(w, b, R, fill, "moe: all-gather", site=(layer.index, "f"))
    with prof.timed("moe: gate/up"):
        grouped.gate_up(b.normed[:R], m.experts, b.plan, b.eact, R)
    with prof.timed("moe: down"):
        grouped.down(b.eact, m.experts, b.plan, b.ey.view(-1, c.hidden), R)

    def fill(lo: int, hi: int) -> None:
        with prof.timed("moe: combine"):
            glue.combine(b.ey[lo:hi], b.wts[lo:hi], b.part[lo:hi])

    return partials(w, b, R, fill, "moe: all-gather", site=(layer.index, "f"))


def _mixer(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None, host_pos: int | None,
           sparse_np: int | None, done: tuple[int, int] | None = None) -> torch.Tensor | None:
    """The layer's KDA or DSA block on b.normed: every rank's partials, or None row-split (``partials``)."""

    if layer.kind == "kda":
        with prof.timed("kda"):
            return kda_block(layer, w, st, b, R, done)
    di = st.dsa_index[layer.index]
    with prof.timed("dsa (total)"):
        return dsa_block(layer, w, st.kc[di], st.vc[di], st.pos_dev, b, R, nch,
                         st.index[di] if st.index is not None else None, host_pos, sparse_np, done)


def _ffn(layer: LayerW, w: Weights, b: Buffers, R: int, done: tuple[int, int] | None = None) -> torch.Tensor | None:
    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
        return mlp_block(layer, w, b, R, done) if layer.mlp is not None else moe_block(layer, w, b, R, done)


def mixer_front(layer: LayerW, w: Weights, b: Buffers):
    """A prompt chunk's KDA / DSA front as ``front(lo, hi)`` (``hcsplit.HcSplit.glue``)."""

    if layer.kind == "kda":
        return lambda lo, hi: kda_front(layer, b, lo, hi)
    return lambda lo, hi: dsa_front(layer, w, b, lo, hi)


def ffn_front(layer: LayerW, w: Weights, b: Buffers):
    """The dense MLP's gate/up or the MoE's shared expert as ``front(lo, hi)``; None for a MoE without a separate
    shared expert (an MLX checkpoint's routes it with the others)."""

    if layer.mlp is not None:
        return lambda lo, hi: mlp_front(layer, w, b, lo, hi)
    if layer.moe.shared is not None:
        return lambda lo, hi: shared_front(layer, w, b, lo, hi)
    return None


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
                  host_pos: int | None = None, sparse_np: int | None = None, mixer=None) -> None:
    """``mixer``: the layer's KDA / DSA block as ``mixer(layer)`` instead of on ``st`` (a multi-stream window's
    segmented blocks, ``verify.BatchedVerify``); everything else is the same calls."""

    c = w.cfg
    x = b.x[:R]
    h = layer.attn_hc
    glue.hc_pre(x, h.fn, h.base, h.scale, layer.in_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters, prompt=b.prefill)
    g = mixer(layer) if mixer is not None else _mixer(layer, w, st, b, R, nch, host_pos, sparse_np)
    with prof.timed("hc"):
        glue.hc_post(x, x, g, b.post[:R], b.comb[:R])
        h = layer.ffn_hc
        glue.hc_pre(x, h.fn, h.base, h.scale, layer.post_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                    b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters, prompt=b.prefill)
    g = _ffn(layer, w, b, R)
    glue.hc_post(x, x, g, b.post[:R], b.comb[:R])


def split_layer_steps(w: Weights, st: State, b: Buffers, R: int, nch: int | None, host_pos: int | None,
                      mixer=None):
    """Every layer of a prompt chunk with its hyper-connection glue split by rows between the ranks
    (``hcsplit``): the first hc_pre runs on every row as unsplit, each later one on this rank's rows, and b.normed,
    the taps and b.hidden come back whole from the peer. The same bits as the unsplit loop and its final mean.
    ``mixer``: the layers' KDA / DSA blocks as ``mixer(layer, done)`` instead of on ``st`` (a chunk of several
    streams' prompts, ``multi_prefill``)."""

    c = w.cfg
    sp = b.split
    layers = w.layers
    sp.begin(R)
    fronts = sp.settings.fronts
    try:
        Rp = sp.Rp                               # R, or R + 1 with a zero pad row for an odd chunk
        first, h = layers[0], layers[0].attn_hc
        glue.hc_pre(b.x[:Rp], h.fn, h.base, h.scale, first.in_norm, b.normed[:Rp], b.xs[:Rp], b.post[:Rp],
                    b.comb[:Rp], b.hcpart[:Rp], c.eps, c.hc_eps, c.hc_iters, prompt=True)
        done = None                              # rows whose front the glue already ran (TF_GLM_PREFILL_OVERLAP=2)
        for i, layer in enumerate(layers):
            if mixer is not None:
                mixer(layer, done)
            else:
                _mixer(layer, w, st, b, R, nch, host_pos, None, done)
            front = ffn_front(layer, w, b) if fronts else None
            with prof.timed("hc"):
                sp.glue(layer.ffn_hc, layer.post_norm, front=front)
            _ffn(layer, w, b, R, sp.own() if front is not None else None)
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            front = mixer_front(nxt, w, b) if fronts and nxt is not None else None
            with prof.timed("hc (layer end)"):
                sp.glue(nxt.attn_hc if nxt is not None else None, nxt.in_norm if nxt is not None else None,
                        taps=tuple(b.tap_at.get(layer.index, ())), final=nxt is None, front=front)
            done = sp.own() if front is not None else None
            yield layer.index
    finally:
        sp.finish()


def split_layers(w: Weights, st: State, b: Buffers, R: int, nch: int | None, host_pos: int | None,
                 mixer=None) -> None:
    for _ in split_layer_steps(w, st, b, R, nch, host_pos, mixer):
        pass


def compute_steps(*args, **kwargs):
    from .kda_mxfp8 import prefill_scope
    with prefill_scope(), prof.event_scope(args[0].rank, args[3]):
        return (yield from _source_compute_steps(*args, **kwargs))


def _source_compute_steps(w: Weights, st: State, b: Buffers, R: int, *, nch: int | None = None,
                  host_pos: int | None = None, inject=None, head: bool = True, mixer=None):
    """Prompt-only forward yielding after each complete layer; returns the head on exhaustion.

    All live rows, taps and overlapped fronts stay in the engine's dedicated pbuf. A split boundary retains
    only this rank's hidden rows plus the already prepared next front, exactly as the uninterrupted split loop.
    The scheduler drains CUDA before a decode round; verify never borrows pbuf. KDA's global temporary scratch
    is consumed inside a layer, so nothing in it needs to survive this yield.
    """
    if not b.prefill:
        raise ValueError("layer-sliced forwards require prompt buffers")
    c = w.cfg
    glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
    if inject is not None:
        rows, features = inject
        b.x.index_copy_(0, rows, features.repeat(1, c.streams))
    split = b.split is not None and b.split.applies(R)
    if split:
        steps = split_layer_steps(w, st, b, R, nch, host_pos, mixer)
        try:
            for i in steps:
                b.split.finish()                    # main stream joins the overlap exchanges/fronts
                yield i
                # A decode round may have used the main stream since the last yield. Geometry and prompt
                # buffers remain intact; the overlap stream must join the new main-stream work before reuse.
                if b.split.stream is not None:
                    b.split.stream.wait_stream(torch.cuda.current_stream())
                b.split.active = True
        finally:
            steps.close()
    else:
        for layer in w.layers:
            layer_forward(layer, w, st, b, R, nch, host_pos, None, mixer)
            for slot in b.tap_at.get(layer.index, ()):
                glue.stream_mean(b.x[:R], b.taps[slot][:R])
            yield layer.index
        glue.stream_mean(b.x[:R], b.hidden[:R])
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    if head:
        return mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    return None


def check_room(w: Weights, st: State, R: int, pos: int | None = None) -> None:
    pos = st.pos if pos is None else pos
    if pos + R > w.cfg.dense_limit and st.index is None:
        raise ValueError(f"context {pos + R} past {w.cfg.dense_limit} tokens: this engine was started without long "
                         "contexts (DSA's sparse top-k)")
    if pos + R > st.capacity:
        raise ValueError("context past the cache capacity")


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int]) -> int:
    """Host work before a forward: the token ids into the static device buffer (pinned copy)."""

    R = len(tokens)
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    check_room(w, st, R)
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = list(tokens)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()
    return R


def compute(w, st, b, R, **kwargs):
    if b.prefill:
        from .kda_mxfp8 import prefill_scope
        with prefill_scope(), prof.event_scope(w.rank, R):
            return _source_compute(w,st,b,R,**kwargs)
    return _source_compute(w,st,b,R,**kwargs)


def _source_compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, nch: int | None = None,
            host_pos: int | None = None, sparse_np: int | None = None, inject=None, head: bool = True, mixer=None):
    """Run capturable GPU work on static buffers and device positions; eager long contexts use host_pos (graphs sparse_np) to select sparse attention; ``inject``: (rows, features) replacing those rows' embeddings (an image prompt's tower output, on every hyper-connection stream)."""

    c = w.cfg
    pf = w.meta.get("l2pf")
    if pf is not None and not b.prefill and R <= pf.s.rows and l2pf.ACTIVE is None:
        l2pf.ACTIVE = pf                 # TF_GLM_L2PF: this decode window's sites prefetch (joined below)
        try:
            return _compute(w, st, b, R, logits=logits, nch=nch, host_pos=host_pos, sparse_np=sparse_np,
                            inject=inject, head=head, mixer=mixer)
        finally:
            pf.join()
            l2pf.ACTIVE = None
    return _compute(w, st, b, R, logits=logits, nch=nch, host_pos=host_pos, sparse_np=sparse_np, inject=inject,
                    head=head, mixer=mixer)


def _compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, nch: int | None = None,
             host_pos: int | None = None, sparse_np: int | None = None, inject=None, head: bool = True, mixer=None):
    c = w.cfg
    glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
    if inject is not None:
        rows, features = inject
        b.x.index_copy_(0, rows, features.repeat(1, c.streams))
    if b.prefill and b.split is not None and sparse_np is None and b.split.applies(R):
        split_layers(w, st, b, R, nch, host_pos, mixer)      # also leaves the taps and b.hidden[:R]
    else:
        for layer in w.layers:
            layer_forward(layer, w, st, b, R, nch, host_pos, sparse_np, mixer)
            for slot in b.tap_at.get(layer.index, ()):
                glue.stream_mean(b.x[:R], b.taps[slot][:R])
        glue.stream_mean(b.x[:R], b.hidden[:R])
    if not logits:
        return None
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    if not head:                         # the normed rows (MTP, last_hidden) without the head: None
        return None
    if b.prefill:                        # the head reads the last row only (fnormed keeps every row for the MTP)
        return mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    return mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])


def chunks_for(st: State, R: int) -> int:
    from .attention import CHUNK

    return -(-(st.pos + R) // CHUNK)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True) -> torch.Tensor | None:
    """Return logits and hidden buffer views for token rows, leaving committed state unchanged until commit."""

    R = stage(w, st, b, tokens)
    return compute(w, st, b, R, logits=logits, nch=chunks_for(st, R), host_pos=st.pos)


@triton.jit
def _row(CONV, PROJ, l, src, c, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr):
    old = tl.load(CONV + l * conv_layer + src * C + c, mask=(src < TAPS) & (c < C), other=0.0)
    new = tl.load(PROJ + l * proj_layer + (src - TAPS) * proj_row + c, mask=(src >= TAPS) & (c < C), other=0.0)
    return tl.where(src < TAPS, old, new)


@triton.jit
def _conv_shift(CONV, PROJ, keep, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program (layer, channel block): the 3 window rows become rows keep .. keep + 2 of [old window; new rows]."""

    l = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v0 = _row(CONV, PROJ, l, keep, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v1 = _row(CONV, PROJ, l, keep + 1, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v2 = _row(CONV, PROJ, l, keep + 2, c, conv_layer, proj_layer, proj_row, C, TAPS)
    tl.store(CONV + l * conv_layer + c, v0, mask=c < C)
    tl.store(CONV + l * conv_layer + C + c, v1, mask=c < C)
    tl.store(CONV + l * conv_layer + 2 * C + c, v2, mask=c < C)


def _shift_conv(conv: torch.Tensor, proj: torch.Tensor, keep: int) -> None:
    """conv [L, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows] (proj [L, R, W >= C])."""

    n, taps, C = conv.shape
    if taps != 3:
        raise ValueError("the conv shift kernel is written for 4-tap convolutions")
    _conv_shift[(n, triton.cdiv(C, 1024))](conv, proj, keep, conv.stride(0), proj.stride(0), proj.stride(1), C=C,
                                           TAPS=taps, BLOCK=1024, num_warps=4)


@triton.jit
def _conv_shift_seg(CONV, PROJ, SEG, conv_slot, conv_layer, proj_layer, proj_row, C: tl.constexpr,
                    TAPS: tl.constexpr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    """Program (layer, channel block, segment): ``_conv_shift`` of the segment's conv slot over its own rows."""

    l = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    e = SEG + tl.program_id(2) * COLS
    row0 = tl.load(e + 0).to(tl.int64)
    rows = tl.load(e + 1)
    cslot = tl.load(e + 4).to(tl.int64)
    keep = tl.load(e + 5)
    if rows > 0:
        conv = CONV + cslot * conv_slot
        proj = PROJ + row0 * proj_row
        v0 = _row(conv, proj, l, keep, c, conv_layer, proj_layer, proj_row, C, TAPS)
        v1 = _row(conv, proj, l, keep + 1, c, conv_layer, proj_layer, proj_row, C, TAPS)
        v2 = _row(conv, proj, l, keep + 2, c, conv_layer, proj_layer, proj_row, C, TAPS)
        tl.store(conv + l * conv_layer + c, v0, mask=c < C)
        tl.store(conv + l * conv_layer + C + c, v1, mask=c < C)
        tl.store(conv + l * conv_layer + 2 * C + c, v2, mask=c < C)


def conv_shift_segments(conv: torch.Tensor, proj: torch.Tensor, seg: torch.Tensor) -> None:
    """``_shift_conv`` per segment of a multi-stream window: conv [conv slots, L, 3, C] (in place; each [3, C]
    contiguous), proj [L, R, W >= C] the window's projection rows; segment i (``kda.segment_table``) has its conv
    slot take rows keep .. keep + 2 of [conv[slot]; its own proj rows]. Segments of 0 rows are skipped."""

    _, n, taps, C = conv.shape
    if taps != 3:
        raise ValueError("the conv shift kernel is written for 4-tap convolutions")
    if conv.stride(3) != 1 or conv.stride(2) != C:
        raise ValueError("each conv slot's [3, C] window must be contiguous")
    if seg.dtype != torch.int32 or seg.dim() != 2 or seg.shape[1] != kda_mod.SEG_COLS or not seg.is_contiguous():
        raise ValueError("segments: a contiguous int32 [nseg, SEG_COLS] table")
    _conv_shift_seg[(n, triton.cdiv(C, 1024), seg.shape[0])](
        conv, proj, seg, conv.stride(0), conv.stride(1), proj.stride(0), proj.stride(1), C=C, TAPS=taps,
        COLS=kda_mod.SEG_COLS, BLOCK=1024, num_warps=4)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the last forward's first ``keep`` rows; a prompt chunk keeps all, its KDA layers already committed."""

    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    n = 0 if b.prefill else len(st.cur)
    if n:
        cur = st.cur[0]
        if keep < R:
            kda_mod.replay_layers(st.rec[cur], st.scratch_set, keep, st.rec[1 - cur])
        st.cur = [1 - cur] * n
        _shift_conv(st.conv, st.proj, keep)
    st.set_pos(st.pos + keep)
