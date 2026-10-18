"""DFlash2 drafts use committed target taps and identical rank-ordered candidate merges; verification accepts only the serial sample."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

from . import glue, qmm, spec_trials, trial_control
from .weights import Weights

EDGE = 0.6        # the selector's edge score weight in a chain pick
NOISE = 0.7       # the target's keyed noise weight when sampling
NO_LIMIT = 1e30
# the noise-aware stop rules' temperature B (fnc<N>:<p>, fcost<N>:noisy): draft_sim's NoisyPredictor beta, fitted by
# log loss (``fit_beta``, its grid) on the 9 teacher-forced dumps of 2026-09-30 (7 sampled prose, 2 greedy code):
# 2.0 on all of them, on the sampled ones alone and on the greedy ones alone (a finer grid: 1.8, log loss -0.2%)
NOISY_BETA = 2.0


def noisy_confidence(score: np.ndarray, noise: np.ndarray | None, j: int, beta: float) -> float:
    """softmax((score + g) / beta) at the pick j, g the target's keyed Gumbel draws for the candidates (None: greedy,
    softmax(score / beta)): the chance the target's keyed sample is the pick when its scaled logits are the drafter's
    plus iid Gumbel(beta) errors. draft_sim's ``noisy_prob`` feature, the same float64 operations in the same order."""

    z = score + noise if noise is not None else score
    z = z / beta
    e = np.exp(z - z.max())
    return float(e[j] / e.sum())


def best_depth(q, round_ms) -> int:
    """The k (0..len(q)) maximizing E[tokens | k drafts] / round_ms[k], q the drafts' conditional acceptances
    (``draft_sim.best_depth`` with round_ms[k] = its Costs.round_ms(k, 1, True))."""

    alive = np.concatenate([[1.0], np.cumprod(np.asarray(q, dtype=np.float64))])
    expect = np.cumsum(alive)
    ms = np.asarray(round_ms[:len(q) + 1], dtype=np.float64)
    return int(np.argmax(expect / ms))


@triton.jit
def _dconv_kernel(X, DYN, BASE, RES, OUT, D: tl.constexpr, G: tl.constexpr, GS: tl.constexpr,
                  BRANCH: tl.constexpr, HAS_RES: tl.constexpr, BLOCK: tl.constexpr, SEG: tl.constexpr = 0):
    """Two-tap grouped dynamic convolution over the block (row r mixes rows r and r - 1), plus the residual; SEG > 0:
    the rows are blocks of SEG rows side by side (``dflash2_multi``), each block's first row mixing no row before."""

    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < D
    x = tl.load(X + row * D + c, mask=ok, other=0.0).to(tl.float32)
    if SEG > 0:
        has_prev = row % SEG != 0
    else:
        has_prev = row > 0
    prev = tl.load(X + (row - 1) * D + c, mask=ok & has_prev, other=0.0).to(tl.float32)
    grp = c // GS
    d0 = tl.load(DYN + ((row * 2 + BRANCH) * 2) * G + grp, mask=ok, other=0.0).to(tl.float32)
    d1 = tl.load(DYN + ((row * 2 + BRANCH) * 2 + 1) * G + grp, mask=ok, other=0.0).to(tl.float32)
    b0 = tl.load(BASE + (BRANCH * 2) * D + c, mask=ok, other=0.0).to(tl.float32)
    b1 = tl.load(BASE + (BRANCH * 2 + 1) * D + c, mask=ok, other=0.0).to(tl.float32)
    k0 = (b0 + d0).to(tl.bfloat16).to(tl.float32)
    k1 = (b1 + d1).to(tl.bfloat16).to(tl.float32)
    y = (x * k0 + prev * k1).to(tl.bfloat16)
    if HAS_RES:
        r = tl.load(RES + row * D + c, mask=ok, other=0.0).to(tl.float32)
        y = (r + y.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * D + c, y, mask=ok)


@triton.jit
def _prep_kernel(QKV, QN, KN, COS, SIN, QO, KO, VO, L, stride, eps,
                 H: tl.constexpr, HKV: tl.constexpr, HALF: tl.constexpr):
    """Normalize and rotate q and k in [q heads | k heads | v heads], copy v, and return [heads, rows, head_dim]."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HALF)
    D: tl.constexpr = 2 * HALF
    src = QKV + row * stride + head * D
    if head < H + HKV:
        a = tl.load(src + d).to(tl.float32)
        b = tl.load(src + HALF + d).to(tl.float32)
        rstd = tl.rsqrt((tl.sum(a * a, axis=0) + tl.sum(b * b, axis=0)) / D + eps)
        if head < H:
            wa = tl.load(QN + d).to(tl.float32)
            wb = tl.load(QN + HALF + d).to(tl.float32)
        else:
            wa = tl.load(KN + d).to(tl.float32)
            wb = tl.load(KN + HALF + d).to(tl.float32)
        a = (a * rstd * wa).to(tl.bfloat16).to(tl.float32)
        b = (b * rstd * wb).to(tl.bfloat16).to(tl.float32)
        cos = tl.load(COS + row * HALF + d)
        sin = tl.load(SIN + row * HALF + d)
        ra = (a * cos - b * sin).to(tl.bfloat16)
        rb = (b * cos + a * sin).to(tl.bfloat16)
        if head < H:
            dst = QO + (head * L + row) * D
        else:
            dst = KO + ((head - H) * L + row) * D
        tl.store(dst + d, ra)
        tl.store(dst + HALF + d, rb)
    else:
        dst = VO + ((head - H - HKV) * L + row) * D
        tl.store(dst + d, tl.load(src + d))
        tl.store(dst + HALF + d, tl.load(src + HALF + d))


@triton.jit
def _dattn_kernel(Q, K, V, OUT, POS, window, scale, N: tl.constexpr, G: tl.constexpr, NH: tl.constexpr,
                  HD: tl.constexpr, CAP: tl.constexpr, BK: tl.constexpr, CAUSAL: tl.constexpr,
                  RING: tl.constexpr = False, SKIP: tl.constexpr = True):
    """Each KV head attends its queries to the window's tiles (masked tiles skipped, same bits) and the block's keys."""

    kvh = tl.program_id(0)
    M: tl.constexpr = G * N
    m = tl.arange(0, M)
    qh = kvh * G + m // N
    qr = m % N
    d = tl.arange(0, HD)
    q = tl.load(Q + (qh[:, None] * N + qr[:, None]) * HD + d[None, :])
    s = tl.load(POS).to(tl.int32)
    klen = s + N
    qpos = s + qr
    m_i = tl.full([M], -1e30, tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, HD], tl.float32)
    lo = 0
    if SKIP:
        lo = tl.maximum(s - window, 0) // BK * BK
    for start in range(lo, klen, BK):
        kk = start + tl.arange(0, BK)
        kin = kk < klen
        if RING:
            row = kk % CAP
        else:
            row = kk
        k = tl.load(K + (kvh * CAP + row[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        v = tl.load(V + (kvh * CAP + row[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        ok = kin[None, :] & (((kk[None, :] < s) & (qpos[:, None] - kk[None, :] <= window)) | (kk[None, :] >= s))
        if CAUSAL:
            ok = ok & (kk[None, :] <= qpos[:, None])
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(OUT + qr[:, None] * (NH * HD) + qh[:, None] * HD + d[None, :], out.to(tl.bfloat16))


def _dconv(x: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, branch: int, group_size: int,
           residual: torch.Tensor | None = None, seg: int = 0) -> torch.Tensor:
    rows, d = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    block = 1024
    _dconv_kernel[(rows, triton.cdiv(d, block))](x, dyn.contiguous(), base, residual if residual is not None else x,
                                                 out, D=d, G=d // group_size, GS=group_size, BRANCH=branch,
                                                 HAS_RES=residual is not None, BLOCK=block, SEG=seg, num_warps=4)
    return out


def _quantize4(w: torch.Tensor) -> qmm.Q4:
    """bf16 (N, K) -> MLX-style affine 4-bit in groups of 64 along K, tiled: min / max ranges (``qmm.quantize4``) or,
    with TF_GLM_DRAFT_QUANT=mse, least-squares clipped ones (``qmm.quantize4_mse``). Drafts only: replies keep."""

    return qmm.draft_quantize4(w)


def _mm(x: torch.Tensor, w: qmm.Q4, xs: torch.Tensor | None = None, *, f32: bool = False) -> torch.Tensor:
    return qmm.matmul(x, w, xs, f32=f32)


class BadCandidates(RuntimeError):
    """A block pass's host rows hold a candidate id outside the vocabulary (a float's bits read as an id)."""


def merge_candidates(g: torch.Tensor, proj: torch.Tensor, k: int,
                     vocab: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A block pass's host rows -> (ids [depth, k], logits, projected rows): ``g`` [ranks, depth, 2 * k] (each rank's
    top-k values, then ids as fp32 bits), merged by value then id over the ranks; ``proj`` [depth, selector rank]."""

    values = torch.cat([g[r, :, :k] for r in range(g.shape[0])], dim=1).numpy().astype(np.float64)
    tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(g.shape[0])],
                       dim=1).numpy().astype(np.int64)
    if vocab is not None and tokens.size and (int(tokens.min()) < 0 or int(tokens.max()) >= vocab):
        bad = tokens[(tokens < 0) | (tokens >= vocab)]
        raise BadCandidates(f"{bad.size} of {tokens.size} candidate ids are outside the vocabulary of {vocab} "
                            f"(first {int(bad[0])} = 0x{int(bad[0]) & 0xFFFFFFFF:08x}, as a float "
                            f"{float(np.array([int(bad[0]) & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]):.4g})")
    if g.shape[0] > 1:
        order = np.lexsort((tokens, -values), axis=-1)[:, :k]
        values = np.take_along_axis(values, order, axis=1)
        tokens = np.take_along_axis(tokens, order, axis=1)
    return tokens, values, proj.numpy().astype(np.float64)          # astype copies: nothing keeps the host buffer


@dataclass
class DraftLayer:
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    a_base: torch.Tensor      # [2, 2, D] bf16
    a_kp: qmm.Q4              # [2 * 2 * D/gs, D]
    m_base: torch.Tensor
    m_kp: qmm.Q4
    qkv: qmm.Q4               # this rank's [q heads | k heads | v heads]
    kv: qmm.Q4                # this rank's [k heads | v heads] (context rows)
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o: qmm.Q4                 # [D, this rank's heads * head_dim]: a row-parallel partial
    gu: qmm.Q4                # this rank's [gate | up]
    down: qmm.Q4              # [D, this rank's MLP width]: a row-parallel partial


class Drafter:
    """Draft one sequence from position-indexed context using device lengths and static buffers shared by eager execution and CUDA graphs."""

    def __init__(self, draft_dir: str | Path, w: Weights, *, block: int | None = None, capacity: int = 2560,
                 tap_rows: int = 0, ring: bool = False) -> None:
        """``tap_rows``: context updates up to this many rows (a verify window's kept rows) replay captured graphs too
        (the block's own size always does). ``ring`` (TF_GLM_DRAFT_RING): the context in a ring of the window and
        block, the same drafts."""

        path = Path(draft_dir)
        cfg = json.loads((path / "config.json").read_text())
        dc = cfg["dflash_config"]
        self.w = w
        self.dev = w.device
        self.rank, self.world = w.rank, w.world
        self.D = int(cfg["hidden_size"])
        self.hd = int(cfg["head_dim"])
        heads, kv_heads = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        from .tp import draft_parts

        # KV heads in whole query groups over the ranks (32 / 8 over 3: 12/12/8 query heads, 3/3/2 KV heads), the MLP
        # in units of 128 (``tp.draft_parts``); two ranks: halves
        qs, kvs, mlps = draft_parts(heads, kv_heads, int(cfg["intermediate_size"]), self.world)
        self.heads, self.kvh = qs[self.rank], kvs[self.rank]
        q0, kv0 = sum(qs[:self.rank]), sum(kvs[:self.rank])
        self.eps = float(cfg["rms_norm_eps"])
        theta = float(cfg["rope_parameters"]["rope_theta"])
        self.mask_id = int(dc["mask_token_id"])
        self.gs = int(dc["conv_group_size"])
        if int(dc["conv_kernel_size"]) != 2:
            raise ValueError("only two-tap convolutions are ported")
        self.block = int(block or dc["block_size"])
        self.top_k = int(dc["selector_top_k"])
        self.tap_layers = tuple(int(i) for i in dc["target_layer_ids"])
        self.window = int(cfg["sliding_window"]) - 1
        self.causal = bool(cfg.get("is_causal", True))
        self.inter = mlps[self.rank]
        m0 = sum(mlps[:self.rank])
        H, KV, hd = self.heads, self.kvh, self.hd
        dev = self.dev

        def gpu(t: torch.Tensor) -> torch.Tensor:
            return t.to(dev, torch.bfloat16).contiguous()

        quantize4 = _quantize4

        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
            def get(name: str) -> torch.Tensor:
                return f.get_tensor(name)

            self.fc = quantize4(gpu(get("fc.weight")))
            self.hidden_norm = gpu(get("hidden_norm.weight"))
            self.norm = gpu(get("norm.weight"))
            self.hproj = gpu(get("candidate_selector.hidden_projection.weight"))
            self.pred = get("candidate_selector.predecessor_codebook").float().numpy().copy()
            self.succ = get("candidate_selector.successor_codebook").float().numpy().copy()
            self.layers: list[DraftLayer] = []
            for i in range(int(cfg["num_hidden_layers"])):
                p = f"layers.{i}."
                q = get(p + "self_attn.q_proj.weight")[q0 * hd:(q0 + H) * hd]
                k = get(p + "self_attn.k_proj.weight")[kv0 * hd:(kv0 + KV) * hd]
                v = get(p + "self_attn.v_proj.weight")[kv0 * hd:(kv0 + KV) * hd]
                o = get(p + "self_attn.o_proj.weight")[:, q0 * hd:(q0 + H) * hd]
                g = get(p + "mlp.gate_proj.weight")[m0:m0 + self.inter]
                u = get(p + "mlp.up_proj.weight")[m0:m0 + self.inter]
                dn = get(p + "mlp.down_proj.weight")[:, m0:m0 + self.inter]
                for conv in ("attention_conv", "mlp_conv"):
                    if tuple(get(p + conv + ".base_kernel").shape) != (2, 2, self.D):
                        raise ValueError(f"unexpected {conv} base kernel shape")
                self.layers.append(DraftLayer(
                    in_norm=gpu(get(p + "input_layernorm.weight")),
                    post_norm=gpu(get(p + "post_attention_layernorm.weight")),
                    a_base=gpu(get(p + "attention_conv.base_kernel")),
                    a_kp=quantize4(gpu(get(p + "attention_conv.kernel_projection.weight"))),
                    m_base=gpu(get(p + "mlp_conv.base_kernel")),
                    m_kp=quantize4(gpu(get(p + "mlp_conv.kernel_projection.weight"))),
                    qkv=quantize4(gpu(torch.cat((q, k, v)))),
                    kv=quantize4(gpu(torch.cat((k, v)))),
                    q_norm=gpu(get(p + "self_attn.q_norm.weight")),
                    k_norm=gpu(get(p + "self_attn.k_norm.weight")),
                    o=quantize4(gpu(o)),
                    gu=quantize4(gpu(torch.cat((g, u)))),
                    down=quantize4(gpu(dn))))
        torch.cuda.empty_cache()
        self.inv_freq = 1.0 / theta ** (torch.arange(hd // 2, device=dev, dtype=torch.float32) * 2 / hd)
        # Block rows sit past committed context and the next context update overwrites them.
        from tensorfold.cuda.geometry import draft_ring_rows

        self.capacity = capacity
        rows = draft_ring_rows(self.window, self.block) if ring and self.window >= 0 else 0
        self.ring = rows if 0 < rows < capacity + self.block else 0     # 0: a flat buffer, logical row p at p
        self.cap = self.ring or capacity + self.block
        self.kc = [torch.zeros((KV, self.cap, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.vc = [torch.zeros((KV, self.cap, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.pos_dev = torch.zeros((1,), dtype=torch.int64, device=dev)
        self.context_end = 0
        n = self.block
        self.ids = torch.full((n,), self.mask_id, dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((1,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.ar = torch.arange(max(64, n), device=dev)
        self.tap_in = torch.zeros((64, len(self.tap_layers) * self.D), dtype=torch.bfloat16, device=dev)
        # a pass's candidates (every rank's [block - 1, 2 * top_k], values then ids as fp32 bits) and projected rows
        # [block - 1, selector rank] fp32 in one device buffer, read back by one pinned copy (``candidates``)
        self.gathered = self.world if self.world > 1 and w.comm is not None else 1
        self.cand_n = self.gathered * (n - 1) * 2 * self.top_k
        cuda = torch.cuda.is_available()
        self.cand = torch.zeros((self.cand_n + (n - 1) * self.hproj.shape[0],), dtype=torch.float32, device=dev)
        self.cand_host = torch.zeros(self.cand.shape, dtype=torch.float32, pin_memory=cuda)
        self.cand_ready = torch.cuda.Event() if cuda else None
        self.packed = self.cand[:self.cand_n].view(self.gathered, n - 1, 2 * self.top_k)
        self.proj = self.cand[self.cand_n:].view(n - 1, self.hproj.shape[0])
        self.pool = None
        self.block_graph = None
        self.tap_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.tap_rows = min(max(self.block, int(tap_rows)), self.tap_in.shape[0])

    def nbytes(self) -> int:
        total = sum(q.nbytes() for L in self.layers for q in (L.a_kp, L.m_kp, L.qkv, L.kv, L.o, L.gu, L.down))
        return total + self.fc.nbytes() + 2 * sum(t.numel() * 2 for t in self.kc)

    def reset(self) -> None:
        self.context_end = 0
        self.pos_dev.zero_()

    def _slots(self, n: int) -> torch.Tensor:
        """Where the rows at positions context_end .. context_end + n - 1 go: those positions, modulo the ring."""

        idx = self.pos_dev + self.ar[:n]
        return idx % self.ring if self.ring else idx

    # -- pieces -------------------------------------------------------------------------------------------------
    def _rotary(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = (self.pos_dev + self.ar[:rows]).to(torch.float32)
        phase = pos[:, None] * self.inv_freq[None, :]
        return phase.cos().contiguous(), phase.sin().contiguous()

    def _prep(self, qkv: torch.Tensor, L: DraftLayer, cos: torch.Tensor, sin: torch.Tensor, heads: int):
        rows = qkv.shape[0]
        hd = self.hd
        q = torch.empty((heads, rows, hd), dtype=torch.bfloat16, device=self.dev) if heads else qkv
        k = torch.empty((self.kvh, rows, hd), dtype=torch.bfloat16, device=self.dev)
        v = torch.empty_like(k)
        _prep_kernel[(rows, heads + 2 * self.kvh)](qkv, L.q_norm, L.k_norm, cos, sin, q, k, v, rows, qkv.stride(0),
                                                   self.eps, H=heads, HKV=self.kvh, HALF=hd // 2, num_warps=1)
        return q, k, v

    def _norm(self, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        out = torch.empty((rows, self.D), dtype=torch.bfloat16, device=self.dev)
        xs = torch.empty((rows, self.D // 64), dtype=torch.float32, device=self.dev)
        glue.rmsnorm(x, weight, self.eps, out, xs)
        return out, xs

    def _row(self, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor | None = None) -> torch.Tensor:
        """A projection whose input is split over the ranks: fp32 partials added in rank order, then bf16."""

        part = _mm(x.contiguous(), q, xs, f32=True)
        if self.world == 1 or self.w.comm is None:
            return part.to(torch.bfloat16)
        got = torch.empty((self.world * part.numel(),), dtype=torch.float32, device=self.dev)
        self.w.comm.all_gather(part.reshape(-1), got)
        g = got.view(self.world, *part.shape)
        acc = g[0].clone()
        for i in range(1, self.world):
            acc += g[i]
        return acc.to(torch.bfloat16)

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        L = self.layers[i]
        rows = x.shape[0]
        normed, xs = self._norm(x, L.in_norm)
        dyn = _mm(normed, L.a_kp, xs)
        q, k, v = self._prep(_mm(_dconv(normed, dyn, L.a_base, 0, self.gs), L.qkv), L, cos, sin, self.heads)
        self.kc[i].index_copy_(1, idx, k)
        self.vc[i].index_copy_(1, idx, v)
        out = torch.empty((rows, self.heads * self.hd), dtype=torch.bfloat16, device=self.dev)
        _dattn_kernel[(self.kvh,)](q, self.kc[i], self.vc[i], out, self.pos_dev, self.window, self.hd ** -0.5,
                                   N=rows, G=self.heads // self.kvh, NH=self.heads, HD=self.hd, CAP=self.cap, BK=64,
                                   CAUSAL=self.causal, RING=bool(self.ring), num_warps=4)
        x = _dconv(self._row(out, L.o), dyn, L.a_base, 1, self.gs, x)
        normed, xs = self._norm(x, L.post_norm)
        dyn = _mm(normed, L.m_kp, xs)
        gu = _mm(_dconv(normed, dyn, L.m_base, 0, self.gs), L.gu)
        act = torch.empty((rows, self.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, self.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, NO_LIMIT)
        return _dconv(self._row(act, L.down, axs), dyn, L.m_base, 1, self.gs, x)

    # -- the context --------------------------------------------------------------------------------------------
    def _taps_compute(self, n: int) -> None:
        ctx, _ = self._norm(_mm(self.tap_in[:n], self.fc), self.hidden_norm)
        cos, sin = self._rotary(n)
        idx = self._slots(n)
        for i, L in enumerate(self.layers):
            _, k, v = self._prep(_mm(ctx, L.kv), L, cos, sin, 0)
            self.kc[i].index_copy_(1, idx, k)
            self.vc[i].index_copy_(1, idx, v)
        self.pos_dev += n

    @torch.no_grad()
    def add_taps(self, taps: torch.Tensor) -> None:
        """Committed rows' taps [n, 5 * D] (bf16) at positions context_end, context_end + 1, ..."""

        for start in range(0, taps.shape[0], self.tap_in.shape[0]):
            part = taps[start:start + self.tap_in.shape[0]]
            n = part.shape[0]
            if self.context_end + n > self.capacity:
                raise ValueError("drafter context past its capacity")
            self.tap_in[:n].copy_(part)
            g = self.tap_graphs.get(n)
            if g is not None:
                g.replay()
            else:
                self._taps_compute(n)
            self.context_end += n

    # -- drafts -------------------------------------------------------------------------------------------------
    def _block_compute(self) -> None:
        """Run [pending, mask x (block - 1)] at the committed length and merge candidates over both ranks' vocabulary halves."""

        n = self.block
        x = torch.empty((n, self.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids, self.w.embed, self.D, 1, x)
        cos, sin = self._rotary(n)
        idx = self._slots(n)
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, idx)
        h, hs = self._norm(x[1:], self.norm)
        logits = _mm(h, self.w.draft_head if self.w.draft_head is not None else self.w.head, hs)
        vals, local = torch.topk(logits.float(), self.top_k, dim=-1)
        gids = (local + self.w.vocab_offset).to(torch.int32)
        packed = torch.cat([vals, gids.view(torch.float32)], dim=1).contiguous()
        if self.gathered > 1:
            self.w.comm.all_gather(packed.view(-1), self.cand[:self.cand_n])
        else:
            self.cand[:self.cand_n].copy_(packed.view(-1))
        self.proj.copy_(F.linear(h, self.hproj).float())

    @torch.no_grad()
    def capture(self) -> None:
        """CUDA graphs for the block pass and for context updates of 1..tap_rows rows (both ranks together)."""

        self.pool = torch.cuda.graph_pool_handle()
        self.reset()
        self.tap_in.zero_()
        self.ids[0] = 0
        for n in range(1, self.tap_rows + 1):
            for _ in range(2):
                self._taps_compute(n)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._taps_compute(n)
            self.tap_graphs[n] = g
            self.reset()
        for _ in range(2):
            self._block_compute()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self._block_compute()
        self.block_graph = g
        torch.cuda.synchronize()
        self.reset()

    @torch.no_grad()
    def candidates(self, pending: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Candidate ids [depth, top_k] for the positions after ``pending``, their logits, the projected rows."""

        self.ids_host[0] = pending
        self.ids[:1].copy_(self.ids_host, non_blocking=True)
        if self.block_graph is not None:
            self.block_graph.replay()
        else:
            self._block_compute()
        # one pinned copy of candidates and projected rows (two .cpu() calls were two syncs and two pageable copies)
        vocab = self.succ.shape[0]
        for attempt in (0, 1):
            if self.cand_ready is not None:
                self.cand_host.copy_(self.cand, non_blocking=True)
                self.cand_ready.record()
                self.cand_ready.synchronize()
            else:
                self.cand_host.copy_(self.cand)
            g = self.cand_host[:self.cand_n].view(self.gathered, self.block - 1, 2 * self.top_k)[:, :depth]
            proj = self.cand_host[self.cand_n:].view(self.block - 1, -1)[:depth]
            try:
                return merge_candidates(g, proj, self.top_k, vocab)
            except BadCandidates as e:
                if attempt:
                    raise
                print(f"[tensorfold] drafter: {e}; copying the rows again once the device is idle", file=sys.stderr,
                      flush=True)
                torch.cuda.synchronize()

    def chain(self, tokens: np.ndarray, values: np.ndarray, proj: np.ndarray, anchor: int, first: int,
              sampling: Sampling | None, confidence: float = 0.0, *, beta: float = 0.0,
              round_ms: tuple[float, ...] | None = None, confs: list | None = None, controller=None) -> list[int]:
        """Choose candidates with selector edges and target keyed noise; stop below cumulative confidence, always
        keeping the first draft. ``beta`` > 0: the confidences are noise-aware (``noisy_confidence``); with
        ``round_ms`` (ms of a round of k drafts, k = 0..) the chain is walked whole and cut at ``best_depth`` of
        them instead (``confidence`` unused). Only where the chain stops changes: the picks are the same. ``confs``
        (a list): the chain is walked whole and each pick's confidence (noise-aware with ``beta``, else the
        softmax share of its score) appended to it, for a caller that cuts the chains itself (``multi``'s joint
        allocation)."""

        depth = tokens.shape[0]
        if controller is not None:
            assert beta > 0 and round_ms is None and confs is None
            controller.start(depth)
        sampled = sampling is not None and sampling.temperature > 0
        temp = float(sampling.temperature) if sampled else 1.0
        noise = None
        if sampled:
            noise = -np.log(-np.log(uniform_rows(sampling.seed, first + np.arange(depth), tokens)))
        noisy = beta > 0
        out: list[int] = []
        conf: list[float] = []
        prev, chain = anchor, 1.0
        noise_weight = trial_control.ACTIVE["draft_noise"]
        for d in range(depth):
            edge = self.succ[tokens[d]].astype(np.float64) @ (self.pred[prev].astype(np.float64) * proj[d])
            score = (values[d] + EDGE * edge) / temp
            pick = score + noise_weight * noise[d] if noise is not None else score
            j = int(np.argmax(pick))
            spec_trials.choice(tokens[d], values[d], edge, score, None if noise is None else noise[d], j)
            stop_chain = False
            if controller is not None:
                stop_chain = not controller.push(noisy_confidence(score, noise[d] if noise is not None else None, j, beta))
            elif confs is not None:
                if noisy:
                    confs.append(noisy_confidence(score, noise[d] if noise is not None else None, j, beta))
                else:
                    p = np.exp(score - score.max())
                    confs.append(float(p[j] / p.sum()))
            elif noisy:
                c = noisy_confidence(score, noise[d] if noise is not None else None, j, beta)
                if round_ms is not None:
                    conf.append(c)
                else:
                    # This guard changes verification work only: normal target-keyed
                    # sampling still chooses the output when no draft is sent.
                    if d == 0 and c < trial_control.ACTIVE.get('fnc_zero',0)/1000.0:
                        return []
                    chain *= c
                    if d > 0 and chain < confidence:
                        break
            elif confidence > 0:
                p = np.exp(score - score.max())
                chain *= float(p[j] / p.sum())
                if d > 0 and chain < confidence:
                    break
            prev = int(tokens[d, j])
            out.append(prev)
            if stop_chain:
                break
        if controller is not None:
            return out[:controller.finish()]
        if noisy and round_ms is not None and confs is None:
            out = out[:best_depth(conf, round_ms)]
        return out

    def propose(self, pending: int, depth: int, sampling: Sampling | None, confidence: float = 0.0, *,
                beta: float = 0.0, round_ms: tuple[float, ...] | None = None) -> list[int]:
        """Up to ``depth`` drafts for the positions after the pending token (which sits at context_end); ``beta``,
        ``round_ms``: the noise-aware stop rules (``chain``)."""

        depth = min(depth, self.block - 1)
        if depth < 1 or self.context_end == 0:
            return []
        tokens, values, proj = self.candidates(pending, depth)
        if beta > 0:
            return self.chain(tokens, values, proj, pending, self.context_end + 1, sampling, confidence, beta=beta,
                              round_ms=round_ms)
        return self.chain(tokens, values, proj, pending, self.context_end + 1, sampling, confidence)
