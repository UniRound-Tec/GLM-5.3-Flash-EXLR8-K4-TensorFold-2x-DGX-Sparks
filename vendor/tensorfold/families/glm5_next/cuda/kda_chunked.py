"""GLM-5.3-Flash's KDA prompt chunks in chunked (WY) form (kda_chunk.cu): 32-row sub-chunks at absolute positions,
one CUDA block per head carrying the 128 x 128 state from sub-chunk to sub-chunk in registers.

Per row the inputs are kda.cu's (conv + SiLU rounded to bf16, fp32 L2 norms, q times 128^-0.5, beta =
bf16(sigmoid(b)), the per-channel decay g = e^l, l = lower * sigmoid(exp(A_log) * (a + dt_bias))); the delta rule
S <- S diag(g); S <- S + beta (v - S k) k^T; o = S q, o rounded to bf16, then the gated RMSNorm. S is [value, key].

A sub-chunk of rows 0..31 entering state S0, E(t -> s) = g_{t+1} ... g_s per key channel (empty: 1):

    L[s, t]  = beta_s sum_c k_s k_t E(t -> s)      t < s          M[s, t] = sum_c q_s k_t E(t -> s)     t <= s
    T        = (I + L)^-1                           Y = beta (v - (k E(0 -> row)) S0^T),   V' = T Y
    O        = (q E(0 -> row)) S0^T + M V'           S_end = S0 diag(E(0 -> 31)) + V'^T (k E(row -> 31))

Every decay is a running product of per-row decays <= 1, multiplied in row order as the serial chain decays its state
(no exponent of a difference of cumulative sums: G reaches -160 in a sub-chunk, and ulp(160) = 1.5e-5 would land on
decays near 1). All products run in fp32 FMAs in a fixed order, so the kernel is deterministic, and a row's bits
depend only on its sub-chunk's rows and the entering state: the same whichever prefill chunk the row came in, as long
as prompt chunks start on multiples of the sub-chunk (TF_GLM_PROMPT_GRID, which TF_GLM_KDA_CHUNKED=1 sets to 64).
Rows outside the prompt chunk (a short first or last sub-chunk) are zero rows: decay 1, no key, no value, beta 0.
It is not kda.cu's arithmetic (different sums, so bf16 outputs may round the other way and states differ at fp32
rounding); its error against the float64 delta rule is the serial kernel's order (tests/cuda/test_glm_kda_chunked.py).

The torch functions (``*_reference``) are the CPU references of the tests; ``chain`` runs the CUDA kernel.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import torch

DK = DV = 128
TAPS = 4
CHUNK = 32              # rows of a sub-chunk (the state is carried between sub-chunks); kda_chunk.cu's BT
SCALE = 1.0 / math.sqrt(DK)


# -- references (torch, CPU or GPU) ----------------------------------------------------------------------------
def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(x.dtype)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def prep_reference(p: torch.Tensor, b_off: int, a: torch.Tensor, conv_state: torch.Tensor, conv_w: torch.Tensor,
                   a_log: torch.Tensor, dt_bias: torch.Tensor, lower: float, rows: int):
    """kda.cu's per-row work in fp32: q, k [rows, H, 128] normalized (q scaled), v [rows, H, 128] (bf16 values),
    the log-decay l [rows, H, 128] (kda.cu's decay is e^l) and beta [rows, H]."""

    H = a_log.numel()
    C = 3 * H * DK
    x = torch.cat([conv_state.float(), p[:rows, :C].float()])
    w = conv_w.float()
    acc = torch.zeros((rows, C), dtype=torch.float32, device=p.device)
    for tap in range(TAPS):
        acc = acc + w[:, tap] * x[tap:tap + rows]
    act = _bf(acc / (1.0 + torch.exp(-acc))).view(rows, 3, H, DK)

    def l2(t: torch.Tensor) -> torch.Tensor:
        return t * (1.0 / torch.sqrt((t * t).sum(-1, keepdim=True) + 1e-6))

    q = l2(act[:, 0]) * torch.tensor(SCALE, dtype=torch.float32)
    k = l2(act[:, 1])
    v = act[:, 2].contiguous()
    rate = torch.exp(a_log.float()).view(1, H, 1)
    ga = a[:rows].float().view(rows, H, DK) + dt_bias.float().view(1, H, DK)
    lg = torch.tensor(lower, dtype=torch.float32) * _sigmoid(rate * ga)
    beta = _bf(_sigmoid(p[:rows, b_off:b_off + H].float()))
    return q, k, v, lg, beta


def recurrent_reference(q, k, v, lg, beta, state, dtype=torch.float64):
    """The delta rule row by row (kda.cu's chain): read-outs [rows, H, 128] and the final state [H, 128, 128]."""

    q, k, v, beta = (t.to(dtype) for t in (q, k, v, beta))
    g = torch.exp(lg.to(dtype))
    S = state.to(dtype).clone()
    out = torch.empty(v.shape, dtype=dtype, device=v.device)
    for r in range(q.shape[0]):
        S = S * g[r][:, None, :]
        kv = (S * k[r][:, None, :]).sum(-1)
        delta = (v[r] - kv) * beta[r][:, None]
        S = S + delta[:, :, None] * k[r][:, None, :]
        out[r] = (S * q[r][:, None, :]).sum(-1)
    return out, S


def chunked_reference(q, k, v, lg, beta, state, pos: int = 0, dtype=torch.float64):
    """The chunked algorithm as the kernel runs it (sub-chunks of CHUNK rows at absolute positions, decays as running
    products, T by forward substitution): read-outs [rows, H, 128] and the final state [H, 128, 128]."""

    rows, H, _ = q.shape
    dev = q.device
    off = pos % CHUNK
    nc = -(-(rows + off) // CHUNK)
    total = nc * CHUNK

    def pad(t, fill=0.0):
        t = t.to(dtype)
        z = torch.full((total,) + tuple(t.shape[1:]), fill, dtype=dtype, device=dev)
        z[off:off + rows] = t
        return z.view((nc, CHUNK) + tuple(t.shape[1:])).transpose(1, 2)     # [nc, H, CHUNK, ...]

    qc, kc, vc, bc = pad(q), pad(k), pad(v), pad(beta)
    gc = pad(torch.exp(lg.to(dtype)), 1.0)
    S = state.to(dtype).clone()                     # [H, V, K]
    out = torch.empty((total, H, DV), dtype=dtype, device=dev)
    for c in range(nc):
        qx, kx, vx, gx, bx = qc[c], kc[c], vc[c], gc[c], bc[c]     # [H, T, 128] (beta [H, T])
        E = torch.zeros((H, CHUNK, CHUNK, DK), dtype=dtype, device=dev)   # E[:, s, t] = E(t -> s), t <= s
        for t in range(CHUNK):
            e = torch.ones((H, DK), dtype=dtype, device=dev)
            E[:, t, t] = e
            for s in range(t + 1, CHUNK):
                e = e * gx[:, s]
                E[:, s, t] = e
        Akk = torch.einsum("hsc,htc,hstc->hst", kx, kx, E)
        Aqk = torch.einsum("hsc,htc,hstc->hst", qx, kx, E)
        tri = torch.ones((CHUNK, CHUNK), dtype=torch.bool, device=dev)
        L = torch.where(torch.tril(tri, -1), bx[:, :, None] * Akk, torch.zeros_like(Akk))
        M = torch.where(torch.tril(tri), Aqk, torch.zeros_like(Aqk))
        T = torch.eye(CHUNK, dtype=dtype, device=dev).expand(L.shape).clone()
        for s in range(1, CHUNK):
            T[:, s, :] = -(L[:, s, :, None] * T).sum(-2)
            T[:, s, s] = 1.0
        fwd = torch.cumprod(gx, dim=1)                              # E(0 -> row)
        back = torch.cat([torch.flip(torch.cumprod(torch.flip(gx[:, 1:], [1]), 1), [1]),
                          torch.ones_like(gx[:, :1])], 1)
        ST = S.transpose(1, 2)                                      # [H, K, V]
        Y = bx[:, :, None] * (vx - (kx * fwd) @ ST)
        Vn = T @ Y
        out[c * CHUNK:(c + 1) * CHUNK] = ((qx * fwd) @ ST + M @ Vn).transpose(0, 1)
        S = S * fwd[:, -1:, :] + Vn.transpose(1, 2) @ (kx * back)
    return out[off:off + rows], S


def gate_norm_reference(y: torch.Tensor, gate: torch.Tensor, norm_w: torch.Tensor, eps: float) -> torch.Tensor:
    """kda.cu's gated RMSNorm of bf16 read-outs y [rows, H, 128] with gate rows [rows, H * 128]: bf16 [rows, H*128]."""

    rows, H, _ = y.shape
    yb = _bf(y.float())
    rinv = 1.0 / torch.sqrt((yb * yb).sum(-1, keepdim=True) / DV + eps)
    yw = norm_w.float() * (yb * rinv)
    return (yw * _sigmoid(gate[:rows].float().view(rows, H, DV))).to(torch.bfloat16).view(rows, H * DV)


def chain_reference(p, b_off, a, g, conv_state, conv_w, state_in, a_log, dt_bias, norm_w, eps, lower, rows, *,
                    pos: int = 0, chunked: bool = True, dtype=torch.float32, mm_round=None):
    """A whole prompt chunk: (out bf16 [rows, H * 128], final state fp32 [H, 128, 128], read-outs y [rows, H, 128])."""

    q, k, v, lg, beta = prep_reference(p, b_off, a, conv_state, conv_w, a_log, dt_bias, lower, rows)
    if chunked:
        y, S = chunked_reference(q, k, v, lg, beta, state_in, pos, dtype=dtype, mm_round=mm_round)
    else:
        y, S = recurrent_reference(q, k, v, lg, beta, state_in, dtype=dtype)
    return gate_norm_reference(y, g, norm_w, eps), S.float(), y


def chain_reference(p, b_off, a, g, conv_state, conv_w, state_in, a_log, dt_bias, norm_w, eps, lower, rows, *,
                    pos: int = 0, chunked: bool = True, dtype=torch.float32):
    """A whole prompt chunk: (out bf16 [rows, H * 128], final state fp32 [H, 128, 128], read-outs y [rows, H, 128])."""

    q, k, v, lg, beta = prep_reference(p, b_off, a, conv_state, conv_w, a_log, dt_bias, lower, rows)
    if chunked:
        y, S = chunked_reference(q, k, v, lg, beta, state_in, pos, dtype=dtype)
    else:
        y, S = recurrent_reference(q, k, v, lg, beta, state_in, dtype=dtype)
    return gate_norm_reference(y, g, norm_w, eps), S.float(), y


# -- the CUDA kernel --------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_kda_chunk", sources=[str(here / "kda_chunk.cpp"), str(here / "kda_chunk.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def chain(p: torch.Tensor, b_off: int, a: torch.Tensor, g: torch.Tensor, conv_state: torch.Tensor,
          conv_w: torch.Tensor, state_in: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
          norm_w: torch.Tensor, eps: float, lower: float, rows: int, out: torch.Tensor, state_out: torch.Tensor,
          pos: int = 0) -> torch.Tensor:
    """kda.chain for a prompt chunk in chunked form: projection rows p [q | k | v | ... | b at b_off ...], bf16 gate
    rows a (decay) and g (output gate), conv state [3, 3 H 128] (read, not shifted), state_in -> state_out
    [H, 128, 128]; ``pos`` is the first row's absolute position. Writes out[:rows] (bf16 [rows, H * 128])."""

    rows = int(rows)
    _ext().chain(p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv_state, conv_w, state_in, a_log,
                 dt_bias, norm_w, float(eps), float(lower), rows, int(pos) % CHUNK, out, state_out)
    return out[:rows]
