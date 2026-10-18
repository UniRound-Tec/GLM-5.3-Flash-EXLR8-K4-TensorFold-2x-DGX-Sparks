"""GLM-5.3-Flash's 4-bit and BF16 dense matmuls; the same groups and K slices at any row count keep each row's bits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels import qmm as shared

BN = 64                   # columns per stored tile
GS = 64                   # inputs per quantization group


@dataclass
class Q4:
    """A 4-bit group-64 matrix [n, k] packed for the shared lane matmul (``shared.pack``)."""

    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    n: int
    k: int
    gs: int = GS

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))


def as_i32(w: torch.Tensor) -> torch.Tensor:
    return w.view(torch.int32) if w.dtype != torch.int32 else w


def make_q4(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> Q4:
    """From MLX arrays: weight (N, K/8) uint32/int32, scales/biases (N, K/64) bf16."""

    w = as_i32(weight)
    n, k8 = w.shape
    if scales.shape != (n, k8 // 8):
        raise ValueError(f"group-64 scales expected ({n}, {k8 // 8}), got {tuple(scales.shape)}")
    p = shared.pack(w, scales, biases, GS)
    return Q4(p.weight, p.scales, p.biases, n, k8 * 8)


def stack_q4(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Q4:
    """Rows of several MLX (weight, scales, biases) with the same K, stacked in order, then tiled."""

    return make_q4(torch.cat([as_i32(p[0]) for p in parts]), torch.cat([p[1] for p in parts]),
                   torch.cat([p[2] for p in parts]))


def to_mlx(q: Q4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return shared.unpack(q)


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Reference: MLX (..., N, K/8) words -> (..., N, K) fp32 values s * q + b."""

    k8 = words.shape[-1]
    w = as_i32(words).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], k8 * 8).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(GS, dim=-1)
    b = biases.to(torch.float32).repeat_interleave(GS, dim=-1)
    return q * s + b


def dequantize_q4(q: Q4) -> torch.Tensor:
    w, s, b = to_mlx(q)
    return dequantize(w.contiguous(), s.contiguous(), b.contiguous())


# K split rule for shapes without a table entry: split until the column tiles times slices reach this many programs
SPLIT_TARGET = 192


# Per-shape K slices determine arithmetic equally for every row; group-step, warp, and stage settings preserve bits.
SHAPE_SK: dict[str, int] = {"12576x4096": 4, "4096x4096": 2, "2048x4096": 4, "8192x1536": 4, "8192x512": 4,
                            "4096x8192": 8}


def split_k(n: int, k: int) -> int:
    """Choose K slices from shape alone until column tiles times slices reach SPLIT_TARGET; changing the target changes every row equally."""

    forced = SHAPE_SK.get(f"{n}x{k}")
    if forced:
        return forced
    tiles = -(-n // BN)
    groups = k // GS
    sk = 1
    while sk < 8 and tiles * sk < SPLIT_TARGET and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


def bucket(m: int) -> int:
    for b in (16, 32, 64, 128):
        if m <= b:
            return b
    raise ValueError(f"at most 128 rows, got {m}")


@triton.jit
def _group_sums(X, XS, x_stride, K: tl.constexpr, GB: tl.constexpr):
    m = tl.program_id(0)
    gb = tl.program_id(1)
    KG: tl.constexpr = K // 64
    g = gb * GB + tl.arange(0, GB)
    k = tl.arange(0, 64)
    ok = g < KG
    x = tl.load(X + m * x_stride + g[:, None] * 64 + k[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    tl.store(XS + m * KG + g, tl.sum(x, axis=1), mask=ok)


def group_sums(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """(M, K) bf16 (rows may be strided) -> (M, K/64) fp32 sums of each 64-input group."""

    m, k = x.shape
    kg = k // GS
    if out is None:
        out = torch.empty((m, kg), dtype=torch.float32, device=x.device)
    _group_sums[(m, triton.cdiv(kg, 16))](x, out, x.stride(0), K=k, GB=16, num_warps=2)
    return out


@triton.jit
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr, F32: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    acc = tl.load(PART + offs, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
    if F32:
        tl.store(OUT + offs, acc, mask=ok)
    else:
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=ok)


@dataclass
class B16:
    """A BF16 matrix [n, k] as the checkpoint stores it (EXL3 checkpoints keep every non-expert weight in BF16)."""

    weight: torch.Tensor      # [n, k] bf16, contiguous
    n: int
    k: int

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()


def make_b16(weight: torch.Tensor) -> B16:
    w = weight.to(torch.bfloat16).contiguous()
    return B16(w, int(w.shape[0]), int(w.shape[1]))


def quantize4(w: torch.Tensor, chunk: int = 8192) -> Q4:
    """Quantize bf16 weights to tiled affine 4-bit groups of 64 for drafting only, never verification."""

    n, k = w.shape
    words = torch.empty((n, k // 8), dtype=torch.int32, device=w.device)
    scales = torch.empty((n, k // 64), dtype=torch.bfloat16, device=w.device)
    biases = torch.empty_like(scales)
    for r in range(0, n, chunk):
        g = w[r:r + chunk].float().view(-1, k // 64, 64)
        lo, hi = g.amin(-1), g.amax(-1)
        scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
        bias = lo.to(torch.bfloat16)
        q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15).to(torch.int32)
        q = q.view(-1, k // 8, 8)
        part = torch.zeros(q.shape[:2], dtype=torch.int32, device=w.device)
        for j in range(8):
            part |= q[..., j] << (4 * j)
        words[r:r + chunk], scales[r:r + chunk], biases[r:r + chunk] = part, scale, bias
    return make_q4(words, scales.contiguous(), biases.contiguous())


def quantize4_mse(w: torch.Tensor, chunk: int = 4096, grid: int = 16) -> Q4:
    """TF_GLM_DENSE=q4: bf16 weights as affine 4-bit groups of 64, each group's range the one of ``grid`` clippings
    (1.0 down to 0.55 of its min / max) with the least squared error, scale and bias as stored (bf16). Lossy."""

    return make_q4(*quantize4_mse_raw(w, chunk, grid))


def quantize4_mse_raw(w: torch.Tensor, chunk: int = 4096,
                      grid: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``quantize4_mse``'s groups in MLX's layout, untiled: words [n, k / 8] int32 (8 nibbles low to high), scales and
    biases [n, k / 64] bf16 (what ``latent.AbsorbQ4`` reads)."""

    n, k = w.shape
    words = torch.empty((n, k // 8), dtype=torch.int32, device=w.device)
    scales = torch.empty((n, k // 64), dtype=torch.bfloat16, device=w.device)
    biases = torch.empty_like(scales)
    for r in range(0, n, chunk):
        g = w[r:r + chunk].float().view(-1, k // 64, 64)
        lo0, hi0 = g.amin(-1), g.amax(-1)
        best_err = best_s = best_b = None
        for i in range(grid):
            a = 1.0 - 0.45 * i / (grid - 1)
            lo, hi = lo0 * a, hi0 * a
            s = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
            b = lo.to(torch.bfloat16)
            q = torch.round((g - b.float()[..., None]) / s.float()[..., None]).clamp(0, 15)
            err = ((q * s.float()[..., None] + b.float()[..., None] - g) ** 2).sum(-1)
            if best_err is None:
                best_err, best_s, best_b = err, s, b
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s = torch.where(take, s, best_s)
                best_b = torch.where(take, b, best_b)
        q = torch.round((g - best_b.float()[..., None]) / best_s.float()[..., None]).clamp(0, 15).to(torch.int32)
        q = q.view(-1, k // 8, 8)
        part = torch.zeros(q.shape[:2], dtype=torch.int32, device=w.device)
        for j in range(8):
            part |= q[..., j] << (4 * j)
        words[r:r + chunk], scales[r:r + chunk], biases[r:r + chunk] = part, best_s, best_b
    return words, scales.contiguous(), biases.contiguous()


DRAFT_QUANTS = ("minmax", "mse")


def draft_quant(value: str | None = None) -> str:
    """TF_GLM_DRAFT_QUANT: how the draft-only 4-bit copies (DFlash2's matrices, an EXL3 checkpoint's draft head) are
    made: minmax (the default: each group's full min / max range, ``quantize4``) or mse (the least-squares clipping of
    ``quantize4_mse``). Drafts only propose, so either keeps every reply; only acceptance can move."""

    import os

    kind = os.environ.get("TF_GLM_DRAFT_QUANT", "") if value is None else value
    kind = kind or "minmax"
    if kind not in DRAFT_QUANTS:
        raise ValueError(f"TF_GLM_DRAFT_QUANT is minmax or mse, not {kind!r}")
    return kind


def draft_quantize4(w: torch.Tensor, kind: str | None = None) -> Q4:
    """A draft-only 4-bit copy of bf16 ``w`` by TF_GLM_DRAFT_QUANT (``draft_quant``); never used to verify."""

    from . import trial_control
    if trial_control.enabled():
        assert draft_quant(kind) == "minmax"
        return trial_control.dual_quantize(w)
    return quantize4_mse(w) if draft_quant(kind) == "mse" else quantize4(w)


def make_dense_q4(weight: torch.Tensor) -> "Q4 | F8":
    return quantize4_mse(weight.to(torch.bfloat16)) if weight.shape[1] % 64 == 0 else make_f8(weight)


def stack_dense_q4(parts: list[torch.Tensor]) -> "Q4 | F8":
    return make_dense_q4(torch.cat([p.to(torch.bfloat16) for p in parts]))


def stack_b16(parts: list[torch.Tensor]) -> B16:
    """Rows of several BF16 matrices with the same K, stacked in order."""

    return make_b16(torch.cat([p.to(torch.bfloat16) for p in parts]))


# BF16 matmuls: columns and K per step; the K slices come from ``split_k`` as for Q4 (fixed by the shape)
B16_BN, B16_BK = 64, 64
# from this many rows (a prompt chunk) one program runs all of a matmul's K slices and sums them in _reduce's order:
# the same bits as the split grid, without writing and reading back fp32 partials for every row. Measured at 2,048
# rows (GB10): 1.3-3.2x on N >= 512, K <= 4,096; slower on narrow outputs (N 32-288) and long K (6,144, 8,192)
SEQ_ROWS = int(__import__("os").environ.get("TF_GLM_SEQ_ROWS", "256"))


def _seq(sk: int, m: int, n: int, k: int) -> bool:
    return sk > 1 and m >= SEQ_ROWS and n >= 512 and k <= 4096


# prompt chunks (SEQ_ROWS rows or more): (BM, BN, warps, stages) and SEQ by per-rank shape (N, K), swept on GB10 at
# 2,048 rows; no choice changes a row's bits. Other shapes take PROMPT_DEFAULT.
F8_PROMPT = {(6432, 4096): (128, 128, 8, 3), (2048, 4096): (64, 64, 8, 3), (4096, 1024): (64, 128, 8, 3),
             (4096, 4096): (64, 128, 8, 3), (8192, 1536): (64, 128, 4, 3), (12288, 4096): (128, 64, 4, 3),
             (4096, 6144): (128, 64, 8, 4), (4096, 8192): (64, 64, 4, 4), (1536, 4096): (64, 64, 8, 3)}
PROMPT_DEFAULT = (64, 128, 8, 3)


def _prompt_cfg(sk: int, m: int, n: int, k: int):
    """(BM, BN, warps, stages, seq) for a prompt chunk's matmul, or None below SEQ_ROWS rows."""
    if m < SEQ_ROWS:
        return None
    if (n, k) in F8_PROMPT:
        return (*F8_PROMPT[(n, k)], sk > 1)
    return (*PROMPT_DEFAULT, _seq(sk, m, n, k))


@triton.jit
def _bmm(X, W, OUT, PART, M, x_stride, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr, SEQ: tl.constexpr = False):
    """Multiply bf16 rows over a fixed K slice in order with fp32 sums, independently of other rows; SEQ: one program
    runs every slice and sums them as _reduce does (the same bits, without the fp32 partials' round trip)."""

    PER: tl.constexpr = K // SK
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    if SEQ:
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SK):
            part = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
            for k0 in range(s * PER, s * PER + PER, BK):
                x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
                w = tl.load(W + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
                part = part + tl.dot(x, tl.trans(w))
            if s == 0:
                acc = part
            else:
                acc = acc + part
    else:
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for k0 in range(pid_s * PER, pid_s * PER + PER, BK):
            x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
            acc = acc + tl.dot(x, tl.trans(w))
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1 or SEQ:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


# (warps, stages) for the BF16 matmul by row bucket: no choice changes bits
B16_CONFIG = {16: (4, 3), 32: (4, 3), 64: (4, 2), 128: (8, 2)}


def matmul(x: torch.Tensor, q: Q4 | B16, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, part: torch.Tensor | None = None) -> torch.Tensor:
    """Multiply strided bf16 x by Q4, BF16 or FP8 q.T, returning bf16 or unrounded fp32 with f32; BF16 and FP8 weights ignore xs."""

    if isinstance(q, F8) and hasattr(q, 'block'):
        from .source_fp8 import matmul as source_matmul
        return source_matmul(x,q,out=out,f32=f32,part=part)
    if isinstance(q, B16) and hasattr(q,'packed'):
        from .zip_k4 import matmul as hybrid_matmul
        return hybrid_matmul(x,q,out=out,f32=f32,part=part)
    if isinstance(q, B16):
        return _matmul_b16(x, q, out=out, f32=f32, part=part)
    if isinstance(q, F8):
        return _matmul_f8(x, q, out=out, f32=f32, part=part)
    if x.shape[1] != q.k or x.stride(1) != 1 or x.dtype != torch.bfloat16:
        raise ValueError(f"matmul: x {tuple(x.shape)} {x.dtype} does not match K={q.k}")
    if out is not None and (out.shape != (x.shape[0], q.n) or not out.is_contiguous()):
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({x.shape[0]}, {q.n})")
    return shared.matmul(x, q, group_sums(x) if xs is None else xs, sk=split_k(q.n, q.k), f32=f32, out=out,
                         variant=dec_tile(q.n, q.k) if x.shape[0] <= 16 else None)


# Decode rows (up to 16) of a Q4 matmul on another column tile / warp / stage count (``shared.qmm_cfg``): every
# output keeps its chain, so every choice gives the same bits. By per-rank shape (n, k), measured on GB10.
DEC_TILES: dict[tuple[int, int], int] = {(4096, 4096): 2, (2048, 4096): 2, (8192, 1536): 2, (4096, 8192): 4,
                                         (4096, 1024): 2}
_dec_env: tuple[str, dict] = ("", {})


def dec_tile(n: int, k: int) -> int | None:
    """TF_GLM_Q4_TILE: unset or "table" the table above, "stock" none, a number that config for every shape."""

    import os

    global _dec_env
    value = os.environ.get("TF_GLM_Q4_TILE", "") or "table"
    if value != _dec_env[0]:
        table = DEC_TILES if value == "table" else {} if value == "stock" else None
        _dec_env = (value, table if table is not None else {"all": int(value)})
    table = _dec_env[1]
    return table.get("all", table.get((n, k)))


def b16_split_k(n: int, k: int) -> int:
    """K slices of a BF16 matmul: like ``split_k``, fixed by the shape, in units of B16_BK."""

    tiles = -(-n // B16_BN)
    steps = k // B16_BK
    sk = 1
    while sk < 8 and tiles * sk < SPLIT_TARGET and steps % (sk * 2) == 0 and steps // (sk * 2) >= 4:
        sk *= 2
    return sk


def _matmul_b16(x: torch.Tensor, q: B16, *, out: torch.Tensor | None, f32: bool,
                part: torch.Tensor | None) -> torch.Tensor:
    m, k = x.shape
    if k != q.k or x.stride(1) != 1 or x.dtype != torch.bfloat16 or k % B16_BK:
        raise ValueError(f"matmul: x {tuple(x.shape)} {x.dtype} does not match K={q.k}")
    bm = bucket(min(m, 128))              # a prompt chunk runs as 128-row blocks: no bucket changes a row's bits
    warps, stages = B16_CONFIG[bm]
    sk = b16_split_k(q.n, q.k)
    seq = _seq(sk, m, q.n, q.k)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    if sk > 1 and not seq:
        need = sk * m * q.n
        if part is None or part.numel() < need:
            part = torch.empty((need,), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, B16_BN), 1 if seq else sk)
    _bmm[grid](x, q.weight, out, part if sk > 1 and not seq else out, m, x.stride(0), N=q.n, K=k, SK=sk, BM=bm,
               BLOCK_N=B16_BN, BK=B16_BK, F32=f32, SEQ=seq, num_warps=warps, num_stages=stages)
    if sk > 1 and not seq:
        total = m * q.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


# -- FP8 weights (TF_GLM_DENSE=fp8): the checkpoint's BF16 matrices stored as e4m3 with an fp32 scale per row and
# 128-column block, quantized once at load. Lossy (replies differ from BF16 weights); the activations stay bf16 and
# the kernel keeps _bmm's structure (a row's bits never depend on the other rows), so drafted replies still equal
# serial ones under it.
F8_BLOCK = 128
F8_MAX = 448.0


@dataclass
class F8:
    """A BF16 matrix [n, k] held as FP8 e4m3 [n, k] and fp32 scales [n, k / 128]: W ~ q * scale."""

    weight: torch.Tensor      # [n, k] torch.float8_e4m3fn
    scale: torch.Tensor       # [n, k // 128] fp32
    n: int
    k: int

    def nbytes(self) -> int:
        return self.weight.numel() + self.scale.numel() * 4


def make_f8(weight: torch.Tensor, rows: int = 8192) -> F8 | B16:
    """Quantize a BF16 matrix a block of 128 columns at a time (absmax / 448 scale, round to nearest); a matrix whose
    width is not a multiple of 128 stays BF16."""

    n, k = weight.shape
    if k % F8_BLOCK:
        return make_b16(weight)
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((n, k // F8_BLOCK), dtype=torch.float32, device=weight.device)
    for r in range(0, n, rows):
        g = weight[r:r + rows].float().view(-1, k // F8_BLOCK, F8_BLOCK)
        s = (g.abs().amax(-1) / F8_MAX).clamp_min(1e-30)
        q[r:r + rows] = (g / s[..., None]).clamp(-F8_MAX, F8_MAX).view(-1, k).to(torch.float8_e4m3fn)
        scale[r:r + rows] = s
    return F8(q, scale, int(n), int(k))


def stack_f8(parts: list[torch.Tensor]) -> F8 | B16:
    return make_f8(torch.cat([p.to(torch.bfloat16) for p in parts]))


@triton.jit
def _fmm(X, W, S, OUT, PART, M, x_stride, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr, SEQ: tl.constexpr = False):
    """_bmm over FP8 weights: each BK step's product (bf16 inputs, fp32 sums) times its 128-column block's scales;
    SEQ as in _bmm."""

    PER: tl.constexpr = K // SK
    KB: tl.constexpr = K // 128
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    if SEQ:
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for p in tl.static_range(SK):
            part = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
            for k0 in range(p * PER, p * PER + PER, BK):
                x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
                w = tl.load(W + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
                s = tl.load(S + rn.to(tl.int64) * KB + k0 // 128, mask=n_ok, other=0.0)
                part = part + tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s[None, :]
            if p == 0:
                acc = part
            else:
                acc = acc + part
    else:
        acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
        for k0 in range(pid_s * PER, pid_s * PER + PER, BK):
            x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
            s = tl.load(S + rn.to(tl.int64) * KB + k0 // 128, mask=n_ok, other=0.0)
            acc = acc + tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1 or SEQ:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


def _matmul_f8(x: torch.Tensor, q: F8, *, out: torch.Tensor | None, f32: bool,
               part: torch.Tensor | None) -> torch.Tensor:
    m, k = x.shape
    if k != q.k or x.stride(1) != 1 or x.dtype != torch.bfloat16 or k % F8_BLOCK:
        raise ValueError(f"matmul: x {tuple(x.shape)} {x.dtype} does not match K={q.k}")
    bm = bucket(min(m, 128))
    warps, stages = F8_CONFIG[bm]
    sk = b16_split_k(q.n, q.k)
    seq = _seq(sk, m, q.n, q.k)
    bn = F8_BN_DECODE if bm == 16 else F8_BN
    cfg = _prompt_cfg(sk, m, q.n, q.k)
    if cfg is not None:
        bm, bn, warps, stages, seq = cfg
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    if sk > 1 and not seq:
        need = sk * m * q.n
        if part is None or part.numel() < need:
            part = torch.empty((need,), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, bn), 1 if seq else sk)
    _fmm[grid](x, q.weight, q.scale, out, part if sk > 1 and not seq else out, m, x.stride(0), N=q.n, K=k, SK=sk,
               BM=bm, BLOCK_N=bn, BK=B16_BK, F32=f32, SEQ=seq, num_warps=warps, num_stages=stages)
    if sk > 1 and not seq:
        total = m * q.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


F8_BN = 64
F8_BN_DECODE = 32          # decode windows (up to 16 rows): narrower column tiles, deeper pipeline (measured +8%)
# (warps, stages) by row bucket; neither they nor the column tile change a row's bits
F8_CONFIG = {16: (4, 6), 32: (4, 3), 64: (4, 2), 128: (8, 2)}


def dense_kind() -> str:
    """TF_GLM_DENSE: how an EXL3 checkpoint's BF16 (non-expert) weights are held: bf16 (as stored) or fp8."""

    import os

    kind = os.environ.get("TF_GLM_DENSE", "bf16")
    if kind not in ("bf16", "fp8", "q4"):
        raise ValueError(f"TF_GLM_DENSE is bf16, fp8 or q4, not {kind!r}")
    return kind


KVB_KINDS = ("bf16", "fp8", "q4")


def kvb_kind(env=None) -> str:
    """TF_GLM_KVB: how an EXL3 checkpoint's kv_b_proj is held for the latent path's absorb and expand (the kernels
    every decode round runs once a DSA layer): bf16 (as stored, the default), fp8 (e4m3, an fp32 scale per row and
    128 latent columns, ``make_f8``'s layout) or q4 (MSE-searched affine groups of 64 along the latent, the MLX
    checkpoint's kernels). fp8 and q4 are lossy, like the rest of TF_GLM_DENSE; a row's bits still never depend on its
    window's other rows, so drafted replies equal serial ones under each."""

    import os

    kind = (os.environ if env is None else env).get("TF_GLM_KVB", "bf16").strip() or "bf16"
    if kind not in KVB_KINDS:
        raise ValueError(f"TF_GLM_KVB is bf16, fp8 or q4, not {kind!r}")
    return kind


def kvb_weights(base, kind: str):
    """A split_weights transform for TF_GLM_KVB: kv_b_proj at a byte a value plus fp32 scales a 128 block (fp8), or
    half a byte plus a bf16 scale and bias a group of 64 (q4); the latent path then holds no other copy of it."""

    def transform(name: str, info: dict) -> tuple[int, int]:
        total, extra = base(name, info)
        shape = info.get("shape") or []
        if kind == "bf16" or not total or "kv_b_proj" not in name or info.get("dtype") != "BF16" or len(shape) != 2:
            return total, extra
        n = total // 2
        if kind == "fp8" and shape[-1] % F8_BLOCK == 0:
            return n + n // F8_BLOCK * 4, extra
        if kind == "q4" and shape[-1] % 64 == 0:
            return n // 2 + n // 64 * 4, extra
        return total, extra
    return transform


# The matrices TF_GLM_DENSE=fp8 converts (weights.py's q4 / stack / head), for the startup memory estimate
_F8_NAMES = __import__("re").compile(
    r"(\.(q_proj|k_proj|v_proj|f_a_proj|g_a_proj|b_proj|f_b_proj|g_b_proj|o_proj|q_a_proj|kv_a_proj_with_mqa|q_b_proj"
    r"|wk|weights_proj|wq_b|gate_proj|up_proj|down_proj|eh_proj)\.weight|^lm_head\.weight)$")


def q4_weights(base):
    """fp8_weights, with the 4-bit matrices of TF_GLM_DENSE=q4 (all but the head and kv_b, which stay FP8) at half a
    byte a value plus a bf16 scale and bias a group of 64."""

    f8 = fp8_weights(base)

    def transform(name: str, info: dict) -> tuple[int, int]:
        total, extra = f8(name, info)
        shape = info.get("shape") or []
        if (total and info.get("dtype") == "BF16" and len(shape) == 2 and shape[-1] % 64 == 0
                and name != "lm_head.weight" and "kv_b_proj" not in name
                and "experts." not in name.replace("shared_experts.", "") and _F8_NAMES.search(name)):
            n = base(name, info)[0] // 2
            return n // 2 + n // 64 * 4, extra
        return total, extra
    return transform


def fp8_weights(base):
    """A split_weights transform that counts the matrices held as FP8 at a byte a value plus their block scales."""

    def transform(name: str, info: dict) -> tuple[int, int]:
        total, extra = base(name, info)
        shape = info.get("shape") or []
        if (total and info.get("dtype") == "BF16" and len(shape) == 2 and shape[-1] % F8_BLOCK == 0
                and "experts." not in name.replace("shared_experts.", "") and _F8_NAMES.search(name)):
            if name == "lm_head.weight":                  # keep the 4-bit draft head split_weights adds
                n = total * 16 // (2 * 16 + 9)          # its values: total = 2n + 9n / 16
                return n + n // F8_BLOCK * 4 + n * 9 // 16, extra
            n = total // 2
            return n + n // F8_BLOCK * 4, extra
        return total, extra
    return transform


def group_sums_for(q, x, out):
    # These correction sums are consumed only by affine Q4 matrices.
    # BF16, original FP8, and ZipK4 subclassing BF16 do not read them.
    from . import trial_control
    if trial_control.ACTIVE.get('d30',0) & 4 and isinstance(q,(B16,F8)):
        return None
    return group_sums(x,out)
