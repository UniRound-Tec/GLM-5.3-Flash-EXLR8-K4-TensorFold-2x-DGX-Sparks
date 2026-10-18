"""TF_GLM_KV=fp8: DSA's caches (the 512-wide latent a token and layer, the indexer's pooled keys) as FP8 e4m3 rows
with a power-of-two scale each, about half the bytes of bf16. Lossy (the owner's call), never inexact: a row's bytes
are a function of its bf16 row alone (one kernel helper, ``store_row``, quantizes prompt chunks' and decode windows'
rows), and the readers take the codes to bf16 (exact) and fold each row's scale into their fp32 products (exact for a
power of two), so a row's attention still depends on its query, its keys and the cache only: drafted == serial and
resumed == fresh hold as with bf16 (tests/cuda/test_glm_kv8.py). The values are those of the bf16 kernels on the
dequantized rows up to fp32 summation order (Triton lays converted tiles out for the tensor cores its own way).

A quantized cache is a uint8 tensor [rows, width + PAD]: the row's e4m3 codes, its fp32 scale, 12 zero bytes (rows
stay 16-byte aligned). One tensor a cache keeps the engine's generic row handling (slices, clones, kept snapshots'
saved rows, ``row_bytes``) as it is. The scale is 2^(ceil(log2 amax) - SHIFT): the row's largest value quantizes to
[128, 256], under e4m3's 448, so nothing saturates; a floating-point format's relative precision does not depend on
the scale, so a per-row power of two loses nothing to a finer or fp32 scale but the values it pushes below e4m3's
normal range (2^-6), under 2^-14 of the row's largest (DeepSeek-V3.2's indexer keeps FP8 keys with per-token ue8m0
scales alike). On GLM-5.3-Flash's own kv_a / kv_a_layernorm rows (amax ~3-5x the rms) per-token fp32, per-128-block
and fixed per-layer scales all land within 0.001 of this one's 2.65% relative error a row. bf16 caches stay as they were (a bf16 tensor [rows, width]); every kernel tells the two by dtype."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

KINDS = ("bf16", "fp8")
PAD = 16             # bytes after an FP8 row's codes: its fp32 scale, then zeros
SHIFT = 8            # a row's scale: 2^(ceil(log2 amax) - SHIFT), so its codes stay within +-256


def kv_kind(env=None) -> str:
    """TF_GLM_KV: bf16 (the default) or fp8, the DSA latent and indexer key caches' format (lossy: replies differ from
    bf16's; drafted replies still equal serial ones, resumed prompts fresh ones)."""

    kind = (os.environ if env is None else env).get("TF_GLM_KV", "bf16").strip() or "bf16"
    if kind not in KINDS:
        raise ValueError(f"TF_GLM_KV is bf16 or fp8, not {kind!r}")
    return kind


def row_bytes(width: int, kind: str) -> int:
    """Bytes a cache row of ``width`` values takes."""
    if kind not in KINDS:
        raise ValueError(f"a cache is bf16 or fp8, not {kind!r}")
    return width + PAD if kind == "fp8" else 2 * width


def zeros(rows: int, width: int, kind: str, device) -> torch.Tensor:
    """A zeroed cache of ``rows`` rows of ``width`` values (zero rows dequantize to zeros in either format)."""
    if kind == "fp8":
        if width % 4:
            raise ValueError(f"an FP8 cache row needs a width divisible by 4, not {width}")
        return torch.zeros((rows, width + PAD), dtype=torch.uint8, device=device)
    return torch.zeros((rows, row_bytes(width, kind) // 2), dtype=torch.bfloat16, device=device)


def quantized(cache: torch.Tensor) -> bool:
    return cache.dtype == torch.uint8


def kind_of(cache: torch.Tensor) -> str:
    return "fp8" if quantized(cache) else "bf16"


def width(cache: torch.Tensor) -> int:
    """Values a row of ``cache`` holds."""
    return cache.shape[-1] - PAD if quantized(cache) else cache.shape[-1]


def parts(cache: torch.Tensor):
    """A kernel's view of a cache: (codes or bf16 values, fp32 view for the scales, row stride in elements of the
    first, FP8). A bf16 cache passes itself as the unused scale argument."""
    if not quantized(cache):
        return cache, cache, cache.shape[-1], False
    if cache.stride(-1) != 1 or cache.shape[-1] % 16:
        raise ValueError("an FP8 cache: contiguous rows of width + 16 bytes")
    return cache.view(torch.float8_e4m3fn), cache.view(torch.float32), cache.shape[-1], True


# -- the format in torch (tests, and the definition the kernels must equal) -------------------------------------------
def scales(amax: torch.Tensor) -> torch.Tensor:
    """fp32 amax -> the fp32 power-of-two scale (the kernels' ``scale_of``, bit for bit)."""
    bits = amax.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((e - SHIFT).clamp(1, 254) << 23).view(torch.float32)


def quantize_rows(x: torch.Tensor) -> torch.Tensor:
    """bf16 (or fp32) rows [n, width] -> an FP8 cache's rows [n, width + PAD], as the write kernels store them."""
    x = x.float()
    s = scales(x.abs().amax(dim=1))
    inv = ((254 - (s.view(torch.int32) >> 23)) << 23).view(torch.float32)
    out = torch.zeros((x.shape[0], x.shape[1] + PAD), dtype=torch.uint8, device=x.device)
    out[:, :x.shape[1]] = (x * inv[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    out[:, x.shape[1]:x.shape[1] + 4] = s[:, None].contiguous().view(torch.uint8)
    return out


def dequantize(cache: torch.Tensor) -> torch.Tensor:
    """A cache's rows as fp32 [n, width] (exact: e4m3 codes times a power of two; bf16 rows as they are)."""
    if not quantized(cache):
        return cache.float()
    w = width(cache)
    codes = cache[..., :w].contiguous().view(torch.float8_e4m3fn).float()
    s = cache[..., w:w + 4].contiguous().view(torch.float32)
    return codes * s


# -- the format in kernels ------------------------------------------------------------------------------------------
@triton.jit
def scale_of(amax):
    """fp32 amax -> 2^(ceil(log2 amax) - SHIFT) and its reciprocal, both exact powers of two (biased exponent held
    within 1 .. 254: a zero row gets 2^-126 and zeros)."""
    bits = amax.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    e = tl.minimum(tl.maximum(e - 8, 1), 254)
    return (e << 23).to(tl.float32, bitcast=True), ((254 - e) << 23).to(tl.float32, bitcast=True)


@triton.jit
def store_row(VALS, SCL, row, x, LW: tl.constexpr, RS: tl.constexpr, FP8: tl.constexpr):
    """Row ``row`` (int64) of a cache <- x fp32 [LW] as bf16 values, or FP8: the bf16 row's codes and scale."""
    k = tl.arange(0, LW)
    xb = x.to(tl.bfloat16)
    if FP8:
        xf = xb.to(tl.float32)
        s, inv = scale_of(tl.max(tl.abs(xf), 0))
        tl.store(VALS + row * RS + k, (xf * inv).to(tl.float8e4nv))
        tl.store(SCL + row * (RS // 4) + LW // 4, s)
    else:
        tl.store(VALS + row * RS + k, xb)


@triton.jit
def row_scales(SCL, rows, ok, LW: tl.constexpr, RS: tl.constexpr):
    """The scales of cache rows ``rows`` (int64 [N]; 1 where not ok)."""
    return tl.load(SCL + rows * (RS // 4) + LW // 4, mask=ok, other=1.0)
