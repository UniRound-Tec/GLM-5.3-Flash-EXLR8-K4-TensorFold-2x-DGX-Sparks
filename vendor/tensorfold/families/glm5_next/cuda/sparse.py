"""Sparse DSA keeps lower-index pools on score ties and visits selected tokens and chunks in position order so window rows preserve serial bits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda.geometry import MLA_SELECT_ROWS as SELECT_ROWS   # a prompt chunk's rows scored at once

from . import kv8
from .kv8 import row_scales

POOL = 4
TOPK_POOLS = 512
BR = 16
# the first position whose row select_tokens counts (npool > TOPK_POOLS): rows before it keep dense attention
SPARSE_FROM = (TOPK_POOLS + 1) * POOL - 1
TOKENS = TOPK_POOLS * POOL + POOL - 1      # a sparse row's attended tokens: its 512 pools', then its incomplete pool's


@triton.jit
def _index_write(KR, k_stride, GR, LNW, LNB, IK, IG, POS, RING, eps, D: tl.constexpr):
    """Row r: LayerNorm(k_raw) -> bf16 into IK[(pos + r) % RING]; the gate row (fp32) -> bf16 into IG alike."""

    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(KR + r * k_stride + d).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / D
    y = xc / tl.sqrt(var + eps) * tl.load(LNW + d).to(tl.float32) + tl.load(LNB + d).to(tl.float32)
    slot = (P + r) % RING
    tl.store(IK + slot * D + d, y.to(tl.bfloat16))
    tl.store(IG + slot * D + d, tl.load(GR + r * D + d).to(tl.bfloat16))


@triton.jit
def _pool_keys(IK, IG, APE, PK, PKS, POS, R, RING, D: tl.constexpr, RS: tl.constexpr, FP8: tl.constexpr):
    """Program i: pool p = pos // 4 + i if it is complete within the window (ends at or before pos + R - 1), from its
    four tokens' keys and gates in the ring (token t at t % RING); a pool's bits depend on those rows only, whichever
    window completes it. FP8 (TF_GLM_KV=fp8): the pooled key stored as kv8.store_row's codes and scale."""

    i = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    p = P // 4 + i
    if 4 * p + 3 > P + R - 1:
        return
    d = tl.arange(0, D)
    s0 = (4 * p + 0) % RING
    s1 = (4 * p + 1) % RING
    s2 = (4 * p + 2) % RING
    s3 = (4 * p + 3) % RING
    l0 = tl.load(IG + s0 * D + d).to(tl.float32) + tl.load(APE + 0 * D + d).to(tl.float32)
    l1 = tl.load(IG + s1 * D + d).to(tl.float32) + tl.load(APE + 1 * D + d).to(tl.float32)
    l2 = tl.load(IG + s2 * D + d).to(tl.float32) + tl.load(APE + 2 * D + d).to(tl.float32)
    l3 = tl.load(IG + s3 * D + d).to(tl.float32) + tl.load(APE + 3 * D + d).to(tl.float32)
    m = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    e2 = tl.exp(l2 - m)
    e3 = tl.exp(l3 - m)
    s = ((e0 + e1) + e2) + e3
    k0 = tl.load(IK + s0 * D + d).to(tl.float32)
    k1 = tl.load(IK + s1 * D + d).to(tl.float32)
    k2 = tl.load(IK + s2 * D + d).to(tl.float32)
    k3 = tl.load(IK + s3 * D + d).to(tl.float32)
    t0 = ((e0 / s).to(tl.bfloat16).to(tl.float32) * k0).to(tl.bfloat16).to(tl.float32)
    t1 = ((e1 / s).to(tl.bfloat16).to(tl.float32) * k1).to(tl.bfloat16).to(tl.float32)
    t2 = ((e2 / s).to(tl.bfloat16).to(tl.float32) * k2).to(tl.bfloat16).to(tl.float32)
    t3 = ((e3 / s).to(tl.bfloat16).to(tl.float32) * k3).to(tl.bfloat16).to(tl.float32)
    kv8.store_row(PK, PKS, p, ((t0 + t1) + t2) + t3, D, RS, FP8)


def index_update(k_raw: torch.Tensor, gate: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, ape: torch.Tensor,
                 ik: torch.Tensor, ig: torch.Tensor, pk: torch.Tensor, pos: torch.Tensor) -> None:
    """Window rows' index keys and gates into their ring (ik, ig: bf16 [ring, 128], token t at row t % ring), then
    every pool the window completes into pk (bf16, or TF_GLM_KV=fp8's kv8 rows). Only the pool a token belongs to
    ever reads its key and gate, so the ring needs the window's rows and the 3 before it: ring >= R + 3
    (forward.index_ring), or at least the context (no row wraps: the full-length caches of before)."""

    R = k_raw.shape[0]
    ring = ik.shape[0]
    if ik.dtype != torch.bfloat16 or ig.shape != ik.shape or not ik.is_contiguous() or not ig.is_contiguous():
        raise ValueError("index_update: bf16 key and gate rings of one shape, contiguous")
    _index_write[(R,)](k_raw, k_raw.stride(0), gate, ln_w, ln_b, ik, ig, pos, ring, 1e-6, D=128, num_warps=1)
    if kv8.width(pk) != 128:
        raise ValueError("index_update: 128-wide pooled keys")
    pkv, pks, rs, fp8 = kv8.parts(pk)
    _pool_keys[(R // 4 + 2,)](ik, ig, ape, pkv, pks, pos, R, ring, D=128, RS=rs, FP8=fp8, num_warps=1)


@triton.jit
def _scores(QI, W, w_stride, PK, PKS, OUT, POS, R, NP, scale, wscale, H: tl.constexpr, HP: tl.constexpr,
            D: tl.constexpr, BP: tl.constexpr, L: tl.constexpr, RS: tl.constexpr, FP8: tl.constexpr):
    """Program (row, L pool blocks): s_p = sum_h w_h relu(scale * qi_h . pool_p) up to the row's position, heads
    padded to HP. FP8 (TF_GLM_KV=fp8): the pooled keys' e4m3 codes on the tensor cores, each pool's power-of-two scale
    on its dots. One row and L blocks a program, the row's queries and head weights loaded once for them (rows a
    program and one block, TF_GLM_SCORE_RB, reloaded a row's queries for every block); each block computes exactly as
    a 1-row program did, the same bits (tools/scores_loop_check.py)."""

    r = tl.program_id(0)
    g = tl.program_id(1)
    P = tl.load(POS)
    d = tl.arange(0, D)
    hh = tl.arange(0, HP)
    hok = hh < H
    npool = (P + r + 1) // 4
    q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None], other=0.0).to(tl.bfloat16)
    w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
    for j in range(0, L):
        pb = g * L + j
        p = pb * BP + tl.arange(0, BP)
        if pb * BP < NP:
            # blocks past the row's last visible complete pool store -inf without dot products (speed only: their
            # dots would be masked to -inf below)
            if pb * BP >= npool:
                tl.store(OUT + r * NP + p, float("-inf"), mask=p < NP)
            else:
                kok = p < npool
                k = tl.load(PK + p[:, None].to(tl.int64) * RS + d[None, :], mask=kok[:, None], other=0.0).to(tl.bfloat16)
                dots = tl.dot(q, tl.trans(k))                     # [HP, BP] fp32; keys past npool loaded as 0
                if FP8:
                    ks = row_scales(PKS, p.to(tl.int64), kok, D, RS)
                    dots = dots * ks[None, :]
                sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
                sc = tl.where(p < npool, sc, float("-inf"))
                tl.store(OUT + r * NP + p, sc, mask=p < NP)


SCORE_LOOP = 32            # pool blocks a _scores program takes (speed only; GB10: 32 >= 16 > 8 > 4)


@triton.jit
def _scores_rows(QI, W, w_stride, PK, PKS, OUT, o_stride, POS, SPR, BASE, R, NP, scale, wscale, H: tl.constexpr,
                 HP: tl.constexpr, D: tl.constexpr, BP: tl.constexpr, G: tl.constexpr, SEG: tl.constexpr,
                 RS: tl.constexpr, FP8: tl.constexpr, K: tl.constexpr):
    """Decode rows, program (row, g): the row's index queries and head weights loaded once, then pool blocks g,
    g + G, .. each through _scores' arithmetic for that row (the same key mask, dot tile, warps and head sum, so
    RB = 1's bits). SEG False: rows from one position (POS[0] + r), the blocks holding a row's first
    _visible_bound columns (pools past the row's -inf; later columns of the NP-column bucket are not written: the
    selection reads the same bound); SEG: a segmented window's row r (skipped when dense) over its own visible
    pools, keys at its stream's base / 4. FP8 (TF_GLM_KV=fp8): pooled keys as kv8 rows RS bytes apart, each pool's
    scale on its dots, as in _scores."""

    r = tl.program_id(0)
    g = tl.program_id(1)
    if SEG:
        if tl.load(SPR + r) == 0:
            return
        npool = (tl.load(POS + r) + 1) // 4
        pbase = tl.load(BASE + r).to(tl.int64) // 4
        nb = (npool + BP - 1) // BP
    else:
        npool = (tl.load(POS) + r + 1) // 4
        pbase = tl.zeros((), tl.int64)
        nb = (_visible_bound(NP, tl.load(POS), r, K) + BP - 1) // BP
    if g >= nb:
        return
    d = tl.arange(0, D)
    hh = tl.arange(0, HP)
    hok = hh < H
    q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None], other=0.0).to(tl.bfloat16)
    w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
    for pb in tl.range(g, nb, G, num_stages=1):          # Triton's pipelining of this loop: slower (measured)
        p = pb * BP + tl.arange(0, BP)
        rows = pbase + p
        k = tl.load(PK + rows[:, None] * RS + d[None, :], mask=(p < npool)[:, None], other=0.0).to(tl.bfloat16)
        kr = tl.where((p < npool)[:, None], k, 0.0)
        dots = tl.dot(q, tl.trans(kr))                                                # [HP, BP] fp32
        if FP8:
            dots = dots * row_scales(PKS, rows, p < npool, D, RS)[None, :]
        sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
        if SEG:
            tl.store(OUT + r * o_stride + p, sc, mask=p < npool)
        else:
            sc = tl.where(p < npool, sc, float("-inf"))
            tl.store(OUT + r * o_stride + p, sc, mask=p < NP)


# decode rows' scoring (_scores_rows): the tile (BP pools, 4 warps) is _scores', which sets the bits (8 warps or
# 32-pool blocks give others); G, the programs a row, is speed only
ROW_SCORE_G = 128


def _top_pools(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Reference top-k: each row's k best pools, ties to the lower pool, ascending, as a stable descending sort keeps them (unique int64 keys)."""
    bits = (scores + 0.0).view(torch.int32)                              # + 0.0: -0 becomes +0, as the sort ties them
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)          # IEEE order as signed ints (negatives flipped)
    keys = ordered.to(torch.int64).bitwise_left_shift_(32)
    keys.bitwise_or_(0xFFFFFFFF - torch.arange(scores.shape[1], device=scores.device, dtype=torch.int64))
    best = torch.topk(keys, k, dim=1, sorted=False).values
    del keys
    return torch.sort(0xFFFFFFFF - (best & 0xFFFFFFFF), dim=1).values


@triton.jit
def _order_key(s):
    """A float32 score as a uint32 whose unsigned order is the scores' order (-0 counted as +0)."""
    bits = (s + 0.0).to(tl.int32, bitcast=True)
    return (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32, bitcast=True)


@triton.jit
def _select_rows(S, OUT, NPS, POS, K: tl.constexpr, BLOCK: tl.constexpr, VIS: tl.constexpr):
    """Program r: a radix select (8 bits a pass) finds the K-th best score, then one pass in pool order writes the pools above it and the lowest ties.
    Rows NPS scores apart; VIS: the row's first visible_bound() columns only (row r at position POS[0] + r)."""

    r = tl.program_id(0).to(tl.int64)
    row = S + r * NPS
    if VIS:
        NP = _visible_bound(NPS, tl.load(POS), r, K)
    else:
        NP = NPS
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = K
    for p in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for c in range(0, NP, BLOCK):
            i = c + tl.arange(0, BLOCK)
            ok = i < NP
            u = _order_key(tl.load(row + i, mask=ok, other=0.0))
            match = ok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (24 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist           # pools with this digit or a higher one
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    for c in range(0, NP, BLOCK):
        i = c + tl.arange(0, BLOCK)
        ok = i < NP
        u = _order_key(tl.load(row + i, mask=ok, other=0.0))
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        tl.store(OUT + r * K + written + tl.cumsum(t, 0) - t, i.to(tl.int64), mask=take)
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)


@triton.jit
def _visible_bound(NPS, P, r, K: tl.constexpr):
    """Columns a decode row at position P + r needs: its complete pools, at least K, at most the NPS scored.
    The scores past its complete pools are -inf with the highest indices, so they rank below every column kept
    (ties to the lower pool): the K best of the first bound columns are the K best of all NPS. A row with fewer
    than K complete pools takes columns 0 .. K - 1 either way (its pools, then the lowest -inf ones)."""
    return tl.minimum(NPS, tl.maximum((P + r + 1) // 4, K))


def top_pools(scores: torch.Tensor, k: int, pos: torch.Tensor | None = None) -> torch.Tensor:
    """``_top_pools``'s pools, ascending, without int64 keys, top-k or sort: one program a row, or for a decode
    window's long rows the split selection (select_split); the same pools either way. ``pos`` (device [1]): the
    rows are decode rows at pos, pos + 1, ..; each is selected over its visible columns only (_visible_bound),
    and the columns past them need not be written."""
    R, NP = scores.shape
    if NP < k or not scores.is_contiguous():
        if pos is not None:
            raise ValueError("top_pools: visible bounds need contiguous rows of at least k scores")
        return _top_pools(scores, k)
    if R < SPLIT_ROWS and NP >= SPLIT_FROM:     # decode rows at long context: each row across chunk programs
        return select_split(scores, k, pos=pos)
    # one program a row (speed only): decode rows past 1,024 scores 4,096 a step, prompt chunks' rows 1,024
    block, warps = (4096, 8) if R < SPLIT_ROWS and NP > 1024 else (1024, 4)
    out = torch.empty((R, k), dtype=torch.int64, device=scores.device)
    _select_rows[(R,)](scores, out, NP, scores if pos is None else pos, K=k, BLOCK=block, VIS=pos is not None,
                       num_warps=warps)
    return out


def pool_bucket(pos: int, R: int, np_max: int) -> int:
    """Pools to score for rows pos .. pos + R - 1: the visible ones rounded up to a power of two (at least 1024), at most the capacity's."""
    visible = (pos + R) // POOL + 1
    return min(np_max, max(1024, 1 << (visible - 1).bit_length()))


PROMPT_ROWS = 64          # windows from this many rows (prompt chunks, eager) take the prompt selection below


def pool_count(pos: int, R: int, np_max: int) -> int:
    """Pools a prompt chunk scores: the visible ones rounded up to a multiple of 1,024 (at most ``pool_bucket``)."""
    visible = (pos + R) // POOL + 1
    return min(np_max, -(-visible // 1024) * 1024)


@triton.jit
def _tokens(POOLS, POS, TOK, CNT, W: tl.constexpr, K: tl.constexpr, PL: tl.constexpr, BLOCK: tl.constexpr):
    """Program r: TOK[r] = the K pools' tokens ascending (pool * PL + j), then the incomplete last pool's visible
    tokens (-1 past row r's position); CNT[r] = their count when the row has more than K complete pools, else 0."""
    r = tl.program_id(0).to(tl.int64)
    q = tl.load(POS).to(tl.int64) + r
    npool = (q + 1) // PL
    for c in tl.static_range(0, K * PL, BLOCK):
        i = c + tl.arange(0, BLOCK)
        p = tl.load(POOLS + r * K + i // PL)
        tl.store(TOK + r * W + i, (p * PL + i % PL).to(tl.int32))
    j = tl.arange(0, 4)
    jok = j < PL - 1
    tail = npool * PL + j
    ok = jok & (tail <= q)
    tl.store(TOK + r * W + K * PL + j, tl.where(ok, tail, -1).to(tl.int32), mask=jok)
    n = K * PL + tl.sum(ok.to(tl.int64), 0)
    tl.store(CNT + r, tl.where(npool > K, n, 0).to(tl.int32))


def select_tokens(qi: torch.Tensor, wts: torch.Tensor, pk: torch.Tensor, pos: int | None, R: int, np_max: int,
                  pos_dev: torch.Tensor, *, bucket: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's attended tokens [R, 2051] ascending (-1 padded) and their count past the dense limit; ``bucket`` fixes the pool count for graphs."""

    if qi.stride(0) != qi.shape[1] or wts.stride(1) != 1:
        raise ValueError("select_tokens: index queries must be contiguous rows, weights unit-stride columns")
    if bucket is None and pos is not None and R >= PROMPT_ROWS:
        return _select_prompt(qi, wts, pk, pos, R, np_max, pos_dev)
    # score only visible pools, rounded up to a power of two so the allocator reuses a few sizes (exact sizes fragmented memory at 128k)
    np_max = bucket if bucket is not None else pool_bucket(pos, R, np_max)
    scores = torch.empty((R, np_max), dtype=torch.float32, device=qi.device)
    # heads and width from the tensors: fixed ones read past a row's index query into its window neighbours
    H = wts.shape[1]
    D = qi.shape[1] // H
    wscale = 1.0 / 5.656854249492381 if H == 32 else H ** -0.5            # 32 ** -0.5 exactly as before
    grid = min(ROW_SCORE_G, triton.cdiv(np_max, 64))
    pkv, pks, rs, fp8 = kv8.parts(pk)
    _scores_rows[(R, grid)](qi, wts, wts.stride(0), pkv, pks, scores, np_max, pos_dev, pos_dev, pos_dev, R, np_max,
                            D ** -0.5, wscale, H=H, HP=max(16, triton.next_power_of_2(H)), D=D, BP=64, G=grid,
                            SEG=False, RS=rs, FP8=fp8, K=TOPK_POOLS, num_warps=4)
    # the graphs' power-of-two bucket past the rows' visible pools: neither scored nor scanned (_visible_bound); the
    # selection writes the tokens and counts itself (split), or _tokens from its pools: select_tokens' formulas
    # (each row's pools' tokens ascending, then its incomplete pool's visible tokens; rows within the dense limit
    # count 0)
    width = TOPK_POOLS * POOL + POOL - 1
    tokens = torch.empty((R, width), dtype=torch.int32, device=qi.device)
    counts = torch.empty((R,), dtype=torch.int32, device=qi.device)
    if R < SPLIT_ROWS and np_max >= SPLIT_FROM:
        select_split(scores, TOPK_POOLS, pos=pos_dev, tokens=tokens, counts=counts)
    else:
        pools = top_pools(scores, TOPK_POOLS, pos=pos_dev)                              # ascending pool index
        _tokens[(R,)](pools, pos_dev, tokens, counts, W=width, K=TOPK_POOLS, PL=POOL, BLOCK=1024, num_warps=4)
    return tokens, counts


def _select_prompt(qi: torch.Tensor, wts: torch.Tensor, pk: torch.Tensor, pos: int, R: int, np_max: int,
                   pos_dev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """select_tokens for a prompt chunk, with select_tokens' counts and, for every counted row, its tokens:

    - it scores NP = pool_count pools, not pool_bucket's power of two: the columns left out lie past every row's
      visible pools, where _scores writes -inf, and they have the highest indices, so they rank below every column
      kept (-inf ties go to the lower pool; a negative NaN, the one key below -inf's, is never an arithmetic result
      on the GPU). A counted row has more than 512 visible pools and NP >= 1024 >= 512, so its best 512 pools, the
      radix select's threshold and ties are the same. (Rows with count 0, whose tokens nothing reads, may differ.)
      The scores buffer keeps pool_bucket's size (the allocator's few sizes), used as a contiguous [R, NP] prefix;
    - _scores takes one row a program and SCORE_LOOP pool blocks, a row's keys masked to its own pools before its
      dot;
    - one kernel (_tokens) writes tokens and counts with select_tokens' integer formulas."""

    dev = qi.device
    NP = pool_count(pos, R, np_max)
    # rows go through in blocks of SELECT_ROWS: every block scores the chunk's NP pools (the same columns, so the same
    # bits a row), and the fp32 scores take SELECT_ROWS rows, not the chunk's, of the capacity-sized memory
    B = min(R, SELECT_ROWS)
    buf = torch.empty((B * pool_bucket(pos, R, np_max),), dtype=torch.float32, device=dev)
    H = wts.shape[1]
    D = qi.shape[1] // H
    wscale = 1.0 / 5.656854249492381 if H == 32 else H ** -0.5            # 32 ** -0.5 exactly as before
    width = TOPK_POOLS * POOL + POOL - 1
    tokens = torch.empty((R, width), dtype=torch.int32, device=dev)
    counts = torch.empty((R,), dtype=torch.int32, device=dev)
    pkv, pks, rs, fp8 = kv8.parts(pk)
    for a in range(0, R, B):
        n = min(B, R - a)
        at = pos_dev if a == 0 else pos_dev + a
        scores = buf[:n * NP].view(n, NP)
        _scores[(n, triton.cdiv(triton.cdiv(NP, 64), SCORE_LOOP))](qi[a:a + n], wts[a:a + n], wts.stride(0), pkv, pks,
                                                                   scores, at, n, NP, D ** -0.5, wscale, H=H,
                                                                   HP=max(16, triton.next_power_of_2(H)), D=D, BP=64,
                                                                   L=SCORE_LOOP, RS=rs, FP8=fp8, num_warps=4)
        pools = top_pools(scores, TOPK_POOLS)                                           # ascending pool index
        _tokens[(n,)](pools, at, tokens[a:a + n], counts[a:a + n], W=width, K=TOPK_POOLS, PL=POOL, BLOCK=1024,
                      num_warps=4)
    return tokens, counts


@triton.jit
def _gtile(q, k, v, m, l, o, valid, SCALE: tl.constexpr):
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _sparse_chunks(Q, KC, VC, TOK, CNT, PO, PM, PL, W: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   CH: tl.constexpr, SCALE: tl.constexpr):
    """Attend selected tokens in list order with the query in row 0 of a 16-row tile and the other tile rows idle."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
    d = tl.arange(0, D)
    g = tl.arange(0, 16)
    q = tl.load(Q + (r * H + h) * D + d[None, :] + g[:, None] * 0, mask=(g == 0)[:, None], other=0).to(tl.bfloat16)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for t in range(CH // 64):
        idx = c * CH + t * 64 + tl.arange(0, 64)
        ok = idx < n
        tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
        kk = tl.load(KC + (tok[:, None] * H + h) * D + d[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
        vv = tl.load(VC + (tok[:, None] * H + h) * D + d[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
        m, l, o = _gtile(q, kk, vv, m, l, o, ok, SCALE)
    base = (c * 128 + r) * H + h
    tl.store(PO + base * D + d[None, :] + g[:, None] * 0, o, mask=(g == 0)[:, None])
    tl.store(PM + base + g * 0, m, mask=g == 0)
    tl.store(PL + base + g * 0, l, mask=g == 0)


@triton.jit
def _sparse_merge(PO, PM, PL, OUT, CNT, H: tl.constexpr, D: tl.constexpr, NCH: tl.constexpr, CH: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.load(CNT + r)
    if n == 0:
        return
    d = tl.arange(0, D)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((D,), tl.float32)
    for c in range(NCH):
        if c * CH < n:
            base = (c * 128 + r) * H + h
            cm = tl.load(PM + base)
            cl = tl.load(PL + base)
            co = tl.load(PO + base * D + d)
            active = cl > 0.0
            next_m = tl.where(active, tl.maximum(m, cm), m)
            a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
            b = tl.where(active, tl.exp(cm - next_m), 0.0)
            o = o * a + co * b
            l = l * a + cl * b
            m = next_m
    tl.store(OUT + (r * H + h) * D + d, (o / l).to(tl.bfloat16))


PART_ROWS = 128          # rows of one launch: the kernels keep a row's chunk partials at c * 128 + r


def sparse_attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
                     out: torch.Tensor, scale: float) -> None:
    """Write attention for rows with positive counts into out [R, H, D] in launches of up to 128 rows, leaving other rows untouched."""

    R, H, D = q.shape
    W = tokens.shape[1]
    CH = 512
    nch = triton.cdiv(W, CH)
    po = torch.empty((nch * PART_ROWS * H * D,), dtype=torch.float32, device=q.device)
    pm = torch.empty((nch * PART_ROWS * H,), dtype=torch.float32, device=q.device)
    pl = torch.empty((nch * PART_ROWS * H,), dtype=torch.float32, device=q.device)
    for r0 in range(0, R, PART_ROWS):
        n = min(PART_ROWS, R - r0)
        rows = slice(r0, r0 + n)
        _sparse_chunks[(n, H, nch)](q[rows], kc, vc, tokens[rows], counts[rows], po, pm, pl, W=W, H=H, D=D, CH=CH,
                                    SCALE=scale, num_warps=4, num_stages=1)
        _sparse_merge[(n, H)](po, pm, pl, out[rows], counts[rows], H=H, D=D, NCH=nch, CH=CH, num_warps=4)


# ------------------------------------------------------------------------------------- multi-stream windows ---
# Segmented windows (latent.py's section of the same name): each stream's index keys and gates in its own part of
# shared [Q, 128] tensors (token t of the stream at ibase + t % ring: its slot's ring, or its extent without rings)
# and its pooled keys at base / 4 + p of [P / 4 + 2, 128]; per-row device tables (``segments.SegRows``) give each row
# its position, bases and kind; pooled keys bf16 or TF_GLM_KV=fp8's kv8 rows (codes, then scales in PKS, rows RS
# elements apart). A row keeps its single-stream arithmetic. Reads and writes of the index caches go through the
# helpers below.

@triton.jit
def _ix_put(IX, row, d, x, D: tl.constexpr):
    """Index key or gate row ``row`` (int64) <- x [D] as bf16."""
    tl.store(IX + row * D + d, x.to(tl.bfloat16))


@triton.jit
def _ix_get(IX, row, d, D: tl.constexpr):
    """Index key or gate row ``row`` (int64) as fp32 [D]."""
    return tl.load(IX + row * D + d).to(tl.float32)


@triton.jit
def _pk_put(PK, PKS, row, x, D: tl.constexpr, RS: tl.constexpr, FP8: tl.constexpr):
    """Pooled key row ``row`` (int64) <- x [D] (fp32) as bf16, or FP8 (kv8.store_row, as _pool_keys stores it)."""
    kv8.store_row(PK, PKS, row, x, D, RS, FP8)


@triton.jit
def _pk_get(PK, PKS, rows, ok, d, D: tl.constexpr, RS: tl.constexpr, FP8: tl.constexpr):
    """Pooled key rows ``rows`` (int64 [n]) as a bf16 tile [n, D] (FP8: the codes, exact) and their scales [n]
    (FP8; ones for bf16, unread); rows with ok false are zeros."""
    k = tl.load(PK + rows[:, None] * RS + d[None, :], mask=ok[:, None], other=0.0).to(tl.bfloat16)
    if FP8:
        return k, row_scales(PKS, rows, ok, D, RS)
    return k, tl.full(rows.shape, 1.0, tl.float32)


@triton.jit
def _seg_index_write(KR, k_stride, GR, LNW, LNB, IK, IG, POS, IBASE, RING, eps, D: tl.constexpr):
    """_index_write for row r at its stream's index row ibase[r] + pos[r] % RING."""

    r = tl.program_id(0)
    row = tl.load(IBASE + r).to(tl.int64) + tl.load(POS + r).to(tl.int64) % RING
    d = tl.arange(0, D)
    x = tl.load(KR + r * k_stride + d).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / D
    y = xc / tl.sqrt(var + eps) * tl.load(LNW + d).to(tl.float32) + tl.load(LNB + d).to(tl.float32)
    _ix_put(IK, row, d, y, D)
    _ix_put(IG, row, d, tl.load(GR + r * D + d), D)


@triton.jit
def _seg_pool_keys(IK, IG, APE, PK, PKS, POS, BASE, IBASE, RING, D: tl.constexpr, RS: tl.constexpr,
                   FP8: tl.constexpr):
    """Program r: the pool row r completes (pos[r] % 4 == 3), pool p = pos[r] // 4 of its stream, from tokens
    4p .. 4p + 3 (index rows ibase + t % RING) into pooled row base / 4 + p: _pool_keys' arithmetic (the pools a
    single-stream window completes are exactly those its rows end)."""

    r = tl.program_id(0)
    q = tl.load(POS + r).to(tl.int64)
    if q % 4 != 3:
        return
    B = tl.load(BASE + r).to(tl.int64)
    IB = tl.load(IBASE + r).to(tl.int64)
    t = (q // 4) * 4
    s0 = IB + (t + 0) % RING
    s1 = IB + (t + 1) % RING
    s2 = IB + (t + 2) % RING
    s3 = IB + (t + 3) % RING
    d = tl.arange(0, D)
    l0 = _ix_get(IG, s0, d, D) + tl.load(APE + 0 * D + d).to(tl.float32)
    l1 = _ix_get(IG, s1, d, D) + tl.load(APE + 1 * D + d).to(tl.float32)
    l2 = _ix_get(IG, s2, d, D) + tl.load(APE + 2 * D + d).to(tl.float32)
    l3 = _ix_get(IG, s3, d, D) + tl.load(APE + 3 * D + d).to(tl.float32)
    m = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    e2 = tl.exp(l2 - m)
    e3 = tl.exp(l3 - m)
    s = ((e0 + e1) + e2) + e3
    k0 = _ix_get(IK, s0, d, D)
    k1 = _ix_get(IK, s1, d, D)
    k2 = _ix_get(IK, s2, d, D)
    k3 = _ix_get(IK, s3, d, D)
    t0 = ((e0 / s).to(tl.bfloat16).to(tl.float32) * k0).to(tl.bfloat16).to(tl.float32)
    t1 = ((e1 / s).to(tl.bfloat16).to(tl.float32) * k1).to(tl.bfloat16).to(tl.float32)
    t2 = ((e2 / s).to(tl.bfloat16).to(tl.float32) * k2).to(tl.bfloat16).to(tl.float32)
    t3 = ((e3 / s).to(tl.bfloat16).to(tl.float32) * k3).to(tl.bfloat16).to(tl.float32)
    _pk_put(PK, PKS, B // 4 + q // 4, ((t0 + t1) + t2) + t3, D, RS, FP8)


def seg_index_update(k_raw: torch.Tensor, gate: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor,
                     ape: torch.Tensor, ik: torch.Tensor, ig: torch.Tensor, pk: torch.Tensor, rows) -> None:
    """index_update for a segmented window: each row's index key and gate at its stream's ibase + pos % ring (its
    slot's ring, or base + pos without rings: ``rows``: segments.SegRows), then every pool a row completes."""

    R = k_raw.shape[0]
    _seg_index_write[(R,)](k_raw, k_raw.stride(0), gate, ln_w, ln_b, ik, ig, rows.pos, rows.ibase, rows.ring_rows,
                           1e-6, D=128, num_warps=1)
    pkv, pks, rs, fp8 = kv8.parts(pk)
    _seg_pool_keys[(R,)](ik, ig, ape, pkv, pks, rows.pos, rows.base, rows.ibase, rows.ring_rows, D=128, RS=rs,
                         FP8=fp8, num_warps=1)


@triton.jit
def _seg_scores(QI, W, w_stride, PK, PKS, OUT, o_stride, POS, SPR, SEG_START, SEG_ROWS, SEG_PBASE, SEG_POOLS, scale,
                wscale, H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, BP: tl.constexpr, G: tl.constexpr,
                RS: tl.constexpr, FP8: tl.constexpr):
    """Program (segment, g): pool blocks g, g + G, .. below the segment's pool count (on the device: the pools its
    last row sees), each block's keys (at the stream's pooled base) loaded once for all the segment's sparse rows;
    a row's scores are _scores' (its keys masked to its own pools before the dot, as with RB > 1), written for its
    visible pools only (the ones the selection reads)."""

    s = tl.program_id(0)
    g = tl.program_id(1)
    npm = tl.load(SEG_POOLS + s)
    nr = tl.load(SEG_ROWS + s)
    r0 = tl.load(SEG_START + s)
    pbase = tl.load(SEG_PBASE + s).to(tl.int64)
    d = tl.arange(0, D)
    hh = tl.arange(0, HP)
    hok = hh < H
    for pb in range(g, (npm + BP - 1) // BP, G):
        p = pb * BP + tl.arange(0, BP)
        k, ks = _pk_get(PK, PKS, pbase + p, p < npm, d, D, RS, FP8)                    # [BP, D], [BP]
        for i in range(nr):
            r = r0 + i
            if tl.load(SPR + r) != 0:
                npool = (tl.load(POS + r) + 1) // 4
                q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None], other=0.0).to(tl.bfloat16)
                kr = tl.where((p < npool)[:, None], k, 0.0)
                dots = tl.dot(q, tl.trans(kr))                                            # [HP, BP] fp32
                if FP8:
                    dots = dots * ks[None, :]
                w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
                sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
                tl.store(OUT + r * o_stride + p, sc, mask=p < npool)


@triton.jit
def _seg_select(S, s_stride, POS, SPR, TOK, CNT, W: tl.constexpr, K: tl.constexpr, PL: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program r (a sparse row): _select_rows over the row's visible pools, its K pools' tokens written as they are
    taken (ascending), then _tokens' tail and count; a dense row gets count 0. Every single-stream path scores more
    columns, but those lie past the row's visible pools, score -inf and have the highest indices: they rank below
    every visible pool (ties go to the lower pool), so a row with more than K visible pools selects the same ones
    (sparse._select_prompt)."""

    r = tl.program_id(0).to(tl.int64)
    if tl.load(SPR + r) == 0:
        tl.store(CNT + r, 0)
        return
    qpos = tl.load(POS + r).to(tl.int64)
    NP = (qpos + 1) // PL
    row = S + r * s_stride
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = K
    for p in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for c in range(0, NP, BLOCK):
            i = c + tl.arange(0, BLOCK)
            ok = i < NP
            u = _order_key(tl.load(row + i, mask=ok, other=0.0))
            match = ok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (24 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    j = tl.arange(0, PL)
    for c in range(0, NP, BLOCK):
        i = c + tl.arange(0, BLOCK)
        ok = i < NP
        u = _order_key(tl.load(row + i, mask=ok, other=0.0))
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        slot = (written + tl.cumsum(t, 0) - t).to(tl.int64)
        tl.store(TOK + r * W + slot[:, None] * PL + j[None, :], (i.to(tl.int64)[:, None] * PL + j[None, :]).to(tl.int32),
                 mask=take[:, None])
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)
    jt = tl.arange(0, 4)
    jok = jt < PL - 1
    tail = NP * PL + jt
    tok = jok & (tail <= qpos)
    tl.store(TOK + r * W + K * PL + jt, tl.where(tok, tail, -1).to(tl.int32), mask=jok)
    n = K * PL + tl.sum(tok.to(tl.int64), 0)
    tl.store(CNT + r, tl.where(NP > K, n, 0).to(tl.int32))


SEG_SCORE_GRID = 256       # _seg_scores programs a segment (grid stride over its pool blocks): speed only
SEG_SPLIT = True           # seg_select_tokens: the split selection (select_split), else one program a row (_seg_select)
SEG_SELECT = (4096, 8)     # _seg_select's scores a step and warps: speed only (the selection is exact integer logic)


def seg_select_tokens(qi: torch.Tensor, wts: torch.Tensor, pk: torch.Tensor, rows, scratch) -> tuple[torch.Tensor,
                                                                                                          torch.Tensor]:
    """select_tokens for a segmented window: each sparse row's attended tokens [R, 2051] (relative to its stream,
    ascending, -1 padded) and counts (0 for dense rows). Scores every segment's pools up to its device-side pool
    count (_seg_scores: a pool block's keys loaded once for all the segment's rows; _scores_rows' row-major loop
    measured no faster here, and slower on dense windows), so one CUDA graph per window size serves every context
    length (``scratch``: segments.SelectScratch)."""

    if qi.stride(0) != qi.shape[1] or wts.stride(1) != 1:
        raise ValueError("seg_select_tokens: index queries must be contiguous rows, weights unit-stride columns")
    R = qi.shape[0]
    if R > scratch.rows:
        raise ValueError(f"seg_select_tokens: {R} rows, the scratch holds {scratch.rows}")
    H = wts.shape[1]
    D = qi.shape[1] // H
    wscale = 1.0 / 5.656854249492381 if H == 32 else H ** -0.5            # select_tokens' constants
    scores = scratch.scores
    grid = min(SEG_SCORE_GRID, triton.cdiv(scores.shape[1], 64))
    pkv, pks, rs, fp8 = kv8.parts(pk)
    _seg_scores[(rows.max_segs, grid)](qi, wts, wts.stride(0), pkv, pks, scores, scores.stride(0), rows.pos,
                                       rows.sparse, rows.seg_start, rows.seg_rows, rows.seg_pbase, rows.seg_pools,
                                       D ** -0.5, wscale, H=H, HP=max(16, triton.next_power_of_2(H)), D=D, BP=64,
                                       G=grid, RS=rs, FP8=fp8, num_warps=4)
    tokens, counts = scratch.tokens[:R], scratch.counts[:R]
    if SEG_SPLIT:
        select_split(scores[:R], TOPK_POOLS, scratch=scratch.split, rows=rows, tokens=tokens, counts=counts)
        return tokens, counts
    _seg_select[(R,)](scores, scores.stride(0), rows.pos, rows.sparse, tokens, counts, W=TOKENS, K=TOPK_POOLS,
                      PL=POOL, BLOCK=SEG_SELECT[0], num_warps=SEG_SELECT[1])
    return tokens, counts


# ----------------------------------------------------------------------------------- split top-k selection ---
# _select_rows' radix select with each row's scores split into CHS-score chunks, one program a (row, chunk), in five
# launches. Passes 0..3: every program takes the digits so far from the row's total histograms (TOT, integer
# atomic sums: exact in any order), keeps its chunk's count of scores above them (GT, first differing at an earlier
# byte) and its chunk's histogram of the next byte (HIST), adding it into the row's total. Step 4 writes the chunk's
# pools at the offset the earlier chunks' counts give (GT and their last histograms, which step 4 only reads):
# scores above the K-th best, then ties in pool order. The same K-th best, ties and ascending output as
# _select_rows: integers only, no floating-point sum order to keep.

@triton.jit
def _split_digits(TOT, r, STEPS: tl.constexpr, K: tl.constexpr):
    """(prefix, fixed, need, last digit) after the first STEPS passes, from the row's total histograms."""
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = tl.full((), K, tl.int32)
    digit = tl.full((), 0, tl.int32)
    for p in tl.static_range(STEPS):
        hist = tl.load(TOT + (r * 4 + p) * 256 + bins)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    return prefix, fixed, need, digit


@triton.jit
def _select_split(S, s_stride, NPC, POS, SPR, TOT, HIST, GT, OUT, TOK, CNT, W: tl.constexpr, K: tl.constexpr,
                  PL: tl.constexpr, CHS: tl.constexpr, CP: tl.constexpr, STEP: tl.constexpr, SEG: tl.constexpr,
                  VIS: tl.constexpr, TOKS: tl.constexpr, BLK: tl.constexpr):
    """Program (row, chunk), STEP 0..3: pass STEP over the chunk; STEP 4: its selected pools, as pool ids OUT[r, :K]
    (SEG False: NPC scores a row, or with VIS the first _visible_bound of them for row r at POS[0] + r; chunks past
    it exit; TOKS, with VIS: as select_tokens' tokens, tail and count instead, every row) or (SEG: the row's
    visible pools, dense rows skipped) as _seg_select's tokens, tail and count. TOT [R, 4, 256] zeroed before step 0;
    HIST [R, CP, 256], GT [R, CP] int32."""

    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    if SEG:
        if tl.load(SPR + r) == 0:
            if STEP == 4:
                if c == 0:
                    tl.store(CNT + r, 0)
            return
        qpos = tl.load(POS + r).to(tl.int64)
        NP = (qpos + 1) // PL
        npool = NP
    else:
        if VIS:
            qpos = tl.load(POS).to(tl.int64) + r
            NP = _visible_bound(NPC, tl.load(POS), r, K)
        else:
            qpos = r * 0
            NP = NPC
        npool = (qpos + 1) // PL
    nch = (NP + CHS - 1) // CHS
    if c >= nch:
        return
    bins = tl.arange(0, 256)
    i = c * CHS + tl.arange(0, CHS)
    ok = i < NP
    u = _order_key(tl.load(S + r * s_stride + i, mask=ok, other=0.0))
    prefix, fixed, need, d = _split_digits(TOT, r, STEP, K)
    own = HIST + (r * CP + c) * 256
    if STEP > 0 and STEP < 4:             # the chunk's scores above the prefix first at the byte just fixed
        above = tl.sum(tl.where(bins > d, tl.load(own + bins), 0), 0)
        if STEP > 1:
            above += tl.load(GT + r * CP + c)
        tl.store(GT + r * CP + c, above)
    if STEP < 4:
        match = ok & ((u & fixed) == prefix)
        hist = tl.histogram(((u >> (24 - 8 * STEP)) & 0xFF).to(tl.int32), 256, mask=match)
        tl.store(own + bins, hist)
        tl.atomic_add(TOT + (r * 4 + STEP) * 256 + bins, hist)
    else:
        # earlier chunks: scores above the prefix at bytes 0..2 (GT), at byte 3 and equal to it (their last histograms),
        # BLK chunks an iteration over the chunks before this one only: a CP x 256 tile masked to them costs every
        # program the whole table (CP is the scratch's, sized for the pool, not the row's context)
        gt = tl.zeros((), dtype=tl.int32)
        equal_seen = tl.zeros((), dtype=tl.int32)
        cb = tl.arange(0, BLK)
        for c0 in range(0, c, BLK):
            ci = c0 + cb
            before = ci < c
            last = tl.load(HIST + (r * CP + ci[:, None]) * 256 + bins[None, :], mask=before[:, None], other=0)
            gt += tl.sum(tl.load(GT + r * CP + ci, mask=before, other=0), 0)
            gt += tl.sum(tl.sum(tl.where(bins[None, :] > d, last, 0), 1), 0)
            equal_seen += tl.sum(tl.sum(tl.where(bins[None, :] == d, last, 0), 1), 0)
        written = gt + tl.minimum(equal_seen, need)
        e = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((e == 1) & (tl.cumsum(e, 0) - e + equal_seen < need))
        t = take.to(tl.int32)
        slot = (written + tl.cumsum(t, 0) - t).to(tl.int64)
        if SEG or TOKS:
            j = tl.arange(0, PL)
            tl.store(TOK + r * W + slot[:, None] * PL + j[None, :],
                     (i.to(tl.int64)[:, None] * PL + j[None, :]).to(tl.int32), mask=take[:, None])
            if c == 0:
                jt = tl.arange(0, 4)
                jok = jt < PL - 1
                tail = npool * PL + jt
                tok = jok & (tail <= qpos)
                tl.store(TOK + r * W + K * PL + jt, tl.where(tok, tail, -1).to(tl.int32), mask=jok)
                n = K * PL + tl.sum(tok.to(tl.int64), 0)
                tl.store(CNT + r, tl.where(npool > K, n, 0).to(tl.int32))
        else:
            tl.store(OUT + r * K + slot, i.to(tl.int64), mask=take)


SPLIT_FROM = 8192          # decode windows split rows of at least this many scores (fewer: one program a row)
SPLIT_ROWS = 64            # ... windows of fewer rows (prompt chunks' hundreds of rows fill the GPU as they are)
SPLIT_WARPS = 4
SPLIT_BLK = 16             # step 4's earlier chunks an iteration (speed only)


def split_chunk(np_: int) -> int:
    """Scores a chunk program takes (speed only): 2,048 up to 16,384 a row, else 4,096."""
    return 2048 if np_ <= 16384 else 4096


def split_chunks(np_: int) -> int:
    """Chunk programs a row of np_ scores takes (a power of two, the chunk tables' row stride)."""
    return triton.next_power_of_2(triton.cdiv(np_, split_chunk(np_)))


def split_scratch(rows: int, np_: int, device) -> torch.Tensor:
    """select_split's int32 scratch for rows of np_ scores: totals [rows, 4, 256], then chunk histograms and counts."""
    return torch.empty((rows * (1024 + split_chunks(np_) * 257),), dtype=torch.int32, device=device)


def select_split(scores: torch.Tensor, k: int, *, scratch: torch.Tensor | None = None, rows=None,
                 tokens: torch.Tensor | None = None, counts: torch.Tensor | None = None,
                 pos: torch.Tensor | None = None) -> torch.Tensor | None:
    """The split selection: top_pools' pools [R, k] (``rows`` None; ``pos`` as in top_pools; with ``pos``, tokens
    and counts: select_tokens' tokens and counts written into them instead, None returned), or with ``rows`` (segments.SegRows) and
    tokens / counts, seg_select_tokens' per-row output over each sparse row's visible pools. ``scratch``: at least
    split_scratch(R, NP)'s (allocated when None)."""

    R, NP = scores.shape
    chs, warps = split_chunk(NP), SPLIT_WARPS
    cp = split_chunks(NP)
    if scratch is None or scratch.numel() < R * (1024 + cp * 257):
        scratch = split_scratch(R, NP, scores.device)
    tot = scratch[:R * 1024]
    hist = scratch[R * 1024:R * (1024 + cp * 256)]
    gt = scratch[R * (1024 + cp * 256):R * (1024 + cp * 257)]
    tot.zero_()
    seg = rows is not None
    toks = not seg and tokens is not None
    if toks and pos is None:
        raise ValueError("select_split: tokens of rows without positions")
    out = None if seg or toks else torch.empty((R, k), dtype=torch.int64, device=scores.device)
    dummy = scratch
    for step in range(5):
        _select_split[(R, cp)](scores, scores.stride(0), NP, rows.pos if seg else (dummy if pos is None else pos),
                               rows.sparse if seg else dummy, tot, hist, gt, out if out is not None else dummy,
                               tokens if tokens is not None else dummy, counts if counts is not None else dummy,
                               W=TOKENS, K=k, PL=POOL, CHS=chs, CP=cp, STEP=step, SEG=seg,
                               VIS=pos is not None and not seg, TOKS=toks, BLK=SPLIT_BLK, num_warps=warps)
    return out
