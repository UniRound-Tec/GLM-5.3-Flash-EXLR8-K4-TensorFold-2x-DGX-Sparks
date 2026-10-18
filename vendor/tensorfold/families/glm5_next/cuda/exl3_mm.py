"""EXL3 expert kernels fix K splits and warps by shape so each routed pair's output is independent of the window's other rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path

import torch

from tensorfold.cuda import experts as grouped

# a rank's expert widths the decode kernel (``dec``) takes: two ranks' 1024, four ranks' 512, three ranks' 768 / 640
# (its down projection's K / 64 k steps a warp: 16, 8, 12, 10); others run the grouped kernel (the same bits)
DEC_WIDTHS = (1024, 768, 640, 512)
# Gate/up and down tile counts, warps, and K splits stay fixed across row counts to preserve each row's bits.
GATEUP_CFG = (8, 4, 4)
DOWN_CFG = (8, 4, 1)
# Prompt chunks launch an item's n blocks next to each other (TF_GLM_EXL3_NFIRST=0: the decode order), so they read
# its rows while L2 still holds them: the same kernel and arithmetic, 2.3-2.5x faster at 4,096 rows.
PROMPT_ROWS = 64          # rows from which a window takes the prompt order


def _pf() -> bool:
    """TF_GLM_EXL3_PF=0: no register prefetch in the expert kernel (the same bits either way)."""
    import os

    return os.environ.get("TF_GLM_EXL3_PF", "1") != "0"


def _nfirst() -> bool:
    import os

    return os.environ.get("TF_GLM_EXL3_NFIRST", "1") != "0"


def dec_settings() -> tuple[bool, int, bool]:
    """Decode windows' EXL3 path. TF_GLM_EXL3_DEC (default on): ``dec``, grouped_kernel's sums with the down
    projection's epilogue fused into it (one launch less a layer, no Z round trip); TF_GLM_EXL3_FUSE: 1 (default) the
    down epilogue, 2 gate/up's too (the last of an item's blocks runs it: measured slower), 0 neither;
    TF_GLM_EXL3_XROW (default on): a layer whose experts share one gate/up suh rotates each row's input once
    (rot_rows: the same fp16 inputs). Every setting gives the same bits; TF_GLM_EXL3_DEC=0: the grouped kernel and
    its epilogue kernels."""
    on = os.environ.get("TF_GLM_EXL3_DEC", "1") != "0"
    fuse = int(os.environ.get("TF_GLM_EXL3_FUSE", "1"))
    if fuse not in (0, 1, 2, 3):
        raise ValueError(f"TF_GLM_EXL3_FUSE is 0 to 3, not {fuse}")
    return on, fuse, os.environ.get("TF_GLM_EXL3_XROW", "1") != "0"


LOADS = {"0": 0, "ldg": 0, "1": 1, "nc": 1, "nc1": 1, "nc2": 2, "nc4": 3}


def dec_loads() -> int:
    """TF_GLM_EXL3_LOADS: the decode kernel's trellis loads (the same bits for every value). ``0`` / ``ldg``
    (default): 32-bit loads one k step ahead; ``nc`` / ``nc1``, ``nc2``, ``nc4``: 16-byte ld.global.nc loads 1, 2 or
    4 k steps ahead through a staging area, the item read with its count in one round trip and the first steps'
    words issued before the member rows arrive (after jayleaton/glm53-tensorfold-spark patches/0580, Apache-2.0)."""

    v = (os.environ.get("TF_GLM_EXL3_LOADS", "") or "0").strip().lower()
    if v not in LOADS:
        raise ValueError(f"TF_GLM_EXL3_LOADS: 0, nc, nc2 or nc4, not {v!r}")
    return LOADS[v]


def _seq() -> bool:
    """TF_GLM_EXL3_SEQ (default on): prompt chunks sum a matmul's K splits in one program (same bits)."""
    import os

    return os.environ.get("TF_GLM_EXL3_SEQ", "1") != "0"


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_exlr8_k4_20261017_v1", sources=[str(here / "exl3.cpp"), str(here / "exl3.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@dataclass
class Exl3Experts:
    """One layer's routed experts on this rank: trellis words [E, K/16, N/16, 32] int32 and scales per expert."""

    gt: torch.Tensor          # gate trellis [E, D/16, NI/16, 32]
    ut: torch.Tensor          # up trellis
    dt: torch.Tensor          # down trellis [E, NI/16, D/16, 32]
    suh_g: torch.Tensor       # [E, D] fp16
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # [E, NI] fp16 (this rank's columns)
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # [E, NI] fp16 (this rank's rows of down)
    svh_d: torch.Tensor       # [E, D]
    count: int
    width: int                # NI: this rank's share of the expert width
    dims: int                 # D
    # [D] fp16 when every expert's gate and up share one suh bitwise (found at construction; GLM-5.3-Flash's
    # EXL3 checkpoint, every MoE layer): prompt chunks then rotate each row's input once (``rot_rows``)
    shared_suh: torch.Tensor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.shared_suh = shared_suh(self.suh_g, self.suh_u)

    def nbytes(self) -> int:
        extra = () if self.shared_suh is None else (self.shared_suh,)
        return sum(t.numel() * t.element_size() for t in (self.gt, self.ut, self.dt, self.suh_g, self.suh_u,
                                                         self.svh_g, self.svh_u, self.suh_d, self.svh_d, *extra))


def shared_suh(suh_g: torch.Tensor, suh_u: torch.Tensor) -> torch.Tensor | None:
    """The one input scale vector [D] if every expert's gate and up suh equal it bit for bit (-0 is not +0), else
    None. Every pair's rotated input x_row * suh @ H is then the row's alone, whatever expert or matrix it feeds."""

    if suh_g.dim() != 2 or suh_g.shape != suh_u.shape or suh_g.dtype != torch.float16 or suh_u.dtype != suh_g.dtype:
        return None
    g, u = suh_g.view(torch.int16), suh_u.view(torch.int16)
    if not (bool(torch.equal(g, g[:1].expand_as(g))) and bool(torch.equal(u, g))):
        return None
    return suh_g[0].clone().contiguous()


def words(trellis: torch.Tensor) -> torch.Tensor:
    """A trellis (int16 [..., 64], 4 bits) as the kernels read it: int32 [..., 32], the same bytes."""

    if trellis.dtype != torch.int16 or trellis.shape[-1] != 64:
        raise ValueError("only 4-bit EXL3 trellises (int16 [..., 64]) are supported")
    return trellis.contiguous().view(torch.int32)


def prompt_pass() -> int:
    """TF_GLM_EXL3_PASS: members a prompt kernel pass takes (64 or 128, the same bits; 128 measured slower on real
    routing: most passes are small, and its one 16-warp block an SM keeps fewer experts in flight)."""
    import os

    value = int(os.environ.get("TF_GLM_EXL3_PASS", "64"))
    if value not in (64, 128):
        raise ValueError(f"TF_GLM_EXL3_PASS is 64 or 128, not {value}")
    return value


def prompt_kernels() -> bool:
    """TF_GLM_EXL3_PROMPT (default on): prompt chunks run the Y^T = W^T X^T kernels on a 64-pair plan (one fp32
    chain a row: other bits than the decode kernels', independent of the chunking); 0: the decode kernels."""
    import os

    return os.environ.get("TF_GLM_EXL3_PROMPT", "1") != "0"


def item_order() -> bool:
    """TF_GLM_EXL3_ORDER (default on): prompt kernels run the items of single-item experts (DRAM-bound) and of
    multi-item experts (mma-bound, weights shared through L2) interleaved; 0: plan order. The same bits either way."""
    import os

    return os.environ.get("TF_GLM_EXL3_ORDER", "1") != "0"


def row_rotation() -> bool:
    """TF_GLM_EXL3_ROWROT (default on): prompt chunks of layers whose experts share one gate/up suh rotate each row's
    input once and load it once for gate and up (the same fp16 inputs, so the same bits); 0: per pair, as decode."""
    import os

    return os.environ.get("TF_GLM_EXL3_ROWROT", "1") != "0"


class Scratch:
    """Per-window buffers for up to ``rows`` rows of ``slots`` slots (the last slot is the shared expert's); a
    prompt window's (``prompt``) kernels keep no split partials."""

    def __init__(self, rows: int, slots: int, dims: int, width: int, device, *, prompt: bool = False) -> None:
        P = rows * slots
        sk = max(GATEUP_CFG[2], DOWN_CFG[2])
        self.xg = torch.zeros((P, dims), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, dims), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, width), dtype=torch.float16, device=device)
        self.z = torch.zeros((1 if prompt else 2 * sk * P * max(width, dims),), dtype=torch.float32, device=device)
        # decode windows' fused gate/up epilogue: a finished-block count per (item, 128 columns), reset by the kernel
        self.done = torch.zeros((1 if prompt else (P + P // grouped.TILE + 1) * max(1, width // 128),),
                                dtype=torch.int32, device=device)
        self.rows, self.slots = rows, slots
        # a prompt plan's launch order (item_order): room for any plan of up to 1,024 experts in items of >= 16 pairs
        self.order = torch.zeros((P // 16 + 1024 if prompt else 1,), dtype=torch.int32, device=device)


def routed(x: torch.Tensor, pick: torch.Tensor, plan: grouped.Plan, ex: Exl3Experts, s: Scratch, y: torch.Tensor,
           R: int, limit: float) -> None:
    """Y[row * slots + slot] (fp32) for rows 0..R-1's routed slots, from normed bf16 x and ``experts.route``'s plan."""

    ext = _ext()
    slots, P = s.slots, s.rows * s.slots
    D, NI = ex.dims, ex.width
    if plan.tile in (64, 128):                       # a prompt window: the Y^T kernels, an item a pass
        shx = ex.shared_suh is not None and row_rotation()
        if shx:                                      # one rotated input a row, read by all its pairs' gate and up
            ext.rot_rows(x, x.stride(0), ex.shared_suh, s.xg, R, D)
            xg = xu = s.xg
        else:
            ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
            xg, xu = s.xg, s.xu
        ext.prompt(xg, xu, ex.gt, ex.ut, ex.dt, plan.items, plan.counts, plan.members, ex.svh_g, ex.svh_u,
                   ex.suh_d, ex.svh_d, s.xd, y, D, NI, ex.count, grouped.max_items(R * slots, plan.experts,
                                                                                   plan.tile), float(limit),
                   plan.tile, slots, shx, s.order if item_order() else None)
        return
    if plan.tile != grouped.TILE:
        raise ValueError("the EXL3 kernel takes items of 16 pairs")
    items = grouped.max_items(R * slots, plan.experts)
    dec, fuse, xrow = dec_settings()
    if dec and R < PROMPT_ROWS and GATEUP_CFG == (8, 4, 4) and DOWN_CFG == (8, 4, 1) and D == 4096 and NI in DEC_WIDTHS:
        xrow = xrow and ex.shared_suh is not None
        if xrow:                                     # one rotated input a row, read by all its pairs' gate and up
            ext.rot_rows(x, x.stride(0), ex.shared_suh, s.xg, R, D)
            xg = xu = s.xg
        else:
            ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
            xg, xu = s.xg, s.xu
        ld = dec_loads()
        dec_ext = ext
        if 2 <= R <= 8:
            from . import trial_control
            if trial_control.ACTIVE.get('expert_order',0):
                from .expert_order import extension
                dec_ext = extension(trial_control.ACTIVE['expert_order'])
        dec_ext.dec(xg, xu, ex.gt, ex.ut, plan.items, plan.counts, plan.members, s.z, 2, D, NI, P, 4, items, slots,
                2 if fuse & 2 else 0, xrow, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, float(limit), s.done, ld)
        if not fuse & 2:
            ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, NI, 4, slots, float(limit))
        dec_ext.dec(s.xd, s.xd, ex.dt, ex.dt, plan.items, plan.counts, plan.members, s.z, 1, NI, D, P, 1, items, slots,
                1 if fuse & 1 else 0, False, ex.svh_d, ex.svh_d, ex.svh_d, y, 0.0, s.done, ld)
        if not fuse & 1:
            ext.down_epilogue(s.z, pick, ex.svh_d, y, R, P, D, 1, slots)
        return
    ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
    nfirst = R >= PROMPT_ROWS and _nfirst()
    nt, w, sk = GATEUP_CFG
    seq = nfirst and sk > 1 and _seq()        # the splits summed in one program: an SK = 1 epilogue, the same bits
    ext.grouped(s.xg, s.xu, ex.gt, ex.ut, plan.items, plan.counts, plan.members, s.z, 2, D, NI, P, sk, items, nt,
                w, nfirst, _pf(), seq)
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, NI, 1 if seq else sk, slots,
                        float(limit))
    nt, w, sk = DOWN_CFG
    ext.grouped(s.xd, s.xd, ex.dt, ex.dt, plan.items, plan.counts, plan.members, s.z, 1, NI, D, P, sk, items, nt,
                w, nfirst, _pf(), nfirst and sk > 1 and _seq())
    ext.down_epilogue(s.z, pick, ex.svh_d, y, R, P, D, sk, slots)
