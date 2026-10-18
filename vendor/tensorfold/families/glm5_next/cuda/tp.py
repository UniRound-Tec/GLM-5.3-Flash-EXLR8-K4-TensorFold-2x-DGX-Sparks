"""GLM-5.3-Flash's tensor-parallel layout over N ranks (N = 2, 3, 4): which share of each split dimension a rank
holds. Every family splits in whole units (``tensorfold.cuda.shares``), the remainder to the lowest ranks, so two
ranks hold exact halves (the two-rank engine's split) and three hold e.g. 22/21/21 heads and 6/5/5 of an expert's
16 Hadamard blocks. Nothing is padded: no exchanged tensor is per-head or per-column (the ranks gather full-width
fp32 partials and top-k candidates), so ranks may hold different shapes.

Families (the split checkpoint tensors, ``split.ROW`` / ``COL`` / EXL3 rules):
  heads   MLA attention heads: q_b and kv_b rows, a DSA layer's o_proj columns        unit 1 head
  lin     KDA heads: q/k/v (and their convs), f_b, g_b, b, A_log, dt_bias rows, o_proj  unit 1 head
  dense   the dense MLP's intermediate (layers before first_k_dense_replace)            unit 128
  moe     a routed expert's intermediate (EXL3: whole 128-column Hadamard blocks)        unit 128
  shared  the shared expert's intermediate                                              unit 128
  vocab   lm_head rows                                                                  unit 64
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tensorfold.cuda.shares import parts

UNITS = {"heads": 1, "lin": 1, "dense": 128, "moe": 128, "shared": 128, "vocab": 64}
SIZES = (2, 3, 4)                     # the engine's tensor-parallel sizes

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_FAMILIES = (
    (re.compile(r"\.mlp\.experts\.\d+\."), "moe"),
    (re.compile(r"\.mlp\.shared_experts\."), "shared"),
    (re.compile(r"\.mlp\.(gate|up|down)_proj\."), "dense"),
    (re.compile(r"\.self_attn\.(q_b|kv_b)_proj\."), "heads"),
    (re.compile(r"\.self_attn\.((q|k|v)_proj|(q|k|v)_conv1d|f_b_proj|g_b_proj|b_proj)\."), "lin"),
    (re.compile(r"\.self_attn\.(A_log|dt_bias)$"), "lin"),
    (re.compile(r"^lm_head\."), "vocab"),
)
_O_PROJ = re.compile(r"\.self_attn\.o_proj\.")


@dataclass(frozen=True)
class Layout:
    """Rank ``rank`` of ``world``: the totals of every family and this rank's [start, stop) of each."""

    rank: int
    world: int
    totals: dict = field(hash=False)
    kinds: tuple = ()                  # per layer "kda" / "dsa" (an o_proj's family); past them: the MTP layer (dsa)

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.world:
            raise ValueError(f"rank {self.rank} of {self.world}")
        for name in self.totals:
            self.sizes(name)                                # every family must split whole

    @classmethod
    def from_config(cls, cfg, rank: int, world: int) -> "Layout":
        """From ``weights.Config``."""

        names = {"heads": "heads", "lin": "lin_heads", "dense": "dense_width", "moe": "moe_width",
                 "shared": "shared_width", "vocab": "vocab"}
        totals = {f: int(getattr(cfg, a)) for f, a in names.items() if getattr(cfg, a, None) is not None}
        return cls(int(rank), int(world), totals, tuple(getattr(cfg, "kinds", ()) or ()))

    @classmethod
    def from_text(cls, t: dict, rank: int, world: int) -> "Layout":
        """From a checkpoint's text config (``config.json``'s text_config or the whole of it)."""

        lin = t.get("linear_attn_config") or {}
        kinds = tuple("kda" if k == "linear_attention" else "dsa" for k in t.get("layer_types", ()))
        totals = {}
        if "num_attention_heads" in t:
            totals["heads"] = int(t["num_attention_heads"])
        if "num_heads" in lin or "linear_num_heads" in t:
            totals["lin"] = int(lin.get("num_heads", t.get("linear_num_heads", 0)))
        if "intermediate_size" in t:
            totals["dense"] = int(t["intermediate_size"])
        if "moe_intermediate_size" in t:
            moe = int(t["moe_intermediate_size"])
            totals.update(moe=moe, shared=moe * int(t.get("n_shared_experts", 1)))
        if "vocab_size" in t:
            totals["vocab"] = int(t["vocab_size"])
        return cls(int(rank), int(world), totals, kinds)

    def sizes(self, family: str) -> list[int]:
        return split_sizes(self.totals[family], self.world, UNITS[family])

    def size(self, family: str) -> int:
        return self.sizes(family)[self.rank]

    def start(self, family: str) -> int:
        return sum(self.sizes(family)[:self.rank])

    def largest(self, family: str) -> int:
        """The largest share any rank holds (rank 0's): what a buffer every rank sizes alike must take."""
        return max(self.sizes(family))

    def cut(self, family: str, length: int) -> tuple[int, int]:
        """This rank's [start, stop) of an axis of ``length`` elements holding ``family``'s total (rows of a weight,
        packed words, trellis tiles, group scales)."""

        total, a = self.totals[family], self.start(family)
        b = a + self.size(family)
        if (a * length) % total or (b * length) % total:
            raise ValueError(f"rank {self.rank} of {self.world}: [{a}, {b}) of {family}'s {total} is not whole on an "
                             f"axis of {length}")
        return a * length // total, b * length // total

    def at(self, rank: int) -> "Layout":
        return Layout(rank, self.world, self.totals, self.kinds)

    # the per-rank widths the engine sizes its weights and buffers by
    @property
    def heads(self) -> int:
        return self.size("heads")

    @property
    def lin_heads(self) -> int:
        return self.size("lin")

    @property
    def dense(self) -> int:
        return self.size("dense")

    @property
    def moe(self) -> int:
        return self.size("moe")

    @property
    def shared(self) -> int:
        return self.size("shared")

    @property
    def vocab(self) -> int:
        return self.size("vocab")

    @property
    def vocab_offset(self) -> int:
        return self.start("vocab")

    def family(self, name: str) -> str | None:
        """The family a split checkpoint tensor belongs to (None: not one this layout splits)."""

        if _O_PROJ.search(name):
            m = _LAYER.search(name)
            layer = int(m.group(1)) if m else -1
            return "lin" if 0 <= layer < len(self.kinds) and self.kinds[layer] == "kda" else "heads"
        for pattern, family in _FAMILIES:
            if pattern.search(name):
                return family
        return None


def split_sizes(total: int, world: int, unit: int) -> list[int]:
    """Each rank's share of a family's ``total``: halves at two ranks whenever the total is even (the two-rank
    engine's split, whatever the unit), else whole ``unit``s as even as they go (``shares.parts``)."""

    if world == 2 and total % 2 == 0:
        return [total // 2, total // 2]
    return parts(total, world, unit)


def of(w) -> Layout:
    """The layout of a ``weights.Weights`` (or a stand-in with cfg, rank and world)."""

    lay = getattr(w, "tp", None)
    if lay is None:
        lay = Layout.from_config(w.cfg, getattr(w, "rank", 0), getattr(w, "world", 1))
    return lay


def draft_parts(heads: int, kv_heads: int, inter: int, world: int) -> tuple[list[int], list[int], list[int]]:
    """DFlash2's split: KV heads in whole groups (each with its heads // kv_heads query heads), the MLP in units of
    128 -> (query heads, KV heads, MLP width) per rank. 32 / 8 over 3 ranks: KV 3/3/2, queries 12/12/8."""

    if heads % kv_heads:
        raise ValueError(f"drafter: {heads} query heads over {kv_heads} KV heads")
    group = heads // kv_heads
    kv = split_sizes(kv_heads, world, 1)
    return [k * group for k in kv], kv, split_sizes(inter, world, 128 if inter % 128 == 0 else 1)
