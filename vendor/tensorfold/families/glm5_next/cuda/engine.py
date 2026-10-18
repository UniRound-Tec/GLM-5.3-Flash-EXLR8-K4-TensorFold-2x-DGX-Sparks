"""GLM-5.3-Flash on N NCCL ranks (2, 3 or 4); all sample by one keyed rule from the same gathered candidates, so no broadcast."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_POLICY = "auto"
# DFlash2 drafts every round: by default up to 5 while their probability product holds 0.3; TF_GLM_DFLASH_POLICY picks
# another DFlash2 policy (e.g. fnc7:0.3, the noise-aware stop rule) for requests that name none
DFLASH_POLICY = os.environ.get("TF_GLM_DFLASH_POLICY", "").strip() or "fc5:0.3"
EXL3_AUTO = DFLASH_POLICY             # what auto runs on an EXL3 checkpoint with the draft model
GRAPH_ROWS = (1, 2, 3, 4, 5, 6, 7, 8)  # verify windows (and MTP steps) captured as CUDA graphs; wider ones run eagerly


def max_rows(value: str | None = None) -> int:
    """TF_GLM_MAX_ROWS: the widest verify window, 16 (default) to 64 in steps of 8 (``geometry.MLA_DECODE_ROWS``,
    which the startup estimate sizes decode buffers for). Wider windows serve copy drafts only (TF_GLM_COPY_MAX up to
    this minus one); past 16 rows a window's all-gathers exceed TF_ROCE_MAX_KB's default 256 KiB (16 KiB a row) and
    go through NCCL unless it is raised (TF_ROCE_MAX_KB=512 for 32 rows)."""

    from tensorfold.cuda.geometry import MLA_DECODE_ROWS

    value = (os.environ.get("TF_GLM_MAX_ROWS", "") if value is None else value).strip()
    if value == "":
        return 16
    if not value.isdecimal() or not 16 <= int(value) <= MLA_DECODE_ROWS or int(value) % 8:
        raise ValueError(f"TF_GLM_MAX_ROWS: 16 to {MLA_DECODE_ROWS} in steps of 8, not {value!r}")
    return int(value)


MAX_ROWS = max_rows()                 # the widest verify window (a pending token and up to 15 drafts: copy drafts)


def wide_graphs(value: str | None = None, most: int | None = None) -> tuple[int, ...]:
    """TF_GLM_WIDE_GRAPHS: verify-window widths past GRAPH_ROWS (9 to TF_GLM_MAX_ROWS) that also replay CUDA graphs,
    comma separated (e.g. ``16`` or ``16,32``; off by default: wider windows run eagerly). Copy drafts pad a window
    that falls short of a captured width up to it (``copy_drafts.CopySettings.pad``). Each width costs a graph a KDA
    parity, and one more a pool bucket past the dense limit, at startup."""

    most = MAX_ROWS if most is None else most
    value = (os.environ.get("TF_GLM_WIDE_GRAPHS", "") if value is None else value).strip()
    if value in ("", "0"):
        return ()
    out = set()
    for part in value.split(","):
        part = part.strip()
        if not part.isdecimal() or not max(GRAPH_ROWS) < int(part) <= most:
            raise ValueError(f"TF_GLM_WIDE_GRAPHS: widths from {max(GRAPH_ROWS) + 1} to {most} (TF_GLM_MAX_ROWS), "
                             f"comma separated, not {value!r}")
        out.add(int(part))
    return tuple(sorted(out))


WIDE_GRAPHS = wide_graphs()
MTP_ROWS = 8                          # rows an MTP step takes at once (a kept backlog is absorbed in steps of this)
DENSE_CAPACITY = 2560                 # cache slots while DSA attention stays dense (contexts up to 2,051 tokens)
# TF_GLM_DRAFT_RING=0: DFlash2's context in a flat buffer of the whole window, not a 2,176-row ring (the same drafts)
DRAFT_RING = os.environ.get("TF_GLM_DRAFT_RING", "1").strip() != "0"
# --vision (rank 0): what startup reserves beside the tower's 1.05 GiB of bf16 weights for an encode and its
# prompt's feature rows (8 KiB a row, which stay through the prefill). The encode's scratch takes turns with a prompt
# chunk's transient scratch (~1 GiB at 164k), so this covers the rows and the scratch past it. A picture cap past
# 4,096 tokens scales it; TENSORFOLD_VISION_WORKSPACE_MIB sets it outright
VISION_WORKSPACE = 3 * 2**28


def vision_workspace() -> int:
    value = os.environ.get("TENSORFOLD_VISION_WORKSPACE_MIB", "")
    if value == "":
        from tensorfold.vision.glm import GlmVisionLimits

        return VISION_WORKSPACE * max(4096, GlmVisionLimits.from_env().image_tokens) // 4096
    if not value.isdecimal() or int(value) > 16384:
        raise ValueError(f"TENSORFOLD_VISION_WORKSPACE_MIB: 0 to 16,384 MiB, not {value!r}")
    return int(value) * 2**20


GRIDS = (16, 32, 64, 128)             # TF_GLM_PROMPT_GRID's values besides 0 (off)


def kda_chunked_on() -> bool:
    from .kda import CHUNKED

    return CHUNKED


def prompt_grid(prefill_rows: int, value: str | None = None, chunked: bool | None = None) -> int:
    """TF_GLM_PROMPT_GRID: 0 (off, the default) or G in ``GRIDS``, which ``prefill_rows`` must be a multiple of.

    With G set, every prompt chunk starts on a multiple of G and a kept prompt state sits at the last multiple of G
    in the prompt (``grid_point``), so prompt kernels whose rows depend on G-row block alignment resume exactly.
    ``chunked`` (default: TF_GLM_KDA_CHUNKED): the chunked KDA prompt path, whose rows depend on the alignment of its
    32-row sub-chunks, needs G a multiple of 32; unset or 0 then means 64."""

    if chunked is None:
        chunked = kda_chunked_on()
    value = os.environ.get("TF_GLM_PROMPT_GRID", "") if value is None else value
    value = value.strip()
    if value in ("", "0"):
        if not chunked:
            return 0
        value = "64"
    if not value.isdecimal() or int(value) not in GRIDS:
        raise ValueError(f"TF_GLM_PROMPT_GRID: 0 or one of {', '.join(map(str, GRIDS))}, not {value!r}")
    grid = int(value)
    if chunked and grid % 32:
        raise ValueError(f"TF_GLM_PROMPT_GRID={grid}: TF_GLM_KDA_CHUNKED=1 runs prompt chunks in 32-row sub-chunks at "
                         f"absolute positions, so prompt chunks must start on a grid of 32, 64 or 128 (0: 64)")
    if prefill_rows % grid:
        raise ValueError(f"TF_GLM_PROMPT_GRID={grid}: prompt chunks of {prefill_rows} rows (TF_GLM_PREFILL_ROWS) "
                         f"must be a multiple of it")
    return grid


def grid_point(n: int, cut: int, grid: int) -> int | None:
    """Where a drafted request of an n-token prompt resumed at ``cut`` keeps its prompt state, or None for nowhere.

    Off (grid 0) and on-grid prompts: at n. Otherwise at B = floor(n / grid) * grid, when B lies past ``cut`` (so
    B > 0); the request then prefills up to B, keeps that state, and resumes from it to n."""

    if not grid or n % grid == 0:
        return n
    point = n // grid * grid
    return point if point > cut else None


SHARED_DEFAULT = 256                  # TF_GLM_SHARED_PREFIX=1: the fewest tokens a shared-prefix state adds
SHARED_MIN = 16
SHARED_MOST = 2                       # shared-prefix states a request keeps at most (the header's two slots)


def shared_prefix(value: str | None = None) -> int:
    """TF_GLM_SHARED_PREFIX: 0 (off, the default), or the fewest tokens past a request's resume point a shared-prefix
    state must add (1 or on: 256; else 16 or more).

    On, a drafted text request also keeps its state where its prompt stops sharing a kept one and at the end of its
    system block (its first user message), each on the prompt grid, so a later prompt sharing only that prefix resumes
    from it; saved states keep DFlash2's context window too."""

    value = os.environ.get("TF_GLM_SHARED_PREFIX", "") if value is None else value
    value = value.strip().lower()
    if value in ("", "0", "off"):
        return 0
    if value in ("1", "on"):
        return SHARED_DEFAULT
    if not value.isdecimal() or int(value) < SHARED_MIN:
        raise ValueError(f"TF_GLM_SHARED_PREFIX: 0 (off), 1 (on: {SHARED_DEFAULT} tokens) or the fewest tokens a "
                         f"shared-prefix state adds past a request's resume point, {SHARED_MIN} or more; not {value!r}")
    return int(value)


def common_prefix(a, b) -> int:
    """How many leading tokens two id sequences (lists or int64 arrays) share."""

    import numpy as np

    n = min(len(a), len(b))
    if n == 0:
        return 0
    x = np.asarray(a[:n], dtype=np.int64)
    y = np.asarray(b[:n], dtype=np.int64)
    diff = np.flatnonzero(x != y)
    return int(diff[0]) if len(diff) else n


def shared_points(prompt: list[int], cut: int, point: int | None, grid: int, least: int, known,
                  opener: int | None = None) -> list[int]:
    """Where a request resumed at ``cut``, keeping its own state at ``point`` (``grid_point``), also keeps shared-prefix
    states: at the longest prefix it shares with any of ``known`` (id sequences: the kept prompts; not the live reply,
    which a conversation's next prompt extends anyway) and at its first ``opener`` token (the end of the system block),
    each rounded down to the grid, kept when it adds at least ``least`` tokens past ``cut`` (and past the other point)
    and lies short of ``point``. Sorted, at most SHARED_MOST."""

    if not least or point is None:
        return []
    import numpy as np

    arr = np.asarray(prompt, dtype=np.int64)
    # only a sequence longer than cut + least can share enough (known may hold int64 arrays of the ids)
    wanted = [max((common_prefix(arr, ids) for ids in known if len(ids) >= cut + least), default=0)]
    if opener is not None:
        at = np.flatnonzero(arr == opener)
        if len(at):
            wanted.append(int(at[0]))
    out: list[int] = []
    for p in sorted(p // grid * grid if grid else p for p in wanted):
        if p - (out[-1] if out else cut) >= least and p < point:
            out.append(p)
    return out[-SHARED_MOST:]


def user_opener(model_dir: Path) -> int | None:
    """The id of GLM's ``<|user|>`` role token (the first one ends the system block), or None without a tokenizer."""

    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        return None
    try:
        added = json.loads(path.read_text()).get("added_tokens", [])
    except (OSError, ValueError):
        return None
    return next((int(t["id"]) for t in added if t.get("content") == "<|user|>"), None)


def encode_policy(spec: str) -> list[int]:
    """Encode policy kind, maximum drafts, and two parameters in millionths as four integers, with 10 added to kind for DFlash2.

    DFlash2 only: ``fncN:P[:B]`` (kind 16: the fcN:P product of noise-aware confidences, B their temperature,
    default ``dflash2.NOISY_BETA``) and ``fcostN:noisy[:B]`` (kind 17: the depth k <= N maximizing E[tokens] / round
    ms by the startup cost table, E from the same confidences; N up to the captured verify windows' drafts)."""

    spec = str(spec).strip()
    bad = ValueError(f"draft policy {spec!r}: expected auto[:E:EVERY:MARGIN], 0, N, a[:LOW:HIGH], cN:P, or one of "
                     f"these after f (N from 1 to {MAX_ROWS - 1}), fncN:P[:B] or fcostN:noisy[:B] "
                     f"(N up to {len(GRAPH_ROWS) - 1})")
    try:
        if spec == "auto" or spec.startswith("auto:"):
            parts = spec.split(":")
            if len(parts) not in (1, 4):
                raise bad
            explore, every, margin = (int(parts[1]), int(parts[2]), float(parts[3])) if len(parts) == 4 else (2, 8, 0.03)
            if explore < 1 or every < 0 or not 0 <= margin < 1:
                raise bad
            return [4 if len(parts) == 1 else 5, explore, every, int(round(margin * 1e6))]
        if spec.startswith("fadapt"):
            parts=spec[6:].split(":")
            if len(parts) not in (1,2):raise bad
            most=int(parts[0]);mode=int(parts[1]) if len(parts)==2 else 1
            if str(most)!=parts[0] or not 1<=most<=7 or mode not in (0,1):raise bad
            return [19,most,mode,0]
        if spec.startswith("femp"):
            most = int(spec[4:])
            if str(most) != spec[4:] or not 1 <= most <= min(7,len(GRAPH_ROWS)-1):
                raise bad
            return [18,most,0,0]
        if spec.startswith(("fnc", "fcost")):
            return _encode_noisy(spec[1:], bad)
        if spec.startswith("f"):
            code = encode_policy(spec[1:])
            return [code[0] + 10] + code[1:] if code[0] else code
        if spec.startswith("a"):
            parts = spec.split(":")
            if parts[0] != "a" or len(parts) not in (1, 3):
                raise bad
            low, high = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.8, 0.9)
            return [2, 3, int(round(low * 1e6)), int(round(high * 1e6))]
        if spec.startswith("c"):
            most_text, conf = spec[1:].split(":")
            most = int(most_text)
            if not 0 < most < MAX_ROWS:
                raise bad
            return [3, most, int(round(float(conf) * 1e6)), 0]
        most = int(spec)
    except ValueError:
        raise bad from None
    if not 0 <= most < MAX_ROWS:
        raise bad
    return [1 if most > 0 else 0, most, 0, 0]


def _encode_noisy(spec: str, bad: ValueError) -> list[int]:
    """``ncN:P[:B]`` -> [16, N, P, B] and ``costN:noisy[:B]`` -> [17, N, B, 0], P and B in millionths (DFlash2 only)."""

    from .dflash2 import NOISY_BETA

    cost = spec.startswith("cost")
    parts = spec[4 if cost else 2:].split(":")
    if len(parts) not in (2, 3) or (cost and parts[1] != "noisy"):
        raise bad
    most = int(parts[0])
    beta = float(parts[2]) if len(parts) == 3 else NOISY_BETA
    if not 0 < most <= len(GRAPH_ROWS) - 1 or not 0 < beta <= 1000 or round(beta * 1e6) < 1:
        raise bad
    if cost:
        return [17, most, int(round(beta * 1e6)), 0]
    conf = float(parts[1])
    if not 0 <= conf <= 1:
        raise bad
    return [16, most, int(round(conf * 1e6)), int(round(beta * 1e6))]


def decode_policy(code: list[int], costs: dict | None = None):
    """Decode a serial, MTP, or automatic policy, retaining exploration, sampling, and margin settings; ``costs``: the
    startup cost table (``GlmEngine._calibrate``) the fcost rule weighs a round's depths by."""

    from .decode import DepthPolicy

    kind, most, a, b = code
    if kind in (4, 5):
        return ("auto", most, a, b / 1e6, kind == 5)
    if kind == 19:
        from .adaptive_depth import AdaptiveDepth
        return AdaptiveDepth(most,costs,mode=a)
    if kind == 18:
        from .empirical_depth import EmpiricalDepth
        return EmpiricalDepth(most,costs)
    kind %= 10
    if kind == 2:
        return DepthPolicy(min(most, MAX_ROWS - 1), low=a / 1e6, high=b / 1e6)
    if kind == 3:
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=a / 1e6)
    if kind == 6:
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=a / 1e6, beta=b / 1e6)
    if kind == 7:
        if costs is None:
            raise ValueError("the fcost policy needs the engine's cost table")
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, beta=a / 1e6, round_ms=round_costs(costs))
    return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True) if kind == 1 else None


def round_costs(costs: dict) -> tuple[float, ...]:
    """ms of a DFlash2 round of k drafts, k = 0 .. len(verify) - 1: the verify window of k + 1 rows, the block and a
    kept row's context update, as draft_sim's Costs(verify, block, 0.0, taps_row).round_ms(k, 1, True)."""

    return tuple(v + costs["block"] + 0.0 + costs["taps_row"] * 1 for v in costs["verify"])


PARALLEL_MOST = 8                     # --parallel: concurrent requests at most (segments.MAX_SEGS: a window's streams)
MULTI_WINDOW = 32                     # --parallel: rows of every stream's verify windows together (multi.MAX_WINDOW)


def multi_window(value: str | None = None) -> int:
    """TF_GLM_MULTI_WINDOW: --parallel's shared verify window, the rows of every stream's windows together in a round
    (the longest tails trimmed past it): 32 (default) or 16 to 64 in steps of 8 (``geometry.MLA_DECODE_ROWS``: the
    widest decode window the kernels serve). The batched window's buffers, token selection scratch and CUDA graphs (one
    a row count) are sized for it, and the startup estimate counts them. Past 32 rows a round's all-gathers exceed
    TF_ROCE_MAX_KB=512 (16 KiB a row) and go through NCCL unless it is raised (1024 for 64 rows)."""

    from tensorfold.cuda.geometry import MLA_DECODE_ROWS

    value = (os.environ.get("TF_GLM_MULTI_WINDOW", "") if value is None else value).strip()
    if value == "":
        return MULTI_WINDOW
    if not value.isdecimal() or not 16 <= int(value) <= MLA_DECODE_ROWS or int(value) % 8:
        raise ValueError(f"TF_GLM_MULTI_WINDOW: 16 to {MLA_DECODE_ROWS} rows in steps of 8, not {value!r}")
    return int(value)


def multi_decode_rows(parallel: int, window: int, rows: int = MAX_ROWS) -> int:
    """Decode-window rows the startup estimate sizes buffers for: MLA_DECODE_ROWS, or under --parallel the engine's own
    window (``rows``: TF_GLM_MAX_ROWS, a lone stream's) and the batched one (``window``) together when they take more
    (TF_GLM_MULTI_WINDOW past 48 at the default 16 rows; 16 + 32 = 48 fit the 64 rows as ever)."""

    from tensorfold.cuda.geometry import MLA_DECODE_ROWS

    return max(MLA_DECODE_ROWS, int(rows) + int(window)) if parallel > 1 else MLA_DECODE_ROWS


def slot_bytes(t: dict, world: int = 2, rows: int = MAX_ROWS, rank: int = 0) -> int:
    """What one more stream slot (``forward.Slots``) takes on rank ``rank`` of ``world``, from the text config:
    every KDA layer's recurrent states (two parities), conv window, a decode window's projections and replay
    scratch."""

    from tensorfold.cuda.geometry import layer_counts

    from .tp import UNITS, split_sizes

    linear, _ = layer_counts(t)
    lin = t.get("linear_attn_config") or {}
    lh = split_sizes(int(lin.get("num_heads", t.get("linear_num_heads", 64))), world, UNITS["lin"])[rank]
    ld = int(lin.get("head_dim", t.get("linear_head_dim", 128)))
    conv = int(lin.get("short_conv_kernel_size", t.get("linear_conv_kernel_dim", 4)))
    width = 3 * lh * ld + 2 * ld + lh                  # the KDA input projection's rows (q, k, v, f_a, g_a, beta)
    scratch = rows * lh * (ld * 2 + ld * 4 + ld * 2 + ld * 4 + 4)
    return linear * (2 * lh * ld * ld * 4 + (conv - 1) * 3 * lh * ld * 2 + rows * width * 2 + scratch)


def multi_draft_bytes(t: dict, streams: int, world: int = 2, *, ring: bool, capacity: int,
                      tap_rows: int = MAX_ROWS, rank: int = 0) -> int:
    """``dflash2_multi.MultiDrafter``'s context pool on rank ``rank``: ``streams`` rings (or flat contexts of
    capacity + block rows) and the padded context updates' trash rows, keys and values per draft layer."""

    from tensorfold.cuda.geometry import draft_ring_rows

    from .tp import split_sizes

    layers = int(t["num_hidden_layers"])
    heads = split_sizes(int(t["num_key_value_heads"]), world, 1)[rank]
    hd = int(t["head_dim"])
    block = int((t.get("dflash_config") or {}).get("block_size", 16))
    window = int(t.get("sliding_window", 0))
    ring_rows = draft_ring_rows(window - 1, block) if ring and window > 0 else 0
    cap = ring_rows if 0 < ring_rows < capacity + block else capacity + block

    def update_rows(n: int) -> int:                 # a context update's rows (MultiDrafter.t_max): the trash
        return -(-max(64, n * tap_rows) // block) * block

    trash = update_rows(streams)
    # past four streams (whose blocks and updates the drafter's own 64-row estimate covers): the batched block pass's
    # activations (dflash2_geometry's 16 fp32 values a row and width) and the context update's tap rows grow too
    four = min(streams, 4)
    hidden, inter = int(t.get("hidden_size", 0)), int(t.get("intermediate_size", 0))
    taps = len((t.get("dflash_config") or {}).get("target_layer_ids") or ()) or 5
    passes = (16 * (max(64, streams * block) - max(64, four * block)) * (hidden + inter) * 4
              + (trash - update_rows(four)) * taps * hidden * 2)
    return 2 * layers * heads * hd * (streams * cap + trash) * 2 + passes


def draft_fraction(draft_dir, rank: int, world: int):
    """``geometry.dflash2_weights``'s per-rank rows of the drafter at ``draft_dir`` (``tp.draft_parts``)."""

    from tensorfold.cuda.capacity import config as read_config

    from .tp import draft_parts

    t = read_config(draft_dir)
    heads, kv_heads, inter = int(t["num_attention_heads"]), int(t["num_key_value_heads"]), int(t["intermediate_size"])
    qs, kvs, mlps = draft_parts(heads, kv_heads, inter, world)
    hd = int(t.get("head_dim") or int(t["hidden_size"]) // heads)
    rows = {"q": qs[rank] * hd, "kv": kvs[rank] * hd, "mlp": mlps[rank]}
    totals = {"q": heads * hd, "kv": kv_heads * hd, "mlp": inter}

    def fraction(kind: str, n: int) -> int:
        if n != totals[kind]:
            raise ValueError(f"drafter {kind} rows {n}, config {totals[kind]}")
        return rows[kind]
    return fraction


def _f64_ints(x: float) -> list[int]:
    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _ints_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


MTP_DEFAULT = "1"                     # TF_GLM_MTP when unset: the MTP head stays beside DFlash2 (auto, 0: left out)


def mtp_head(drafter: bool, serial_only: bool, layers: int, value: str | None = None) -> bool:
    """TF_GLM_MTP: load the MTP head? auto: not beside DFlash2 or with --no-drafts; 1: whenever it exists; 0: never."""

    value = os.environ.get("TF_GLM_MTP", "") if value is None else value
    value = value.strip().lower() or MTP_DEFAULT
    if value not in ("0", "1", "auto"):
        raise ValueError(f"TF_GLM_MTP: 0, 1 or auto, not {value!r}")
    if value == "auto":
        return bool(layers) and not drafter and not serial_only
    return value == "1" and bool(layers)


def without_mtp(transform, layers: int):
    """A startup weight transform that leaves out the MTP layer's tensors (``layers.<num_hidden_layers>.``)."""

    prefix = f"model.language_model.layers.{layers}."
    return lambda name, info: (0, 0) if name.startswith(prefix) else transform(name, info)


def bell_key(rank: int, n: int) -> str:
    """The store key that wakes follower ``rank`` for request ``n`` (rank 1's is the two-rank engine's name)."""

    return f"tf_glm_request_{n}" if rank == 1 else f"tf_glm_request_r{rank}_{n}"


class GlmEngine:
    """GLM-5.3-Flash on ``world`` ranks (this one ``rank``): weights, MTP and DFlash2 drafting, per-request policies."""

    world = 2                   # ranks the model is split over (``tp.Layout``; set by __init__)
    shared = 0                  # TF_GLM_SHARED_PREFIX (``shared_prefix``): 0 off, else the fewest tokens it adds
    opener = None               # rank 0, shared on: ``<|user|>``'s id, whose first appearance ends the system block
    parallel = 1                # --parallel N: requests decoded together (``multi``); 1 serves one at a time
    concurrent = False          # the server hands every request straight to ``generate`` (the scheduler orders them)
    multi = None                # --parallel > 1: the ``multi.MultiDecoder`` (both ranks)
    scheduler = None            # ... and rank 0's ``multi.GlmScheduler``

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, world: int = 2,
                 policy: str = DEFAULT_POLICY,
                 drafter: Path | None = None, context: int | None = None, context_explicit: bool | None = None, serial_only: bool = False, comm=None,
                 prefill_rows: int | None = None, vision: bool = False, vision_urls: bool = False,
                 split=None, parallel: int = 1, graphs: bool = True) -> None:
        """``comm``: a communicator with ``all_gather`` and ``barrier`` instead of NCCL between two machines (tests);
        ``vision``: rank 0 runs the image tower (rank 1 only takes its rows); ``split``: a ``hcsplit.SplitSettings``
        instead of TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP (tests); ``parallel``: requests decoded together
        (``multi``: their caches share one pool, each stream up to the context window; DFlash2 drafts only);
        ``world``: the ranks the model is split over (2, 3 or 4; ``tp.Layout``); ``graphs``: False runs every window
        eagerly (tests whose communicator cannot be captured)."""

        import torch

        from tensorfold.cuda.comm import NCCL
        from .decode import Engine
        from .weights import Config, load
        from .split import rule
        from tensorfold.cuda.capacity import admit
        from tensorfold.cuda.geometry import (PREFILL_ROWS, dflash2_geometry, dflash2_weights, mla_geometry,
                                              mla_ring_bytes, split_weights)

        encode_policy(policy)                           # a bad default fails here, not in the first request
        if not encode_policy(DFLASH_POLICY)[0] >= 10:
            raise ValueError(f"TF_GLM_DFLASH_POLICY={DFLASH_POLICY!r}: a DFlash2 policy (f...), e.g. fc5:0.3 or fnc7:0.3")
        from .copy_drafts import CopySettings

        # TF_GLM_COPY_DRAFTS=1: a reply that repeats earlier text drafts its continuation (prompt lookup) ahead of
        # the MTP head and DFlash2; both ranks must agree, so the settings join the startup comparison below
        self.copy = None if serial_only else CopySettings.from_env(MAX_ROWS - 1, pad=WIDE_GRAPHS)
        from .hcsplit import SplitSettings

        # TF_GLM_HC_SPLIT=1 / TF_GLM_PREFILL_OVERLAP=1: prompt chunks' hyper-connection glue split by rows between the
        # ranks (and its exchanges overlapped); the same bits, both ranks must agree (the comparison below)
        self.split = SplitSettings.from_env() if split is None else split
        world = int(world) if comm is None else int(getattr(comm, "world", world))
        from .tp import SIZES

        if world not in SIZES:
            raise ValueError(f"GLM-5.3-Flash's CUDA engine runs on {', '.join(map(str, SIZES))} ranks, not {world}")
        if int(parallel) > 1 and drafter is None and not serial_only:
            # before any connection or memory: the MTP head is not loaded under --parallel (``mtp_on`` below), so
            # this used to fail later as "TF_GLM_MTP=0 leaves no MTP head" whatever TF_GLM_MTP said (issue #38)
            raise ValueError(f"--parallel {int(parallel)}: concurrent requests draft with DFlash2 only (the "
                             "checkpoint's MTP head drafts one request at a time, --parallel 1): give every rank "
                             "the DFlash2 draft model (--drafter), serve with --parallel 1 for MTP drafts, or pass "
                             f"--no-drafts to every rank to decode {int(parallel)} requests together without drafts")
        torch.cuda.set_device(0)
        self.torch = torch
        self.rank, self.world = rank, world
        parallel = int(parallel)
        if not 1 <= parallel <= PARALLEL_MOST:
            raise ValueError(f"--parallel: 1 to {PARALLEL_MOST} requests at once for GLM-5.3-Flash, not {parallel}")
        self.parallel = parallel
        # TF_GLM_MULTI_WINDOW: the rows of a round's shared verify window (--parallel), read on every rank (compared below)
        self.multi_window = multi_window()
        self.policy = "0" if serial_only else os.environ.get("TF_GLM_DEFAULT_POLICY", policy)
        self.serial_only = serial_only
        self.comm = comm if comm is not None else NCCL(rank, world, master, port)
        self.comm.barrier()
        if comm is None and os.environ.get("TF_GLM_COMM", "nccl") == "roce":
            # small all-gathers (a decode round's partials, the samplers) over b12x's one-shot RoCE transport
            from tensorfold.cuda.roce import RoceComm

            self.comm = RoceComm(self.comm, rank, world)
            if rank == 0:
                roce = self.comm.roce
                routes = getattr(roce, "routes", None)
                via = ", ".join(roce.hcas) if routes is None else "; ".join(
                    f"rank {p} over {'+'.join(roce.hcas[h] for h, _ in pairs)}" for p, pairs in sorted(routes.items()))
                print(f"[tensorfold] all-gathers up to {roce.max_bytes >> 10} KiB over RoCE ({via}; b12x's RoCEnante "
                      "transport), larger ones over NCCL", flush=True)
        cfg = Config.read(model_dir)
        # Without --context the window stays dense, attending every key without indexer work.
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        from . import LATENT

        from .qmm import KVB_KINDS, dense_kind, fp8_weights, kvb_kind, kvb_weights, q4_weights

        # TF_GLM_DENSE=fp8 (an EXL3 checkpoint): its BF16 matrices take a byte a value, which the estimate counts
        from . import tp as tp_mod
        from .split import rank_cut

        lay = tp_mod.Layout.from_config(cfg, rank, world)
        self.tp = lay

        def cut(name: str, shape: list[int], kind: str) -> tuple[int, int]:
            """This rank's share of a split tensor (``split.rank_cut``) or of lm_head's rows, for the estimate."""
            if kind == "vocab":
                if "vocab" not in lay.totals:
                    return 0, shape[0] // world
                return lay.vocab_offset, lay.vocab_offset + lay.vocab
            got = rank_cut(lay, name, shape, kind)
            if got is None:                          # a family the config does not size (two ranks): halves
                n = shape[{"row": 0, "col": -1, "dim1": 1}[kind]]
                return 0, n // world
            return got

        def share(family: str, total: int) -> int:
            """This rank's width of a split dimension, for the estimate."""
            return lay.size(family) if family in lay.totals else int(total) // world
        weights_estimate = split_weights(rule, world, cut)
        if (Path(model_dir) / "exlr8-source.json").exists():
            from .nvfp4_memory import configure, checkpoint
            configure()
        if cfg.quant == "nvfp4_csf":
            from .nvfp4_capacity import wrap_weight_estimate
            from .nvfp4_memory import configure, checkpoint, scratch_budget
            configure()
            weights_estimate = wrap_weight_estimate(weights_estimate, cfg, world)
        if cfg.quant in ("exl3", "nvfp4_csf") and dense_kind() == "fp8":
            weights_estimate = fp8_weights(weights_estimate)
        elif cfg.quant in ("exl3", "nvfp4_csf") and dense_kind() == "q4":
            weights_estimate = q4_weights(weights_estimate)
        # TF_GLM_KVB=fp8/q4 (an EXL3 checkpoint, latent cache): kv_b as the absorb and expand kernels read it
        kvb = kvb_kind() if cfg.quant in ("exl3", "nvfp4_csf") and LATENT else "bf16"
        weights_estimate = kvb_weights(weights_estimate, kvb)
        # TF_GLM_KV=fp8: the DSA latent cache and the indexer's pooled keys as e4m3 rows (``kv8``), half the bytes
        from .kv8 import KINDS as KV_KINDS, kv_kind

        kv = kv_kind()
        if kv != "bf16" and not LATENT:
            raise ValueError("TF_GLM_KV=fp8 holds the latent cache: it needs TF_GLM_LATENT=1")
        self.kv = kv
        # TF_GLM_MTP off: the MTP layer's tensors, caches and buffers are neither loaded nor estimated
        self.mtp_on = mtp_head(drafter is not None, serial_only, cfg.mtp_layers) and parallel == 1   # multi: DFlash2
        if not self.mtp_on:
            weights_estimate = without_mtp(weights_estimate, cfg.layers)
        # rows a prompt chunk runs at once: TF_GLM_PREFILL_ROWS (both ranks alike), more amortizes each chunk's expert
        # weight decode over more tokens and costs buffer memory
        if prefill_rows is None:
            prefill_rows = int(os.environ.get("TF_GLM_PREFILL_ROWS", "") or PREFILL_ROWS)
        prefill_rows = int(prefill_rows)
        if not 64 <= prefill_rows <= 16384 or prefill_rows % 64:
            raise ValueError(f"prefill rows: a multiple of 64 from 64 to 16,384, not {prefill_rows}")
        # TF_GLM_PROMPT_GRID=G: prompt chunks start on multiples of G and kept prompt states sit on them
        self.grid = prompt_grid(prefill_rows)
        # TF_GLM_SHARED_PREFIX: kept states where prompts stop sharing, and saved states keep DFlash2's window; rank 0
        # picks the points (the header carries them), but both ranks save and evict alike, so both must agree
        self.shared = shared_prefix()
        vision = bool(vision) and rank == 0
        workspace = vision_workspace() if vision else 0
        if vision:                           # the tower's bf16 weights, which the split rule drops
            import math

            text_weights = weights_estimate
            weights_estimate = lambda name, info: ((math.prod(info["shape"]) * 2, 0)
                                                   if name.startswith("model.visual.") else text_weights(name, info))

        from .exl3_mm import prompt_kernels
        from .hcsplit import buffer_rows
        from .latent import SPARSE_ONEPASS

        # the prompt buffers' rows, the split's pad rows included (``hcsplit.buffer_rows``; the prompt chunk's rows at
        # two and four ranks)
        prompt_buffer_rows = buffer_rows(prefill_rows, world, self.split)

        nvfp4_scratch = 0
        if cfg.quant == "nvfp4_csf":
            nvfp4_scratch = scratch_budget(experts=cfg.experts, hidden=cfg.hidden,
                intermediate=lay.moe, top_k=cfg.top_k, max_tokens=max(64, prompt_buffer_rows),
                mode=os.environ.get("TF_GLM_NVFP4_ACT", "a4"), limit=cfg.limit)
            print(f"[tensorfold] NVFP4 shared preparation/execution storage: {nvfp4_scratch} bytes", flush=True)

        def geometry(text):
            from tensorfold.cuda.capacity import Geometry

            g = mla_geometry(text, world, MAX_ROWS, share=share, minimum_slots=DENSE_CAPACITY, latent=LATENT,
                             prefill_rows=prompt_buffer_rows,
                             mtp=self.mtp_on, onepass=SPARSE_ONEPASS, exl3_prompt=prompt_kernels(),
                             prompt_split_k=cfg.quant in ("exl3", "nvfp4_csf") and dense_kind() != "q4", kv=kv,
                             decode_rows=multi_decode_rows(parallel, self.multi_window))
            # --parallel: every stream slot past the first (KDA states, conv windows, window scratch); the batched
            # verify window's buffers (TF_GLM_MULTI_WINDOW rows) beside the engine's own fit the decode rows the
            # estimate sizes (``multi_decode_rows``), its token selection's fp32 scores (window rows x a pool of 4
            # tokens) grow with the pool
            nvfp4_reserve = 0
            if cfg.quant == "nvfp4_csf":
                # Actual shared scratch + scales; keep a separate 2 GiB allowance
                # for compiled programs/decoder metadata and allocator rounding.
                nvfp4_reserve = cfg.experts * 3 * cfg.hidden * lay.moe // 16 + nvfp4_scratch + (2 << 30)
            extra = workspace + nvfp4_reserve + (parallel - 1) * (slot_bytes(text, world, rank=rank) + mla_ring_bytes(text, prefill_rows))
            per = 4 * self.multi_window // 4 if parallel > 1 else 0
            if not extra and not per:
                return g
            # The shared pool rounds its token count down to ALIGN. Admit one
            # aligned reserve so an explicit window still fits when larger
            # prefill buffers leave no budget for extra cached conversations.
            reserve = g.reserve
            if cfg.quant == "nvfp4_csf" and parallel > 1:
                from .pool import ALIGN
                reserve = ((reserve + ALIGN - 1) // ALIGN) * ALIGN
            return Geometry(lambda slots: g.bytes_at(slots) + extra + per * slots, reserve, g.minimum_slots)

        def draft_geometry(text):
            from tensorfold.cuda.capacity import Geometry

            g = dflash2_geometry(text, world, MAX_ROWS, ring=DRAFT_RING,
                                 kv_heads=tp_mod.split_sizes(int(text["num_key_value_heads"]), world, 1)[rank])
            if parallel == 1:
                return g
            # --parallel: the multi-stream drafter's own pool of contexts beside the solo drafter's
            return Geometry(lambda slots: g.bytes_at(slots) + multi_draft_bytes(text, parallel, world, rank=rank, ring=DRAFT_RING,
                                                                                 capacity=slots), g.reserve)
        self.capacity_plan = admit(model_dir, context if explicit else cfg.dense_limit, explicit, torch,
                                   geometry,
                                   weights_estimate, rank=rank, world=world, gather=self._gather_ints,
                                   draft_dir=drafter, draft_weights=lambda d: dflash2_weights(d, world, draft_fraction(d, rank, world)),
                                   draft_geometry=draft_geometry)
        self.limit = self.capacity_plan["context_window"]
        capacity = self.capacity_plan["cache_slots"]
        long_context = self.limit > cfg.dense_limit
        # TF_GLM_DRAFT_DUMP=<dir>: DFlash2 requests decode serially and record the drafter's view for the offline
        # simulator (``draft_dump``, ``draft_sim``); TF_GLM_DRAFT_QUANT=minmax|mse: how DFlash2's 4-bit copies are made
        from .draft_dump import dump_dir
        from .qmm import DRAFT_QUANTS, draft_quant

        self.dump = dump_dir() if drafter is not None and not serial_only and parallel == 1 else None
        quant_code = DRAFT_QUANTS.index(draft_quant())
        from tensorfold.cuda.sampling import union_cover

        verify_code = 0
        if parallel > 1:                      # serial and batched verify windows make different collectives
            from .multi import verify_kind

            verify_code = 1 + ("batched", "serial").index(verify_kind())
            from .multi_tune import MultiSettings

            from .sliced_fill import budget_ms

            verify_code = [verify_code] + MultiSettings.from_env().code() + _f64_ints(budget_ms()) + [self.multi_window]
        else:
            verify_code = [verify_code]
        # every rank must run the same calls: refuse to start when they were given different settings
        mine = [int(drafter is not None), capacity, int(long_context), int(serial_only), int(LATENT),
                prefill_rows, self.grid, KVB_KINDS.index(kvb)] + (self.copy.code() if self.copy is not None else [0, 0, 0, 0]) + \
            self.split.code() + [int(self.dump is not None), quant_code, self.shared, int(DRAFT_RING), int(self.mtp_on),
                                 parallel, *verify_code, int(union_cover()), MAX_ROWS, len(WIDE_GRAPHS),
                                 sum(WIDE_GRAPHS), max(WIDE_GRAPHS, default=0),
                                 KV_KINDS.index(kv), len(cfg.eos), sum(cfg.eos)]
        # other conversations' kept prompts get what the window leaves, at most TF_GLM_CACHE_GIB, the same on both ranks
        from .cache_budget import read_budget
        self.cache_entries = int(os.environ.get("TF_GLM_CACHE_ENTRIES", "8"))
        if self.cache_entries < 1:
            raise ValueError("TF_GLM_CACHE_ENTRIES must be positive")
        self.snapshot_budget = read_budget()
        mine += [self.cache_entries, self.snapshot_budget >> 20]
        plan = self.capacity_plan
        wanted = int(float(os.environ.get("TF_GLM_CACHE_GIB", "3")) * 2 ** 30)
        spare = max(0, min(wanted, plan["budget_bytes"] - plan["total_bytes_estimate"]))
        rows = self._gather_ints(mine + [spare >> 20])
        odd = [r for r, row in enumerate(rows) if row[:-1] != rows[0][:-1]]
        if odd:
            raise RuntimeError("the ranks were started with different settings (draft model, context, drafts, "
                               "TF_GLM_LATENT, TF_GLM_PREFILL_ROWS, TF_GLM_PROMPT_GRID, TF_GLM_KVB, TF_GLM_COPY_*, "
                               "TF_GLM_HC_SPLIT*, TF_GLM_PREFILL_OVERLAP, TF_GLM_OVERLAP_PIECES, TF_GLM_DRAFT_DUMP, "
                               "TF_GLM_DRAFT_QUANT, TF_GLM_SHARED_PREFIX, TF_GLM_DRAFT_RING, TF_GLM_MTP, --parallel, "
                               "TF_GLM_MULTI_VERIFY, TF_GLM_MULTI_SAMPLER / _DEPTH / _OVERHEAD_MS / _ASYNC / _LONE / _PROFILE / _WINDOW, "
                               "TENSORFOLD_NUCLEUS_UNION, TF_GLM_MAX_ROWS, TF_GLM_WIDE_GRAPHS, TF_GLM_KV, TF_GLM_ASSISTANT_ENDS): "
                               f"rank 0 {rows[0][:-1]}, " + ", ".join(f"rank {r} {rows[r][:-1]}" for r in odd) +
                               "; pull the draft model on every machine (or pass --drafter none to all) and give "
                               "every rank the same flags")
        self.cache_bytes = min(row[-1] for row in rows) << 20
        plan["kept_bytes"] = self.cache_bytes
        for key in ("serving_peak_bytes_estimate", "total_bytes_estimate"):
            plan[key] = plan[key] + self.cache_bytes
        # --parallel: one pool of per-token caches for every stream and kept prompt: the window's slots, plus what
        # TF_GLM_CACHE_GIB gives kept prompts (their rows in the pool, their KDA states and DFlash2 windows beside it),
        # whole 2,048-token extents; each stream holds up to the window (pool.py, multi.py)
        pool_rows = None
        display = 0
        if parallel > 1:
            from tensorfold.cuda.capacity import config as read_config

            from . import display_kv
            from .pool import ALIGN

            text = read_config(model_dir)
            g = geometry(text)
            per_token = max(1, -(-(g.bytes_at(capacity + 65536) - g.bytes_at(capacity)) // 65536))
            states = int(os.environ.get("TF_GLM_CACHE_ENTRIES", "8")) * (slot_bytes(text, world, rows=0, rank=rank) // 2 +
                                                                          (22 << 20))
            states = max(states, self.snapshot_budget)
            plan["snapshot_budget_bytes"] = states
            # TF_GLM_DISPLAY_KV_MIB: the display reservation's bytes too, which free memory never counted
            display = display_kv.credit()
            pool_rows = (capacity + max(0, self.cache_bytes + display - states) // per_token) // ALIGN * ALIGN
            display_rows = pool_rows - (capacity + max(0, self.cache_bytes - states) // per_token) // ALIGN * ALIGN
            if pool_rows < ALIGN:
                raise ValueError(f"--parallel {parallel}: a pool of {pool_rows} tokens is less than one {ALIGN}-token "
                                 f"extent; free memory or lower --context")
            if cfg.quant in ("exl3", "nvfp4_csf") and os.environ.get("TF_GLM_POOL_TOKENS"):
                requested_pool = int(os.environ["TF_GLM_POOL_TOKENS"])
                if requested_pool % ALIGN or requested_pool < self.limit + MAX_ROWS:
                    raise ValueError("TF_GLM_POOL_TOKENS must be aligned and fit the explicit context")
                # Keep the present 512K pool. Extra allowance is for transient
                # inference buffers, not automatically spent on a larger KV pool.
                pool_rows = min(pool_rows, requested_pool)
            self.limit = min(self.limit, pool_rows - MAX_ROWS)
            self.cache_bytes = 0                 # kept prompts live in the pool
            plan["pool_tokens"] = pool_rows
        if rank == 0 and self.cache_bytes < wanted and parallel == 1:
            print(f"[tensorfold] other conversations' prompts are kept in {self.cache_bytes / 2 ** 30:.1f} GiB, what "
                  f"the {self.limit}-token window leaves (TF_GLM_CACHE_GIB asks {wanted / 2 ** 30:.1f})", flush=True)
        if not self.mtp_on and drafter is None and not serial_only:
            raise ValueError(("TF_GLM_MTP=0 leaves" if cfg.mtp_layers else "this checkpoint has") + " no MTP head and "
                             "no DFlash2 draft model was given, so every round would decode one token: pull the draft "
                             "model on both machines (--drafter), or pass --no-drafts to both for the serial reference")
        if self.split.split and self.split.exchange == "p2p" and hasattr(self.comm, "exchange" if world == 2
                                                                         else "exchange_all"):
            # every rank: open the split's send/receive connections before the weights and the cache pool take the
            # memory; opened at the split's warm-up (after the pool), NCCL's buffer registration could fail with
            # ENOMEM (ibv_reg_mr, issue #36). The connections stay open; nothing they send changes. Past two ranks,
            # one to every peer, as ``hcsplit.RowSplit.warm`` opens them
            a = torch.zeros((1,), dtype=torch.float32, device="cuda")
            if world == 2:
                self.comm.exchange([a], [torch.empty_like(a)], 1 - rank)
            else:
                peers = [p for p in range(world) if p != rank]
                self.comm.exchange_all({p: [a] for p in peers}, {p: [torch.empty_like(a)] for p in peers})
            torch.cuda.synchronize()
        # Open the experimental index send/recv connection before model/KV allocation.
        from . import trial_control as prefill_trials
        if world == 2 and prefill_trials.enabled():
            exchange_warm = torch.zeros(1, dtype=torch.int32, device="cuda")
            self.comm.exchange([exchange_warm], [torch.empty_like(exchange_warm)], 1 - rank)
            torch.cuda.synchronize()
        w = load(model_dir, rank=rank, world=world, mtp=self.mtp_on, nvfp4_rows=prompt_buffer_rows)
        w.comm = self.comm
        self.comm.ready("loading")                   # a peer stuck loading is named, not waited on in NCCL
        self.comm.barrier()
        self.w = w
        if rank == 0 and cfg.mtp_layers and not self.mtp_on:
            print("[tensorfold] the checkpoint's MTP head is not loaded (TF_GLM_MTP=" +
                  (os.environ.get("TF_GLM_MTP", "").strip() or MTP_DEFAULT) + "): " +
                  ("DFlash2 drafts every request" if drafter is not None else "--no-drafts"), flush=True)
        if cfg.quant == "nvfp4_csf":
            checkpoint('target_load_complete')
        self.drafter = None
        if drafter is not None:
            from .dflash2 import Drafter

            self.drafter = Drafter(drafter, w, capacity=pool_rows or capacity, tap_rows=MAX_ROWS, ring=DRAFT_RING)
            from . import draft_checkpoints
            draft_checkpoints.preload(self.drafter,drafter)
        if cfg.quant == "nvfp4_csf":
            checkpoint('draft_loaded_before_kv_and_graphs')
        self.e = Engine(w, capacity=pool_rows or capacity, max_rows=MAX_ROWS, prefill_rows=prefill_rows, graphs=graphs,
                        graph_rows=GRAPH_ROWS + WIDE_GRAPHS, long_context=long_context,
                        taps=self.drafter.tap_layers if self.drafter is not None else (), grid=self.grid,
                        split=self.split, mtp_rows=MTP_ROWS, streams=parallel, pool_rows=pool_rows, kv=kv,
                        display=display)
        if display and rank == 0:
            print(f"[tensorfold] {display >> 20} MiB of the pool in the GPU's display reservation "
                  f"(TF_GLM_DISPLAY_KV_MIB), which free memory does not count: +{display_rows} tokens, "
                  f"{sum(p.spanned for p in self.e.caches.arena.planes)} latent planes in its span", flush=True)
        if self.e.pbuf.split is not None:
            self.e.pbuf.split.warm()                 # both ranks: open the send/receive connection now
            if rank == 0:
                print(f"[tensorfold] {self.split.describe()}", flush=True)
                if not self.e.pbuf.split.applies(prefill_rows):      # e.g. TF_GLM_HC_SPLIT_MIN_ROWS past a chunk
                    print(f"[tensorfold] note: a full {prefill_rows}-row prompt chunk runs unsplit at {world} ranks "
                          "(only shorter ones split)", flush=True)
        c = self.copy
        if rank == 0 and (WIDE_GRAPHS or MAX_ROWS != 16 or (c is not None and (c.reply_match or c.miss_most))):
            print(f"[tensorfold] verify windows up to {MAX_ROWS} rows, CUDA graphs for "
                  f"{', '.join(map(str, GRAPH_ROWS + WIDE_GRAPHS))} rows (TF_GLM_MAX_ROWS, TF_GLM_WIDE_GRAPHS)"
                  + (f"; copies from the reply match {c.reply_match} tokens (TF_GLM_COPY_REPLY_MATCH)"
                     if c is not None and c.reply_match else "")
                  + (f"; {c.miss_most} copied drafts after a copied round that missed (TF_GLM_COPY_MISS_MAX)"
                     if c is not None and c.miss_most else ""), flush=True)
        if rank == 0 and self.dump is not None:
            print(f"[tensorfold] TF_GLM_DRAFT_DUMP: DFlash2 requests decode serially and record the drafter's view "
                  f"into {self.dump} (one .npz a request; replies equal draft: false)", flush=True)
        if rank == 0 and kv != "bf16":
            print("[tensorfold] TF_GLM_KV=fp8: the DSA latent cache and the indexer's pooled keys as e4m3 rows with a "
                  "power-of-two scale each (lossy; drafted replies still equal serial ones)", flush=True)
        if rank == 0 and quant_code:
            print("[tensorfold] TF_GLM_DRAFT_QUANT=mse: DFlash2's 4-bit copies (and an EXL3 draft head) with "
                  "least-squares clipped ranges", flush=True)
        if rank == 0 and self.grid:
            print(f"[tensorfold] prompt chunks and kept prompt states on a {self.grid}-token grid "
                  f"(TF_GLM_PROMPT_GRID)", flush=True)
        if rank == 0 and kda_chunked_on():
            print("[tensorfold] TF_GLM_KDA_CHUNKED=1: KDA prompt chunks in chunked (WY) form, 32-row sub-chunks "
                  "(close to the serial kernel, not its bits; decode keeps the serial kernel)", flush=True)
        # rank 0: the role token whose first appearance ends a prompt's system block (a shared-prefix point)
        self.opener = user_opener(model_dir) if self.shared and rank == 0 else None
        if rank == 0 and self.shared:
            print(f"[tensorfold] shared-prefix prompt states: kept where a prompt stops sharing another and at the end "
                  f"of its system block, when they add {self.shared}+ tokens (TF_GLM_SHARED_PREFIX)", flush=True)
        if self.drafter is not None:
            if graphs:
                self.drafter.capture()
        if cfg.quant == "nvfp4_csf":
            checkpoint('graphs_before_calibration')
        self.costs = self._calibrate()
        if rank == 0:
            c = self.costs
            print(f"[tensorfold] drafter timings (ms, fastest of 7): {c['timed']}", flush=True)
            mtp = (f"; MTP draft {c['mtp']:.2f} (+{c['mtp_step']:.2f} a chained draft, +{c['mtp_row']:.2f} a row)"
                   if w.mtp is not None else "")
            print("[tensorfold] drafter costs (ms): verify " + " ".join(f"{v:.1f}" for v in c["verify"]) + mtp +
                  f"; DFlash2 block {c['block']:.2f} (+{c['taps_row']:.3f} a tap row)", flush=True)
        if cfg.quant == "nvfp4_csf":
            checkpoint('calibration_complete')
        self.vision = None                   # rank 0's image tower (``tensorfold.vision.glm.GlmVision``), --vision
        if vision:
            from tensorfold.vision.glm import GlmVision

            self.vision = GlmVision(model_dir, torch.device("cuda", 0), allow_urls=vision_urls)
            self.vision.warm()
            torch.cuda.empty_cache()
            lim = self.vision.limits
            print(f"[tensorfold] vision: image and video input, a {self.vision.weight_bytes / 2**30:.2f} GiB tower "
                  f"with {workspace / 2**30:.2f} GiB of workspace; at most {lim.image_tokens} tokens a picture, "
                  f"{lim.video_tokens} a clip ({lim.video_frames} frames at {lim.fps:g} a second)"
                  f"{'; https URLs allowed' if vision_urls else ''}", flush=True)
        self.eos = tuple(w.cfg.eos)
        self.model_dir = Path(model_dir)
        self.request = threading.local()    # the calling request's policy and stop-at-EOS (``app.GlmApp``)
        # kept conversations (decode.Snapshot, least recently used first) and the live caches' ids; states and saved rows stay within cache_bytes
        self.cache: list = []
        self.live: list[int] = []
        if parallel > 1:
            from .multi import GlmScheduler, MultiDecoder

            self.multi = MultiDecoder(self, parallel)
            self.multi.model_dir = self.model_dir
            if rank == 0:
                self.scheduler = GlmScheduler(self.multi, max_streams=parallel)
                self.concurrent = True
                print(f"[tensorfold] --parallel {parallel}: up to {parallel} requests decode together, their caches "
                      f"in one pool of {pool_rows} tokens (each up to {self.limit}), their verify windows "
                      f"{self.multi.window} rows together (TF_GLM_MULTI_WINDOW); DFlash2 drafts, prompt chunks of "
                      f"{self.multi.fill_rows} rows while others decode (TF_GLM_FILL_ROWS, TF_GLM_FILL_SHARE "
                      f"{self.multi.share:g}){'; waiting prompts share prompt chunks (TF_GLM_MULTI_PREFILL)' if self.multi.group else ''}; "
                      f"{self.multi.tune.describe()}"
                      f"{f'; prompt layer slices target {self.multi.fill_budget:g} ms' if self.multi.fill_budget else ''}", flush=True)
                if self.multi.row_ms is not None:
                    print("[tensorfold] --parallel batched window ms by rows: " +
                          " ".join(f"{i + 1}:{ms:.1f}" for i, ms in enumerate(self.multi.row_ms)), flush=True)
        if hasattr(self.comm, "settle"):                # startup is over: RoCE gathers go without a barrier first
            self.comm.settle()

    def _calibrate(self) -> dict:
        """Per-piece ms for ``drafter_choice.DrafterChoice``: fastest of interleaved passes, equal on both ranks."""

        import statistics

        import numpy as np

        from .decode import draft, prefill

        torch = self.torch
        e, st = self.e, self.e.st
        rng = np.random.default_rng(0)
        vocab = self.w.cfg.vocab

        def tokens(n: int) -> list[int]:
            return [int(t) for t in rng.integers(0, vocab, n)]

        prefill(e, tokens(64), None, mtp=True, drafter=self.drafter)
        hidden = e.pbuf.fnormed[:8].clone()                 # rows for timing the draft steps
        one, six = tokens(1), tokens(6)
        start = st.mtp_len

        def rewind() -> None:
            st.set_mtp_len(start)
            st.mtp_drafted = 0

        # the verify windows the drafters' rounds use (MTP chains of 3, DFlash2 blocks of 5: the captured graphs);
        # wider (copied) windows are not the choice's, and their eager launches would bend the fitted line
        widest = max(GRAPH_ROWS)
        pieces: dict[str, tuple] = {f"v{r}": (lambda w=tokens(r): e.forward(w), None) for r in range(1, widest + 1)}
        if self.w.mtp is not None:
            pieces["m1"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 1, None), rewind)
            pieces["m3"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 3, None), rewind)
            pieces["m6"] = (lambda: draft(e, hidden[:6], six, st.pos + 1, 1, None), rewind)
        if self.drafter is not None:
            d = self.drafter
            taps = e.tap_rows(8, e.pbuf).clone()
            ctx = d.context_end

            def back() -> None:
                if d.context_end != ctx:
                    d.pos_dev.sub_(d.context_end - ctx)
                    d.context_end = ctx

            pieces["block"] = (lambda: d.propose(one[0], 5, None, 0.0), None)
            pieces["taps8"] = (lambda: d.add_taps(taps), back)
        best = {name: float("inf") for name in pieces}
        for turn in range(9):
            for name, (fn, prep) in pieces.items():
                if prep is not None:
                    prep()
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                if turn >= 2:
                    best[name] = min(best[name], (time.perf_counter() - t) * 1e3)
            rewind()
            if self.drafter is not None:
                back()
        names = list(best)
        mine = torch.tensor([best[n] for n in names], dtype=torch.float32, device="cuda")
        got = torch.empty((self.world * mine.numel(),), dtype=torch.float32, device="cuda")
        self.comm.all_gather(mine, got)
        both = dict(zip(names, got.view(self.world, -1).max(dim=0).values.tolist()))
        e.reset()
        if self.drafter is not None:
            self.drafter.reset()
        rows = list(range(2, widest + 1))
        ys = [both[f"v{r}"] for r in rows]
        slope = statistics.median((ys[j] - ys[i]) / (rows[j] - rows[i]) for i in range(len(rows))
                                  for j in range(i + 1, len(rows)))
        base = statistics.median(y - slope * r for r, y in zip(rows, ys))
        verify = [both["v1"]] + [base + slope * r for r in rows]
        mtp = both.get("m1", 0.0)
        return {"verify": verify, "mtp": mtp, "mtp_step": max((both.get("m3", 0.0) - mtp) / 2, 0.0),
                "mtp_row": max((both.get("m6", 0.0) - mtp) / 5, 0.0), "block": both.get("block", 0.0),
                "taps_row": max(both.get("taps8", 0.0) / 8, 0.0), "timed": {k: round(v, 2) for k, v in both.items()}}

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int32, device="cuda")
        got = torch.empty((self.world * len(values),), dtype=torch.int32, device="cuda")
        self.comm.all_gather(mine, got)
        return got.view(self.world, len(values)).tolist()

    # the idle doorbell: the followers wait for each request on the rendezvous store (no store: no doorbell), not in
    # NCCL; a key a follower and request, which only that follower deletes
    def _store(self):
        return getattr(self.comm, "store", None)

    def _ring(self) -> None:
        store = self._store()
        if store is not None:
            self._bell = getattr(self, "_bell", 0) + 1
            for r in range(1, getattr(self, "world", 2)):
                store.set(bell_key(r, self._bell), b"1")

    def _await_bell(self) -> None:
        store = self._store()
        if store is None:
            return
        from datetime import timedelta

        key = bell_key(getattr(self, "rank", 1), getattr(self, "_bell", 0) + 1)
        while True:
            try:
                store.wait([key], timedelta(hours=1))
                break
            except Exception as e:            # an idle hour: wait again (a lost rank 0 is a connection error instead)
                if "timeout" not in str(e).lower():
                    raise
        store.delete_key(key)
        self._bell = getattr(self, "_bell", 0) + 1

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (a length, then the values, through the all-gather)."""

        torch = self.torch
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32, device="cuda")
        got = torch.empty((self.world,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = (torch.tensor(values, dtype=torch.int32, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int32, device="cuda"))
        allv = torch.empty((self.world * count,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    def _effective(self, code: list[int]) -> list[int]:
        """Resolve auto and MTP policies to the available heads, using EXL3_AUTO for EXL3 with DFlash2 and DFlash2 when MTP is absent."""

        if code[0] == 4 and self.drafter is not None and self.w.cfg.quant == "exl3":
            return encode_policy(EXL3_AUTO)
        if self.w.mtp is None and code[0] in (1, 2, 3, 4, 5):
            return encode_policy(DFLASH_POLICY) if code[0] in (4, 5) else [code[0] + 10] + code[1:]
        return code

    def _drafters(self, code: list[int]) -> tuple[bool, bool, bool]:
        """(auto, MTP drafts, DFlash2 drafts) for a policy code."""

        auto = code[0] in (4, 5)
        dflash = (auto or code[0] // 10 == 1) and self.drafter is not None
        return auto, auto or not dflash, dflash

    def _resume(self, prompt: list[int], code: list[int]):
        """The longest snapshot of a strict prefix of ``prompt`` whose draft caches fit the request's drafters, or of
        the whole prompt when it kept its head's logits row (a replay of an earlier prompt, ``decode.whole``)."""

        _, mtp, dflash = self._drafters(code)
        best = None
        for snap in self.cache:
            fits = (not dflash or snap.drafter_end == len(snap.ids)) and (not mtp or snap.mtp_len >= 0)
            fits = fits and not (self.grid and len(snap.ids) % self.grid)     # prompt chunks start on the grid
            short = len(snap.ids) < len(prompt) or (len(snap.ids) == len(prompt) and snap.head is not None)
            if fits and short and prompt[:len(snap.ids)] == snap.ids and (
                    best is None or len(snap.ids) > len(best.ids)):
                best = snap
        return best

    def _drop(self, snap) -> None:
        """Forget a kept snapshot and free its saved rows now, even while a caller still holds the object."""
        snap.rows, snap.nbytes, snap.drafter_rows = None, 0, None
        self.cache.remove(snap)

    def _remember(self, snap) -> None:
        for c in [c for c in self.cache if c.ids == snap.ids and c is not snap]:
            self._drop(c)
        self.cache[:] = [c for c in self.cache if c is not snap] + [snap]   # a resumed prompt kept again moves last
        dropped = False
        while len(self.cache) > 1 and (len(self.cache) > self.cache_entries or self._held_bytes() > self.cache_bytes):
            self._drop(self.cache[0])
            dropped = True
        if dropped:
            import torch

            torch.cuda.empty_cache()

    def _take_over(self, keep: list[int]) -> None:
        """Save the rows of every kept snapshot the next prefill overwrites, dropping the oldest entries past the memory budget; both ranks decide alike."""
        from .decode import row_bytes, save_rows

        drafter = self._kept_drafter()
        live = self.live
        dropped = False

        def resumes(c) -> bool:
            return len(c.ids) <= len(keep) and keep[:len(c.ids)] == c.ids

        for snap in list(self.cache):
            n = len(snap.ids)
            if snap not in self.cache or snap.rows is not None or resumes(snap):
                continue
            if live[:n] != snap.ids:                  # its rows are already gone: nothing to resume from
                self._drop(snap)
                continue
            need = row_bytes(self.e, snap, drafter=drafter)
            while self._held_bytes() + need > self.cache_bytes:
                old = next((c for c in self.cache if c is not snap and not resumes(c)), None)
                if old is None:
                    break
                self._drop(old)
                dropped = True
            if self._held_bytes() + need > self.cache_bytes:
                self._drop(snap)
                dropped = True
                continue
            save_rows(self.e, snap, drafter=drafter)
        if dropped:
            import torch

            torch.cuda.empty_cache()             # give the freed rows back rather than keep them in torch's pool

    def _kept_drafter(self):
        """The drafter whose context window saved snapshots keep (TF_GLM_SHARED_PREFIX), or None: they drop it."""

        return getattr(self, "drafter", None) if self.shared else None

    def _shared(self, prompt: list[int], hit) -> list[int]:
        """Rank 0: the shared-prefix points a drafted text request keeps (``shared_points``), sent to rank 1."""

        if not self.shared:
            return []
        import numpy as np

        cut = len(hit.ids) if hit is not None else 0
        known = []
        for c in self.cache:                  # a kept prompt's ids as an array, made once (ids never change)
            if getattr(c, "ids_array", None) is None:
                c.ids_array = np.asarray(c.ids, dtype=np.int64)
            known.append(c.ids_array)
        return shared_points(prompt, cut, grid_point(len(prompt), cut, self.grid), self.grid, self.shared, known,
                             self.opener)

    def _held_bytes(self) -> int:
        from .decode import snapshot_bytes

        return sum(snapshot_bytes(c) for c in self.cache)

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable[[list[int]], Any],
             code: list[int], hit, draft: bool, constraint=None, feed=None, shared=()) -> dict[str, Any]:
        """``shared``: rank 0's shared-prefix points (``_shared``), which both ranks keep states at."""

        self.e.constraint, self.e.window = constraint, None       # both ranks walk and mask the same rows
        try:
            return self._run_once(prompt, max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, feed, shared)
        finally:
            self.e.constraint = self.e.window = self.e.vote = None

    def _run_once(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool,
                  on_tokens: Callable[[list[int]], Any], code: list[int], hit, draft: bool,
                  feed=None, shared=()) -> dict[str, Any]:
        from .decode import (DepthPolicy, StopVote, dflash_decode, mtp_decode, prefill, serial_decode, take_snapshot,
                             whole)
        from .drafter_choice import DrafterChoice, auto_decode

        auto, use_mtp, use_dflash = self._drafters(code)
        drafter = self.drafter if use_dflash else None
        t0 = time.perf_counter()
        # a request writes the caches from its resume point: other conversations' rows are saved first, a saved resume point's restored
        from .decode import load_rows

        cut = len(hit.ids) if hit is not None else 0
        self._take_over(list(hit.ids) if hit is not None else [])
        if hit is not None and hit.rows is not None:
            load_rows(self.e, hit, drafter=self._kept_drafter())
            hit.rows, hit.nbytes = None, 0            # live again
        # an image prompt's rows are not its token ids' rows: nothing resumes from them, and they are not kept
        self.live = list(prompt) if feed is None else []
        # a drafted text prompt keeps its state at ``point``: its end, or with TF_GLM_PROMPT_GRID the last grid
        # multiple in it. Short of the end, the prompt prefills to the point as a prompt of that length would (no
        # first token), keeps that state, and resumes from it as a later request would: so any prompt extending
        # prompt[:point] passes through the same state there, fresh or resumed. Both ranks decide alike
        point = grid_point(len(prompt), cut, self.grid) if draft and feed is None else None
        # TF_GLM_SHARED_PREFIX: rank 0's shared-prefix points (past the resume point, short of ``point``) are kept the
        # same way, each prefilled to as a prompt of its length and resumed from, so fresh and resumed prompts pass
        # through the same states there too
        stops = sorted(p for p in set(shared) if point is not None and cut < p < point)
        stats: dict[str, Any] = {"shared": list(stops)} if stops else {}
        if point is not None and point < len(prompt):
            stops.append(point)
        resume, kept = hit, []
        for stop in stops:
            head = list(prompt[:stop])
            prefill(self.e, head, sampling, mtp=use_mtp, drafter=drafter, resume=resume, sample=False)
            resume = take_snapshot(self.e, head, self.e.last_hidden if use_mtp else None, mtp=use_mtp,
                                   drafter=drafter)
            kept.append(resume)
        # a replay of a kept prompt (``decode.whole``: its state and head's logits row) prefills nothing and keeps
        # nothing new; a prompt's own end state keeps the head's row it sampled from, for a later replay
        replay = whole(hit, prompt)
        end = point == len(prompt) and not replay
        first = prefill(self.e, prompt, sampling, mtp=use_mtp, drafter=drafter, resume=resume, feed=feed,
                        keep_head=end)
        prefill_s = time.perf_counter() - t0
        if end:
            snap = take_snapshot(self.e, prompt, self.e.last_hidden if use_mtp else None, mtp=use_mtp,
                                 drafter=drafter)
            snap.head, self.e.head = self.e.head, None
            kept.append(snap)
        if self.shared and draft and feed is None and hit is not None:     # what it resumed from is used again
            self.cache = [c for c in self.cache if c is not hit] + [hit]
        for snap in kept:
            self._remember(snap)
        if not kept and draft and feed is None and hit is not None:     # the resume point is the grid point: newest
            self.cache = [c for c in self.cache if c is not hit] + [hit]
        stats.update(prefill_s=prefill_s, cached=cut)
        # rank 0's on_tokens may ask to stop (the client has gone, a stop string, a gate's cut); rank 1's never does.
        # Every rank hears it on the next round's verify sample (``decode.StopVote``) and ends after that round, so
        # a stopped request frees the engine within a round or two instead of decoding to max_tokens (issue #38)
        on_tokens = StopVote(on_tokens)
        on_tokens([first])
        if max_tokens <= 1 or (stop_eos and first in self.eos):
            return stats
        self.e.vote = on_tokens                     # on every rank alike: each verify gather carries one word more
        policy = decode_policy(code, self.costs)
        copies = None
        if policy is not None and self.copy is not None:     # prompt and pending token, alike on both ranks
            copies = self.copy.drafts(list(prompt) + [first])
        if policy is None:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens)
        elif self.dump is not None and drafter is not None:      # serial's reply, the drafter's view recorded
            from .draft_dump import dump_decode

            meta = None if self.rank else dict(
                prompt_tokens=len(prompt), cached=cut, policy=list(code), images=feed is not None,
                prompt_sha256=hashlib.sha256(json.dumps(list(prompt)).encode()).hexdigest()[:16])
            res = dump_decode(self.e, drafter, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                              out_dir=self.dump if self.rank == 0 else None, meta=meta)
            if self.rank == 0:
                stats["draft_dump"] = res.dump_path
        elif auto:
            greedy = sampling is None or sampling.temperature <= 0
            m_policy = DepthPolicy(3, fixed=True, confidence=0.35) if greedy else DepthPolicy(3, low=0.6, high=0.85)
            _, explore, every, margin, sampled_too = policy
            choice = None
            if drafter is not None and (greedy or sampled_too):
                choice = DrafterChoice(self.costs, first="f" if greedy else "m", explore=explore, every=every,
                                       margin=margin)
            res = auto_decode(self.e, drafter, first, max_tokens, sampling, choice=choice, m_policy=m_policy,
                              f_policy=DepthPolicy(5, fixed=True, confidence=0.3), stop_eos=stop_eos,
                              on_tokens=on_tokens, copies=copies)
        elif use_dflash:
            res = dflash_decode(self.e, self.drafter, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                                on_tokens=on_tokens, copies=copies)
        else:
            res = mtp_decode(self.e, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                             on_tokens=on_tokens, copies=copies)
        # the caches now hold prompt and reply; only prompts are snapshotted, since a later prompt prefills the reply again
        if feed is None:
            self.live = list(prompt) + res.tokens[:self.e.st.pos - len(prompt)]
        stats.update(decode_s=res.seconds, rounds=res.rounds, min_rows=1 + min(res.depths, default=0),
                     tokens_per_round=round((len(res.tokens) - 1) / max(res.rounds, 1), 3),
                     sha256=hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        if on_tokens.stop:                          # ended early on a stop rank 0 asked for (``StopVote``)
            stats["stopped"] = True
        if res.arms:
            stats.update(drafters=res.arms, keeps=res.keeps)
        if policy is not None:
            stats.update(drafted=res.drafted, accepted=res.accepted)
        if copies is not None:
            stats.update(copy_rounds=res.copy_rounds, copy_drafted=res.copy_drafted, copy_accepted=res.copy_accepted)
        if res.stages:
            stats["stages_ms"] = {k: round(v * 1e3, 1) for k, v in res.stages.items()}
        return stats

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 constraint=None, *, vision=None, background: bool = False) -> dict[str, Any]:
        """Mirror one rank-0 request on rank 1; draft=False uses serial decoding and fresh prefill as the reference drafted replies must equal; ``vision``: an image prompt's prepared pictures (``GlmPrepared``); ``background``: under --parallel, after the other requests and yielding to them."""

        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        if not draft or self.serial_only:
            spec = "0"
        else:
            spec = getattr(self.request, "policy", None)
            if not spec:
                default_file=os.environ.get('TF_GLM_POLICY_DEFAULT_FILE')
                spec=Path(default_file).read_text().strip() if default_file else self.policy
        code = self._effective(encode_policy(spec))
        if code[0]==18 and self.scheduler is None:
            raise ValueError('femp needs the shared scheduler (--parallel > 1)')
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        if self.scheduler is not None:          # --parallel: the scheduler's worker runs it with the others
            if vision is not None and self.vision is None:
                raise ValueError("image inputs require starting this server with --vision")
            stats = self.scheduler.submit(list(prompt), max_tokens, sampling, bool(draft) and not self.serial_only,
                                          on_tokens, stop_eos=stop_eos, vision=vision, constraint=constraint,
                                          background=background, glm={"code": code, "spec": spec, "trial": getattr(self.request, "trial", None)})
            stats.update(policy=spec, drafts=draft)
            return stats
        positions: list[int] = []
        feed = None
        if vision is not None:               # encoded here, before rank 1 is told anything, so a failure stops both
            if self.vision is None:
                raise ValueError("image inputs require starting this server with --vision")
            positions = [int(p) for p in vision.positions]
            if positions and (positions[-1] >= len(prompt) or positions != sorted(set(positions))):
                raise ValueError("image rows must sit at increasing positions inside the prompt")
            t = time.perf_counter()
            # the tower's scratch and a prompt chunk's (token selection, sparse partials) take turns in the same
            # memory: torch's cache goes back to the driver before and after the encode
            self.torch.cuda.empty_cache()
            rows = self.vision.features(vision)
            self.torch.cuda.empty_cache()
            if rows.shape[0] != len(positions):
                raise ValueError(f"the tower gave {rows.shape[0]} rows for {len(positions)} image positions")
            feed = VisionFeed(self, positions, rows)
            feed.encode_s = time.perf_counter() - t
        hit = self._resume(list(prompt), code) if draft and feed is None else None
        shared = self._shared(list(prompt), hit) if draft and feed is None else []
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        header = [max_tokens, int(stop_eos), int(draft), len(hit.ids) if hit is not None else 0,
                  seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0), *_f64_ints(sampling.min_p if sampling else 0.0),
                  int(constraint is not None), len(positions),
                  *(shared + [0] * SHARED_MOST)[:SHARED_MOST]] + code
        from tensorfold.engine.grammar import pack

        self._ring()                                   # wakes rank 1, which idles on the store, not in the all-gather
        self._share(header)
        self._share(list(prompt))
        if constraint is not None:                     # the request's grammar: rank 1 compiles the same
            self._share(pack(constraint))
        if positions:
            self._share(positions)
        stats = self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, constraint, feed,
                          shared)
        stats.update(policy=spec, drafts=draft)
        if hasattr(self.comm, "check"):                  # a RoCE wait that timed out inside a graph surfaces here
            self.comm.check()
        if feed is not None:
            stats.update(image_rows=len(positions), encode_s=round(feed.encode_s, 3))
        return stats

    def follow(self) -> None:
        """Ranks past 0: mirror every request rank 0 serves, forever."""

        if self.multi is not None:
            self.multi.follow()
            return
        while True:
            self.follow_one()

    def follow_one(self) -> None:
        """A rank past 0: mirror one request rank 0 serves."""

        from tensorfold.engine.exact_sampling import Sampling

        self._await_bell()
        (max_tokens, stop_eos, draft, cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi, shaped,
         images, *rest) = self._share(None)
        shared, code = [p for p in rest[:SHARED_MOST] if p], rest[SHARED_MOST:]
        prompt = self._share(None)
        packed = self._share(None) if shaped else []
        constraint = None
        if packed:                                  # compiled here as on rank 0
            from tensorfold.engine import grammar

            constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
        feed = VisionFeed(self, self._share(None), None) if images else None
        temperature = _ints_f64(t_lo, t_hi)
        seed = (s_top << 62) | (s_hi << 31) | s_lo
        sampling = (Sampling(seed, temperature, top_k, _ints_f64(p_lo, p_hi), _ints_f64(m_lo, m_hi))
                    if temperature > 0 else None)
        hit = None
        if cached:
            hit = next((c for c in self.cache if len(c.ids) == cached and prompt[:cached] == c.ids), None)
            if hit is None:
                raise RuntimeError(f"rank {self.rank} has no snapshot of the {cached} tokens rank 0 resumes from")
        self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, code, hit, bool(draft),
                  constraint, feed, shared)
        if hasattr(self.comm, "check"):
            self.comm.check()


class VisionFeed:
    """An image prompt's feature rows for each prefill chunk: rank 0 holds them all, and hands a chunk's rows to
    the other ranks through the all-gather as the chunk comes (they send zeros), so no rank holds a second copy."""

    def __init__(self, engine: GlmEngine, positions: list[int], rows) -> None:
        self.engine, self.positions, self.rows = engine, positions, rows
        self.width = engine.w.cfg.hidden
        self.encode_s = 0.0

    def chunk(self, start: int, count: int):
        """(row indices within the chunk, their features) for prompt rows start .. start + count, or None."""

        import bisect

        torch = self.engine.torch
        lo = bisect.bisect_left(self.positions, start)
        hi = bisect.bisect_left(self.positions, start + count)
        if lo == hi:
            return None
        n = hi - lo
        mine = (self.rows[lo:hi].contiguous() if self.rows is not None else
                torch.zeros((n, self.width), dtype=torch.bfloat16, device="cuda"))
        both = torch.empty((self.engine.world * n, self.width), dtype=torch.bfloat16, device="cuda")
        self.engine.comm.all_gather(mine, both)
        index = torch.tensor([p - start for p in self.positions[lo:hi]], dtype=torch.int64, device="cuda")
        return index, both[:n]
