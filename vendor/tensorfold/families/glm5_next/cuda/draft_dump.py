"""TF_GLM_DRAFT_DUMP=<dir>: teacher-forced DFlash2 records for the offline draft simulator (``draft_sim``).

With it set (on both ranks: the engine refuses to start otherwise), a request that would draft with DFlash2 decodes
one token a round instead, through the same one-row verify as ``serial_decode``, so its reply is serial's
(``draft: false``). Before each round it runs the DFlash2 block for the pending token as ``dflash_decode`` would and
records, for the block's positions: the merged candidates and their logits, the selector's edge scores between
consecutive candidates, the request's keyed Gumbel draws for them, the drafter's projected rows, and the chain the
production rule (``fc5:0.3``) would propose; after the verify, the target's top-32 logits and the committed token.
Both ranks run every call (the block and the target's top-32 gather are collectives); rank 0 writes one ``.npz`` per
request into <dir>. Off (unset), nothing here runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

from . import dflash2
from .decode import DecodeResult, _sync, commit, stopped

DUMP_ENV = "TF_GLM_DRAFT_DUMP"
TARGET_K = 32          # the target's top candidates kept a row (>= top_k 20 + MARGIN 8, so its sample re-derives)
VERSION = 1
PROD_MOST, PROD_CONFIDENCE = 5, 0.3      # engine.DFLASH_POLICY, the chain recorded as ``prod_chain``

_count = 0


def dump_dir(env: dict | None = None) -> Path | None:
    """TF_GLM_DRAFT_DUMP's directory, or None when unset or empty (off)."""

    value = (os.environ if env is None else env).get(DUMP_ENV, "")
    return Path(value) if value else None


def edges(drafter, tokens: np.ndarray, proj: np.ndarray, anchor: int) -> tuple[np.ndarray, np.ndarray]:
    """The selector edges Drafter.chain computes, for every pair it could meet: (pending -> depth-0 candidates [K],
    depth d-1 candidate a -> depth d candidates [D - 1, K, K]); each the chain's own float64 expression (same bits)."""

    depth, k = tokens.shape
    e0 = drafter.succ[tokens[0]].astype(np.float64) @ (drafter.pred[anchor].astype(np.float64) * proj[0])
    out = np.zeros((max(depth - 1, 0), k, k), dtype=np.float64)
    for d in range(1, depth):
        succ = drafter.succ[tokens[d]].astype(np.float64)
        for a in range(k):
            out[d - 1, a] = succ @ (drafter.pred[int(tokens[d - 1, a])].astype(np.float64) * proj[d])
    return e0, out


def target_top(w, row: torch.Tensor, k: int = TARGET_K) -> tuple[np.ndarray, np.ndarray, float]:
    """The target's top-k over the whole vocabulary of one logits row (this rank's slice [1, V/world]) by (value desc,
    id asc), and the row's logsumexp; gathered from every rank alike."""

    k = min(k, row.shape[1])
    x = row.float()
    vals, ids = torch.topk(x, k, dim=-1)
    lse = torch.logsumexp(x, dim=-1, keepdim=True)
    ids = (ids + w.vocab_offset).to(torch.int32)
    packed = torch.cat([vals, ids.view(torch.float32), lse], dim=1).contiguous()
    if w.comm is None:
        g = packed.view(1, -1).cpu()
    else:
        got = torch.empty((w.world * packed.numel(),), dtype=torch.float32, device=packed.device)
        w.comm.all_gather(packed.view(-1), got)
        g = got.view(w.world, -1).cpu()
    values = g[:, :k].reshape(-1).numpy().astype(np.float64)
    tokens = g[:, k:2 * k].contiguous().view(torch.int32).reshape(-1).numpy().astype(np.int64)
    lses = g[:, 2 * k].numpy().astype(np.float64)
    order = np.lexsort((tokens, -values))[:k]
    return tokens[order], values[order], float(np.logaddexp.reduce(lses))


@torch.no_grad()
def dump_decode(e, drafter, pending: int, count: int, sampling: Sampling | None, *, out_dir: Path | None,
                stop_eos: bool = False, on_tokens=None, meta: dict | None = None) -> DecodeResult:
    """``serial_decode``'s tokens (one-row verifies), with the DFlash2 block run and recorded before each; rank 0
    passes ``out_dir`` and writes the records there, every rank runs the same calls."""

    w, st, b = e.w, e.st, e.buf
    depth = drafter.block - 1
    sampled = sampling is not None and sampling.temperature > 0
    out = [pending]
    rec: dict[str, list] = {k: [] for k in ("first", "cand_ids", "cand_vals", "proj", "noise", "edge0", "edges",
                                            "prod_chain", "tgt_ids", "tgt_vals", "tgt_lse")}
    stages = {"block": 0.0, "forward": 0.0, "sample": 0.0, "commit": 0.0}
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos) and not stopped(e):
        t0 = time.perf_counter()
        if drafter.context_end != st.pos:
            raise RuntimeError(f"draft dump: the drafter's context ends at {drafter.context_end}, "
                               f"the reply at {st.pos}")
        first = drafter.context_end + 1
        tokens, values, proj = drafter.candidates(out[-1], depth)
        e0, ed = edges(drafter, tokens, proj, out[-1])
        noise = (-np.log(-np.log(uniform_rows(sampling.seed, first + np.arange(depth), tokens)))
                 if sampled else None)
        most = min(PROD_MOST, depth)
        chain = drafter.chain(tokens[:most], values[:most], proj[:most], out[-1], first, sampling, PROD_CONFIDENCE)
        t1 = time.perf_counter()
        logits = e.forward(e.verify_window([out[-1]]))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        top_ids, top_vals, lse = target_top(w, logits[:1])
        e.follow([tok])
        t3 = time.perf_counter()
        commit(w, st, b, 1, 1)
        drafter.add_taps(e.tap_rows(1))
        t4 = time.perf_counter()
        stages["block"] += t1 - t0
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
        rec["first"].append(first)
        rec["cand_ids"].append(tokens.astype(np.int32))
        rec["cand_vals"].append(values.astype(np.float32))       # float32 logits widened by candidates(): exact
        rec["proj"].append(proj.astype(np.float32))
        if noise is not None:
            rec["noise"].append(noise)
        rec["edge0"].append(e0)
        rec["edges"].append(ed)
        rec["prod_chain"].append(chain + [-1] * (depth - len(chain)))
        rec["tgt_ids"].append(top_ids.astype(np.int32))
        rec["tgt_vals"].append(top_vals.astype(np.float32))
        rec["tgt_lse"].append(lse)
        out.append(tok)
        if on_tokens is not None:
            on_tokens([tok])
    _sync(w)
    res = DecodeResult(out, time.perf_counter() - start, len(out) - 1, stages=stages)
    if out_dir is not None:
        info = dict(meta or {})
        info.update(version=VERSION, block=drafter.block, selector_top_k=drafter.top_k, edge=dflash2.EDGE,
                    noise=dflash2.NOISE, prod_most=PROD_MOST, prod_confidence=PROD_CONFIDENCE, stop_eos=bool(stop_eos),
                    eos=[int(t) for t in w.cfg.eos], count=int(count), steps=len(out) - 1,
                    seed=int(sampling.seed) if sampling is not None else 0,
                    temperature=float(sampling.temperature) if sampling is not None else 0.0,
                    top_k=int(sampling.top_k) if sampling is not None else 0,
                    top_p=float(sampling.top_p) if sampling is not None else 1.0,
                    min_p=float(sampling.min_p) if sampling is not None else 0.0,
                    sha256=hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16])
        res.dump_path = str(write(out_dir, info, out, rec, depth, drafter.top_k))
    return res


def write(out_dir: Path, meta: dict, tokens: Sequence[int], rec: dict, depth: int, k: int) -> Path:
    """One request's records as a compressed .npz (written to a temporary name, then renamed)."""

    global _count
    _count += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rec["first"])

    def stack(key, shape, dtype):
        return np.stack(rec[key]).astype(dtype) if rec[key] else np.zeros((0, *shape), dtype=dtype)

    arrays = dict(meta=np.array(json.dumps(meta)), tokens=np.asarray(tokens, dtype=np.int64),
                  first=np.asarray(rec["first"], dtype=np.int64), cand_ids=stack("cand_ids", (depth, k), np.int32),
                  cand_vals=stack("cand_vals", (depth, k), np.float32),
                  proj=np.stack(rec["proj"]) if rec["proj"] else np.zeros((0, depth, 0), np.float32),
                  noise=stack("noise", (depth, k), np.float64) if rec["noise"] else np.zeros((0,), np.float64),
                  edge0=stack("edge0", (k,), np.float64), edges=stack("edges", (max(depth - 1, 0), k, k), np.float64),
                  prod_chain=stack("prod_chain", (depth,), np.int32), tgt_ids=stack("tgt_ids", (TARGET_K,), np.int32),
                  tgt_vals=stack("tgt_vals", (TARGET_K,), np.float32),
                  tgt_lse=np.asarray(rec["tgt_lse"], dtype=np.float64).reshape(n))
    name = f"dump-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{_count:04d}"
    tmp = out_dir / f".{name}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    path = out_dir / f"{name}.npz"
    os.replace(tmp, path)
    return path
