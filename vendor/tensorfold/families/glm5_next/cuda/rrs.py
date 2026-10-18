"""Exact rejection speculative sampling over a sparse DFlash2 proposal.

P is the target's existing temperature/top-k/top-p/min-p distribution. Q is
the normalized top-16 selector distribution, conditional on the draft prefix.
Stopping before a current draw is allowed; a draw-dependent stop must include
that draw. An independent pilot may instead decide K before the actual chain.
No weight changes. This preserves P, not the old seed-to-text mapping.
"""
from __future__ import annotations

import numpy as np
from tensorfold.engine.exact_sampling import uniform

# Independent random domains; position-keyed randomness makes discarded suffixes
# irrelevant. Each domain is consumed at most once at a committed position.
PROPOSE = 0x243F6A8885A308D3
ACCEPT = 0x13198A2E03707344
TARGET = 0xA4093822299F31D0
PILOT = 0x082EFA98EC4E6C89


def softmax(score):
    score = np.asarray(score, dtype=np.float64)
    out = np.exp(score - score.max())
    return out / out.sum()


def distribution(values, ids, sampling):
    """Same filtering and tie order as exact_sampling.choose; then normalize."""
    order = np.lexsort((ids, -values))
    k = max(1, min(int(sampling.top_k) if sampling.top_k else len(ids), len(ids)))
    ids = np.asarray(ids, dtype=np.int64)[order][:k]
    scaled = np.asarray(values)[order][:k].astype(np.float64) / max(float(sampling.temperature), 1e-6)
    probs = softmax(scaled)
    if 0.0 < sampling.top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(probs), sampling.top_p) + 1)
        ids, scaled = ids[:keep], scaled[:keep]
    if sampling.min_p > 0.0:
        keep = int((scaled >= scaled[0] + sampling.min_log).sum())
        ids, scaled = ids[:keep], scaled[:keep]
    return ids, softmax(scaled)


def draw(ids, probs, seed, position, domain):
    positive = probs > 0
    ids, probs = ids[positive], probs[positive]
    if not len(ids):
        raise RuntimeError('RRS distribution has no mass')
    keys = np.log(probs) - np.log(-np.log(uniform(seed ^ domain, position, ids)))
    return int(ids[int(np.argmax(keys))])


def selector(drafter, ids, values, proj, previous, temperature, edge_scale):
    edge = drafter.succ[ids].astype(np.float64) @ (drafter.pred[previous].astype(np.float64) * proj)
    return (values + edge_scale * edge) / float(temperature)


def confidence(score, gumbel, index, beta):
    return float(softmax((score + gumbel) / beta)[index])


def pilot_length(drafter, tokens, values, proj, anchor, first, sampling, noise, floor, beta):
    """Original exclusive FNC walk in an independent RNG domain, used only for K."""
    previous, reach, kept = anchor, 1.0, 0
    for i in range(len(tokens)):
        ids = np.asarray(tokens[i], dtype=np.int64)
        score = selector(drafter, ids, values[i], proj[i], previous, sampling.temperature, 0.6)
        g = -np.log(-np.log(uniform(sampling.seed ^ PILOT, first+i, ids)))
        index = int(np.argmax(score + noise*g))
        reach *= confidence(score,g,index,beta)
        if i and reach < floor:
            break
        kept += 1
        previous = int(ids[index])
    return kept


def proposal(drafter, tokens, values, proj, anchor, first, sampling, noise=1.0,
             mode=0, floor=0.3, beta=2.0):
    """0 fixed; 1 inclusive FNC; 2 independent pilot FNC; 3 inclusive native edge.

    Inclusion of draw i is determined by the prefix through i-1. Stopping after
    draw i may depend on it. Therefore Q_i remains the actual conditional law of
    every verified proposal. Mode 2's pilot is independent of all proposal and
    acceptance RNG domains, including when it inspects future positions.
    """
    if sampling is None or sampling.temperature <= 0 or noise <= 0:
        raise ValueError('RRS proposal needs positive temperature and noise')
    if mode not in (0,1,2,3) or not 0 <= floor <= 1 or beta <= 0:
        raise ValueError('Invalid adaptive RRS rule')
    depth = len(tokens)
    if mode == 2:
        depth = pilot_length(drafter,tokens,values,proj,anchor,first,sampling,noise,floor,beta)
    output, laws = [], []
    previous, reach = anchor, 1.0
    for i in range(depth):
        ids = np.asarray(tokens[i], dtype=np.int64)
        score = selector(drafter,ids,values[i],proj[i],previous,sampling.temperature,1.0 if mode==3 else 0.6)
        probs = softmax(score/noise)
        previous = draw(ids,probs,sampling.seed,first+i,PROPOSE)
        output.append(previous)
        laws.append((ids,probs))
        if mode in (1,3):
            g = -np.log(-np.log(uniform(sampling.seed ^ PROPOSE,first+i,ids)))
            index = int(np.flatnonzero(ids==previous)[0])
            reach *= confidence(score,g,index,beta)
            # Crucial: this candidate remains in output/laws when it triggers stop.
            if reach < floor:
                break
    return output,laws


def accept_or_correct(pids, p, qids, q, candidate, seed, position):
    """One exact maximal coupling step, including absent P/Q support."""
    qmatch = np.flatnonzero(qids == candidate)
    if len(qmatch) != 1 or q[qmatch[0]] <= 0:
        raise RuntimeError('RRS candidate is not in its proposal law')
    pmatch = np.flatnonzero(pids == candidate)
    px = float(p[pmatch[0]]) if len(pmatch) else 0.0
    qx = float(q[qmatch[0]])
    u = float(uniform(seed ^ ACCEPT, position, np.array([0], dtype=np.int64))[0])
    if u < min(1.0, px / qx):
        return candidate, True
    # Sparse P is complete after top-k. All residual mass is in P's support.
    aligned_q = (pids[:, None] == qids[None, :]) @ q
    residual = np.maximum(p - aligned_q, 0.0)
    if not residual.sum() > 0:
        raise RuntimeError('RRS rejected without positive residual mass')
    token = draw(pids, residual, seed, position, TARGET)
    if token == candidate:
        raise RuntimeError('RRS correction selected the rejected candidate')
    return token, False


def verify(values, ids, positions, sampling, candidates, laws):
    """Return the committed prefix plus correction/bonus; unused rows are -1.

    The normal engine equality loop then commits accepted drafts + one token.
    EOS may shorten this further. Never inspect rows beyond the first mismatch.
    """
    if len(values) != len(candidates) + 1 or len(laws) != len(candidates):
        raise RuntimeError('RRS proposal/verification row mismatch')
    output = [-1] * len(values)
    for i, (candidate, (qids, q)) in enumerate(zip(candidates, laws)):
        pids, p = distribution(values[i], ids[i], sampling)
        token, accepted = accept_or_correct(pids, p, qids, q, candidate, sampling.seed, int(positions[i]))
        output[i] = token
        if not accepted:
            return output
    i = len(candidates)
    pids, p = distribution(values[i], ids[i], sampling)
    output[i] = draw(pids, p, sampling.seed, int(positions[i]), TARGET)
    return output
