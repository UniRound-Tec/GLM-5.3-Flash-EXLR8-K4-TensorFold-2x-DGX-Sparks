"""Offline DFlash2 draft-policy simulator over TF_GLM_DRAFT_DUMP records (``draft_dump``): pure numpy, CPU only.

A dump holds, for every committed position of a serial (``draft: false``-equal) reply, what the DFlash2 block proposed
there (its merged top-k candidates, their logits, the selector's edge scores, the keyed Gumbel draws of the request)
and what the target committed. Sampling is keyed by (seed, position, token), so a round that starts at position i and
proposes a chain keeps exactly the drafts that equal the committed tokens i + 1, i + 2, ... (up to the first mismatch)
plus one sampled token. A policy's rounds are therefore replayed exactly (up to the drafter's numerics: the dump's
context taps come from one-row verifies, a live round's from wider windows), and a cost table (verify window V(R)
for R = 1..8 rows, the block, per-round overhead) turns them into ms a round and tokens a second.

Policies (``parse_policy``):
  fc<N>:<p>        today's rule: up to N drafts while the product of the chain's (noise-free) confidences holds p
  f<N>             N drafts every round
  fa[:LOW:HIGH]    DepthPolicy's adaptive 1-3 drafts
  nc<N>:<p>[:B]    noise-aware product: each draft's confidence is softmax((score + g) / B) at the pick, where g are the
                   target's own keyed Gumbel draws (known before the verify); B defaults to --beta
  cost<N>:<pred>   the depth k <= N maximizing E[tokens | k] / (V(k + 1) + block + overhead), E from the per-draft
                   acceptance predictor <pred>: conf (today's confidence), noisy[:B], const (per-depth rates), logit
  oracle<N>        hindsight: the per-round depths that minimize the reply's total time (an upper bound)
Any policy takes ``@noise=W,edge=E,trunc=1`` to change the chain's pick: the weight of the keyed noise (0.7 today;
1.0 is the drafter's estimate of the target's keyed sample), of the selector edges (0.6), and whether picks are cut to
the drafter's own top-k / top-p nucleus like the target's sampler.

Run: python -m tensorfold.families.glm5_next.cuda.draft_sim DUMP.npz ... [--policy SPEC ...] [--train DUMP ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

# the measured EXL3 + DFlash2 costs (ms): verify windows of 1..8 rows and one DFlash2 block
VERIFY_MS = (29.2, 35.6, 42.6, 48.8, 53.6, 58.9, 64.5, 66.0)
BLOCK_MS = 3.08
EDGE = 0.6          # dflash2.EDGE
NOISE = 0.7         # dflash2.NOISE
MARGIN = 8          # exact_sampling.MARGIN


# -- dumps ------------------------------------------------------------------------------------------------------------
@dataclass
class Dump:
    """One request's teacher-forced records; step i: pending token tokens[i], drafts for tokens[i + 1 ...]."""

    meta: dict
    tokens: np.ndarray          # [n + 1] int64: the reply (the first token, then one a step)
    first: np.ndarray           # [n] int64: position of draft 0 (the drafter's context_end + 1), = the target's
    cand_ids: np.ndarray        # [n, D, K] int64: merged candidates by (value desc, id asc), D = block - 1
    cand_vals: np.ndarray       # [n, D, K] float64 (float32 logits)
    edge0: np.ndarray           # [n, K] float64: selector edge, pending token -> depth-0 candidates
    edges: np.ndarray           # [n, D - 1, K, K] float64: edge from depth d-1 candidate a to depth d candidate b
    noise: np.ndarray | None    # [n, D, K] float64 keyed Gumbel draws (None: greedy)
    tgt_ids: np.ndarray         # [n, T] int64: the target's top-T at the committed row
    tgt_vals: np.ndarray        # [n, T] float64
    tgt_lse: np.ndarray         # [n] float64: logsumexp of the whole row (raw logits)
    prod_chain: np.ndarray      # [n, D] int64: the engine's own chain under its production rule, -1 padded
    path: str = ""
    proj: np.ndarray | None = None   # [n, D, rank] float32: the selector's projected rows (edges already hold them)

    @property
    def n(self) -> int:
        return int(self.first.shape[0])

    @property
    def depth(self) -> int:
        return int(self.cand_ids.shape[1])

    @property
    def sampled(self) -> bool:
        return self.noise is not None

    @property
    def temperature(self) -> float:
        return float(self.meta["temperature"]) if self.sampled else 1.0

    @property
    def eos(self) -> frozenset:
        return frozenset(int(t) for t in self.meta.get("eos", ())) if self.meta.get("stop_eos") else frozenset()

    def room(self, i: int) -> int:
        """Drafts a round at step i may propose (dflash_decode's count - len(out)): up to the request's max_tokens
        when the reply ended at an EOS (drafts past it are rejected there), else up to the reply's end."""

        if self.eos and int(self.tokens[-1]) in self.eos and "count" in self.meta:
            return int(self.meta["count"]) - (i + 1)
        return self.n - i

    def accepted(self, i: int, drafts: Sequence[int]) -> int:
        """Drafts a round at step i keeps: equal to the committed tokens, a matched EOS not counted (as the loop)."""

        k = 0
        for d in drafts:
            if i + 1 + k > self.n:
                break
            t = int(self.tokens[i + 1 + k])
            if d != t or t in self.eos:
                break
            k += 1
        return k


def load_dump(path: str | Path) -> Dump:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        noise = z["noise"].astype(np.float64) if "noise" in z.files and z["noise"].size else None
        return Dump(meta, z["tokens"].astype(np.int64), z["first"].astype(np.int64), z["cand_ids"].astype(np.int64),
                    z["cand_vals"].astype(np.float64), z["edge0"].astype(np.float64), z["edges"].astype(np.float64),
                    noise, z["tgt_ids"].astype(np.int64), z["tgt_vals"].astype(np.float64),
                    z["tgt_lse"].astype(np.float64), z["prod_chain"].astype(np.int64), str(path),
                    z["proj"] if "proj" in z.files and z["proj"].size else None)


def load_dumps(paths: Sequence[str]) -> list[Dump]:
    files: list[str] = []
    for p in paths:
        q = Path(p)
        files += sorted(str(f) for f in q.glob("*.npz")) if q.is_dir() else sorted(glob.glob(p)) or [p]
    return [load_dump(f) for f in files]


# -- the chain --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PickRule:
    edge: float = EDGE
    noise: float = NOISE
    trunc: bool = False         # cut picks to the drafter's own top-k / top-p nucleus (the target sampler's shape)


RULE = PickRule()      # today's pick


@dataclass
class Trace:
    """A chain walked to its full depth (no stop): per depth the pick and what a stop rule or predictor reads."""

    tokens: list[int]
    j: list[int]
    score: list[np.ndarray]      # (logits + edge * EDGE) / temperature, as Drafter.chain
    keep: list[np.ndarray]       # bool mask of pickable candidates (all, or the nucleus with trunc)
    noise: np.ndarray | None


def _nucleus(score: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    order = np.argsort(-score, kind="stable")
    keep = np.zeros(score.shape, dtype=bool)
    k = min(top_k, len(score)) if top_k else len(score)
    s = score[order[:k]]
    p = np.exp(s - s.max())
    p /= p.sum()
    kept = int((np.cumsum(p) < top_p).sum()) + 1 if 0.0 < top_p < 1.0 else k
    keep[order[:kept]] = True
    return keep


def walk(d: Dump, i: int, depth: int, rule: PickRule = RULE, confidence: float = 0.0) -> Trace:
    """Drafter.chain at step i, bit for bit with the default rule (same float64 operations in the same order);
    ``confidence`` > 0 stops as fc does (product of noise-free confidences, the first draft always kept)."""

    depth = min(depth, d.depth)
    temp = d.temperature
    tr = Trace([], [], [], [], d.noise[i, :depth] if d.sampled else None)
    chain = 1.0
    prev = -1
    top_k = int(d.meta.get("top_k") or 0)
    top_p = float(d.meta.get("top_p", 1.0))
    for k in range(depth):
        edge = d.edge0[i] if k == 0 else d.edges[i, k - 1, prev]
        score = (d.cand_vals[i, k] + rule.edge * edge) / temp
        pick = score + rule.noise * tr.noise[k] if tr.noise is not None else score
        keep = _nucleus(score, top_k, top_p) if rule.trunc and d.sampled else np.ones(score.shape, dtype=bool)
        if rule.trunc and d.sampled:
            pick = np.where(keep, pick, -np.inf)
        j = int(np.argmax(pick))
        if confidence > 0:
            p = np.exp(score - score.max())
            chain *= float(p[j] / p.sum())
            if k > 0 and chain < confidence:
                break
        prev = j
        tr.tokens.append(int(d.cand_ids[i, k, j]))
        tr.j.append(j)
        tr.score.append(score)
        tr.keep.append(keep)
    return tr


FEATURES = ("conf", "log_conf", "log_cum_conf", "noisy_prob", "log_noisy_prob", "noisy_margin", "top_gap", "depth",
            "bias")


def features(tr: Trace, beta: float = 1.0) -> np.ndarray:
    """[depth, len(FEATURES)] for a trace's drafts (see FEATURES); greedy: noise-aware columns equal the plain ones."""

    rows = []
    cum = 0.0
    for k, (score, keep, j) in enumerate(zip(tr.score, tr.keep, tr.j)):
        s = np.where(keep, score, -np.inf)
        p = np.exp(s - s.max())
        conf = float(p[j] / p.sum())
        cum += math.log(max(conf, 1e-300))
        z = s + tr.noise[k] if tr.noise is not None else s
        q = _softmax_at(z / beta, j)
        others = np.delete(z, j)
        margin = float(z[j] - others.max()) if others.size and np.isfinite(others.max()) else 30.0
        top = np.sort(s[np.isfinite(s)])[::-1]
        gap = float(top[0] - top[1]) if top.size > 1 else 30.0
        rows.append((conf, math.log(max(conf, 1e-300)), cum, q, math.log(max(q, 1e-300)), min(margin, 30.0),
                     min(gap, 30.0), float(k), 1.0))
    return np.array(rows, dtype=np.float64).reshape(len(rows), len(FEATURES))


def _softmax_at(z: np.ndarray, j: int) -> float:
    e = np.exp(z - z.max())
    return float(e[j] / e.sum())


# -- predictors of a draft's acceptance (given the drafts before it were accepted) -----------------------------------
class Predictor:
    name = "?"

    def fit(self, X: np.ndarray, y: np.ndarray) -> Predictor:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConfPredictor(Predictor):
    """Today's belief: a draft is kept with its noise-free confidence."""

    name = "conf"

    def predict(self, X):
        return X[:, FEATURES.index("conf")]


class NoisyPredictor(Predictor):
    """softmax((score + g) / beta) at the pick: the target's scaled logits taken as the drafter's plus iid Gumbel(beta)
    errors, the keyed draws g known, make exactly this the probability that the target's keyed sample is the pick.
    ``fit`` picks beta on a grid by log loss (its features must be made with beta 1)."""

    name = "noisy"

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def predict(self, X):
        return X[:, FEATURES.index("noisy_prob")]


class ConstPredictor(Predictor):
    """The mean acceptance at each depth (a baseline that knows nothing of the round)."""

    name = "const"

    def __init__(self) -> None:
        self.rates = np.full(8, 0.5)

    def fit(self, X, y):
        dep = X[:, FEATURES.index("depth")].astype(int)
        for k in range(len(self.rates)):
            m = dep == k
            self.rates[k] = (y[m].sum() + 1) / (m.sum() + 2)
        return self

    def predict(self, X):
        return self.rates[np.minimum(X[:, FEATURES.index("depth")].astype(int), len(self.rates) - 1)]


class LogisticPredictor(Predictor):
    """Logistic regression (Newton / IRLS, a small L2) on chosen FEATURES columns."""

    name = "logit"

    def __init__(self, cols: Sequence[str] = ("log_conf", "log_noisy_prob", "noisy_margin", "top_gap", "depth", "bias"),
                 l2: float = 1e-3) -> None:
        self.cols = tuple(cols)
        self.idx = [FEATURES.index(c) for c in self.cols]
        self.l2 = l2
        self.w = np.zeros(len(self.idx))

    def fit(self, X, y):
        A = X[:, self.idx]
        w = np.zeros(A.shape[1])
        for _ in range(50):
            p = 1 / (1 + np.exp(-np.clip(A @ w, -40, 40)))
            g = A.T @ (p - y) + self.l2 * w
            H = (A * (p * (1 - p))[:, None]).T @ A + self.l2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-9:
                break
        self.w = w
        return self

    def predict(self, X):
        return 1 / (1 + np.exp(-np.clip(X[:, self.idx] @ self.w, -40, 40)))

    def describe(self) -> str:
        return ", ".join(f"{c} {v:+.3f}" for c, v in zip(self.cols, self.w))


def training_rows(dumps: Sequence[Dump], rule: PickRule = RULE, depth: int = 7,
                  beta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) over every step of the dumps: each draft of the full chain while the drafts before it were accepted,
    labelled 1 when it equals the committed token (a round's conditional acceptance, the quantity E[tokens] needs)."""

    X, y = [], []
    for d in dumps:
        for i in range(d.n):
            tr = walk(d, i, min(depth, d.n - i), rule)
            if not tr.tokens:
                continue
            F = features(tr, beta)
            kept = d.accepted(i, tr.tokens)
            rows = min(kept + 1, len(tr.tokens))
            X.append(F[:rows])
            y.append((np.arange(rows) < kept).astype(np.float64))
    if not X:
        return np.zeros((0, len(FEATURES))), np.zeros(0)
    return np.concatenate(X), np.concatenate(y)


def fit_beta(dumps: Sequence[Dump], rule: PickRule, depth: int = 7,
             grid: Sequence[float] = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)) -> tuple[float, float]:
    """The NoisyPredictor beta with the least log loss on the dumps: (beta, log loss)."""

    best = (1.0, math.inf)
    for b in grid:
        X, y = training_rows(dumps, rule, depth, b)
        if not len(y):
            break
        loss = log_loss(NoisyPredictor(b).predict(X), y)
        if loss < best[1]:
            best = (b, loss)
    return best


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(y) else float("nan")


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    """Log loss, Brier score and the ranking AUC of acceptance predictions."""

    out = {"rows": len(y), "rate": float(y.mean()) if len(y) else float("nan"), "log_loss": log_loss(p, y),
           "brier": float(((p - y) ** 2).mean()) if len(y) else float("nan")}
    pos, neg = p[y > 0.5], p[y < 0.5]
    if len(pos) and len(neg):
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
        out["auc"] = float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return out


# -- costs and policies -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Costs:
    verify: tuple[float, ...] = VERIFY_MS    # V(R), R = 1..len
    block: float = BLOCK_MS                  # one DFlash2 block (a round that drafts)
    overhead: float = 0.0                    # every round: sampling, commit, host work
    taps_row: float = 0.0                    # each kept row's context update

    def round_ms(self, drafted: int, kept: int, drafting: bool) -> float:
        return self.verify[drafted] + (self.block if drafting else 0.0) + self.overhead + self.taps_row * kept


COSTS = Costs()


class Policy:
    """A round's drafts at step i (at most ``room``), then what the round kept."""

    spec = "?"
    rule = RULE

    def most(self) -> int:
        raise NotImplementedError

    def propose(self, d: Dump, i: int, room: int, costs: Costs) -> list[int]:
        raise NotImplementedError

    def record(self, drafted: int, accepted: int) -> None:
        pass


class ProductPolicy(Policy):
    """fc<N>:<p> (``conf``) and f<N> (p = 0): Drafter.propose with DepthPolicy(N, fixed=True, confidence=p)."""

    def __init__(self, most: int, confidence: float, rule: PickRule = RULE, spec: str = "") -> None:
        self.n, self.confidence, self.rule, self.spec = most, confidence, rule, spec

    def most(self):
        return self.n

    def propose(self, d, i, room, costs):
        depth = min(self.n, room)
        return walk(d, i, depth, self.rule, self.confidence).tokens if depth > 0 else []


class AdaptivePolicy(ProductPolicy):
    """fa[:LOW:HIGH]: DepthPolicy's running acceptance rate chooses 1, 2 or 3 drafts."""

    def __init__(self, low: float = 0.8, high: float = 0.9, rule: PickRule = RULE, spec: str = "") -> None:
        super().__init__(3, 0.0, rule, spec)
        self.low, self.high, self.rate = low, high, 0.8
        self.first = True

    def propose(self, d, i, room, costs):
        n = max(1, min(3, 1 if self.rate < self.low else 2 if self.rate < self.high else 3))
        depth = min(n, room)
        return walk(d, i, depth, self.rule).tokens if depth > 0 else []

    def record(self, drafted, accepted):
        if drafted:
            self.rate = 0.875 * self.rate + 0.125 * (accepted / drafted)


class NoisyProductPolicy(ProductPolicy):
    """nc<N>:<p>[:B]: stop when the product of noise-aware confidences softmax((score + g) / B)[pick] drops below p,
    always keeping the first draft (the fc rule with the keyed draws taken into account)."""

    def __init__(self, most: int, confidence: float, beta: float, rule: PickRule = RULE, spec: str = "") -> None:
        super().__init__(most, confidence, rule, spec)
        self.beta = beta

    def propose(self, d, i, room, costs):
        depth = min(self.n, room)
        if depth <= 0:
            return []
        tr = walk(d, i, depth, self.rule)
        q = features(tr, self.beta)[:, FEATURES.index("noisy_prob")]
        chain, out = 1.0, []
        for k, t in enumerate(tr.tokens):
            chain *= q[k]
            if k > 0 and chain < self.confidence:
                break
            out.append(t)
        return out


def best_depth(p: np.ndarray, costs: Costs, drafting_cost: bool = True) -> int:
    """The k (0..len(p)) maximizing E[tokens | k drafts] / round ms, p the conditional acceptances of the drafts."""

    alive = np.concatenate([[1.0], np.cumprod(p)])          # P(the first j drafts all kept), j = 0..len(p)
    expect = np.cumsum(alive)                               # 1 + sum_{j <= k} P(first j kept)
    ms = np.array([costs.round_ms(k, 1, drafting_cost) for k in range(len(p) + 1)])
    return int(np.argmax(expect / ms))


class CostPolicy(Policy):
    """cost<N>:<pred>: walk the whole chain, predict each draft's conditional acceptance, keep the most profitable
    prefix (E[tokens] / ms, the block counted in every round since it ran to decide)."""

    def __init__(self, most: int, predictor: Predictor, beta: float = 1.0, rule: PickRule = RULE,
                 spec: str = "") -> None:
        self.n, self.predictor, self.beta, self.rule, self.spec = most, predictor, beta, rule, spec

    def most(self):
        return self.n

    def propose(self, d, i, room, costs):
        depth = min(self.n, room)
        if depth <= 0:
            return []
        tr = walk(d, i, depth, self.rule)
        p = self.predictor.predict(features(tr, self.beta))
        return tr.tokens[:best_depth(p, costs)]


# -- the replay -------------------------------------------------------------------------------------------------------
@dataclass
class SimResult:
    spec: str
    tokens: int = 0             # committed tokens past the first (what DecodeResult.tokens_per_second counts)
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    ms: float = 0.0
    keeps: list[int] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)

    def add(self, other: SimResult) -> None:
        self.tokens += other.tokens
        self.rounds += other.rounds
        self.drafted += other.drafted
        self.accepted += other.accepted
        self.ms += other.ms
        self.keeps += other.keeps
        self.depths += other.depths

    def summary(self) -> dict:
        r = max(self.rounds, 1)
        return {"policy": self.spec, "tokens": self.tokens, "rounds": self.rounds,
                "tokens_per_round": self.tokens / r, "drafts_per_round": self.drafted / r,
                "accept_rate": self.accepted / self.drafted if self.drafted else 0.0,
                "ms_per_round": self.ms / r, "tok_per_s": 1e3 * self.tokens / self.ms if self.ms else 0.0}


def simulate(d: Dump, policy: Policy, costs: Costs = COSTS) -> SimResult:
    """dflash_decode's rounds over the dump's reply: every round verifies the pending token and the policy's drafts
    and keeps the matching prefix plus one sampled token; the reply ends where the dump's does."""

    res = SimResult(policy.spec)
    i = 0
    while i < d.n:                                   # out holds tokens[:i + 1]; the loop runs while len(out) < count
        room = d.room(i)
        drafts = policy.propose(d, i, room, costs)
        kept = d.accepted(i, drafts)
        res.ms += costs.round_ms(len(drafts), kept + 1, min(policy.most(), room) > 0)
        res.rounds += 1
        res.drafted += len(drafts)
        res.accepted += kept
        res.keeps.append(kept + 1)
        res.depths.append(len(drafts))
        policy.record(len(drafts), kept)
        i += kept + 1
    res.tokens = min(i, d.n)          # a last round whose drafts all held: its extra sample lies past max_tokens
    return res


def oracle(d: Dump, most: int, costs: Costs = COSTS, rule: PickRule = RULE) -> SimResult:
    """The least total time any per-round depth choice (<= most) of this pick rule could reach, by dynamic
    programming over positions with hindsight: an upper bound for stop rules."""

    n = d.n
    run = np.zeros(n, dtype=np.int64)                # how many drafts of the full chain at i would be kept
    for i in range(n):
        run[i] = d.accepted(i, walk(d, i, min(most, n - i), rule).tokens)
    best = np.full(n + 1, np.inf)
    choice = np.zeros(n + 1, dtype=np.int64)
    best[n] = 0.0
    for i in range(n - 1, -1, -1):
        for k in range(min(most, n - i) + 1):
            kept = min(k, int(run[i]))
            t = costs.round_ms(k, kept + 1, k > 0) + best[min(i + kept + 1, n)]
            if t < best[i]:
                best[i], choice[i] = t, k
    res = SimResult(f"oracle{most}")
    i = 0
    while i < n:
        k = int(choice[i])
        kept = min(k, int(run[i]))
        res.ms += costs.round_ms(k, kept + 1, k > 0)
        res.rounds += 1
        res.drafted += k
        res.accepted += kept
        res.keeps.append(kept + 1)
        res.depths.append(k)
        i += kept + 1
    res.tokens = min(i, n)
    return res


def parse_rule(text: str) -> PickRule:
    rule = RULE
    for part in filter(None, text.split(",")):
        key, _, value = part.partition("=")
        if key == "noise":
            rule = replace(rule, noise=float(value))
        elif key == "edge":
            rule = replace(rule, edge=float(value))
        elif key == "trunc":
            rule = replace(rule, trunc=value not in ("0", "false", ""))
        else:
            raise ValueError(f"pick rule {text!r}: noise=W, edge=E or trunc=0/1")
    return rule


def parse_policy(spec: str, predictors: dict[str, Predictor] | None = None, beta: float = 1.0) -> Policy | str:
    """A policy from its spec (module docstring); ``oracle<N>`` comes back as its spec (``oracle``, not a Policy)."""

    base, _, rule_text = spec.partition("@")
    rule = parse_rule(rule_text)
    predictors = predictors or {}
    try:
        if base.startswith("oracle"):
            return spec
        if base.startswith("cost"):
            most, name = base[4:].split(":", 1)
            key, _, arg = name.partition(":")
            if key == "noisy":
                return CostPolicy(int(most), NoisyPredictor(), float(arg) if arg else beta, rule, spec)
            if key == "conf":
                return CostPolicy(int(most), ConfPredictor(), beta, rule, spec)
            if name in predictors:
                return CostPolicy(int(most), predictors[name], beta, rule, spec)
            raise ValueError(f"no predictor {name!r} (conf, noisy[:B], or a fitted one: {', '.join(predictors)})")
        if base.startswith("fa"):
            parts = base.split(":")
            low, high = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.8, 0.9)
            return AdaptivePolicy(low, high, rule, spec)
        if base.startswith("fc"):
            most, conf = base[2:].split(":")
            return ProductPolicy(int(most), float(conf), rule, spec)
        if base.startswith("nc"):
            parts = base[2:].split(":")
            return NoisyProductPolicy(int(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else beta,
                                      rule, spec)
        if base.startswith("f"):
            return ProductPolicy(int(base[1:]), 0.0, rule, spec)
    except (IndexError, ValueError) as err:
        raise ValueError(f"policy {spec!r}: {err}") from None
    raise ValueError(f"policy {spec!r}: fc<N>:<p>, f<N>, fa[:LOW:HIGH], nc<N>:<p>[:B], cost<N>:<pred>, oracle<N>")


def run_policy(dumps: Sequence[Dump], spec: str, costs: Costs, predictors: dict[str, Predictor] | None = None,
               beta: float = 1.0) -> SimResult:
    total = SimResult(spec)
    for d in dumps:
        pol = parse_policy(spec, predictors, beta)
        if isinstance(pol, str):
            base, _, rule_text = spec.partition("@")
            total.add(oracle(d, int(base[6:] or d.depth), costs, parse_rule(rule_text)))
        else:
            total.add(simulate(d, pol, costs))
    total.spec = spec
    return total


# -- checks -----------------------------------------------------------------------------------------------------------
def check(d: Dump) -> dict:
    """Consistency of a dump: its keyed draws, the target's top-T resampling to the committed token, and the sim's
    replay of the engine's production chain (``prod_chain``, bit for bit)."""

    from tensorfold.engine.exact_sampling import Sampling, choose_rows, uniform_rows

    out = {"file": d.path, "steps": d.n}
    m = d.meta
    if d.sampled:
        s = Sampling(int(m["seed"]), float(m["temperature"]), int(m["top_k"]), float(m["top_p"]),
                     float(m.get("min_p", 0.0)))
        g = np.stack([-np.log(-np.log(uniform_rows(s.seed, d.first[i] + np.arange(d.depth), d.cand_ids[i])))
                      for i in range(d.n)]) if d.n else d.noise
        out["noise_ok"] = bool(np.array_equal(g, d.noise))
        if s.top_k and s.top_k + MARGIN <= d.tgt_ids.shape[1]:
            w = s.top_k + MARGIN
            got = [choose_rows(d.tgt_vals[i:i + 1, :w], d.tgt_ids[i:i + 1, :w], [int(d.first[i])], s)[0]
                   for i in range(d.n)]
            out["target_ok"] = float(np.mean(np.array(got) == d.tokens[1:])) if d.n else 1.0
    else:
        order = [np.lexsort((d.tgt_ids[i], -d.tgt_vals[i]))[0] for i in range(d.n)]
        got = [int(d.tgt_ids[i, o]) for i, o in enumerate(order)]
        out["target_ok"] = float(np.mean(np.array(got) == d.tokens[1:])) if d.n else 1.0
    most, conf = int(m.get("prod_most", 5)), float(m.get("prod_confidence", 0.3))
    same = 0
    for i in range(d.n):
        mine = walk(d, i, min(most, d.depth), PickRule(float(m.get("edge", EDGE)), float(m.get("noise", NOISE))),
                    conf).tokens
        theirs = [int(t) for t in d.prod_chain[i] if t >= 0]
        same += mine == theirs
    out["prod_chain_ok"] = same / d.n if d.n else 1.0
    return out


def depth_table(dumps: Sequence[Dump], rule: PickRule = RULE, depth: int = 7) -> list[dict]:
    """Per depth: how often the full chain's draft is kept given the drafts before it were, and the mean confidences."""

    X, y = training_rows(dumps, rule, depth)
    rows = []
    for k in range(depth):
        mk = X[:, FEATURES.index("depth")] == k
        if not mk.any():
            break
        rows.append({"depth": k + 1, "rows": int(mk.sum()), "kept": float(y[mk].mean()),
                     "conf": float(X[mk, FEATURES.index("conf")].mean()),
                     "noisy_prob": float(X[mk, FEATURES.index("noisy_prob")].mean())})
    return rows


# -- command line -----------------------------------------------------------------------------------------------------
DEFAULT_POLICIES = ("fc5:0.3", "f1", "f2", "f3", "f4", "f5", "f7", "fc7:0.3", "fc5:0.2", "fc5:0.4", "fa",
                    "fc5:0.3@noise=1.0", "fc5:0.3@noise=1.0,trunc=1", "nc7:0.3", "nc7:0.3@noise=1.0,trunc=1",
                    "cost7:conf", "cost7:noisy", "cost7:const", "cost7:logit", "cost7:logit@noise=1.0,trunc=1",
                    "oracle7")


def _split(dumps: list[Dump]) -> tuple[list[Dump], list[Dump], str]:
    if len(dumps) >= 2:
        return dumps[0::2], dumps[1::2], "fit on the even-numbered dumps, evaluated on the odd-numbered ones"
    return dumps, dumps, "one dump: fitted and evaluated on the same records (in-sample)"


def main(argv: Sequence[str] | None = None) -> int:
    head, _, rest = __doc__.partition("\n\n")
    ap = argparse.ArgumentParser(prog="draft_sim", description=head, epilog=rest,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help=".npz dumps, globs or directories (evaluated on)")
    ap.add_argument("--train", nargs="*", default=None,
                    help="dumps to fit predictors on (default: every other one of DUMPS, the rest evaluated)")
    ap.add_argument("--policy", action="append", default=None, help="a policy spec (repeatable; default: a sweep)")
    ap.add_argument("--verify", default=",".join(map(str, VERIFY_MS)), help="V(R) ms for R = 1..8, comma separated")
    ap.add_argument("--block", type=float, default=BLOCK_MS, help="DFlash2 block ms")
    ap.add_argument("--overhead", type=float, default=0.0, help="ms every round beyond V(R) and the block")
    ap.add_argument("--taps-row", type=float, default=0.0, help="ms a kept row's context update")
    ap.add_argument("--beta", type=float, default=0.0, help="noise-aware temperature (0: fitted on the train dumps)")
    ap.add_argument("--check", action="store_true", help="verify each dump's consistency first")
    ap.add_argument("--json", action="store_true", help="print JSON lines instead of a table")
    ap.add_argument("--only", choices=("sampled", "greedy"), help="keep only sampled (temperature > 0) or greedy dumps")
    args = ap.parse_args(argv)

    keep = (lambda d: d.sampled == (args.only == "sampled")) if args.only else (lambda d: True)
    dumps = [d for d in load_dumps(args.dumps) if keep(d)]
    if not dumps:
        print("draft_sim: no dumps", file=sys.stderr)
        return 1
    costs = Costs(tuple(float(v) for v in args.verify.split(",")), args.block, args.overhead, args.taps_row)
    if args.train is not None:
        train, test, how = [d for d in load_dumps(args.train) if keep(d)], dumps, "fit on --train, evaluated on DUMPS"
    else:
        train, test, how = _split(dumps)
    emit = (lambda obj: print(json.dumps(obj))) if args.json else None
    if args.check:
        for d in dumps:
            c = check(d)
            emit(c) if emit else print("check", c)

    specs = args.policy or list(DEFAULT_POLICIES)
    rules = {parse_rule(s.partition("@")[2]) for s in specs}
    beta = args.beta
    if beta <= 0:
        beta, loss = fit_beta(train, PickRule())
        note = {"beta": beta, "log_loss": loss}
        emit(note) if emit else print(f"noise-aware beta {beta:g} (log loss {loss:.4f} on the train dumps)")
    predictors_by_rule: dict[PickRule, dict[str, Predictor]] = {}
    for rule in rules:
        Xtr, ytr = training_rows(train, rule, 7, beta)
        Xte, yte = training_rows(test, rule, 7, beta)
        fitted: dict[str, Predictor] = {"const": ConstPredictor().fit(Xtr, ytr),
                                        "logit": LogisticPredictor().fit(Xtr, ytr)}
        predictors_by_rule[rule] = fitted
        report = {"rule": asdict(rule), "split": how}
        for name, pred in [("conf", ConfPredictor()), ("noisy", NoisyPredictor(beta))] + list(fitted.items()):
            report[name] = scores(pred.predict(Xte), yte) if len(yte) else {}
        report["logit_weights"] = fitted["logit"].describe()
        if emit:
            emit({"predictors": report})
        else:
            print(f"\npredictors ({report['rule']}; {how}):")
            for name in ("conf", "noisy", "const", "logit"):
                s = report[name]
                if s:
                    print(f"  {name:6s} rows {s['rows']:6d}  kept {s['rate']:.3f}  log loss {s['log_loss']:.4f}  "
                          f"brier {s['brier']:.4f}  auc {s.get('auc', float('nan')):.3f}")
            print(f"  logit weights: {report['logit_weights']}")
    table = depth_table(test)
    if emit:
        emit({"depths": table})
    else:
        print("\nper-depth acceptance of the full chain (today's pick), given the drafts before were kept:")
        for r in table:
            print(f"  depth {r['depth']}: rows {r['rows']:6d}  kept {r['kept']:.3f}  conf {r['conf']:.3f}  "
                  f"noisy {r['noisy_prob']:.3f}")
        print(f"\n{'policy':34s} {'tok/round':>9s} {'drafts':>7s} {'accept':>7s} {'ms/round':>9s} {'tok/s':>7s}")
    for spec in specs:
        rule = parse_rule(spec.partition("@")[2])
        s = run_policy(test, spec, costs, predictors_by_rule.get(rule), beta).summary()
        if emit:
            emit(s)
        else:
            print(f"{spec:34s} {s['tokens_per_round']:9.3f} {s['drafts_per_round']:7.2f} {s['accept_rate']:7.3f} "
                  f"{s['ms_per_round']:9.2f} {s['tok_per_s']:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
