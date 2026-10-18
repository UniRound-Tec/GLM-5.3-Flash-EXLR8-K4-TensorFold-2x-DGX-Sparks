"""Request-local port of the old vLLM cost-K controller for TensorFold.

Only rank 0 chooses and learns elapsed costs. Its choice travels in the existing
ROUND payload. Target sampling, verification and commit stay unchanged.
"""
import math
import statistics


class EmpiricalDepth:
    empirical = True

    def __init__(self, most, costs):
        if not 1 <= most <= 7 or not costs or len(costs['verify']) <= most:
            raise ValueError('empirical depth needs captured K=0..N and measured costs')
        self.most = most
        self.confidence = 0.0
        self.beta = 2.0
        self.round_ms = None
        self.prior = [float(v) + (float(costs['block']) if k else 0.0)
                      + float(costs.get('taps_row', 0.0))
                      for k, v in enumerate(costs['verify'][:most + 1])]
        if any(not math.isfinite(v) or v <= 0 for v in self.prior):
            raise ValueError('invalid measured cost prior')
        self.p = [0.8] * most
        self.n = [0] * most
        self.times = {}
        self.rounds = 0
        self.current = most
        self.last = None
        self.overhead = 0.0
        self.hist = {}
        self.probes = 0
        self.fallbacks = 0
        self.probe_due = 32
        self.probe_gap = 32
        self.probe_k = None
        self.rejected_costs = 0
        self.regime_resets = 0
        self.shared = False

    @property
    def chain_rule(self):
        # beta changes confidence only; candidate picks still use the same keyed noise.
        return {'beta': self.beta}

    def next(self, drafted, accepted):
        # Engine calls this on both ranks. Rank 0 selects in the scheduler, never here.
        return self.current

    def set_shared(self, shared):
        self.shared = bool(shared)
        self.confidence = 0.3 if self.shared else 0.0

    def cost(self, k):
        rec = self.times.get(k)
        return rec[0] if rec and self.rounds - rec[1] <= 64 else self.prior[k] + self.overhead

    def scores(self, most):
        expected = reach = 1.0
        out = []
        for k in range(most + 1):
            if k:
                reach *= self.p[k - 1]
                expected += reach
            out.append(expected / self.cost(k))
        return out

    def choose(self, remaining):
        most = min(self.most, max(0, remaining))
        if self.shared:
            self.fallbacks += 1
            self.current = most
            return most
        if self.rounds < 2:
            self.current = most
            return most
        scores = self.scores(most)
        best = max(range(most + 1), key=lambda k: scores[k])
        if self.current <= most and scores[best] < 1.04 * scores[self.current]:
            best = self.current
        if self.probe_k is not None:
            # A failed exploratory gear backs off; profitable changes allow fresh probes.
            self.probe_gap = min(96, self.probe_gap + 32) if best != self.probe_k else 32
            self.probe_due = self.rounds + self.probe_gap
            self.probe_k = None
        if self.rounds >= self.probe_due and most:
            stale = [k for k in range(most + 1) if k != best
                     and (k not in self.times or self.rounds - self.times[k][1] > 64)]
            # Alternate high-tail visibility and a nearby stale cost in both directions.
            if self.probes % 2 == 0 and best < most:
                candidate = most
            elif stale:
                candidate = min(stale, key=lambda k: (abs(k - best), k))
            else:
                candidate = None
            if candidate is not None:
                best = candidate
                self.probe_k = best
                self.probes += 1
            self.probe_due = self.rounds + self.probe_gap
        self.current = best
        return best

    def record(self, drafted, accepted, usable=True):
        self.last = None
        if not usable or self.shared:
            return
        if not 0 <= accepted <= drafted <= self.most:
            raise ValueError('invalid acceptance observation')
        self.rounds += 1
        self.hist[drafted] = self.hist.get(drafted, 0) + 1
        for j in range(min(drafted, accepted + 1)):
            self.p[j] = .85 * self.p[j] + .15 * (j < accepted)
            self.n[j] += 1
        self.last = drafted

    def observe_ms(self, ms):
        k, self.last = self.last, None
        if k is None or not math.isfinite(ms) or not 1 < ms < 2000:
            return
        rec = self.times.get(k)
        if rec is None:
            self.times[k] = [ms, self.rounds, [ms], 0]
        else:
            avg, age, samples, rejects = rec
            med = statistics.median(samples)
            mad = statistics.median(abs(x - med) for x in samples)
            if len(samples) >= 4 and abs(ms - med) > max(3 * mad, .15 * med, 2) and rejects < 7:
                rec[3] += 1
                self.rejected_costs += 1
                return
            reset = rejects >= 7
            self.regime_resets += int(reset)
            self.times[k] = [ms if reset else .9 * avg + .1 * ms, self.rounds,
                             [ms] if reset else (samples + [ms])[-16:], 0]
        # Runtime overhead also calibrates unmeasured gears; no stale optimistic plateau.
        delta = max(0.0, min(self.prior[0], ms - self.prior[k]))
        self.overhead = .9 * self.overhead + .1 * delta

    def describe(self):
        return {'rounds': self.rounds, 'k_histogram': self.hist, 'probes': self.probes,
                'shared_fallbacks': self.fallbacks, 'conditional_p': self.p,
                'position_observations': self.n,
                'costs_ms': {str(k): v[0] for k, v in self.times.items()},
                'cost_sample_age': {str(k): self.rounds-v[1] for k, v in self.times.items()},
                'prior_ms': self.prior, 'overhead_ms': self.overhead,
                'cost_rejections': self.rejected_costs, 'regime_resets': self.regime_resets}
