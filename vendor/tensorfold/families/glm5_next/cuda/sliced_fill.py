"""CPU policy and saved control state for opt-in layer-sliced concurrent prompt fills.

The GPU rows live in Engine.pbuf, already counted by the startup memory estimate. Only one prompt chunk may
be suspended; its generator retains the HC split loop's next-front/done state. No additional GPU storage.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Sequence


def budget_ms(value: str | None = None) -> float:
    value = os.environ.get("TF_GLM_FILL_BUDGET_MS", "") if value is None else value
    try:
        ms = float(value.strip() or "0")
    except ValueError:
        ms = -1.0
    if not math.isfinite(ms) or ms < 0:
        raise ValueError(f"TF_GLM_FILL_BUDGET_MS: finite milliseconds >= 0 (0: off), not {value!r}")
    return ms


def drafted_rounds(value: str | None = None) -> bool:
    """TF_GLM_FILL_DRAFTS: 1 makes the decode rounds between prompt slices full drafted rounds (more tokens for the
    decoding streams during a fill, a later first token for the filling one); 0 (default): one token per stream."""

    raw = (os.environ.get("TF_GLM_FILL_DRAFTS", "") if value is None else value).strip() or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"TF_GLM_FILL_DRAFTS: 0 or 1, not {raw!r}")
    return raw == "1"


class LayerCosts:
    """Startup timings at several row counts, interpolated separately for every KDA/DSA/dense layer.

    The first cost includes embedding. Tail work and per-slice overhead are learned separately. A layer slower than the budget runs alone: CUDA
    kernels cannot be preempted, so this setting is a target, not a hard latency guarantee.
    """
    def __init__(self, samples: dict[int, Sequence[float]]) -> None:
        if not samples:
            raise ValueError("prompt cost model needs samples")
        self.samples = {r: tuple(float(v) for v in costs) for r, costs in sorted(samples.items())}
        self.layers = len(next(iter(self.samples.values())))
        if (not self.layers or any(r <= 0 or len(c) != self.layers or
                                  any(not math.isfinite(v) or v <= 0 for v in c)
                                  for r, c in self.samples.items())):
            raise ValueError("prompt costs need positive rows and equally sized finite positive layer timings")
        self.scale = 1.0
        self.overhead = 0.0
        self.tails: dict[int, float] = {}

    def tail_cost(self, rows: int) -> float:
        if not self.tails:
            return 0.0
        lo = max((r for r in self.tails if r <= rows), default=min(self.tails))
        hi = min((r for r in self.tails if r >= rows), default=max(self.tails))
        if lo == hi:
            return self.tails[lo] * rows / lo
        t = (rows - lo) / (hi - lo)
        return self.tails[lo] + t * (self.tails[hi] - self.tails[lo])

    def observe_tail(self, rows: int, ms: float) -> None:
        if rows > 0 and math.isfinite(ms) and ms >= 0:
            self.tails[rows] = 0.8 * self.tails[rows] + 0.2 * ms if rows in self.tails else ms

    def observe_overhead(self, ms: float) -> None:
        if math.isfinite(ms) and ms >= 0:
            self.overhead = 0.8 * self.overhead + 0.2 * ms

    def costs(self, rows: int) -> tuple[float, ...]:
        keys = list(self.samples)
        lo = max((r for r in keys if r <= rows), default=keys[0])
        hi = min((r for r in keys if r >= rows), default=keys[-1])
        if lo == hi:
            return tuple(v * self.scale for v in self.samples[lo])
        t = (rows - lo) / (hi - lo)
        return tuple((a + t * (b - a)) * self.scale for a, b in zip(self.samples[lo], self.samples[hi]))

    def stop(self, rows: int, lo: int, budget: float) -> int:
        if not 0 <= lo < self.layers or not math.isfinite(budget) or budget <= 0:
            raise ValueError("slice needs a valid first layer and positive finite budget")
        costs = list(self.costs(rows))
        costs[-1] += self.tail_cost(rows)
        budget = max(0.001, budget - self.overhead)
        def greedy_end(start):
            end, ms = start + 1, costs[start]
            while end < self.layers and ms + costs[end] <= budget:
                ms += costs[end]
                end += 1
            return end

        def pieces(start):
            count = 0
            while start < self.layers:
                start = greedy_end(start)
                count += 1
            return count

        # Greedy gives the minimum number of contiguous slices. Among first cuts
        # that preserve that count, approach the mean cost instead of leaving a
        # tiny tail after several near-budget stalls. Never exceed the budget
        # except for one indivisible layer, as on the ordinary greedy path.
        count = pieces(lo)
        target = sum(costs[lo:]) / count
        ends = [hi for hi in range(lo + 1, greedy_end(lo) + 1) if pieces(hi) <= count - 1]
        return min(ends, key=lambda hi: (abs(sum(costs[lo:hi]) - target), -hi))

    def observe(self, rows: int, lo: int, hi: int, ms: float) -> None:
        # Layer GPU time only: tail and dispatch cost must not inflate every layer's prediction.
        predicted = sum(self.costs(rows)[lo:hi])
        if predicted > 0 and math.isfinite(ms) and ms > 0:
            self.scale = min(8.0, max(0.5, self.scale * (0.8 + 0.2 * ms / predicted)))


@dataclass
class PendingFill:
    group: list[tuple[Any, int]]
    many: bool = False
    layer: int = 0
    steps: Any = None
    retired: set[int] = field(default_factory=set)
    layer_ms: float = 0.0
    layer_event: Any = None
    slice_layer_ms: float = 0.0
    slice_wall_ms: float = 0.0
    starts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.starts = [lane.st.pos for lane, _ in self.group]

    @property
    def rows(self) -> int:
        return sum(stop - start for (_, stop), start in zip(self.group, self.starts))

    def contains(self, sid: int) -> bool:
        return any(lane.sid == sid for lane, _ in self.group)

    def close(self) -> None:
        if self.steps is not None:
            self.steps.close()
            self.steps = None


def advance(pending: PendingFill, lo: int, hi: int, layers: int, make_steps, before_finalize=None):
    """Run the requested contiguous layer range; return (complete, forward result)."""
    if not lo == pending.layer < hi <= layers:
        raise RuntimeError(f"out-of-order prompt slice [{lo}, {hi}), expected layer {pending.layer} of {layers}")
    if pending.steps is None:
        if lo:
            raise RuntimeError("a resumed prompt slice has no saved forward")
        pending.steps = make_steps()
    for _ in range(lo, hi):
        try:
            next(pending.steps)
        except StopIteration as exc:
            raise RuntimeError("prompt forward ended before its final layer") from exc
    pending.layer = hi
    if hi < layers:
        return False, None
    if before_finalize is not None:
        before_finalize()
    try:
        next(pending.steps)                   # final norm, taps, commit (after the last layer's yield)
    except StopIteration as result:
        pending.steps = None
        return True, result.value
    raise RuntimeError("prompt forward has more layers than the slice protocol")
