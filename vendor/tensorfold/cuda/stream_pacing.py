"""Smooth streaming (``TF_GLM_STREAM_SMOOTH=1``): a streamed reply's text goes out token by token at a steady pace
from a short playout buffer, instead of a burst of drafted tokens every round and a pause whenever the engine fills
another request's prompt.

The engine's rounds come in bursts: a round accepts ~3 drafted tokens, a round takes longer while other requests
decode, and while a new prompt fills the rounds wait for its layer slices. The request's tokens enter a queue as they
arrive and a writer thread sends them one event each: the next goes out ``hold / queued`` after the previous one
(never before it arrived), so the queue settles at about ``hold`` seconds of tokens and a pause in the rounds shorter
than that is spread over the tokens already queued instead of showing as a freeze. No token is held longer than
``LONGEST_HOLD`` times ``hold``.

The text, its order and the reply's end are unchanged: only when events go out. Events that are not text (tool-call
deltas, keep-alives) keep their place in the queue and are not paced; the round that ends the reply flushes the queue
at once, so the reply ends when it did. The engine never waits: the request thread only queues."""

from __future__ import annotations

import collections
import os
import threading
import time
from typing import Any, Callable

HOLD_MS = 400               # the default playout buffer: about this much text is held, so shorter pauses never show
LONGEST_HOLD = 2.5          # no token is held longer than this many times the buffer


def smooth_streaming(value: str | None = None) -> bool:
    """``TF_GLM_STREAM_SMOOTH``: 1 paces a streamed reply's tokens, 0 (the default) sends each round as one event."""

    raw = (os.environ.get("TF_GLM_STREAM_SMOOTH", "0") if value is None else value).strip() or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"TF_GLM_STREAM_SMOOTH={raw!r}: expected 0 or 1")
    return raw == "1"


def hold_seconds(value: str | None = None) -> float:
    """``TF_GLM_STREAM_SMOOTH_MS``: the playout buffer in milliseconds (default ``HOLD_MS``)."""

    raw = (os.environ.get("TF_GLM_STREAM_SMOOTH_MS", "") if value is None else value).strip() or str(HOLD_MS)
    try:
        ms = int(raw)
    except ValueError:
        raise ValueError(f"TF_GLM_STREAM_SMOOTH_MS={raw!r}: expected whole milliseconds") from None
    if not 1 <= ms <= 5000:
        raise ValueError(f"TF_GLM_STREAM_SMOOTH_MS={ms}: expected 1-5000")
    return ms / 1000.0


class _Item:
    __slots__ = ("delta", "paced", "arrived")

    def __init__(self, delta: dict[str, Any], paced: bool, arrived: float) -> None:
        self.delta, self.paced, self.arrived = delta, paced, arrived


class Playout:
    """One streamed reply's playout queue, and the writer thread that empties it through ``emit``.

    ``put`` queues an event (``paced``: a text delta; else it goes out as soon as it reaches the front); ``flush``
    sends everything queued at once from then on; ``close`` flushes, waits for the queue to empty and stops the writer;
    ``abort`` drops what is queued and stops it. ``gone`` holds once ``emit`` failed or ``cancelled()`` said the
    client left (checked before each held event)."""

    def __init__(self, emit: Callable[[dict[str, Any]], bool], hold: float | None = None,
                 cancelled: Callable[[], bool] | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.emit, self.cancelled, self.clock = emit, cancelled, clock
        self.hold = hold_seconds() if hold is None else hold
        self.queue: collections.deque[_Item] = collections.deque()
        self.paced = 0                       # paced items queued
        self.last: float | None = None       # when the previous event went out
        self.flushing = self.closed = self.gone = False
        self.error: BaseException | None = None
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, name="tf-playout", daemon=True)
        self.thread.start()

    # -- the request thread -------------------------------------------------------------------------------------------

    def put(self, delta: dict[str, Any], paced: bool = True) -> None:
        with self.cond:
            if self.closed or self.gone:
                return
            self.queue.append(_Item(delta, paced, self.clock()))
            self.paced += paced
            self.cond.notify()

    def flush(self) -> None:
        with self.cond:
            self.flushing = True
            self.cond.notify()

    def close(self) -> None:
        """Send everything still queued at once and stop the writer (the reply's end follows)."""

        with self.cond:
            self.flushing = self.closed = True
            self.cond.notify()
        self.thread.join()

    def abort(self) -> None:
        with self.cond:
            self.queue.clear()
            self.paced = 0
            self.closed = True
            self.cond.notify()
        self.thread.join()

    # -- the writer ---------------------------------------------------------------------------------------------------

    def due(self, now: float) -> float:
        """When the front of the queue goes out (the queue is not empty)."""

        item = self.queue[0]
        if self.flushing or not item.paced or self.last is None:
            return now
        at = max(item.arrived, self.last + self.hold / max(self.paced, 1))
        return min(at, item.arrived + LONGEST_HOLD * self.hold)

    def _run(self) -> None:
        while True:
            with self.cond:
                while True:
                    if not self.queue:
                        if self.closed:
                            return
                        self.cond.wait()
                        continue
                    now = self.clock()
                    at = self.due(now)
                    if at <= now:
                        break
                    self.cond.wait(at - now)          # a new token shortens the wait; a flush ends it
                item = self.queue.popleft()
                self.paced -= item.paced
                held = item.paced and not self.flushing and now > item.arrived
            try:
                ok = not (held and self.cancelled is not None and self.cancelled()) and self.emit(item.delta)
            except BaseException as exc:              # noqa: BLE001  re-raised by the request after generate
                self.error, ok = exc, False
            with self.cond:
                self.last = self.clock()
                if not ok:
                    self.gone = True
                    self.queue.clear()
                    self.paced = 0
                    return
