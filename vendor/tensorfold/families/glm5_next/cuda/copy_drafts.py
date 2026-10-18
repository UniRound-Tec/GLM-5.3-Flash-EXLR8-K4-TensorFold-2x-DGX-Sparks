"""Prompt-lookup ("copy") drafts: when the context's last tokens occurred before, propose what followed them.

A reply that quotes or edits its prompt (or repeats itself) is cheap to draft: its last ``match`` tokens (the
pending token included) are searched in the prompt and the reply so far, and the tokens after the latest earlier
occurrence become the drafts. They only propose; verification keeps the serial sample, so replies stay exact.

Three refinements, off by default (they only change what is proposed, never what is kept):
- ``reply_match`` (TF_GLM_COPY_REPLY_MATCH): an occurrence that starts inside the reply must match the context's last
  ``reply_match`` tokens, not ``match``. Code replies repeat short boilerplate (``    def __init__(self``, ``}); ``)
  whose continuations differ: over 39 live code replies (2026-09-30) reply-sourced copies kept 23% of their drafts,
  prompt-sourced ones 74%, and a missed copy round costs a wide window and the DFlash2 round it displaced.
- ``miss_most`` (TF_GLM_COPY_MISS_MAX): after a copied round whose drafts were not all kept, the next copied round
  proposes at most this many (a narrow window), until a copied round keeps all of its drafts again.
- ``pad`` (the engine's TF_GLM_WIDE_GRAPHS widths): a proposal whose window (pending token + drafts) falls between 8
  rows and a captured width is padded to that width by repeating its last draft, so the window replays a CUDA graph
  instead of running eagerly. Padded drafts are drafts like any other: kept only where they equal the sample.

Both ranks hold the same prompt and sample the same tokens, so each computes the same proposals with no exchange.
Pure numpy (no torch), so it is testable anywhere."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

MATCH = 8        # the context's last this many tokens must have occurred before (TF_GLM_COPY_MATCH)
MOST = 5         # drafts a round (TF_GLM_COPY_MAX, up to 15): graphs cover windows of 8 rows, wider ones run eagerly


GRAPHED = 8      # verify windows up to this many rows always replay CUDA graphs (engine.GRAPH_ROWS)


@dataclass(frozen=True)
class CopySettings:
    match: int = MATCH
    most: int = MOST
    reply_match: int = 0             # 0: occurrences in the reply need ``match`` tokens like the prompt's
    pad: tuple[int, ...] = ()        # window widths past GRAPHED rows that replay CUDA graphs (TF_GLM_WIDE_GRAPHS)
    miss_most: int = 0               # 0: off; else drafts a copied round after one that missed

    @classmethod
    def from_env(cls, max_drafts: int, env=None, pad: Sequence[int] = ()) -> CopySettings | None:
        """TF_GLM_COPY_DRAFTS=1 turns copy drafts on (off by default); TF_GLM_COPY_MATCH, TF_GLM_COPY_MAX and
        TF_GLM_COPY_REPLY_MATCH (0 or more than the match, up to 64) and TF_GLM_COPY_MISS_MAX (0: off, or 1 up to
        TF_GLM_COPY_MAX) tune; ``pad``: the engine's wide graph widths."""

        env = os.environ if env is None else env
        on = env.get("TF_GLM_COPY_DRAFTS", "0").strip()
        if on in ("", "0"):
            return None
        if on != "1":
            raise ValueError(f"TF_GLM_COPY_DRAFTS: 0 or 1, not {on!r}")

        def number(name: str, default: int, low: int, high: int) -> int:
            text = env.get(name, "").strip()
            if text == "":
                return default
            if not text.isdecimal() or not low <= int(text) <= high:
                raise ValueError(f"{name}: {low} to {high}, not {text!r}")
            return int(text)

        match = number("TF_GLM_COPY_MATCH", MATCH, 2, 64)
        reply = number("TF_GLM_COPY_REPLY_MATCH", 0, 0, 64)
        if reply and reply <= match:
            raise ValueError(f"TF_GLM_COPY_REPLY_MATCH: 0 (off) or more than TF_GLM_COPY_MATCH ({match}), not {reply}")
        widths = tuple(sorted(int(r) for r in pad if GRAPHED < int(r) <= max_drafts + 1))
        most = number("TF_GLM_COPY_MAX", min(MOST, max_drafts), 1, max_drafts)
        return cls(match, most, reply, widths, number("TF_GLM_COPY_MISS_MAX", 0, 0, most))

    def code(self) -> list[int]:
        """The settings as ints both ranks compare at startup (zeros: off; the pad widths are the engine's
        TF_GLM_WIDE_GRAPHS, compared on their own)."""

        return [self.match, self.most, self.reply_match, self.miss_most]

    def drafts(self, context: Sequence[int]) -> CopyDrafts:
        """A request's index: ``context`` is its prompt and pending token (the reply's first token)."""

        return CopyDrafts(context, self.match, self.most, prompt=max(len(context) - 1, 0),
                          reply_match=self.reply_match, pad=self.pad, miss_most=self.miss_most)


class CopyDrafts:
    """One sequence's context (prompt, then committed reply tokens and the pending one) and its copy proposals."""

    def __init__(self, context: Sequence[int], match: int = MATCH, most: int = MOST, *, prompt: int = 0,
                 reply_match: int = 0, pad: Sequence[int] = (), miss_most: int = 0) -> None:
        """``prompt``: how many leading context tokens are the prompt (occurrences starting at or past it are the
        reply's, which need ``reply_match`` tokens when that is more than ``match``); ``pad``: window widths past
        GRAPHED rows to pad a proposal's window up to (``propose``); ``miss_most``: drafts a copied round after one
        that missed (0: ``most`` always)."""

        if match < 1 or most < 1:
            raise ValueError("match and most must be positive")
        self.match, self.most = int(match), int(most)
        self.prompt = int(prompt)
        self.reply_match = int(reply_match) if int(reply_match) > self.match else 0
        self.pad = tuple(sorted(int(r) for r in pad if int(r) > GRAPHED))
        self.miss_most = int(miss_most)
        self.proposed = 0              # drafts of the last proposal not yet settled by ``extend``
        self.missed = False            # the last settled copied round kept fewer than all of its drafts
        n = len(context)
        self.buf = np.empty((max(1024, 2 * n),), dtype=np.int32)
        self.buf[:n] = np.asarray(context, dtype=np.int32) if n else 0
        self.length = n

    def __len__(self) -> int:
        return self.length

    def tokens(self) -> list[int]:
        return self.buf[:self.length].tolist()

    def extend(self, tokens: Sequence[int]) -> None:
        """Committed tokens (the last one is the next round's pending token) join the context; after a proposal,
        they settle whether its round kept all of its copied drafts (they then number at least those plus one)."""

        k = len(tokens)
        if not k:
            return
        if self.proposed:
            self.missed = k <= self.proposed
            self.proposed = 0
        if self.length + k > self.buf.shape[0]:
            grown = np.empty((2 * (self.length + k),), dtype=np.int32)
            grown[:self.length] = self.buf[:self.length]
            self.buf = grown
        self.buf[self.length:self.length + k] = np.asarray(tokens, dtype=np.int32)
        self.length += k

    def starts(self) -> np.ndarray:
        """Ascending starts of the earlier occurrences of the context's last ``match`` tokens (not the suffix)."""

        L, n = self.length, self.match
        if L <= n:
            return np.empty((0,), dtype=np.int64)
        ctx = self.buf
        q = ctx[L - n:L]
        # starts 0 .. L - n - 1 (each leaves a token after its match): by the last token, then the others
        hits = np.flatnonzero(ctx[n - 1:L - 1] == q[n - 1])
        for k in range(n - 1):
            if not hits.size:
                break
            hits = hits[ctx[hits + k] == q[k]]
        extra = self.reply_match - n if self.reply_match else 0
        if extra and hits.size:
            # an occurrence starting inside the reply: the extra tokens before it must match too
            inside = hits >= self.prompt
            ok = ~inside | ((hits >= extra) & (L - n - extra >= 0))
            for k in range(1, extra + 1):
                sel = inside & ok
                if not sel.any():
                    break
                ok[sel] = ctx[hits[sel] - k] == ctx[L - n - k]
            hits = hits[ok]
        return hits

    def propose(self, room: int | None = None) -> list[int]:
        """Up to ``min(most, room)`` drafts: what followed the latest earlier occurrence that has that many tokens
        after it, else the most after any occurrence (the earliest); [] when the suffix never occurred before. With
        ``pad``, a window of more than GRAPHED rows short of a padded width is filled up to it (``room`` allowing)
        by repeating the last draft."""

        most = min(self.most, self.miss_most) if self.missed and self.miss_most else self.most
        k = most if room is None else min(most, int(room))
        self.proposed = 0
        if k < 1:
            return []
        hits = self.starts()
        if not hits.size:
            return []
        L, n = self.length, self.match
        full = hits[hits <= L - n - k]
        s = int(full[-1]) if full.size else int(hits[0])
        out = self.buf[s + n:min(s + n + k, L)].tolist()
        self.proposed = len(out)                  # the copied drafts (a pad's repeats may miss without counting)
        return self.padded(out, room)

    def padded(self, drafts: list[int], room: int | None = None) -> list[int]:
        """``drafts`` filled up to the next ``pad`` width's window by repeating the last one, when their window is
        wider than GRAPHED rows, narrower than that width, and ``room`` (default ``most``) leaves space for it."""

        rows = 1 + len(drafts)
        if not self.pad or not drafts or rows <= GRAPHED:
            return drafts
        width = next((w for w in self.pad if w >= rows), 0)
        if width <= rows or width - 1 > (self.most if room is None else int(room)):
            return drafts
        return drafts + [drafts[-1]] * (width - rows)
