"""The reasoning of GLM's tool-calling replies, put back into later requests whose client dropped it.

Parts adapted from jayleaton/glm53-tensorfold-spark patch 0620 (Apache-2.0): its ``reasoning`` fix (the
``ReasoningMemory`` LRU, keyed by the server's call ids and by a signature of the calls for clients that renumber
ids). Here the signature covers the whole conversation before the calls, not only the last user message, so a reply
is only ever put back into the conversation it was written for.

GLM-5.3's template renders the reasoning of every assistant step after the last user message (interleaved thinking
inside one tool loop). A client that does not send a step's reasoning back (spark-bench, many agent frameworks) gets
``<think></think>`` for it: a context the model never wrote, and a prompt that stops matching the previous one at
that step, so the prefix cache re-reads the rest. With the reasoning kept here, each prompt of a chain continues the
last one as it was generated. A step that carries its own reasoning (``reasoning_content`` or ``reasoning``, or a
``<think>`` block in its content) is never changed; an empty one (clients that send ``""`` for every step) is none.

``TF_GLM_KEEP_REASONING``: replies kept (default 1024; 0 turns this off); ``TF_GLM_KEEP_REASONING_MB``: their text
at most (default 32 MiB); the oldest go first.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from typing import Any

_OUR_ID = re.compile(r"^call_[0-9a-f]{24}$")      # the ids ``parse_tool_calls`` / ``GlmCallStreamer`` make


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: expected a whole number") from None
    if value < 0:
        raise ValueError(f"{name}={raw!r}: expected 0 or more")
    return value


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value) if value.strip() else {}
        except ValueError:
            return value
    return {} if value is None else value


def _calls(calls: Any) -> list[list[Any]]:
    """Calls as [name, arguments]: ids dropped (clients renumber them), arguments parsed (clients re-serialise)."""

    out = []
    for call in calls if isinstance(calls, list) else []:
        fn = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
        out.append([str(fn.get("name") or ""), _arguments(fn.get("arguments"))])
    return out


def signature(history: list[Any], calls: Any) -> str:
    """The key of ``calls`` written after ``history``: each message's role, text (stripped, as the template renders
    it) and calls, without ids or reasoning, then the calls."""

    shape = []
    for m in history:
        if not isinstance(m, dict):
            continue
        entry = [str(m.get("role") or ""), _text(m.get("content")).strip()]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            entry.append(_calls(m.get("tool_calls")))
        shape.append(entry)
    text = json.dumps([shape, _calls(calls)], ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "sig:" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _keys(history: list[Any], calls: Any) -> list[str]:
    ids = [c.get("id") for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []
    return [i for i in ids if isinstance(i, str) and _OUR_ID.match(i)] + [signature(history, calls)]


class KeptReasoning:
    """Recent tool-calling replies' reasoning (LRU, bounded in replies and characters), found by any of its keys."""

    def __init__(self, entries: int = 1024, chars: int = 32 << 20) -> None:
        self.entries, self.chars = int(entries), int(chars)
        self.items: OrderedDict[int, tuple[str, tuple[str, ...]]] = OrderedDict()
        self.by_key: dict[str, int] = {}
        self.size = self.next = 0
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "KeptReasoning | None":
        entries = _env_int("TF_GLM_KEEP_REASONING", 1024)
        return cls(entries, _env_int("TF_GLM_KEEP_REASONING_MB", 32) << 20) if entries else None

    def __len__(self) -> int:
        return len(self.items)

    def remember(self, messages: Any, calls: Any, reasoning: Any) -> None:
        """Keep the ``reasoning`` of a reply that made ``calls`` after the request's ``messages``."""

        if not calls or not isinstance(reasoning, str) or not reasoning.strip() or len(reasoning) > self.chars:
            return
        keys = tuple(_keys(messages if isinstance(messages, list) else [], calls))
        with self.lock:
            n, self.next = self.next, self.next + 1
            self.items[n] = (reasoning, keys)
            self.size += len(reasoning)
            for key in keys:                    # a key kept again points at the newer reply
                self.by_key[key] = n
            while len(self.items) > self.entries or self.size > self.chars:
                old, (text, old_keys) = self.items.popitem(last=False)
                self.size -= len(text)
                for key in old_keys:
                    if self.by_key.get(key) == old:
                        del self.by_key[key]

    def _find(self, keys: list[str]) -> str | None:
        with self.lock:
            for key in keys:
                n = self.by_key.get(key)
                if n is not None:
                    self.items.move_to_end(n)
                    return self.items[n][0]
        return None

    def restore(self, messages: Any) -> Any:
        """``messages`` with the kept reasoning of each assistant step that calls tools and carries none (a copy;
        the caller's list and messages are left alone)."""

        if not isinstance(messages, list) or not self.items:
            return messages
        out = None
        for i, m in enumerate(messages):
            if not (isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
                    and not _text(m.get("reasoning_content")).strip() and not _text(m.get("reasoning")).strip()
                    and "</think>" not in _text(m.get("content"))):
                continue
            kept = self._find(_keys(messages[:i], m["tool_calls"]))
            if kept is not None:
                out = out if out is not None else list(messages)
                out[i] = {**m, "reasoning_content": kept}
        return messages if out is None else out
