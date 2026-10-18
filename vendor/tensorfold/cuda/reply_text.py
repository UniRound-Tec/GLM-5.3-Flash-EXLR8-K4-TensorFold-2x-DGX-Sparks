"""The CUDA server's reply text: streamed decoding, think blocks and tool-call parsing (the Mac lane server's rules).

Parts adapted from jayleaton/glm53-tensorfold-spark patch 0620 (Apache-2.0): the rule that calls written inside the
think block are the reply's calls when they end it (its ``trailing_calls``), in ``ThinkSplit``.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tensorfold.server.stopping import StopPolicy
from tensorfold.server.tools import _GLM_NAME_RE as _GLM_NAME, parse_glm_tool_call_block
from tensorfold.tool_parameters import decode_parameter, nullable_text, parameter_schemas, typed_parameter

_CALL_OPEN, _CALL_CLOSE = "<tool_call>", "</tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.IGNORECASE | re.DOTALL)



def _partial_tag(text: str, tag: str) -> int:
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class StreamDecoder:
    """Decode a shared token window to preserve leading spaces and byte boundaries, deferring incomplete characters."""

    def __init__(self, tok, skip: tuple[int, ...] = ()):
        self.tok, self.skip = tok, frozenset(skip)
        self.ids: list[int] = []
        self.text = ""
        self.prefix = 0             # window start
        self.read = 0               # tokens already reflected in ``text``

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def add(self, new: list[int]) -> str:
        self.ids.extend(t for t in new if t not in self.skip)
        before = self._decode(self.ids[self.prefix:self.read])
        after = self._decode(self.ids[self.prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self.prefix, self.read = self.read, len(self.ids)
        return self.text

    def final(self) -> str:
        """Everything, including a trailing partial character (as decoding it all at once gives)."""

        before = self._decode(self.ids[self.prefix:self.read])
        return self.text + self._decode(self.ids[self.prefix:])[len(before):]



class StopStrings:
    """A request's ``stop`` strings, decided token by token, so a reply cuts at the same token however its tokens arrive."""

    def __init__(self, strings: tuple[str, ...], tok, skip: tuple[int, ...] = ()):
        self.strings, self.tok, self.skip = strings, tok, frozenset(skip)
        # a new match lies in the last (its UTF-8 length) tokens, plus eight as the Mac's ``StopPolicy`` keeps
        self.tail = max((len(s.encode()) for s in strings), default=0) + 8

    def hit(self, tokens: list[int]) -> bool:
        """Whether the text through the newest token holds a stop string (earlier tokens were checked already)."""

        ids = [t for t in tokens[-self.tail:] if t not in self.skip]
        text = self.tok.decode(ids, skip_special_tokens=False)
        return any(stop in text for stop in self.strings)

    def visible(self, text: str, *, partial: bool = False) -> str:
        """The text before the first match; ``partial`` also holds back an end that may begin one (the Mac's rule)."""

        return StopPolicy.visible(self, text, partial=partial)


def hide_tool_calls(text: str, *, finished: bool) -> str:
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(_CALL_OPEN, pos)
        if start < 0:
            tail = text[pos:]
            out.append(tail[: len(tail) - (0 if finished else _partial_tag(tail, _CALL_OPEN))])
            return "".join(out)
        out.append(text[pos:start])
        end = text.find(_CALL_CLOSE, start)
        if end < 0:
            return "".join(out)
        pos = end + len(_CALL_CLOSE)


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return str((fn or tool).get("name") or "").strip() if isinstance(tool, dict) else ""


def parse_tool_calls(text: str, tools: list[dict[str, Any]], *, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
    """Qwen ``<function=name><parameter=k>v</parameter></function>``, GLM ``name<arg_key>..`` or JSON calls."""

    if not tools:
        return text, None
    known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
    schemas = parameter_schemas(tools)
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    cursor = 0
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        residue.append(text[cursor:match.start()])
        cursor = match.end()
        if max_calls is not None and len(calls) >= max_calls:
            continue
        block = match.group(1).strip()
        name, args = None, {}
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
                name = fn.get("name")
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            m = _TOOL_FUNCTION_BLOCK_RE.match(block)
            if m:
                if max_calls is not None and _TOOL_PARAMETER_BLOCK_RE.sub("", m.group(2)).strip():
                    continue
                name = m.group(1).strip()
                # typed parameters (array, object, number...) decode per the tool's schema, as the Mac server does
                props = schemas.get(name.lower(), {})
                args = {key: decode_parameter(value, props.get(key, {}))
                        for key, value in ((p.group(1).strip(), p.group(2))
                                           for p in _TOOL_PARAMETER_BLOCK_RE.finditer(m.group(2)))}
            else:
                glm = parse_glm_tool_call_block(repair_glm_keys(block), tools, complete=max_calls is not None)
                if glm is not None:
                    name, args = glm
        if not name or str(name).lower() not in known:
            if max_calls is None:
                residue.append(match.group(0))
            continue
        if max_calls is not None:
            try:
                if not isinstance(args, dict):
                    continue
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError):
                continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": known[str(name).lower()],
                                   "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    residue.append(text[cursor:])
    return "".join(residue).strip(), calls or None


_THINK_CLOSE = "</think>"
_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
THINK_CALL_HOLD = 1024      # characters of a call written inside the think block held before it streams as a call
_GLM_TAG = re.compile(r"</?arg_(?:key|value)>")
_LAST_WORD = re.compile(r"(\S+)\s*$")


def repair_glm_keys(block: str) -> str:
    """A GLM call whose ``<arg_key>`` the model left out, writing whitespace where that token belongs (issue #22:
    ``read path</arg_key><arg_value>..``, or ``</arg_value>\npath</arg_key>``), with it put back: the word before an
    unopened ``</arg_key>`` is its key. A block where nothing is missing comes back as it is."""

    if block.count(_KEY_END) <= block.count(_KEY):
        return block
    out, at, prev, prev_end = [], 0, None, 0     # prev: the last tag seen (ending at prev_end); at: text copied
    for tag in _GLM_TAG.finditer(block):
        if tag.group(0) == _KEY_END and prev in (None, _VALUE_END):
            seg_start = 0 if prev is None else prev_end
            word = _LAST_WORD.search(block, seg_start, tag.start())
            # the name's segment keeps a name before the key (``read path``): no word, or no name, is not repaired
            if word is None or (prev is None and not block[seg_start:word.start()].strip()):
                return block
            out += [block[at:word.start()], _KEY]
            at = word.start()
        prev, prev_end = tag.group(0), tag.end()
    return "".join(out) + block[at:]


def close_glm_call(text: str, tools: list[dict[str, Any]]) -> str:
    """What closes the GLM call a reply ended inside when it parses as a whole call to an offered tool (issue #22:
    the model wrote its end token before ``</tool_call>``): ``</tool_call>``, after ``</arg_value>`` when a value is
    open; "" when the reply is not inside a call, or the closed call would not parse (its markup is then text)."""

    at = text.rfind(_CALL_OPEN)
    if at < 0:
        return ""
    block = text[at + len(_CALL_OPEN):]
    if _CALL_CLOSE in block or _THINK_CLOSE in block:
        return ""
    suffix = (_VALUE_END if block.rfind(_VALUE) > block.rfind(_VALUE_END) else "") + _CALL_CLOSE
    # the strict reading (one whole call, nothing but its markup, arguments a JSON object)
    return suffix if parse_tool_calls(_CALL_OPEN + block + suffix, tools, max_calls=1)[1] else ""


def _call_run(text: str, start: int, finished: bool) -> tuple[int, str]:
    """(where the run of complete ``<tool_call>`` blocks from ``start`` ends, what it is): ``"calls"`` when the reply
    ended on it (nothing after it but whitespace and a closing ``</think>``), ``"pending"`` while that may still
    come true, ``"reasoning"`` once text follows it (more reasoning, or an answer after the think block)."""

    end = pos = start
    while (block := _CALL_BLOCK.match(text, pos)) is not None:
        end = block.end()
        pos = len(text) - len(text[end:].lstrip())
    after = text[end:].lstrip()
    if after.startswith(_THINK_CLOSE) and not after[len(_THINK_CLOSE):].strip():
        after = ""
    if not after:
        return end, "calls" if finished and end > start else "pending"
    if not finished and (after.startswith(_CALL_OPEN) or _CALL_OPEN.startswith(after)
                         or _THINK_CLOSE.startswith(after)):
        return end, "pending"                   # a block, or the think block's close, being written
    return end, "reasoning"


class ThinkSplit:
    """(reasoning, answer) of a thinking GLM reply with tools, streamed or whole: the think block ends at its
    ``</think>``, and calls the model wrote inside it are its calls only when they end the reply (nothing after them
    but whitespace and the block's close; theirs is the D5 rule of jayleaton/glm53-tensorfold-spark patch 0620, and
    upstream's split read any call in an unclosed block as the answer). A call in the think block is held: its text
    stays out of the reasoning until the reply ends on it (a call) or text follows it (reasoning after all). With
    ``hold``, a held call longer than ``hold`` characters is taken as a call at once (``committed``) so a long file
    write streams while it is written; nothing after it can make it reasoning again.

    ``stream_from`` is where the text a call streamer reads begins once known (the answer, or a committed call); a
    reply whose calls it never reaches has them from the end parser."""

    def __init__(self, hold: int | None = None) -> None:
        self.hold = hold
        self.floor = 0                  # calls before this are reasoning
        self.committed: int | None = None
        self.stream_from: int | None = None

    def __call__(self, text: str, finished: bool) -> tuple[str, str]:
        if self.committed is not None:
            at = self.committed
            return text[:at], text[at:].replace(_THINK_CLOSE, "", 1).lstrip()
        close = text.find(_THINK_CLOSE)
        while True:
            call = text.find(_CALL_OPEN, self.floor)
            if close >= 0 and (call < 0 or close < call):
                self.stream_from = close + len(_THINK_CLOSE)
                return text[:close], text[self.stream_from:].lstrip("\n")
            if call < 0:
                held = 0 if finished else max(_partial_tag(text, tag) for tag in (_THINK_CLOSE, _CALL_OPEN))
                return text[:len(text) - held], ""
            end, kind = _call_run(text, call, finished)
            if kind == "calls":
                return text[:call], text[call:end]
            if kind == "reasoning":
                self.floor = max(end, call + len(_CALL_OPEN))
                continue
            if self.hold is not None and len(text) - call > self.hold and close < 0:
                self.committed = self.stream_from = call
                return self(text, finished)
            return text[:call], ""


_KEY, _KEY_END, _VALUE, _VALUE_END = "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"
_VALUE_OPENS = re.compile(r"\s*<arg_value>")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class GlmCallStreamer:
    """GLM's ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`` calls as OpenAI tool-call
    deltas: each call is followed while it is written (its arguments' JSON built a string value as it arrives and a
    typed one when whole, joining to what ``parse_tool_calls`` gives) and sent whole, header and arguments in one delta,
    once its ``</tool_call>`` arrives. A call the reply ends inside (its token limit) is never sent: a client could
    store or run its cut arguments.

    ``feed`` takes the reply's answer so far (text that only grows) and returns the new deltas; ``sent`` counts the
    calls sent. A block it cannot follow (not GLM's markup, a tool the request did not offer) is skipped and stays
    with the end parser."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.tools = tools
        self.known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
        self.schemas = parameter_schemas(tools)
        self.calls: list[dict[str, Any]] = []          # every call whose header was sent, its arguments as sent
        self.skipped: list[str] = []                    # whole blocks it did not follow, for the end parser
        self.pos, self.state, self.start = 0, "outside", 0
        self.open = False                               # the last call's header went out, its </tool_call> has not
        self.schema: dict[str, Any] = {}               # the current call's parameter schemas
        self.value_schema: dict[str, Any] | None = None     # the current typed value's schema, None for text
        self.maybe_null = False                         # a nullable text value that may still be ``null``: held
        self.sent = 0                                   # calls sent whole (the last one may still be open)

    def _args(self, out: list[dict[str, Any]], piece: str) -> None:
        index = len(self.calls) - 1
        self.calls[index]["function"]["arguments"] += piece
        last = out[-1]["tool_calls"][0] if out else None
        if last is not None and last["index"] == index and "id" not in last:      # one delta a call a feed
            last["function"]["arguments"] += piece
        else:
            out.append({"tool_calls": [{"index": index, "function": {"arguments": piece}}]})

    def feed(self, text: str) -> list[dict[str, Any]]:
        self._follow(text)
        whole = len(self.calls) - (1 if self.open else 0)
        out = [{"tool_calls": [{"index": i, "id": c["id"], "type": "function",
                                "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}]}
               for i, c in enumerate(self.calls[self.sent:whole], self.sent)]
        self.sent = whole
        return out

    def _follow(self, text: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while True:
            rest = text[self.pos:]
            if self.state == "outside":
                at = rest.find(_CALL_OPEN)
                if at < 0:
                    return out
                self.start = self.pos + at
                self.pos, self.state = self.start + len(_CALL_OPEN), "name"
            elif self.state == "name":                  # the end parser's name: the text before <arg_key> or the end
                ends = [at for at in (rest.find(_KEY), rest.find(_CALL_CLOSE)) if at >= 0]
                if not ends:
                    if len(rest) > 256:
                        self.state = "skip"
                        continue
                    return out
                name = rest[:min(ends)].strip()
                if not self._callable(name):
                    self.state = "skip"
                    continue
                name, id_ = self.known[name.lower()], f"call_{uuid.uuid4().hex[:24]}"
                self.calls.append({"id": id_, "type": "function", "function": {"name": name, "arguments": ""}})
                self.open = True
                out.append({"tool_calls": [{"index": len(self.calls) - 1, "id": id_, "type": "function",
                                            "function": {"name": name, "arguments": ""}}]})
                self.schema = self.schemas.get(name.lower(), {})
                self.pos, self.state = self.pos + min(ends), "arguments"
            elif self.state == "skip":                  # a block the end parser decides: to its end
                at = rest.find(_CALL_CLOSE)
                if at < 0:
                    return out
                self.pos, self.state = self.pos + at + len(_CALL_CLOSE), "outside"
                self.skipped.append(text[self.start:self.pos])
            elif self.state == "arguments":             # text between arguments is ignored, as the end parser does
                key, close = rest.find(_KEY), rest.find(_CALL_CLOSE)
                if close >= 0 and (key < 0 or close < key):
                    self._args(out, "}" if self.calls[-1]["function"]["arguments"] else "{}")
                    self.pos, self.state, self.open = self.pos + close + len(_CALL_CLOSE), "outside", False
                    self._repaired(text[self.start:self.pos])
                elif key >= 0:
                    self.pos, self.state = self.pos + key + len(_KEY), "key"
                else:
                    return out
            elif self.state == "key":
                end = rest.find(_KEY_END)
                opens = _VALUE_OPENS.match(rest, end + len(_KEY_END)) if end >= 0 else None
                if opens is None:
                    after = rest[end + len(_KEY_END):] if end >= 0 else ""
                    cut = _CALL_CLOSE in (rest if end < 0 else after)
                    if not cut and (end < 0 or not after.strip() or _VALUE.startswith(after.lstrip())):
                        return out                      # the key or its value's opener is still being written
                    self.state = "arguments"            # not an argument the end parser reads the same way
                    continue
                key = rest[:end].strip()
                schema = self.schema.get(key, {})
                self.value_schema = schema if typed_parameter(schema) else None
                self.maybe_null = self.value_schema is None and nullable_text(schema)
                opened = "{" if not self.calls[-1]["function"]["arguments"] else ","
                quote = "" if self.value_schema is not None or self.maybe_null else '"'
                self._args(out, opened + _json(key) + ":" + quote)
                self.pos, self.state = self.pos + opens.end(), "value"
            elif self.state == "value":
                end = rest.find(_VALUE_END)
                if self.value_schema is not None:       # typed: decoded whole, as the end parser does
                    if end < 0:
                        return out
                    self._args(out, _json(decode_parameter(rest[:end], self.value_schema, python=False)))
                elif self.maybe_null:                   # text or null: held while it may still be ``null``
                    written = rest[:end] if end >= 0 else rest[:len(rest) - _partial_tag(rest, _VALUE_END)]
                    if end >= 0 and written == "null":
                        self._args(out, "null")
                    elif end < 0 and "null".startswith(written):
                        return out
                    else:
                        self._args(out, '"')
                        self.maybe_null = False
                        continue
                else:                                   # text: as it arrives, less a tail that may begin the closer
                    written = rest[:end] if end >= 0 else rest[:len(rest) - _partial_tag(rest, _VALUE_END)]
                    piece = _json(written)[1:-1] + ('"' if end >= 0 else "")
                    if piece:
                        self._args(out, piece)
                    if end < 0:
                        self.pos += len(written)
                        return out
                self.pos, self.state = self.pos + end + len(_VALUE_END), "arguments"

    def _repaired(self, block: str) -> None:
        """A whole call whose markup lost an ``<arg_key>`` (``repair_glm_keys``): its arguments as the end parser reads
        it repaired (the streamer skipped the argument it could not see); it has not been sent yet."""

        if block.count(_KEY_END) <= block.count(_KEY):
            return
        calls = parse_tool_calls(block, self.tools)[1] or []
        if len(calls) == 1 and calls[0]["function"]["name"] == self.calls[-1]["function"]["name"]:
            self.calls[-1]["function"]["arguments"] = calls[0]["function"]["arguments"]

    def _callable(self, name: str) -> bool:
        """Whether the end parser reads a block beginning ``name`` as a GLM call to an offered tool."""

        if _GLM_NAME.fullmatch(name) is None or name.lower() not in self.known:
            return False
        try:
            json.loads(name)                            # a name that is JSON (``true``, ``1``) is read as JSON
        except ValueError:
            return True
        return False

    def rest(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The calls the end parser reads in the blocks this streamer skipped (JSON or Qwen markup): sent at the end."""

        return [call for block in self.skipped for call in parse_tool_calls(block, tools)[1] or []]


# -- chat template -------------------------------------------------------------------------
