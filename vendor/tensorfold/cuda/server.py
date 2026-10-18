"""OpenAI server for the CUDA engines: a family's ``cuda_engine`` gives ``eos``, ``generate`` and ``follow``."""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tensorfold.engine import grammar
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import CONTEXT_LIMIT, RequestError, refusal
from tensorfold.server.messages import validate_modalities
from tensorfold.server.probabilities import TokenBytes, probability_options
from tensorfold.server.request_options import heard_effort, parse_numbers, thinking_fields
from tensorfold.server.stopping import stop_options
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.engine.call_gate import CallGate, ThinkBudget, call_format, generate_gated
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server.tools import active_tool_specs, tool_choice_requires_call

from tensorfold.cuda import health
from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.cuda.reply_text import (THINK_CALL_HOLD, GlmCallStreamer, StopStrings, StreamDecoder, ThinkSplit,
                                        close_glm_call, hide_tool_calls, parse_tool_calls)
from tensorfold.cuda.stream_pacing import Playout, smooth_streaming
from tensorfold.cuda.turns import Turns, Yield
from tensorfold.server.text import is_title_request, reasoning_count, split_thinking

# While a GLM tool call is written it is held until whole (a call the reply ends inside is never sent); an idle client
# (upstream #114: one that drops a reply sending nothing) gets an empty delta this often meanwhile
CALL_KEEPALIVE_S = 2.0


# -- requests --------------------------------------------------------------------------------

_MADE = threading.Lock()                # guards the lazily made per-app ``Turns``

_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed")


@dataclass(slots=True)
class PreparedRequest:
    prompt: list[int]
    max_tokens: int
    tools: list[dict[str, Any]]
    thinking: bool
    sampling: Any          # the engine's ``Sampling``, or None for greedy decoding
    ignore_eos: bool = False
    stop: tuple[str, ...] = ()
    vision: Any = None
    grammar: Any = None     # (spec, compiled grammar) of the request's response_format, or None
    think_budget: int = 0   # reply tokens before the server closes a think block the reply leaves open (0: no limit)


def _native_context(model_dir: Path) -> int:
    path = model_dir / "config.json"
    if not path.exists():
        return 0
    config = json.loads(path.read_text())
    text = config.get("text_config") or config
    limit = text.get("max_position_embeddings") or config.get("max_position_embeddings")
    return int(limit) if isinstance(limit, int) and limit > 0 else 0


class App:
    """Serve one engine with sampling and reply-length defaults for requests that omit them."""

    reads_ignore_eos = False            # True where the engine reads ``ignore_eos`` itself; a ``stop_eos`` engine is given it

    def __init__(self, engine, model_dir: Path, served: str, *, default_thinking: bool = False,
                 sampling: dict[str, Any] | None = None, max_tokens: int = 4096,
                 context_window: int | None = None, reasoning_effort: str | None = None, thinking_budget: int = 0,
                 aliases: tuple[str, ...] | list[str] = ()):
        from tokenizers import Tokenizer

        self.engine = engine
        self.vision = getattr(engine, "vision", None)
        self.served = served
        self.aliases = tuple(str(alias).strip() for alias in aliases if str(alias).strip())
        self.model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.template = ChatTemplate(model_dir)
        self.default_thinking = default_thinking
        self.reasoning_effort, self.thinking_budget = reasoning_effort, int(thinking_budget)   # the Mac's defaults
        self.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, **(sampling or {})}
        self.max_tokens = int(max_tokens)
        self.native_context_window = _native_context(model_dir)
        self.context_window = self.native_context_window if context_window is None else int(context_window)
        if self.context_window < 0:
            raise ValueError("context_window must be 0 or a positive token count")
        self.turns = Turns()                # one request at a time where the engine decodes one

    @property
    def model_ids(self) -> list[str]:
        """The ids this endpoint answers to, as the MLX server lists them: ``--name`` first, then each ``--alias``."""

        ids: list[str] = []
        for model_id in (self.served, *getattr(self, "aliases", ())):
            if model_id and model_id not in ids:
                ids.append(model_id)
        return ids

    def reply_model(self, body: Any) -> str:
        """The id a reply names: the one the request asked for when this endpoint answers to it, else ``--name``."""

        asked = body.get("model") if isinstance(body, dict) else None
        return asked if isinstance(asked, str) and asked in self.model_ids else self.served

    def _check_fields(self, body: dict[str, Any]) -> str | None:
        import inspect

        if not isinstance(body, dict):
            return "the request body must be a JSON object"
        try:
            probability_options(body, supported=bool(getattr(self.engine, "supports_logprobs", False)))
        except RequestError as exc:
            return str(exc)
        if body.get("draft", True) is False and "draft" not in inspect.signature(self.engine.generate).parameters:
            return "this model's CUDA engine has no serial switch (\"draft\": false)"
        if not isinstance(body.get("messages", []), list):
            return "messages must be a list"
        problem = grammar.refusal(body)                 # a malformed grammar field, or one beside a required call
        if problem is None and grammar.request_spec(body) and "constraint" not in inspect.signature(
                self.engine.generate).parameters:
            problem = "this model's engine does not enforce structured output"
        return problem

    def _grammars(self) -> grammar.Grammars:
        return grammar.compiler(self, getattr(self, "model_dir", None), self.engine.eos)

    def _engine_capacity(self) -> int | None:
        capacities = []
        for name in ("context_window", "limit"):
            limit = getattr(self.engine, name, None)
            if isinstance(limit, int):
                capacities.append(max(0, limit))
        return min(capacities) if capacities else None

    def _restart(self, need: int, ranks: str = "") -> str:
        """A larger ``--context`` to restart with, only where the startup admission would accept it."""

        largest = (getattr(self.engine, "capacity_plan", None) or {}).get("largest_window")
        if largest is None or need > largest:
            return ""
        return f", or restart{ranks} with --context {need} or more (this memory admits up to {largest})"

    def _context_limit(self) -> int | None:
        limits = [self.context_window] if self.context_window > 0 else []
        capacity = self._engine_capacity()
        if capacity is not None:
            limits.append(capacity)
        return min(limits) if limits else None

    @property
    def effective_context_window(self) -> int | None:
        """Safe prompt-plus-reply capacity; None is unlimited, while zero refuses every prompt."""

        return self._context_limit()

    def _requested_tokens(self, body: dict[str, Any]) -> int:
        for name in ("max_tokens", "max_completion_tokens"):
            value = body.get(name)
            if value is not None:
                try:
                    int(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RequestError(f"{name} must be an integer token count") from exc
        return max(1, int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens))

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        from jinja2.exceptions import TemplateError

        validate_modalities(body)
        ToolCallPolicy(body)
        ignore_eos, stop = stop_options(body)
        max_tokens = self._requested_tokens(body)
        try:
            tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        kwargs = body.get("chat_template_kwargs")
        if kwargs is None:                   # absent or null: the server's defaults
            kwargs = {}
        elif not isinstance(kwargs, dict):   # [], "", false and 0 included
            raise RequestError("chat_template_kwargs must be a JSON object or null")
        kwargs = dict(kwargs)
        # reasoning_effort and enable_thinking as the Mac server reads them: the template hears an effort when thinking
        levels = getattr(self.template, "efforts", frozenset())
        fields = thinking_fields(body, levels)
        kwargs.pop("enable_thinking", None)
        kwargs.pop("reasoning_effort", None)
        thinking = bool(fields.get("enable_thinking", self.default_thinking))
        effort = heard_effort(fields.get("reasoning_effort"), getattr(self, "reasoning_effort", None), levels)
        if thinking and effort:
            kwargs["reasoning_effort"] = effort
        budget = parse_numbers({"thinking_budget": body.get("thinking_budget")})["thinking_budget"]
        budget = int(budget or getattr(self, "thinking_budget", 0)) if chat and thinking else 0     # 0: the default
        spec = grammar.request_spec(body)
        top = probability_options(body, supported=bool(getattr(self.engine, "supports_logprobs", False)))
        if top is not None:
            if not chat or body.get("stream") or thinking or tools or stop or spec is not None or budget:
                raise RequestError("logprobs support nonstreamed text chat with thinking off, without tools, "
                                   "stop strings or structured output")
            if not hasattr(self, "_probability_decoder"):
                self._probability_decoder = TokenBytes(self.tok)
        compiled = (spec, self._grammars().compile(spec)) if spec is not None else None
        if chat:
            if not isinstance(body.get("messages"), list):
                raise RequestError("messages must be a list")
            from tensorfold.server.prompts import has_images, prepare_images, tool_media_parts

            def render(messages: list[dict[str, Any]], **images: bool) -> str:   # text renders as it always has
                try:
                    return self.template.render(messages, tools=tools, enable_thinking=thinking, extra=kwargs,
                                                **images)
                except TemplateError as exc:     # the checkpoint's template refuses the request (``raise_exception``)
                    raise RequestError(f"the chat template rejected the request: {exc}") from exc

            # no --vision: a tool result's pictures as the template's own "unable to process" reminder (GLM)
            reminded = (tool_media_parts(body["messages"]) if getattr(self, "tool_media_reminders", False)
                        and getattr(self, "vision", None) is None else None)
            if reminded is not None:
                text = render(reminded, allow_images=True)
            elif has_images(body["messages"]):
                rendered = prepare_images(self.vision, body["messages"],
                                          lambda messages: render(messages, allow_images=True),
                                          context_limit=self._context_limit())
                return PreparedRequest(rendered.tokens, max_tokens, tools, thinking,
                                       self.sampling_for(body, rendered.tokens), ignore_eos=ignore_eos, stop=stop,
                                       vision=rendered.vision, grammar=compiled, think_budget=budget)
            else:
                text = render(body["messages"])
            prompt = self.tok.encode(text, add_special_tokens=False).ids
        elif isinstance(body.get("prompt"), list):       # token ids (vLLM's and OpenAI's form): served as given
            prompt = self.token_ids(body["prompt"])
        else:
            text = body.get("prompt")
            if not isinstance(text, str):
                raise RequestError("prompt must be a string or a list of token ids")
            prompt = self.tok.encode(text, add_special_tokens=_flag(body, "add_special_tokens", False)).ids
        if not prompt:
            raise RequestError("rendered prompt is empty")
        # sampling is resolved here, so a malformed control is refused before a stream opens
        return PreparedRequest(prompt, max_tokens, tools, thinking, self.sampling_for(body, prompt),
                               ignore_eos=ignore_eos, stop=stop, grammar=compiled, think_budget=budget)

    def token_ids(self, value: Any, field: str = "prompt") -> list[int]:
        """Token ids as a request gives them (a list, or a list holding one list); RequestError outside the vocabulary."""

        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        if not isinstance(value, list) or any(type(t) is not int for t in value):
            raise RequestError(f"{field} must be a list of integer token ids (one prompt a request)")
        size = getattr(self.tok, "get_vocab_size", None)
        vocab = size(with_added_tokens=True) if size is not None else None
        if any(t < 0 or (vocab is not None and t >= vocab) for t in value):
            top = "" if vocab is None else f" to {vocab - 1}"
            raise RequestError(f"{field} token ids must be in the vocabulary's range 0{top}")
        return list(value)

    def tokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        """vLLM's ``/tokenize``: a prompt's ids (``add_special_tokens`` as vLLM, default true), or ``messages``' as the
        chat route renders them (``add_generation_prompt``, default true)."""

        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        if "messages" in body:
            fields = dict(body)
            if "add_generation_prompt" in body:
                kwargs = body.get("chat_template_kwargs")
                kwargs = {} if kwargs is None else kwargs
                if not isinstance(kwargs, dict):
                    raise RequestError("chat_template_kwargs must be a JSON object or null")
                fields["chat_template_kwargs"] = {**kwargs,
                                                  "add_generation_prompt": _flag(body, "add_generation_prompt", True)}
            ids = self._prepare(fields, True).prompt
        else:
            text = body.get("prompt")
            if not isinstance(text, str):
                raise RequestError("prompt must be a string (or send messages)")
            ids = self.tok.encode(text, add_special_tokens=_flag(body, "add_special_tokens", True)).ids
        limit = self._context_limit()
        return {"count": len(ids), "max_model_len": limit if limit is not None else self.native_context_window,
                "tokens": [int(t) for t in ids]}

    def detokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        """vLLM's ``/detokenize``: the text of ``tokens``, special tokens included."""

        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        return {"prompt": self.tok.decode(self.token_ids(body.get("tokens"), "tokens"), skip_special_tokens=False)}

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Why the request cannot run, or None; rendered before a stream's headers are sent."""

        problem = self._check_fields(body)
        if problem:
            return problem
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        limit = self._context_limit()
        if limit is not None and len(prepared.prompt) >= limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            native = f" (model window: {self.native_context_window} tokens)" if self.native_context_window else ""
            return (f"{CONTEXT_LIMIT} {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and leaves "
                    f"no room for a reply in the server's {limit}-token {kind}{native}, which exceeds the context "
                    f"window; shorten the prompt"
                    f"{self._restart(len(prepared.prompt) + 1)}")
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        if limit is not None and asked and len(prepared.prompt) + prepared.max_tokens > limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            return (f"{CONTEXT_LIMIT} {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and "
                    f"requests {prepared.max_tokens} reply tokens, which exceeds the context window (the server's "
                    f"{limit}-token {kind}); reduce the prompt or reply length"
                    f"{self._restart(len(prepared.prompt) + prepared.max_tokens)}")
        return None

    def prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        problem = self._check_fields(body)
        if problem:
            raise refusal(problem)
        prepared = self._prepare(body, chat)
        problem = self.check(body, prepared=prepared)
        if problem:
            raise refusal(problem)
        limit = self._context_limit()
        if limit is not None:
            prepared.max_tokens = min(prepared.max_tokens, limit - len(prepared.prompt))
        return prepared

    def admit(self, body: dict[str, Any], chat: bool, prepared: PreparedRequest) -> None:
        """Refuse a request with ``CapacityError`` while the scheduler's admission cap holds, before a streamed reply's
        200 and headers go out (``Scheduler.submit`` checks again, but by then a stream can only carry the error as an
        event, with status 200, which a client's retry logic never sees)."""

        check = getattr(getattr(self.engine, "scheduler", None), "_check_admission", None)
        if check is not None:
            check(body.get("priority") == "background" or (chat and is_title_request(body.get("messages"), prepared.tools)))

    def sampling_for(self, body: dict[str, Any], prompt: list[int]):
        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy; RequestError if malformed."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        fields = parse_numbers({k: body[k] for k in _SAMPLING_FIELDS if body.get(k) is not None})
        temp = float(fields.get("temperature", self.sampling["temperature"]))
        if temp <= 0:
            return None
        seed = fields.get("seed")
        top_k = fields.get("top_k", self.sampling["top_k"])
        top_p = fields.get("top_p", self.sampling["top_p"])
        min_p = fields.get("min_p", self.sampling.get("min_p", 0.0))
        return Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p),
                        float(min_p))

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        """One reply; once ``cancelled()`` holds, a waiting request raises ``RequestCancelled`` unstarted, a running one stops at its next round and raises it after ``generate``."""

        arrived = time.perf_counter()
        prepared = prepared if prepared is not None else self.prepare(body, chat)
        prompt, max_tokens = prepared.prompt, prepared.max_tokens
        tools, thinking = prepared.tools, prepared.thinking
        policy = ToolCallPolicy(body)
        sampling = prepared.sampling
        # end tokens end the reply and stay out of its text, unless it asks ignore_eos of an engine that reads it
        takes_stop_eos = "stop_eos" in inspect.signature(self.engine.generate).parameters
        ends = () if prepared.ignore_eos and (self.reads_ignore_eos or takes_stop_eos) else tuple(self.engine.eos)
        stops = StopStrings(prepared.stop, self.tok, ends)
        out: list[int] = []
        sent = {"reasoning": 0, "content": 0}
        stopped = {"client": False, "stop": False}
        failed: list[Exception] = []
        stream = StreamDecoder(self.tok, ends)
        # a streamed reply's GLM calls go out while they are written (a long file write would otherwise send nothing
        # for minutes, past the idle cut of clients such as LiteLLM and Node's fetch); one call a reply waits
        glm_calls = bool(tools) and self._glm_calls()
        calls_stream = GlmCallStreamer(tools) if glm_calls and body.get("stream") and not policy.single else None
        quiet_since = [time.monotonic()]     # the last delta sent while a call is held (CALL_KEEPALIVE_S)
        # other families' calls stream as argument deltas while written (as on the Mac); one-call requests keep the
        # end parser
        xml_stream = ToolCallStreamer(tools) if tools and not glm_calls and not policy.single else None
        answer_raw = [""]
        closing = [""]                       # what closes a GLM call the reply ended inside on an end token (#22)
        # GLM's calls written inside the think block: calls when the reply ends on them (``ThinkSplit``)
        think = (ThinkSplit(THINK_CALL_HOLD if calls_stream is not None else None) if glm_calls and chat and thinking
                 else None)

        def split(raw: str, finished: bool) -> tuple[str, str]:
            if not (chat and thinking):
                return "", raw
            return think(raw, finished) if think is not None else split_thinking(raw, finished=finished)

        def call_text(raw: str) -> str:
            """The text the call streamer reads: the answer, or a long call taken from the think block; "" before."""

            if think is None:
                return raw
            return raw[think.stream_from:] if think.stream_from is not None else ""

        def visible(finished: bool) -> tuple[str, str, str]:
            raw = stream.final() if finished else stream.text
            # stop strings match the generated text, reasoning included, before it is split (as on the Mac)
            raw = stops.visible(raw, partial=not finished) if stops.strings else raw
            raw = raw + closing[0] if finished else raw
            reasoning, answer = split(raw, finished)
            answer_raw[0] = answer
            if tools:
                answer = (policy.content(answer, finished=finished) if policy.single
                          else hide_tool_calls(answer, finished=finished))
            return reasoning, answer, raw

        serving: list[Any] = [None]

        # TF_GLM_STREAM_SMOOTH=1: a streamed reply's tokens go out one event each at a steady pace from a short
        # playout buffer (``stream_pacing.Playout``), so drafted bursts and other requests' prompt fills do not show
        playout = Playout(emit, cancelled=cancelled) if body.get("stream") and smooth_streaming() else None

        def send(delta: dict[str, Any], paced: bool) -> bool:
            """Write an event (through the playout queue when smoothing): False once the client has gone."""

            if playout is None:
                return emit(delta)
            playout.put(delta, paced)
            return not playout.gone

        def text_delta(tokens: list[int]) -> tuple[dict[str, Any], str]:
            """The reasoning and answer text these tokens add (the part not sent yet), and the raw text so far."""

            stream.add(tokens)
            reasoning, answer, raw = visible(False)
            delta: dict[str, Any] = {}
            if len(reasoning) > sent["reasoning"]:
                delta["reasoning_content"] = reasoning[sent["reasoning"]:]
                sent["reasoning"] = len(reasoning)
            if len(answer) > sent["content"]:
                delta["content"] = answer[sent["content"]:]
                sent["content"] = len(answer)
            return delta, raw

        def on_tokens(new: list[int]) -> bool:
            # True stops the engine after this round (two-rank GLM: every rank after the next one, ``StopVote``);
            # an engine still running keeps calling and gets True
            if playout is not None and playout.gone:
                stopped["client"] = True
            if stopped["client"] or stopped["stop"] or failed:
                return True
            try:
                if stops.strings:
                    kept = []
                    for token in new:             # token by token: the round's width cannot move the cut
                        kept.append(token)
                        out.append(token)
                        if stops.hit(out):
                            stopped["stop"] = True
                            break
                    new = kept
                else:
                    out.extend(new)
                # smooth streaming: token by token into the playout queue, unless the reply ends in this round (then
                # the queue is flushed: the end is never later)
                ending = stopped["stop"] or (bool(new) and (new[-1] in ends or len(out) >= max_tokens))
                if playout is not None and ending:
                    playout.flush()
                pieces = [[t] for t in new] if playout is not None and not ending and len(new) > 1 else [new]
                raw, wrote = "", False
                for piece in pieces:
                    delta, raw = text_delta(piece)
                    if not delta:
                        continue
                    if not send(delta, True):
                        stopped["client"] = True
                        break
                    wrote = True
                parts: list[dict[str, Any]] = []
                if calls_stream is not None and not stopped["client"]:
                    parts += calls_stream.feed(call_text(raw))
                    if parts or wrote:
                        quiet_since[0] = time.monotonic()
                    elif calls_stream.open and time.monotonic() - quiet_since[0] >= CALL_KEEPALIVE_S:
                        parts.append({})          # a call held until whole: an empty delta keeps idle clients waiting
                        quiet_since[0] = time.monotonic()
                if not all(send(part, False) for part in parts):
                    stopped["client"] = True
                if xml_stream is not None and not stopped["client"]:
                    for call_delta in xml_stream.feed(answer_raw[0]):     # never the reasoning
                        if not send(call_delta, False):
                            stopped["client"] = True
                            break
                if not stopped["client"] and cancelled is not None and cancelled():   # every round, text or not
                    stopped["client"] = True
                if serving[0] is not None:
                    serving[0].saw()
            except Exception as exc:        # noqa: BLE001  raised after generate returns, never into the engine
                failed.append(exc)
                return True
            return stopped["client"] or stopped["stop"]

        draft = body.get("draft", True) is not False
        gate = self._call_gate(prompt, tools) if tools and tool_choice_requires_call(body.get("tool_choice")) else None

        options: dict[str, Any] = {} if draft else {"draft": False}
        probabilities = None
        if body.get("logprobs"):
            from tensorfold.engine.probabilities import Probabilities

            probabilities = Probabilities(body.get("top_logprobs") or 0, len(prompt), max_tokens)
            options["probabilities"] = probabilities
        if takes_stop_eos:
            options["stop_eos"] = not prepared.ignore_eos
        shaped = prepared.grammar is not None or prepared.think_budget > 0
        think_end = self.tok.token_to_id("</think>") if chat and thinking and shaped else None
        budget = self._think_budget(prepared, think_end)
        # priority "background" (or a session-title request): after the others, as on the Mac
        background = body.get("priority") == "background" or (chat and is_title_request(body.get("messages"), tools))
        concurrent = getattr(self.engine, "concurrent", False)
        turns = None if concurrent else self._turns()
        if concurrent and background and "background" in inspect.signature(self.engine.generate).parameters:
            options["background"] = True            # the engine's scheduler orders its lanes and prompts
        # one engine at a time: a background reply yields between rounds (not on two ranks, which decode to the end)
        yielding = background and turns is not None and getattr(self.engine, "tp", 1) == 1
        gates = [g for g in (gate, budget, Yield(turns) if yielding else None) if g is not None]

        cached: list[int] = []              # the prompt tokens the first run found cached (usage's cached_tokens)

        def generate(ids: list[int], count: int, feed: Callable[[list[int]], bool]) -> Any:
            if yielding and cached:                 # a run after a cut: foreground requests waiting go first
                turns.give()
                turns.take(True, cancelled)
            extra = dict(options)
            if prepared.grammar is not None:    # response_format: a fresh grammar state, after </think> when thinking
                spec, compiled = prepared.grammar
                # a run after the thinking budget's close (it ends at </think> under a grammar) starts in the grammar
                end = None if think_end is None or think_end in ids[len(prompt):] else think_end
                extra["constraint"] = self._grammars().constraint(compiled, think_end=end, spec=spec)
            if prepared.vision is not None:          # a gate's continuation keeps the images, positions extended
                same = list(ids) == list(prepared.vision.token_ids)
                if same:
                    extra["vision"] = prepared.vision
                elif hasattr(prepared.vision, "continued"):     # a frontend that extends its own prompts
                    extra["vision"] = prepared.vision.continued(ids)
                else:
                    from tensorfold.vision.qwen_processing import continued

                    extra["vision"] = continued(prepared.vision, ids, self.vision.frontend.config)
            # generate_gated wraps on_tokens: attach the poll to the actual callback
            # reaching the scheduler, without invoking its streaming/pacing state.
            feed.cancelled = cancelled
            stats = self.engine.generate(ids, count, sampling, feed, **extra)
            if not cached:
                cached.append(int((stats or {}).get("cached") or 0))
            return stats

        # an engine that decodes concurrent requests together (``concurrent``) takes them as they come
        if turns is not None:
            turns.take(background, cancelled)
        try:
            if cancelled is not None and cancelled():                # the client left while this request waited
                raise RequestCancelled("the client left before the request started")
            with health.of(self).running(len(prompt), out, arrived) as request:  # /health reads ``out``; rounds never call in
                serving[0] = request
                try:
                    stats = request.stats = generate_gated(generate, prompt, max_tokens, gates, on_tokens)
                except BaseException:
                    if playout is not None:
                        playout.abort()
                    raise
                finally:
                    serving[0] = None
                if playout is not None:             # what is still queued goes out now, before the reply's end
                    (playout.abort if stopped["client"] or failed else playout.close)()
                    if playout.error is not None:
                        failed.append(playout.error)
                    elif playout.gone:
                        stopped["client"] = True
        finally:
            if turns is not None:
                turns.give()
        if failed:
            raise failed[0]
        if stopped["client"]:                                        # as the Mac server: nothing more is written
            raise RequestCancelled("the client left during the reply")
        stats = {**(stats or {}), "token_sha": token_sha(out)}
        # the model's end token (not a stop string or the token limit) ended the reply: a call it ended inside is
        # closed when it then parses (#22), else its markup is the reply's text; it is never dropped
        ended = bool(out) and out[-1] in ends and not stopped["stop"]
        if ended and glm_calls:
            final_raw = stream.final()
            closing[0] = close_glm_call(stops.visible(final_raw) if stops.strings else final_raw, tools)
        reasoning, answer, raw = visible(True)
        final: dict[str, Any] = {}
        if len(reasoning) > sent["reasoning"]:
            final["reasoning_content"] = reasoning[sent["reasoning"]:]
        text = stops.visible(self.tok.decode([t for t in out if t not in ends], skip_special_tokens=False)) + closing[0]
        raw_answer = split(text, True)[1]
        content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)
        content = policy.content(content) if tools else content
        call_deltas, streamed = [], 0
        # the calls streamed as sent, then any the streamer did not follow; calls held in the think block to the end
        # (never streamed) are the end parser's, sent whole
        if calls_stream is not None and (think is None or think.stream_from is not None):
            call_deltas = calls_stream.feed(call_text(raw))
            streamed = calls_stream.sent
            calls = [*calls_stream.calls[:streamed], *calls_stream.rest(tools)] or None
        # a call the token limit cut (the reply ended inside it): never sent, nor its markup as content; one the
        # model's end token left open that would not parse, its markup as text (``closing`` closed the others)
        cut = calls_stream is not None and calls_stream.open and not ended
        if cut:
            content = answer.strip()
        elif tools and not ended and (at := content.rfind("<tool_call>")) >= 0 and "</tool_call>" not in content[at:]:
            content, cut = content[:at].rstrip(), True     # the same for the end parser: an unclosed call is cut text
        tail = content[sent["content"]:] if content.startswith(answer[:sent["content"]]) else ""
        if tail:
            final["content"] = tail
        finish = "tool_calls" if calls and not cut else ("stop" if stopped["stop"] or (out and out[-1] in ends)
                                                         else "length")
        if body.get("return_token_ids"):              # the reply's ids in the "tensorfold" block, for exactness checks
            stats = {**(stats or {}), "token_ids": [int(t) for t in out]}
        logprobs = (self._probability_decoder.format(probabilities.emitted(out), ends)
                    if probabilities is not None else None)
        # the calls already sent as deltas; the handler sends the rest (a call the streamer could not follow)
        if xml_stream is not None and xml_stream.streamed:
            streamed = xml_stream.index + 1
        return {"final": final, "calls": calls, "call_deltas": call_deltas, "calls_streamed": streamed,
                "finish": finish, "content": content, "reasoning": reasoning,
                **({"logprobs": logprobs} if logprobs is not None else {}),
                "prompt_tokens": len(prompt), "completion_tokens": len(out), "cached_tokens": (cached or [0])[0],
                "reasoning_tokens": reasoning_count(out, self.tok.token_to_id("</think>") if chat and thinking else None),
                "stats": stats}

    def _turns(self) -> Turns:
        """The engine's turns (one request at a time, background ones last), made on first use."""

        with _MADE:
            return self.__dict__.setdefault("turns", Turns())

    def _think_budget(self, prepared: PreparedRequest, think_end: int | None) -> ThinkBudget | None:
        """The Mac's thinking budget: the cut becomes a newline, </think> and a blank line (</think> alone under a grammar)."""

        if prepared.think_budget <= 0 or think_end is None:
            return None
        close = [*self.tok.encode("\n", add_special_tokens=False).ids, think_end]
        if prepared.grammar is None:
            close += self.tok.encode("\n\n", add_special_tokens=False).ids
        return ThinkBudget(prepared.think_budget, close, think_end)

    def _glm_calls(self) -> bool:
        """Whether this model writes GLM's ``<arg_key>``/``<arg_value>`` calls (streamed while they are written): its
        tokenizer has those marks as tokens."""

        lookup = getattr(self.tok, "token_to_id", None)
        return lookup is not None and all(lookup(mark) is not None for mark in ("<tool_call>", "<arg_key>",
                                                                                "</arg_value>"))

    def _call_gate(self, prompt: list[int], tools: list[dict[str, Any]]) -> CallGate:
        """The gate a required tool call needs, from this template's call markup and the rendered prompt."""

        if not hasattr(self, "_form"):
            probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}]}]
            try:
                text = self.template.render(probe, tools=None, enable_thinking=False)
            except Exception:  # noqa: BLE001 - a template that renders no calls: the opener alone
                text = ""
            openers = [o for o in ("<tool_call>", "<|tool_call>") if self.tok.token_to_id(o) is not None]
            self._form = call_format(text, "tfprobe_fn", openers) or ((openers[0], None, None) if openers else None)
        if self._form is None:
            raise RequestError('tool_choice "required" or a named function needs a chat template that marks tool calls '
                               '(<tool_call> or <|tool_call>), and this one does not: send "auto"')
        opener, lead, tail = self._form
        eos = set(self.engine.eos)

        def text(token: int) -> str:
            return self.tok.decode([token], skip_special_tokens=False)

        def blank(token: int) -> bool:
            return token not in eos and not text(token).strip()

        think = [-1 if self.tok.token_to_id(t) is None else self.tok.token_to_id(t) for t in ("<think>", "</think>")]
        names = [str((t.get("function") or t).get("name") or "") for t in tools] if lead is not None else []
        return CallGate.after_prompt(prompt, self.tok.token_to_id(opener), blank, think_open=think[0],
                                     think_end=think[1], text=text, lead=lead or "", names=names, tail=tail or "",
                                     encode=lambda t: list(self.tok.encode(t, add_special_tokens=False).ids))


def _flag(body: dict[str, Any], name: str, default: bool) -> bool:
    value = body.get(name, default)
    if not isinstance(value, bool):
        raise RequestError(f"{name} must be a boolean")
    return value


def token_sha(tokens: list[int]) -> str:
    """A reply's token ids, hashed as the Mac server does: drafted and ``"draft": false`` replies must match."""

    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


from tensorfold.cuda.http import Server, make_handler, serve, usage_of  # noqa: E402,F401  (the HTTP side)
