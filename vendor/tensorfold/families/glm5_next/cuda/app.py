"""GLM request routes validate context before streaming and render an empty think block without a reasoning-effort line when thinking is off."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest, RequestError
from tensorfold.server.errors import CONTEXT_LIMIT
from tensorfold.families.glm5_next.cuda.kept_reasoning import KeptReasoning
from tensorfold.families.glm5_next.prompts import agent_history, thinking_off


class ThinkingOffTemplate:
    """The checkpoint's chat template, rendered as GLM-5.3's thinking-off template renders it when thinking is off
    (``prompts.thinking_off``, as the Mac's tokenizer renders it), an agent's history made renderable first
    (``prompts.agent_history``), with the reasoning of tool-calling steps the client dropped put back (``kept``,
    ``kept_reasoning.KeptReasoning``; None: off)."""

    def __init__(self, inner, kept: KeptReasoning | None = None, clear_thinking: bool = False) -> None:
        self.inner = inner
        self.efforts = getattr(inner, "efforts", frozenset())
        self.kept = kept
        self.clear_thinking = clear_thinking

    def render(self, messages, *, tools, enable_thinking, extra=None, allow_images: bool = False) -> str:
        if self.kept is not None:
            messages = self.kept.restore(messages)
        # earlier turns keep their reasoning unless the request says otherwise, as in zai-org's current template
        extra = {"clear_thinking": self.clear_thinking, **(extra or {})}
        text = self.inner.render(agent_history(messages), tools=tools, enable_thinking=enable_thinking, extra=extra,
                                 allow_images=allow_images)
        return text if enable_thinking else thinking_off(text)


# the checkpoint's template answers a picture or clip with this reminder; with --vision it writes GLM's markers instead
# (Glm5NextProcessor's layout), and every text-only prompt renders as before. A part's ``tf_mark`` (server.prompts'
# MediaMarks, taken out again before the prompt is encoded) goes before its markers
_NO_MEDIA = ('{{- "<reminder>You are unable to process this " ~ media_type ~ " because you don\'t have multi-modal '
             'input ability. Try different methods.</reminder>" }}')
_MEDIA = ("{%- if item.tf_mark is defined -%}{{- item.tf_mark -}}{%- endif -%}"
          "{%- if media_type == 'image' -%}<|begin_of_image|><|image|><|end_of_image|>"
          "{%- elif media_type == 'video' -%}<|begin_of_video|><|video|><|end_of_video|>"
          "{%- else -%}" + _NO_MEDIA + "{%- endif -%}")


def media_template(inner, model_dir) -> None:
    """Swap the checkpoint template's no-media reminder for GLM's image and video markers."""

    source = (model_dir / "chat_template.jinja").read_text()
    mark = "{%- if item.tf_mark is defined -%}{{- item.tf_mark -}}{%- endif -%}"
    if source.count(_NO_MEDIA) == 1:
        source = source.replace(_NO_MEDIA, _MEDIA)
    elif all(source.count("{{- emit_" + kind + "() -}}") == 1 for kind in ("image", "video")):
        # Native multimodal templates still need TensorFold's transient marks:
        # tool results may be sorted, and MediaMarks maps images to that order.
        for kind in ("image", "video"):
            call = "{{- emit_" + kind + "() -}}"
            source = source.replace(call, mark + call)
    else:
        raise ValueError("--vision: unsupported GLM media template")
    inner.template = inner.template.environment.from_string(source)


class GlmApp(App):
    reads_ignore_eos = True             # ``run`` hands it to the engine's request
    # without --vision, a tool result's pictures render as the template's own reminder (issue #28): a user's are refused
    tool_media_reminders = True

    def __init__(self, engine, model_dir, served: str, **kwargs: Any) -> None:
        super().__init__(engine, model_dir, served, **kwargs)
        if self.vision is not None:
            media_template(self.template, Path(model_dir))
        clear = os.environ.get("TF_GLM_CLEAR_THINKING", "0").strip() or "0"
        if clear not in ("0", "1"):
            raise ValueError(f"TF_GLM_CLEAR_THINKING={clear!r}: expected 0 or 1")
        self.template = ThinkingOffTemplate(self.template, KeptReasoning.from_env(), clear_thinking=clear == "1")

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Validate the rendered prompt plus max_tokens against the engine context limit before streaming."""

        from .trial_control import parse
        try:
            trials = parse(body.get("tf_trials"))
            if trials["spec"] and int(body.get("max_tokens", 0)) not in range(1, 1025):
                return "spec diagnostics require max_tokens from 1 to 1024"
        except ValueError as exc:
            return str(exc)
        problem = self._check_fields(body)
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem or super().check(body, prepared=prepared)
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        prompt = len(prepared.prompt)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return super().check(body, prepared=prepared)
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        # v0.6.0 (68c6e35, Apache-2.0): OpenAI's wording, so prepare refuses it as context_length_exceeded
        return (f"{CONTEXT_LIMIT} {limit} tokens: this request needs a {need}-token context ({detail}), which exceeds "
                f"the context window this server was started for; shorten the prompt or reply"
                f"{self._restart(need, ' both ranks')}")

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        model = str(body.get("model") or "")
        self.engine.request.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
        from .trial_control import parse
        self.engine.request.trial = parse(body.get("tf_trials"))
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        result = super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
        kept = getattr(self.template, "kept", None)
        if chat and kept is not None and result.get("calls") and result.get("finish") == "tool_calls":
            # the calls' reasoning, for a client that will not send it back
            kept.remember(body.get("messages"), result["calls"], result.get("reasoning"))
        return result
