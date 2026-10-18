"""Prepare bounded text or image prompts before queueing GPU work."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from tensorfold.server.errors import CapacityError, RequestError, refusal
from tensorfold.server.messages import _normalize_tool_call_arguments, normalize_messages


@dataclass
class RenderedPrompt:
    tokens: list[int]
    history_len: int = 0
    vision: Any = None


def has_images(messages):
    return any(isinstance(m, dict) and isinstance(m.get('content'), list)
               and any(isinstance(p, dict) and p.get('type') in ('image_url', 'video_url') for p in m['content'])
               for m in messages or [])


def tool_media_parts(messages):
    """For a server without a vision frontend whose template answers media with its own reminder (GLM-5.3's "You are
    unable to process this image ..."): messages whose only pictures and clips are in tool results, those parts as the
    template's bare media parts (never fetched or decoded), so an agent session that keeps a screenshot in its history
    goes on (issue #28); None when there are none, or a user (or other) message has some."""

    if not has_images(messages):
        return None
    out = []
    for m in messages:
        content = m.get('content') if isinstance(m, dict) else None
        if not (isinstance(content, list)
                and any(isinstance(p, dict) and p.get('type') in ('image_url', 'video_url') for p in content)):
            out.append(m)
            continue
        if m.get('role') != 'tool':
            return None
        out.append({**m, 'content': [{'type': p['type'][:-len('_url')]}
                                     if isinstance(p, dict) and p.get('type') in ('image_url', 'video_url') else p
                                     for p in content]})
    return out


IMAGE_SLOTS = threading.BoundedSemaphore(16)      # requests decoding and processing images at once (host memory)
IMAGE_WAITERS = threading.BoundedSemaphore(128)   # requests waiting for a slot; past that, refused at once
IMAGE_WAIT_S = 60.0


def image_slot():
    """Hold one image-preparation slot, or refuse with 503 when the queue is full or the wait runs out."""
    if not IMAGE_WAITERS.acquire(blocking=False):
        raise CapacityError('image request queue is full; retry shortly')
    try:
        if not IMAGE_SLOTS.acquire(timeout=IMAGE_WAIT_S):
            raise CapacityError('image processing capacity is busy; retry shortly')
    finally:
        IMAGE_WAITERS.release()
    return IMAGE_SLOTS


class MediaMarks:
    """Each picture's and clip's place in the rendered prompt: a template may sort tool results (GLM-5.3's orders
    them by their calls' ids) or leave one out (``agent_history``), so the media are matched to the markers by a mark
    rendered before each (``tf_mark``, a per-request nonce), then the marks are taken out: the prompt is the one
    rendered without them.

    Every request with media is marked, not only one whose pictures sit in tool results: the marks are also what
    tells a marker the template wrote from one the conversation merely quotes (a pasted template, a log, this
    server's own refusal), which the frontend's ``escape_quoted`` neutralizes."""

    def __init__(self, template) -> None:
        import re
        import secrets

        nonce = secrets.token_hex(8)
        self.pattern = re.compile("\x00tf-media-" + nonce + r":(\d+)\x00")
        self.count = 0
        for message in template:
            for part in message.get("content") if isinstance(message.get("content"), list) else ():
                if isinstance(part, dict) and part.get("type") in ("image", "video"):
                    part["tf_mark"] = f"\x00tf-media-{nonce}:{self.count}\x00"
                    self.count += 1

    @staticmethod
    def wanted(template) -> bool:
        """Whether the request carries pictures or clips at all (a user's and a tool result's alike)."""

        return any(isinstance(m.get("content"), list)
                   and any(isinstance(p, dict) and p.get("type") in ("image", "video") for p in m["content"])
                   for m in template)

    def read(self, text: str, escape=None) -> tuple[str, list[int]]:
        """(the text without its marks, the media's indices in the order the template rendered them).

        ``escape`` (a frontend's ``escape_quoted``) is given the text and where this request's marks end, so it can
        neutralize the markers the conversation only quotes -- without it, a chat that keeps such text beside a
        picture is refused on every turn that carries one."""

        found = [(m.start(), m.end(), int(m.group(1))) for m in self.pattern.finditer(text)]
        order = [k for _, _, k in found]
        if any(k >= self.count for k in order):
            raise ValueError("the rendered prompt's media marks do not match the request's media")
        if escape is not None:
            text = escape(text, [e for _, e, _ in found])
        return self.pattern.sub("", text), order


def prepare_images(frontend, messages, render, *, context_limit=None):
    from tensorfold.vision.images import DEFAULT_LIMITS, ImageInputError, load_images, split_images

    if frontend is None:
        raise RequestError('image input requires a supported vision checkpoint served with --vision')
    allow_urls = bool(getattr(frontend, 'allow_urls', False))
    videos = bool(getattr(frontend, 'videos', False))       # a frontend that encodes video frames too
    # a frontend's own limits (GLM: up to 50 pictures and 4 clips a request); the shared ones otherwise
    limits = getattr(frontend, 'image_limits', None) or DEFAULT_LIMITS
    # a frontend that renders tool results' media in place (GLM: inside their <tool_response>)
    tool_media = bool(getattr(frontend, 'tool_media', False))
    try:
        template, sources = split_images(messages, limits=limits, allow_urls=allow_urls, allow_videos=videos,
                                         max_videos=getattr(frontend, 'max_videos', None), tool_media=tool_media)
        text = None
        if tool_media and MediaMarks.wanted(template):
            marks = MediaMarks(template)
            text, order = marks.read(render(template), escape=getattr(frontend, 'escape_quoted', None))
            sources = [sources[k] for k in order]          # as rendered: sorted, or left out with their result
    except (ImageInputError, ValueError) as exc:
        raise RequestError(str(exc)) from exc
    slot = image_slot()
    try:
        from tensorfold.vision.images import ImageSource

        pictures = [s for s in sources if isinstance(s, ImageSource)]
        if hasattr(frontend, 'load_images'):              # a frontend that fits each picture as it decodes it
            images = frontend.load_images(pictures)
        else:
            images = load_images(pictures, limits=limits, allow_urls=allow_urls)
        clips = [s for s in sources if not isinstance(s, ImageSource)]
        if clips:
            if hasattr(frontend, 'load_videos'):             # a frontend with its own frame sampling
                clips = frontend.load_videos(clips)
            else:
                from tensorfold.vision.videos import load_videos

                clips = load_videos(clips, frontend.video_size, allow_urls=allow_urls)
            prepared = frontend.prepare(text if text is not None else render(template), images, videos=clips,
                                        max_prompt_tokens=context_limit)
        else:
            prepared = frontend.prepare(text if text is not None else render(template), images,
                                        max_prompt_tokens=context_limit)
    except (ImageInputError, ValueError, ImportError) as exc:
        raise refusal(str(exc)) from exc               # a prompt past the window: context_length_exceeded
    finally:
        slot.release()
    return RenderedPrompt(list(prepared.token_ids), vision=prepared)


def prepare_prompt(app, messages, tools, thinking, prompt, fields):
    if prompt is not None:
        if isinstance(prompt, str):
            with app.tokenizer_lock:
                tokens = [int(t) for t in app.tokenizer.encode(prompt)]
        else:
            tokens = [int(t) for t in prompt]
        return RenderedPrompt(tokens)
    if not has_images(messages):
        tokens, history = app.render(messages, tools, thinking=thinking)
        return RenderedPrompt(tokens, history)
    messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=app.late_system,
                                                                 allow_images=True))
    effort = app.effort_for(fields.get('reasoning_effort'))

    def render(template):
        kwargs = dict(add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        if tools:
            kwargs['tools'] = tools
        if thinking and effort:
            kwargs['reasoning_effort'] = effort
        with app.tokenizer_lock:
            return app.tokenizer.apply_chat_template(template, **kwargs)

    return prepare_images(getattr(app, 'vision', None), messages, render, context_limit=app.context_window or None)
