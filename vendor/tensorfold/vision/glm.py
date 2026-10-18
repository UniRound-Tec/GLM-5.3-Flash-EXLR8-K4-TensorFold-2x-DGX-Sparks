"""GLM-5.3 Flash image and video input on CUDA: the checkpoint's processors (resize with pad, frame choice, prompt
layout) and its vision tower, run on rank 0.

The arithmetic follows transformers 5.17's ``Glm5NextImageProcessor`` / ``Glm5NextVideoProcessor`` /
``Glm5NextVisionModel``: images and frame pairs become 14x14x2 patches, 24 blocks of 2-D rotary attention over one
image (or one frame pair) at a time, a 2x2 downsample and a SwiGLU merger into the language model's 4,096 columns.
The language model takes the features in place of its ``<|image|>`` rows; its positions are the token positions
(the MLA layers carry no rotary part), so nothing else about the prompt changes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from tensorfold.server.errors import CONTEXT_LIMIT

from .images_http import ImageInputError
from .glm_picture_cache import PictureCache, FittedPicture

IMAGE, VIDEO = "<|image|>", "<|video|>"
IMAGE_SPAN = "<|begin_of_image|><|image|><|end_of_image|>"     # what the chat template writes for a picture
VIDEO_SPAN = "<|begin_of_video|><|video|><|end_of_video|>"     # and for a clip, expanded frame pair by frame pair

_MENTION_ZWSP = "\u200b"           # keeps a marker a conversation merely mentions from being read as the token
_MENTIONS = ("<|begin_of_image|>", "<|end_of_image|>", "<|begin_of_video|>", "<|end_of_video|>",
             IMAGE, VIDEO)


def _as_text(text: str) -> str:
    """A marker a conversation mentions -- a patch, a log, this server's own refusal -- must stay text.

    The encoder reads a bare ``<|image|>`` as the picture token itself, and the engine then finds a position
    with no features for it (``prepare`` counts them). A zero-width space inside the marker leaves the
    characters there for the model while the encoder tokenizes them as the text they are."""

    if "<|" not in text:
        return text
    for mark in _MENTIONS:
        if mark in text:
            text = text.replace(mark, mark[:2] + _MENTION_ZWSP + mark[2:])
    return text


MEAN = (0.48145466, 0.4578275, 0.40821073)                    # CLIP's, from processor_config.json
STD = (0.26862954, 0.26130258, 0.27577711)
PASS_PATCHES = 8192                  # patches the tower encodes at once (its workspace scales with this): a
                                     # 2,048-token picture is one pass


def _env_int(name: str, default: int, low: int, high: int) -> int:
    value = os.environ.get(name, "")
    if value == "":
        return default
    if not value.isdecimal() or not low <= int(value) <= high:
        raise ValueError(f"{name}: {low:,} to {high:,}, not {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class GlmVisionLimits:
    """Token caps: the checkpoint allows 8,000 tokens a picture and 240,000 a clip; these keep a request's
    encode within the tower's reserved workspace and the prompt within a sensible share of the window.

    A request takes up to ``max_images`` pictures and ``max_videos`` clips. Its pictures share
    ``request_image_tokens`` and its clips ``request_video_tokens`` in equal parts, each still within its own cap:
    up to 8 pictures and 2 clips (the defaults) keep their full caps, so their canvases and bits are as before."""

    image_tokens: int = 2048         # a picture's tokens at most (a 1080p frame is 2,040 at 2,048)
    video_tokens: int = 16384        # a clip's tokens at most, over all its frame pairs
    video_frames: int = 128          # frames sampled from a clip at most (2 a second, spread wider past this)
    fps: float = 2.0                 # the processor's rate
    min_tokens: int = 16
    max_images: int = 50             # pictures a request
    max_videos: int = 4              # clips a request
    request_image_tokens: int = 16384    # a request's pictures' tokens, shared
    request_video_tokens: int = 32768    # a request's clips' tokens, shared

    @classmethod
    def from_env(cls) -> "GlmVisionLimits":
        d = cls()                            # a slots class keeps its defaults on instances, not the class
        return cls(image_tokens=_env_int("TENSORFOLD_GLM_IMAGE_TOKENS", d.image_tokens, 16, 8000),
                   video_tokens=_env_int("TENSORFOLD_GLM_VIDEO_TOKENS", d.video_tokens, 64, 240000),
                   video_frames=_env_int("TENSORFOLD_GLM_VIDEO_FRAMES", d.video_frames, 2, 2048),
                   max_images=_env_int("TENSORFOLD_GLM_MAX_IMAGES", d.max_images, 1, 256),
                   max_videos=_env_int("TENSORFOLD_GLM_MAX_VIDEOS", d.max_videos, 1, 16),
                   request_image_tokens=_env_int("TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS", d.request_image_tokens,
                                                 16, 262144),
                   request_video_tokens=_env_int("TENSORFOLD_GLM_REQUEST_VIDEO_TOKENS", d.request_video_tokens,
                                                 64, 1048576))

    def picture_tokens(self, pictures: int) -> int:
        """Each picture's token cap in a request with ``pictures`` of them."""

        return max(self.min_tokens, min(self.image_tokens, self.request_image_tokens // max(1, pictures)))

    def clip_tokens(self, clips: int) -> int:
        """Each clip's token cap in a request with ``clips`` of them."""

        return max(64, min(self.video_tokens, self.request_video_tokens // max(1, clips)))


# -- the processors' geometry ---------------------------------------------------------------------------------
def smart_resize(frames: int, height: int, width: int, max_tokens: int, min_tokens: int = 16, temporal: int = 2,
                 factor: int = 28) -> tuple[int, int]:
    """The canvas (height, width), multiples of ``factor``, within ``max_tokens`` over ``frames`` frames."""

    per_token = temporal * factor ** 2
    low_pixels, high_pixels = min_tokens * per_token, max_tokens * per_token

    def align(v: float) -> int:
        return math.ceil(v / factor) * factor

    aligned = max(temporal, round(frames / temporal) * temporal)
    h, w = align(height), align(width)
    if aligned * h * w < low_pixels:
        scale = math.sqrt(low_pixels / (frames * height * width))
        h, w = align(max(1, math.ceil(height * scale))), align(max(1, math.ceil(width * scale)))
    if aligned * h * w > high_pixels:
        if high_pixels < aligned * factor ** 2:
            raise ImageInputError("the token cap is too small for one patch per frame pair")
        lo, hi, h, w = 1, height, factor, factor
        while lo <= hi:                      # the largest content height whose aligned canvas fits
            mid = (lo + hi) // 2
            ch, cw = align(mid), align(max(1, math.floor(width * mid / height)))
            if aligned * ch * cw <= high_pixels:
                h, w, lo = ch, cw, mid + 1
            else:
                hi = mid - 1
    return h, w


def content_size(frames: int, height: int, width: int, canvas: tuple[int, int], min_tokens: int = 16,
                 temporal: int = 2, factor: int = 28) -> tuple[int, int]:
    """The resized picture inside the canvas (the rest is zero padding, right and bottom)."""

    th, tw = canvas
    scale = min(th / height, tw / width)
    if frames * height * width >= temporal * factor ** 2 * min_tokens:
        scale = min(1.0, scale)
    return max(1, min(th, math.floor(height * scale))), max(1, min(tw, math.floor(width * scale)))


def fit(frame, canvas: tuple[int, int], content: tuple[int, int]):
    """One uint8 [3, H, W] frame resized (bicubic, antialiased, as torchvision does for the processor) and padded."""

    import torch
    from torchvision.transforms.v2 import functional as tvF

    if tuple(frame.shape[-2:]) != content:
        frame = tvF.resize(frame, list(content), interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    out = torch.zeros((3, *canvas), dtype=torch.uint8)
    out[:, :content[0], :content[1]] = frame
    return out


def sample_frames(total: int, fps: float, duration: float, limits: GlmVisionLimits) -> list[int]:
    """``Glm5NextVideoProcessor.sample_frames``: one frame each 1/fps seconds from the start, at most
    ``video_frames`` (then spread evenly), duplicates dropped, an even count (the last frame repeated)."""

    duration = duration or round((total - 1) / fps) + 1
    max_seconds = int(duration)
    want = min(int(duration * limits.fps), limits.video_frames)
    if total < want:
        picks = np.linspace(0, total - 1, want, dtype=int).tolist()
    else:
        picks, second, step = [], 0.0, 1 / limits.fps
        for i in range(total):
            if i / fps >= second:
                second += step
                picks.append(i)
                if second >= max_seconds:
                    break
    if len(picks) < want:
        start, end = (picks[0], picks[-1]) if picks else (0, max(total - 1, 0))
        picks = np.linspace(start, end, want, dtype=int).tolist()
    elif len(picks) > want:
        picks = np.linspace(0, total - 1, want, dtype=int).tolist()
    seen: set[int] = set()
    unique = [i for i in picks if not (i in seen or seen.add(i))]
    if not unique:
        unique = [0]
    if len(unique) % 2:
        unique.append(unique[-1])
    return unique


# -- inputs -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GlmClip:
    """A clip's sampled frames at the tower's canvas: uint8 [frames, 3, H, W], and each frame pair's time."""

    frames: Any
    seconds: tuple[float, ...]           # one per frame pair: its first frame's time
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha256(b"tensorfold-glm-clip-v1\0" + json.dumps(list(self.frames.shape)).encode())
        digest.update(self.frames.numpy().tobytes())
        object.__setattr__(self, "content_hash", digest.hexdigest())


def decode_clip(data: bytes, limits: GlmVisionLimits, deadline: float, *, max_dimension: int = 8192,
                max_seconds: float = 3600.0) -> GlmClip:
    """Decode one clip's sampled frames, each resized onto the clip's canvas as it is decoded."""

    import torch

    from .videos import _av

    av = _av()
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.video:
                raise ImageInputError("the video has no video stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            width, height = int(stream.codec_context.width or 0), int(stream.codec_context.height or 0)
            if min(width, height) <= 0 or max(width, height) > max_dimension:
                raise ImageInputError("video dimensions are missing or exceed the pixel limit")
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            total = int(stream.frames or 0)
            if not total and fps > 0:        # WebM and some MKV files carry no frame count
                seconds = (float(stream.duration * stream.time_base) if stream.duration else
                           container.duration / 1e6 if container.duration else 0.0)
                total = int(round(seconds * fps))
            if not 0 < fps <= 1000 or total <= 0:
                raise ImageInputError("the video's frame count or rate is missing")
            if total / fps > max_seconds:
                raise ImageInputError(f"videos are limited to {max_seconds / 60:.0f} minutes")
            picks = sample_frames(total, fps, total / fps, limits)
            canvas = smart_resize(len(picks), height, width, limits.video_tokens, limits.min_tokens)
            content = content_size(len(picks), height, width, canvas, limits.min_tokens)
            wanted = set(picks)
            got: dict[int, Any] = {}
            for n, frame in enumerate(container.decode(stream)):
                if n > picks[-1] or time.monotonic() >= deadline:
                    break
                if n in wanted:
                    rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                    got[n] = fit(rgb, canvas, content)
            if time.monotonic() >= deadline:
                raise ImageInputError("video decoding exceeds the time limit")
    except ImageInputError:
        raise
    except (OSError, ValueError, MemoryError) as exc:
        raise ImageInputError(f"video bytes are invalid or unsupported ({type(exc).__name__}); use MP4 or "
                              "WebM") from None
    if not got:
        raise ImageInputError("the video has no decodable frames")
    last = max(got)                           # a stream shorter than its header: repeat the last frame decoded
    frames = torch.stack([got.get(i, got[last]) if i <= last else got[last] for i in picks])
    return GlmClip(frames, tuple(i / fps for i in picks[::2]))


@dataclass
class GlmPrepared:
    """A prompt with pictures or clips: its tokens, where the tower's rows go, and what to encode.

    ``items``: in prompt order, ("image", uint8 [3, H, W]) or ("video", uint8 [frames, 3, H, W]); ``positions``:
    the prompt positions of their feature rows, in order (a picture's rows, then the next item's)."""

    token_ids: list[int]
    items: list[tuple[str, Any]]
    positions: list[int]
    content_hash: str = ""
    # per item, in order: its feature rows, and a 62-bit key of its kind, shape and pixels (prompt reuse tells
    # pictures apart by these: their placeholder tokens are all the same)
    item_rows: list[int] = field(default_factory=list)
    item_keys: list[int] = field(default_factory=list)

    def continued(self, ids: list[int]) -> "GlmPrepared":
        """The same pictures under a longer prompt (a tool gate's continuation): positions are token positions."""

        if list(ids[:len(self.token_ids)]) != list(self.token_ids):
            raise ValueError("a continued image prompt must extend the original prompt")
        return GlmPrepared(list(ids), self.items, self.positions, self.content_hash, self.item_rows, self.item_keys)

    def from_item(self, k: int) -> "GlmPrepared":
        """Only items k.. and their feature rows' positions (a prompt resumed past the earlier items' rows)."""

        skip = sum(self.item_rows[:k])
        return GlmPrepared(self.token_ids, self.items[k:], self.positions[skip:], self.content_hash,
                           self.item_rows[k:], self.item_keys[k:])

    def keyed_ids(self) -> list[int]:
        """The prompt's ids with each item's feature rows replaced by -1 - its key: equal prefixes then mean the same
        pictures in the same places, not only the same placeholder tokens."""

        ids = list(self.token_ids)
        at = 0
        for rows, key in zip(self.item_rows, self.item_keys):
            for p in self.positions[at:at + rows]:
                ids[p] = -1 - key
            at += rows
        return ids


# -- the tower --------------------------------------------------------------------------------------------------
class GlmVisionTower:
    """``Glm5NextVisionModel`` in plain torch on bf16 weights; one image or frame pair attends within itself."""

    def __init__(self, model_dir: Path, device) -> None:
        import torch
        from safetensors import safe_open

        cfg = json.loads((model_dir / "config.json").read_text())
        v = cfg["vision_config"]
        self.hidden, self.heads = int(v["hidden_size"]), int(v["num_heads"])
        self.head_dim = self.hidden // self.heads
        self.depth, self.eps = int(v["depth"]), float(v["rms_norm_eps"])
        self.patch, self.temporal = int(v["patch_size"]), int(v["temporal_patch_size"])
        self.merge, self.out = int(v["spatial_merge_size"]), int(v["out_hidden_size"])
        self.limit = float(v.get("swiglu_limit", 10.0))
        theta = float((v.get("rope_parameters") or {}).get("rope_theta", 10000.0))
        if self.merge != 2 or self.patch != 14 or self.temporal != 2 or self.head_dim % 4:
            raise ValueError("this GLM vision tower's geometry is not the one TensorFold implements")
        spatial = self.head_dim // 2
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.float32, device=device) / spatial))
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        names = sorted(n for n in index if n.startswith("model.visual."))
        if not names:
            raise ValueError("this checkpoint has no vision tower (model.visual.*)")
        t: dict[str, Any] = {}
        for file in sorted({index[n] for n in names}):
            with safe_open(str(model_dir / file), framework="pt", device="cpu") as f:
                for n in names:
                    if index[n] == file:
                        t[n[len("model.visual."):]] = f.get_tensor(n).to(device=device, dtype=torch.bfloat16)
        self.weight_bytes = sum(x.numel() * x.element_size() for x in t.values())
        self.t = t
        self.device = device
        self.mean = torch.tensor(MEAN, device=device).view(3, 1, 1) * 255
        self.std = torch.tensor(STD, device=device).view(3, 1, 1) * 255

    # the processors' patch layout: rows are merge blocks (2x2 patches), a patch's values (channel, time, y, x)
    def patches(self, frames):
        """uint8 [T, 3, H, W] (T even) on the device -> normalized [T/2 * H/14 * W/14, 3 * 2 * 14 * 14] float32."""

        T, C, H, W = frames.shape
        p, m = self.patch, self.merge
        x = (frames.float() - self.mean) / self.std
        x = x.view(T // 2, 2, C, H // p // m, m, p, W // p // m, m, p)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(T // 2 * (H // p) * (W // p), C * 2 * p * p)

    def _rms(self, x, w):
        import torch

        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)

    def _rope(self, gh: int, gw: int):
        import torch

        m = self.merge
        hpos, wpos = torch.meshgrid(torch.arange(gh, device=self.device), torch.arange(gw, device=self.device),
                                    indexing="ij")
        shape = (gh // m, m, gw // m, m)
        pos = torch.stack([hpos.reshape(shape).transpose(1, 2).flatten(),
                           wpos.reshape(shape).transpose(1, 2).flatten()], dim=-1).float()
        freqs = (pos[..., None] * self.inv_freq).flatten(1)        # [S, h freqs then w freqs]
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

    @staticmethod
    def _rotate(x, cos, sin):
        import torch

        xf = x.float()
        half = xf.shape[-1] // 2
        return (xf * cos + torch.cat((-xf[..., half:], xf[..., :half]), dim=-1) * sin).to(x.dtype)

    def _block(self, i: int, x, cos, sin):
        import torch
        import torch.nn.functional as F

        t, L = self.t, self.limit
        g = f"blocks.{i}."
        B, S, D = x.shape
        h = self._rms(x, t[g + "norm1.weight"])
        q, k, v = F.linear(h, t[g + "attn.qkv.weight"], t[g + "attn.qkv.bias"]).view(B, S, 3, self.heads,
                                                                                     self.head_dim).unbind(2)
        q = self._rotate(self._rms(q, t[g + "attn.q_norm.weight"]), cos, sin)
        k = self._rotate(self._rms(k, t[g + "attn.k_norm.weight"]), cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           scale=self.head_dim ** -0.5)
        x = x + F.linear(o.transpose(1, 2).reshape(B, S, D), t[g + "attn.proj.weight"], t[g + "attn.proj.bias"])
        h = self._rms(x, t[g + "norm2.weight"])
        gate = F.linear(h, t[g + "mlp.gate_proj.weight"], t[g + "mlp.gate_proj.bias"]).clamp(max=L)
        up = F.linear(h, t[g + "mlp.up_proj.weight"], t[g + "mlp.up_proj.bias"]).clamp(-L, L)
        del h
        return x + F.linear(F.silu(gate) * up, t[g + "mlp.down_proj.weight"], t[g + "mlp.down_proj.bias"])

    def encode(self, patches, groups: int, gh: int, gw: int):
        """``groups`` images (or frame pairs) of gh x gw patches each, [groups * gh * gw, 1176] -> their
        [groups * gh * gw / 4, out] features in the language model's dtype."""

        import torch
        import torch.nn.functional as F

        t = self.t
        S = gh * gw
        # the reference's convolutions, not the equal matmuls: 24 blocks turn their last-bit differences into ~3%
        x = F.conv3d(patches.to(torch.bfloat16).view(-1, 3, self.temporal, self.patch, self.patch),
                     t["patch_embed.proj.weight"], t["patch_embed.proj.bias"], stride=(self.temporal, self.patch,
                                                                                      self.patch))
        x = x.view(groups, S, self.hidden)
        cos, sin = self._rope(gh, gw)
        for i in range(self.depth):
            x = self._block(i, x, cos, sin)
        x = self._rms(x, t["post_layernorm.weight"]).reshape(-1, self.merge, self.merge, self.hidden)
        x = F.conv2d(x.permute(0, 3, 1, 2), t["downsample.weight"], t["downsample.bias"],
                     stride=self.merge).view(-1, self.out)
        x = F.linear(x, t["merger.proj.weight"])
        x = F.gelu(F.layer_norm(x, (self.out,), t["merger.post_projection_norm.weight"],
                                t["merger.post_projection_norm.bias"]))
        gate = F.linear(x, t["merger.gate_proj.weight"]).clamp(max=self.limit)
        up = F.linear(x, t["merger.up_proj.weight"]).clamp(-self.limit, self.limit)
        return F.linear(F.silu(gate) * up, t["merger.down_proj.weight"])

    def features(self, items: list[tuple[str, Any]]):
        """Every item's feature rows, in order: [rows, out] bf16 on the device."""

        import torch

        # each pass writes its rows into the request's output at once: the rows are held once, not twice over a cat
        rows = sum(GlmVision._rows(kind, frames) for kind, frames in items)
        out = torch.empty((rows, self.out), dtype=torch.bfloat16, device=self.device)
        at = 0
        for _, frames in items:
            frames = frames if frames.dim() == 4 else frames.unsqueeze(0).expand(2, -1, -1, -1)
            T, _, H, W = frames.shape
            gh, gw = H // self.patch, W // self.patch
            per = max(1, PASS_PATCHES // (gh * gw))          # frame pairs a pass
            for start in range(0, T // 2, per):
                pairs = frames[2 * start:2 * min(T // 2, start + per)].to(self.device, non_blocking=True)
                x = self.encode(self.patches(pairs), pairs.shape[0] // 2, gh, gw)
                out[at:at + x.shape[0]].copy_(x)
                at += x.shape[0]
                del x
        return out


# -- the frontend ------------------------------------------------------------------------------------------------
class GlmVision:
    """What the server needs for GLM pictures and clips: ``prepare`` (rendered text -> tokens and items),
    ``load_videos``, and the tower's ``features`` (rank 0)."""

    videos = True
    tool_media = True       # tool results' pictures and clips, rendered inside their <tool_response> (issue #28)

    def __init__(self, model_dir: Path, device, *, allow_urls: bool = False,
                 limits: GlmVisionLimits | None = None) -> None:
        from tokenizers import Tokenizer

        cfg = json.loads((model_dir / "config.json").read_text())
        self.image_token = int(cfg.get("image_token_id", 154854))
        self.allow_urls = allow_urls
        self.limits = limits or GlmVisionLimits.from_env()
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        if self.tok.token_to_id(IMAGE) != self.image_token:
            raise ValueError("the tokenizer's <|image|> id does not match config.json's image_token_id")
        self.tower = GlmVisionTower(model_dir, device)
        self.weight_bytes = self.tower.weight_bytes
        self.picture_cache = PictureCache()

    @property
    def image_limits(self):
        """The shared image limits with this frontend's picture count; bytes and pixels scale with it (each picture
        is fitted to its canvas as it is decoded, so one full-size picture is in host memory at a time)."""

        from .images import DEFAULT_LIMITS as D

        n = self.limits.max_images
        if n <= D.max_images:
            return replace(D, max_images=n)
        return replace(D, max_images=n, max_total_encoded_bytes=64 * 1024 * 1024,
                       max_total_pixels=max(D.max_total_pixels, n * 8 * 1024 * 1024),
                       total_timeout_seconds=max(D.total_timeout_seconds, 2.0 * n))

    @property
    def max_videos(self) -> int:
        return self.limits.max_videos

    def load_images(self, sources) -> list:
        """The request's pictures, each decoded and fitted to its canvas (uint8 [3, H, W]) in turn."""

        from .images import _check_source, _data_bytes, _decode
        from .images_http import fetch_image

        L = self.image_limits
        if len(sources) > L.max_images:
            raise ImageInputError(f"a request supports at most {L.max_images} images")
        tokens = self.limits.picture_tokens(len(sources))
        total_bytes, total_pixels, out = 0, 0, []
        deadline = time.monotonic() + L.total_timeout_seconds
        for source in sources:
            _check_source(source, L, self.allow_urls)
            remaining = min(L.max_encoded_bytes, L.max_total_encoded_bytes - total_bytes)
            if remaining <= 0 or time.monotonic() >= deadline:
                raise ImageInputError("image request exceeds the total byte or time limit")
            if source.url.startswith("data:"):
                data = _data_bytes(source.url, remaining)
            else:
                data, _ = fetch_image(source.url, max_bytes=remaining,
                                      deadline=min(deadline, time.monotonic() + L.timeout_seconds),
                                      max_redirects=L.max_redirects, max_url_chars=L.max_url_chars)
            total_bytes += len(data)
            picture, pixels = self.picture_cache.fit(
                data, source.detail, tokens, self.limits.min_tokens, L,
                L.max_total_pixels - total_pixels, self._picture)
            total_pixels += pixels
            out.append(picture)
            if time.monotonic() >= deadline:
                raise ImageInputError("image request exceeds the total time limit")
        return out

    def load_videos(self, sources) -> list[GlmClip]:
        from .videos import DEFAULT_VIDEO_LIMITS as L, VIDEO_MEDIA_TYPES
        from .images import _data_bytes
        from .images_http import fetch_image

        if len(sources) > self.limits.max_videos:
            raise ImageInputError(f"a request supports at most {self.limits.max_videos} videos")
        limits = replace(self.limits, video_tokens=self.limits.clip_tokens(len(sources)))
        total, out = 0, []
        deadline = time.monotonic() + L.total_timeout_seconds
        for source in sources:
            remaining = min(L.max_encoded_bytes, L.max_total_encoded_bytes - total)
            if remaining <= 0 or time.monotonic() >= deadline:
                raise ImageInputError("video request exceeds the total byte or time limit")
            if source.url.startswith("data:"):
                data = _data_bytes(source.url, remaining)
            elif self.allow_urls:
                data, _ = fetch_image(source.url, max_bytes=remaining,
                                      deadline=min(deadline, time.monotonic() + L.timeout_seconds),
                                      max_redirects=L.max_redirects, max_url_chars=L.max_url_chars,
                                      media_types=VIDEO_MEDIA_TYPES)
            else:
                raise ImageInputError("video URLs are off on this server")
            total += len(data)
            out.append(decode_clip(data, limits, deadline, max_dimension=L.max_dimension,
                                   max_seconds=L.max_seconds))
        return out

    def _picture(self, image, tokens: int):
        import torch

        rgb = torch.frombuffer(bytearray(image.pixels), dtype=torch.uint8).view(image.height, image.width, 3)
        rgb = rgb.permute(2, 0, 1)
        canvas = smart_resize(2, image.height, image.width, tokens, self.limits.min_tokens)
        return fit(rgb, canvas, content_size(2, image.height, image.width, canvas, self.limits.min_tokens))

    @staticmethod
    def _marker_spans(text: str):
        """(start, end) of every picture and clip marker in ``text``, left to right."""

        out, pos = [], 0
        while True:
            i, v = text.find(IMAGE_SPAN, pos), text.find(VIDEO_SPAN, pos)
            if i < 0 and v < 0:
                return out
            if v < 0 or 0 <= i < v:
                out.append((i, i + len(IMAGE_SPAN)))
                pos = i + len(IMAGE_SPAN)
            else:
                out.append((v, v + len(VIDEO_SPAN)))
                pos = v + len(VIDEO_SPAN)

    def escape_quoted(self, text: str, mark_ends) -> str:
        """``text`` with the markers a conversation only quotes turned into text, for ``server.prompts``.

        A marker the template wrote for this request's own picture or clip stands right behind the mark rendered for
        it (``MediaMarks``); every other one -- a pasted template, a log, this server's own refusal -- is a
        quotation. The encoder would read it as the picture token itself and the engine would then find a position
        with no features for it, so a quotation is escaped (a zero-width space after ``<|``) and ``prepare`` leaves
        it as the text it is. ``mark_ends`` are where this request's marks end."""

        ends = set(mark_ends)
        out, pos = [], 0
        for start, stop in self._marker_spans(text):
            out.append(_as_text(text[pos:start]))          # the text before it (this request's mark included:
            own = start in ends                            # the server takes the marks out afterwards)
            out.append(text[start:stop] if own else _as_text(text[start:stop]))
            pos = stop
        out.append(_as_text(text[pos:]))
        return "".join(out)

    def prepare(self, text: str, images, *, videos=(), max_prompt_tokens: int | None = None) -> GlmPrepared:
        """The rendered prompt with each picture's and clip's placeholder expanded as Glm5NextProcessor does."""

        videos = list(videos or ())
        # A conversation may mention a marker as text -- a patch, a log, this server's own refusal. What the
        # template wrote for the request's own pictures and clips is the spans the walk below expands; a mention
        # is only text, and refusing the prompt over it ended a session that kept such text beside a screenshot
        # (a text-only request, which never reaches here, sends the mention on as it is). A mention of a whole
        # span is still refused: it cannot be told from a marker, and the walk would spend a picture on it.
        if text.count(IMAGE_SPAN) != len(images) or text.count(VIDEO_SPAN) != len(videos):
            raise ImageInputError("the prompt's image and video markers do not match its images and videos "
                                  "(is <|image|> or <|video|> in the text?)")
        tokens = self.limits.picture_tokens(len(images))
        items: list[tuple[str, Any]] = []
        pieces: list[str] = []
        image_iter, video_iter = iter(images), iter(videos)
        rest = text
        while True:
            i, v = rest.find(IMAGE_SPAN), rest.find(VIDEO_SPAN)
            if i < 0 and v < 0:
                pieces.append(_as_text(rest))
                break
            if v < 0 or 0 <= i < v:
                picture = next(image_iter)          # load_images fits pictures as it decodes them
                picture = picture if isinstance(picture, FittedPicture) or hasattr(picture, "dim") else self._picture(picture, tokens)
                n = picture.shape[1] // 28 * (picture.shape[2] // 28)
                pieces.append(_as_text(rest[:i]) + "<|begin_of_image|>" + IMAGE * n + "<|end_of_image|>")
                items.append(("image", picture))
                rest = rest[i + len(IMAGE_SPAN):]
            else:
                clip = next(video_iter)
                n = clip.frames.shape[2] // 28 * (clip.frames.shape[3] // 28)
                frames = "".join(f"<|begin_of_image|>{IMAGE * n}<|end_of_image|>{s:.1f} seconds"
                                 for s in clip.seconds)
                pieces.append(_as_text(rest[:v]) + "<|begin_of_video|>" + frames + "<|end_of_video|>")
                items.append(("video", clip.frames))
                rest = rest[v + len(VIDEO_SPAN):]
        ids = self.tok.encode("".join(pieces), add_special_tokens=False).ids
        if max_prompt_tokens and len(ids) >= max_prompt_tokens:
            # v0.6.0's context refusal wording (68c6e35, Apache-2.0): served as context_length_exceeded
            raise ValueError(f"{CONTEXT_LIMIT} {max_prompt_tokens} tokens: the prompt with its images is {len(ids):,} "
                             f"tokens, which exceeds the context window; send fewer or smaller images, or a shorter "
                             f"video")
        positions = [p for p, t in enumerate(ids) if t == self.image_token]
        rows = sum(self._rows(kind, x) for kind, x in items)
        if len(positions) != rows:
            raise ImageInputError(f"the prompt has {len(positions)} image positions for {rows} feature rows")
        content_hash, keys = self.picture_cache.identities(items)
        return GlmPrepared(ids, items, positions, content_hash, [self._rows(kind, x) for kind, x in items], keys)

    @staticmethod
    def _rows(kind: str, x) -> int:
        pairs = x.shape[0] // 2 if kind == "video" else 1
        return pairs * (x.shape[-2] // 28) * (x.shape[-1] // 28)

    def features(self, prepared: GlmPrepared):
        return self.tower.features([(kind, x.canvas if isinstance(x, FittedPicture) else x)
                                    for kind, x in prepared.items])

    def warm(self) -> None:
        """Run a small picture through the tower once, so its kernels are loaded before the first request."""

        import torch

        self.tower.features([("image", torch.zeros((3, 56, 56), dtype=torch.uint8))])
        torch.cuda.synchronize()
