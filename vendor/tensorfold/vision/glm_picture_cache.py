"""Bounded CPU picture preprocessing cache, inspired by Mia PR 63 (Apache-2.0).

Keep fitted canvases rather than source bytes/lazy mutable canvases. Request references
stay valid after LRU eviction. Preserve the original per-image and aggregate hashes.
No GPU allocation, URL cache, request-field override, or on-disk picture storage.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass

from .images_http import ImageInputError


@dataclass(frozen=True, slots=True, eq=False)
class FittedPicture:
    canvas: object
    shape: tuple
    width: int
    height: int
    pixels: int
    digest: bytes
    key: int
    nbytes: int


class PictureCache:
    def __init__(self, enabled=None, max_bytes=None, entries=256):
        self.enabled = (os.environ.get('TENSORFOLD_GLM_PICTURE_CACHE', '1').lower()
                        not in ('0', 'off', 'no', 'false')) if enabled is None else bool(enabled)
        if max_bytes is None:
            value = os.environ.get('TENSORFOLD_GLM_PICTURE_CACHE_MB', '384')
            if not value.isdecimal() or not 16 <= int(value) <= 8192:
                raise ValueError('TENSORFOLD_GLM_PICTURE_CACHE_MB: expected 16 to 8192')
            max_bytes = int(value) * 1024 * 1024
        self.max_bytes, self.max_entries = int(max_bytes), int(entries)
        self.held_bytes = 0
        self.pictures = OrderedDict()
        self.prefixes = OrderedDict()
        self.lock = threading.RLock()

    def fit(self, data, detail, tokens, min_tokens, limits, remaining_pixels, fitter):
        from .images import _decode

        cache_key = (hashlib.sha256(data).digest(), detail, int(tokens), int(min_tokens)) if self.enabled else None
        seen = None
        if self.enabled:
            with self.lock:
                seen = self.pictures.get(cache_key)
                if seen is not None:
                    self.pictures.move_to_end(cache_key)
        if seen is not None:
            # EXIF orientation only swaps dimensions; area and maximum dimension are unchanged.
            if max(seen.width, seen.height) > limits.max_dimension or seen.pixels > min(limits.max_pixels, remaining_pixels):
                raise ImageInputError('image dimensions exceed the decoded pixel limit')
            return seen, seen.pixels

        decoded = _decode(data, detail, limits, remaining_pixels)
        canvas = fitter(decoded, tokens)
        pixels = decoded.width * decoded.height
        if not self.enabled:
            return canvas, pixels
        raw = canvas.numpy().tobytes()
        head = b'image' + json.dumps(list(canvas.shape)).encode()
        seen = FittedPicture(canvas, tuple(canvas.shape), decoded.width, decoded.height, pixels,
                             hashlib.sha256(raw).digest(),
                             int.from_bytes(hashlib.sha256(head + b'\0' + raw).digest()[:8], 'big') >> 2,
                             canvas.numel() * canvas.element_size())
        if seen.nbytes <= self.max_bytes and self.max_entries > 0:
            with self.lock:
                previous = self.pictures.pop(cache_key, None)
                if previous is not None:
                    self.held_bytes -= previous.nbytes
                self.pictures[cache_key] = seen
                self.held_bytes += seen.nbytes
                while self.held_bytes > self.max_bytes or len(self.pictures) > self.max_entries:
                    self.held_bytes -= self.pictures.popitem(last=False)[1].nbytes
        return seen, pixels

    def identities(self, items):
        # Prefix SHA states preserve sha256(head0 + pixels0 + head1 + pixels1 ...).
        # Hashes of hashes would change the original content_hash even for identical pixels.
        digest = hashlib.sha256(b'tensorfold-glm-vision-v1\0')
        keys, prefix = [], ()
        for kind, item in items:
            head = kind.encode() + json.dumps(list(item.shape)).encode()
            if isinstance(item, FittedPicture):
                keys.append(item.key)
                prefix += ((kind, item.shape, item.digest),)
                with self.lock:
                    saved = self.prefixes.get(prefix)
                    if saved is not None:
                        self.prefixes.move_to_end(prefix)
                        digest = saved.copy()
                if saved is None:
                    digest.update(head)
                    digest.update(item.canvas.numpy().tobytes())
                    with self.lock:
                        self.prefixes[prefix] = digest.copy()
                        self.prefixes.move_to_end(prefix)
                        while len(self.prefixes) > 256:
                            self.prefixes.popitem(last=False)
            else:
                raw = item.numpy().tobytes()
                digest.update(head)
                digest.update(raw)
                keys.append(int.from_bytes(hashlib.sha256(head + b'\0' + raw).digest()[:8], 'big') >> 2)
                # Include non-cached images/videos in the prefix's identity too.
                prefix += ((kind, tuple(item.shape), hashlib.sha256(raw).digest()),)
        return digest.hexdigest(), keys
