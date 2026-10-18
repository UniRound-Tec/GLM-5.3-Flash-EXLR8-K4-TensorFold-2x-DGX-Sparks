"""The cache pool's first latent planes in the GB10's UEFI display reservation (TF_GLM_DISPLAY_KV_MIB).

The firmware sets aside a framebuffer region that nothing claims on a headless Spark and that MemAvailable never
counts. A DRM dumb buffer on /dev/dri/card0, mmap'd and registered with cuMemHostRegister(DEVICEMAP|IOMEMORY), is
memory the GPU reads like any other (on GB10 the device pointer is the host address). ``--parallel`` sizes its pool
from free memory; with this setting it is credited the reservation's bytes too, and the first latent planes (one per
DSA layer, every token's row) are carved from one span: an ordinary host mapping, then the display buffer right above
it, so the planes take exactly the credited bytes of the reservation and nothing more of the host's memory than
without it. Placement only: the planes hold the same bytes.

``TF_GLM_DISPLAY_KV_MIB``: unset or 0 off; a multiple of 16 up to 2032 (2048 fails ENOMEM on GB10).
``TF_GLM_DISPLAY_KV_CARD``: the DRM card (default /dev/dri/card0; the container needs it as a device).
For headless Sparks only: a card with a connected output (a monitor) keeps its reservation, and the start stops.
Fails closed: the credited bytes were never budgeted against host memory, so a span that cannot be mapped stops the
start instead of falling back to ordinary memory.
"""
from __future__ import annotations

import os

import torch

ENV = "TF_GLM_DISPLAY_KV_MIB"
CARD_ENV = "TF_GLM_DISPLAY_KV_CARD"
LARGEST_MIB = 2032
SYSFS = "/sys/class/drm"


def mib() -> int:
    """The display reservation credited to the pool, in MiB; 0 when off."""
    raw = (os.environ.get(ENV) or "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{ENV}={raw!r}: a number of MiB") from None
    if value == 0:
        return 0
    if value < 0 or value % 16 or value > LARGEST_MIB:
        raise ValueError(f"{ENV}={value}: a multiple of 16 from 16 to {LARGEST_MIB} (1792 measured on GB10; 2048 "
                         "fails ENOMEM)")
    return value


def credit() -> int:
    """The bytes the pool is credited (0 when off)."""
    return mib() << 20


def span_planes(count: int, plane_bytes: int, display: int) -> int:
    """How many of ``count`` latent planes of ``plane_bytes`` the span takes: the fewest that hold ``display``."""
    k = -(-int(display) // int(plane_bytes))
    if k > count:
        raise ValueError(f"{ENV}: {display >> 20} MiB is more than the {count} latent planes hold "
                         f"({count * plane_bytes >> 20} MiB); lower it")
    return max(1, k)


def connected_outputs(sysfs: str, card: str) -> list[str]:
    """The outputs of ``card`` (e.g. card0) with a display connected, by their sysfs status."""
    import glob

    found = []
    for path in sorted(glob.glob(os.path.join(sysfs, f"{card}-*", "status"))):
        with open(path) as f:
            if f.read().strip() == "connected":
                found.append(os.path.basename(os.path.dirname(path))[len(card) + 1:])
    return found


class System:
    """Every DRM and CUDA driver call the span makes, one method a call (the seam the checks replace)."""

    CREATE_DUMB = 0xC02064B2       # DRM_IOCTL_MODE_CREATE_DUMB
    MAP_DUMB = 0xC01064B3          # DRM_IOCTL_MODE_MAP_DUMB
    GEM_CLOSE = 0x40086409         # DRM_IOCTL_GEM_CLOSE
    DEVICEMAP, IOMEMORY = 0x02, 0x04
    PITCH = 16384                  # 4096 pixels at 32 bits
    MAP_FAILED = (1 << 64) - 1

    def __init__(self) -> None:
        import ctypes

        self.card = os.environ.get(CARD_ENV, "/dev/dri/card0")
        self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_long]
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        # the _v2 entry points: the unversioned ones predate I/O-memory registration and fail on the display mapping
        self.cuda = ctypes.CDLL("libcuda.so.1")
        torch.zeros(1, device="cuda")      # registration needs a context current on this thread

    def open_card(self) -> int:
        return os.open(self.card, os.O_RDWR | os.O_CLOEXEC)

    def create_dumb(self, fd: int, size: int) -> tuple[int, int]:
        """(GEM handle, the bytes the kernel gave)."""
        import fcntl
        import struct

        layout = "=6IQ"            # drm_mode_create_dumb: height width bpp flags handle pitch size
        out = struct.unpack(layout, fcntl.ioctl(fd, self.CREATE_DUMB,
                                                struct.pack(layout, -(-size // self.PITCH), 4096, 32, 0, 0, 0, 0)))
        return out[4], out[6]

    def map_offset(self, fd: int, handle: int) -> int:
        import fcntl
        import struct

        layout = "=2IQ"            # drm_mode_map_dumb: handle pad offset
        return struct.unpack(layout, fcntl.ioctl(fd, self.MAP_DUMB, struct.pack(layout, handle, 0, 0)))[2]

    def reserve(self, size: int) -> int:
        """A PROT_NONE placeholder for the whole span, so the halves land next to each other."""
        return self._mmap(None, size, 0, 0x02 | 0x20 | 0x4000, -1, 0)

    def map_ordinary(self, address: int, size: int) -> None:
        self._mmap(address, size, 0x03, 0x02 | 0x20 | 0x10, -1, 0)

    def map_display(self, address: int, size: int, fd: int, offset: int) -> None:
        self._mmap(address, size, 0x03, 0x01 | 0x10, fd, offset)

    def register(self, address: int, size: int, *, io_memory: bool) -> None:
        import ctypes

        flags = self.DEVICEMAP | (self.IOMEMORY if io_memory else 0)
        self._check(self.cuda.cuMemHostRegister_v2(ctypes.c_void_p(address), ctypes.c_size_t(size),
                                                   ctypes.c_uint(flags)))

    def device_pointer(self, address: int) -> int:
        import ctypes

        pointer = ctypes.c_void_p()
        self._check(self.cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(pointer), ctypes.c_void_p(address),
                                                           ctypes.c_uint(0)))
        return int(pointer.value or 0)

    def unregister(self, address: int) -> None:
        import ctypes

        self._check(self.cuda.cuMemHostUnregister(ctypes.c_void_p(address)))

    def unmap(self, address: int, size: int) -> None:
        import ctypes

        if self.libc.munmap(ctypes.c_void_p(address), ctypes.c_size_t(size)) != 0:
            raise OSError(ctypes.get_errno(), "munmap")

    def close_gem(self, fd: int, handle: int) -> None:
        import fcntl
        import struct

        fcntl.ioctl(fd, self.GEM_CLOSE, struct.pack("=2I", handle, 0))

    def close_fd(self, fd: int) -> None:
        os.close(fd)

    def _mmap(self, address, size: int, prot: int, flags: int, fd: int, offset: int) -> int:
        import ctypes

        ctypes.set_errno(0)
        result = int(self.libc.mmap(address, size, prot, flags, fd, offset) or 0)
        if result in (0, self.MAP_FAILED):
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        return result

    def _check(self, result: int) -> None:
        import ctypes

        if result:
            name = ctypes.c_char_p()
            self.cuda.cuGetErrorName(ctypes.c_int(result), ctypes.byref(name))
            raise OSError(f"CUDA driver error {result} ({(name.value or b'?').decode()})")


SYSTEM = System


class Span:
    """The mapped span as device memory (``__cuda_array_interface__``)."""

    def __init__(self, pointer: int, size: int) -> None:
        self.pointer, self.size = int(pointer), int(size)

    @property
    def __cuda_array_interface__(self) -> dict:
        return {"shape": (self.size,), "typestr": "|u1", "data": (self.pointer, False), "version": 3}


def _step(name: str, call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"{ENV}: {name} failed: {exc}") from exc


def _quiet(call, *args) -> None:
    try:
        call(*args)
    except Exception:
        pass                       # unwinding: the first failure is the one to report


def map_span(ordinary: int, display: int) -> Span:
    """``ordinary`` bytes of host memory and the ``display`` bytes of the reservation right above them, both
    registered with CUDA, as one span. Unwound completely on any failure."""
    card = os.path.basename(os.environ.get(CARD_ENV, "/dev/dri/card0"))
    if busy := connected_outputs(SYSFS, card):
        raise RuntimeError(f"{ENV}: {', '.join(f'{card}-{o}' for o in busy)} has a display connected, which uses the "
                           f"reservation; unset {ENV} on a Spark with a monitor")
    total = ordinary + display
    system = SYSTEM()
    fd = handle = base = None
    registered: list[int] = []
    try:
        fd = _step("open(display card)", system.open_card)
        handle, got = _step("DRM_IOCTL_MODE_CREATE_DUMB", system.create_dumb, fd, display)
        if got < display:
            raise RuntimeError(f"{ENV}: the display reserve gave {got} B, short of {display} B")
        offset = _step("DRM_IOCTL_MODE_MAP_DUMB", system.map_offset, fd, handle)
        base = _step("mmap(span)", system.reserve, total)
        if ordinary:
            _step("mmap(ordinary half)", system.map_ordinary, base, ordinary)
        _step("mmap(display half)", system.map_display, base + ordinary, display, fd, offset)
        if ordinary:
            _step("cuMemHostRegister(ordinary half)", system.register, base, ordinary, io_memory=False)
            registered.append(base)
        _step("cuMemHostRegister(display half)", system.register, base + ordinary, display, io_memory=True)
        registered.append(base + ordinary)
        pointer = _step("cuMemHostGetDevicePointer", system.device_pointer, base)
        top = _step("cuMemHostGetDevicePointer", system.device_pointer, base + ordinary)
        if top - pointer != ordinary:
            raise RuntimeError(f"{ENV}: the halves are {top - pointer} B apart on the device, not {ordinary}")
        return Span(pointer, total)
    except BaseException:
        for address in reversed(registered):
            _quiet(system.unregister, address)
        if base is not None:
            _quiet(system.unmap, base, total)
        if handle is not None:
            _quiet(system.close_gem, fd, handle)
        if fd is not None:
            _quiet(system.close_fd, fd)
        raise


def as_tensor(span: Span, device) -> torch.Tensor:
    return torch.as_tensor(span, device=device)


_OWNERS: list[Span] = []           # the span lives as long as the process: the pool never gives it back


def latent_planes(count: int, rows: int, width: int, kind: str, device, display: int) -> list[torch.Tensor]:
    """The first of ``count`` latent planes (``kv8.zeros(rows, width, kind)`` each) as views of one span whose top
    ``display`` bytes are the display reservation: as many planes as hold it. Once a process."""
    from . import kv8

    if _OWNERS:
        raise RuntimeError(f"{ENV}: the display reservation already holds a cache pool in this process")
    like = kv8.zeros(rows, width, kind, "meta")
    plane_bytes = like.numel() * like.element_size()
    k = span_planes(count, plane_bytes, display)
    span = map_span(k * plane_bytes - display, display)
    _OWNERS.append(span)
    flat = as_tensor(span, device)
    return [flat[i * plane_bytes:(i + 1) * plane_bytes].view(like.dtype).view(like.shape).zero_() for i in range(k)]
