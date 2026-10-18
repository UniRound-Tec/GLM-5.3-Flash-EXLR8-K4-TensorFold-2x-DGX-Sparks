"""One-shot RoCE all-gather for two DGX Sparks: b12x's "RoCEnante" transport (https://github.com/local-inference-lab/b12x,
Apache License 2.0; the C proxy is theirs, roce_proxy.c; the GPU side is roce.cu, rewritten from their CuTe kernel).

A decode round's all-gathers are small (a verify window's partials: rows x 4,096 fp32) and NCCL spends most of each
in fixed latency (~30-50 us). Here the GB10 GPU writes its shard into pinned host memory the ConnectX-7 RDMA-writes
to the peer, and reads the peer's shard from its own pinned memory in place: one kernel, no NCCL proxy handoff.
``RoceComm`` sends gathers of up to ``max_bytes`` a rank this way and the rest through NCCL, with the same output
layout (rank order), the same bits, and a device-side epoch so CUDA graphs replay it.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("roce_proxy.c")
_LOCK = threading.Lock()
SLOT_ALIGN = 4096
PACK = 16
THREADS = 512
BLOCKS = 8                               # the largest grid; smaller shards launch fewer (power-of-two) blocks
SPIN_LIMIT = 20_000_000                  # flag polls (~1 us each) before a wait poisons the runtime: ~20 s


def spin_limit() -> int:
    """TF_ROCE_WAIT_S: how long a GPU wait for the peer's shard may take before it poisons the runtime (default 20 s;
    graphs keep the value they were captured with)."""

    value = os.environ.get("TF_ROCE_WAIT_S", "").strip()
    if not value:
        return SPIN_LIMIT
    try:
        seconds = float(value)
    except ValueError:
        seconds = -1.0
    if not 1 <= seconds <= 3600:
        raise ValueError(f"TF_ROCE_WAIT_S: 1 to 3600 seconds, not {value!r}")
    return int(seconds * 1_000_000)


def _cache_dir() -> Path:
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "tensorfold")
    return Path(root) / "roce"


@lru_cache(maxsize=1)
def proxy_library() -> ctypes.CDLL:
    """Build roce_proxy.c with the host C compiler and libibverbs once per source hash, and bind it."""

    with _LOCK:
        source = _SOURCE.read_bytes()
        target = _cache_dir() / f"roce_proxy-{hashlib.sha256(source).hexdigest()[:16]}.so"
        if not target.exists():
            cc = next((c for c in (os.environ.get("CC"), "gcc", "cc", "clang") if c and shutil.which(c)), None)
            if cc is None:
                raise RuntimeError("the RoCE all-gather needs a C compiler and libibverbs headers")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{os.getpid()}")
            cmd = [cc, "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", str(tmp), str(_SOURCE), "-libverbs", "-lpthread"]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError("building the RoCE proxy failed: " + " ".join(cmd) + "\n" + done.stderr)
            os.replace(tmp, target)
        lib = ctypes.CDLL(str(target), use_errno=True)
    u64, p, i = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
    for name, res, args in (
            ("roce_abi_version", i, []), ("roce_layout", i, [i, u64, ctypes.POINTER(u64)]),
            ("roce_blob_bytes", u64, []),
            ("roce_create", p, [i, i, ctypes.POINTER(ctypes.c_char_p), i, ctypes.POINTER(i), i, p, u64, u64,
                                ctypes.c_char_p, u64]),
            ("roce_set_routes", i, [p, i, i, ctypes.POINTER(i), ctypes.POINTER(i), i]),
            ("roce_local_blob", i, [p, p, u64]), ("roce_connect", i, [p, p, u64]), ("roce_start", i, [p]),
            ("roce_stop", None, [p]), ("roce_failed", i, [p]), ("roce_error", ctypes.c_char_p, [p]),
            ("roce_stat", u64, [p, i]), ("roce_hca_stat", u64, [p, i, i]), ("roce_destroy", None, [p])):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    if lib.roce_abi_version() != 5:
        raise RuntimeError("unexpected RoCE proxy ABI version")
    return lib


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_roce_v1", sources=[str(here / "roce.cpp"), str(here / "roce.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


MAX_HCAS = 4                             # roce_proxy.c's ROCE_MAX_HCAS
_SYS = Path("/sys/class/infiniband")


def _hcas(world: int = 2) -> list[str]:
    """The HCAs the transport uses: TF_ROCE_HCA (or NCCL_IB_HCA), else every ACTIVE one; two ranks take the first
    two (their one cable's rails), more ranks up to four (a triangle's two rails a peer)."""

    raw = os.environ.get("TF_ROCE_HCA") or os.environ.get("NCCL_IB_HCA") or ""
    names = [x.strip().lstrip("=^").split(":")[0] for x in raw.split(",") if x.strip()]
    if not names:
        names = [d.name for d in sorted(_SYS.glob("*")) if "ACTIVE" in (d / "ports/1/state").read_text()]
    return names[:2] if world == 2 else names[:MAX_HCAS]


def _ipv4_gid(hca: str) -> tuple[int, int, int] | None:
    """(GID index, IPv4 as an int, prefix length) of ``hca``'s first RoCE v2 IPv4 GID (sysfs and ``ip``), or None."""

    import ipaddress

    port = _SYS / hca / "ports" / "1"
    entries = []
    for path in (port / "gids").glob("*"):
        try:
            gid = path.read_text().strip()
            kind = (port / "gid_attrs" / "types" / path.name).read_text().strip()
        except OSError:
            continue
        if "v2" not in kind.lower() or not gid.replace(":", "").lower().startswith("0" * 20 + "ffff"):
            continue
        raw = bytes.fromhex(gid.replace(":", ""))
        if raw == bytes(16):
            continue
        entries.append((int(path.name), int(ipaddress.IPv4Address(raw[12:])), path))
    if not entries:
        return None
    index, ip, path = min(entries)
    prefix = 24
    try:
        dev = (port / "gid_attrs" / "ndevs" / path.name).read_text().strip()
        out = subprocess.run(["ip", "-o", "-4", "addr", "show", "dev", dev], capture_output=True, text=True).stdout
        for word in out.split():
            if "/" in word and word.split("/")[0] == str(ipaddress.IPv4Address(ip)):
                prefix = int(word.split("/")[1])
    except (OSError, ValueError):
        pass
    return index, ip, prefix


def plan_routes(addrs: list[list[tuple[int, int] | None]]) -> tuple[dict, int]:
    """Every rank's (IPv4, prefix) a listed HCA (None: unknown) -> ({(rank, peer): [(local HCA, peer HCA), ...]},
    stripes a peer). HCAs pair when they share a subnet, each HCA once a peer, in name order on both sides (so the
    two ends of a pair list the same pairs mirrored); every pair of ranks gets the fewest stripes any pair has,
    which must be at least one."""

    world = len(addrs)

    def net(a):
        return None if a is None else (a[0] >> (32 - a[1]) if a[1] else 0, a[1])

    pairs: dict = {}
    for r in range(world):
        for p in range(world):
            if p == r:
                continue
            used, got = set(), []
            for h, a in enumerate(addrs[r]):
                for rh, b in enumerate(addrs[p]):
                    if rh not in used and net(a) is not None and net(a) == net(b):
                        used.add(rh)
                        got.append((h, rh))
                        break
            pairs[(r, p)] = got
    if world < 2:
        return pairs, 0
    least = min(len(v) for v in pairs.values())
    if least == 0:
        missing = ", ".join(f"{r}-{p}" for (r, p), v in sorted(pairs.items()) if not v and r < p)
        raise RuntimeError(f"RoCE: no HCA of rank(s) {missing} shares a subnet with the peer's: every pair of ranks "
                           "needs a direct path (a cable or a switch); use TF_GLM_COMM=nccl on a ring")
    return {k: v[:least] for k, v in pairs.items()}, least


class RoceGather:
    """One rank's side of the exchange: the pinned region, the proxy thread and the gather launches."""

    def __init__(self, comm, rank: int, world: int, max_bytes: int = 256 << 10) -> None:
        """Two ranks: every listed HCA to the peer's of the same index (b12x's striping), GID index TF_ROCE_GID_INDEX
        (or NCCL_IB_GID_INDEX, 3). More ranks: each HCA's own RoCE v2 IPv4 GID (unless TF_ROCE_GID_INDEX) and routes
        by subnet (``plan_routes``), so a triangle of direct cables reaches each peer over that pair's rails."""
        lib = proxy_library()
        self.lib, self.rank, self.world, self.max_bytes = lib, rank, world, int(max_bytes)
        self.slot_bytes = -(-self.max_bytes // SLOT_ALIGN) * SLOT_ALIGN
        out = (ctypes.c_uint64 * 7)()
        if lib.roce_layout(world, self.slot_bytes, out) != 0:
            raise ValueError(f"RoCE layout refused world {world}, slot {self.slot_bytes}")
        recv_off, flag_off, send_off, ctrl_off, total, self.flag_stride, self.slots = (int(v) for v in out)
        self.region = torch.zeros(total, dtype=torch.uint8, pin_memory=True)   # flags and ctrl start at 0
        base = self.region.data_ptr()
        self.recv, self.flag, self.send, self.ctrl = base + recv_off, base + flag_off, base + send_off, base + ctrl_off
        self.ctrl_words = self.region[ctrl_off:ctrl_off + 28].view(torch.int32).numpy()
        self.classes = BLOCKS.bit_length()
        # epoch, stage arrivals and tail arrivals by power-of-two grid, poison
        self.counters = torch.zeros(2 + 2 * self.classes, dtype=torch.int32, device="cuda")
        self.hcas = _hcas(world)
        fixed = os.environ.get("TF_ROCE_GID_INDEX") or os.environ.get("NCCL_IB_GID_INDEX")
        found = [None] * len(self.hcas) if world == 2 else [_ipv4_gid(h) for h in self.hcas]
        if world == 2 or fixed:
            gids = [int(fixed or 3)] * len(self.hcas)
        else:
            gids = [f[0] if f is not None else 3 for f in found]
        self.addrs = [None if f is None else (f[1], f[2]) for f in found]
        tc = int(os.environ.get("NCCL_IB_TC", "0"), 0)
        names = (ctypes.c_char_p * len(self.hcas))(*[n.encode() for n in self.hcas])
        err = ctypes.create_string_buffer(512)
        self.ctx = lib.roce_create(world, rank, names, len(self.hcas), (ctypes.c_int * len(gids))(*gids), tc,
                                   ctypes.c_void_p(base), total, self.slot_bytes, err, len(err))
        failed = 0 if self.ctx else 1
        blob_len = int(lib.roce_blob_bytes())
        blob = ctypes.create_string_buffer(blob_len)
        if not failed and lib.roce_local_blob(self.ctx, blob, blob_len) != 0:
            failed = 1
        # every rank's verdict, geometry and connection blob through the existing communicator (more than two ranks:
        # each HCA's IPv4 and prefix too, for the routes)
        words = -(-blob_len // 4)
        extra = 0 if world == 2 else 2 * MAX_HCAS
        mine = torch.zeros(4 + extra + words, dtype=torch.int32)
        n_hca = len(self.hcas) if world == 2 else 0
        mine[:4] = torch.tensor([failed, n_hca, self.slot_bytes >> 12, self.flag_stride])
        if extra:                                        # (IPv4 as int32 bits, prefix) a slot; prefix -1: none
            mine[4:4 + extra] = torch.tensor([0, -1] * MAX_HCAS, dtype=torch.int32)
            for h, a in enumerate(self.addrs):
                if a is not None:
                    mine[4 + 2 * h] = a[0] - (1 << 32) if a[0] >= 1 << 31 else a[0]
                    mine[5 + 2 * h] = a[1]
        raw = torch.frombuffer(bytearray(blob.raw.ljust(words * 4, b"\0")), dtype=torch.int32)
        mine[4 + extra:] = raw
        got = torch.empty(world * mine.numel(), dtype=torch.int32, device="cuda")
        comm.all_gather(mine.cuda(), got)
        rows = got.view(world, -1).cpu()
        if int(rows[:, 0].sum()) or len(set(tuple(r[1:4].tolist()) for r in rows)) != 1:
            why = err.value.decode(errors="replace") if not self.ctx else "the ranks' RoCE geometry differs"
            self.close()
            raise RuntimeError(f"RoCE all-gather setup failed on some rank: {why}")
        self.n_flag = len(self.hcas)                     # stripes a peer: every HCA at two ranks
        if extra:
            addrs = [[None if int(r[5 + 2 * h]) < 0 else (int(r[4 + 2 * h]) & 0xFFFFFFFF, int(r[5 + 2 * h]))
                      for h in range(MAX_HCAS)] for r in rows]
            try:
                routes, self.n_flag = plan_routes(addrs)
            except RuntimeError:
                self.close()
                raise
            for p in range(world):
                if p == rank:
                    continue
                local, remote = zip(*routes[(rank, p)])
                n = len(local)
                if lib.roce_set_routes(self.ctx, p, n, (ctypes.c_int * n)(*local), (ctypes.c_int * n)(*remote),
                                       self.n_flag) != 0:
                    why = lib.roce_error(self.ctx).decode(errors="replace")
                    self.close()
                    raise RuntimeError(f"RoCE routes: {why}")
            self.routes = {p: routes[(rank, p)] for p in range(world) if p != rank}
        blobs = b"".join(bytes(r[4 + extra:].numpy().tobytes())[:blob_len] for r in rows)
        self.spin = spin_limit()
        _ext()                      # its first build (tens of seconds) here, before the verdict all-gather below
        ok = lib.roce_connect(self.ctx, blobs, len(blobs)) == 0 and lib.roce_start(self.ctx) == 0
        verdict = torch.tensor([0 if ok else 1], dtype=torch.int32, device="cuda")
        both = torch.empty(world, dtype=torch.int32, device="cuda")
        comm.all_gather(verdict, both)
        if int(both.sum()):
            why = lib.roce_error(self.ctx).decode(errors="replace") if self.ctx else ""
            self.close()
            raise RuntimeError(f"RoCE all-gather could not connect: {why}")

    def fits(self, send: torch.Tensor) -> bool:
        n = send.numel() * send.element_size()
        return self.fits_bytes(n) and send.data_ptr() % PACK == 0 and send.is_contiguous()

    def fits_bytes(self, n: int) -> bool:
        """Whether a gather of ``n`` bytes a rank goes over RoCE (the same answer on every rank)."""
        return 0 < n <= self.max_bytes and n % PACK == 0

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        nbytes = send.numel() * send.element_size()
        packs = nbytes // PACK
        need = max(1, -(-packs // (2 * THREADS)))
        grid = min(1 << (need - 1).bit_length(), BLOCKS)
        klass = grid.bit_length() - 1
        c = self.counters.data_ptr()
        _ext().gather(send, recv, packs, nbytes, packs, self.recv, self.flag, self.send, self.ctrl, self.slot_bytes,
                      c, c + 4 * (1 + klass), c + 4 * (1 + self.classes + klass), c + 4 * (1 + 2 * self.classes),
                      self.spin, self.world, self.rank, self.slots, self.flag_stride, self.n_flag, grid, THREADS)
        if not torch.cuda.is_current_stream_capturing():
            self.check()

    def check(self) -> None:
        """Raise if a wait timed out on the GPU or the proxy thread failed."""

        if self.ctrl_words[2] != 0 or (self.ctx and self.lib.roce_failed(self.ctx)):
            peer, hca = int(self.ctrl_words[3]), int(self.ctrl_words[6])
            why = self.lib.roce_error(self.ctx).decode(errors="replace") if self.ctx else ""
            raise RuntimeError(f"RoCE all-gather failed (sequence {int(self.ctrl_words[2])}, peer {peer}, hca {hca})"
                               f"{': ' + why if why else ''} [{self.stats()}]")

    def stats(self) -> str:
        """The proxy's counters, for a failure report: the doorbell against what was posted and completed."""

        if not self.ctx:
            return "no proxy"
        lib, c = self.lib, self.ctx
        hcas = " ".join(f"hca{h} completed {lib.roce_hca_stat(c, h, 0)} bytes {lib.roce_hca_stat(c, h, 1)}"
                        for h in range(len(self.hcas)))
        return (f"doorbell {int(self.ctrl_words[0])}, posted {lib.roce_stat(c, 0)}, completed {lib.roce_stat(c, 1)}, "
                f"last {lib.roce_stat(c, 2)}; {hcas}; wait limit {self.spin / 1e6:g} s (TF_ROCE_WAIT_S)")

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.roce_stop(self.ctx)
            self.lib.roce_destroy(self.ctx)
            self.ctx = None


class RoceComm:
    """NCCL with small all-gathers over RoCE: the same call, output layout and bits."""

    def __init__(self, nccl, rank: int, world: int, max_bytes: int | None = None) -> None:
        self.nccl, self.rank, self.world = nccl, rank, world
        limit = int(os.environ.get("TF_ROCE_MAX_KB", "256")) << 10 if max_bytes is None else max_bytes
        self.roce = RoceGather(nccl, rank, world, limit)
        self.small = self.large = 0
        self.settled = False

    def settle(self) -> None:
        """Startup is over on this rank (its engine is built). Until then each eager RoCE gather waits for the peer in
        an NCCL barrier first: the ranks build their CUDA extensions on their own and can drift tens of seconds apart
        on a first start, past what a RoCE wait allows; NCCL waits without that limit. Captured gathers never do."""

        self.settled = True

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """RoCE for a gather whose size fits, NCCL otherwise: decided by the byte count alone, which every rank's
        matching call shares. A send or receive buffer that is not 16-byte aligned or contiguous goes through an
        aligned copy: choosing by the buffers' addresses let two ranks pick different transports for the same gather
        (one waiting in RoCE, the other in NCCL, forever)."""

        n = send.numel() * send.element_size()
        if self.roce.fits_bytes(n):
            self.small += 1
            s = send if send.is_contiguous() and send.data_ptr() % PACK == 0 else send.contiguous().clone()
            r = recv if recv.is_contiguous() and recv.data_ptr() % PACK == 0 else torch.empty_like(recv,
                                                                                               memory_format=torch.contiguous_format)
            if not self.settled and not torch.cuda.is_current_stream_capturing():
                self.nccl.barrier()
            self.roce.all_gather(s, r)
            if r is not recv:
                recv.copy_(r)
        else:
            self.large += 1
            self.nccl.all_gather(send, recv)

    def barrier(self) -> None:
        self.nccl.barrier()

    def check(self) -> None:
        self.roce.check()

    def __getattr__(self, name):
        return getattr(self.nccl, name)
