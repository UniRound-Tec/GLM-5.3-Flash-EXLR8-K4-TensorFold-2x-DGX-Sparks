"""NCCL all-gather (and paired send/receive) on the current stream so CUDA graphs capture it; a rank-order sum after
it keeps ranks bit-equal."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os

import torch

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4}


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
    candidates = [os.environ.get("TF_NCCL_LIB", "")]
    found = ctypes.util.find_library("nccl")
    if found:
        candidates.append(found)
    candidates += glob.glob("/usr/lib/*/libnccl.so.2") + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/nccl/lib/libnccl.so.2")
    candidates += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl*.so*"))
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError("libnccl not found (set TF_NCCL_LIB)")


class NCCL:
    def __init__(self, rank: int, world: int, master: str, port: int) -> None:
        from datetime import timedelta

        from torch.distributed import TCPStore

        self.rank, self.world = rank, world
        self.lib = _library()
        lib = self.lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
        for name in ("ncclSend", "ncclRecv"):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p]
        lib.ncclGroupStart.argtypes = []
        lib.ncclGroupEnd.argtypes = []
        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=600))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set("tf_nccl_uid", bytes(uid.internal))
        else:
            raw = self.store.get("tf_nccl_uid")
            ctypes.memmove(ctypes.addressof(uid), raw, 128)
        self.comm = ctypes.c_void_p()
        torch.cuda.current_device()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank))

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NCCL error {code}: {self.lib.ncclGetErrorString(code).decode()}")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order (contiguous tensors, same dtype)."""

        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclAllGather(send.data_ptr(), recv.data_ptr(), send.numel(), _DTYPES[send.dtype],
                                           self.comm, stream))

    def ready(self, label: str, *, every: float = 60.0, timeout: float = 3600.0) -> None:
        """Every rank finishes ``label`` before any goes on; a rank missing after ``timeout`` s is named."""

        import time
        from datetime import timedelta

        self.store.set(f"tf_ready/{label}/{self.rank}", "1")
        others = [r for r in range(self.world) if r != self.rank]
        started = time.monotonic()
        while True:
            try:
                self.store.wait([f"tf_ready/{label}/{r}" for r in others], timedelta(seconds=every))
                return
            except Exception as exc:                  # noqa: BLE001  (the store's timeout; anything else goes up)
                if "timeout" not in str(exc).lower():
                    raise
            waited = time.monotonic() - started
            missing = ", ".join(str(r) for r in others)
            if waited >= timeout:
                raise RuntimeError(f"rank {self.rank} finished {label} but rank {missing} has not after "
                                   f"{waited / 60:.0f} min: check that rank's log (a CUDA extension build waiting on "
                                   "a lock names the lock there)")
            print(f"[tensorfold] rank {self.rank} finished {label}; waiting for rank {missing} ({waited:.0f} s)",
                  flush=True)

    def exchange(self, sends: list[torch.Tensor], recvs: list[torch.Tensor], peer: int) -> None:
        """One NCCL group on the current stream: every ``sends[i]`` to ``peer`` and ``recvs[i]`` from it (contiguous
        tensors; the peer's matching call has the same count, sizes and dtypes in the same order)."""

        if len(sends) != len(recvs) or peer == self.rank or not 0 <= peer < self.world:
            raise ValueError("exchange: one receive per send, from another rank")
        for s, r in zip(sends, recvs):
            if s.numel() != r.numel() or s.dtype != r.dtype or not (s.is_contiguous() and r.is_contiguous()):
                raise ValueError("exchange: each send and receive must be contiguous and alike in size and dtype")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclGroupStart())
        try:
            for s, r in zip(sends, recvs):
                if s.numel():
                    self._check(self.lib.ncclSend(s.data_ptr(), s.numel(), _DTYPES[s.dtype], peer, self.comm, stream))
                    self._check(self.lib.ncclRecv(r.data_ptr(), r.numel(), _DTYPES[r.dtype], peer, self.comm, stream))
        finally:
            self._check(self.lib.ncclGroupEnd())

    def exchange_all(self, sends: dict[int, list[torch.Tensor]], recvs: dict[int, list[torch.Tensor]]) -> None:
        """One NCCL group on the current stream with several peers: every ``sends[p][i]`` to peer p and
        ``recvs[p][i]`` from it (an all-to-all of row blocks; each peer's matching call names this rank with the
        same sizes and dtypes in the same order)."""

        if set(sends) != set(recvs) or any(p == self.rank or not 0 <= p < self.world for p in sends):
            raise ValueError("exchange_all: sends and receives with the same other ranks")
        for p in sends:
            if len(sends[p]) != len(recvs[p]):
                raise ValueError("exchange_all: one receive per send")
            for s, r in zip(sends[p], recvs[p]):
                if s.numel() != r.numel() or s.dtype != r.dtype or not (s.is_contiguous() and r.is_contiguous()):
                    raise ValueError("exchange_all: each send and receive must be contiguous and alike")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclGroupStart())
        try:
            for p in sorted(sends):
                for s, r in zip(sends[p], recvs[p]):
                    if s.numel():
                        self._check(self.lib.ncclSend(s.data_ptr(), s.numel(), _DTYPES[s.dtype], p, self.comm, stream))
                        self._check(self.lib.ncclRecv(r.data_ptr(), r.numel(), _DTYPES[r.dtype], p, self.comm, stream))
        finally:
            self._check(self.lib.ncclGroupEnd())

    def barrier(self) -> None:
        x = torch.zeros((1,), dtype=torch.float32, device="cuda")
        y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
        self.all_gather(x, y)
        torch.cuda.synchronize()
