"""Bounded candidate packing; retain TensorFold's exact CPU sampling rules.

Transport tensors always start at their allocation base. This scratch is owned
by one scheduler/weights object and consumed on its single CUDA stream.
"""
import ctypes
from pathlib import Path
import torch

MAX_ROWS = 32
MAX_K = 256


class Buffers:
    def __init__(self, device, vocabulary, world):
        self.device, self.vocabulary, self.world = device, vocabulary, world
        self.logits = torch.empty((MAX_ROWS, vocabulary), dtype=torch.float32, device=device)
        self.values = torch.empty(MAX_ROWS * MAX_K, dtype=torch.float32, device=device)
        self.ids = torch.empty(MAX_ROWS * MAX_K, dtype=torch.int64, device=device)
        self.flat = torch.empty(2 * MAX_ROWS * MAX_K, dtype=torch.float32, device=device)
        self.got = torch.empty(world * self.flat.numel(), dtype=torch.float32, device=device)
        self.host = torch.empty(self.got.numel(), dtype=torch.float32, pin_memory=True)
        self.ready = torch.cuda.Event()
        self.lib = ctypes.CDLL(str(Path(__file__).with_name('libatlas_experts.so')))
        self.pack_kernel = self.lib.atlas_pack_sampling
        self.pack_kernel.argtypes = [ctypes.c_uint64]*3+[ctypes.c_int]*3+[ctypes.c_uint64]
        self.pack_kernel.restype = ctypes.c_int

    def pack(self, w, parts, layout):
        at = 0
        for k, rows, n in layout:
            logits = parts[k][0]
            if logits.dtype != torch.float32:
                self.logits[:rows].copy_(logits)
                logits = self.logits[:rows]
            count = rows * n
            values, ids = self.values[:count].view(rows, n), self.ids[:count].view(rows, n)
            torch.topk(logits, n, dim=-1, out=(values, ids))
            packed = self.flat[at:at + count * 2].view(rows, n * 2)
            code = self.pack_kernel(values.data_ptr(),ids.data_ptr(),packed.data_ptr(),count,n,
                                    w.vocab_offset,torch.cuda.current_stream().cuda_stream)
            if code:
                raise RuntimeError(f'Sampling pack launch error {code}')
            at += count * 2
        flat = self.flat[:at]
        if w.comm is None:
            source = flat
        else:
            source = self.got[:w.world * at]
            w.comm.all_gather(flat, source)
        self.host[:source.numel()].copy_(source, non_blocking=True)
        self.ready.record()
        self.ready.synchronize()
        return self.host[:source.numel()].view(self.world, at)


def packed_host(w, parts, margin):
    """Return None outside the bounded common path; caller uses the original implementation."""
    layout = []
    vocabulary, device = None, None
    for k, (logits, positions, sampling) in enumerate(parts):
        greedy = sampling is None or sampling.temperature <= 0
        if not greedy and not sampling.top_k:
            continue
        n = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + margin)
        if n > MAX_K or logits.shape[0] > MAX_ROWS:
            return None
        if vocabulary is not None and (vocabulary != logits.shape[1] or device != logits.device):
            return None
        vocabulary, device = logits.shape[1], logits.device
        layout.append((k, logits.shape[0], n))
    if not layout or sum(r for _, r, _ in layout) > MAX_ROWS:
        return None
    world = 1 if w.comm is None else w.world
    buffers = getattr(w, '_sampling_reuse', None)
    if buffers is None:
        buffers = w._sampling_reuse = Buffers(device, vocabulary, world)
    elif (buffers.device, buffers.vocabulary, buffers.world) != (device, vocabulary, world):
        return None
    return buffers.pack(w, parts, layout), layout
