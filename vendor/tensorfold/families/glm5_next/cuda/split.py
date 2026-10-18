"""Read each rank directly from checkpoint bytes or a saved rank folder, preserving packed groups at every split boundary."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROW = (
    r"\.mlp\.experts\.\d+\.(gate|up)_proj\.",
    r"\.mlp\.shared_experts\.(gate|up)_proj\.",
    r"\.mlp\.(gate|up)_proj\.",
    r"\.self_attn\.(q|k|v)_proj\.",
    r"\.self_attn\.(q|k|v)_conv1d\.",
    r"\.self_attn\.(f_b|g_b|b)_proj\.",
    r"\.self_attn\.(A_log|dt_bias)$",
    r"\.self_attn\.(q_b|kv_b)_proj\.",
)
COL = (
    r"\.mlp\.experts\.\d+\.down_proj\.",
    r"\.mlp\.shared_experts\.down_proj\.",
    r"\.mlp\.down_proj\.",
    r"\.self_attn\.o_proj\.",
)
REP = (
    r"^lm_head\.", r"embed_tokens\.", r"^model\.language_model\.norm\.weight$",
    r"_layernorm\.weight$", r"\.hc_(attn|ffn)_(fn|base|scale)$", r"\.mlp\.gate\.(weight|e_score_correction_bias)$",
    r"\.self_attn\.indexer\.", r"\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.self_attn\.(f_a|g_a)_proj\.",
    r"\.self_attn\.o_norm\.weight$", r"\.(eh_proj)\.", r"\.(enorm|hnorm)\.weight$", r"\.shared_head\.norm\.weight$",
)
RUN = 128 << 20          # most bytes one read of neighbouring tensors takes (``RankReader.prefetch``)
GAP = 1 << 20            # most unused bytes such a read spans between two tensors (more reads other layers twice)
READERS = 8              # reads in flight: past this the layers are built slower than they are read
DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "U8": 1, "I8": 1, "I64": 8,
               "F64": 8}
# the files a rank folder needs besides its weights (the tokenizer, chat template and configs)
SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "processor_config.json", "model.safetensors.index.json")


# EXL3 gate/up split tile columns and svh; down splits tile rows and suh; each replicates the remaining tensors.
EXL3_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(trellis|suh|svh|mcg)$")
EXL3_RULES = {("gate", "trellis"): "dim1", ("gate", "suh"): "rep", ("gate", "svh"): "row",
              ("down", "trellis"): "row", ("down", "suh"): "row", ("down", "svh"): "rep"}


def rule(name: str) -> str:
    if name.startswith("model.visual."):
        return "drop"
    m = EXL3_EXPERT.search(name)
    if m:
        proj, part = m.groups()
        return "rep" if part == "mcg" else EXL3_RULES[("gate" if proj == "up" else proj, part)]
    hits = [kind for kind, pats in (("row", ROW), ("col", COL), ("rep", REP)) if any(re.search(p, name) for p in pats)]
    if len(hits) != 1:
        raise ValueError(f"{name}: split rule is ambiguous or missing ({hits})")
    return hits[0]


def read_header(path: str | Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


AXIS = {"row": 0, "col": 1, "dim1": 1}


def _cut(shape: list[int], kind: str, rank: int, cut: tuple[int, int] | None) -> tuple[int, int]:
    """[start, stop) of the split axis: ``cut`` (``tp.Layout.cut``), or rank's half of an even axis (two ranks)."""

    if kind == "col" and len(shape) != 2:
        raise ValueError(f"column split needs a 2-D shape, got {shape}")
    if kind == "dim1" and len(shape) < 2:
        raise ValueError(f"split of the second axis needs a second dim, got {shape}")
    n = shape[AXIS[kind]]
    if cut is None:
        if n % 2:
            raise ValueError(f"{kind} split of an odd dim {shape}")
        return rank * (n // 2), (rank + 1) * (n // 2)
    a, b = int(cut[0]), int(cut[1])
    if not 0 <= a < b <= n:
        raise ValueError(f"{kind} split [{a}, {b}) outside {shape}")
    return a, b


def split_bytes(raw: np.ndarray, shape: list[int], itemsize: int, kind: str, rank: int,
                cut: tuple[int, int] | None = None) -> tuple[np.ndarray, list[int]]:
    """A tensor's bytes -> rank's part of them and its shape; ``cut``: the rank's [start, stop) of the split axis
    (``tp.Layout.cut``; default rank's half, the two-rank split)."""

    if kind == "rep":
        return raw, list(shape)
    if kind not in AXIS:
        raise ValueError(kind)
    a, b = _cut(shape, kind, rank, cut)
    if kind == "row":
        per = raw.size // shape[0]
        return raw[a * per:b * per], [b - a] + list(shape[1:])
    inner = int(np.prod(shape[2:])) * itemsize
    view = raw.reshape(shape[0], shape[1] * inner)
    part = np.ascontiguousarray(view[:, a * inner:b * inner])
    return part.reshape(-1), [shape[0], b - a] + list(shape[2:])


def split_device(raw, shape: list[int], itemsize: int, kind: str, rank: int, cut: tuple[int, int] | None = None):
    """``split_bytes`` for a uint8 tensor on the GPU: the rank's part (a new contiguous tensor) and its shape."""

    if kind == "rep":
        return raw.clone(), list(shape)
    if kind not in AXIS:
        raise ValueError(kind)
    a, b = _cut(shape, kind, rank, cut)
    if kind == "row":
        per = raw.numel() // shape[0]
        return raw[a * per:b * per].clone(), [b - a] + list(shape[1:])
    inner = int(np.prod(shape[2:])) * itemsize
    part = raw.view(shape[0], shape[1] * inner)[:, a * inner:b * inner]
    return part.contiguous().reshape(-1), [shape[0], b - a] + list(shape[2:])


def layout_for(model_dir: str | Path, rank: int, world: int):
    """``tp.Layout`` of rank ``rank`` of ``world`` from the checkpoint's config."""

    from .tp import Layout

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    return Layout.from_text(dict(raw.get("text_config") or raw), rank, world)


def rank_cut(layout, name: str, shape: list[int], kind: str) -> tuple[int, int] | None:
    """The rank's [start, stop) of ``name``'s split axis by ``layout`` (None for a replicated tensor)."""

    if kind in ("rep", "drop"):
        return None
    if layout is None:                                    # two ranks without a config: halves
        return None
    family = layout.family(name)
    if family is None or family not in layout.totals:
        if layout.world == 2:                             # a family the config does not size: halves, as ever
            return None
        raise ValueError(f"{name}: split ({kind}) but the config sizes no tensor-parallel family for it")
    return layout.cut(family, shape[AXIS[kind]])


def torch_dtype(dtype: str):
    import torch

    return {"U32": torch.uint32, "I32": torch.int32, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16,
            "I16": torch.int16, "U16": torch.uint16, "U8": torch.uint8, "I8": torch.int8, "I64": torch.int64,
            "F64": torch.float64}[dtype]


def rank_files(model_dir: str | Path, rank: int) -> list[Path]:
    return sorted(Path(model_dir).glob(f"*.rank{rank}.safetensors"))


def folder_world(files: list[Path]) -> int:
    """The tensor-parallel size a rank folder was split for (its ``tf_tp`` metadata; none: the two-rank split)."""

    worlds = set()
    for path in files:
        meta = read_header(path)[0].get("__metadata__") or {}
        worlds.add(int(meta.get("tf_tp", 2)))
    if len(worlds) > 1:
        raise ValueError(f"rank files split for different sizes: {sorted(worlds)}")
    return worlds.pop() if worlds else 2


class RankReader:
    """Read stored-dtype CPU tensors for one rank from the full checkpoint or its pre-split folder."""

    def __init__(self, model_dir: str | Path, rank: int, world: int = 2) -> None:
        from tensorfold.cuda.direct_read import ReadAhead, Reader, SafeTensors

        self.dir, self.rank, self.world = Path(model_dir), rank, int(world)
        mine = rank_files(self.dir, rank)
        other = sorted(p for p in self.dir.glob("*.rank*.safetensors") if p not in mine)
        if other and not mine:
            raise ValueError(f"{self.dir} holds another rank's share ({other[0].name}): give rank {rank} its own "
                             "folder or the full checkpoint")
        if mine and folder_world(mine) != self.world:
            raise ValueError(f"{self.dir} holds rank {rank}'s share for --tp {folder_world(mine)}, not --tp "
                             f"{self.world}: split it again (python -m {__name__} --tp {self.world}) or give the "
                             "full checkpoint")
        self._layout = False                              # read from config.json at the first split tensor
        self.io = Reader()                                # O_DIRECT reads where the file system allows them
        self.reads = ReadAhead(self.io, READERS, RUN, GAP)
        self.split = bool(mine)
        if self.split:
            self.folder = SafeTensors(mine, self.io)
            self.index = dict.fromkeys(self.folder.keys())
            return
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            names = json.loads(index.read_text())["weight_map"]
        else:
            names = {k: p.name for p in sorted(self.dir.glob("*.safetensors")) for k in read_header(p)[0]
                     if k != "__metadata__"}
        self.files: dict[str, tuple[dict, int]] = {}
        self.index = dict(names)

    def prefetch(self, names, device=None) -> None:
        """Start reading ``names`` (a layer's hundreds of small expert tensors) in shared reads; with a CUDA ``device``, ``get`` returns them uploaded."""

        items = []
        for name in dict.fromkeys(names):
            span = self._span(name)
            items.append((name, span[0], span[1], span[2], span))
        self.reads.queue(items, device, self._cut)

    @property
    def layout(self):
        """This rank's ``tp.Layout`` (None without a config.json at two ranks: plain halves)."""

        if self._layout is False:
            if self.world == 2 and not (self.dir / "config.json").exists():
                self._layout = None
            else:
                self._layout = layout_for(self.dir, self.rank, self.world)
        return self._layout

    @property
    def ahead(self) -> dict:
        return self.reads.ahead

    def get(self, name: str):
        out = self.reads.take(name)
        return out if out is not None else self._read(name)

    def close(self) -> None:
        self.reads.close()

    def _span(self, name: str) -> tuple:
        """(file, first byte, end byte, split kind, shape, dtype, cut) of the bytes this rank reads for ``name``;
        ``cut``: the rank's [start, stop) of the split axis."""

        if self.split:                                    # a rank folder holds the rank's tensors as they are
            path, begin, n, dtype, shape = self.folder.where[name]
            return str(path), begin, begin + n, "rep", list(shape), dtype, None
        file = str(self.dir / self.index[name])
        if file not in self.files:
            self.files[file] = read_header(file)
        header, base = self.files[file]
        info = header[name]
        kind = rule(name)
        if kind == "drop":
            raise KeyError(f"{name} is not used by the engine")
        a, b = info["data_offsets"]
        shape = list(info["shape"])
        cut = rank_cut(self.layout, name, shape, kind)
        if kind == "row" and cut is None and shape and shape[0] % 2 == 0:
            cut = _cut(shape, kind, self.rank, None)
        if kind == "row" and cut is not None:             # the rank's rows are one run: read only those
            per = (b - a) // shape[0]
            a, b = a + cut[0] * per, a + cut[1] * per
            kind, shape, cut = "rep", [cut[1] - cut[0]] + shape[1:], None
        return file, base + a, base + b, kind, shape, info["dtype"], cut

    def _tensor(self, raw: np.ndarray, span: tuple, own: bool):
        """The rank's tensor from the span's bytes; ``own``: never a view of ``raw`` (a shared read's buffer)."""

        import torch

        _, _, _, kind, shape, dtype, cut = span
        data, shape = split_bytes(raw, shape, DTYPE_BYTES[dtype], kind, self.rank, cut)
        if own and np.may_share_memory(data, raw):
            data = data.copy()
        return torch.from_numpy(data).view(torch_dtype(dtype)).reshape(shape)

    def _read(self, name: str):
        if self.split:
            return self.folder.get(name)
        span = self._span(name)
        return self._tensor(self.io.read(span[0], span[1], span[2] - span[1]).numpy(), span, own=False)

    def _cut(self, raw, span: tuple):
        """The rank's tensor from its span's bytes, a view of a shared read (so copied), on the host or the device."""

        if not raw.is_cuda:
            return self._tensor(raw.numpy(), span, own=True)
        _, _, _, kind, shape, dtype, cut = span
        data, shape = split_device(raw, shape, DTYPE_BYTES[dtype], kind, self.rank, cut)
        return data.view(torch_dtype(dtype)).reshape(shape)

def write(path: str, tensors: list[tuple[str, str, list[int], np.ndarray]], metadata: dict | None) -> None:
    header: dict = {}
    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.size]}
        offset += data.size
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in tensors:
            f.write(memoryview(data))
    os.replace(tmp, path)


def split_file(src: str | Path, out: str | Path, rank: int, world: int = 2, layout=None) -> dict:
    """One checkpoint file -> OUT/<stem>.rank<R>.safetensors with the rank's part of every tensor it keeps (of
    ``world`` ranks by ``layout``, a ``tp.Layout``; the metadata names a size past 2 as ``tf_tp``)."""

    header, base = read_header(src)
    metadata = header.pop("__metadata__", None)
    if layout is None and (int(world) != 2 or (Path(src).parent / "config.json").exists()):
        layout = layout_for(Path(src).parent, rank, world)
    if int(world) != 2:
        metadata = {**(metadata or {}), "tf_tp": str(int(world))}
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    stem = os.path.basename(str(src)).replace(".safetensors", "")
    summary = {"rep": 0, "row": 0, "col": 0, "dim1": 0, "drop": 0}
    part = []
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        info = header[name]
        kind = rule(name)
        summary[kind] += 1
        if kind == "drop":
            continue
        a, b = info["data_offsets"]
        itemsize = DTYPE_BYTES[info["dtype"]]
        data, shape = split_bytes(mm[base + a:base + b], info["shape"], itemsize, kind, rank,
                                  rank_cut(layout, name, list(info["shape"]), kind))
        if int(np.prod(shape)) * itemsize != data.size:
            raise ValueError(f"{name}: {shape} does not match {data.size} bytes")
        part.append((name, info["dtype"], shape, data))
    os.makedirs(out, exist_ok=True)
    if part:
        write(os.path.join(str(out), f"{stem}.rank{rank}.safetensors"), part, metadata)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="the checkpoint (e.g. the snapshot `tensorfold pull` downloaded)")
    p.add_argument("--tp", type=int, choices=(2, 3, 4), default=2, help="ranks the model is split over")
    p.add_argument("--rank", type=int, choices=(0, 1, 2, 3), required=True, help="the rank this machine serves")
    p.add_argument("out", type=Path, help="the folder to write (then: tensorfold serve OUT --tp N --rank R ...)")
    args = p.parse_args(argv)
    if args.rank >= args.tp:
        p.error(f"--rank {args.rank} needs --tp {args.rank + 1} or more")
    layout = layout_for(args.model_dir, args.rank, args.tp)
    files = sorted(args.model_dir.glob("model-*.safetensors"))
    if not files:
        raise SystemExit(f"{args.model_dir}: no model-*.safetensors files")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        if (args.model_dir / name).exists():
            shutil.copyfile(args.model_dir / name, args.out / name)
    for src in files:
        stem = src.name.replace(".safetensors", "")
        if (args.out / f"{stem}.rank{args.rank}.safetensors").exists():
            continue
        print(src.name, split_file(src, args.out, args.rank, args.tp, layout), flush=True)
    print(f"rank {args.rank}'s share of {len(files)} files in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
