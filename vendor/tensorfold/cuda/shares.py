"""How a dimension splits over tensor-parallel ranks: whole units (a head, a 128-column Hadamard block, 64 vocabulary
rows), as even as they go, the remainder to the lowest ranks. Two ranks of an even count get exact halves."""

from __future__ import annotations


def parts(total: int, world: int, unit: int = 1) -> list[int]:
    """Each rank's share of ``total`` (a multiple of ``unit``) in rank order: e.g. 64 heads over 3 ranks 22, 21, 21;
    2048 columns in units of 128 over 3 ranks 768, 640, 640."""

    total, world, unit = int(total), int(world), int(unit)
    if world < 1 or unit < 1:
        raise ValueError(f"shares of {total} over {world} ranks in units of {unit}")
    if total % unit:
        raise ValueError(f"{total} is not a whole number of {unit}-wide units")
    units = total // unit
    if units < world:
        raise ValueError(f"{total} ({units} units of {unit}) cannot give each of {world} ranks a unit")
    base, extra = divmod(units, world)
    return [(base + (r < extra)) * unit for r in range(world)]


def share(total: int, world: int, rank: int, unit: int = 1) -> tuple[int, int]:
    """Rank ``rank``'s [start, stop) of ``total`` (``parts``)."""

    sizes = parts(total, world, unit)
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} of {world}")
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


def scaled(total: int, world: int, rank: int, unit: int, length: int) -> tuple[int, int]:
    """Rank ``rank``'s [start, stop) of an axis of ``length`` elements that holds the ``total`` (e.g. a trellis's
    128 tile columns for 2,048 expert columns, a packed row's words): the share of ``total`` scaled to the axis,
    which must land on whole elements."""

    a, b = share(total, world, rank, unit)
    if (a * length) % total or (b * length) % total:
        raise ValueError(f"rank {rank} of {world}: [{a}, {b}) of {total} is not whole on an axis of {length}")
    return a * length // total, b * length // total
