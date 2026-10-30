"""
Change a voxel theater's resolution without changing its real size.

The theater is a fixed piece of space; its resolution says how finely that space
is cut. ``factor`` 2 doubles the resolution: every block becomes 2×2×2 blocks
half as wide, each 1/8 of its matter. ``factor`` 0.5 halves it: every 2×2×2
group becomes one block twice as wide. Any whole number n (or 1/n) works the
same way, n³ blocks per block.

  - Finer is EXACT: the same material fills the same space, so mass, volume,
    centre of mass and every contact face are unchanged.
  - Coarser LOSES detail, and says how much. A merged block must be one
    material (sigma-ground can ground a material, not an arbitrary mixture),
    so it becomes matter only when at least half its blocks were matter, and
    then the material most of them were made of (a tie goes to the denser
    one). A one-block railing vanishes at half resolution; a merged
    granite/iron block weighs what granite does. The report gives the mass and
    the block count of every material before and after.

The grid's low corner stays put, so the theater keeps its place in the world;
a grid that does not divide evenly is padded with air at its high end first.
"""
from __future__ import annotations

from fractions import Fraction


def parse_factor(factor):
    """``factor`` → ("refine" | "coarsen", n). 2 → refine by 2; 0.5 → coarsen by 2."""
    f = Fraction(factor).limit_denominator(64)
    if f <= 0 or abs(float(f) - float(factor)) > 1e-9 or (f.numerator != 1 and f.denominator != 1):
        raise ValueError(f"resolution factor must be a whole number n or 1/n, got {factor!r}")
    if f.denominator == 1:
        return "refine", f.numerator
    return "coarsen", f.denominator


def pad_to(a, n, fill):
    """Pad ``a`` at the high end of each axis to a multiple of ``n``."""
    import numpy as np
    pad = [(0, (-s) % n) for s in a.shape]
    return np.pad(a, pad, constant_values=fill) if any(p for _, p in pad) else a


def blocks(a, n):
    """(X, Y, Z) → (X/n, Y/n, Z/n, n³): each coarse cell's n³ fine cells, last axis."""
    X, Y, Z = a.shape
    return (a.reshape(X // n, n, Y // n, n, Z // n, n)
             .transpose(0, 2, 4, 1, 3, 5).reshape(X // n, Y // n, Z // n, n ** 3))


def refine(a, n):
    """Every cell → n³ copies."""
    return a.repeat(n, 0).repeat(n, 1).repeat(n, 2)


def coarsen_labels(label, n, densities):
    """Merge n³ blocks into one. ``densities[k]`` ranks label k on a tie (index 0 = air).

    Returns ``(coarse_label, fine_blocks)``; ``fine_blocks`` is the padded fine
    labels in ``blocks`` layout, for callers that carry per-cell state along.
    """
    import numpy as np

    fine = blocks(pad_to(label, n, 0), n)
    nmat = len(densities)
    counts = np.stack([(fine == k).sum(-1) for k in range(1, nmat)], -1)   # matter only
    rank = np.argsort(np.argsort(np.asarray(densities[1:], dtype=float)))   # 0 = lightest
    winner = np.argmax(counts * nmat + rank, -1) + 1
    coarse = np.where(counts.sum(-1) * 2 >= n ** 3, winner, 0).astype(label.dtype)
    return coarse, fine


def resample_field(field, factor):
    """A ``VoxelField`` at another resolution, same space. Returns ``(field, report)``."""
    import numpy as np
    from .voxelize import _finalize

    how, n = parse_factor(factor)
    if how == "refine" and n == 1:
        return field, {"factor": 1, "block_m": field.voxel_size, "mass_before_kg": field.mass_kg,
                       "mass_after_kg": field.mass_kg, "mass_change_kg": {}}
    dens = field.density_by_label or {}
    densities = [0.0] + [float(dens[m]) for m in field.materials[1:]]
    if how == "refine":
        label, pitch = refine(field.label_grid, n), field.voxel_size / n
    else:
        label, _ = coarsen_labels(field.label_grid, n, densities)
        pitch = field.voxel_size * n
    lo = np.asarray(field.center) - np.asarray(field.label_grid.shape) * field.voxel_size / 2
    out = _finalize(label, field.materials, pitch, lo, lambda m: float(dens[m]), field.watertight_frac,
                    extra_note=f"resampled x{factor} from {field.voxel_size:g} m blocks")

    def tally(lab, cell_m3):
        return {m: {"blocks": int((lab == k).sum()), "mass_kg": float((lab == k).sum()) * cell_m3 * densities[k]}
                for k, m in enumerate(field.materials) if k}

    before = tally(field.label_grid, field.voxel_size ** 3)
    after = tally(label, pitch ** 3)
    change = {m: after[m]["mass_kg"] - before[m]["mass_kg"] for m in before        # − lost, + gained
              if abs(after[m]["mass_kg"] - before[m]["mass_kg"]) > 1e-12 * max(field.mass_kg, 1e-30)}
    report = {"factor": factor, "block_m": pitch, "blocks_before": int((field.label_grid > 0).sum()),
              "blocks_after": int((label > 0).sum()), "mass_before_kg": field.mass_kg, "mass_after_kg": out.mass_kg,
              "materials_before": before, "materials_after": after, "mass_change_kg": change}
    return out, report


__all__ = ["parse_factor", "resample_field", "coarsen_labels", "refine", "blocks", "pad_to"]
