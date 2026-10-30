"""
A voxel theater's resolution: the same real space, cut finer or coarser.

The theater is a fixed piece of space; its resolution says how finely that space
is cut. It is chosen once, when the theater is built (``voxel_exec``'s
``resolution``), and never changes while the world runs. ``factor`` 2 doubles the resolution: every block becomes 2×2×2 blocks
half as wide, each 1/8 of its matter. ``factor`` 0.5 halves it: every 2×2×2
group becomes one block twice as wide. Any whole number n (or 1/n) works the
same way, n³ blocks per block.

  - Finer is EXACT: the same material fills the same space, so mass, volume,
    centre of mass and every contact face are unchanged.
  - Coarser cannot keep everything: a merged block must be one material
    (sigma-ground can ground a material, not an arbitrary mixture). Two ways:

    keep="mass" (the default): each material's mass is a budget. It gets as many
    merged blocks as its blocks fill (its count / n³, rounded), placed first
    where the merged cell is most full of it, so the shape holds; among equally
    full cells an ordered-dither pattern spreads them evenly. A one-block floor
    (half of every merged cell) becomes a checkerboard of full blocks with the
    floor's mass, not a slab twice as thick. Each material's mass is then right
    to within half a merged block; a lone block (1/n³ of one) vanishes.

    keep="shape": every merged block votes alone: matter when at least half its
    blocks were, of the material most of them were (a tie goes to the denser).
    Solid-looking, but a thin floor doubles in thickness and its mass with it.

    Either way the report gives every material's blocks and mass before and
    after.

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


def _dither(shape):
    """An ordered-dither rank per cell, 3-D Bayer style: the most significant bit
    alternates like a checkerboard (x^y^z), then rows (y^z), then layers (z),
    then the same one scale up. Equally full cells are filled in this order, so a
    half-filled layer comes out a checkerboard, a quarter-filled one every other
    row of it, and so on."""
    import numpy as np
    x, y, z = np.indices(shape)
    key = np.zeros(shape, dtype=np.int64)
    for level in range(6):                       # the finest level is the most significant
        b = lambda a: (a >> level) & 1
        key = key * 8 + (b(x) ^ b(y) ^ b(z)) * 4 + (b(y) ^ b(z)) * 2 + b(z)
    return key


def coarsen_labels(label, n, densities, keep="mass"):
    """Merge n³ blocks into one; ``keep`` is "mass" or "shape" (see the module).
    ``densities[k]`` is label k's density (index 0 = air): a tie goes to the denser.

    Returns ``(coarse_label, fine_blocks)``; ``fine_blocks`` is the padded fine
    labels in ``blocks`` layout, for callers that carry per-cell state along.
    """
    import numpy as np

    fine = blocks(pad_to(label, n, 0), n)
    nmat = len(densities)
    counts = np.stack([(fine == k).sum(-1) for k in range(1, nmat)], -1)   # matter only
    rank = np.argsort(np.argsort(np.asarray(densities[1:], dtype=float)))   # 0 = lightest
    if keep == "shape":
        winner = np.argmax(counts * nmat + rank, -1) + 1
        return np.where(counts.sum(-1) * 2 >= n ** 3, winner, 0).astype(label.dtype), fine
    if keep != "mass":
        raise ValueError(f"keep must be 'mass' or 'shape', got {keep!r}")

    quota = np.rint(counts.reshape(-1, nmat - 1).sum(0) / n ** 3).astype(np.int64)
    cell, mat = np.nonzero(counts.reshape(-1, nmat - 1))
    full = counts.reshape(-1, nmat - 1)[cell, mat]
    order = np.lexsort((-rank[mat], _dither(counts.shape[:3]).ravel()[cell], -full))
    coarse = np.zeros(counts.shape[0] * counts.shape[1] * counts.shape[2], dtype=label.dtype)
    left = int(quota.sum())
    for c, m in zip(cell[order], mat[order]):   # fullest first; one material per cell
        if left == 0:
            break
        if quota[m] and not coarse[c]:
            coarse[c] = m + 1
            quota[m] -= 1
            left -= 1
    return coarse.reshape(counts.shape[:3]), fine


def resample_field(field, factor, keep="mass"):
    """A ``VoxelField`` at another resolution, same space. Returns ``(field, report)``.
    ``keep`` ("mass" | "shape") only matters when merging; see the module."""
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
        label, _ = coarsen_labels(field.label_grid, n, densities, keep)
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
    report = {"factor": factor, "keep": keep if how == "coarsen" else None, "block_m": pitch, "blocks_before": int((field.label_grid > 0).sum()),
              "blocks_after": int((label > 0).sum()), "mass_before_kg": field.mass_kg, "mass_after_kg": out.mass_kg,
              "materials_before": before, "materials_after": after, "mass_change_kg": change}
    return out, report


__all__ = ["parse_factor", "resample_field", "coarsen_labels", "refine", "blocks", "pad_to"]
