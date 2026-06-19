"""
Voxel — a shape defined by a sampled SIGNED-DISTANCE GRID.

Where the analytic primitives (Sphere, Box, …) carry their geometry as a closed
formula, a ``Voxel`` carries it as a 3-D grid of signed distances on a regular
cubic lattice — the discretization of a real scanned / CAD mesh. Its
``surface_distance()`` trilinearly interpolates that grid, so it satisfies the
exact same ``Shape`` contract as every other primitive: the renderer that
raymarches an SDF and the integrator that samples ``material_at`` both work on a
``Voxel`` unchanged. That is the whole point of the voxel pivot — real geometry
becomes "just another SDF."

Kernel stays import-clean: this module never imports numpy at module scope. It
*indexes* the grid arrays Deckard builds (numpy ``ndarray`` supports ``g[i,j,k]``
and ``.shape`` with no import here), and the interpolation math is plain floats.
A grid is built/owned by ``deckard/voxelize.py`` (the opt-in numpy/trimesh/scipy
path); the kernel only knows how to read one.

Convention: the grid is centred on ``self.center`` and symmetric — cell (i,j,k)
sits at ``center + (i-(nx-1)/2)·voxel_size`` (and similarly j,k). Signed distance
is negative inside the solid, positive outside (metres). Deckard pads the grid
with a margin of empty cells so the boundary samples are safely positive.
"""

import math

from .shapes import Shape


class Voxel(Shape):
    """A voxelized solid: an SDF grid that behaves like any analytic Shape.

    Args:
        sdf_grid:    a 3-D array (``ndarray``-like, supports ``g[i,j,k]`` and
                     ``.shape``) of SIGNED DISTANCES in metres — negative inside.
        voxel_size_m: edge length of one cubic cell (m).
        center:      placement centre (x,y,z) in m; the grid is centred here.
        volume_m3:   the solid's true volume (m³) — Deckard computes it exactly
                     as ``(sdf<0).sum()·cell³`` with numpy; cheap to pass in.
        label_grid:  optional per-cell material-id grid (same dims) for the merged
                     labeled-field / interface scan; None ⇒ material is carried at
                     the layer level like every other shape.
        source:      provenance note (e.g. "PartNet 44164 @ 2mm, igl winding").
    """

    def __init__(self, sdf_grid, voxel_size_m, center=None, volume_m3=None,
                 label_grid=None, source=""):
        self._g = sdf_grid
        self._nx, self._ny, self._nz = (int(n) for n in sdf_grid.shape)
        self.voxel_size = float(voxel_size_m)
        self._labels = label_grid
        self.source = source
        self.__init_center__(center)
        if volume_m3 is not None:
            self._volume = float(volume_m3)
        else:
            self._volume = self._count_inside() * self.voxel_size ** 3

    # ── extents ──────────────────────────────────────────────────────────
    def _extents(self):
        """Full (x,y,z) span of the grid box in metres."""
        return (self._nx * self.voxel_size,
                self._ny * self.voxel_size,
                self._nz * self.voxel_size)

    def _defining_dims(self):
        return self._extents()

    def volume(self):
        return self._volume

    def bounding_radius(self):
        a, b, c = self._extents()
        return 0.5 * math.sqrt(a * a + b * b + c * c)

    # The analytic moment / area methods below are COARSE box fallbacks over the
    # grid's bounding extent. The voxel Construct computes the EXACT mass / CoM /
    # inertia with numpy directly from the grid, so these are only used if some
    # caller asks a bare Voxel for them (rare). Honest: flagged as approximate.
    def inertia_factor(self, axis='z'):
        a, b, c = self._extents()
        if axis == 'x':
            return (1.0 / 12.0) * (b * b + c * c)
        if axis == 'y':
            return (1.0 / 12.0) * (a * a + c * c)
        return (1.0 / 12.0) * (a * a + b * b)

    def cross_section(self, axis='z'):
        a, b, c = self._extents()
        # bounding-box face × fill fraction (occupied volume / box volume)
        box_vol = a * b * c
        fill = (self._volume / box_vol) if box_vol > 0 else 1.0
        if axis == 'x':
            return b * c * fill
        if axis == 'y':
            return a * c * fill
        return a * b * fill

    def surface_area(self):
        a, b, c = self._extents()
        return 2.0 * (a * b + a * c + b * c)   # coarse box estimate

    # ── the SDF (renderer + physics queries) ─────────────────────────────
    def _trilinear(self, fi, fj, fk):
        """Trilinear interpolation of the SDF grid at fractional indices."""
        g = self._g
        i0 = int(fi); j0 = int(fj); k0 = int(fk)
        i1 = i0 + 1 if i0 + 1 < self._nx else i0
        j1 = j0 + 1 if j0 + 1 < self._ny else j0
        k1 = k0 + 1 if k0 + 1 < self._nz else k0
        di = fi - i0; dj = fj - j0; dk = fk - k0
        c000 = float(g[i0, j0, k0]); c001 = float(g[i0, j0, k1])
        c010 = float(g[i0, j1, k0]); c011 = float(g[i0, j1, k1])
        c100 = float(g[i1, j0, k0]); c101 = float(g[i1, j0, k1])
        c110 = float(g[i1, j1, k0]); c111 = float(g[i1, j1, k1])
        c00 = c000 * (1 - di) + c100 * di
        c01 = c001 * (1 - di) + c101 * di
        c10 = c010 * (1 - di) + c110 * di
        c11 = c011 * (1 - di) + c111 * di
        c0 = c00 * (1 - dj) + c10 * dj
        c1 = c01 * (1 - dj) + c11 * dj
        return c0 * (1 - dk) + c1 * dk

    def surface_distance(self, px, py, pz):
        """Signed distance to the voxelized surface (negative inside).

        Inside the grid box → trilinear interpolation. Outside the box → the
        boundary sample (≥0, since the grid is padded) plus the Euclidean
        distance from the point to the box, so a raymarcher converges cleanly.
        """
        nx, ny, nz = self._nx, self._ny, self._nz
        vs = self.voxel_size
        cx, cy, cz = self.center
        fi = (px - cx) / vs + (nx - 1) * 0.5
        fj = (py - cy) / vs + (ny - 1) * 0.5
        fk = (pz - cz) / vs + (nz - 1) * 0.5
        ci = 0.0 if fi < 0.0 else (nx - 1.0 if fi > nx - 1 else fi)
        cj = 0.0 if fj < 0.0 else (ny - 1.0 if fj > ny - 1 else fj)
        ck = 0.0 if fk < 0.0 else (nz - 1.0 if fk > nz - 1 else fk)
        val = self._trilinear(ci, cj, ck)
        if ci == fi and cj == fj and ck == fk:
            return val
        bx = cx + (ci - (nx - 1) * 0.5) * vs
        by = cy + (cj - (ny - 1) * 0.5) * vs
        bz = cz + (ck - (nz - 1) * 0.5) * vs
        extra = math.sqrt((px - bx) ** 2 + (py - by) ** 2 + (pz - bz) ** 2)
        return (val if val > 0.0 else 0.0) + extra

    def material_at(self, px, py, pz):
        """Per-cell material id at a point, or None if the grid is unlabeled.

        (Single-material voxels carry their material at the layer level, like
        every other shape; only the merged labeled field uses this.)
        """
        if self._labels is None:
            return None
        nx, ny, nz = self._nx, self._ny, self._nz
        vs = self.voxel_size
        cx, cy, cz = self.center
        i = int(round((px - cx) / vs + (nx - 1) * 0.5))
        j = int(round((py - cy) / vs + (ny - 1) * 0.5))
        k = int(round((pz - cz) / vs + (nz - 1) * 0.5))
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return None
        return self._labels[i, j, k]

    def _count_inside(self):
        """Occupied-cell count (fallback when volume isn't supplied)."""
        try:
            import numpy as _np
            return int((_np.asarray(self._g) < 0.0).sum())
        except Exception:
            return 0

    def __repr__(self):
        a, b, c = self._extents()
        return (f"Voxel({self._nx}x{self._ny}x{self._nz} @ "
                f"{self.voxel_size * 1000:.1f}mm, {a:.3g}x{b:.3g}x{c:.3g}m)")
