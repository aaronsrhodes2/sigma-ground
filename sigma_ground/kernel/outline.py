"""Outline — a researched 2D profile swept into a 3D solid (extrude or revolve).

The analytic primitives (sphere/cylinder/box/cone/torus/ellipsoid) cannot express
organic, human-scale shapes — a feather vane, a leaf, a blade, a bottle
silhouette. Outline closes that gap WITHOUT a mesh: a 2D polyline (a *researched
outline* — e.g. a Quick Draw medoid stroke, a fact, not a raster image) is swept
by a primitive op:

  * ``extrude`` — a closed polygon in the part's x-y plane, given a THICKNESS
    along z (a flat sheet with a complex rim: feather, leaf, blade, card).
  * ``revolve`` — a half-profile ``r(z)`` (points ordered by z) spun about the
    z axis (a solid of revolution: bottle, vase, cup).

It exposes the same duck-typed surface (``surface_distance`` / ``volume`` /
``bounding_radius`` / ``center``) the geometry kernel and Deckard's SDF
integrator already use, so mass / centre-of-mass / inertia / interfaces all flow
through the existing, validated machinery. Outlines AUGMENT the analytic kit;
they do not replace it.

Keep researched outlines simplified (few points): ``surface_distance`` is O(points)
and the integrator samples it across a grid.
"""
from __future__ import annotations

import math

from .shapes import Shape


def _shoelace_area(poly):
    """Signed area of a 2D polygon (CCW positive)."""
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def _seg_distance(px, py, ax, ay, bx, by):
    """Distance from point (px, py) to the segment (ax, ay)-(bx, by)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _polygon_signed_distance(px, py, poly):
    """Signed distance from (px, py) to a closed polygon — negative INSIDE.

    Magnitude = distance to the nearest edge; sign by even-odd ray casting.
    """
    inside = False
    min_d = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        d = _seg_distance(px, py, ax, ay, bx, by)
        if d < min_d:
            min_d = d
        if (ay > py) != (by > py):
            xint = ax + (py - ay) / (by - ay) * (bx - ax)
            if px < xint:
                inside = not inside
    return -min_d if inside else min_d


def _section_signed_distance(z, rho, section):
    """Signed distance for a revolve half-section in (z, rho) — negative INSIDE.

    Like ``_polygon_signed_distance`` but the rotation axis (an edge with both
    endpoints at rho=0) is NOT a real surface, so it's excluded from the distance
    (a point on the axis is interior to the revolved solid) while still counting
    for the even-odd inside test.
    """
    inside = False
    min_d = float("inf")
    n = len(section)
    for i in range(n):
        z0, r0 = section[i]
        z1, r1 = section[(i + 1) % n]
        if (r0 > rho) != (r1 > rho):
            zint = z0 + (rho - r0) / (r1 - r0) * (z1 - z0)
            if z < zint:
                inside = not inside
        if abs(r0) < 1e-12 and abs(r1) < 1e-12:
            continue                                  # the rotation axis is not a surface
        d = _seg_distance(z, rho, z0, r0, z1, r1)
        if d < min_d:
            min_d = d
    return -min_d if inside else min_d


class Outline(Shape):
    """A 2D outline swept into a 3D solid. See the module docstring."""

    def __init__(self, profile, mode="extrude", thickness=None, center=None):
        pts = [(float(u), float(v)) for u, v in profile]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]                                   # drop a closing duplicate
        self.profile = pts
        self.mode = mode
        self.__init_center__(center)
        if mode == "extrude":
            if len(pts) < 3:
                raise ValueError("extrude Outline needs >=3 profile points")
            if not thickness or thickness <= 0.0:
                raise ValueError("extrude Outline needs a positive thickness")
            self.thickness = float(thickness)
            self._area = abs(_shoelace_area(pts))
            self._rmax = max(math.hypot(u, v) for u, v in pts)
        elif mode == "revolve":
            if len(pts) < 2:
                raise ValueError("revolve Outline needs >=2 profile (z, r) points")
            zs = [z for z, _ in pts]
            self.thickness = 0.0
            self._zmin, self._zmax = min(zs), max(zs)
            self._rmax = max(abs(r) for _, r in pts)
            # closed (z, r) half-section: the profile, returned along the axis r=0
            self._section = pts + [(self._zmax, 0.0), (self._zmin, 0.0)]
        else:
            raise ValueError(f"Outline mode must be 'extrude' or 'revolve', got {mode!r}")

    # -- the duck-typed surface Deckard + the kernel integrator use -----------
    def volume(self):
        if self.mode == "extrude":
            return self._area * self.thickness
        # solid of revolution: sum of cone-frustum volumes (exact for linear r(z))
        v = 0.0
        for (z0, r0), (z1, r1) in zip(self.profile, self.profile[1:]):
            v += (math.pi / 3.0) * (r0 * r0 + r0 * r1 + r1 * r1) * abs(z1 - z0)
        return v

    def surface_distance(self, px, py, pz):
        cx, cy, cz = self.center
        if self.mode == "extrude":
            d2 = _polygon_signed_distance(px - cx, py - cy, self.profile)
            dz = abs(pz - cz) - 0.5 * self.thickness
            if d2 <= 0.0 and dz <= 0.0:
                return max(d2, dz)
            if d2 > 0.0 and dz > 0.0:
                return math.hypot(d2, dz)
            return max(d2, dz)
        # revolve: signed distance in the (z, rho) half-section (axis = interior)
        rho = math.hypot(px - cx, py - cy)
        return _section_signed_distance(pz - cz, rho, self._section)

    def bounding_radius(self):
        if self.mode == "extrude":
            return math.hypot(self._rmax, 0.5 * self.thickness)
        return math.hypot(self._rmax, 0.5 * (self._zmax - self._zmin))

    # -- kernel-analysis surface (approximate; Deckard uses the integrator) ---
    def _defining_dims(self):
        span = self.thickness if self.mode == "extrude" else (self._zmax - self._zmin)
        return (self._rmax, self._rmax, span)

    def inertia_factor(self, axis="z"):
        r = self.bounding_radius()                           # bbox approximation
        return 0.4 * r * r

    def cross_section(self, axis="z"):
        if self.mode == "extrude":
            return self._area if axis == "z" else 2.0 * self._rmax * self.thickness
        return math.pi * self._rmax * self._rmax

    def surface_area(self):
        if self.mode == "extrude":
            per = 0.0
            n = len(self.profile)
            for i in range(n):
                ax, ay = self.profile[i]
                bx, by = self.profile[(i + 1) % n]
                per += math.hypot(bx - ax, by - ay)
            return 2.0 * self._area + per * self.thickness
        return 2.0 * math.pi * self._rmax * (self._zmax - self._zmin)

    def __repr__(self):
        return f"Outline({self.mode}, {len(self.profile)} pts, r~{self._rmax:.4g}m)"


__all__ = ["Outline"]
