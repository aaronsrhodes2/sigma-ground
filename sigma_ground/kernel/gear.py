"""Gear — an involute spur gear cross-section swept along z into a 3D solid.

Follows the same pattern as ``Outline`` (kernel/outline.py): a researched 2D
profile — here, an *analytic* involute tooth profile rather than a polyline —
swept by extrusion along the axis. Exposes the same duck-typed ``Shape``
surface (``surface_distance`` / ``volume`` / ``bounding_radius`` /
``inertia_factor`` / ``center``) so the CSG engine (kernel/csg.py) and mass /
inertia machinery work unmodified.

SDF convention (matches shapes.py / csg.py exactly):
    negative = inside the solid
    zero     = on the surface
    positive = outside the solid

Gear axis convention: the gear's rotation axis is the local z-axis, same as
``Cylinder`` (kernel/shapes.py) — face_width is centered at z=0, same as
``Outline``'s extrude mode.

NOT_EXACT: the 2D gear cross-section SDF (``_gear_sdf_2d``) is a
CONSERVATIVE, topologically-correct signed distance field — standard
practice for raymarched gear teeth (union of an angular "tooth wedge" with
a root disk). It is exact along the involute flank and along the addendum/
root circles, but it is NOT exact Euclidean distance in the corner regions
near the tooth tip and the tooth/root transition. That is fine for
raymarching and for the numerical (grid quadrature) volume/inertia
estimates below, but callers wanting exact corner distances should not
rely on this class.

NOT_PHYSICS: the root fillet is a Quilez polynomial smooth-min (reusing
``csg.sdf_smooth_union``) between the tooth wedge and the root disk, with
blend radius ``fillet_coeff * module``. A true hobbed/trochoidal root
fillet (the shape a hobbing cutter actually leaves behind) has no simple
closed form and is NOT what this produces — this is a cosmetic corner
rounding, not a manufacturing-accurate fillet.

tooth_form: only 'involute' (the standard form for power-transmission
spur gears) is implemented. Any other value raises NotImplementedError —
it is never silently treated as involute.
"""
from __future__ import annotations

import math

from .shapes import Shape
from .csg import sdf_smooth_union


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class InvoluteGear(Shape):
    """Solid involute spur gear (axis along z), extruded to face_width.

    Args:
        module: gear module m (metres per tooth of pitch diameter / N).
        teeth: number of teeth N (int, >= 3).
        pressure_angle: pressure angle alpha, radians (e.g. radians(20)).
        addendum_coeff: addendum height as a multiple of module (r_a = r_p + coeff*m).
        dedendum_coeff: dedendum depth as a multiple of module (r_f = r_p - coeff*m).
        fillet_coeff: root fillet blend radius as a multiple of module
            (NOT_PHYSICS — cosmetic corner rounding, see module docstring).
        face_width: axial gear thickness (metres), centered at z=0.
        tooth_form: only 'involute' is supported; anything else raises
            NotImplementedError.
        center: (x, y, z) placement of the gear's geometric center.
        grid_resolution: default NxNxN grid used by volume()/inertia_factor()
            when called with no override (see those methods). Higher = more
            accurate, slower. Default 60.

    Standard radii (FIRST_PRINCIPLES: standard involute gear geometry,
    e.g. Shigley's Mechanical Engineering Design):
        r_p = m*N/2                    pitch radius
        r_b = r_p*cos(alpha)           base radius (involute originates here)
        r_a = r_p + addendum_coeff*m   addendum / tip radius
        r_f = r_p - dedendum_coeff*m   dedendum / root radius
    """

    def __init__(self, module, teeth, pressure_angle,
                 addendum_coeff=1.0, dedendum_coeff=1.25, fillet_coeff=0.35,
                 face_width=1.0, tooth_form='involute', center=None,
                 grid_resolution=60):
        if tooth_form != 'involute':
            raise NotImplementedError(
                f"InvoluteGear only implements tooth_form='involute', "
                f"got {tooth_form!r}. Non-involute tooth forms (cycloidal, "
                f"trochoidal, etc.) are out of scope for this class."
            )

        self.module = float(module)
        self.teeth = int(teeth)
        self.pressure_angle = float(pressure_angle)
        self.addendum_coeff = float(addendum_coeff)
        self.dedendum_coeff = float(dedendum_coeff)
        self.fillet_coeff = float(fillet_coeff)
        self.face_width = float(face_width)
        self.tooth_form = tooth_form
        self.grid_resolution = int(grid_resolution)
        self.__init_center__(center)

        if self.teeth < 3:
            raise ValueError(f"InvoluteGear needs teeth >= 3, got {self.teeth}")
        if self.face_width <= 0.0:
            raise ValueError("InvoluteGear needs a positive face_width")

        m, N, alpha = self.module, self.teeth, self.pressure_angle

        # -- standard radii --------------------------------------------------
        self.r_p = m * N / 2.0
        self.r_b = self.r_p * math.cos(alpha)
        self.r_a = self.r_p + self.addendum_coeff * m
        self.r_f = self.r_p - self.dedendum_coeff * m

        # -- angles -----------------------------------------------------------
        self.psi = math.pi / (2.0 * N)      # half tooth-thickness angle at pitch circle
        self.beta = math.pi / N             # half the angular tooth pitch (tooth+gap period / 2)

        # local-frame reference angle: theta(r_b) in the gear's own folded
        # frame — the involute's t=0 point sits here (see module docstring
        # "local frame" note in the spec this class implements).
        self._theta_rb = self.psi + self._inv(alpha)

        # root fillet blend radius (NOT_PHYSICS, see module docstring)
        self.fillet_radius = self.fillet_coeff * m

        self._quad_cache = {}  # resolution -> (volume, (Ix_sum, Iy_sum, Iz_sum))

    # -- involute geometry helpers --------------------------------------------

    @staticmethod
    def _inv(x):
        """Involute function: inv(x) = tan(x) - x."""
        return math.tan(x) - x

    def _theta(self, rho):
        """Half tooth-thickness angle at radius rho, in the gear's folded frame.

        theta(rho) = psi + inv(alpha) - inv(alpha_rho), alpha_rho = arccos(clamp(r_b/rho)).

        For rho <= r_b the ratio r_b/rho clamps to 1, giving alpha_rho = 0
        and theta(rho) == theta(r_b) — a flat extension used for the sign
        test in _local_surface_distance, not a claim that the involute
        exists below the base circle.
        """
        r_b = self.r_b
        ratio = r_b / rho if rho > 1e-15 else 1.0
        alpha_rho = math.acos(_clamp(ratio, -1.0, 1.0))
        return self.psi + self._inv(self.pressure_angle) - self._inv(alpha_rho)

    def _involute_point(self, t):
        """C(t) = r_b*(cos(t)+t*sin(t), sin(t)-t*cos(t)) — involute curve, local frame."""
        r_b = self.r_b
        cos_t, sin_t = math.cos(t), math.sin(t)
        x = r_b * (cos_t + t * sin_t)
        y = r_b * (sin_t - t * cos_t)
        return x, y

    def _nearest_involute_t(self, rho, phi_local):
        """Closed-form nearest-point roll angle t* on the involute flank.

        CRITICAL clamp 1: rho <= r_b -> t* = 0 (no involute below the base
        circle; the curve starts at t=0 on the base circle).
        CRITICAL clamp 2: t* = max(0, phi_local + arccos(clamp(r_b/rho)))
        — without the max(0, ...) floor, points angularly "before" the
        involute's start get a spurious negative-t "nearest point" that
        isn't actually on the curve (t >= 0 only), producing wrong
        distances for a wide angular window (verified >3 rad error in t
        without this clamp during spec verification).
        """
        if rho <= self.r_b:
            return 0.0
        arc = math.acos(_clamp(self.r_b / rho, -1.0, 1.0))
        return max(0.0, phi_local + arc)

    # -- 2D cross-section SDF --------------------------------------------------

    def _gear_sdf_2d(self, x, y):
        """Signed distance to the 2D gear cross-section (local x, y).

        See module docstring: conservative/topological SDF, exact on the
        flank/addendum/root circles, not exact in the tip/root corners.
        """
        rho = math.hypot(x, y)
        phi = math.atan2(y, x)
        beta = self.beta

        # fold into one tooth+gap period, centered; fold again for L/R symmetry
        phi_folded_signed = ((phi + beta) % (2.0 * beta)) - beta
        phi_folded = abs(phi_folded_signed)

        # Gear frame -> involute frame is a REFLECTION, not a rotation: in
        # its own frame the involute C(t) curls CCW (polar angle +inv(a_rho)
        # going outward), while the gear's flank curls CW away from the
        # tooth centerline (theta(rho) = theta_rb - inv(a_rho), DECREASING
        # going outward). So phi_inv = theta_rb - phi_folded — and the query
        # point must be rebuilt with that SAME reflected angle so point and
        # curve live in one frame (reflection is an isometry, so distances
        # survive). Using the rotation phi_folded - theta_rb here was a real
        # bug: sign classification stayed correct but flank distance
        # MAGNITUDES were off by up to ~1 module-unit (caught by adversarial
        # brute-force verification against an independent dense-polygon
        # ground truth; on-flank points read ~0.6-0.9 instead of ~1e-15).
        phi_inv = self._theta_rb - phi_folded
        t_star = self._nearest_involute_t(rho, phi_inv)
        cx_t, cy_t = self._involute_point(t_star)

        qx, qy = rho * math.cos(phi_inv), rho * math.sin(phi_inv)
        d_flank_mag = math.hypot(qx - cx_t, qy - cy_t)

        theta_rho = self._theta(rho)
        # negative if angularly inside the tooth material at this radius
        d_flank = -d_flank_mag if phi_folded < theta_rho else d_flank_mag

        d_addendum = rho - self.r_a                  # positive = beyond the tip
        d_root_disk = rho - self.r_f                 # positive = outside the root disk (disk's own SDF)

        d_tooth_wedge = max(d_flank, d_addendum)      # intersection: inside tooth AND inside addendum

        # Union of the tooth wedge with the root disk. NOTE: both operands
        # already use the positive-outside convention (same as
        # csg.sdf_union's contract), so this is a plain SDF union — see
        # the root-fillet smoothing below, which replaces the plain min
        # with csg.sdf_smooth_union between these same two operands.
        # (An earlier draft of this spec negated d_root_disk here; that
        # sign was checked numerically against the required self-test
        # 2b — "far outside the addendum, surface_distance should be
        # positive" — and fails it: negating d_root_disk makes every far
        # point register as deeply *inside* the gear, since -d_root_disk
        # runs to -infinity as rho grows. The non-negated union is the
        # one that reproduces the correct outside/inside sign everywhere
        # it was checked, including at the gear center and far outside
        # the tip, so that is what this implementation uses.)
        d_gear_2d = sdf_smooth_union(d_tooth_wedge, d_root_disk, self.fillet_radius)

        return d_gear_2d

    # -- Shape interface ---------------------------------------------------

    def _local_surface_distance(self, x, y, z):
        """surface_distance, already translated into the shape's local frame."""
        d_2d = self._gear_sdf_2d(x, y)
        d_extrude = abs(z) - self.face_width / 2.0
        # Same extrusion idiom as Outline's extrude mode (kernel/outline.py):
        # intersect the 2D profile with the [-hw, hw] z-slab.
        if d_2d <= 0.0 and d_extrude <= 0.0:
            return max(d_2d, d_extrude)
        if d_2d > 0.0 and d_extrude > 0.0:
            return math.hypot(d_2d, d_extrude)
        return max(d_2d, d_extrude)

    def surface_distance(self, px, py, pz):
        """Signed distance from a point (global frame) to the gear surface.

        Positive = outside, negative = inside, matching Shape's convention
        (see csg.py's docstring).
        """
        cx, cy, cz = self.center
        return self._local_surface_distance(px - cx, py - cy, pz - cz)

    def _defining_dims(self):
        return (self.r_a, self.r_a, self.face_width)

    def bounding_radius(self):
        # Half-diagonal of the [-r_a, r_a] x [-r_a, r_a] x [-hw, hw] bounding
        # box — same convention as Cylinder.bounding_radius().
        return math.hypot(self.r_a, self.face_width / 2.0)

    # -- volume / inertia: numerical grid quadrature ------------------------
    #
    # No closed-form Green's-theorem contour integral is attempted here
    # (out of scope, too error-prone for a conservative/non-exact SDF).
    # Instead we classify a 3D grid of cell centers as inside/outside using
    # this class's own surface_distance() sign and accumulate volume and
    # the moment-of-inertia integrals directly. See the self-test for the
    # required convergence check (40^3 vs 80^3 grids, <2% difference).

    def _grid_quadrature(self, resolution):
        """Numerically integrate volume and I/m sums over an NxNxN grid.

        Returns (volume, (Ix_sum, Iy_sum, Iz_sum)) where Iaxis_sum is the
        raw sum of perp_r^2 * cell_volume over inside cells (divide by
        volume to get inertia_factor(axis), matching Shape's I/m contract).
        """
        cached = self._quad_cache.get(resolution)
        if cached is not None:
            return cached

        r_a = self.r_a
        hw = self.face_width / 2.0
        n = resolution

        dx = 2.0 * r_a / n
        dy = 2.0 * r_a / n
        dz = 2.0 * hw / n
        cell_vol = dx * dy * dz

        volume = 0.0
        ix_sum = 0.0  # sum (y^2+z^2) * cell_vol  -> inertia about x
        iy_sum = 0.0  # sum (x^2+z^2) * cell_vol  -> inertia about y
        iz_sum = 0.0  # sum (x^2+y^2) * cell_vol  -> inertia about z

        for i in range(n):
            x = -r_a + (i + 0.5) * dx
            x2 = x * x
            for j in range(n):
                y = -r_a + (j + 0.5) * dy
                y2 = y * y
                # cheap 2D reject before paying for the z loop
                if self._gear_sdf_2d(x, y) > math.hypot(dx, dy):
                    continue
                for k in range(n):
                    z = -hw + (k + 0.5) * dz
                    if self._local_surface_distance(x, y, z) < 0.0:
                        z2 = z * z
                        volume += cell_vol
                        ix_sum += (y2 + z2) * cell_vol
                        iy_sum += (x2 + z2) * cell_vol
                        iz_sum += (x2 + y2) * cell_vol

        result = (volume, (ix_sum, iy_sum, iz_sum))
        self._quad_cache[resolution] = result
        return result

    def volume(self, resolution=None):
        """Volume in m^3, via numerical grid quadrature (see class docstring).

        Args:
            resolution: optional override of the NxNxN grid resolution used
                for this call (does not change self.grid_resolution). Used
                by the convergence self-test; normal callers should omit it.
        """
        res = self.grid_resolution if resolution is None else int(resolution)
        vol, _ = self._grid_quadrature(res)
        return vol

    def inertia_factor(self, axis='z', resolution=None):
        """I/m (m^2) about the given axis, via numerical grid quadrature.

        Args:
            axis: 'x', 'y', or 'z'.
            resolution: optional override, see volume().
        """
        res = self.grid_resolution if resolution is None else int(resolution)
        vol, (ix_sum, iy_sum, iz_sum) = self._grid_quadrature(res)
        if vol <= 0.0:
            return 0.0
        sums = {'x': ix_sum, 'y': iy_sum, 'z': iz_sum}
        if axis not in sums:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        return sums[axis] / vol

    def _analytic_area_2d(self, n_steps=2000):
        """Independent 1D radial sanity check for the 2D cross-section area.

        area ~= N * (tooth wedge area) + root disk area, where the tooth
        wedge area is found by integrating 2*theta(rho)*rho drho from r_f
        to r_a (the angular width of one tooth at each radius, integrated
        radially) — NOT via the SDF at all, so it's a genuinely independent
        check on the SDF-based volume() grid quadrature. Not expected to
        match exactly since the SDF construction is conservative, not
        exact (see module docstring); ~10% agreement is the bar.
        """
        r_f, r_a = self.r_f, self.r_a
        if r_a <= r_f:
            return math.pi * r_a * r_a
        dr = (r_a - r_f) / n_steps
        tooth_area = 0.0
        for i in range(n_steps):
            rho = r_f + (i + 0.5) * dr
            tooth_area += 2.0 * self._theta(rho) * rho * dr
        root_disk_area = math.pi * r_f * r_f
        return self.teeth * tooth_area + root_disk_area

    # -- remaining Shape interface (approximate; not the focus of this class) --

    def cross_section(self, axis='z'):
        if axis == 'z':
            return self._analytic_area_2d()
        # Side-on projection: approximate as the bounding rectangle, same
        # coarse convention Cylinder uses for its transverse cross_section.
        return 2.0 * self.r_a * self.face_width

    def surface_area(self):
        # Coarse approximation: two end-face areas (analytic 2D area) plus
        # a lateral band at the pitch radius. Not exact — a true involute
        # gear's lateral surface area requires integrating flank arc length
        # per tooth; out of scope here (this class's numerical rigor is
        # spent on volume/inertia, per the spec this class implements).
        area_2d = self._analytic_area_2d()
        lateral = 2.0 * math.pi * self.r_p * self.face_width
        return 2.0 * area_2d + lateral

    def __repr__(self):
        return (f"InvoluteGear(m={self.module:.4g}, N={self.teeth}, "
                f"alpha={math.degrees(self.pressure_angle):.3g}deg, "
                f"r_a={self.r_a:.4g}m, r_f={self.r_f:.4g}m, "
                f"face_width={self.face_width:.4g}m)")


__all__ = ["InvoluteGear"]
