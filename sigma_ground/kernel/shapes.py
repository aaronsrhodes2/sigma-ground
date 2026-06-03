"""
Geometric primitives — shape defines inertia.

Five primary smooth shapes that bypass polygonal rendering:

  Sphere(radius)              — 1 value, doubly curved (+)
  Torus(major_R, minor_r)     — 2 values, mixed curvature (+/-)
  Cone(radius, height)        — 2 values, developable surface
  Cylinder(radius, height)    — 2 values, developable surface
  Box(x, y, z)                — 3 values, flat faces

Degenerate cases (dimensionally compromised):

  Ring(R)       = Torus(R, 0)       — 1D curve, no volume
  Plane(w, d)   = Box(w, d, 0)      — 2D surface, no volume
  HollowSphere  = thin shell        — 2D surface, no volume

A dimension is "compromised" when it falls at or below the Planck
length floor (1.616e-35 m). Shapes classify themselves:

  3D  solid   — all defining dims real → can hold matter
  2D  surface — one dim compromised → boundary only
  1D  line    — two dims compromised → topological skeleton
  0D  point   — all dims compromised → degenerate

Only 3D shapes define the extents of matter.

Structure = additive list of (shape, center, material).
  Hollow regions use material='air'.
  Volume efficiency = used_volume / target_volume.
  Resolution = shapes_per_mole (adjustable packing granularity).

Future: Möbius strip (non-orientable surface).

Compatible with matter-shaper Sigma Signatures:
  .shape.json layers map 1:1 to these primitives.

Derivation chains:

  1. Volume — exact analytical formulae for each primitive.
     FIRST_PRINCIPLES: Archimedes (sphere), Pappus (torus),
     Cavalieri (cylinder, cone).

  2. Moment of inertia — I/m factor for each shape about each axis.
     FIRST_PRINCIPLES: ∫ r² dm over uniform density body.
     Euler (1750), Steiner (1840).

  3. Cross-sectional area — projected area perpendicular to a given axis.
     Needed for drag, radiation pressure, impact calculations.

  σ-dependence:
    Shapes are σ-independent — pure geometry.
    Mass scales with σ, but I/m (the inertia factor) does not.
    Volume, area, bounding radius — all σ-independent.
"""

import math

# Planck length — the dimensional floor.
# Any spatial dimension at or below this is physically meaningless.
_L_PLANCK = 1.616255e-35  # m


def _is_real(dim):
    """True if a spatial dimension is physically real (above Planck floor)."""
    return abs(dim) > _L_PLANCK


class Shape:
    """Base class for geometric primitives.

    Every shape carries a geometric center (cx, cy, cz) for placement
    within a structure. Default is the origin (0, 0, 0).

    Subclasses must implement:
      volume(), inertia_factor(), bounding_radius(),
      cross_section(), surface_area(), _defining_dims()
    """

    def __init_center__(self, center=None):
        """Set geometric center. Call from subclass __init__."""
        if center is None:
            self.center = (0.0, 0.0, 0.0)
        else:
            self.center = (float(center[0]), float(center[1]), float(center[2]))

    def _defining_dims(self):
        """Return tuple of this shape's independent spatial dimensions.

        Used by dimensionality to count how many are physically real.
        Subclasses must override.
        """
        raise NotImplementedError

    @property
    def dimensionality(self):
        """Effective spatial dimensionality (0, 1, 2, or 3).

        Counts how many defining dimensions are above the Planck floor.
        A shape with all dims real is 3D (solid, can hold matter).
        Fewer real dims → surface (2D), line (1D), or point (0D).
        """
        return sum(1 for d in self._defining_dims() if _is_real(d))

    @property
    def is_volumetric(self):
        """True if this shape can hold matter (dimensionality == 3)."""
        return self.dimensionality == 3

    def volume(self):
        """Volume in m³."""
        raise NotImplementedError

    def inertia_factor(self, axis='z'):
        """I/m ratio (m²) — the purely geometric part of moment of inertia.

        I = mass × inertia_factor(axis)

        This is σ-independent because mass cancels in the ratio I/(mr²)
        but the dimensions don't cancel, so we return I/m in m² units.

        Args:
            axis: 'x', 'y', or 'z' — rotation axis.

        Returns:
            I/m in m².
        """
        raise NotImplementedError

    def bounding_radius(self):
        """Bounding sphere radius in m — for broad-phase collision."""
        raise NotImplementedError

    def cross_section(self, axis='z'):
        """Projected cross-sectional area perpendicular to axis (m²).

        Used for drag force F_d = ½ρCdAv² and radiation pressure.

        Args:
            axis: 'x', 'y', or 'z' — direction of motion.

        Returns:
            Area in m².
        """
        raise NotImplementedError

    def surface_area(self):
        """Total surface area in m² (for thermal radiation, surface energy)."""
        raise NotImplementedError

    def surface_distance(self, px, py, pz):
        """Minimum distance from point (px, py, pz) to this shape's surface.

        The point is in the shape's local frame (relative to self.center).
        Returns 0.0 if the point is exactly on the surface.
        Positive = outside, negative = inside (signed distance).

        Used by boundary_agreement to measure how well a set of primitives
        approximates a target shape's outer boundary.

        Subclasses must override for accurate boundary testing.
        Default falls back to bounding sphere (coarse).
        """
        cx, cy, cz = self.center
        dx, dy, dz = px - cx, py - cy, pz - cz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dist - self.bounding_radius()

    @classmethod
    def from_sigma_signature(cls, layer):
        """Create a Shape from a Sigma Signature .shape.json layer.

        Args:
            layer: dict with 'type', 'radii'/'dimensions', etc.

        Returns:
            Shape instance.
        """
        shape_type = layer.get('type', 'ellipsoid')

        if shape_type == 'sphere':
            radii = layer.get('radii', [1.0, 1.0, 1.0])
            return Sphere(radii[0])

        elif shape_type == 'ellipsoid':
            radii = layer.get('radii', [1.0, 1.0, 1.0])
            # If all radii equal, it's a sphere
            if abs(radii[0] - radii[1]) < 1e-10 and abs(radii[1] - radii[2]) < 1e-10:
                return Sphere(radii[0])
            return Ellipsoid(radii[0], radii[1], radii[2])

        elif shape_type == 'cylinder':
            r = layer.get('radius_m', layer.get('radii', [1.0])[0])
            h = layer.get('height_m', 1.0)
            return Cylinder(r, h)

        elif shape_type == 'box':
            dims = layer.get('dimensions', {})
            return Box(dims.get('x_m', 1.0),
                       dims.get('y_m', 1.0),
                       dims.get('z_m', 1.0))

        elif shape_type == 'cone':
            r = layer.get('radius_m', layer.get('radii', [1.0])[0])
            h = layer.get('height_m', 1.0)
            return Cone(r, h)

        elif shape_type == 'torus':
            R = layer.get('major_radius_m', 1.0)
            r = layer.get('minor_radius_m', 0.25)
            return Torus(R, r)

        elif shape_type == 'plane':
            w = layer.get('width_m', 1.0)
            d = layer.get('depth_m', 1.0)
            return Box(w, d, 0.0)

        elif shape_type == 'ring':
            r = layer.get('radius_m', layer.get('radii', [1.0])[0])
            return Torus(r, 0.0)

        else:
            raise ValueError(f"Unknown shape type '{shape_type}'")


class Sphere(Shape):
    """Solid sphere: the simplest 3D body.

    I = ⅖mr² about any axis (symmetry).
    FIRST_PRINCIPLES: ∫ r² dm with uniform ρ, spherical coordinates.
    """

    def __init__(self, radius, center=None):
        self.radius = float(radius)
        self.__init_center__(center)

    def _defining_dims(self):
        return (self.radius, self.radius, self.radius)

    def volume(self):
        return (4.0 / 3.0) * math.pi * self.radius ** 3

    def inertia_factor(self, axis='z'):
        # Sphere is symmetric — same about any axis
        return (2.0 / 5.0) * self.radius ** 2

    def bounding_radius(self):
        return self.radius

    def cross_section(self, axis='z'):
        # Circle: πr²
        return math.pi * self.radius ** 2

    def surface_area(self):
        return 4.0 * math.pi * self.radius ** 2

    def surface_distance(self, px, py, pz):
        cx, cy, cz = self.center
        dx, dy, dz = px - cx, py - cy, pz - cz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dist - self.radius

    def __repr__(self):
        return f"Sphere(r={self.radius:.4g}m)"


class HollowSphere(Shape):
    """Thin spherical shell — 2D surface, no volume.

    I = ⅔mr² about any axis.
    FIRST_PRINCIPLES: surface integral of r² dm over shell.

    Dimensionally compromised: surface has extent in 2 angular dims
    but zero radial thickness. Cannot hold matter volume.
    """

    def __init__(self, radius, center=None):
        self.radius = float(radius)
        self.__init_center__(center)

    def _defining_dims(self):
        # Shell: two angular extents (real) but zero thickness
        return (self.radius, self.radius, 0.0)

    def volume(self):
        return 0.0

    def inertia_factor(self, axis='z'):
        return (2.0 / 3.0) * self.radius ** 2

    def bounding_radius(self):
        return self.radius

    def cross_section(self, axis='z'):
        return math.pi * self.radius ** 2

    def surface_area(self):
        return 4.0 * math.pi * self.radius ** 2

    def surface_distance(self, px, py, pz):
        cx, cy, cz = self.center
        dx, dy, dz = px - cx, py - cy, pz - cz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dist - self.radius

    def __repr__(self):
        return f"HollowSphere(r={self.radius:.4g}m)"


class Cylinder(Shape):
    """Solid cylinder (axis along z by default).

    I_axial = ½mr² (rotation about cylinder axis)
    I_transverse = ¹⁄₁₂mh² + ¼mr² (rotation perpendicular to axis)

    FIRST_PRINCIPLES: volume integral in cylindrical coordinates.
    """

    def __init__(self, radius, height, center=None):
        self.radius = float(radius)
        self.height = float(height)
        self.__init_center__(center)

    def _defining_dims(self):
        return (self.radius, self.radius, self.height)

    def volume(self):
        return math.pi * self.radius ** 2 * self.height

    def inertia_factor(self, axis='z'):
        if axis == 'z':
            # Axial: I/m = ½r²
            return 0.5 * self.radius ** 2
        else:
            # Transverse (x or y): I/m = ¹⁄₁₂h² + ¼r²
            return (1.0 / 12.0) * self.height ** 2 + 0.25 * self.radius ** 2

    def bounding_radius(self):
        # Half-diagonal of bounding box
        return math.sqrt(self.radius ** 2 + (self.height / 2.0) ** 2)

    def cross_section(self, axis='z'):
        if axis == 'z':
            # Looking down the axis: circle
            return math.pi * self.radius ** 2
        else:
            # Looking from the side: rectangle
            return 2.0 * self.radius * self.height

    def surface_area(self):
        # 2 circles + lateral
        return 2.0 * math.pi * self.radius ** 2 + \
               2.0 * math.pi * self.radius * self.height

    def surface_distance(self, px, py, pz):
        """Signed distance to cylinder surface (axis along z)."""
        cx, cy, cz = self.center
        dx, dy = px - cx, py - cy
        dz = pz - cz
        # Radial distance from axis
        rho = math.sqrt(dx * dx + dy * dy)
        # Distance to lateral surface (radial)
        d_radial = rho - self.radius
        # Distance to end caps (axial)
        half_h = self.height / 2.0
        d_axial = abs(dz) - half_h
        # Signed distance: negative inside, positive outside
        if d_radial <= 0 and d_axial <= 0:
            # Inside: distance to nearest surface (negative)
            return max(d_radial, d_axial)
        elif d_radial > 0 and d_axial > 0:
            # Outside both: distance to edge
            return math.sqrt(d_radial ** 2 + d_axial ** 2)
        else:
            # Outside one, inside the other
            return max(d_radial, d_axial)

    def __repr__(self):
        return f"Cylinder(r={self.radius:.4g}m, h={self.height:.4g}m)"


class Box(Shape):
    """Rectangular parallelepiped (cuboid).

    I about z-axis = ¹⁄₁₂m(x² + y²)
    FIRST_PRINCIPLES: triple integral of (x² + y²) dm.

    Degenerate case: Box(w, d, 0) is a Plane (2D thin plate).
    """

    def __init__(self, x, y, z, center=None):
        """Args: x, y, z are full dimensions (not half-widths) in metres."""
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.__init_center__(center)

    def _defining_dims(self):
        return (self.x, self.y, self.z)

    def volume(self):
        return self.x * self.y * self.z

    def inertia_factor(self, axis='z'):
        if axis == 'x':
            return (1.0 / 12.0) * (self.y ** 2 + self.z ** 2)
        elif axis == 'y':
            return (1.0 / 12.0) * (self.x ** 2 + self.z ** 2)
        else:
            return (1.0 / 12.0) * (self.x ** 2 + self.y ** 2)

    def bounding_radius(self):
        return 0.5 * math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def cross_section(self, axis='z'):
        if axis == 'x':
            return self.y * self.z
        elif axis == 'y':
            return self.x * self.z
        else:
            return self.x * self.y

    def surface_area(self):
        return 2.0 * (self.x * self.y + self.x * self.z + self.y * self.z)

    def surface_distance(self, px, py, pz):
        """Signed distance to box surface (centered at self.center)."""
        cx, cy, cz = self.center
        # Distance from center in each axis
        dx = abs(px - cx) - self.x / 2.0
        dy = abs(py - cy) - self.y / 2.0
        dz = abs(pz - cz) - self.z / 2.0
        # Signed distance
        if dx <= 0 and dy <= 0 and dz <= 0:
            # Inside: distance to nearest face (negative)
            return max(dx, dy, dz)
        else:
            # Outside: Euclidean distance to nearest corner/edge/face
            return math.sqrt(max(dx, 0) ** 2 + max(dy, 0) ** 2 + max(dz, 0) ** 2)

    def __repr__(self):
        if not _is_real(self.z):
            return f"Plane({self.x:.4g}x{self.y:.4g}m)"
        return f"Box({self.x:.4g}x{self.y:.4g}x{self.z:.4g}m)"


class Ellipsoid(Shape):
    """Solid ellipsoid with semi-axes (rx, ry, rz).

    I about z-axis = ⅕m(rx² + ry²)
    FIRST_PRINCIPLES: volume integral in ellipsoidal coordinates.

    A sphere is a special case where rx = ry = rz.
    """

    def __init__(self, rx, ry, rz, center=None):
        """Args: rx, ry, rz are semi-axis lengths in metres."""
        self.rx = float(rx)
        self.ry = float(ry)
        self.rz = float(rz)
        self.__init_center__(center)

    def _defining_dims(self):
        return (self.rx, self.ry, self.rz)

    def volume(self):
        return (4.0 / 3.0) * math.pi * self.rx * self.ry * self.rz

    def inertia_factor(self, axis='z'):
        if axis == 'x':
            return (1.0 / 5.0) * (self.ry ** 2 + self.rz ** 2)
        elif axis == 'y':
            return (1.0 / 5.0) * (self.rx ** 2 + self.rz ** 2)
        else:
            return (1.0 / 5.0) * (self.rx ** 2 + self.ry ** 2)

    def bounding_radius(self):
        return max(self.rx, self.ry, self.rz)

    def cross_section(self, axis='z'):
        # Projected ellipse area
        if axis == 'x':
            return math.pi * self.ry * self.rz
        elif axis == 'y':
            return math.pi * self.rx * self.rz
        else:
            return math.pi * self.rx * self.ry

    def surface_area(self):
        # Knud Thomsen approximation (±1.061% max error)
        p = 1.6075
        ap = self.rx ** p
        bp = self.ry ** p
        cp = self.rz ** p
        return 4.0 * math.pi * ((ap * bp + ap * cp + bp * cp) / 3.0) ** (1.0 / p)

    def surface_distance(self, px, py, pz):
        """Approximate signed distance to ellipsoid surface.

        Exact SDF for an ellipsoid requires solving a quartic.
        This uses the scaled-sphere approximation: transform to
        unit sphere space, measure, scale back. Accurate near surface.
        """
        cx, cy, cz = self.center
        # Normalize to unit sphere
        nx = (px - cx) / self.rx if self.rx > 0 else 0
        ny = (py - cy) / self.ry if self.ry > 0 else 0
        nz = (pz - cz) / self.rz if self.rz > 0 else 0
        dist_norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        # Scale back by average radius for distance estimate
        avg_r = (self.rx + self.ry + self.rz) / 3.0
        return (dist_norm - 1.0) * avg_r

    def __repr__(self):
        return f"Ellipsoid({self.rx:.4g}x{self.ry:.4g}x{self.rz:.4g}m)"


class Cone(Shape):
    """Solid right circular cone (apex at top, base at bottom).

    I_axial = 3/10 mr² (about cone axis)
    I_transverse = 3/20 mr² + 3/80 mh²  (about base center)

    FIRST_PRINCIPLES: volume integral in cylindrical coordinates
    with linearly tapering radius.
    """

    def __init__(self, radius, height, center=None):
        self.radius = float(radius)
        self.height = float(height)
        self.__init_center__(center)

    def _defining_dims(self):
        return (self.radius, self.radius, self.height)

    def volume(self):
        return (1.0 / 3.0) * math.pi * self.radius ** 2 * self.height

    def inertia_factor(self, axis='z'):
        if axis == 'z':
            # About cone axis: I/m = 3/10 r²
            return (3.0 / 10.0) * self.radius ** 2
        else:
            # About transverse axis through center of mass:
            # CM is at h/4 from base
            # I/m = 3/20 r² + 3/80 h²  (about apex-to-base axis CM)
            return (3.0 / 20.0) * self.radius ** 2 + \
                   (3.0 / 80.0) * self.height ** 2

    def bounding_radius(self):
        # From center of mass (h/4 from base) to farthest point
        cm_to_apex = 3.0 * self.height / 4.0
        cm_to_base_edge = math.sqrt(self.radius ** 2 + (self.height / 4.0) ** 2)
        return max(cm_to_apex, cm_to_base_edge)

    def cross_section(self, axis='z'):
        if axis == 'z':
            # Looking down: circle
            return math.pi * self.radius ** 2
        else:
            # Looking from side: triangle
            return 0.5 * 2.0 * self.radius * self.height

    def surface_area(self):
        slant = math.sqrt(self.radius ** 2 + self.height ** 2)
        return math.pi * self.radius * slant + math.pi * self.radius ** 2

    def surface_distance(self, px, py, pz):
        """Signed distance to cone surface (apex at top, base at z=0)."""
        cx, cy, cz = self.center
        # Cone center of mass is at h/4 from base, but we place base at cz - h/4
        # and apex at cz + 3h/4. Simpler: treat base at z=0, apex at z=h locally.
        base_z = cz - self.height / 4.0
        dx, dy = px - cx, py - cy
        local_z = pz - base_z
        rho = math.sqrt(dx * dx + dy * dy)
        # Cone radius at height z: r(z) = radius * (1 - z/height) for z in [0, h]
        if local_z < 0:
            # Below base
            d_base = -local_z
            d_radial = rho - self.radius
            if d_radial <= 0:
                return d_base
            return math.sqrt(d_base ** 2 + d_radial ** 2)
        elif local_z > self.height:
            # Above apex
            return math.sqrt(rho ** 2 + (local_z - self.height) ** 2)
        else:
            # Alongside cone
            r_at_z = self.radius * (1.0 - local_z / self.height)
            # Distance to cone lateral surface along the normal
            slant = math.sqrt(self.radius ** 2 + self.height ** 2)
            # Signed distance: negative inside, positive outside
            d_lateral = (rho - r_at_z) * self.height / slant
            d_base = -local_z  # negative (inside) if above base
            d_top = local_z - self.height  # negative if below apex
            if rho <= r_at_z:
                # Inside cone laterally
                return max(d_lateral, d_base)
            else:
                return d_lateral

    def __repr__(self):
        return f"Cone(r={self.radius:.4g}m, h={self.height:.4g}m)"


class Torus(Shape):
    """Solid torus (donut) — mixed curvature surface.

    Defined by major_radius R (center of hole to center of tube)
    and minor_radius r (tube cross-section radius).

    Volume: V = 2π²Rr²  (Pappus' centroid theorem, 4th century AD)
    Surface area: A = 4π²Rr  (Pappus)

    FIRST_PRINCIPLES: Pappus' centroid theorem — volume of a solid of
    revolution = 2π × distance_of_centroid × area_of_cross_section.
    Cross-section is a circle of area πr², centroid at distance R.

    Moment of inertia (exact, from volume integral in toroidal coords):
      I_z (about symmetry axis) = m(R² + ¾r²)
      I_x = I_y (about a diameter) = m(½R² + ⅝r²)

    Degenerate case: Torus(R, 0) is a Ring (1D hoop).
      I_z = mR², I_x = ½mR² — all mass at distance R from axis.
    """

    def __init__(self, major_radius, minor_radius, center=None):
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.__init_center__(center)

    def _defining_dims(self):
        # Major radius gives the curve, minor radius gives the tube thickness.
        # A torus has two independent spatial extents: the ring (1D) and
        # the tube cross-section (2D). All three dims real → 3D solid.
        return (self.major_radius, self.minor_radius, self.minor_radius)

    def volume(self):
        R, r = self.major_radius, self.minor_radius
        return 2.0 * math.pi ** 2 * R * r ** 2

    def inertia_factor(self, axis='z'):
        R, r = self.major_radius, self.minor_radius
        if axis == 'z':
            # About symmetry axis: I/m = R² + ¾r²
            # At r=0 (Ring): I/m = R² ✓
            return R ** 2 + 0.75 * r ** 2
        else:
            # About a diameter: I/m = ½R² + ⅝r²
            # At r=0 (Ring): I/m = ½R² ✓
            return 0.5 * R ** 2 + 0.625 * r ** 2

    def bounding_radius(self):
        return self.major_radius + self.minor_radius

    def cross_section(self, axis='z'):
        R, r = self.major_radius, self.minor_radius
        if axis == 'z':
            # Looking down: annulus (outer circle minus inner circle)
            return math.pi * ((R + r) ** 2 - (R - r) ** 2)
        else:
            # Looking from side: two circles of radius r, separated by 2R
            # Approximate as bounding rectangle
            return 2.0 * r * (2.0 * (R + r))

    def surface_area(self):
        R, r = self.major_radius, self.minor_radius
        return 4.0 * math.pi ** 2 * R * r

    def surface_distance(self, px, py, pz):
        """Signed distance to torus surface (symmetry axis along z)."""
        cx, cy, cz = self.center
        dx, dy, dz = px - cx, py - cy, pz - cz
        R, r = self.major_radius, self.minor_radius
        # Distance from point to the tube center ring
        rho_xy = math.sqrt(dx * dx + dy * dy)
        # Distance from point to nearest point on the center ring
        d_ring = math.sqrt((rho_xy - R) ** 2 + dz * dz)
        # Signed distance to torus surface
        return d_ring - r

    def __repr__(self):
        if not _is_real(self.minor_radius):
            return f"Ring(R={self.major_radius:.4g}m)"
        return f"Torus(R={self.major_radius:.4g}m, r={self.minor_radius:.4g}m)"


# ── Convenience constructors ────────────────────────────────────────

def sphere(radius):
    """Create a Sphere shape."""
    return Sphere(radius)


def cylinder(radius, height):
    """Create a Cylinder shape."""
    return Cylinder(radius, height)


def box(x, y, z):
    """Create a Box shape."""
    return Box(x, y, z)


def ellipsoid(rx, ry, rz):
    """Create an Ellipsoid shape."""
    return Ellipsoid(rx, ry, rz)


def cone(radius, height):
    """Create a Cone shape."""
    return Cone(radius, height)


def torus(major_radius, minor_radius):
    """Create a Torus shape."""
    return Torus(major_radius, minor_radius)


# ── Degenerate-case aliases ─────────────────────────────────────────
# These are not separate classes — they are dimensionally compromised
# versions of Box and Torus. The math degenerates cleanly.

def Ring(radius, center=None):
    """Thin ring (hoop) — Torus with zero tube radius (1D curve).

    I_axial = mR² (all mass at distance R from axis)
    I_diameter = ½mR²

    This is Torus(R, 0): Pappus with zero cross-section area.
    """
    return Torus(radius, 0.0, center=center)


def Plane(width, depth, center=None):
    """Finite rectangular plane — Box with zero thickness (2D surface).

    I_z = ¹⁄₁₂m(w² + d²)   (thin plate about normal)
    I_x = ¹⁄₁₂md²           (about width axis)
    I_y = ¹⁄₁₂mw²           (about depth axis)

    This is Box(w, d, 0): cuboid with zero z-extent.
    """
    return Box(width, depth, 0.0, center=center)


def plane(width, depth):
    """Create a Plane (thin Box)."""
    return Plane(width, depth)


# ── Mass from shape + density ───────────────────────────────────────

def mass_from_shape(shape, density_kg_m3):
    """Compute mass from shape volume and density.

    m = ρ × V

    FIRST_PRINCIPLES: definition of density.

    Args:
        shape: Shape instance
        density_kg_m3: material density in kg/m³

    Returns:
        Mass in kg. Returns 0 for dimensionally compromised shapes.
    """
    return density_kg_m3 * shape.volume()


def moment_of_inertia(shape, mass, axis='z'):
    """Moment of inertia from shape geometry and mass.

    I = m × shape.inertia_factor(axis)

    This is the correct way to compute I: geometry determines the
    distribution, mass determines the scale. σ enters through mass only.

    Args:
        shape: Shape instance
        mass: mass in kg
        axis: rotation axis ('x', 'y', or 'z')

    Returns:
        Moment of inertia in kg·m².
    """
    return mass * shape.inertia_factor(axis)


# ── Structure ───────────────────────────────────────────────────────
# Additive list of shaped material volumes.

# Avogadro's number — shapes_per_mole resolution scaling
_N_A = 6.02214076e23


class Structure:
    """Additive shape-packing of a material volume.

    A structure is a flat list of (shape, material) layers that
    together approximate a physical object. Each shape carries its
    own geometric center for placement.

    Volume-filling strategy:
      1. Know target_volume from mass / density.
      2. Start with the largest primitive fitting the outer boundary.
      3. Fill remaining gaps with progressively smaller shapes.
      4. shapes_per_mole controls the resolution of the finest layer.
         More shapes/mol = smoother approximation, fewer = coarser.

    volume_efficiency measures how well the packing fills the target:
      1.0 = perfect fill, <1.0 = gaps remain, >1.0 = overpacked.
    """

    def __init__(self, target_volume, shapes_per_mole=1.0):
        """
        Args:
            target_volume: total volume to fill (m³), from mass/density.
            shapes_per_mole: resolution — shapes per mole of matter.
                Higher = finer packing. Default 1.0 = one shape per mole.
        """
        self.target_volume = float(target_volume)
        self.shapes_per_mole = float(shapes_per_mole)
        self.layers = []  # list of (Shape, material_name) tuples
        self._operations = []  # CSG operation per layer, parallel index
        self._csdf_cache = None  # cached ComposedSDF

    def add(self, shape, material, operation='add'):
        """Add a shaped material layer to the structure.

        Args:
            shape: Shape instance (must be volumetric for matter).
            material: material name string (e.g. 'iron', 'air').
            operation: CSG operation — 'add' (union), 'subtract', 'intersect',
                'smooth_union', 'smooth_subtract', 'smooth_intersect'.
                Default 'add'. Air layers auto-infer 'subtract' in the CSG tree.
        """
        self.layers.append((shape, material))
        self._operations.append(operation)
        self._csdf_cache = None  # invalidate

    @property
    def used_volume(self):
        """Total envelope volume of all layers (m³), including air."""
        return sum(s.volume() for s, _ in self.layers)

    @property
    def material_volume(self):
        """Volume occupied by actual matter (m³).

        Air layers carve hollow regions out of material layers.
        Material volume = (sum of non-air volumes) − (sum of air volumes).
        This is the actual volume of stuff.
        """
        solid = sum(s.volume() for s, m in self.layers if m != 'air')
        hollow = sum(s.volume() for s, m in self.layers if m == 'air')
        return solid - hollow

    @property
    def air_volume(self):
        """Volume of hollow (air) regions (m³)."""
        return sum(s.volume() for s, m in self.layers if m == 'air')

    @property
    def volume_efficiency(self):
        """Ratio of material volume to target volume.

        1.0 = perfect packing. <1.0 = gaps. >1.0 = overpacked.
        Air layers don't count — they represent hollow regions.
        Returns 0.0 if target_volume is zero.
        """
        if self.target_volume <= 0.0:
            return 0.0
        return self.material_volume / self.target_volume

    @property
    def remaining_volume(self):
        """Material volume not yet filled (m³). Negative if overpacked."""
        return self.target_volume - self.material_volume

    @property
    def shape_count(self):
        """Number of shapes in the structure."""
        return len(self.layers)

    @property
    def shape_budget(self, moles=None):
        """Maximum number of shapes at current resolution.

        Args:
            moles: amount of matter in moles. If None, estimates from
                   target_volume assuming ~1e-5 m³/mol (condensed matter).
        """
        if moles is not None:
            return int(self.shapes_per_mole * moles)
        # Rough estimate: condensed matter molar volume ~1e-5 m³/mol
        est_moles = self.target_volume / 1e-5
        return max(1, int(self.shapes_per_mole * est_moles))

    def materials(self):
        """Set of unique material names in the structure."""
        return {m for _, m in self.layers}

    def volume_by_material(self):
        """Dict of material_name → total volume (m³)."""
        result = {}
        for shape, mat in self.layers:
            result[mat] = result.get(mat, 0.0) + shape.volume()
        return result

    def __repr__(self):
        n = len(self.layers)
        eff = self.volume_efficiency
        mats = len(self.materials() - {'air'})
        return f"Structure({n} layers, {mats} materials, efficiency={eff:.3f})"

    def composed_sdf(self):
        """Lazily build and cache a ComposedSDF for this structure.

        The ComposedSDF composes all layers' signed distance fields
        using CSG boolean operations (union, subtract, intersect, smooth).
        Air layers are automatically treated as subtractions.

        Returns:
            ComposedSDF instance.
        """
        if self._csdf_cache is None:
            from .csg import ComposedSDF
            self._csdf_cache = ComposedSDF(self)
        return self._csdf_cache

    def point_inside(self, px, py, pz):
        """Is this point inside the composed solid?

        Uses the composed SDF — accounts for boolean operations
        and air subtractions. A point inside the bore of a pipe
        returns False (air was subtracted).
        """
        return self.composed_sdf().point_inside(px, py, pz)

    def material_at(self, px, py, pz):
        """Which material occupies this point?

        Returns the material name at the given point, accounting
        for layer ordering (last added wins). Returns 'air' for
        hollow regions, None for points outside the structure.
        """
        return self.composed_sdf().material_at(px, py, pz)

    def min_surface_distance(self, px, py, pz):
        """Minimum absolute distance from a point to any shape surface.

        Returns the closest distance to any surface in the structure,
        including air shapes. Air shapes define real physical boundaries
        (e.g. the inner surface of a hollow pipe is a real surface even
        though the interior is air).

        Used by boundary_agreement to test how well our primitives match
        a target shape's outer boundary.
        """
        if not self.layers:
            return float('inf')
        return min(
            abs(shape.surface_distance(px, py, pz))
            for shape, _mat in self.layers
        )

    def boundary_agreement(self, sample_points):
        """Measure how well this structure's surface matches sample points.

        Takes a list of (x, y, z) points that should lie on the target
        shape's boundary. For each point, measures the minimum distance
        to any primitive surface (including air shapes, which define
        real physical boundaries like pipe interiors). Reports statistics.

        Args:
            sample_points: list of (x, y, z) tuples on the target boundary.

        Returns:
            dict with:
                max_deviation_m: worst-case boundary error (metres)
                mean_deviation_m: average boundary error (metres)
                rms_deviation_m: root-mean-square deviation (metres)
                score: 0.0 to 1.0 (1.0 = perfect boundary match)
                n_points: number of sample points tested
                grade: 'exact'/'excellent'/'good'/'fair'/'poor'
        """
        if not sample_points:
            return {
                'max_deviation_m': 0.0, 'mean_deviation_m': 0.0,
                'rms_deviation_m': 0.0, 'score': 1.0,
                'n_points': 0, 'grade': 'exact',
            }

        deviations = []
        for px, py, pz in sample_points:
            d = self.min_surface_distance(px, py, pz)
            deviations.append(d)

        n = len(deviations)
        max_dev = max(deviations)
        mean_dev = sum(deviations) / n
        rms_dev = math.sqrt(sum(d * d for d in deviations) / n)

        # Score: exponential decay based on RMS deviation relative to
        # bounding radius. A deviation of 0 → score 1.0.
        # Deviation equal to bounding radius → score ~0.37.
        if self.layers:
            ref_size = max(s.bounding_radius() for s, _ in self.layers)
        else:
            ref_size = 1.0
        if ref_size > 0:
            score = math.exp(-rms_dev / (ref_size * 0.01))
        else:
            score = 1.0 if rms_dev == 0 else 0.0

        score = max(0.0, min(1.0, score))

        # Grade
        if max_dev < 1e-10:
            grade = 'exact'
        elif score >= 0.95:
            grade = 'excellent'
        elif score >= 0.85:
            grade = 'good'
        elif score >= 0.70:
            grade = 'fair'
        else:
            grade = 'poor'

        return {
            'max_deviation_m': max_dev,
            'mean_deviation_m': mean_dev,
            'rms_deviation_m': rms_dev,
            'score': score,
            'n_points': n,
            'grade': grade,
        }

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)
