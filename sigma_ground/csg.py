"""
CSG Composition Engine — Constructive Solid Geometry via SDF.

Composes analytic primitives (Sphere, Cylinder, Box, Cone, Ellipsoid, Torus)
into complex solids using boolean operations on their signed distance fields.

SDF convention (same as shapes.py):
    negative = inside the solid
    zero     = on the surface
    positive = outside the solid

Boolean operations on SDFs:
    union(a, b)        = min(a, b)       — merge two solids
    subtract(a, b)     = max(a, -b)      — carve b out of a
    intersect(a, b)    = max(a, b)       — keep only the overlap
    smooth_union(a, b) = smooth-min      — organic blending

The CSG tree:
    CSGNode (abstract)
    ├── CSGLeaf   — wraps a single Shape + material
    └── CSGBranch — binary operation on two CSGNodes

ComposedSDF is the public API: takes a Structure, builds the CSG tree,
and provides sdf(), point_inside(), material_at() queries.

FIRST_PRINCIPLES: all operations are exact for the primitives they compose.
No approximation, no sampling, no mesh conversion. The SDF is evaluated
analytically at every query point.

Pure geometry — no rendering, no physics.
"""

import math


# ── SDF composition functions ────────────────────────────────────────

def sdf_union(a, b):
    """Union of two signed distances: min(a, b).

    The resulting solid contains all points inside either a or b.
    """
    return min(a, b)


def sdf_subtract(a, b):
    """Subtract b from a: max(a, -b).

    The resulting solid contains points inside a but outside b.
    Carves b's volume out of a.
    """
    return max(a, -b)


def sdf_intersect(a, b):
    """Intersection of two signed distances: max(a, b).

    The resulting solid contains only points inside both a and b.
    """
    return max(a, b)


def sdf_smooth_union(a, b, k):
    """Smooth union (polynomial smooth-min) with blending radius k.

    Produces a smooth, organic transition between two solids instead
    of a sharp crease. k controls the blend radius in metres — larger k
    means a wider, softer blend.

    k = 0 degenerates to exact union (min).

    FIRST_PRINCIPLES: Inigo Quilez's polynomial smooth minimum.
    Widely used in SDF ray-marching (ShaderToy, Dreams PS4).
    """
    if k <= 0.0:
        return min(a, b)
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


def sdf_smooth_subtract(a, b, k):
    """Smooth subtraction with blending radius k.

    Carves b from a with a smooth fillet at the intersection.
    """
    if k <= 0.0:
        return max(a, -b)
    h = max(k - abs(-b - a), 0.0) / k
    return max(a, -b) + h * h * k * 0.25


def sdf_smooth_intersect(a, b, k):
    """Smooth intersection with blending radius k."""
    if k <= 0.0:
        return max(a, b)
    h = max(k - abs(a - b), 0.0) / k
    return max(a, b) + h * h * k * 0.25


# ── Operation dispatch ───────────────────────────────────────────────

# Maps operation name → (function, needs_k)
_OPS = {
    'add':              (sdf_union, False),
    'union':            (sdf_union, False),
    'subtract':         (sdf_subtract, False),
    'intersect':        (sdf_intersect, False),
    'smooth_union':     (sdf_smooth_union, True),
    'smooth_subtract':  (sdf_smooth_subtract, True),
    'smooth_intersect': (sdf_smooth_intersect, True),
}

_DEFAULT_SMOOTH_K = 0.01  # 1cm default blend radius


def _apply_op(op_name, a, b, k=None):
    """Apply a named SDF operation."""
    func, needs_k = _OPS[op_name]
    if needs_k:
        return func(a, b, k if k is not None else _DEFAULT_SMOOTH_K)
    return func(a, b)


# ── CSG Node hierarchy ──────────────────────────────────────────────

class CSGNode:
    """Abstract base for CSG tree nodes."""

    def sdf(self, px, py, pz):
        """Signed distance from point to composed surface."""
        raise NotImplementedError

    def point_inside(self, px, py, pz):
        """Is the point inside the composed solid?"""
        return self.sdf(px, py, pz) < 0.0

    def bounding_radius(self):
        """Conservative bounding sphere radius from origin."""
        raise NotImplementedError


class CSGLeaf(CSGNode):
    """Wraps a single Shape as a CSG tree leaf.

    Delegates sdf() to the Shape's surface_distance() method.
    """

    def __init__(self, shape, material='default'):
        self.shape = shape
        self.material = material

    def sdf(self, px, py, pz):
        return self.shape.surface_distance(px, py, pz)

    def bounding_radius(self):
        cx, cy, cz = self.shape.center
        return self.shape.bounding_radius() + math.sqrt(cx*cx + cy*cy + cz*cz)

    def __repr__(self):
        return f"CSGLeaf({self.shape}, '{self.material}')"


class CSGBranch(CSGNode):
    """Binary CSG operation on two child nodes.

    Composes left and right children using a boolean SDF operation.
    """

    def __init__(self, left, right, operation='add', k=None):
        """
        Args:
            left:      CSGNode (the base solid)
            right:     CSGNode (the operand)
            operation: 'add', 'subtract', 'intersect',
                       'smooth_union', 'smooth_subtract', 'smooth_intersect'
            k:         blend radius for smooth operations (metres)
        """
        if operation not in _OPS:
            raise ValueError(f"Unknown CSG operation: '{operation}'. "
                             f"Valid: {list(_OPS.keys())}")
        self.left = left
        self.right = right
        self.operation = operation
        self.k = k

    def sdf(self, px, py, pz):
        a = self.left.sdf(px, py, pz)
        b = self.right.sdf(px, py, pz)
        return _apply_op(self.operation, a, b, self.k)

    def bounding_radius(self):
        # Conservative: max of children
        return max(self.left.bounding_radius(), self.right.bounding_radius())

    def __repr__(self):
        return f"CSGBranch({self.operation}, {self.left}, {self.right})"


# ── ComposedSDF — public API ────────────────────────────────────────

class ComposedSDF:
    """Composes a Structure's layers into a single evaluable SDF.

    Takes a Structure (flat list of shapes + materials + operations)
    and builds a CSG tree. Provides point queries: sdf, point_inside,
    material_at.

    Air layers are automatically treated as subtractions — a pipe built
    as [Cylinder(outer, steel), Cylinder(inner, air)] correctly carves
    the bore out of the shell without explicit operation='subtract'.
    """

    def __init__(self, structure):
        """Build CSG tree from a Structure.

        Args:
            structure: Structure instance with .layers and ._operations.
        """
        self._structure = structure
        self._root = None
        self._leaves = []  # ordered list of (CSGLeaf, operation) for material_at
        self._build()

    def _build(self):
        """Construct the CSG tree from the structure's layers."""
        layers = self._structure.layers
        ops = getattr(self._structure, '_operations', None)
        if ops is None:
            ops = ['add'] * len(layers)

        if not layers:
            self._root = None
            return

        # Build leaves with inferred operations
        for i, (shape, material) in enumerate(layers):
            op = ops[i] if i < len(ops) else 'add'

            # Air layers automatically become subtractions
            if material == 'air' and op == 'add':
                op = 'subtract'

            leaf = CSGLeaf(shape, material)
            self._leaves.append((leaf, op))

        # Build tree left-to-right: first leaf is the root,
        # each subsequent leaf is composed with the accumulated tree
        self._root = self._leaves[0][0]  # first leaf, ignore its op
        for leaf, op in self._leaves[1:]:
            self._root = CSGBranch(self._root, leaf, op)

    def sdf(self, px, py, pz):
        """Signed distance from point to composed surface.

        Negative = inside, zero = surface, positive = outside.
        Returns inf if structure is empty.
        """
        if self._root is None:
            return float('inf')
        return self._root.sdf(px, py, pz)

    def point_inside(self, px, py, pz):
        """Is the point inside the composed solid?

        A point is inside if sdf < 0. Points exactly on the surface
        (sdf = 0) are considered outside (boundary convention).
        """
        return self.sdf(px, py, pz) < 0.0

    def material_at(self, px, py, pz):
        """Which material occupies this point?

        Iterates layers in reverse order (last added wins).
        Returns the material of the last leaf whose shape contains
        the point. Air is returned as 'air' — the caller decides
        what that means.

        Returns None if the point is outside all shapes.
        """
        if not self._leaves:
            return None

        # Check leaves in reverse: last added takes priority
        for leaf, _op in reversed(self._leaves):
            if leaf.shape.surface_distance(px, py, pz) < 0.0:
                return leaf.material

        return None

    def sample_surface(self, n_points, bounds=None):
        """Sample points approximately on the composed surface.

        Uses rejection sampling within the bounding box: generates
        random points, evaluates SDF, keeps those close to zero.

        Args:
            n_points: target number of surface points.
            bounds:   ((xmin,ymin,zmin), (xmax,ymax,zmax)) bounding box.
                      If None, estimated from structure.

        Returns:
            list of (x, y, z) tuples near the composed surface.
        """
        import random

        if self._root is None:
            return []

        if bounds is None:
            r = self._root.bounding_radius() * 1.1
            bounds = ((-r, -r, -r), (r, r, r))

        (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds

        # Tolerance: points within this distance of surface are accepted
        diag = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
        tol = diag * 0.005  # 0.5% of bounding diagonal

        points = []
        max_attempts = n_points * 200  # safety limit
        attempts = 0

        while len(points) < n_points and attempts < max_attempts:
            px = xmin + random.random() * (xmax - xmin)
            py = ymin + random.random() * (ymax - ymin)
            pz = zmin + random.random() * (zmax - zmin)
            d = abs(self.sdf(px, py, pz))
            if d < tol:
                points.append((px, py, pz))
            attempts += 1

        return points

    def slice_at_z(self, z, resolution, bounds=None):
        """Evaluate composed SDF on a 2D grid at height z.

        Returns a 2D list of SDF values. Useful for visualization
        and debugging — shows the cross-section of the solid.

        Args:
            z:          height to slice at (metres).
            resolution: grid cells per axis (NxN grid).
            bounds:     ((xmin,ymin), (xmax,ymax)) 2D bounds.
                        If None, estimated from structure.

        Returns:
            list of lists: grid[iy][ix] = sdf value at that cell.
        """
        if self._root is None:
            return []

        if bounds is None:
            r = self._root.bounding_radius() * 1.1
            bounds = ((-r, -r), (r, r))

        (xmin, ymin), (xmax, ymax) = bounds
        dx = (xmax - xmin) / resolution
        dy = (ymax - ymin) / resolution

        grid = []
        for iy in range(resolution):
            row = []
            py = ymin + (iy + 0.5) * dy
            for ix in range(resolution):
                px = xmin + (ix + 0.5) * dx
                row.append(self.sdf(px, py, z))
            grid.append(row)

        return grid

    def __repr__(self):
        n = len(self._leaves)
        return f"ComposedSDF({n} leaves)"
