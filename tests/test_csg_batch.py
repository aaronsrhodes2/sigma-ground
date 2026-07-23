"""
Tests for the vectorized/batch SDF query path (ComposedSDF.sdf_batch et al).

Correctness discipline: every batch method here is checked against its
scalar sibling called in a Python loop, at rtol=atol=1e-9 — tight enough to
catch a real formula transcription bug, loose enough to absorb the few-ULP
noise from numpy's vectorized reduction order vs Python's math.sqrt.

Covers:
  * Every primitive with a hand-vectorized override (Sphere, HollowSphere,
    Cylinder, Box, Ellipsoid, Cone, Torus).
  * Branch-forcing fixtures for the three primitives with per-point control
    flow (Cylinder, Box, Cone) — random sampling alone doesn't reliably hit
    every named branch.
  * The three deckard/construct.py wrapper shapes (_Rotated, _Clipped,
    _Subtracted) that CSGLeaf.sdf_batch must delegate through correctly.
  * Full ComposedSDF trees exercising every _OPS entry.
  * The universal Shape.surface_distance_batch fallback (no override).
  * A performance canary (not a correctness gate) so a future accidental
    revert to a Python-loop shim inside sdf_batch doesn't pass silently.
"""

import math
import random
import time
import unittest

import numpy as np

from sigma_ground.kernel.csg import ComposedSDF, CSGLeaf, CSGBranch, _OPS
from sigma_ground.kernel.shapes import (
    Shape, Sphere, HollowSphere, Cylinder, Box, Ellipsoid, Cone, Torus,
    Structure,
)
from sigma_ground.kernel.outline import Outline
from sigma_ground.deckard.construct import _Rotated, _Clipped, _Subtracted, _rotation


TOL = dict(rtol=1e-9, atol=1e-9)


def _scalar_loop(shape, points):
    """Reference implementation: surface_distance() called once per point."""
    return np.array([shape.surface_distance(p[0], p[1], p[2]) for p in points],
                     dtype=np.float64)


def _random_points(n, r, seed, center=(0.0, 0.0, 0.0)):
    """n uniform random points in [-2r, 2r]^3 about center — deliberately
    wide enough to cross interior, exterior, and near-surface regions."""
    rng = random.Random(seed)
    cx, cy, cz = center
    pts = [(cx + rng.uniform(-2 * r, 2 * r),
            cy + rng.uniform(-2 * r, 2 * r),
            cz + rng.uniform(-2 * r, 2 * r)) for _ in range(n)]
    return np.array(pts, dtype=np.float64)


# ── Per-shape parity, dense random sampling ─────────────────────────────

class TestShapeBatchParity(unittest.TestCase):
    """surface_distance_batch() must match surface_distance() looped, for
    every shape type this pass hand-vectorizes."""

    N = 2000

    def _check(self, shape, seed, center=(0.0, 0.0, 0.0)):
        r = shape.bounding_radius()
        pts = _random_points(self.N, max(r, 1e-6), seed, center)
        expected = _scalar_loop(shape, pts)
        actual = shape.surface_distance_batch(pts)
        self.assertEqual(actual.shape, (self.N,))
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_sphere(self):
        self._check(Sphere(1.0), seed=1)
        self._check(Sphere(0.37, center=(1.0, -2.0, 0.5)), seed=2, center=(1.0, -2.0, 0.5))

    def test_hollow_sphere(self):
        self._check(HollowSphere(2.5), seed=3)
        self._check(HollowSphere(0.05, center=(-1.0, 0.0, 3.0)), seed=4, center=(-1.0, 0.0, 3.0))

    def test_cylinder(self):
        self._check(Cylinder(1.0, 2.0), seed=5)
        self._check(Cylinder(0.05, 0.2, center=(0.1, -0.2, 0.3)), seed=6, center=(0.1, -0.2, 0.3))
        # Squat / tall degenerate-ish aspect ratios
        self._check(Cylinder(3.0, 0.01), seed=7)
        self._check(Cylinder(0.01, 3.0), seed=8)

    def test_box(self):
        self._check(Box(1.0, 2.0, 3.0), seed=9)
        self._check(Box(0.5, 0.5, 0.5, center=(2.0, 2.0, 2.0)), seed=10, center=(2.0, 2.0, 2.0))
        self._check(Box(5.0, 0.1, 0.1), seed=11)

    def test_ellipsoid(self):
        self._check(Ellipsoid(1.0, 2.0, 0.5), seed=12)
        self._check(Ellipsoid(0.1, 0.1, 0.1, center=(1.0, 1.0, 1.0)), seed=13, center=(1.0, 1.0, 1.0))

    def test_cone(self):
        self._check(Cone(1.0, 2.0), seed=14)
        self._check(Cone(0.05, 0.2, center=(0.1, -0.2, 0.3)), seed=15, center=(0.1, -0.2, 0.3))
        self._check(Cone(2.0, 0.2), seed=16)

    def test_torus(self):
        self._check(Torus(2.0, 0.5), seed=17)
        self._check(Torus(0.2, 0.05, center=(1.0, 0.0, -1.0)), seed=18, center=(1.0, 0.0, -1.0))
        # Ring degenerate case (minor_radius = 0)
        self._check(Torus(1.0, 0.0), seed=19)


# ── Branch-forcing fixtures ──────────────────────────────────────────────

class TestBranchForcingFixtures(unittest.TestCase):
    """Hand-constructed points that hit every named branch at least once —
    don't rely solely on random sampling for shapes with per-point control
    flow (Cylinder, Box, Cone)."""

    def test_cylinder_branches(self):
        cyl = Cylinder(1.0, 2.0)   # radius 1, height 2 -> half_h = 1
        pts = np.array([
            (0.0, 0.0, 0.0),     # fully inside (both <=0)
            (2.0, 0.0, 0.0),     # purely radial-outside (d_radial>0, d_axial<=0)
            (0.0, 0.0, 2.0),     # purely axial-outside (d_radial<=0, d_axial>0)
            (2.0, 0.0, 2.0),     # both-outside corner
            (1.0, 0.0, 0.0),     # exactly on lateral surface
            (0.0, 0.0, 1.0),     # exactly on cap
        ], dtype=np.float64)
        expected = _scalar_loop(cyl, pts)
        actual = cyl.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_box_branches(self):
        box = Box(2.0, 2.0, 2.0)   # half-extents 1,1,1
        pts = np.array([
            (0.0, 0.0, 0.0),      # inside
            (2.0, 0.0, 0.0),      # outside, face-nearest
            (2.0, 2.0, 0.0),      # outside, edge-nearest
            (2.0, 2.0, 2.0),      # outside, corner-nearest
            (1.0, 0.0, 0.0),      # exactly on face
        ], dtype=np.float64)
        expected = _scalar_loop(box, pts)
        actual = box.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_cone_branches(self):
        cone = Cone(1.0, 2.0)     # base at cz - h/4 = -0.5, apex at 1.5 (local frame 0..2)
        pts = np.array([
            (0.0, 0.0, -2.0),     # below base, radially inside -> d_base
            (5.0, 0.0, -2.0),     # below base, radially outside -> hypot
            (0.0, 0.0, 5.0),      # above apex
            (0.1, 0.0, 0.0),      # alongside, near axis (inside laterally, local_z small)
            (5.0, 0.0, 0.5),      # alongside, well outside laterally
        ], dtype=np.float64)
        expected = _scalar_loop(cone, pts)
        actual = cone.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)


# ── Wrapper parity (_Rotated, _Clipped, _Subtracted) ─────────────────────

class TestWrapperBatchParity(unittest.TestCase):
    """deckard/construct.py's wrapper shapes must delegate sdf_batch
    correctly — these matter because CSGLeaf.sdf_batch is fed whatever
    _shape_from()/_expand_fill()/_make_conform() produce, which is often
    one of these wrappers, not a raw Shape."""

    def test_rotated_box(self):
        base = Box(1.0, 2.0, 3.0)
        R = _rotation((30.0, 45.0, 60.0))   # non-trivial Euler rotation
        rot = _Rotated(base, R)
        r = base.bounding_radius()
        pts = _random_points(2000, r, seed=101)
        expected = _scalar_loop(rot, pts)
        actual = rot.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_rotated_cone(self):
        base = Cone(0.5, 1.5, center=(0.2, -0.1, 0.3))
        R = _rotation((15.0, 80.0, -40.0))
        rot = _Rotated(base, R)
        r = base.bounding_radius()
        pts = _random_points(2000, r, seed=102, center=base.center)
        expected = _scalar_loop(rot, pts)
        actual = rot.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_clipped(self):
        base = Sphere(1.0)
        clipped = _Clipped(base, axis=2, sign=1.0, level=0.2,
                            volume=1.0, center=(0.0, 0.0, 0.0))
        pts = _random_points(2000, 1.0, seed=103)
        expected = _scalar_loop(clipped, pts)
        actual = clipped.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_clipped_gas_side(self):
        base = Cylinder(0.5, 1.0)
        clipped = _Clipped(base, axis=2, sign=-1.0, level=-0.1,
                            volume=1.0, center=(0.0, 0.0, 0.0))
        pts = _random_points(2000, 0.5, seed=104)
        expected = _scalar_loop(clipped, pts)
        actual = clipped.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_subtracted(self):
        base = Box(2.0, 2.0, 2.0)
        cut = Sphere(0.8)
        sub = _Subtracted(base, cut, volume=1.0, center=(0.0, 0.0, 0.0))
        pts = _random_points(2000, 1.0, seed=105)
        expected = _scalar_loop(sub, pts)
        actual = sub.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)


# ── Tree-composition parity (every _OPS entry through ComposedSDF) ───────

class TestComposedSDFBatchParity(unittest.TestCase):
    """composed.sdf_batch(points) must match composed.sdf(*p) looped, for
    a tree exercising every named CSG operation."""

    def _check_structure(self, structure, seed, extent=3.0):
        composed = ComposedSDF(structure)
        pts = _random_points(3000, extent, seed)
        expected = np.array([composed.sdf(p[0], p[1], p[2]) for p in pts])
        actual = composed.sdf_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_every_op(self):
        for op in _OPS:
            with self.subTest(op=op):
                s = Structure(target_volume=1.0)
                s.add(Sphere(1.0), 'iron', operation='add')
                s.add(Box(1.0, 1.0, 1.0, center=(0.5, 0.5, 0.5)), 'steel', operation=op)
                self._check_structure(s, seed=hash(op) % 10000)

    def test_explicit_k_and_default_k(self):
        s = Structure(target_volume=1.0)
        s.add(Sphere(1.0), 'iron', operation='add')
        s.add(Sphere(0.7, center=(1.0, 0.0, 0.0)), 'iron', operation='smooth_union')
        self._check_structure(s, seed=201)

        s2 = Structure(target_volume=1.0)
        s2._csdf_cache = None
        s2.add(Sphere(1.0), 'iron', operation='add')
        s2.layers.append((Sphere(0.7, center=(1.0, 0.0, 0.0)), 'iron'))
        s2._operations.append('smooth_union')
        composed = ComposedSDF(s2)
        # exercise the explicit-k branch of CSGBranch directly
        composed._root.k = 0.25
        pts = _random_points(2000, 3.0, seed=202)
        expected = np.array([composed.sdf(p[0], p[1], p[2]) for p in pts])
        actual = composed.sdf_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_pipe_pattern(self):
        """Mirrors deckard/construct.py's actual usage: outer shell +
        subtracted bore, air auto-subtract via Structure.add()."""
        s = Structure(target_volume=1.0)
        s.add(Cylinder(0.02, 0.5), 'steel', operation='add')
        s.add(Cylinder(0.015, 0.5), 'air', operation='add')  # auto -> subtract
        self._check_structure(s, seed=301, extent=0.05)

    def test_deep_mixed_tree(self):
        """2-4 leaves deep, mixed primitive types, mixed ops."""
        s = Structure(target_volume=1.0)
        s.add(Sphere(1.0), 'iron', operation='add')
        s.add(Cylinder(0.4, 3.0, center=(0.0, 0.0, 0.0)), 'iron', operation='intersect')
        s.add(Box(0.3, 0.3, 0.3, center=(0.5, 0.0, 0.0)), 'air', operation='subtract')
        s.add(Torus(0.6, 0.15, center=(0.0, 0.0, 0.5)), 'brass', operation='smooth_union')
        self._check_structure(s, seed=302)

    def test_empty_structure(self):
        s = Structure(target_volume=1.0)
        composed = ComposedSDF(s)
        pts = np.zeros((5, 3))
        actual = composed.sdf_batch(pts)
        self.assertTrue(np.all(np.isinf(actual)))


# ── Fallback-path parity (no override yet) ────────────────────────────────

class TestFallbackBatchParity(unittest.TestCase):
    """The §2 universal default (Shape.surface_distance_batch) must produce
    identical results to the scalar loop for shapes with no hand-vectorized
    override. Trivially true by construction, but this is the regression
    test for the safety net everything unvectorized relies on."""

    def test_outline_extrude(self):
        profile = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)]
        shp = Outline(profile, mode='extrude', thickness=0.2)
        r = shp.bounding_radius()
        pts = _random_points(500, r, seed=401)
        expected = _scalar_loop(shp, pts)
        actual = shp.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_outline_revolve(self):
        profile = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.3)]
        shp = Outline(profile, mode='revolve')
        r = shp.bounding_radius()
        pts = _random_points(500, r, seed=402)
        expected = _scalar_loop(shp, pts)
        actual = shp.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_dummy_shape_bounding_sphere_default(self):
        """A minimal Shape subclass that doesn't override surface_distance
        at all — exercises the base class's own bounding-sphere fallback,
        through the batch default."""
        class _Dummy(Shape):
            def __init__(self, r, center=None):
                self._r = r
                self.__init_center__(center)

            def bounding_radius(self):
                return self._r

        shp = _Dummy(1.5, center=(0.1, 0.2, 0.3))
        pts = _random_points(500, 1.5, seed=403, center=(0.1, 0.2, 0.3))
        expected = _scalar_loop(shp, pts)
        actual = shp.surface_distance_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)

    def test_composed_with_unvectorized_leaf(self):
        """A ComposedSDF tree with an Outline leaf mixed into ops that ARE
        vectorized — confirms CSGLeaf.sdf_batch falls through correctly for
        just that one leaf without breaking the rest of the tree."""
        s = Structure(target_volume=1.0)
        s.add(Sphere(1.0), 'iron', operation='add')
        profile = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.3), (0.0, 0.3)]
        s.add(Outline(profile, mode='extrude', thickness=0.1, center=(0.3, 0.0, 0.0)),
              'brass', operation='smooth_union')
        composed = ComposedSDF(s)
        pts = _random_points(1000, 2.0, seed=404)
        expected = np.array([composed.sdf(p[0], p[1], p[2]) for p in pts])
        actual = composed.sdf_batch(pts)
        np.testing.assert_allclose(actual, expected, **TOL)


# ── Zero-regression: sdf_batch must not mutate/step on the scalar path ───

class TestNoScalarRegression(unittest.TestCase):
    """Calling the new batch methods must not change what the pre-existing
    scalar methods return (no shared mutable state, no monkey-patching)."""

    def test_scalar_sdf_unaffected_by_batch_call(self):
        s = Structure(target_volume=1.0)
        s.add(Sphere(1.0), 'iron', operation='add')
        s.add(Box(0.5, 0.5, 0.5), 'air', operation='subtract')
        composed = ComposedSDF(s)

        before = composed.sdf(0.3, 0.1, -0.2)
        composed.sdf_batch(_random_points(500, 2.0, seed=501))
        after = composed.sdf(0.3, 0.1, -0.2)
        self.assertEqual(before, after)


# ── Performance canary (not a correctness gate) ───────────────────────────

class TestBatchPerformance(unittest.TestCase):
    """Sanity-check that sdf_batch is actually faster than the scalar loop
    on a realistic grid size — a canary against a future accidental revert
    to a Python-loop shim inside sdf_batch. Loose threshold, not flaky."""

    def test_speedup_on_voxel_grid(self):
        s = Structure(target_volume=1.0)
        s.add(Sphere(1.0), 'iron', operation='add')
        s.add(Cylinder(0.4, 3.0), 'iron', operation='intersect')
        s.add(Box(0.3, 0.3, 0.3, center=(0.5, 0.0, 0.0)), 'air', operation='subtract')
        s.add(Torus(0.6, 0.15, center=(0.0, 0.0, 0.5)), 'brass', operation='smooth_union')
        composed = ComposedSDF(s)

        n = 50
        lo, hi = -2.0, 2.0
        xs = np.linspace(lo, hi, n)
        grid = np.array([(x, y, z) for x in xs for y in xs for z in xs], dtype=np.float64)
        self.assertEqual(grid.shape, (n ** 3, 3))

        t0 = time.perf_counter()
        batch_result = composed.sdf_batch(grid)
        t_batch = time.perf_counter() - t0

        t0 = time.perf_counter()
        scalar_result = np.array([composed.sdf(p[0], p[1], p[2]) for p in grid])
        t_scalar = time.perf_counter() - t0

        np.testing.assert_allclose(batch_result, scalar_result, **TOL)

        speedup = t_scalar / t_batch if t_batch > 0 else float('inf')
        print(f"\n[batch perf] N={n**3} points: scalar={t_scalar:.4f}s "
              f"batch={t_batch:.4f}s speedup={speedup:.1f}x")
        self.assertGreater(speedup, 5.0)


if __name__ == '__main__':
    unittest.main()
