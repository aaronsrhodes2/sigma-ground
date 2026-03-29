"""
Tests for CSG Composition Engine (A3DVS — Adaptive 3D Volumetric Shaper).

Tests SDF composition operations, CSG tree construction, ComposedSDF queries,
and backward compatibility with existing Structure usage.
"""

import math
import unittest

from sigma_ground.csg import (
    CSGLeaf, CSGBranch, ComposedSDF,
    sdf_union, sdf_subtract, sdf_intersect,
    sdf_smooth_union, sdf_smooth_subtract, sdf_smooth_intersect,
)
from sigma_ground.shapes import (
    Sphere, Cylinder, Box, Cone, Ellipsoid, Torus, Structure,
)


# ── SDF composition functions ───────────────────────────────────────

class TestSDFOperations(unittest.TestCase):
    """Test the standalone SDF composition functions."""

    def test_union_takes_minimum(self):
        self.assertEqual(sdf_union(-2.0, -1.0), -2.0)
        self.assertEqual(sdf_union(0.5, 1.5), 0.5)
        self.assertEqual(sdf_union(-0.5, 0.5), -0.5)

    def test_subtract_carves(self):
        # Inside a but outside b → inside result
        self.assertLess(sdf_subtract(-1.0, 1.0), 0)
        # Inside both a and b → outside result (b carved out)
        self.assertGreater(sdf_subtract(-1.0, -1.0), 0)
        # Outside a → outside result regardless of b
        self.assertGreater(sdf_subtract(1.0, -1.0), 0)

    def test_intersect_takes_maximum(self):
        self.assertEqual(sdf_intersect(-2.0, -1.0), -1.0)
        self.assertEqual(sdf_intersect(0.5, 1.5), 1.5)
        # One inside one outside → outside
        self.assertGreater(sdf_intersect(-0.5, 0.5), 0)

    def test_smooth_union_blends(self):
        # At zero k, degenerates to exact union
        self.assertAlmostEqual(sdf_smooth_union(-1.0, 0.5, 0.0), -1.0)
        # With positive k, result is <= min (smooth-min is always <= min)
        result = sdf_smooth_union(-0.1, -0.1, 0.5)
        self.assertLessEqual(result, -0.1)
        # Far apart values: smooth union ≈ exact union
        result = sdf_smooth_union(-5.0, 5.0, 0.01)
        self.assertAlmostEqual(result, -5.0, places=3)

    def test_smooth_subtract(self):
        self.assertAlmostEqual(sdf_smooth_subtract(-1.0, 1.0, 0.0),
                               sdf_subtract(-1.0, 1.0))

    def test_smooth_intersect(self):
        self.assertAlmostEqual(sdf_smooth_intersect(-2.0, -1.0, 0.0),
                               sdf_intersect(-2.0, -1.0))


# ── CSGLeaf ──────────────────────────────────────────────────────────

class TestCSGLeaf(unittest.TestCase):
    """Test CSGLeaf wrapping individual shapes."""

    def test_sphere_leaf(self):
        leaf = CSGLeaf(Sphere(0.1), 'iron')
        self.assertAlmostEqual(leaf.sdf(0, 0, 0), -0.1)
        self.assertAlmostEqual(leaf.sdf(0.1, 0, 0), 0.0)
        self.assertAlmostEqual(leaf.sdf(0.2, 0, 0), 0.1)
        self.assertTrue(leaf.point_inside(0, 0, 0))
        self.assertFalse(leaf.point_inside(0.2, 0, 0))
        self.assertEqual(leaf.material, 'iron')

    def test_cylinder_leaf(self):
        leaf = CSGLeaf(Cylinder(0.05, 0.2), 'steel')
        self.assertTrue(leaf.point_inside(0, 0, 0))
        self.assertFalse(leaf.point_inside(0.06, 0, 0))

    def test_box_leaf(self):
        leaf = CSGLeaf(Box(0.2, 0.2, 0.2), 'wood')
        self.assertTrue(leaf.point_inside(0, 0, 0))
        self.assertFalse(leaf.point_inside(0.15, 0, 0))

    def test_bounding_radius_accounts_for_center(self):
        leaf = CSGLeaf(Sphere(0.1, center=(1, 0, 0)))
        # bounding_radius from origin = shape radius + distance from origin
        self.assertGreater(leaf.bounding_radius(), 1.0)


# ── CSGBranch ────────────────────────────────────────────────────────

class TestCSGBranch(unittest.TestCase):
    """Test binary CSG operations on shape pairs."""

    def test_union_of_two_spheres(self):
        left = CSGLeaf(Sphere(0.1, center=(0, 0, 0)))
        right = CSGLeaf(Sphere(0.1, center=(0.15, 0, 0)))
        branch = CSGBranch(left, right, 'add')

        # Inside left sphere
        self.assertTrue(branch.point_inside(0, 0, 0))
        # Inside right sphere
        self.assertTrue(branch.point_inside(0.15, 0, 0))
        # Outside both
        self.assertFalse(branch.point_inside(0.3, 0, 0))

    def test_subtract_sphere_from_sphere(self):
        # Large sphere minus small sphere at center = hollow shell
        outer = CSGLeaf(Sphere(0.1))
        inner = CSGLeaf(Sphere(0.05))
        branch = CSGBranch(outer, inner, 'subtract')

        # Center: inside inner, so subtracted out
        self.assertFalse(branch.point_inside(0, 0, 0))
        # In the shell: inside outer but outside inner
        self.assertTrue(branch.point_inside(0.07, 0, 0))
        # Outside outer
        self.assertFalse(branch.point_inside(0.15, 0, 0))

    def test_intersect_overlapping_spheres(self):
        left = CSGLeaf(Sphere(0.1, center=(0, 0, 0)))
        right = CSGLeaf(Sphere(0.1, center=(0.1, 0, 0)))
        branch = CSGBranch(left, right, 'intersect')

        # At overlap region (midpoint)
        self.assertTrue(branch.point_inside(0.05, 0, 0))
        # Inside left only
        self.assertFalse(branch.point_inside(-0.05, 0, 0))
        # Inside right only
        self.assertFalse(branch.point_inside(0.15, 0, 0))

    def test_smooth_union(self):
        left = CSGLeaf(Sphere(0.1, center=(0, 0, 0)))
        right = CSGLeaf(Sphere(0.08, center=(0, 0, 0.15)))
        branch = CSGBranch(left, right, 'smooth_union', k=0.02)

        # Inside both
        self.assertTrue(branch.point_inside(0, 0, 0))
        self.assertTrue(branch.point_inside(0, 0, 0.15))
        # Junction region — smoothed, should be inside
        self.assertTrue(branch.point_inside(0, 0, 0.08))

    def test_invalid_operation_raises(self):
        left = CSGLeaf(Sphere(0.1))
        right = CSGLeaf(Sphere(0.1))
        with self.assertRaises(ValueError):
            CSGBranch(left, right, 'explode')


# ── ComposedSDF ──────────────────────────────────────────────────────

class TestComposedSDF(unittest.TestCase):
    """Test ComposedSDF built from Structure."""

    def test_single_sphere(self):
        s = Structure(target_volume=0.004)
        s.add(Sphere(0.1), 'iron')
        csdf = ComposedSDF(s)
        self.assertTrue(csdf.point_inside(0, 0, 0))
        self.assertFalse(csdf.point_inside(0.2, 0, 0))
        self.assertAlmostEqual(csdf.sdf(0.1, 0, 0), 0.0, places=10)

    def test_hollow_pipe(self):
        """Pipe: outer steel cylinder + inner air cylinder."""
        s = Structure(target_volume=0.001)
        s.add(Cylinder(0.025, 1.0), 'steel')
        s.add(Cylinder(0.020, 1.0), 'air')
        csdf = ComposedSDF(s)

        # In the wall (between r=0.020 and r=0.025)
        self.assertTrue(csdf.point_inside(0.022, 0, 0))
        # In the bore (r < 0.020) — air subtracted
        self.assertFalse(csdf.point_inside(0, 0, 0))
        # Outside (r > 0.025)
        self.assertFalse(csdf.point_inside(0.03, 0, 0))

    def test_air_auto_subtract(self):
        """Air layers automatically become subtract operations."""
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'glass')
        s.add(Sphere(0.08), 'air')
        csdf = ComposedSDF(s)

        # Shell between r=0.08 and r=0.1
        self.assertTrue(csdf.point_inside(0.09, 0, 0))
        # Center: air subtracted
        self.assertFalse(csdf.point_inside(0, 0, 0))

    def test_material_at_pipe(self):
        """material_at correctly reports steel vs air vs outside."""
        s = Structure(target_volume=0.001)
        s.add(Cylinder(0.025, 1.0), 'steel')
        s.add(Cylinder(0.020, 1.0), 'air')
        csdf = ComposedSDF(s)

        self.assertEqual(csdf.material_at(0.022, 0, 0), 'steel')
        self.assertEqual(csdf.material_at(0, 0, 0), 'air')
        self.assertIsNone(csdf.material_at(0.03, 0, 0))

    def test_material_at_solid(self):
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'iron')
        csdf = ComposedSDF(s)
        self.assertEqual(csdf.material_at(0, 0, 0), 'iron')
        self.assertIsNone(csdf.material_at(0.2, 0, 0))

    def test_smooth_union_snowman(self):
        """Two spheres with smooth union form a snowman."""
        s = Structure(target_volume=0.01)
        s.add(Sphere(0.10, center=(0, 0, 0)), 'snow')
        s.add(Sphere(0.08, center=(0, 0, 0.15)), 'snow',
              operation='smooth_union')
        csdf = ComposedSDF(s)

        self.assertTrue(csdf.point_inside(0, 0, 0))       # body
        self.assertTrue(csdf.point_inside(0, 0, 0.15))    # head
        self.assertTrue(csdf.point_inside(0, 0, 0.08))    # junction
        self.assertFalse(csdf.point_inside(0.2, 0, 0))    # outside

    def test_empty_structure(self):
        s = Structure(target_volume=0.0)
        csdf = ComposedSDF(s)
        self.assertEqual(csdf.sdf(0, 0, 0), float('inf'))
        self.assertFalse(csdf.point_inside(0, 0, 0))
        self.assertIsNone(csdf.material_at(0, 0, 0))

    def test_repr(self):
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'iron')
        s.add(Sphere(0.05), 'air')
        csdf = ComposedSDF(s)
        self.assertIn('2 leaves', repr(csdf))

    def test_slice_at_z(self):
        """Slice through a sphere produces a circular cross-section."""
        s = Structure(target_volume=0.004)
        s.add(Sphere(0.1), 'iron')
        csdf = ComposedSDF(s)

        grid = csdf.slice_at_z(0.0, 10, bounds=((-0.15, -0.15), (0.15, 0.15)))
        self.assertEqual(len(grid), 10)
        self.assertEqual(len(grid[0]), 10)
        # Center cell should be negative (inside)
        self.assertLess(grid[5][5], 0)
        # Corner cell should be positive (outside)
        self.assertGreater(grid[0][0], 0)


# ── Structure integration ───────────────────────────────────────────

class TestStructureCSGIntegration(unittest.TestCase):
    """Test that Structure's new CSG methods work correctly."""

    def test_structure_point_inside(self):
        s = Structure(target_volume=0.004)
        s.add(Sphere(0.1), 'iron')
        self.assertTrue(s.point_inside(0, 0, 0))
        self.assertFalse(s.point_inside(0.2, 0, 0))

    def test_structure_material_at(self):
        s = Structure(target_volume=0.001)
        s.add(Cylinder(0.025, 1.0), 'steel')
        s.add(Cylinder(0.020, 1.0), 'air')
        self.assertEqual(s.material_at(0.022, 0, 0), 'steel')
        self.assertEqual(s.material_at(0, 0, 0), 'air')

    def test_composed_sdf_cached(self):
        s = Structure(target_volume=0.004)
        s.add(Sphere(0.1), 'iron')
        csdf1 = s.composed_sdf()
        csdf2 = s.composed_sdf()
        self.assertIs(csdf1, csdf2)  # same object, cached

    def test_cache_invalidated_on_add(self):
        s = Structure(target_volume=0.004)
        s.add(Sphere(0.1), 'iron')
        csdf1 = s.composed_sdf()
        s.add(Sphere(0.05), 'air')
        csdf2 = s.composed_sdf()
        self.assertIsNot(csdf1, csdf2)  # rebuilt after add


# ── Backward compatibility ──────────────────────────────────────────

class TestBackwardCompatibility(unittest.TestCase):
    """Ensure existing Structure usage patterns are unbroken."""

    def test_layers_are_two_tuples(self):
        """Existing code does `for shape, mat in s.layers`."""
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'iron')
        s.add(Cylinder(0.05, 0.2), 'steel')
        for shape, mat in s.layers:
            self.assertIsNotNone(shape)
            self.assertIsInstance(mat, str)

    def test_volume_calculations_unchanged(self):
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'iron')
        vol = (4.0 / 3.0) * math.pi * 0.1 ** 3
        self.assertAlmostEqual(s.used_volume, vol, places=10)
        self.assertAlmostEqual(s.material_volume, vol, places=10)

    def test_material_volume_with_air(self):
        """Air volume correctly subtracted from material volume."""
        s = Structure(target_volume=0.001)
        s.add(Cylinder(0.025, 1.0), 'steel')
        s.add(Cylinder(0.020, 1.0), 'air')
        outer_v = math.pi * 0.025**2 * 1.0
        inner_v = math.pi * 0.020**2 * 1.0
        self.assertAlmostEqual(s.material_volume, outer_v - inner_v, places=10)

    def test_materials_set(self):
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'iron')
        s.add(Sphere(0.05), 'air')
        self.assertEqual(s.materials(), {'iron', 'air'})

    def test_shape_count(self):
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'iron')
        s.add(Sphere(0.05), 'air')
        self.assertEqual(s.shape_count, 2)

    def test_operations_parallel_to_layers(self):
        """_operations list stays in sync with layers."""
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'iron')
        s.add(Sphere(0.05), 'air')
        s.add(Sphere(0.03), 'gold', operation='smooth_union')
        self.assertEqual(len(s._operations), len(s.layers))
        self.assertEqual(s._operations[0], 'add')
        self.assertEqual(s._operations[1], 'add')  # stored as 'add', inferred as subtract in CSG
        self.assertEqual(s._operations[2], 'smooth_union')


# ── Complex composed shapes ─────────────────────────────────────────

class TestComplexShapes(unittest.TestCase):
    """Test CSG with real-world-like compound shapes."""

    def test_bolt_head_and_shank(self):
        """Bolt: cylinder head + cylinder shank, unioned."""
        s = Structure(target_volume=0.0001)
        s.add(Cylinder(0.005, 0.004, center=(0, 0, 0.017)), 'steel')  # head
        s.add(Cylinder(0.003, 0.030, center=(0, 0, 0)), 'steel')      # shank
        csdf = ComposedSDF(s)

        # Inside head
        self.assertTrue(csdf.point_inside(0, 0, 0.017))
        # Inside shank
        self.assertTrue(csdf.point_inside(0, 0, 0))
        # Outside
        self.assertFalse(csdf.point_inside(0.01, 0, 0))

    def test_i_beam_cross_section(self):
        """W-beam: web box + top flange + bottom flange."""
        depth = 0.15
        flange_w = 0.10
        web_t = 0.004
        flange_t = 0.006
        length = 0.5

        s = Structure(target_volume=0.001)
        # Web
        s.add(Box(web_t, length, depth), 'steel')
        # Top flange
        s.add(Box(flange_w, length, flange_t,
                  center=(0, 0, (depth - flange_t) / 2)), 'steel')
        # Bottom flange
        s.add(Box(flange_w, length, flange_t,
                  center=(0, 0, -(depth - flange_t) / 2)), 'steel')
        csdf = ComposedSDF(s)

        # In the web center
        self.assertTrue(csdf.point_inside(0, 0, 0))
        # In the top flange
        self.assertTrue(csdf.point_inside(0.04, 0, 0.07))
        # In the gap between web and flange edge (outside web, outside flange)
        self.assertFalse(csdf.point_inside(0.04, 0, 0.03))

    def test_hollow_sphere_shell(self):
        """Hollow sphere: outer sphere minus inner sphere."""
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.1), 'glass')
        s.add(Sphere(0.09), 'air')

        # In the shell
        self.assertTrue(s.point_inside(0.095, 0, 0))
        # In the cavity
        self.assertFalse(s.point_inside(0, 0, 0))
        # Outside
        self.assertFalse(s.point_inside(0.15, 0, 0))

    def test_multi_material_layers(self):
        """Nested layers: steel outer, copper middle, air core."""
        s = Structure(target_volume=0.001)
        s.add(Sphere(0.10), 'steel')
        s.add(Sphere(0.08), 'copper')
        s.add(Sphere(0.03), 'air')

        # In the steel shell (r=0.08 to r=0.10)
        self.assertEqual(s.material_at(0.09, 0, 0), 'steel')
        # In the copper layer (r=0.03 to r=0.08)
        self.assertEqual(s.material_at(0.05, 0, 0), 'copper')
        # In the air core (r < 0.03)
        self.assertEqual(s.material_at(0, 0, 0), 'air')
        # Outside
        self.assertIsNone(s.material_at(0.15, 0, 0))


if __name__ == '__main__':
    unittest.main()
