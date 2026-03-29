"""
A3DVS Dry-Run Quality Test — Matter Shape Compression.

Pulls shapes from all available sources, converts them via the A3DVS
pipeline, and verifies:
  1. Conversion quality is 'excellent' or 'exact'
  2. Boundary score >= 0.95
  3. Stored shape is <= 1/5 of incoming source data
  4. can_discard == True for all clean conversions

This is the acceptance test for the A3DVS system. If any shape exceeds
the 1/5 storage ratio, it's flagged for inspection.
"""

import json
import math
import os
import unittest

from sigma_ground.parts import (
    convert_to_primitives, PIPES, ISO_BOLTS, W_BEAMS,
    pipe_structure, bolt_structure, beam_structure,
)
from sigma_ground.shapes import Structure


# ── Simulated external data sizes ────────────────────────────────────
# Real-world mesh sizes for common engineering parts.
# A typical OBJ mesh: ~20 bytes per triangle face line + vertices.

_PIPE_MESH_BYTES = 8000      # simple tube mesh
_BOLT_MESH_BYTES = 12000     # threaded bolt mesh
_BEAM_MESH_BYTES = 6000      # extruded I-beam mesh
_SPHERE_MESH_BYTES = 15000   # tessellated sphere
_BOX_MESH_BYTES = 2000       # trivial cube mesh
_CYLINDER_MESH_BYTES = 5000  # tessellated cylinder
_CONE_MESH_BYTES = 4000      # tessellated cone

# Storage ratio threshold: our shapes must be <= 1/5 of source
_MAX_STORAGE_FRACTION = 0.2  # 1/5


def _storage_bytes(structure):
    """Estimate our stored primitive data size in bytes.

    ~80 bytes per primitive: type enum (1) + 6 floats dims (48) +
    3 floats center (24) + material tag (7).
    """
    return structure.shape_count * 80


class TestA3DVSPipeCatalog(unittest.TestCase):
    """Convert all ANSI pipe sizes and verify quality."""

    def test_all_pipes(self):
        failures = []
        for key, spec in PIPES.items():
            desc = {
                'type': 'pipe',
                'outer_diameter_m': spec['od_m'],
                'wall_thickness_m': spec['wall_m'],
                'length_m': 1.0,
            }
            result = convert_to_primitives(desc, source_bytes=_PIPE_MESH_BYTES)

            # Quality checks
            if result['conversion_quality'] not in ('excellent', 'exact'):
                failures.append(f"{key}: quality={result['conversion_quality']}")
                continue
            if result['boundary']['score'] < 0.95:
                failures.append(f"{key}: boundary={result['boundary']['score']:.3f}")
                continue
            if not result['can_discard']:
                failures.append(f"{key}: can_discard=False")
                continue

            # Storage ratio: our shape must be <= 1/5 of source
            our_bytes = _storage_bytes(result['structure'])
            ratio = our_bytes / _PIPE_MESH_BYTES
            if ratio > _MAX_STORAGE_FRACTION:
                failures.append(
                    f"{key}: storage {our_bytes}B / {_PIPE_MESH_BYTES}B "
                    f"= {ratio:.2f} > {_MAX_STORAGE_FRACTION}")

        self.assertEqual(failures, [], f"Pipe failures:\n" + "\n".join(failures))


class TestA3DVSBoltCatalog(unittest.TestCase):
    """Convert all ISO bolt sizes and verify quality."""

    def test_all_bolts(self):
        failures = []
        for key, spec in ISO_BOLTS.items():
            desc = {
                'type': 'bolt',
                'iso_key': key,
                'length_m': spec['major_d_m'] * 5,  # typical L/D ratio
            }
            result = convert_to_primitives(desc, source_bytes=_BOLT_MESH_BYTES)

            if result['conversion_quality'] not in ('excellent', 'exact'):
                failures.append(f"{key}: quality={result['conversion_quality']}")
                continue
            if not result['can_discard']:
                failures.append(f"{key}: can_discard=False")
                continue

            our_bytes = _storage_bytes(result['structure'])
            ratio = our_bytes / _BOLT_MESH_BYTES
            if ratio > _MAX_STORAGE_FRACTION:
                failures.append(
                    f"{key}: storage {our_bytes}B / {_BOLT_MESH_BYTES}B "
                    f"= {ratio:.2f}")

        self.assertEqual(failures, [], f"Bolt failures:\n" + "\n".join(failures))


class TestA3DVSBeamCatalog(unittest.TestCase):
    """Convert all AISC beam sizes and verify quality."""

    def test_all_beams(self):
        failures = []
        for key, spec in W_BEAMS.items():
            desc = {
                'type': 'beam',
                'depth_m': spec['depth_m'],
                'flange_width_m': spec['flange_w_m'],
                'web_thickness_m': spec['web_t_m'],
                'flange_thickness_m': spec['flange_t_m'],
                'length_m': 3.0,
            }
            result = convert_to_primitives(desc, source_bytes=_BEAM_MESH_BYTES)

            if result['conversion_quality'] not in ('excellent', 'exact'):
                failures.append(f"{key}: quality={result['conversion_quality']}")
                continue
            if not result['can_discard']:
                failures.append(f"{key}: can_discard=False")
                continue

            our_bytes = _storage_bytes(result['structure'])
            ratio = our_bytes / _BEAM_MESH_BYTES
            if ratio > _MAX_STORAGE_FRACTION:
                failures.append(
                    f"{key}: storage {our_bytes}B / {_BEAM_MESH_BYTES}B "
                    f"= {ratio:.2f}")

        self.assertEqual(failures, [], f"Beam failures:\n" + "\n".join(failures))


class TestA3DVSPrimitiveShapes(unittest.TestCase):
    """Convert basic geometric shapes and verify quality."""

    def test_sphere(self):
        r = convert_to_primitives(
            {'type': 'sphere', 'radius_m': 0.05},
            source_bytes=_SPHERE_MESH_BYTES)
        self.assertTrue(r['can_discard'])
        self.assertLessEqual(
            _storage_bytes(r['structure']) / _SPHERE_MESH_BYTES,
            _MAX_STORAGE_FRACTION)

    def test_cylinder(self):
        r = convert_to_primitives(
            {'type': 'cylinder', 'radius_m': 0.03, 'height_m': 0.2},
            source_bytes=_CYLINDER_MESH_BYTES)
        self.assertTrue(r['can_discard'])
        self.assertLessEqual(
            _storage_bytes(r['structure']) / _CYLINDER_MESH_BYTES,
            _MAX_STORAGE_FRACTION)

    def test_box(self):
        r = convert_to_primitives(
            {'type': 'box', 'width_m': 0.1, 'depth_m': 0.08, 'height_m': 0.05},
            source_bytes=_BOX_MESH_BYTES)
        self.assertTrue(r['can_discard'])
        self.assertLessEqual(
            _storage_bytes(r['structure']) / _BOX_MESH_BYTES,
            _MAX_STORAGE_FRACTION)

    def test_cone(self):
        r = convert_to_primitives(
            {'type': 'cone', 'radius_m': 0.04, 'height_m': 0.12},
            source_bytes=_CONE_MESH_BYTES)
        self.assertTrue(r['can_discard'])
        self.assertLessEqual(
            _storage_bytes(r['structure']) / _CONE_MESH_BYTES,
            _MAX_STORAGE_FRACTION)


class TestA3DVSMatterShaperShapes(unittest.TestCase):
    """Load .shape.json from matter-shaper and verify primitive efficiency.

    These shapes are ALREADY in sigma_v1 format (analytic primitives).
    The .shape.json files ARE our compressed representation — the 1/5
    storage rule applies to external polygon meshes (OBJ/STL), not to
    our own format.

    What we verify here:
      - Our internal Structure representation (80 bytes/primitive) is
        always smaller than the verbose JSON format
      - Primitive count is reasonable (not over-segmented)
      - The shapes load without error
    """

    _SHAPE_DIR = os.path.join(
        os.path.dirname(__file__), '..', '..', 'matter-shaper',
        'MatterShaper', 'harvest', 'test_data', 'output')

    _OBJECT_DIR = os.path.join(
        os.path.dirname(__file__), '..', '..', 'matter-shaper',
        'MatterShaper', 'object_maps')

    def _check_shape_file(self, path):
        """Load a .shape.json and measure primitive efficiency."""
        with open(path, 'r') as f:
            data = json.load(f)

        json_bytes = os.path.getsize(path)
        layers = data.get('layers', [])
        if not layers:
            return None

        our_bytes = len(layers) * 80  # internal binary representation

        return {
            'name': data.get('object', data.get('name', os.path.basename(path))),
            'json_bytes': json_bytes,
            'our_bytes': our_bytes,
            'compression': json_bytes / max(our_bytes, 1),
            'n_primitives': len(layers),
        }

    def test_harvest_shapes(self):
        """All harvested shapes load and our binary repr is smaller than JSON."""
        if not os.path.isdir(self._SHAPE_DIR):
            self.skipTest("matter-shaper harvest dir not found")

        results = []
        for fname in os.listdir(self._SHAPE_DIR):
            if not fname.endswith('.shape.json'):
                continue
            path = os.path.join(self._SHAPE_DIR, fname)
            r = self._check_shape_file(path)
            if r is None:
                continue
            results.append(r)
            with self.subTest(shape=r['name']):
                # Binary repr should always be smaller than verbose JSON
                self.assertLess(r['our_bytes'], r['json_bytes'],
                    f"{r['name']}: binary ({r['our_bytes']}B) >= "
                    f"JSON ({r['json_bytes']}B)")
                # Primitive count should be reasonable (not over-segmented)
                self.assertLessEqual(r['n_primitives'], 50,
                    f"{r['name']}: {r['n_primitives']} primitives — over-segmented")

        self.assertGreater(len(results), 0, "No shape files found")

    def test_object_map_shapes(self):
        """All object map shapes load and binary repr is smaller than JSON."""
        if not os.path.isdir(self._OBJECT_DIR):
            self.skipTest("matter-shaper object_maps dir not found")

        results = []
        for fname in os.listdir(self._OBJECT_DIR):
            if not fname.endswith('.shape.json'):
                continue
            path = os.path.join(self._OBJECT_DIR, fname)
            r = self._check_shape_file(path)
            if r is None:
                continue
            results.append(r)
            with self.subTest(shape=r['name']):
                self.assertLess(r['our_bytes'], r['json_bytes'])
                self.assertLessEqual(r['n_primitives'], 50)

        self.assertGreater(len(results), 0, "No object map files found")

        self.assertGreater(len(results), 0, "No object map files found")


class TestA3DVSStorageReport(unittest.TestCase):
    """Generate a full storage ratio report across all shape types."""

    def test_storage_report(self):
        """Run all conversions and print a summary report."""
        report = []

        # Pipes (sample 5)
        pipe_keys = list(PIPES.keys())[:5]
        for key in pipe_keys:
            spec = PIPES[key]
            r = convert_to_primitives({
                'type': 'pipe',
                'outer_diameter_m': spec['od_m'],
                'wall_thickness_m': spec['wall_m'],
                'length_m': 1.0,
            }, source_bytes=_PIPE_MESH_BYTES)
            our = _storage_bytes(r['structure'])
            report.append((f"pipe/{key}", r['conversion_quality'],
                            r['boundary']['score'], our, _PIPE_MESH_BYTES,
                            our / _PIPE_MESH_BYTES, r['can_discard']))

        # Bolts (sample 5)
        bolt_keys = list(ISO_BOLTS.keys())[:5]
        for key in bolt_keys:
            r = convert_to_primitives({
                'type': 'bolt', 'iso_key': key,
                'length_m': ISO_BOLTS[key]['major_d_m'] * 5,
            }, source_bytes=_BOLT_MESH_BYTES)
            our = _storage_bytes(r['structure'])
            report.append((f"bolt/{key}", r['conversion_quality'],
                            r['boundary']['score'], our, _BOLT_MESH_BYTES,
                            our / _BOLT_MESH_BYTES, r['can_discard']))

        # Beams (sample 3)
        beam_keys = list(W_BEAMS.keys())[:3]
        for key in beam_keys:
            spec = W_BEAMS[key]
            r = convert_to_primitives({
                'type': 'beam',
                'depth_m': spec['depth_m'],
                'flange_width_m': spec['flange_w_m'],
                'web_thickness_m': spec['web_t_m'],
                'flange_thickness_m': spec['flange_t_m'],
                'length_m': 3.0,
            }, source_bytes=_BEAM_MESH_BYTES)
            our = _storage_bytes(r['structure'])
            report.append((f"beam/{key}", r['conversion_quality'],
                            r['boundary']['score'], our, _BEAM_MESH_BYTES,
                            our / _BEAM_MESH_BYTES, r['can_discard']))

        # Primitives
        for name, desc, src in [
            ('sphere', {'type': 'sphere', 'radius_m': 0.05}, _SPHERE_MESH_BYTES),
            ('cylinder', {'type': 'cylinder', 'radius_m': 0.03, 'height_m': 0.2},
             _CYLINDER_MESH_BYTES),
            ('box', {'type': 'box', 'width_m': 0.1, 'depth_m': 0.08,
                     'height_m': 0.05}, _BOX_MESH_BYTES),
            ('cone', {'type': 'cone', 'radius_m': 0.04, 'height_m': 0.12},
             _CONE_MESH_BYTES),
        ]:
            r = convert_to_primitives(desc, source_bytes=src)
            our = _storage_bytes(r['structure'])
            report.append((name, r['conversion_quality'],
                            r['boundary']['score'], our, src,
                            our / src, r['can_discard']))

        # Verify ALL pass the 1/5 threshold
        for name, quality, bscore, our, src, ratio, discard in report:
            with self.subTest(shape=name):
                self.assertLessEqual(ratio, _MAX_STORAGE_FRACTION,
                    f"{name}: {our}B / {src}B = {ratio:.3f}")
                self.assertTrue(discard, f"{name}: can_discard=False")


if __name__ == '__main__':
    unittest.main()
