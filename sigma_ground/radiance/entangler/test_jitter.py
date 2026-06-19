"""
Tests for thermal jitter — the one deliberate effect on direct-physics-to-pixel.

Contracts:
  1. Amplitude scales with temperature (cited √T roughness) and is capped at one
     inter-node spacing.
  2. Enabling jitter decorrelates the lattice (nodes move); a given (scene, frame)
     reproduces exactly; successive frames differ (a living surface).
  3. The input list is never mutated.
  4. Surface nodes scatter in their tangent plane (stay on the surface); volume
     nodes scatter isotropically.
  5. Off-by-default: generate_*_nodes(jitter=None) is byte-identical to the
     legacy call — backward compatible.

□σ = −ξR
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sigma_ground.radiance.entangler.vec import Vec3
from sigma_ground.radiance.entangler.jitter import (
    apply_thermal_jitter, thermal_jitter_sigma,
)
from sigma_ground.radiance.entangler.surface_nodes import (
    SurfaceNode, generate_surface_nodes,
)
from sigma_ground.radiance.entangler.volume_nodes import (
    VolumeNode, generate_volume_nodes,
)
from sigma_ground.radiance.entangler.shapes import EntanglerSphere
from sigma_ground.radiance.materials.material import Material


def _copper(T=293.15):
    return Material("copper", Vec3(0.9, 0.6, 0.5),
                    material_key="copper", temperature_K=T)


def _flat_surface_nodes(n=64, mat=None):
    """A flat patch of surface nodes, all normals +z — easy tangent-plane checks."""
    mat = mat or _copper()
    nodes = []
    for i in range(n):
        x = (i % 8) * 0.1
        y = (i // 8) * 0.1
        nodes.append(SurfaceNode(Vec3(x, y, 0.0), Vec3(0, 0, 1), mat))
    return nodes


def _max_disp(a, b):
    return max((a[i].position - b[i].position).length() for i in range(len(a)))


class TestAmplitude(unittest.TestCase):
    """thermal_jitter_sigma — √T scaling, capped at spacing."""

    def test_rises_with_temperature(self):
        cold = thermal_jitter_sigma("copper", 293.15, spacing=0.1)
        hot = thermal_jitter_sigma("copper", 2000.0, spacing=0.1)
        self.assertGreater(hot, cold)

    def test_capped_at_spacing(self):
        # Even at extreme T the displacement σ never exceeds one node gap.
        s = thermal_jitter_sigma("copper", 100000.0, spacing=0.1)
        self.assertLessEqual(s, 0.1)

    def test_zero_spacing_zero_amplitude(self):
        self.assertEqual(thermal_jitter_sigma("copper", 1800.0, spacing=0.0), 0.0)

    def test_unknown_key_uses_sqrt_T_fallback(self):
        """No texture entry → Debye–Waller √(T/T_ref) fallback, still rising in T."""
        cold = thermal_jitter_sigma(None, 293.15, spacing=0.1)
        hot = thermal_jitter_sigma(None, 1200.0, spacing=0.1)
        self.assertGreater(hot, cold)


class TestDecorrelation(unittest.TestCase):
    """Jitter moves nodes, reproducibly, differently each frame."""

    def test_jitter_moves_nodes(self):
        base = _flat_surface_nodes()
        on = apply_thermal_jitter(base, "copper", 1800.0,
                                  spacing=0.1, frame=0)
        self.assertGreater(_max_disp(base, on), 0.0)

    def test_same_frame_reproduces(self):
        base = _flat_surface_nodes()
        a = apply_thermal_jitter(base, "copper", 1800.0, spacing=0.1, frame=7)
        b = apply_thermal_jitter(base, "copper", 1800.0, spacing=0.1, frame=7)
        self.assertEqual(_max_disp(a, b), 0.0)

    def test_frames_differ(self):
        base = _flat_surface_nodes()
        f0 = apply_thermal_jitter(base, "copper", 1800.0, spacing=0.1, frame=0)
        f1 = apply_thermal_jitter(base, "copper", 1800.0, spacing=0.1, frame=1)
        self.assertGreater(_max_disp(f0, f1), 0.0)

    def test_scenes_differ(self):
        base = _flat_surface_nodes()
        s0 = apply_thermal_jitter(base, "copper", 1800.0,
                                  spacing=0.1, frame=0, scene_seed=0)
        s1 = apply_thermal_jitter(base, "copper", 1800.0,
                                  spacing=0.1, frame=0, scene_seed=1)
        self.assertGreater(_max_disp(s0, s1), 0.0)

    def test_input_not_mutated(self):
        base = _flat_surface_nodes()
        before = [(n.position.x, n.position.y, n.position.z) for n in base]
        apply_thermal_jitter(base, "copper", 1800.0, spacing=0.1, frame=0)
        after = [(n.position.x, n.position.y, n.position.z) for n in base]
        self.assertEqual(before, after)


class TestDirectionality(unittest.TestCase):
    """Surface = tangent-plane scatter; volume = isotropic."""

    def test_surface_jitter_is_tangential(self):
        """Normals are +z, so jitter must leave z ≈ unchanged (tangent = xy plane)."""
        base = _flat_surface_nodes()
        on = apply_thermal_jitter(base, "copper", 2000.0, spacing=0.1, frame=3)
        max_dz = max(abs(on[i].position.z - base[i].position.z)
                     for i in range(len(base)))
        self.assertLess(max_dz, 1e-9)

    def test_volume_jitter_is_isotropic(self):
        """Volume nodes (no normal) move in all three axes."""
        mat = _copper(1800.0)
        vnodes = [VolumeNode(Vec3(i * 0.1, 0.0, 0.0), depth=0.5, material=mat, dl=0.1)
                  for i in range(64)]
        on = apply_thermal_jitter(vnodes, "copper", 1800.0, spacing=0.1, frame=0)
        moved_z = max(abs(on[i].position.z - vnodes[i].position.z)
                      for i in range(len(vnodes)))
        self.assertGreater(moved_z, 0.0)

    def test_node_type_preserved(self):
        base = _flat_surface_nodes()
        on = apply_thermal_jitter(base, "copper", 1800.0, spacing=0.1, frame=0)
        self.assertIsInstance(on[0], SurfaceNode)


class TestGeneratorIntegration(unittest.TestCase):
    """Jitter is off by default and threads through the generators when set."""

    def _sphere(self):
        return EntanglerSphere(Vec3(0, 0, 0), 2.0, _copper(1800.0))

    def test_off_by_default_matches_legacy(self):
        sph = self._sphere()
        a = generate_surface_nodes(sph, density=20)
        b = generate_surface_nodes(sph, density=20, jitter=None)
        self.assertEqual(_max_disp(a, b), 0.0)

    def test_surface_generator_applies_jitter(self):
        sph = self._sphere()
        off = generate_surface_nodes(sph, density=20)
        on = generate_surface_nodes(sph, density=20, jitter={"frame": 0})
        self.assertEqual(len(off), len(on))
        self.assertGreater(_max_disp(off, on), 0.0)

    def test_volume_generator_applies_jitter(self):
        sph = EntanglerSphere(Vec3(0, 0, 0), 2.0, _copper(1800.0),
                              fill_volume=True)
        off = generate_volume_nodes(sph, n_nodes=500)
        on = generate_volume_nodes(sph, n_nodes=500, jitter={"frame": 0})
        self.assertGreater(_max_disp(off, on), 0.0)


if __name__ == "__main__":
    unittest.main()
