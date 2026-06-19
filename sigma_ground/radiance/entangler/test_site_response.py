"""
Tests for site_response — direct physics → pixel (the deterministic-mean core).

A lattice node answers a ray from CITED physics, not a stored colour:
  1. Cold reflectance — emergent copper from measured n+k (optics.metal_rgb).
  2. Planck glow — colour from Planck×Kirchhoff, gated at the Draper point (~798 K),
     brightness a tonemap of the cited εσT⁴ power; red→white as T climbs.
  3. Every returned field carries its cited source string.
  4. Non-metals return cold_rgb None (caller keeps its own colour) — no crash.

These lock the contract the entangler shade path depends on.

□σ = −ξR
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sigma_ground.radiance.entangler.site_response import site_response


class TestColdReflectance(unittest.TestCase):
    """Cold copper is emergent (measured n+k), warm-toned, never a chosen RGB."""

    def test_copper_cold_is_warm(self):
        r = site_response("copper", 293.15)
        cold = r["cold_rgb"]
        self.assertIsNotNone(cold, "copper must ground to a metal RGB")
        # Copper reflects red > green > blue — the emergent warm tone.
        self.assertGreater(cold[0], cold[1])
        self.assertGreater(cold[1], cold[2])
        # All channels in gamut.
        for c in cold:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_cold_is_temperature_stable(self):
        """EM reflectance is σ- and (cold-)T-invariant: the hue doesn't drift
        with temperature below incandescence — only the added glow does."""
        a = site_response("copper", 293.15)["cold_rgb"]
        b = site_response("copper", 500.0)["cold_rgb"]
        self.assertEqual(a, b)

    def test_non_metal_returns_none_cold(self):
        """A key optics can't ground returns cold_rgb None (no raise); the caller
        then keeps its own colour. Evaluated cold (293 K) so no glow is computed."""
        r = site_response("definitely_not_a_material", 293.15)
        self.assertIsNone(r["cold_rgb"])
        self.assertFalse(r["glowing"])


class TestPlanckGlow(unittest.TestCase):
    """Incandescence is gated at the Draper point and ramps red→white."""

    def test_cold_does_not_glow(self):
        r = site_response("copper", 293.15)
        self.assertFalse(r["glowing"])
        self.assertEqual(r["glow_level"], 0.0)
        self.assertEqual(r["glow_rgb"], (0.0, 0.0, 0.0))

    def test_draper_gate(self):
        """Draper point (~700 K rendering threshold): below is dark, above glows."""
        self.assertFalse(site_response("copper", 650.0)["glowing"])
        self.assertTrue(site_response("copper", 750.0)["glowing"])

    def test_glow_power_rises_with_T(self):
        """εσT⁴ — emitted power is strictly increasing in T."""
        p1 = site_response("copper", 1000.0)["glow_power"]
        p2 = site_response("copper", 1800.0)["glow_power"]
        p3 = site_response("copper", 3000.0)["glow_power"]
        self.assertGreater(p2, p1)
        self.assertGreater(p3, p2)

    def test_glow_level_rises_with_T(self):
        l1 = site_response("copper", 1000.0)["glow_level"]
        l2 = site_response("copper", 1800.0)["glow_level"]
        l3 = site_response("copper", 3000.0)["glow_level"]
        self.assertLess(l1, l2)
        self.assertLessEqual(l2, l3)
        for lv in (l1, l2, l3):
            self.assertGreaterEqual(lv, 0.0)
            self.assertLessEqual(lv, 1.0)

    def test_red_to_white_ramp(self):
        """Dull red at 1000 K → whiter (more blue/green) by 3000 K."""
        g1 = site_response("copper", 1000.0)["glow_rgb"]
        g2 = site_response("copper", 1800.0)["glow_rgb"]
        g3 = site_response("copper", 3000.0)["glow_rgb"]
        # Red is present from the start.
        self.assertGreater(g1[0], 0.0)
        # Green fills in as it heats.
        self.assertGreater(g2[1], g1[1])
        # Blue (the "white" tail) only appears at the hot end.
        self.assertGreater(g3[2], g1[2])


class TestProvenance(unittest.TestCase):
    """Every value the shade uses traces to a cited field function."""

    def test_sources_present(self):
        r = site_response("copper", 1800.0)
        src = r["sources"]
        self.assertIn("optics.metal_rgb", src["cold_rgb"])
        self.assertIn("thermal", src["glow"])
        self.assertIn("Rayleigh", src["specular_fraction"])

    def test_specular_fraction_grounded(self):
        """Specular fraction comes from the Rayleigh/roughness path (a real number
        for a metal in the texture DB), exposed for the shade."""
        r = site_response("copper", 293.15)
        self.assertIsNotNone(r["specular_fraction"])
        self.assertGreaterEqual(r["specular_fraction"], 0.0)
        self.assertLessEqual(r["specular_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
