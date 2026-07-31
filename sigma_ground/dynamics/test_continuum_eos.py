"""Phase E3: the isothermal (velocity-dispersion) EOS for dilute matter.

Found live (2026-07-30): the Tait EOS with condensed-matter K extrapolates
a dilute swarm's kernel density (~0.24 kg/m³ vs iron's 7874) to ~ −1.7e11
Pa of spurious tension, ejecting the first pair that shears inside the
kernel radius. The fix is the standard ring/disk closure — P = ρ·c_s² with
c_s the velocity dispersion (Goldreich & Tremaine 1978) — selected per
scene, never a clamp on Tait. These tests pin the law, the guards, the
no-tension property, and the ring smoke run that originally exploded.
"""
import math
import unittest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import (
    ContinuumParcel, ContinuumScene, continuum_step,
    _pressure_isothermal,
)
from sigma_ground.dynamics.seeding import (
    G_CODATA, seed_ring_annulus, to_nbody, to_continuum,
)

ZERO_G = Vec3(0.0, 0.0, 0.0)
RING = dict(central_mass_kg=5.0e15, central_radius_m=800.0,
            r_inner_m=6000.0, r_outer_m=12000.0, ring_mass_kg=1.0e9,
            particle_density_kg_m3=920.0)


class TestEosSelection(unittest.TestCase):
    def _pair(self):
        return [ContinuumParcel('iron', 0.01, position=Vec3(0, 0, 0)),
                ContinuumParcel('iron', 0.01, position=Vec3(1, 0, 0))]

    def test_default_is_tait(self):
        scene = ContinuumScene(self._pair(), gravity=ZERO_G, ground=False)
        self.assertEqual(scene.eos, "tait")
        self.assertIsNone(scene.sound_speed_m_s)

    def test_unknown_eos_rejected(self):
        with self.assertRaises(ValueError):
            ContinuumScene(self._pair(), gravity=ZERO_G, ground=False,
                           eos="polytrope")

    def test_isothermal_requires_sound_speed(self):
        with self.assertRaises(ValueError):
            ContinuumScene(self._pair(), gravity=ZERO_G, ground=False,
                           eos="isothermal")

    def test_isothermal_law_and_no_tension(self):
        """P = ρ·c_s² exactly, and P ≥ 0 no matter how dilute — the
        property whose absence blew up the ring."""
        p = ContinuumParcel('iron', 0.01, position=Vec3(0, 0, 0))
        p.sph_density = 0.244            # the actual dilute value found live
        P = _pressure_isothermal(p, 2.0)
        self.assertAlmostEqual(P, 0.244 * 4.0)
        self.assertGreaterEqual(P, 0.0)
        self.assertEqual(p.pressure, P)


class TestRingSurvives(unittest.TestCase):
    def test_ring_smoke_both_engines_hold_the_annulus(self):
        """The exact scenario that exploded under Tait: 30-particle ring,
        50 steps (~a quarter mid-ring orbit), both engines. With the
        isothermal EOS at c_s = the seeded dispersion scale, both sides
        keep every particle inside a sane annulus and agree closely (SPH
        pressure is a small correction at these separations, not a bomb)."""
        bodies = seed_ring_annulus(30, seed=42, **RING)
        r_mid = 9000.0
        T = 2 * math.pi * math.sqrt(r_mid ** 3
                                    / (G_CODATA * RING["central_mass_kg"]))
        dt = T / 200.0

        sysnb = to_nbody(bodies, softening_m=50.0)
        for _ in range(50):
            sysnb.step(dt)
        r_nb = [math.hypot(*b.position_m[:2]) for b in sysnb.bodies[1:]]

        # c_s: a disclosed Choice at the dispersion scale — 1% of the
        # mid-ring circular speed, the same order the seeder would inject
        v_kep = math.sqrt(G_CODATA * RING["central_mass_kg"] / r_mid)
        scene = to_continuum(bodies, material_key='iron', G=G_CODATA,
                             gravity_softening_m=50.0,
                             eos="isothermal",
                             sound_speed_m_s=0.01 * v_kep)
        for _ in range(50):
            continuum_step(scene, dt)
        r_sph = [math.hypot(p.position.x, p.position.y)
                 for p in scene.parcels[1:]]

        for name, rr in (("nbody", r_nb), ("sph", r_sph)):
            self.assertTrue(all(math.isfinite(x) for x in rr), name)
            self.assertGreater(min(rr), 4000.0, name)
            self.assertLess(max(rr), 16000.0, name)
        # cross-engine agreement: gravity dominates at these separations,
        # so the two descriptions may only differ by a small fraction of
        # the annulus width this early
        drift = max(abs(a - b) for a, b in zip(r_nb, r_sph))
        self.assertLess(drift, 600.0, f"cross-engine drift {drift:.1f} m")


if __name__ == "__main__":
    unittest.main()
