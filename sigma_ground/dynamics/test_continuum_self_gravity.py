"""Phase E1: pairwise Newtonian self-gravity in the SPH continuum bridge.

The fluid side of the fluid-vs-gravity comparison (Phase E) needs the fluid
to gravitate. These tests pin:

  1. OFF by default — self_gravity_G=None reproduces pre-Phase-E behavior
     (the accumulator is never built, accelerations are untouched).
  2. The force law — two well-separated parcels attract with a = G·m/r²
     (checked against a hand-computed value, far outside both the SPH
     kernel radius and the softening length so no other term contributes).
  3. Symmetry — Newton's third law: total momentum stays ~zero for an
     isolated pair starting at rest.
  4. Energy — a bound two-body pair integrated over many steps conserves
     E = KE + PE to integrator tolerance (leapfrog is symplectic; drift
     should be bounded, not secular).

G here is scaled up (a disclosed test choice) so that centimeter-scale
parcels produce measurable accelerations without waiting geological time —
the force LAW is what's under test, and it is linear in G.
"""
import math
import unittest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import (
    ContinuumParcel,
    ContinuumScene,
    continuum_step,
    _sph_density_sum,
)

G_REAL = 6.674e-11        # CODATA 2018, m^3 kg^-1 s^-2
ZERO_G = Vec3(0.0, 0.0, 0.0)


def _pair(sep_m=1.0, radius=0.01):
    """Two iron parcels at rest, ``sep_m`` apart on the x-axis."""
    a = ContinuumParcel('iron', radius, position=Vec3(0.0, 0.0, 0.0),
                        velocity=Vec3(0, 0, 0), label='a')
    b = ContinuumParcel('iron', radius, position=Vec3(sep_m, 0.0, 0.0),
                        velocity=Vec3(0, 0, 0), label='b')
    return [a, b]


class TestOffByDefault(unittest.TestCase):
    def test_default_scene_has_no_self_gravity(self):
        scene = ContinuumScene(_pair(), gravity=ZERO_G, ground=False)
        self.assertIsNone(scene.self_gravity_G)

    def test_density_sum_returns_none_when_gravity_off(self):
        parcels = _pair()
        scene = ContinuumScene(parcels, gravity=ZERO_G, ground=False)
        self.assertIsNone(_sph_density_sum(parcels, scene.h))

    def test_parcels_at_rest_stay_at_rest_without_self_gravity(self):
        """Two separated parcels, no background g, no self-gravity: nothing
        moves — the pre-Phase-E regression."""
        parcels = _pair(sep_m=1.0)
        scene = ContinuumScene(parcels, gravity=ZERO_G, ground=False)
        for _ in range(50):
            continuum_step(scene, 1e-3)
        for p in parcels:
            self.assertEqual((p.velocity.x, p.velocity.y, p.velocity.z),
                             (0.0, 0.0, 0.0), p.label)


class TestForceLaw(unittest.TestCase):
    def test_two_body_acceleration_matches_newton(self):
        """a = G·m_other/r² at 1 m separation, far outside the kernel
        radius (h ~ cm scale) and softening — pure gravity, hand-checked."""
        parcels = _pair(sep_m=1.0)
        scene = ContinuumScene(parcels, gravity=ZERO_G, ground=False,
                               self_gravity_G=G_REAL,
                               gravity_softening_m=0.0)
        grav = _sph_density_sum(parcels, scene.h, gravity_G=G_REAL,
                                softening=0.0)
        m = parcels[1].mass
        expected = G_REAL * m / 1.0 ** 2
        self.assertAlmostEqual(grav[0].x, expected, delta=abs(expected) * 1e-9)
        self.assertAlmostEqual(grav[1].x, -expected, delta=abs(expected) * 1e-9)
        self.assertEqual((grav[0].y, grav[0].z), (0.0, 0.0))

    def test_newtons_third_law_momentum_stays_zero(self):
        """Isolated pair from rest under strong scaled gravity: total
        momentum must remain ~0 while the parcels visibly accelerate."""
        parcels = _pair(sep_m=1.0)
        G_scaled = 1.0e-3                       # disclosed test scaling
        scene = ContinuumScene(parcels, gravity=ZERO_G, ground=False,
                               self_gravity_G=G_scaled)
        for _ in range(100):
            continuum_step(scene, 1e-3)
        px = sum(p.mass * p.velocity.x for p in parcels)
        speed = max(abs(p.velocity.x) for p in parcels)
        self.assertGreater(speed, 0.0)          # gravity actually acted
        self.assertLess(abs(px), 1e-12 * max(p.mass for p in parcels))

    def test_parcels_fall_toward_each_other(self):
        parcels = _pair(sep_m=1.0)
        scene = ContinuumScene(parcels, gravity=ZERO_G, ground=False,
                               self_gravity_G=1.0e-3)
        r0 = parcels[1].position.x - parcels[0].position.x
        for _ in range(200):
            continuum_step(scene, 1e-3)
        r1 = parcels[1].position.x - parcels[0].position.x
        self.assertLess(r1, r0)


class TestEnergyConservation(unittest.TestCase):
    def test_bound_orbit_conserves_energy(self):
        """A circular two-body orbit (v_circ = sqrt(G·m_other²/(M·r)) about
        the barycenter) stepped for several periods: E = KE + PE drifts by
        < 1% — leapfrog is symplectic, so error is bounded, not secular."""
        parcels = _pair(sep_m=1.0)
        m = parcels[0].mass                     # equal masses
        G_s = 1.0e-3
        # circular orbit about the barycenter at r/2: v² = G·m/(4·(r/2))·...
        # For equal masses at separation r: v_each = sqrt(G·m/(2·r))·... do it
        # from first principles: a = G·m/r², circular about center needs
        # v²/(r/2) = G·m/r²  →  v = sqrt(G·m/(2·r))
        v = math.sqrt(G_s * m / 2.0)
        parcels[0].velocity = Vec3(0.0, +v, 0.0)
        parcels[1].velocity = Vec3(0.0, -v, 0.0)
        scene = ContinuumScene(parcels, gravity=ZERO_G, ground=False,
                               self_gravity_G=G_s, gravity_softening_m=0.0)

        def energy():
            ke = sum(0.5 * p.mass * (p.velocity.x ** 2 + p.velocity.y ** 2
                                     + p.velocity.z ** 2) for p in parcels)
            dx = parcels[1].position.x - parcels[0].position.x
            dy = parcels[1].position.y - parcels[0].position.y
            dz = parcels[1].position.z - parcels[0].position.z
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            pe = -G_s * parcels[0].mass * parcels[1].mass / r
            return ke + pe

        e0 = energy()
        period = 2 * math.pi * 0.5 / v          # circumference/speed at r/2
        steps = 2000
        dt = (3 * period) / steps               # three full orbits
        for _ in range(steps):
            continuum_step(scene, dt)
        e1 = energy()
        self.assertLess(abs(e1 - e0), abs(e0) * 0.01,
                        f"energy drifted {e0} -> {e1}")
        # and the pair is still bound at roughly the same separation
        dx = parcels[1].position.x - parcels[0].position.x
        dy = parcels[1].position.y - parcels[0].position.y
        r = math.sqrt(dx * dx + dy * dy)
        self.assertGreater(r, 0.5)
        self.assertLess(r, 2.0)


class TestSoftening(unittest.TestCase):
    def test_softening_keeps_close_pair_finite(self):
        """Overlapping parcels with Plummer softening: acceleration stays
        finite and well below the unsoftened 1/r² blow-up."""
        parcels = _pair(sep_m=1e-6)             # essentially on top of each other
        scene = ContinuumScene(parcels, gravity=ZERO_G, ground=False,
                               self_gravity_G=G_REAL)
        eps = scene.gravity_softening_m
        self.assertGreater(eps, 0.0)            # default 0.5·h engaged
        grav = _sph_density_sum(parcels, scene.h, gravity_G=G_REAL,
                                softening=eps)
        m = parcels[1].mass
        cap = G_REAL * m / eps ** 2             # softened ceiling scale
        self.assertTrue(math.isfinite(grav[0].x))
        self.assertLess(abs(grav[0].x), cap)


if __name__ == "__main__":
    unittest.main()
