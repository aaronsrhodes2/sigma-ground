"""Regression sentinels for the N-body 1PN general-relativity layers (fixed 2026-10-10).

Before the fix, `gr_1pn` applied minus the textbook 1PN correction and `eih_cross` had its velocity cross-term reversed,
so Mercury's perihelion moved −43″/cy and −72″/cy. The old tests asserted only that GR moved Mercury "by more than zero",
so neither sign nor size was ever checked (unification round 2, U9; KNOWN_GAPS.md). These tests pin both:

1. the engine's correction vectors equal the textbook formulas, written out independently here;
2. Mercury's perihelion advances by 6πGM/(c²a(1−e²)) per orbit (≈ +43″/cy) for both layers;
3. the many-body EIH layer gives the same precession in a boosted frame (post-Galilean invariance);
4. the orbit adapters do not switch GR on unless asked.

References: the β = γ = 1 PPN test-particle equation of motion, a = −GMn/r² + (GM/c²r²)[(4GM/r − v²)n + 4(n·v)v] with n
pointing from the attractor to the body (Iorio, arXiv:gr-qc/0405053, eq. 1; Will 2014, Living Rev. Relativ. 17, 4);
the Einstein–Infeld–Hoffmann equations in the form used for solar-system ephemerides (Will 1993, Theory and Experiment in
Gravitational Physics; Moyer 2003, JPL Monograph 2).
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.constants import C as _C, G as _G
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

M_SUN = 1.98892e30          # kg (Mentat solar_system_body('sun'))
M_MERCURY = 3.3011e23       # kg
A_MERCURY = 5.7909e10       # m, semimajor axis [LIT, NASA fact sheet]
E_MERCURY = 0.2056          # eccentricity [LIT, NASA fact sheet]


def _textbook_1pn(gm: float, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Test-particle 1PN correction; x and v are the body's position and velocity relative to the attractor."""
    r = float(np.linalg.norm(x))
    n = x / r                                    # outward, attractor -> body
    return gm / (_C**2 * r**2) * ((4.0 * gm / r - float(v @ v)) * n + 4.0 * float(n @ v) * v)


def _textbook_eih_corrections(gms, xs, vs) -> np.ndarray:
    """1PN EIH acceleration minus the Newtonian one, for every body, written out from the textbook form."""
    n = len(gms); c2 = _C**2
    a_n = np.zeros((n, 3)); phi = np.zeros(n)
    for i in range(n):
        for k in range(n):
            if k != i:
                d = xs[k] - xs[i]; r = float(np.linalg.norm(d))
                a_n[i] += gms[k] * d / r**3
                phi[i] += gms[k] / r
    out = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            xij = xs[i] - xs[j]; r = float(np.linalg.norm(xij))
            A = (-4.0 * phi[i] / c2 - phi[j] / c2 + float(vs[i] @ vs[i]) / c2 + 2.0 * float(vs[j] @ vs[j]) / c2
                 - 4.0 * float(vs[i] @ vs[j]) / c2 - 1.5 * (float(xij @ vs[j]) / r) ** 2 / c2
                 + 0.5 * float((-xij) @ a_n[j]) / c2)
            out[i] += gms[j] * (-xij) / r**3 * A
            out[i] += gms[j] / (c2 * r**3) * float(xij @ (4.0 * vs[i] - 3.0 * vs[j])) * (vs[i] - vs[j])
            out[i] += 3.5 / c2 * gms[j] * a_n[j] / r
    return out


def _mercury_pair(boost: np.ndarray | None = None) -> list[CelestialBody]:
    boost = np.zeros(3) if boost is None else boost
    mu = _G * (M_SUN + M_MERCURY)
    rp = A_MERCURY * (1 - E_MERCURY)
    vp = math.sqrt(mu * (1 + E_MERCURY) / rp)
    return [CelestialBody(M_SUN, np.zeros(3), boost.copy(), 6.957e8, 0.0),
            CelestialBody(M_MERCURY, np.array([rp, 0.0, 0.0]), np.array([0.0, vp, 0.0]) + boost, 2.44e6, 0.0)]


def _perihelion_advance_per_orbit(toggles: PhysicsToggles, boost: np.ndarray | None = None,
                                  n_orbits: int = 2, dt: float = 1200.0) -> float:
    """Orbit-averaged advance of the Laplace–Runge–Lenz angle per orbit, minus the Newtonian run's (removes
    integrator precession). Averaging over whole orbits removes the short-period 1PN oscillation."""
    mu = _G * (M_SUN + M_MERCURY)
    period = 2 * math.pi * math.sqrt(A_MERCURY**3 / mu)
    per_orbit = int(round(period / dt)); n_steps = n_orbits * per_orbit

    def lrl_angles(tog):
        system = NBodySystem(_mercury_pair(boost), toggles=tog)
        out = np.empty(n_steps)
        for k in range(n_steps):
            system.forest_ruth_step(dt)
            r = system.bodies[1].position_m - system.bodies[0].position_m
            v = system.bodies[1].velocity_m_s - system.bodies[0].velocity_m_s
            A = np.cross(v, np.cross(r, v)) - mu * r / np.linalg.norm(r)
            out[k] = math.atan2(A[1], A[0])
        return np.unwrap(out)

    d = lrl_angles(toggles) - lrl_angles(PhysicsToggles())
    return float((d[-per_orbit:].mean() - d[:per_orbit].mean()) / (n_orbits - 1))


def _gr_advance_per_orbit() -> float:
    mu = _G * (M_SUN + M_MERCURY)
    return 6 * math.pi * mu / (_C**2 * A_MERCURY * (1 - E_MERCURY**2))


class TestGRCorrectionVectors(unittest.TestCase):
    """The engine's 1PN accelerations equal the textbook formulas (fast; no integration)."""

    def test_gr_1pn_equals_textbook_test_particle_law(self):
        rng = np.random.default_rng(7)
        for _ in range(8):
            x = rng.normal(size=3) * 0.4 * 1.496e11
            v = rng.normal(size=3) * 4.0e4
            bodies = [CelestialBody(M_SUN, np.zeros(3), np.zeros(3), 6.957e8, 0.0),
                      CelestialBody(1.0, x, v, 1.0, 0.0)]
            a_gr = NBodySystem(bodies, toggles=PhysicsToggles(gr_1pn=True)).compute_accelerations()[1]
            a_n = NBodySystem(bodies, toggles=PhysicsToggles()).compute_accelerations()[1]
            expected = _textbook_1pn(bodies[0].gm_m3_s2, x, v)
            np.testing.assert_allclose(a_gr - a_n, expected, rtol=1e-6, atol=1e-6 * np.linalg.norm(expected))

    def test_static_body_is_attracted_less_by_4GM_over_rc2(self):
        """At rest, the harmonic-gauge 1PN acceleration is −(GM/r²)(1 − 4GM/(rc²)) n: weaker, not stronger."""
        r = 0.1 * 1.496e11
        bodies = [CelestialBody(M_SUN, np.zeros(3), np.zeros(3), 6.957e8, 0.0),
                  CelestialBody(1.0, np.array([r, 0.0, 0.0]), np.zeros(3), 1.0, 0.0)]
        a_gr = NBodySystem(bodies, toggles=PhysicsToggles(gr_1pn=True)).compute_accelerations()[1]
        a_n = NBodySystem(bodies, toggles=PhysicsToggles()).compute_accelerations()[1]
        x = bodies[0].gm_m3_s2 / (r * _C**2)
        self.assertAlmostEqual(float(np.linalg.norm(a_gr) / np.linalg.norm(a_n)) - 1.0, -4.0 * x, delta=1e-6 * 4.0 * x)

    def test_eih_cross_equals_textbook_eih(self):
        rng = np.random.default_rng(11)
        gms_mass = [M_SUN, 5.97e24, 1.9e27]
        xs = [np.zeros(3), np.array([1.496e11, 0.0, 0.0]), np.array([0.0, 7.78e11, 1e9])]
        for _ in range(4):
            vs = [rng.normal(size=3) * 1e3, np.array([0.0, 2.98e4, 0.0]) + rng.normal(size=3) * 1e3,
                  np.array([-1.31e4, 0.0, 0.0]) + rng.normal(size=3) * 1e3]
            bodies = [CelestialBody(m, x.copy(), v.copy(), 1e6, 0.0) for m, x, v in zip(gms_mass, xs, vs)]
            a_eih = NBodySystem(bodies, toggles=PhysicsToggles(eih_cross=True)).compute_accelerations()
            a_n = NBodySystem(bodies, toggles=PhysicsToggles()).compute_accelerations()
            expected = _textbook_eih_corrections([b.gm_m3_s2 for b in bodies], xs, vs)
            for i in range(3):
                np.testing.assert_allclose(a_eih[i] - a_n[i], expected[i], rtol=1e-6,
                                           atol=1e-6 * np.linalg.norm(expected[i]))


class TestMercuryPerihelion(unittest.TestCase):
    """Mercury's perihelion advances by the GR value, ≈ +43″/cy (Will 2014), in both layers."""

    def test_gr_1pn_advances_mercury_by_the_GR_value(self):
        ratio = _perihelion_advance_per_orbit(PhysicsToggles(gr_1pn=True)) / _gr_advance_per_orbit()
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)

    def test_eih_cross_advances_mercury_by_the_GR_value(self):
        ratio = _perihelion_advance_per_orbit(PhysicsToggles(eih_cross=True)) / _gr_advance_per_orbit()
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)

    def test_eih_cross_precession_is_the_same_in_a_boosted_frame(self):
        """EIH is post-Galilean invariant: a 30 km/s boost along the perihelion velocity must not change the advance.
        (The single-body gr_1pn layer is valid only in the attractor's rest frame and is not tested here.)"""
        rest = _perihelion_advance_per_orbit(PhysicsToggles(eih_cross=True))
        boosted = _perihelion_advance_per_orbit(PhysicsToggles(eih_cross=True), boost=np.array([0.0, 3.0e4, 0.0]))
        self.assertAlmostEqual(boosted / rest, 1.0, delta=0.01)


class TestAdapterDefaults(unittest.TestCase):
    """The orbit adapters are plain Newtonian unless GR is requested."""

    def _pair(self):
        return [CelestialBody(M_SUN, np.zeros(3), np.zeros(3), 6.957e8, 0.0),
                CelestialBody(M_MERCURY, np.array([4.6e10, 0.0, 0.0]), np.array([0.0, 5.9e4, 0.0]), 2.44e6, 0.0)]

    def test_sg_nbody_default_is_newtonian(self):
        from sigma_ground.field.interface.adapters.integrator import sg_nbody
        default, _ = sg_nbody(self._pair(), dt_s=3600.0, t_total_s=10 * 86400.0)
        newton, _ = sg_nbody(self._pair(), dt_s=3600.0, t_total_s=10 * 86400.0, include_gr=False)
        with_gr, _ = sg_nbody(self._pair(), dt_s=3600.0, t_total_s=10 * 86400.0, include_gr=True)
        np.testing.assert_array_equal(default[1].position_m, newton[1].position_m)
        self.assertFalse(np.array_equal(default[1].position_m, with_gr[1].position_m))

    def test_sg_propagate_orbit_default_is_newtonian(self):
        from sigma_ground.field.interface.adapters.base import SGPhysicsMixin
        obj = SGPhysicsMixin()
        default = obj.sg_propagate_orbit(self._pair(), dt_s=3600.0, n_steps=240)
        newton = obj.sg_propagate_orbit(self._pair(), dt_s=3600.0, n_steps=240, include_gr=False)
        np.testing.assert_array_equal(default[1].position_m, newton[1].position_m)


if __name__ == "__main__":
    unittest.main()
