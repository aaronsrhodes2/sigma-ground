"""Gravitational-wave damping in the N-body engine (fixed 2026-10-10, KNOWN_GAPS A11).

Before the fix, `_apply_gw_damping` used the Peters power with r⁴ instead of r⁵ (units W·m; Earth–Sun 2.9e13 W instead
of 196 W, PSR J1738+0333 decaying 1.73e9 times too fast), rounded any per-step loss below half an ulp of 1 to zero, and
rewrote both velocities from the centre of mass every step, which injected a spurious decay. The tests below check the
engine against independent ground truth:

* the quadrupole power of a circular binary, L = (32/5) G⁴ m₁² m₂² (m₁+m₂)/(c⁵ r⁵) (Peters 1964, Phys. Rev. 136, B1224),
  and its orbit average for eccentric orbits, × f(e) = (1 + 73e²/24 + 37e⁴/96)/(1−e²)^(7/2) (Peters & Mathews 1963);
* Earth–Sun: ≈ 200 W (e.g. the quadrupole table in arXiv:gr-qc/0606111);
* PSR J1738+0333 (Freire et al. 2012, MNRAS 423, 3328): Pb = 0.3547907398724 d, e = 3.4e-7, m_c = 0.181 M☉,
  m_p = 1.46 M☉ (Antoniadis et al. 2012); general relativity predicts Ṗb = −27.7 (+1.5 −1.9) fs/s;
* bookkeeping: losses far below an ulp per step still add up to the right total.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.constants import AU_M, C as _C, G as _G, M_SUN_KG
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

M_EARTH = 5.97219e24          # kg, Mentat solar_system_body('earth') (NASA fact sheet 2024)
DAY_S = 86400.0


def _peters_circular(m1: float, m2: float, r: float) -> float:
    return (32.0 / 5.0) * _G**4 * m1**2 * m2**2 * (m1 + m2) / (_C**5 * r**5)


def _circular_pair(m1: float, m2: float, a: float, theta: float = 0.0) -> list[CelestialBody]:
    M = m1 + m2
    v = math.sqrt(_G * M / a)
    rot = np.array([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    x = rot @ np.array([a, 0.0, 0.0]); u = rot @ np.array([0.0, v, 0.0])
    return [CelestialBody(m1, -m2 / M * x, -m2 / M * u, 1.0, 0.0), CelestialBody(m2, m1 / M * x, m1 / M * u, 1.0, 0.0)]


class TestGWLuminosity(unittest.TestCase):

    def test_earth_sun_radiates_196_W(self):
        system = NBodySystem(_circular_pair(M_SUN_KG, M_EARTH, AU_M))
        L = system.gw_luminosity_W()
        self.assertAlmostEqual(L / _peters_circular(M_SUN_KG, M_EARTH, AU_M), 1.0, delta=1e-12)
        self.assertAlmostEqual(L, 196.4, delta=2.0)          # before the fix: 2.94e13 W

    def test_eccentric_orbit_average_is_peters_f_of_e(self):
        """Time average of the instantaneous power over a Kepler orbit (e = 0.617, Hulse–Taylor-like) equals
        Peters' orbit-averaged power with the enhancement f(e) = 11.86."""
        m1, m2, a, e = 1.438 * M_SUN_KG, 1.390 * M_SUN_KG, 2.0e8, 0.617
        M = m1 + m2
        n = math.sqrt(_G * M / a**3); b = a * math.sqrt(1 - e * e)
        samples = 4000; total = 0.0
        for k in range(samples):                      # uniform in mean anomaly = uniform in time
            mean_anom = 2 * math.pi * (k + 0.5) / samples
            E = mean_anom
            for _ in range(50):
                E -= (E - e * math.sin(E) - mean_anom) / (1 - e * math.cos(E))
            r = np.array([a * (math.cos(E) - e), b * math.sin(E), 0.0])
            Edot = n / (1 - e * math.cos(E))
            v = np.array([-a * math.sin(E) * Edot, b * math.cos(E) * Edot, 0.0])
            total += NBodySystem._gw_power(m1, m2, r, v)
        f_e = (1 + 73 / 24 * e**2 + 37 / 96 * e**4) / (1 - e**2) ** 3.5
        self.assertAlmostEqual((total / samples) / (_peters_circular(m1, m2, a) * f_e), 1.0, delta=1e-9)


class TestGWDampingBookkeeping(unittest.TestCase):

    def test_losses_below_half_an_ulp_accumulate(self):
        """10,000 damping steps of ε = L·dt/KE = 1e-17 each (below the half-ulp of 1, 5.6e-17) must remove 1e-13 of
        the relative kinetic energy. Before the fix every step rounded to a factor of exactly 1."""
        system = NBodySystem(_circular_pair(1.46 * M_SUN_KG, 0.181 * M_SUN_KG, 1.73e9),
                             toggles=PhysicsToggles(gw_damping=True))
        b0, b1 = system.bodies
        mu = b0.mass_kg * b1.mass_kg / (b0.mass_kg + b1.mass_kg)
        vel = [b0.velocity_m_s.copy(), b1.velocity_m_s.copy()]
        ke0 = 0.5 * mu * float(np.dot(vel[1] - vel[0], vel[1] - vel[0]))
        dt = 1e-17 * ke0 / system.gw_luminosity_W()
        p0 = b0.mass_kg * vel[0] + b1.mass_kg * vel[1]
        for _ in range(10_000):
            vel = system._apply_gw_damping(vel, dt)
        v_rel = vel[1] - vel[0]
        ke1 = 0.5 * mu * float(np.dot(v_rel, v_rel))
        self.assertAlmostEqual((ke0 - ke1) / ke0 / 1e-13, 1.0, delta=0.01)
        p1 = b0.mass_kg * vel[0] + b1.mass_kg * vel[1]
        self.assertLess(float(np.linalg.norm(p1 - p0)), 1e-15 * b1.mass_kg * float(np.linalg.norm(vel[1])))

    def test_earth_sun_damping_adds_no_spurious_drift(self):
        """Earth–Sun loses 2.3e-24 of its orbital energy per year, far below an ulp: a month of hourly damped Verlet
        steps must leave the orbit bit-identical to the undamped run (before the fix, the centre-of-mass rewrite
        changed it every step)."""
        runs = []
        for gw in (False, True):
            system = NBodySystem(_circular_pair(M_SUN_KG, M_EARTH, AU_M))
            for _ in range(24 * 30):
                system.step(3600.0, include_gw_loss=gw)
            runs.append(np.concatenate([b.position_m for b in system.bodies] + [b.velocity_m_s for b in system.bodies]))
        np.testing.assert_array_equal(runs[0], runs[1])


class TestPSRJ1738(unittest.TestCase):
    """PSR J1738+0333: the engine's orbital decay equals the Peters (GR quadrupole) value within 2 %."""

    PB = 0.3547907398724 * DAY_S
    MP, MC = 1.46 * M_SUN_KG, 0.181 * M_SUN_KG

    def _peters_pbdot(self) -> float:
        M = self.MP + self.MC
        return (-(192 * math.pi / 5) * (2 * math.pi * _G / (_C**3 * self.PB)) ** (5 / 3)
                * self.MP * self.MC * M ** (-1 / 3))

    def _a(self) -> float:
        return (_G * (self.MP + self.MC) * self.PB**2 / (4 * math.pi**2)) ** (1 / 3)

    def test_peters_value_matches_published_gr_prediction(self):
        """Ṗb from the engine's power and Kepler's law, −(3/2) Pb L/|E|, against Freire et al. 2012: −27.7 (+1.5 −1.9) fs/s."""
        system = NBodySystem(_circular_pair(self.MP, self.MC, self._a()))
        E = -_G * self.MP * self.MC / (2 * self._a())
        pbdot = -1.5 * self.PB * system.gw_luminosity_W() / abs(E)
        self.assertAlmostEqual(pbdot / self._peters_pbdot(), 1.0, delta=1e-9)
        self.assertTrue(-27.7e-15 - 1.9e-15 <= pbdot <= -27.7e-15 + 1.5e-15, pbdot)

    def test_integrated_orbital_decay_matches_peters(self):
        """Damped minus undamped energy over 200 orbits (dt = Pb/100), Ṗb = −(3/2) Pb Ė/E, mean of four orbital
        phases (the Verlet round-off random walk is ~1 % per run at this length). Before the fix: 1.73e9 × Peters;
        with r⁵ but the old per-step factor and centre-of-mass rewrite: 2.2 × Peters."""
        a = self._a(); dt = self.PB / 100; n = 100 * 200
        t = np.arange(n + 1) * dt
        ratios = []
        for theta in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            energies = {}
            for gw in (False, True):
                system = NBodySystem(_circular_pair(self.MP, self.MC, a, theta), toggles=PhysicsToggles(gw_damping=gw))
                e = np.empty(n + 1); e[0] = system.total_energy()
                for k in range(1, n + 1):
                    system.step(dt)
                    e[k] = system.total_energy()
                energies[gw] = e
            edot = np.polyfit(t, energies[True] - energies[False], 1)[0]
            ratios.append(-1.5 * self.PB * edot / energies[False][0] / self._peters_pbdot())
        self.assertAlmostEqual(float(np.mean(ratios)), 1.0, delta=0.02, msg=str(ratios))


if __name__ == "__main__":
    unittest.main()
