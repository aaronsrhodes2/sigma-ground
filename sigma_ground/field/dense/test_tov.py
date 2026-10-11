"""field.dense.tov and eos: the Schwarzschild interior solution, Buchdahl's bound, the Newtonian limit, the Tolman identity,
the first law on the tables, and the free-neutron star (Oppenheimer & Volkoff 1939).

Ground truth is independent: the exact incompressible-star solution (Schwarzschild 1916) written out here, the n = 1 Lane-Emden polytrope, and the
published Oppenheimer-Volkoff maximum mass of a free neutron gas. Labels follow misc/NUCLEAR_MATTER_PLAN.md.
"""
from __future__ import annotations

import math
import time
import unittest

from sigma_ground.field.constants import C, G, M_SUN_KG
from sigma_ground.field.dense import eos as eosmod
from sigma_ground.field.dense import fermi_gas as fg
from sigma_ground.field.dense.polytrope import lane_emden
from sigma_ground.field.dense.tov import mass_radius_curve, maximum_mass, solve_tov

EPS0 = 6.0e33                                     # an arbitrary uniform energy density, J/m^3 (~ nuclear matter scale)


def _schwarzschild(pc_over_eps):
    """Exact incompressible star with central pressure P_c = pc_over_eps * eps: returns (C, R, M)."""
    s = (1.0 + pc_over_eps) / (1.0 + 3.0 * pc_over_eps)        # sqrt(1 - 2C) from P_c/eps = (1 - s)/(3 s - 1)
    comp = (1.0 - s * s) / 2.0
    R = math.sqrt(3.0 * comp * C ** 4 / (4.0 * math.pi * G * EPS0))
    return comp, R, comp * R * C * C / G


class TestSchwarzschildInterior(unittest.TestCase):
    def test_exact_star_at_compactness_one_quarter(self):                     # BUG: C = 0.25 (r_s/R = 0.5)
        pc = 0.2 * 0.0 + (1.0 - math.sqrt(0.5)) / (3.0 * math.sqrt(0.5) - 1.0)   # the P_c/eps of a C = 1/4 star
        star = solve_tov(eosmod.UniformEOS(EPS0), pc * EPS0)
        comp, R, M = _schwarzschild(pc)
        self.assertAlmostEqual(comp, 0.25, delta=1e-12)
        self.assertAlmostEqual(star.compactness, 0.25, delta=1e-8)
        self.assertAlmostEqual(star.radius_m / R, 1.0, delta=1e-8)
        self.assertAlmostEqual(star.mass_kg / M, 1.0, delta=1e-8)

    def test_several_central_pressures(self):                                 # BUG
        for pc in (1e-3, 0.05, 0.5, 2.0):
            star = solve_tov(eosmod.UniformEOS(EPS0), pc * EPS0)
            comp, R, M = _schwarzschild(pc)
            self.assertAlmostEqual(star.mass_kg / M, 1.0, delta=1e-7, msg=f"pc/eps={pc}")
            self.assertAlmostEqual(star.radius_m / R, 1.0, delta=1e-7, msg=f"pc/eps={pc}")

    def test_buchdahl_bound(self):                                            # BUG: C < 4/9 for every star
        for pc in (1e-2, 1.0, 50.0, 1e3):
            self.assertLess(solve_tov(eosmod.UniformEOS(EPS0), pc * EPS0).compactness, 4.0 / 9.0)
        eos = eosmod.free_fermi_eos(fg.NEUTRON)
        for s in mass_radius_curve(eos, fg.chandrasekhar_pressure(0.1, fg.NEUTRON), fg.chandrasekhar_pressure(20.0, fg.NEUTRON), 12):
            self.assertLess(s.compactness, 4.0 / 9.0)


class TestNewtonianLimit(unittest.TestCase):
    def test_n1_polytrope_at_low_central_pressure(self):                      # BUG: M/M_LE - 1 and R/(pi a) - 1 below 1e-4 at w_c = 1e-5
        K = 1.0e5
        wc = 1e-5
        rho_c = wc * C * C / K                                                # P_c = K rho_c^2 = w_c rho_c c^2
        star = solve_tov(eosmod.PolytropeEOS(K, 2.0), K * rho_c ** 2)
        a = math.sqrt(K / (2.0 * math.pi * G))
        m_le = 4.0 * math.pi * a ** 3 * rho_c * lane_emden(1.0).omega
        self.assertLess(abs(star.mass_kg / m_le - 1.0), 1e-4)
        self.assertLess(abs(star.radius_m / (math.pi * a) - 1.0), 1e-4)


class TestIdentities(unittest.TestCase):
    def setUp(self):
        self.eos = eosmod.free_fermi_eos(fg.NEUTRON)

    def test_tolman_mass_equals_gravitational_mass(self):                     # BY CONSTRUCTION (catches a wrong nu)
        for x in (0.3, 1.0, 2.0):
            s = solve_tov(self.eos, fg.chandrasekhar_pressure(x, fg.NEUTRON))
            self.assertAlmostEqual(s.tolman_mass_kg / s.mass_kg, 1.0, delta=1e-8, msg=f"x={x}")

    def test_baryon_mass_exceeds_gravitational_mass(self):                    # BUG: bound stars
        s = solve_tov(self.eos, fg.chandrasekhar_pressure(1.0, fg.NEUTRON))
        self.assertGreater(s.baryon_mass_kg, s.mass_kg)
        self.assertGreater(s.binding_energy_kg, 0.0)

    def test_first_law_on_the_tables(self):                                   # BUG: d eps / dn = (eps + P)/n to 1e-6
        for eos in (self.eos, eosmod.electron_ion_eos(2.0)):
            lo, hi = eos.pressure_range_pa
            for f in (1e3, 1e6, 1e9, 1e12):
                p = lo * f
                if lo < p < hi / 10:
                    self.assertLess(eosmod.first_law_error(eos, p), 1e-6, msg=f"{eos.name} P={p:.2e}")

    def test_table_reproduces_the_analytic_gas(self):                         # BUG
        for x in (0.05, 0.7, 5.0):
            p = fg.chandrasekhar_pressure(x, fg.NEUTRON)
            self.assertAlmostEqual(self.eos.eps(p) / fg.chandrasekhar_energy(x, fg.NEUTRON), 1.0, delta=1e-8, msg=f"x={x}")
            self.assertAlmostEqual(self.eos.n(p) / (fg.NEUTRON.n0_m3 * x ** 3), 1.0, delta=1e-8, msg=f"x={x}")


class TestFreeNeutronStar(unittest.TestCase):
    def test_oppenheimer_volkoff_maximum_mass(self):                          # LIT: OV 1939; plan: 0.710 +/- 0.005 M_sun
        eos = eosmod.free_fermi_eos(fg.NEUTRON)
        star = maximum_mass(eos, fg.chandrasekhar_pressure(0.3, fg.NEUTRON), fg.chandrasekhar_pressure(3.0, fg.NEUTRON))
        self.assertAlmostEqual(star.mass_msun, 0.710, delta=0.005)
        self.assertTrue(0.25 < star.trace_ratio_central < 0.34)               # plan: 3P/eps ~ 0.30 at the maximum
        self.assertLess(star.compactness, 4.0 / 9.0)

    def test_mass_rises_then_falls(self):                                     # BUG: the unstable branch exists
        eos = eosmod.free_fermi_eos(fg.NEUTRON)
        curve = mass_radius_curve(eos, fg.chandrasekhar_pressure(0.2, fg.NEUTRON), fg.chandrasekhar_pressure(10.0, fg.NEUTRON), 16)
        masses = [s.mass_kg for s in curve]
        k = masses.index(max(masses))
        self.assertTrue(0 < k < len(masses) - 1)

    def test_one_star_is_fast(self):                                          # engineering (plan: < 0.5 s; relaxed on a loaded machine)
        eos = eosmod.free_fermi_eos(fg.NEUTRON)
        t = time.perf_counter()
        solve_tov(eos, fg.chandrasekhar_pressure(1.0, fg.NEUTRON))
        self.assertLess(time.perf_counter() - t, 2.0)


class TestInputs(unittest.TestCase):
    def test_validation(self):                                                # BY CONSTRUCTION
        with self.assertRaises(ValueError):
            solve_tov(eosmod.UniformEOS(EPS0), -1.0)
        with self.assertRaises(ValueError):
            solve_tov(eosmod.UniformEOS(EPS0))
        with self.assertRaises(ValueError):
            eosmod.TabulatedEOS("bad", [2.0, 1.0, 3.0, 4.0], [1, 1, 1, 1], [1, 1, 1, 1], [0.1] * 4, 1.0)


if __name__ == "__main__":
    unittest.main()
