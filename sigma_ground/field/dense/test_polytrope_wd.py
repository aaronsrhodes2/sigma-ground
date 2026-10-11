"""field.dense.polytrope and white_dwarf: closed forms, published Lane-Emden constants, Chandrasekhar's limit and its approach.

Ground truth is independent of the code: exact n = 0 and n = 1 solutions, the literature values of xi_1(3) and omega_3 (Chandrasekhar 1939),
the closed-form Chandrasekhar mass written out here from the constants, the n = 3/2 and n = 3 polytrope scalings, and a theorem of GR
(the white-dwarf sequence turns over below the Newtonian limit). Labels follow misc/NUCLEAR_MATTER_PLAN.md.
"""
from __future__ import annotations

import math
import unittest

from sigma_ground.field.constants import AMU_KG, C, G, HBAR, M_NEUTRON_KG, M_SUN_KG
from sigma_ground.field.dense import white_dwarf as wd
from sigma_ground.field.dense.polytrope import lane_emden, polytrope_structure


class TestLaneEmden(unittest.TestCase):
    def test_n0_and_n1_exact(self):                                           # BUG
        a = lane_emden(0.0)
        self.assertAlmostEqual(a.xi1, math.sqrt(6.0), delta=1e-9)
        self.assertAlmostEqual(a.omega, 2.0 * math.sqrt(6.0), delta=1e-9)     # omega_0 = xi_1^3 / 3
        b = lane_emden(1.0)
        self.assertAlmostEqual(b.xi1, math.pi, delta=1e-9)
        self.assertAlmostEqual(b.omega, math.pi, delta=1e-9)

    def test_n3_published_constants(self):                                    # LIT: Chandrasekhar 1939, table 4
        c = lane_emden(3.0)
        self.assertAlmostEqual(c.xi1, 6.89685, delta=1e-5)
        self.assertAlmostEqual(c.omega, 2.01824, delta=1e-5)

    def test_n_out_of_range_is_rejected(self):                                # BUG (regime BEYOND)
        for n in (-0.5, 5.0, 6.0):
            with self.assertRaises(ValueError):
                lane_emden(n)

    def test_n1_structure_analytic(self):                                     # BUG: a^2 = K / (2 pi G), R = pi a, M = pi^2 ... closed form
        K, rho_c = 1.0e5, 3.0e6
        s = polytrope_structure(1.0, K, rho_c)
        a = math.sqrt(K / (2.0 * math.pi * G))
        self.assertAlmostEqual(s.radius_m / (math.pi * a), 1.0, delta=1e-9)
        self.assertAlmostEqual(s.mass_kg / (4.0 * math.pi * a ** 3 * rho_c * math.pi), 1.0, delta=1e-9)

    def test_n3_mass_is_independent_of_central_density(self):                 # BUG: the n = 3 polytrope has one mass
        K = 1.2e15
        m1 = polytrope_structure(3.0, K, 1e12).mass_kg
        m2 = polytrope_structure(3.0, K, 1e17).mass_kg
        self.assertAlmostEqual(m1 / m2, 1.0, delta=1e-12)


class TestChandrasekharMass(unittest.TestCase):
    def test_closed_form_with_literature_omega3(self):                        # BUG: independent evaluation of the formula
        want = math.sqrt(3.0 * math.pi) / 2.0 * 2.01824 * (HBAR * C / G) ** 1.5 / (2.0 * AMU_KG) ** 2
        self.assertAlmostEqual(wd.chandrasekhar_mass(2.0) / want, 1.0, delta=5e-6)       # omega_3 is quoted to 6 digits

    def test_plan_values(self):                                               # LIT/DERIVED: plan 1.4559 (m_u), 1.4310 (m_n) at mu_e = 2
        self.assertAlmostEqual(wd.chandrasekhar_mass(2.0, AMU_KG) / M_SUN_KG, 1.4559, delta=1.4559 * 1e-3)
        self.assertAlmostEqual(wd.chandrasekhar_mass(2.0, M_NEUTRON_KG) / M_SUN_KG, 1.4310, delta=1.4310 * 1e-3)

    def test_inverse_square_in_mu_e(self):                                    # BUG
        self.assertAlmostEqual(wd.chandrasekhar_mass(1.0) / wd.chandrasekhar_mass(2.0), 4.0, delta=1e-12)


class TestWhiteDwarfStructure(unittest.TestCase):
    def test_exact_eos_approaches_the_chandrasekhar_limit(self):              # BUG: M1 (exact EOS) + M0 (integrator) together
        m = wd.white_dwarf_structure(1e18, 2.0).mass_kg
        self.assertAlmostEqual(m / wd.chandrasekhar_mass(2.0), 1.0, delta=2e-3)
        self.assertLess(m, wd.chandrasekhar_mass(2.0))                        # approaches from below

    def test_mass_rises_with_central_density_and_radius_falls(self):          # BY CONSTRUCTION of the sequence (Newtonian)
        stars = wd.white_dwarf_mass_radius([1e7, 1e8, 1e9, 1e10, 1e11, 1e13])
        for a, b in zip(stars, stars[1:]):
            self.assertGreater(b.mass_kg, a.mass_kg)
            self.assertLess(b.radius_m, a.radius_m)

    def test_low_mass_slope_is_one_half(self):                                # BUG: n = 3/2 limit, M ~ rho_c^((3-n)/(2n)) = rho_c^(1/2)
        a, b = wd.white_dwarf_structure(1e5, 2.0), wd.white_dwarf_structure(2e5, 2.0)
        slope = math.log(b.mass_kg / a.mass_kg) / math.log(2.0)
        self.assertAlmostEqual(slope, 0.5, delta=0.01)

    def test_nonrelativistic_limit_matches_the_n_three_halves_polytrope(self):  # BUG: P = K rho^(5/3), K from the NR Fermi gas
        rho = 1e5
        K = (3.0 * math.pi ** 2) ** (2.0 / 3.0) * HBAR ** 2 / (5.0 * 9.1093837015e-31 * (2.0 * AMU_KG) ** (5.0 / 3.0))
        poly = polytrope_structure(1.5, K, rho)
        w = wd.white_dwarf_structure(rho, 2.0)
        self.assertAlmostEqual(w.mass_kg / poly.mass_kg, 1.0, delta=5e-3)     # the gap is the relativistic correction ~ x^2
        self.assertAlmostEqual(w.radius_m / poly.radius_m, 1.0, delta=5e-3)

    def test_gr_sequence_peaks_below_the_chandrasekhar_mass(self):            # BUG (a theorem of GR; fails only if GR is not applied)
        m_max = wd.white_dwarf_max_mass_gr(2.0)
        self.assertLess(m_max, wd.chandrasekhar_mass(2.0))
        self.assertGreater(m_max, 0.97 * wd.chandrasekhar_mass(2.0))

    def test_gr_and_newtonian_agree_at_low_density(self):                     # BUG
        n, g = wd.white_dwarf_structure(1e9, 2.0), wd.white_dwarf_structure(1e9, 2.0, gr=True)
        self.assertAlmostEqual(g.mass_kg / n.mass_kg, 1.0, delta=2e-3)

    def test_regime_labels_and_validation(self):                              # BY CONSTRUCTION
        self.assertEqual(wd.white_dwarf_structure(1e5, 2.0).regime, "EDGE")
        self.assertEqual(wd.white_dwarf_structure(1e9, 2.0).regime, "SAFE")
        with self.assertRaises(ValueError):
            wd.white_dwarf_structure(-1.0)


if __name__ == "__main__":
    unittest.main()
